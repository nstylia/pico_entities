import tensorflow as tf

import sys
sys.path.append('../')
import src.eval as eval

from src.data_utils import minibatches, pad_sequences, NONE
from src.general_utils import Progbar
from models.base_model import BaseModel

import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers


#highway and more
from models.nns import highway_layer, attention


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           list(self.config.vocab_tags.items())}
        if self.config.use_elmo:
            # self.elmo_inputs = []
            self.batcher = Batcher(self.config.filename_words, 50)
            self.bilm = BidirectionalLanguageModel(
                self.config.filename_elmo_options,
                self.config.filename_elmo_weights)
            self.elmo_token_ids = tf.placeholder('int32', shape=(None, None, 50))
            self.elmo_embeddings_op = self.bilm(self.elmo_token_ids)
            self.elmo_embeddings_input = weight_layers('input', self.elmo_embeddings_op, l2_coef=0.0)

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, \
                    shape=[self.config.batch_size, self.config.max_length_words], name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[self.config.batch_size],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32,\
                    shape=[self.config.batch_size, self.config.max_length_words, self.config.max_length_chars],
                        name="char_ids")

        # shape = (batch size, max length sentences, 1024)
        self.elmo_embeddings = tf.placeholder(tf.float32, shape=(self.config.batch_size, self.config.max_length_words,
                                                                 self.config.elmo_size))

        self.elmo_and_char_embeddings = tf.placeholder(tf.float32,
                                                       shape=[self.config.batch_size,self.config.max_length_words,
                                                              self.config.elmo_chars_size])

        #shape = (batch_size, max_length sentences, Elmo emb size + Char emb size + word2vec emb size_
        self.word_elmo_char_embeddings = tf.placeholder(tf.float32, \
                                                        shape=[self.config.batch_size, self.config.max_length_words,
                                                               self.config.elmo_chars_size+self.config.dim_char])
        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.max_length_words],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.max_length_words],
                        name="labels")

        #bool
        self.is_test = tf.placeholder(tf.bool)

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        # self.is_test = self.config.is_test

        if self.config.use_elmo:
            if self.config.use_chars and self.config.use_elmo_and_words:
                char_ids, words_embs, word_ids = list(zip(*words))
                char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                                       nlevels=2)
                word_ids, sequence_lengths = pad_sequences(word_ids, '_PAD_')

            elif self.config.use_chars:
                char_ids, word_ids = list(zip(*words))
                char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                                       nlevels=2)
                word_ids, sequence_lengths = pad_sequences(word_ids, '_PAD_')
            else:
                word_ids, sequence_lengths = pad_sequences(words, '_PAD_')
            elmo_ids = self.batcher.batch_sentences(word_ids)
            elmo_embeddings = self.sess.run(
                [self.elmo_embeddings_input['weighted_op']],
                feed_dict={self.elmo_token_ids: elmo_ids}
            )

        elif self.config.use_chars:
            char_ids, word_ids = list(zip(*words))
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        if self.config.use_elmo:
            feed = {
                self.elmo_embeddings : elmo_embeddings[0],
                self.sequence_lengths: sequence_lengths
            }

        else:
            feed = {
                self.word_ids: word_ids,
                self.sequence_lengths: sequence_lengths
            }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if self.config.use_elmo_and_words:
            feed[self.word_ids] = words_embs

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        feed[self.is_test] = self.config.testing


        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

    def add_chars_elmo_highway_op(self):
        elmo_embeddings = self.elmo_embeddings
        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")

            cnns_list = []
            for filter_size, kernel_size in zip(self.config.filters, self.config.kernels):
                cnn_2d = tf.layers.conv2d(inputs=char_embeddings,
                                         filters=filter_size,
                                         kernel_size=kernel_size,
                                         strides=self.config.strides,
                                         padding='same',
                                         name="kernel_%d" % kernel_size)
                cnn_2d_bn = tf.layers.batch_normalization(inputs=cnn_2d,
                                                       name="batchnorm_%d" % kernel_size)
                cnn_2d_act = tf.nn.tanh(cnn_2d_bn, name="cnn_tanh_%d" % kernel_size)
                cnn_2d_mp = tf.layers.max_pooling2d(inputs=cnn_2d_act, pool_size=1, strides=1, name="cnn_mp_%d" % kernel_size)
                cnn_2d_rd = tf.reduce_mean(cnn_2d_mp, axis=[2])
                cnns_list.append(cnn_2d_rd)

            cnns = tf.concat(cnns_list, axis=-1)

            cnns_h = highway_layer(cnns, bias=self.config.highway_bias,
                                   bias_start=self.config.highway_bias_start,
                                   scope='highway_layer')
        elmo_embeddings = tf.concat([elmo_embeddings, cnns_h], axis=-1)
        self.elmo_and_char_embeddings = tf.nn.dropout(elmo_embeddings, self.dropout)

    def add_word_char_highway_embeddings_op(self):
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")


            cnns_list = []
            for filter_size, kernel_size in zip(self.config.filters, self.config.kernels):
                cnn_2d = tf.layers.conv2d(inputs=char_embeddings,
                                         filters=filter_size,
                                         kernel_size=kernel_size,
                                         strides=1,
                                         padding='same',
                                         name="kernel_%d" % kernel_size)
                cnn_2d_bn = tf.layers.batch_normalization(inputs=cnn_2d,
                                                       name="batchnorm_%d" % kernel_size)
                cnn_2d_act = tf.nn.tanh(cnn_2d_bn, name="cnn_tanh_%d" % kernel_size)
                cnn_2d_mp = tf.layers.max_pooling2d(inputs=cnn_2d_act, pool_size=1, strides=1, name="cnn_mp_%d" % kernel_size)
                cnn_2d_rd = tf.reduce_mean(cnn_2d_mp, axis=[2])
                cnns_list.append(cnn_2d_rd)

            cnns = tf.concat(cnns_list, axis=-1)

            cnns_h = highway_layer(cnns, bias=self.config.highway_bias,
                                   bias_start=self.config.highway_bias_start,
                                   scope='highway_layer')

        word_embeddings = tf.concat([word_embeddings, cnns_h], axis=-1)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_word_char_1d_highway_embeddings_op(self):
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1 for cnn1d
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                # word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                cnns_list = []
                for filter_size, kernel_size in zip(self.config.filters, self.config.kernels):
                    cnn_1d = tf.layers.conv1d(inputs=char_embeddings,
                                             filters=filter_size,
                                             kernel_size=kernel_size,
                                             strides=1,
                                             padding='same',
                                             name="kernel_%d" % kernel_size)
                    cnn_1d_bn = tf.layers.batch_normalization(inputs=cnn_1d,
                                                              name="batchnorm_%d" % kernel_size)
                    cnn_1d_act = tf.nn.tanh(cnn_1d_bn, name="cnn_tanh_%d" % kernel_size)
                    cnn_1d_mp = tf.layers.max_pooling1d(inputs=cnn_1d_act, pool_size=1, strides=1,
                                                        name="cnn_mp_%d" % kernel_size)
                    cnn_f = tf.reshape(cnn_1d_mp, shape=[s[0], s[1], s[2] * filter_size])
                    cnns_list.append(cnn_f)

                cnns = tf.concat(cnns_list, axis=-1)

        word_embeddings = tf.concat([word_embeddings, cnns], axis=-1)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op_elmo_rdbilstm(self):
        if self.config.use_chars:
            input = self.elmo_and_char_embeddings
        else:
            input = self.elmo_embeddings

        # attention-global
        with tf.variable_scope('attention'):
            attn, alphas = attention(input)
            input = tf.expand_dims(attn, axis=1) * input

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, input,
                    dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)


        input_h = highway_layer(input, bias=self.config.highway_bias,
                               bias_start=self.config.highway_bias_start,
                               scope='highway_layer')
       
        output_h = tf.math.add(input_h, output)

        with tf.variable_scope("rdbilstm"):
            cell_fw_rd = tf.contrib.rnn.LSTMCell(self.config.rdsize)
            cell_bw_rd = tf.contrib.rnn.LSTMCell(self.config.rdsize)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw_rd, cell_bw_rd, output_h,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output_rd = tf.concat([output_fw, output_bw], axis=-1)
            output_rd = tf.nn.dropout(output_rd, self.dropout)

        with tf.variable_scope("projrd"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.config.rdsize, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                                dtype=tf.float32, initializer=tf.glorot_uniform_initializer())

            nsteps = tf.shape(output_rd)[1]
            output = tf.reshape(output_rd, [-1, 2 * self.config.rdsize])

            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)


        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])

            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def build(self):
        self.add_placeholders()
        if self.config.use_elmo:
            self.add_chars_elmo_highway_op()
            self.add_logits_op_elmo_rdbilstm()
        elif self.config.char_max:
            self.add_word_embeddings_op()
            self.add_word_char_cnn_bilstm_embeddings_op()
            self.add_logits_highway_op()
        else:
            self.add_word_char_highway_embeddings_op() #original -lstm+cnn2d
            self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init



    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths

    def batch_to_batchsize(self, batch_size, words, labels):
        import numpy as np
        # size = 5
        size = len(words)
        if batch_size > size:
            diff = batch_size - size
            if not self.config.use_elmo:
                pad_id = 0
                pad_char = [0] * self.config.max_length_chars
            else:
                pad_id = '_PAD_'
                pad_char = [self.config.vocab_chars[pad_id]] * self.config.max_length_chars
            pad_label = self.config.vocab_tags[NONE]
            pad_char_entries = ([pad_char] * self.config.max_length_words)
            pad_word_entries = [pad_id] * self.config.max_length_words
            if self.config.use_elmo_and_words:
                pad_word_id = [self.config.vocab_words[pad_id]] *self.config.max_length_words
                pad_entry_words = (pad_char_entries, pad_word_id, pad_word_entries)
            else:
                pad_entry_words = (pad_char_entries,pad_word_entries)
            pad_entry_labels = np.array([pad_label] * self.config.max_length_words, dtype=object)
            for _ in range(0,diff):
                words.append(pad_entry_words)
                labels.append(pad_entry_labels)
            return words,labels
        else:
            return words,labels

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        # self.is_test = self.config.testing
        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            self.config.iter = i
            # self.is_test = self.config.testing
            words, labels =  self.batch_to_batchsize(batch_size,words,labels)
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)


        # metrics = self.run_evaluate(dev)
        metrics = self.run_eval_pad(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in list(metrics.items())])
        self.logger.info(msg)

        return metrics["f1"]

    def run_epoch_eval(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        import itertools
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        # self.is_test = self.config.testing
        # iterate over dataset
        for (i, (words,labels)),(j, (eval_words, eval_labels)) in zip(enumerate(minibatches(train, batch_size)),
                                                  # itertools.cycle(enumerate(minibatches(dev, batch_size)))):
                                                enumerate(minibatches(itertools.cycle(dev), batch_size))):
            self.config.iter = i
            words, labels = self.batch_to_batchsize(batch_size, words, labels)
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                                       self.config.dropout)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            eval_words, eval_labels = self.batch_to_batchsize(batch_size, eval_words, eval_labels)
            eval_fd, _ = self.get_feed_dict(eval_words, eval_labels, self.config.lr,
                                   self.config.dropout)
            _, val_loss, _ = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=eval_fd)

            prog.update(i + 1, [("train loss", train_loss),("val loss", val_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)
                #loss tracking
                tf.summary.scalar("train loss", train_loss)
                tf.summary.scalar("eval loss", val_loss)

        # metrics = self.run_evaluate(dev)
        metrics = self.run_eval_pad(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in list(metrics.items())])
        self.logger.info(msg)
        # eval loss

        return metrics["f1"]

    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """

        def div_or_zero(num, den):
          return num/den if den else 0.0

        l_true = []
        l_pred = []

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                l_true += lab
                l_pred += lab_pred

        # Token stats
        print('Passing LSTM-CRF tags to eval func:')
        print('\t', self.idx_to_tag.items())
        tags = [idx for idx, tag in self.idx_to_tag.items() if tag != NONE]
        return eval.token_f1(true = l_true, pred = l_pred, labels = tags)

    def run_eval_pad(self,test):
        def div_or_zero(num, den):
          return num/den if den else 0.0

        l_true = []
        l_pred = []

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):

            words, labels = self.batch_to_batchsize(self.config.batch_size, words, labels)
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = list(lab[:length])
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                l_true += lab
                l_pred += lab_pred
        # l_true = [list(lab) for lab in labels]
        # l_pred = labels_pred
        # Token stats
        print('Passing LSTM-CRF tags to eval func:')
        print('\t', self.idx_to_tag.items())
        tags = [idx for idx, tag in self.idx_to_tag.items() if tag != NONE]
        return eval.token_f1(true = l_true, pred = l_pred, labels = tags)

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = list(zip(*words))
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

    def predict_elmo(self, sentence):
        words = [self.config.processing_word_elmo(w) for w in sentence]
        if type(words[0]) == tuple:
            words = list(zip(*words))
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

    def predict_abstract(self,pred_entry):
        all_pred_labels = []
        for words, labels in minibatches(pred_entry, self.config.batch_size):
            words, labels = self.batch_to_batchsize(self.config.batch_size, words, labels)
            labels_pred, _ = self.predict_batch(words)
            all_pred_labels+=labels_pred
        return all_pred_labels