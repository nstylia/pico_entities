import numpy as np
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import tensorflow as tf


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "N"
ELMo_Start = '<S>'
ELMo_End = '</S>'
PAD_TOKEN = '_PAD_'
# PAD_LABEL = "O"
PAD_CHAR = '_PAD_'

# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None, padding = False):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield [words, tags]
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, pos, tag = ls[0], ls[1], ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]



    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)

def get_processing_word_elmo(vocab_words=None, vocab_chars=None,
                             lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]


        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def get_processing_word_elmo_plus(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. keep copy of word for ELMo usage
        word_elmo = word

        # 3. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 4. return tuple char ids, word id, word
        if vocab_chars is not None and chars == True:
            return char_ids, word, word_elmo
        else:
            return word

    return f

def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        #skipped - need word tokens for ELMo
        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    # raise Exception("Unknow key is not allowed. Check that "\
                    #                 "your vocab (tags?) is correct")
                    pass

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length

def list_padder(seq,wlen,wtok,clen,ctok):
    entries = []
    for entry in seq:
        entries.append(_list_padder(entry, wlen, wtok,level=1))

    seq_ = []
    for entry in entries:
        entry_chars = []
        for char in entry:
            entry_chars.append(_list_padder(char,clen,ctok,level=2))
        seq_.append(entry_chars)

    return seq_

def _list_padder(seq,length,token,level=1):
    if level == 1:
        seq_ = seq[:length] + [token] * max(length - len(seq), 0)
    if level == 2:
        chars = seq[0]
        word = seq[1]
        chars = chars[:length] + [token] * max(length - len(chars), 0)
        seq_ = (chars,word)
    return seq_

# second iteration to handle [chars,word_id,word] instead of [chars,word]
def list_padder_adv(seq,wlen,wtok,clen,ctok):
    entries = []
    for entry in seq:
        entries.append(_list_padder_adv(entry, wlen, wtok,level=1))

    seq_ = []
    for entry in entries:
        entry_chars = []
        for char in entry:
            entry_chars.append(_list_padder_adv(char,clen,ctok,level=2))
        seq_.append(entry_chars)

    return seq_

def _list_padder_adv(seq,length,token,level=1):
    if level == 1:
        seq_ = seq[:length] + [token] * max(length - len(seq), 0)
    if level == 2:
        chars = seq[0]
        word_id = seq[1]
        word = seq[2]
        chars = chars[:length] + [token] * max(length - len(chars), 0)
        seq_ = (chars,word_id,word)
    return seq_

def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def transform_data(dataset,vocab_chars,vocab_labels,max_length,max_length_char, use_elmo):
    """
    Function to transform data pre-given to neural network.
    Performs paddings on words & characters if true flag and labels.
    :param data:
        dataset: iteratior for sequence of data
        vocab_words = vocabulary file of words
        vocab_chars = vocabulary file of chars
    :return:
        (padded_data,padded_labels)
        padded_data = list of tuple (

    """
    dataset = [entry for entry in dataset]
    dataset = np.array(dataset)
    data = dataset[:,0]
    labels = dataset[:,1]
    if max_length == None and max_length_char == None:
        max_length = max(map(lambda x: len(x), data))
        max_length_char = max([max(map(lambda x: len(x), seq[0])) for seq in data])
    pad_char = vocab_chars[PAD_CHAR]
    if not use_elmo:
        PAD_TOKEN = 0
    else:
        PAD_TOKEN = '_PAD_'
    pad_token = ([pad_char]*max_length_char,PAD_TOKEN)
    pad_label = vocab_labels[NONE]
    new_data = list_padder(data, max_length, pad_token, max_length_char, pad_char)
    new_labels = [_list_padder(_, max_length, pad_label,level=1) for _ in labels]
    
    return np.array(list(zip(new_data,new_labels))), max_length, max_length_char

def transform_data_all(dataset,vocab_chars,vocab_words,vocab_labels,max_length,max_length_char):
    """
    Function to transform data pre-given to neural network.
    Performs paddings on words & characters if true flag and labels.
    :param data:
        dataset: iteratior for sequence of data
        vocab_words = vocabulary file of words
        vocab_chars = vocabulary file of chars
    :return:
        (padded_data,padded_labels)
        padded_data = list of tuple (

    """
    dataset = [entry for entry in dataset]
    dataset = np.array(dataset)
    data = dataset[:,0]
    labels = dataset[:,1]
    if max_length == None and max_length_char == None:
        max_length = max(map(lambda x: len(x), data))
        max_length_char = max([max(map(lambda x: len(x), seq[0])) for seq in data])

    pad_char = vocab_chars[PAD_CHAR]
    pad_word_id = vocab_words[PAD_TOKEN]
    pad_token = ([pad_char] * max_length_char, pad_word_id, PAD_TOKEN)
    pad_label = vocab_labels[NONE]
    new_data = list_padder_adv(data, max_length, pad_token, max_length_char, pad_char)
    new_labels = [_list_padder_adv(_, max_length, pad_label,level=1) for _ in labels]
    return np.array(list(zip(new_data,new_labels))), max_length, max_length_char

def get_elmo_embeddings(config):

    batcher = Batcher(config.filename_words, 50)

    token_ids = tf.placeholder('int32', shape=(None, None, 50))
    bilm = BidirectionalLanguageModel(
        config.filename_elmo_options,
        config.filename_elmo_weights,
    )

    elmo_embeddings_op = bilm(token_ids)
    elmo_context_input = weight_layers('input', elmo_embeddings_op, l2_coef=0.0)

    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.

        sess.run(tf.global_variables_initializer())

        # Create batches of data.
        train = CoNLLDataset(config.filename_train)
        sents_train = [entry[0] for entry in train]
        sent_ids_train = batcher.batch_sentences(sents_train)

        # Compute ELMo representations (here for the input only, for simplicity).

        elmo_input = sess.run(
            [elmo_context_input['weighted_op']],
            feed_dict={token_ids: sent_ids_train[0]}
        )
        for batch in sent_ids_train[1:]:
            elmo_input_ = sess.run(
                [elmo_context_input['weighted_op']],
                feed_dict={token_ids: batch}
            )
            elmo_input = np.hstack(elmo_input,elmo_input_)

        test = CoNLLDataset(config.filename_test)
        sents_test = [entry[0] for entry in test]
        sent_ids_test = batcher.batch_sentences(sents_test)

        elmo_context_output_ = sess.run(
            [elmo_context_input['weighted_op']],
            feed_dict={token_ids: sent_ids_test}
        )

    return elmo_context_input_, elmo_context_output_

#Wan et al. (2013) for RNNs
def dropconnect(W, p):
    return tf.nn.dropout(W, keep_prob=p) * p