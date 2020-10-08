import os
import numpy as np
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word, get_processing_word_elmo, get_processing_word_elmo_plus
from .sent_data_utils import get_processing_sent_words

class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_word_elmo = get_processing_word_elmo(self.vocab_words,
                self.vocab_chars, lowercase=False, chars=self.use_chars)
        self.processing_word_elmo_plus = get_processing_word_elmo_plus(self.vocab_words,
                self.vocab_chars, lowercase=False, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)
        # self.embeddings = None #we are not using embeddings - we are using elmo
        self.word_embeddings = None # need to fix this
    cwd = os.getcwd()

    # general config
    dir_output = os.path.join(cwd, "results/test/")
    dir_model  = os.path.join(dir_output, "model.weights/")
    path_log   = os.path.join(dir_output, "log.txt")

    input_save_dir = os.path.join(cwd, "data/pico_sent_iob/")
    output_save_dir = os.path.join(cwd, "data/pico_sent_xml_pred/")

    # embeddings
    dim_word = 200
    dim_char = 300

    #PubMed Embeddings
    filename_glove = os.path.join(cwd, "data/PubMed-w2v.txt")

    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = os.path.join(cwd, "data/embeddings.{}d.trimmed.npz".format(dim_word))
    use_pretrained = True


    #bioelmo model
    filename_elmo_options = os.path.join(cwd, "elmo_models/elmo_2x4096_512_2048cnn_2xhighway_options_PubMed_only.json")
    filename_elmo_weights = os.path.join(cwd, "elmo_models/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5")
    filename_elmo_token_embeddings = os.path.join(cwd, "elmo_models/elmo_token_embeddings.hdf5")
    elmo_size = 1024

    # dataset
    filename_dev = os.path.join(cwd,   "data/dev.txt")
    filename_test = os.path.join(cwd,  "data/gold.txt")
    filename_train = os.path.join(cwd, "data/train.txt")


    max_iter = None # if not None, max number of examples in Dataset
    
    # vocab (created from dataset with build_data.py)
    filename_words = os.path.join(cwd, "data/words.txt")
    filename_tags = os.path.join(cwd, "data/tags.txt")
    filename_chars = os.path.join(cwd, "data/chars.txt")

    #padding info - generated on the fly
    max_length_words = 188
    max_length_chars = 50

    # training
    train_embeddings = False
    nepochs          = 20
    dropout          = 0.5 
    batch_size       = 32
    lr_method        = "adam" 
    lr               = 0.0001
    lr_decay         = 0.9
    clip = .1  # if negative, no clipping
    nepoch_no_imprv  = 5

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 712 # main model lstm
    rdsize = 356 #residual lstm 

    #highway
    highway_bias = True
    highway_bias_start = -2.0

    use_crf = True # CRF or Softmax
    use_chars = True # char embedding
    use_elmo = True # if elmo embeddings instead of word embeddings
    use_elmo_and_words = False #if word embeddings with elmo embeddings
    char_max = False #when elmo and chars - chars -> cnns -> BiLSTM. // Concat(with Elmo) - > Bilstm -> Concat (highway) -> proj.
    
    #variables for CNN and batchnorm
    iter = 1
    testing = np.bool(False)
    kernels = [1, 2, 3, 4]
    filters = [40,80,120,160]
    strides = 1
    
    elmo_chars_size = elmo_size + sum(filters) 


class PICOConfig():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary - same as with NER
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)
        #
        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_sent_words = get_processing_sent_words(self.vocab_words,
                                                   self.vocab_chars, lowercase=False, chars=self.use_chars)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                           if self.use_pretrained else None)
        # self.embeddings = None #we are not using embeddings - we are using elmo
        self.word_embeddings = None  # need to fix this

    cwd = os.getcwd()

    #gold ners
    file_data_gold = os.path.join(cwd, "data/pico_sent_xml/")
    file_data_pred = os.path.join(cwd, "data/pico_sent_xml_pred/")
    # file_data_pred = os.path.join("/home/nikos/Desktop/3/")
    file_data_pred_tags = os.path.join(cwd, "data/pico_sent_pred/")



    # vocab (created from dataset with build_data.py)
    filename_words = os.path.join(cwd, "data/words.txt")
    filename_tags = os.path.join(cwd, "data/tags.txt")
    filename_chars = os.path.join(cwd, "data/chars.txt")

    #PubMed embeddings
    filename_glove = os.path.join(cwd, "data/PubMed-w2v.txt")

    # general config
    data_dir = os.path.join('pico_sents', 'aggregated')

    dir_output = os.path.join(cwd, "pico_results/test/")
    dir_model = os.path.join(dir_output, "pico_model/")
    path_log = os.path.join(dir_output, "log.txt")

    use_chars = False
    use_pretrained = False
    use_w2v = True
    use_gold = False

    n_splits = int(10)

    features = {
        'sent_len' : True,
        'struct': False,
        'punct': True,
        'ners': False,
        'has_P': False,
        'has_IC': False,
        'has_O': False,
        'has_all': False
    }

    predict_prob = False
    model = 'xgb'

    model_params = None

    max_iter = None