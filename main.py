import numpy as np
import time, argparse, os

from build_data import generate_model_data, CoNLLDataset, generate_model_data_elmo
from src.data_utils import transform_data, transform_data_all
from src.config import Config
from models.ner_model import NERModel


if __name__ == "__main__":

    runtime = time.gmtime()
    timestamp = runtime.tm_mday.__str__() + '_' + runtime.tm_mon.__str__() + '_' + runtime.tm_year.__str__() + '-' + \
                runtime.tm_hour.__str__() + '_' + runtime.tm_min.__str__() + '_' + runtime.tm_sec.__str__()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_prefix', help='data prefix')

    args = parser.parse_args()
    print("Running experiment with prefix:" + args.data_prefix)
    
    # generate_model_data_elmo(data_prefix=args.data_prefix) #only for first run
    

    config = Config()


    if args.data_prefix:
        cwd = os.getcwd()
        config.filename_dev = os.path.join(cwd, 'data', args.data_prefix + '_' + os.path.basename(config.filename_dev))
        config.filename_test = os.path.join(cwd, 'data', args.data_prefix + '_' + os.path.basename(config.filename_test))
        config.filename_train = os.path.join(cwd, 'data', args.data_prefix + '_' + os.path.basename(config.filename_train))

    if config.use_elmo_and_words:
        dev = CoNLLDataset(config.filename_dev, config.processing_word_elmo_plus,
                           config.processing_tag, config.max_iter)
        train = CoNLLDataset(config.filename_train, config.processing_word_elmo_plus,
                             config.processing_tag, config.max_iter)

        print("Transforming train and dev data...")

        train, config.max_length_words, config.max_length_chars = \
            transform_data_all(train, config.vocab_chars, config.vocab_words, config.vocab_tags, \
                               config.max_length_words, config.max_length_chars)
        dev, _, _ = transform_data_all(dev, config.vocab_chars, config.vocab_words, config.vocab_tags, \
                                       config.max_length_words, config.max_length_chars)

    elif config.use_elmo:
        dev = CoNLLDataset(config.filename_dev, config.processing_word_elmo,
                           config.processing_tag, config.max_iter)
        train = CoNLLDataset(config.filename_train, config.processing_word_elmo,
                             config.processing_tag, config.max_iter)

        print("Transforming train and dev data...")
        train, config.max_length_words, config.max_length_chars = \
            transform_data(train, config.vocab_chars, config.vocab_tags, \
                               config.max_length_words, config.max_length_chars,config.use_elmo)
        dev, _, _ = transform_data(dev, config.vocab_chars, config.vocab_tags, \
                                       config.max_length_words, config.max_length_chars,config.use_elmo)


    else :
        dev = CoNLLDataset(config.filename_dev, config.processing_word,
                           config.processing_tag, config.max_iter)
        train = CoNLLDataset(config.filename_train, config.processing_word,
                             config.processing_tag, config.max_iter)
        print("Transforming train and dev data...")
        train, config.max_length_words, config.max_length_chars = \
            transform_data(train, config.vocab_chars, config.vocab_tags, \
                           config.max_length_words, config.max_length_chars,config.use_elmo)
        dev, _, _ = transform_data(dev, config.vocab_chars, config.vocab_tags, \
                                   config.max_length_words, config.max_length_chars, config.use_elmo)

    model = NERModel(config)
    model.build()
    print('Training...')
    model.train(train, dev)

    print('Testing..')
    config.testing=np.bool(True)
    if config.use_elmo_and_words:
        test = CoNLLDataset(config.filename_test, config.processing_word_elmo_plus,
                            config.processing_tag, config.max_iter)

        print("Transforming test data...")
        test, _, _ = transform_data_all(test, config.vocab_chars, config.vocab_words, config.vocab_tags, \
                                        config.max_length_words, config.max_length_chars)

        model.evaluate(test)
    elif config.use_elmo:
        test = CoNLLDataset(config.filename_test, config.processing_word_elmo,
                            config.processing_tag, config.max_iter)

        print("Transforming test data...")
        test, _, _ = transform_data(test, config.vocab_chars, config.vocab_tags, \
                                        config.max_length_words, config.max_length_chars,config.use_elmo)

        model.evaluate(test)
    else:
        test = CoNLLDataset(config.filename_test, config.processing_word,
                            config.processing_tag, config.max_iter)
        print("Transforming test data...")
        test, _, _ = transform_data(test, config.vocab_chars, config.vocab_tags,
                                    config.max_length_words, config.max_length_chars, config.use_elmo)
        model.evaluate(test)