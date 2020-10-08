from src.data_utils import CoNLLDataset
from src.ner_model import NERModel
from src.config import Config

import sys, os

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def main(data_prefix = None):
    # create instance of config
    config = Config()
    
    if data_prefix:
      cwd = os.getcwd()
      config.filename_dev   = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_dev))
      config.filename_test  = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_test))
      config.filename_train = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_train))

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)
    model.evaluate(test)

    # evaluate and interact
    #interactive_shell(model)


if __name__ == "__main__":
    main(data_prefix = sys.argv[1])
