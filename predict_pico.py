import tensorflow as tf
import os
from build_data import generate_model_data, CoNLLDataset, generate_model_data_elmo
from src.data_utils import transform_data, transform_data_all
from src.sent_data_utils import save_predicted
from src.config import Config
from models.ner_model import NERModel


if __name__ == "__main__":
    config = Config()
    config.load()


    model = NERModel(config)
    tf.reset_default_graph()
    model.build()
    print('Loading model...')
    model.restore_session(config.dir_model)
    print('Model load completed...')


    files = os.listdir(config.input_save_dir)
    for file in files:
        input_file = os.path.join(config.input_save_dir, file)
        extra = CoNLLDataset(input_file, config.processing_word_elmo,
                             config.processing_tag, config.max_iter)
        extra, _, _ = transform_data(extra, config.vocab_chars, config.vocab_tags, \
                                     config.max_length_words, config.max_length_chars, config.use_elmo)

        predicted_labels = model.predict_abstract(extra)
        output_file = os.path.join(config.output_save_dir, file)

        idx2tag = dict([(v, k) for (k, v) in config.vocab_tags.items()])
        save_predicted(file, output_file, extra, predicted_labels)



