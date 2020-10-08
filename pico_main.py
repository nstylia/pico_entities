import os
import glob
from src.sent_data_utils import PICODataset_xml, PICODataset_pkl, build_pred_with_tags
from src.config import PICOConfig
from sklearn.model_selection import KFold
from models.sentence_classifier import Classifier


if __name__ == "__main__":
    print("Initializing...")
    config = PICOConfig()

    cwd = os.getcwd()
    if config.use_gold:
        files = glob.glob(config.file_data_gold+'*.xml')
    else:
        # build_pred_with_tags(config) #only for first run
        files = glob.glob(config.file_data_pred_tags+'*.pkl')

    kf = KFold(n_splits=config.n_splits, shuffle=True,random_state=84264)
    results = []



    print("Starting splits...")
    for train_indexes, test_indexes in kf.split(files):
        train_files = [files[index] for index in train_indexes]
        test_files = [files[index] for index in test_indexes]

        if config.use_gold:
            train = PICODataset_xml(config.file_data_gold,train_files,config.processing_sent_words,
                                config.max_iter)
            test = PICODataset_xml(config.file_data_gold, test_files,config.processing_sent_words,
                               config.max_iter)
        else:
            train = PICODataset_pkl(config.file_data_pred, train_files, config.processing_sent_words,
                                    config.max_iter)
            test = PICODataset_pkl(config.file_data_pred, test_files, config.processing_sent_words,
                                   config.max_iter)

        classifier = Classifier(config)
        classifier.model_selector()
        train, test = classifier.add_features(train,test)

        print("training split")
        predictions = classifier.train_predict(train,test)
        print("train completed")
        print("Scores:")
        scores = classifier.score(test,predictions)
        print("\n")
        results.append(scores)


      
    print(results)
    f1_total = 0
    prec_total = 0
    rec_total = 0
    for scores in results:
        f1_total += scores[0][1]
        prec_total += scores[1]
        rec_total += scores[2]
    print("\n\nFINAL AVERAGE SCORES")
    print(f1_total / config.n_splits)
    print(prec_total / config.n_splits)
    print(rec_total / config.n_splits)