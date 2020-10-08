from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,auc,\
    roc_auc_score,precision_recall_curve,roc_curve
from gensim.models import KeyedVectors
from nltk import word_tokenize
import numpy as np

class Classifier():

    def __init__(self, config):
        super(Classifier, self).__init__()
        self.model = None
        self.config = config

    def model_selector(self):
        if self.config.model.lower() == 'svc':
            # defaults
            penalty = 'l2'
            loss = 'squared_hinge'
            dual = True
            tol = 0.0001
            c = 2.0
            class_weight = None
            verbose = 1
            max_iter = 2000
            if self.config.model_params is not None:
                if 'penalty' in self.config.model_params : penalty = self.config.model_params['penalty']
                if 'loss'  in self.config.model_params : loss = self.config.model_params['loss']
                if 'dual'  in self.config.model_params : dual = self.config.model_params['dual']
                if 'tol'  in self.config.model_params : tol=self.config.model_params['tol']
                if 'c'  in self.config.model_params : c = self.config.model_params['c']
                if 'class_weight'  in self.config.model_params : class_weight = self.config.model_params['class_weight']
                if 'verbose'  in self.config.model_params : verbose = self.config.model_params['verbose']
                if 'max_iter'  in self.config.model_params : max_iter = self.config.model_params['max_iter']

            self.model = LinearSVC(penalty=penalty,loss=loss,dual=dual,
                                   tol=tol,C=c,class_weight=class_weight,
                                   max_iter=max_iter,verbose=verbose)
        if self.config.model.lower() == 'lr':
            self.model = LogisticRegression(solver='lbfgs', penalty='l2')
        if self.config.model.lower() == 'linearreg':
            self.model = LinearRegression()
        if self.config.model.lower() == 'gnb':
            self.model = GaussianNB()
        if self.config.model.lower() == 'mnb':
            self.model = GaussianNB()
        if self.config.model.lower() == 'edt':
            n_est = 5
            if self.config.model_params is not None:
                if 'est' in self.config.model_params: n_est = self.config.model_params['n_est']
            self.model = ExtraTreesClassifier(n_estimators=n_est)
        if self.config.model.lower() == 'dt':
            self.model = DecisionTreeClassifier()
        if self.config.model.lower() == 'rf':
            self.model = RandomForestClassifier()
        if self.config.model.lower() =='gb':
            loss='deviance'
            lr= 0.1
            n_estimators=100
            criterion='friedman_mse' #or mse
            verbose=1
            if self.config.model_params is not None:
                if 'loss' in self.config.model_params: loss = self.config.model_params['loss']
                if 'lr' in self.config.model_params: lr = self.config.model_params['lr']
                if 'n_estimators' in self.config.model_params: n_estimators = self.config.model_params['n_estimators']
                if 'criterion' in self.config.model_params: criterion = self.config.model_params['criterion']
                if 'verbose' in self.config.model_params: verbose = self.config.model_params['verbose']
            self.model = GradientBoostingClassifier(loss=loss,learning_rate=lr,n_estimators=n_estimators,
                                                    criterion=criterion,verbose=verbose)
        if self.config.model.lower() == 'xgb':
            self.model = XGBClassifier(learning_rate=0.3)

    def featurize(self,train,test):
        x_train, ner_train, y_train, x_test, ner_test, y_test = [], [], [], [], [], []
        for sentence, ner, tag in train:
            x_train += sentence
            ner_train += ner
            y_train += [item[-1] for item in tag]
        for sentence, ner, tag in test:
            x_test += sentence
            ner_test += ner
            y_test += [item[-1] for item in tag]

        return  x_train, ner_train, y_train, x_test, ner_test, y_test

    def transform_w2v(self,model,data,pad_size):
        new_data = []
        for entry in data:
            new_entry = []
            tokens = word_tokenize(entry)
            for token in tokens:
                try:
                    vector = np.mean(model.wv[token])
                except KeyError:
                    vector = np.mean(np.random.random(200))
                new_entry.append(vector)
            new_data.append(new_entry)

        for i in range(0,len(new_data)):
            size = len(new_data[i])
            if size > pad_size:
                new_data[i] = new_data[i][:pad_size]
            else:
                new_data[i]+= [0]*(pad_size - size)
        return new_data

    def add_all(self,data):
        all_occs = []
        for entry in data:
            if sum(entry) == 3:
                all_occs.append(0)
            else:
                all_occs.append(0)
        return all_occs

    def add_sent_len(self,data):
        new_data = []
        for sent in data:
            new_data.append([len(sent)])
        return new_data

    def add_struct(self,data):
        new_data = []
        struct_dict = {0:"None",
                    1:"Background",
                    2:"Introduction",
                    3:"Methods",
                    4:"Methodology"
                    4:"Results",
                    5:"Conclusions"}
        for sent in data:

        return new_data

    def add_punct(self,data):
        import string
        punct = [entry for entry in string.punctuation]
        punct_data = []
        for token in data:
            if token in punct or True in [char in punct for char in token]:
                punct_data.append([1])
            else:
                punct_data.append([0])
        return punct_data

    def add_to_features(self,all_features, feature):
        if len(all_features) == 0:
            return feature
        else:
            new_features = []
            for i in range(0,len(feature)):
                new_features.append(all_features[i]+feature[i])
            return new_features

    def add_features(self,train,test):
        x_train, ner_train, y_train, x_test, ner_test, y_test = self.featurize(train,test)
        if self.config.features['ners']:
            features_train = ner_train
            features_test = ner_test
        else:
            features_train = []
            features_test = []

        if self.config.features['has_P']:
            p_train = self.add_p(x_train)
            p_test = self.add_p(x_test)
            features_train = self.add_to_features(features_train, p_train)
            features_test = self.add_to_features(features_test, p_test)

        if self.config.features['has_IC']:
            ic_train = self.add_ic(x_train)
            ic_test = self.add_ic(x_test)
            features_train = self.add_to_features(features_train, ic_train)
            features_test = self.add_to_features(features_test, ic_test)

        if self.config.features['has_O']:
            o_train = self.add_o(x_train)
            o_test = self.add_o(x_test)
            features_train = self.add_to_features(features_train, o_train)
            features_test = self.add_to_features(features_test, o_test)

        if self.config.features['has_all']:
            all_train = self.add_all(ner_train)
            all_test = self.add_all(ner_test)
            features_train = self.add_to_features(features_train, all_train)
            features_test = self.add_to_features(features_test, all_test)

        if self.config.features['sent_len']:
            lens_train = self.add_sent_len(x_train)
            lens_test = self.add_sent_len(x_test)
            features_train = self.add_to_features(features_train,lens_train)
            features_test = self.add_to_features(features_test,lens_test)

        if self.config.features['struct']:
            struct_train = self.add_struct(x_train)
            struct_test = self.add_struct(x_test)
            features_train = self.add_to_features(features_train, struct_train)
            features_test = self.add_to_features(features_test, struct_test)

        if self.config.features['punct']:
            punct_train = self.add_punct(x_train)
            punct_test = self.add_punct(x_test)
            features_train = self.add_to_features(features_train, punct_train)
            features_test = self.add_to_features(features_test, punct_test)

        sent_lens = [len(s) for s in x_train]
        pad_size = int(np.round(np.mean(sent_lens) + (2*np.std(sent_lens))))

        if self.config.use_w2v:
            print("loading word2vec model...")
            w2v_model = KeyedVectors.load_word2vec_format(self.config.filename_glove, binary = True)
            print("model loaded.")
            x_train = self.transform_w2v(w2v_model, x_train, pad_size)
            x_test = self.transform_w2v(w2v_model, x_test, pad_size)
        else:
            #1hot count/tfidf vectorizer - not required
            exit(0)

        for i in range(0,len(x_train)):
            x_train[i]+=features_train[i]

        for i in range(0,len(x_test)):
            x_test[i]+=features_test[i]

        return [x_train, y_train], [x_test, y_test]

    def train_predict(self,train,test):
        if self.config.model.lower() == 'xgb':
            train_data = np.array(train[0])
            train_labels = np.array(train[1])
            test = np.array(test[0])
            self.model.fit(train_data, train_labels)
            return self.model.predict(test)
        else:
            self.model.fit(train[0],train[1])
        if self.config.predict_prob:
            return self.model.predict_proba(test[0])
        else:
            return self.model.predict(test[0])

    def score(self,test,predictions):
        f_score = f1_score(test[1], predictions, average=None)
        print('F1:', f_score)
        prec_score = precision_score(test[1], predictions)
        print('precision:', prec_score)
        rec_score = recall_score(test[1], predictions)
        print('recall:', rec_score)
        fpr, tpr, thresholds = roc_curve(test[1], predictions)
        print("FPR & TPR: ", [fpr, tpr])
        print('AUC:', auc(fpr, tpr))
        print('ROC AUC Score:', roc_auc_score(test[1], predictions))
        print('Precision Recall Curve:', precision_recall_curve(test[1], predictions))
        return [f_score,prec_score,rec_score]