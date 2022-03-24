import multiprocessing
from time import time

import nltk
import numpy as np
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from matplotlib import pyplot as plt
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils.np_utils import to_categorical


class TextClassifier_DBOW:
    def __init__(self, train_data, test_data, n_epochs, solver, c, penalty, id):
        self.history = None
        self.train_data = train_data
        self.test_data = test_data
        self.n_epochs = n_epochs
        self.solver = solver
        self.c = c
        self.penalty = penalty
        self.id = id
        self.model = 0
        self.ys_test = None
        self.xs_test = None

    def train(self):
        cores = multiprocessing.cpu_count()

        # Training Data
        xs_train, ys_train = self.train_data
        self.xs_test, self.ys_test = self.test_data

        train = pd.DataFrame({"label": ys_train, "text": xs_train})
        test = pd.DataFrame({"label": self.ys_test, "text": self.xs_test})

        nltk.download('punkt')

        # Text Tokenization
        # print("tokenize text...")
        # train_tagged = train.apply(lambda r: TaggedDocument(words=TextClassifier_DBOW.tokenize_text(r['text']),
        #                                                     tags=[r.label]), axis=1)
        # test_tagged = test.apply(lambda r: TaggedDocument(words=TextClassifier_DBOW.tokenize_text(r['text']),
        #                                                   tags=[r.label]), axis=1)
        # Modeling
        start_training = time()
        #
        # # Distributed Bag of Words
        # print("train Doc2Vec-DBOW model...")
        # model_dbow = Doc2Vec(workers=cores)
        # model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
        #
        # ys_train, xs_train, self.ys_test, self.xs_test = TextClassifier_DBOW.train_Doc2Vec(model_dbow,
        #                                                                                    train_tagged,
        #                                                                                    test_tagged, self.n_epochs)

        print("vectorize text ...")
        vectoriser = TfidfVectorizer(max_features=100000)
        vectoriser.fit(xs_train)

        xs_train = vectoriser.transform(xs_train)
        self.xs_test = vectoriser.transform(self.xs_test)

        # self.model = LogisticRegression(verbose=1, solver=self.solver, C=self.c, penalty=self.penalty, max_iter=100000)
        self.model = RandomForestClassifier(n_estimators=1000)
        print("Build Model ...")
        self.model.fit(xs_train, ys_train)

        end_training = time()

        # Time
        duration_training = end_training - start_training
        duration_training = round(duration_training, 2)

        # # Number of Parameter
        # trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        # nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        # n_params = trainableParams + nonTrainableParams

        # Prediction for Training mse
        y_pred = self.model.predict(xs_train)
        error = mean_squared_error(ys_train, y_pred)
        error = round(error, 2)

        # Summary
        print('------ DBOW-Model + LogReg ------')
        print(f'Duration Training: {duration_training} seconds')
        print('Accuracy Training: ', error)
        # print("Number of Parameter: ", n_params)

        return duration_training, error

    def test(self):
        # Test Data

        # Predict Data
        start_test = time()
        y_pred = self.model.predict(self.xs_test)
        error = mean_squared_error(self.ys_test, y_pred)
        error = round(error, 2)
        end_test = time()

        # Time
        duration_test = end_test - start_test
        duration_test = round(duration_test, 2)

        print(f'Duration Inference: {duration_test} seconds')

        print("Accuracy Testing: %.2f" % error)
        print("")

        return duration_test, error

    @staticmethod
    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

    @staticmethod
    def train_Doc2Vec(model, train_tagged, test_tagged, n_epochs):
        for epoch in range(n_epochs):
            model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]),
                        total_examples=len(train_tagged.values), epochs=1)
            model.alpha -= 0.001
            model.min_alpha = model.alpha

        y_train, X_train = TextClassifier_DBOW.vec_for_learning(model, train_tagged)
        y_test, X_test = TextClassifier_DBOW.vec_for_learning(model, test_tagged)

        return y_train, X_train, y_test, X_test

    @staticmethod
    def vec_for_learning(model, tagged_docs):
        sents = tagged_docs.values
        targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
        return targets, regressors

    def plot(self):
        # Plot loss and val_loss
        px = 1 / plt.rcParams['figure.dpi']
        __fig = plt.figure(figsize=(800 * px, 600 * px))
        plt.plot(self.history.history['loss'], 'blue')
        plt.plot(self.history.history['val_loss'], 'red')
        plt.title('Neural Network Training loss history')
        plt.ylabel('loss (log scale)')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        url = f"plots/training-history/TensorFlow_{self.id}_Loss-Epochs-Plot.png"
        plt.savefig(url)
        # plt.show()
        print("TensorFlow loss Plot saved...")
        print("")
