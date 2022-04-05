from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


class NeuralNetworkEmbeddingLayer:
    def __init__(self, train_data, test_data, learning_rate, n_epochs, opt, i):
        self.history = None
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.i = i
        self.opt = opt
        self.model = 0
        self.xs_test = None
        self.ys_test = None

    def train(self):
        # Training Data
        xs_train, ys_train = self.train_data
        xs_test, ys_test = self.test_data

        train = pd.DataFrame({"label": ys_train, "text": xs_train})
        test = pd.DataFrame({"label": ys_test, "text": xs_test})

        # number of words to consider in the dataset
        max_words = 20000
        tokenizer = Tokenizer(num_words=20000)
        train_texts = list(train['text'].values)
        test_texts = list(test['text'].values)
        # create the token index based on tweets
        tokenizer.fit_on_texts(train_texts)
        tokenizer.fit_on_texts(test_texts)

        # transform the tweets to sequences
        train_sequences = tokenizer.texts_to_sequences(train_texts)
        test_sequences = tokenizer.texts_to_sequences(test_texts)
        # set the maximum length of each tweet based on dataset
        lens = [len(x) for x in train_sequences]
        max_length_train = max(lens)

        lens = [len(x) for x in train_sequences]
        max_length_test = max(lens)

        xs_train = pad_sequences(train_sequences, maxlen=max_length_train)
        ys_train = train['label'].values

        self.xs_test = pad_sequences(test_sequences, maxlen=max_length_test)
        self.ys_test = test['label'].values

        # define model architecture
        embedding_dim = 100

        self.model = tf.keras.Sequential()
        self.model.add(layers.Embedding(max_words, embedding_dim, input_length=max_length_train))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.i, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        # Define Optimizer
        if self.opt == "SGD":
            opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        elif self.opt == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        else:
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # define loss and optimizer
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Modeling
        start_training = time()
        self.history = self.model.fit(xs_train, ys_train, epochs=self.n_epochs, validation_split=.2,
                                      batch_size=128, verbose=1)
        end_training = time()

        # Time
        duration_training = end_training - start_training
        duration_training = round(duration_training, 2)

        # Number of Parameter
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        n_params = trainableParams + nonTrainableParams

        # Prediction for Training mse
        loss, error = self.model.evaluate(xs_train, ys_train, verbose=0)
        error = round(error, 2)

        # Summary
        print('------ Embedding Layer + Neural Network ------')
        print('Number of Neurons: ', self.i)
        print(f'Duration Training: {duration_training} seconds')
        print('Accuracy Training: ', error)
        print("Number of Parameter: ", n_params)

        return duration_training, error, n_params

    def test(self):
        # Predict Data
        start_test = time()
        loss, error = self.model.evaluate(self.xs_test, self.ys_test, verbose=0)
        error = round(error, 2)
        end_test = time()

        # Time
        duration_test = end_test - start_test
        duration_test = round(duration_test, 2)

        print(f'Duration Inference: {duration_test} seconds')

        print("Accuracy Testing: %.2f" % error)
        print("")

        return duration_test, error

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
