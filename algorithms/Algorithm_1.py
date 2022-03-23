from time import time

import keras
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt


class TensorFlow_CNN:
    def __init__(self, train_data, test_data, learning_rate, n_epochs, id, opt):
        self.history = None
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.id = id
        self.opt = opt
        self.model = 0

    def train(self):
        # Training Data
        xs_train, ys_train = self.train_data
        ys_train = to_categorical(ys_train)

        # define model architecture
        MODEL = "universal-sentence-encoder"
        VERSION = 4
        URL = "https://tfhub.dev/google/" + MODEL + "/" + str(VERSION)
        hub_layer2 = hub.KerasLayer(URL, output_shape=[512], input_shape=[], dtype=tf.string)
        self.model = keras.Sequential()
        self.model.add(hub_layer2)
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(2, activation='softmax'))

        # Define Optimizer
        if self.opt == "SGD":
            opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # define loss and optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

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
        print('------ TensorFlow - Universal Sentence Encoder Model ------')
        print(f'Duration Training: {duration_training} seconds')
        print('Accuracy Training: ', error)
        print("Number of Parameter: ", n_params)

        return duration_training, error

    def test(self):
        # Test Data
        xs_test, ys_test = self.test_data
        ys_test = to_categorical(ys_test)

        # Predict Data
        start_test = time()
        loss, error = self.model.evaluate(xs_test, ys_test, verbose=0)
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
