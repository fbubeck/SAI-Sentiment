import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from time import time


class TensorFlow_ANN:
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

        # Convert y_train into one-hot format
        temp = []
        for i in range(len(ys_train)):
            temp.append(to_categorical(ys_train[i], num_classes=10))
        ys_train = np.array(temp)

        xs_train, xs_val, ys_train, ys_val = train_test_split(xs_train, ys_train, test_size=0.25, random_state=8)

        # normalize pixel values
        xs_train = xs_train / 255
        xs_val = xs_val / 255

        # define model architecture
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        # Define Optimizer
        if self.opt == "SGD":
            opt = keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=self.learning_rate)

        # define loss and optimizer
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Modeling
        start_training = time()
        self.history = self.model.fit(xs_train, ys_train, epochs=self.n_epochs, validation_data=(xs_val, ys_val),
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
        print('------ TensorFlow - ANN ------')
        print(f'Duration Training: {duration_training} seconds')
        print('Accuracy Training: ', error)
        print("Number of Parameter: ", n_params)

        return duration_training, error

    def test(self):
        # Test Data
        xs_test, ys_test = self.test_data

        # normalize pixel values
        xs_test = xs_test / 255

        # Convert y_test into one-hot format
        temp = []
        for i in range(len(ys_test)):
            temp.append(to_categorical(ys_test[i], num_classes=10))
        ys_test = np.array(temp)

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
