import tensorflow as tf


class DataProvider():

    def get_Data(self):
        print("Preprocess data ...")
        mnist = tf.keras.datasets.mnist

        (xs_train, ys_train), (xs_test, ys_test) = mnist.load_data()

        train_data = xs_train, ys_train
        test_data = xs_test, ys_test

        return train_data, test_data
