import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import gdown


class DataProvider:

    def get_Data(self):
        print("Read data ...")

        columns = ['polarity', 'id', 'date', 'query_string', 'twitter_user', 'tweet']

        data = pd.read_csv('data/training_data.csv', header=None, names=columns, encoding='latin-1')
        # test_data = pd.read_csv('data/test_data.csv', header=None, names=columns, encoding='latin-1')

        train_data, test_data = train_test_split(data, test_size=0.25, random_state=8)

        return train_data, test_data

    def import_data(self):
        PATH = "data/preprocessedTexts/"
        xs_train_lemma_file = PATH + "xs_train_lemma.csv"
        xs_test_lemma_file = PATH + "xs_test_lemma.csv"
        ys_train_lemma_file = PATH + "ys_train_lemma.csv"
        ys_test_lemma_file = PATH + "ys_test_lemma.csv"

        xs_train = pd.read_csv(xs_train_lemma_file, encoding='utf8')
        xs_test = pd.read_csv(xs_test_lemma_file, encoding='utf8')
        ys_train = pd.read_csv(ys_train_lemma_file, encoding='utf8')
        ys_test = pd.read_csv(ys_test_lemma_file, encoding='utf8')

        xs_train = xs_train["clean_tweet"].values
        xs_test = xs_test["clean_tweet"].values

        # Label Encoding
        print("Encode Labels ...")
        label_encoder = LabelEncoder()
        ys_train = label_encoder.fit_transform(ys_train)
        ys_test = label_encoder.fit_transform(ys_test)

        train_data = xs_train, ys_train
        test_data = xs_test, ys_test

        return train_data, test_data

