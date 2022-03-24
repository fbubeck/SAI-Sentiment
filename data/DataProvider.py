import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


class DataProvider:

    def get_Data(self):
        print("Read data ...")

        columns = ['polarity', 'id', 'date', 'query_string', 'twitter_user', 'tweet']

        data = pd.read_csv('data/training_data.csv', header=None, names=columns, encoding='latin-1')
        # test_data = pd.read_csv('data/test_data.csv', header=None, names=columns, encoding='latin-1')

        train_data, test_data = train_test_split(data, test_size=0.25, random_state=8)

        return train_data, test_data





