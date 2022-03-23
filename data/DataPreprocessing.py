from data import DataProvider
import keras
from sklearn.preprocessing import LabelEncoder
import re
import spacy


class DataPreprocessing:

    def clean_text(self):
        data = DataProvider.DataProvider()
        train_data, test_data = data.get_Data()

        print("Preprocess data ...")

        # Drop unnecessary columns
        train_data = train_data[['id', 'polarity', 'tweet']].sample(150000)
        test_data = test_data[['id', 'polarity', 'tweet']]

        xs_train = train_data.drop("polarity", axis=1)  # drop labels for training set
        ys_train = train_data["polarity"].copy()

        xs_test = test_data.drop("polarity", axis=1)  # drop labels for training set
        ys_test = test_data["polarity"].copy()

        def text_processing(tweet):
            # remove https links
            clean_tweet = re.sub(r'http\S+', '', tweet)
            # remove punctuation marks
            punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
            clean_tweet = ''.join(ch for ch in clean_tweet if ch not in set(punctuation))
            # convert text to lowercase
            clean_tweet = clean_tweet.lower()
            # remove numbers
            clean_tweet = re.sub('\d', ' ', clean_tweet)
            # remove whitespaces
            clean_tweet = ' '.join(clean_tweet.split())
            return clean_tweet

        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        def lemmatization(tweets):
            lemma_tweet = []
            for i in tweets:
                t = [token.lemma_ for token in nlp(i)]
                lemma_tweet.append(' '.join(t))
            return lemma_tweet

        print("clean data ...")
        xs_train['clean_tweet'] = xs_train['tweet'].apply(lambda x: text_processing(x))
        xs_test['clean_tweet'] = xs_test['tweet'].apply(lambda x: text_processing(x))

        print("lemmatize data ...")
        xs_train["clean_tweet"] = lemmatization(xs_train["clean_tweet"])
        xs_test['clean_tweet'] = lemmatization(xs_test['clean_tweet'])

        # Defining Files
        PATH = "data/preprocessedTexts/"
        xs_train_lemma_file = PATH + "xs_train_lemma.csv"
        xs_test_lemma_file = PATH + "xs_test_lemma.csv"
        ys_train_lemma_file = PATH + "ys_train_lemma.csv"
        ys_test_lemma_file = PATH + "ys_test_lemma.csv"

        # Save to files
        xs_train.to_csv(xs_train_lemma_file, index=False)
        xs_test.to_csv(xs_test_lemma_file, index=False)
        ys_train.to_csv(ys_train_lemma_file, index=False)
        ys_test.to_csv(ys_test_lemma_file, index=False)

