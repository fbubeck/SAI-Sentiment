from nltk import RegexpTokenizer

from data import DataProvider
import keras
from sklearn.preprocessing import LabelEncoder
import re
import spacy
import nltk


class DataPreprocessing:

    def clean_text(self):
        data = DataProvider.DataProvider()
        train_data, test_data = data.get_Data()

        print("Preprocess data ...")

        # Drop unnecessary columns
        train_data = train_data[['id', 'polarity', 'tweet']].sample(150000)
        test_data = test_data[['id', 'polarity', 'tweet']].sample(50000)

        xs_train = train_data.drop("polarity", axis=1)  # drop labels for training set
        ys_train = train_data["polarity"].copy()

        xs_test = test_data.drop("polarity", axis=1)  # drop labels for training set
        ys_test = test_data["polarity"].copy()

        print("clean data ...")

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

        xs_train['clean_tweet'] = xs_train['tweet'].apply(lambda x: text_processing(x))
        xs_test['clean_tweet'] = xs_test['tweet'].apply(lambda x: text_processing(x))

        print("tokenize text...")
        tokenizer = RegexpTokenizer(r'w+')
        xs_train['clean_tweet'] = xs_train['clean_tweet'].apply(tokenizer.tokenize)
        xs_test['clean_tweet'] = xs_test['tweet'].apply(tokenizer.tokenize)

        print("stemming text ...")
        st = nltk.PorterStemmer()

        def stemming_on_text(data):
            text = [st.stem(word) for word in data]
            return data

        xs_train['clean_tweet'] = xs_train['clean_tweet'].apply(lambda x: stemming_on_text(x))
        xs_test['clean_tweet'] = xs_test['clean_tweet'].apply(lambda x: stemming_on_text(x))

        print("lemmatize data ...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        lm = nltk.WordNetLemmatizer()

        def lemmatizer_on_text(data):
            text = [lm.lemmatize(word) for word in data]
            return data

        xs_train["clean_tweet"] = xs_train["clean_tweet"].apply(lambda x: lemmatizer_on_text(x))
        xs_test['clean_tweet'] = xs_test['clean_tweet'].apply(lambda x: lemmatizer_on_text(x))

        # nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # def lemmatization(tweets):
        #     lemma_tweet = []
        #     for i in tweets:
        #         t = [token.lemma_ for token in nlp(i)]
        #         lemma_tweet.append(' '.join(t))
        #     return lemma_tweet

        # xs_train["clean_tweet"] = lemmatization(xs_train["clean_tweet"])
        # xs_test['clean_tweet'] = lemmatization(xs_test['clean_tweet'])

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

