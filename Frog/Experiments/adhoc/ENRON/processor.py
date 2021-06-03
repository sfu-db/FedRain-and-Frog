from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import numpy as np
import os


def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        with open(folder + a_file, 'r', encoding='latin-1') as f:
            result = f.read()
            a_list.append(result)
    return a_list


class NLProcessor(object):

    def __init__(self):
        self.vectorizer = CountVectorizer(min_df=15, stop_words='english')

    def process_spam(self, spam, ham):
        Y = np.zeros(len(spam) + len(ham), dtype=np.int64)
        Y[:len(spam)] = 1
        Y[len(spam):] = 0
        Y = np.array(Y)
        return spam + ham, Y

    def learn_vocab(self, docs):
        """
        Learns a vocabulary from docs.
        """
        self.vectorizer.fit(docs)

    def get_bag_of_words(self, docs):
        """
        Takes in a list of documents and converts it into a bag of words
        representation. Returns X, a sparse matrix where each row is an example
        and each column is a feature (word in the vocab).
        """
        X = self.vectorizer.transform(docs)
        return X

    def load_spam(self):
        np.random.seed(0)

        spam = init_lists('../../data/spam/enron1/spam/')
        ham = init_lists('../../data/spam/enron1/ham/')

        docs, Y = self.process_spam(spam, ham)

        docs_train, docs_test, self.y_train, self.y_test = train_test_split(docs, Y, train_size=0.8, stratify=Y)

        self.learn_vocab(docs_train)
        self.X_train = self.get_bag_of_words(docs_train)
        self.X_test = self.get_bag_of_words(docs_test)

    def corrupt_random(self, seed, corrupt_rate):
        np.random.seed(seed)
        corruptions = int(len(self.y_train) * corrupt_rate)
        corrupt_idx = np.random.choice(len(self.y_train), size=corruptions, replace=False)

        ycrptd = self.y_train.copy()
        ycrptd[corrupt_idx] = (1 - ycrptd[corrupt_idx])

        return self.X_train, ycrptd

    # Choose a word and mark all documents in training as spam
    def corrupt_by_word(self, word):

        word_id = self.vectorizer.vocabulary_[word]
        targeted_documents = self.X_train[:, word_id] > 0
        targeted_documents = np.squeeze(targeted_documents.toarray())
        ycrptd = self.y_train.copy()

        ycrptd[targeted_documents] = 1

        return self.X_train, ycrptd
