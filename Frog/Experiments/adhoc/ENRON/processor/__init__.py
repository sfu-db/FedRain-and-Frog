from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

import mnist
import numpy as np
from scipy.sparse import csr_matrix
from skimage import img_as_float
from skimage.filters import gaussian
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import tensorflow as tf
from mlsql.logreg import LogisticRegression
from mlsql.processor import Processor

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import numpy as np
import os

cwd = Path(__file__).resolve().parent


def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        with open(folder + a_file, 'r', encoding='latin-1') as f:
            result = f.read()
            a_list.append(result)
    return a_list


class ENRONProcessor(Processor):
    seed: int
    vectorizer: CountVectorizer
    corrupt_word: str
    docs: List[str]

    Xtrain: np.ndarray
    ytrain: np.ndarray
    Xtest: np.ndarray
    ytest: np.ndarray
    ycrptd: np.ndarray
    sel_corrupt: np.ndarray

    def __init__(self, seed: int, corrupt_word: str, corrupt_rate: float, sparse: bool = True):

        self.params = {
            "corrupt_word": corrupt_word,
            "corrupt_rate": corrupt_rate
        }
        self.seed = seed
        self.corrupt_word = corrupt_word

        self.vectorizer = CountVectorizer(min_df=15, stop_words='english')

        docs, y = load_dataset()
        self.docs = docs
        X = self.vectorizer.fit_transform(docs)
        if not sparse:
            X = X.toarray()

        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(X, y, train_size=0.8, stratify=y, random_state=seed)

        np.random.seed(seed)

        word_id = self.vectorizer.vocabulary_[corrupt_word]
        corrupt_idx = self.Xtrain[:, word_id] > 0
        if sparse:
            corrupt_idx = corrupt_idx.toarray()
        corrupt_idx = np.squeeze(corrupt_idx)
        self.ycrptd = self.ytrain.copy()
        self.ycrptd[corrupt_idx] = 1

        self.sel_corrupt = np.zeros(len(self.ytrain), dtype=np.bool)
        self.sel_corrupt[corrupt_idx] = True

    def get_corrupted(self):
        return self.Xtrain, self.ycrptd, self.sel_corrupt

    def get_clean(self):
        return self.Xtrain, self.ytrain

    def complain(self, model: LogisticRegression, return_value: bool = False):

        word_id = self.vectorizer.vocabulary_[self.corrupt_word]
        targets = self.Xtest[:, word_id] > 0
        C = (self.ytest[targets] == 1).sum()
        proba = model.predict_proba_tensor(self.Xtest[targets])
        Q = tf.reduce_sum(proba)
        nquery = len(targets)

        if return_value:
            return 1 / nquery * tf.norm(C - Q), C, Q
        else:
            return 1 / nquery * tf.norm(C - Q)


def load_dataset():
    spam = init_lists(f"{cwd}/enron1/spam/")
    ham = init_lists(f"{cwd}/enron1/ham/")

    y = np.zeros(len(spam) + len(ham), dtype=np.int64)
    y[:len(spam)] = 1
    y[len(spam):] = 0

    return spam + ham, y
