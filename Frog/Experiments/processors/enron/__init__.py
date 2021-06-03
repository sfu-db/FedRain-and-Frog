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
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from mlsql import ModelManager, binary_ambiguity_count
from mlsql.processor import Processor, ComplaintRet, TiresiasOutput


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import numpy as np
import os

cwd = Path(__file__).resolve().parent


def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        with open(folder + a_file, "r", encoding="latin-1") as f:
            result = f.read()
            a_list.append(result)
    return a_list


class ENRONProcessor(Processor):
    QUERY: str = "SELECT COUNT(*) FROM D WHERE D.text like '%corrupt_word%' AND M(D.text)"
    seed: int
    vectorizer: CountVectorizer
    corrupted_word: str
    docs: List[str]

    Xtrain: tf.Tensor
    ytrain: tf.Tensor
    Xtest: tf.Tensor
    ytest: tf.Tensor
    ycorr: tf.Tensor
    corrsel: tf.Tensor
    enc: OneHotEncoder

    def __init__(self, seed: int, corrupted_word: str):

        super().__init__(seed=seed, corrupted_word=corrupted_word)

        self.vectorizer = CountVectorizer(min_df=15, stop_words="english")
        self.enc = OneHotEncoder(categories=[np.arange(2)], sparse=False)

        docs, y = load_dataset()
        self.docs = docs
        self.enc.fit(y.reshape(-1, 1))

        X = self.vectorizer.fit_transform(docs).toarray()

        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, train_size=0.8, stratify=y, random_state=self.auto_seed
        )

        word_id = self.vectorizer.vocabulary_[corrupted_word]
        corrupt_idx = Xtrain[:, word_id] > 0
        corrupt_idx = np.squeeze(corrupt_idx)
        ycorr = ytrain.copy()
        ycorr[corrupt_idx] = 1

        corrsel = np.zeros(len(ytrain), dtype=np.bool)
        corrsel[corrupt_idx] = True

        self.set_tensor_variables(
            Xtrain=Xtrain,
            ytrain=self.enc.transform(ytrain.reshape(-1, 1)),
            Xtest=Xtest,
            ytest=self.enc.transform(ytest.reshape(-1, 1)),
            ycorr=self.enc.transform(ycorr.reshape(-1, 1)),
        )
        self.set_tensor_variables(tf.bool, corrsel=corrsel)

    def ambiguity(self, manager: ModelManager) -> Optional[float]:
        word_id = self.vectorizer.vocabulary_[self.corrupted_word]
        targets = self.Xtest[:, word_id] > 0

        C, Q, _, _ = self.complain(manager, True)
        return binary_ambiguity_count(C.numpy()[0], Q.numpy()[0], targets.shape[0])

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        word_id = self.vectorizer.vocabulary_[self.corrupted_word]
        targets = self.Xtest[:, word_id] > 0
        C = tf.reduce_sum(self.ytest[targets][:, 1])
        proba = manager.predict_proba(self.Xtest[targets])
        if exact:
            Q = tf.cast(tf.reduce_sum(tf.argmax(proba, axis=1)), tf.float32)
        else:
            Q = tf.reduce_sum(proba)

        return ComplaintRet(AC=tf.reshape(C, (1,)), AQ=tf.reshape(Q, (1,)))

    def tiresias(self, manager: ModelManager, n: int = 1) -> TiresiasOutput:
        C, Q, _, _ = self.complain(manager, exact=True)
        word_id = self.vectorizer.vocabulary_[self.corrupted_word]
        targets = self.Xtest[:, word_id] > 0
        proba = manager.predict_proba(self.Xtest[targets])[:, 1].numpy()

        results = []
        for _ in range(n):
            if Q > C:
                disc = int((Q - C)[0])
                X = self.Xtest[targets][proba > 0.5]
                choice = np.random.choice(len(X), disc)
                results.append(
                    (tf.gather(X, choice), tf.tile([[0.0, 1.0]], (len(choice), 1)))
                )
            elif Q <= C:
                disc = int((C - Q)[0])
                X = self.Xtest[targets][proba <= 0.5]
                choice = np.random.choice(len(X), disc)
                results.append(
                    (tf.gather(X, choice), tf.tile([[1.0, 0.0]], (len(choice), 1)))
                )
        return results


def load_dataset():
    spam = init_lists(f"{cwd}/enron1/spam/")
    ham = init_lists(f"{cwd}/enron1/ham/")

    y = np.zeros(len(spam) + len(ham), dtype=np.int64)
    y[: len(spam)] = 1
    y[len(spam) :] = 0

    return spam + ham, y
