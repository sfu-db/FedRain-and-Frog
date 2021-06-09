from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from mlsql import ModelManager

from mlsql import ComplaintRet, Processor


class BreastCancerProcessor(Processor):
    def __init__(self, seed, corr_rate: float = 0.3) -> None:
        super().__init__(seed=seed)

        x, y = datasets.load_breast_cancer(return_X_y=True)
        x = preprocessing.normalize(x, norm="l2")

        x = preprocessing.normalize(x, norm="l2")

        Xtrain, Xtest, ytrain, ytest = train_test_split(
            x, y, test_size=0.2, random_state=self.auto_seed,
        )
        Xtest, Xquery, ytest, yquery = train_test_split(
            Xtest, ytest, test_size=0.5, random_state=self.auto_seed,
        )

        print(f"Dataset size: {x.shape}")

        (candidates,) = np.where(ytrain == 1)

        np.random.seed(self.auto_seed)
        corrupt_idx = np.random.choice(
            candidates, size=int(len(candidates) * corr_rate), replace=False
        )
        corrsel = np.full((len(ytrain),), False)
        corrsel[corrupt_idx] = True
        ycorr = ytrain.copy()

        ycorr[corrsel] = 0

        self.enc = OneHotEncoder(categories=[np.arange(2)], sparse=False)
        self.enc.fit(ytrain.reshape(-1, 1))

        self.set_tensor_variables(
            Xtrain=Xtrain,
            ytrain=self.enc.transform(ytrain.reshape(-1, 1)),
            ycorr=self.enc.transform(ycorr.reshape(-1, 1)),
            Xtest=Xtest,
            ytest=self.enc.transform(ytest.reshape(-1, 1)),
            corrsel=corrsel,
            XOOS=Xquery,
            yOOS=self.enc.transform(yquery.reshape(-1, 1)),
        )

    def complain(self, manager, exact=False):
        C = tf.reduce_sum(self.ytest[:, 1])
        if exact:
            Q = tf.reduce_sum(tf.cast(manager.predict(self.Xtest), tf.float32))
        else:
            # The relaxed count of the target class is the sum of the relevant probabilities
            Q = tf.reduce_sum(manager.predict_proba(self.Xtest)[:, 1])

        return ComplaintRet(AC=tf.reshape(C, [1]), AQ=tf.reshape(Q, [1]))
