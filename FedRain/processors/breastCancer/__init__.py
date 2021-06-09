from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets

from mlsql import ComplaintRet, Processor

# 6000 vs 25000
class BreastCancerProcessor(Processor):
    def __init__(self, mode: str, corr_rate: float = 0.3) -> None:
        assert mode in {"A", "B"}
        super().__init__()

        x, y = datasets.load_breast_cancer(return_X_y=True)
        x = preprocessing.normalize(x, norm="l2")

        # rowids = np.arange(n) % x.shape[0]
        # colids = np.arange(m) % x.shape[1]

        # x = x[:, colids]
        # x = x[rowids]
        # y = y[rowids]

        x = preprocessing.normalize(x, norm="l2")

        random_state = int(tf.random.uniform([], maxval=2 ** 10, seed=1))
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            x, y, test_size=0.2, random_state=random_state,
        )
        Xtest, Xquery, ytest, yquery = train_test_split(
            Xtest, ytest, test_size=0.5, random_state=random_state,
        )

        print(f"Dataset size: {x.shape}")

        (candidates,) = np.where(ytrain == 1)

        np.random.seed(1024)
        corrupt_idx = np.random.choice(
            candidates, size=int(len(candidates) * corr_rate), replace=False
        )
        corrsel = np.full((len(ytrain),), False)
        corrsel[corrupt_idx] = True
        ycorr = ytrain.copy()

        ycorr[corrsel] = 0

        # Extract numpy arrays for labels
        #         ytrain = tf.one_hot(ytrain.astype(int), 2)
        #         ytest = tf.one_hot(ytest.astype(int), 2)
        a_features = x.shape[1] // 2

        if mode == "A":
            Xtrain_a = Xtrain[:, :a_features]
            Xtest_a = Xtest[:, 0:a_features]
            Xquery_a = Xquery[:, 0:a_features]

            self.set_tensor_variables(
                Xtrain=Xtrain_a,
                ytrain=ytrain,
                ycorr=ycorr,
                Xtest=Xtest_a,
                ytest=ytest,
                Xquery=Xquery_a,
                yquery=yquery,
                corrsel=corrsel,
            )

        else:
            Xtrain_b = Xtrain[:, a_features:]
            Xtest_b = Xtest[:, a_features:]
            Xquery_b = Xquery[:, a_features:]

            self.set_tensor_variables(
                Xtrain=Xtrain_b, Xtest=Xtest_b, Xquery=Xquery_b,
            )
