# Standard imports
# Import for typing stuff
from typing import List, Optional, Tuple

import numpy as np

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

# To fix the single-class-count complaints we can use tiresias
from mlsql import ModelManager
from mlsql.processor import ComplaintRet, Processor, TiresiasOutput
from mlsql.utils import safe_transform
from sklearn.model_selection import train_test_split
from .utils import load_dataset, single_digit_corruption
import mnist


class MNISTProcessor(Processor):
    def __init__(self, seed: int, corr_rate: float = 0.3) -> None:
        super().__init__(seed=seed)

        train_images = mnist.train_images()
        train_images = train_images.reshape([train_images.shape[0], -1]) / 256
        train_labels = mnist.train_labels() % 2

        test_images = mnist.test_images()
        test_images = test_images.reshape([test_images.shape[0], -1]) / 256
        test_labels = mnist.test_labels() % 2

        print(f"Dataset size: {train_images.shape}")

        Xtrain, Xquery, ytrain, yquery = train_test_split(
            train_images, train_labels, test_size=0.1, random_state=self.auto_seed,
        )

        Xtest, ytest = test_images, test_labels

        (candidates,) = np.where(np.array(ytrain) == 1)

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
            XOOS=Xquery,
            yOOS=self.enc.transform(yquery.reshape(-1, 1)),
            corrsel=corrsel,
        )

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        C = tf.reduce_sum(self.ytest[:, 1])
        if exact:
            Q = tf.reduce_sum(tf.cast(manager.predict(self.Xtest), tf.float32))
        else:
            # The relaxed count of the target class is the sum of the relevant probabilities
            Q = tf.reduce_sum(manager.predict_proba(self.Xtest)[:, 1])

        return ComplaintRet(AC=tf.reshape(C, [1]), AQ=tf.reshape(Q, [1]))
