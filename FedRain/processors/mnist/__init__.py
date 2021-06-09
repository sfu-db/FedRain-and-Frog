# from .count_aggregation_complain import MNISTCountAggregationComplaintProcessor
# from .mnist_join_count import MNISTJoinAggregationProcessor
# from .mnist_join_rows import MNISTJoinRowsProcessor
# from .mnist_join_ambiguity import MNISTJoinAmbiguityProcessor
# from .mnist_groupby_count import MNISTGroupbyCountProcessor
# from .count_point_complaint import MNISTCountPointComplaintProcessor
# from .count_wrong_complaint import MNISTCountWrongComplaintProcessor
# from .count_aggregation_complain_short_eyesight import (
#     MNISTCountAggregationComplaintShortEyeSightProcessor,
# )
# from .mnist_join_count_non_disjoint import MNISTNonDisjointJoinAggregationProcessor


from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets

from mlsql import ComplaintRet, Processor
import mnist

# 6000 vs 25000
class MNISTProcessor(Processor):
    def __init__(self, mode: str, corr_rate: float = 0.3) -> None:
        assert mode in {"A", "B"}
        super().__init__()

        train_images = mnist.train_images()
        train_images = train_images.reshape([train_images.shape[0], -1]) / 256
        train_labels = mnist.train_labels() % 2

        test_images = mnist.test_images()
        test_images = test_images.reshape([test_images.shape[0], -1]) / 256
        test_labels = mnist.test_labels() % 2

        print(f"Dataset size: {train_images.shape}")

        Xtrain, Xquery, ytrain, yquery = train_test_split(
            train_images, train_labels, test_size=0.1, random_state=4096,
        )

        Xtest, ytest = test_images, test_labels

        (candidates,) = np.where(np.array(ytrain) == 1)

        np.random.seed(1024)
        corrupt_idx = np.random.choice(
            candidates, size=int(len(candidates) * corr_rate), replace=False
        )
        corrsel = np.full((len(ytrain),), False)
        corrsel[corrupt_idx] = True
        ycorr = ytrain.copy()
        ycorr[corrsel] = 0

        a_features = 300

        if mode == "A":
            Xtrain_a = Xtrain[:, :a_features]
            Xtest_a = Xtest[:, 0:a_features]
            Xquery_a = Xquery[:, 0:a_features]

            self.set_tensor_variables(
                Xtrain_full=Xtrain,
                Xtrain=Xtrain_a,
                ytrain=ytrain,
                ycorr=ycorr,
                Xtest=Xtest_a,
                ytest=ytest,
                Xquery=Xquery_a,
                Xquery_full=Xquery,
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

