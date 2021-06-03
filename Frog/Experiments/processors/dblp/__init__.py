from pathlib import Path
from typing import Any, Dict, Set, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from mlsql import (
    ComplaintRet,
    ModelManager,
    Processor,
    minimal_single_count_fix_multi,
    binary_ambiguity_count,
)
from tensorflow.python.framework.dtypes import DType

cwd = Path(__file__).resolve().parent


class DBLPProcessor(Processor):
    _corruption_types: Set[str] = set(["random", "positive", "negative", "titlelev"])

    enc: OneHotEncoder

    Xtrain: tf.Tensor
    Xtest: tf.Tensor
    ytrain: tf.Tensor
    ytest: tf.Tensor

    ycorr: tf.Tensor
    corrsel: tf.Tensor

    corruption_type: str
    corruption_rate: float
    neg_size: float
    train_size: float

    def __init__(
        self,
        seed: int,
        corruption_type: str,
        corruption_rate: float,
        neg_size: float = 1,
        train_size: float = 0.5,
        verbose: bool = False,
    ) -> None:
        assert corruption_type in self._corruption_types

        super().__init__(
            seed=seed,
            corruption_rate=corruption_rate,
            corruption_type=corruption_type,
            neg_size=neg_size,
            train_size=train_size,
        )

        df = pd.read_csv(f"{cwd}/dblp_scholar_features_wo_id.csv")
        dfl = pd.read_csv(f"{cwd}/data-dblp.csv")
        dfr = pd.read_csv(f"{cwd}/data-google_scholar.csv")

        npos = (df["label"] == 1).sum()
        nneg = (df["label"] == 0).sum()
        if verbose:
            print(f"# of Positive {npos}, # of negs {nneg}")

        if neg_size is not None:
            (pos_choice,) = np.where(df["label"] == 1)
            (neg_indexes,) = np.where(df["label"] == 0)
            to_sample = neg_size * len(pos_choice)
            neg_choice = np.random.choice(neg_indexes, size=to_sample, replace=False)
            reduced_choice = np.concatenate([neg_choice, pos_choice])
            sel_reduce = np.full(len(df), False)
            sel_reduce[reduced_choice] = True
            X = df.iloc[sel_reduce][df.columns[3:-1]].values
            y = df.iloc[sel_reduce][df.columns[-1]].values
        else:
            X = df[df.columns[3:-1]].values
            y = df[df.columns[-1]].values

        if verbose:
            print(f"Size of dblp {len(dfl)}, size of google scholar {len(dfr)}")
            print(f"Size of train + test: {len(X)}")

        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, train_size=train_size, test_size=1 - train_size, stratify=y
        )

        self.enc = OneHotEncoder(categories=[np.arange(2)], sparse=False)
        # To initialize it we have
        self.enc.fit(ytrain.reshape(-1, 1))

        # Corruptions
        if corruption_type == "random":
            corrupt_idx = np.random.choice(
                range(0, len(ytrain)),
                size=int(len(ytrain) * corruption_rate),
                replace=False,
            )
        elif corruption_type == "positive":
            (positives,) = np.where(ytrain == 1)
            corrupt_idx = np.random.choice(
                positives, size=int(len(positives) * corruption_rate), replace=False
            )
        elif corruption_type == "negative":
            (negatives,) = np.where(ytrain == 0)
            corrupt_idx = np.random.choice(
                negatives, size=int(len(negatives) * corruption_rate), replace=False
            )
        elif corruption_type == "titlelev":
            (corrupt_idx,) = np.where((Xtrain[:, 3] > 5) & (ytrain == 1))
        else:
            raise RuntimeError(f"corrupt '{corruption_type}' type not supported")

        corrsel = np.zeros(len(ytrain), dtype=np.bool)
        corrsel[corrupt_idx] = True
        ycorr = ytrain.copy()
        ycorr[corrsel] = 1 - ycorr[corrsel]

        self.set_tensor_variables(
            tf.float32,
            Xtrain=Xtrain,
            ytrain=self.enc.transform(ytrain.reshape(-1, 1)),
            Xtest=Xtest,
            ytest=self.enc.transform(ytest.reshape(-1, 1)),
            ycorr=self.enc.transform(ycorr.reshape(-1, 1)),
        )

        self.set_tensor_variables(tf.bool, corrsel=corrsel)

    def ambiguity(self, manager: ModelManager) -> Optional[float]:
        Q, C, _, _ = self.complain(manager, True)
        return binary_ambiguity_count(C.numpy()[0], Q.numpy()[0], self.Xtest.shape[0])

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        return complain_impl(manager, self.Xtest, self.ytest, exact)

    # Returns the correction of the count based on how-to-provenance
    def tiresias(
        self, manager: ModelManager, n: int = 1
    ) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        # The predictions as a numpy array
        predictions_value = manager.predict(self.Xtest).numpy()

        # The count that needs to be enforced
        count_true = np.sum(self.ytest.numpy()[:, 1])

        # Fix the test predictions based on complaint
        predictions_fixed = minimal_single_count_fix_multi(
            predictions_value, 2, 1, count_true, n
        )

        results = []
        for i in range(predictions_fixed.shape[0]):
            # Detect the label differences between the fix and the predictions
            changed_mask = predictions_fixed[i] != predictions_value
            changed_X = self.Xtest[changed_mask]
            changed_y = predictions_fixed[i, changed_mask]
            changed_y = tf.constant(
                self.enc.transform(changed_y.reshape(-1, 1)),
                dtype=tf.float32,
                name="tsy",
            )
            results.append((changed_X, changed_y))

        return results


@tf.function
def complain_impl(
    manager: ModelManager, Xtest: tf.Tensor, ytest: tf.Tensor, exact: bool = False
) -> ComplaintRet:
    C = tf.reduce_sum(ytest[:, 1])
    proba = manager.predict_proba(Xtest)

    if exact:
        Q = tf.cast(tf.reduce_sum(tf.argmax(proba, axis=1)), tf.float32)
    else:
        Q = tf.reduce_sum(proba[:, 1])

    return ComplaintRet(AC=tf.reshape(C, (1,)), AQ=tf.reshape(Q, (1,)))
