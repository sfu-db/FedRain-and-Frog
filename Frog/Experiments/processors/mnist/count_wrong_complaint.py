# Standard imports
# Import for typing stuff
from typing import Optional, Tuple
from numbers import Number

import numpy as np

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

# To fix the single-class-count complaints we can use tiresias
from mlsql import ModelManager, multi_ambiguity_count, minimal_single_count_fix_multi
from mlsql.processor import ComplaintRet, Processor

from .utils import load_dataset, single_digit_corruption


class MNISTCountWrongComplaintProcessor(Processor):

    Xtrain: tf.Tensor
    Xtest: tf.Tensor
    corrsel: tf.Tensor
    ytrain: tf.Tensor
    ycorr: tf.Tensor
    ytest: tf.Tensor

    C: tf.Tensor

    enc: OneHotEncoder

    seed: int
    corrupted_class: int
    corrupted_to_class: int
    corruption_rate: float
    complaint_mode: str
    size: int
    """
    This processor relates to the query
    Query `SELECT COUNT(*) FROM D WHERE predict(D) = :digit`,
    Complaint `COUNT(*) == true_number`, but the complaint number is not accurate
    Corruption: Corrupt :digit to :another_digit
    """

    def __init__(
        self,
        seed: int,
        corrupted_class: int,
        corrupted_to_class: int,
        corruption_rate: float,
        complaint_mode: str = "correct",
        size: int = 3000,
    ):
        """
        Parameters
        ----------
        seed : int
            randomization seed for corruption
        corrupted_class : int
            class of MINST that is corrupted a.k.a. its training count is reduced
            It is also used as the target class of the count complaint
        corrupted_to_class : int
            corruptions are flipped to this MNIST label
        corruption_rate : float
            percentage of corrupted_class that will be flipped
        complaint_mode : str
            "correct", "wrong", "partial".
        """

        # First things first. Let us do some sanity checks
        assert corrupted_class in range(10)
        assert corrupted_to_class in range(10)
        assert corrupted_class != corrupted_to_class
        assert 0 < corruption_rate <= 1
        assert complaint_mode in ["correct", "wrong", "partial"]

        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__(
            seed=seed,
            corrupted_class=corrupted_class,
            corrupted_to_class=corrupted_to_class,
            corruption_rate=corruption_rate,
            complaint_mode=complaint_mode,
            size=size,
        )

        # It is now time to prepare the corruption
        # Let us set the numpy seed
        np.random.seed(seed)
        Xtrain, ytrain, Xtest, ytest = load_dataset(
            self.auto_seed, train_size=size, test_size=size
        )

        ycorr, corrsel = single_digit_corruption(
            ytrain,
            self.corrupted_class,
            self.corrupted_to_class,
            self.corruption_rate,
            self.auto_seed,
        )

        self.enc = OneHotEncoder(categories=[np.arange(10)], sparse=False)
        self.enc.fit(ytrain.reshape(-1, 1))

        self.set_tensor_variables(
            Xtrain=Xtrain,
            Xtest=Xtest,
            ytrain=self.enc.transform(ytrain.reshape(-1, 1)),
            ytest=self.enc.transform(ytest.reshape(-1, 1)),
            ycorr=self.enc.transform(ycorr.reshape(-1, 1)),
        )

        self.corrsel = tf.constant(corrsel, name="corrsel")

    def post_init(self, manager: ModelManager) -> None:
        ypred = manager.predict_proba(self.Xtest)

        Q = tf.reduce_sum(ypred[:, self.corrupted_class])
        C = tf.reduce_sum(self.ytest[:, self.corrupted_class])

        if self.complaint_mode == "wrong":
            if Q > C:
                C = tf.math.round(Q * 1.2)
            else:
                C = tf.math.round(Q * 0.8)
        elif self.complaint_mode == "correct":
            if Q > C:
                C = tf.math.round(C * 0.8)
            else:
                C = tf.math.round(C * 1.2)
        elif self.complaint_mode == "partial":
            C = tf.math.round((Q + C) / 2)
        else:
            raise RuntimeError("Unreachable")

        self.C = C

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        return complain_impl(self.Xtest, self.corrupted_class, self.C, manager, exact)

    def ambiguity(self, manager: ModelManager) -> Optional[float]:
        C = tf.reduce_sum(self.ytest[:, self.corrupted_class])
        # The model predictions
        test_probas = manager.predict_proba(self.Xtest)
        predictions = tf.argmax(test_probas, axis=1)
        Q = tf.reduce_sum(tf.cast(predictions == self.corrupted_class, tf.float32))

        return multi_ambiguity_count(int(C.numpy()), int(Q.numpy()), len(self.Xtest), 10)

    # Returns the correction of the count based on how-to-provenance
    def tiresias(self, manager: ModelManager) -> Tuple[tf.Tensor, tf.Tensor]:
        # The predictions as a numpy array
        predictions_value = manager.predict(self.Xtest).numpy()

        # Fix the test predictions based on complaint
        predictions_fixed = minimal_single_count_fix_multi(
            predictions_value, 10, self.corrupted_class, self.C.numpy()
        )

        # Detect the label differences between the fix and the predictions
        changed_mask = predictions_fixed != predictions_value
        changed_X = self.Xtest[changed_mask]
        changed_y = predictions_fixed[changed_mask]
        changed_y = tf.constant(
            self.enc.transform(changed_y.reshape(-1, 1)), dtype=tf.float32, name="tsy"
        )

        return changed_X, changed_y


@tf.function
def complain_impl(
    Xtest: tf.Tensor, corrupted_class: int, C: tf.Tensor, manager: ModelManager, exact: bool = False
) -> ComplaintRet:
    if exact:
        ypred = manager.predict(Xtest)
        Q = tf.reduce_sum(tf.cast(ypred == corrupted_class, tf.float32))
    else:
        Q = tf.reduce_sum(manager.predict_proba(Xtest)[:, corrupted_class])

    return ComplaintRet(AC=tf.reshape(C, [1]), AQ=tf.reshape(Q, [1]))
