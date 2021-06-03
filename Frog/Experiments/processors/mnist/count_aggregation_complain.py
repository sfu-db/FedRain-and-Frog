# Standard imports
# Import for typing stuff
from typing import List, Optional, Tuple

import numpy as np

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

# To fix the single-class-count complaints we can use tiresias
from mlsql import ModelManager, minimal_single_count_fix_multi, multi_ambiguity_count
from mlsql.processor import ComplaintRet, Processor, TiresiasOutput
from mlsql.utils import safe_transform

from .utils import load_dataset, single_digit_corruption


class MNISTCountAggregationComplaintProcessor(Processor):

    Xtrain: tf.Tensor
    Xtest: tf.Tensor
    corrsel: tf.Tensor
    ytrain: tf.Tensor
    ycorr: tf.Tensor
    ytest: tf.Tensor

    enc: OneHotEncoder

    seed: int
    corrupted_class: int
    corrupted_to_class: int
    corruption_rate: float
    size: int

    """
    This processor relates to the query
    `SELECT COUNT(*) FROM D WHERE predict(D) = $digit`,
    Complaint `COUNT(*) == x`
    Corruption: `$digit to $another_digit`
    """

    def __init__(
        self,
        seed: int,
        corrupted_class: int,
        corrupted_to_class: int,
        corruption_rate: float,
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
        size : int
            The size of the dataset [default: 3000].
        """

        # First things first. Let us do some sanity checks
        assert corrupted_class in range(10)
        assert corrupted_to_class in range(10)
        assert corrupted_class != corrupted_to_class
        assert 0 < corruption_rate <= 1

        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__(
            seed=seed,
            corrupted_class=corrupted_class,
            corrupted_to_class=corrupted_to_class,
            corruption_rate=corruption_rate,
            size=size,
        )

        Xtrain, ytrain, Xtest, ytest = load_dataset(
            self.auto_seed, train_size=self.size, test_size=self.size
        )

        ycorr, corrsel = single_digit_corruption(
            ytrain,
            self.corrupted_class,
            self.corrupted_to_class,
            self.corruption_rate,
            self.auto_seed,
        )

        # To actualy conclude the corruption process
        # We need to convert all labels to one-hot representations

        # Here is our one-hot encoder
        self.enc = OneHotEncoder(categories=[np.arange(10)], sparse=False)
        # To initialize it we have
        self.enc.fit(ytrain.reshape(-1, 1))

        self.set_tensor_variables(
            Xtrain=Xtrain,
            Xtest=Xtest,
            ytrain=self.enc.transform(ytrain.reshape(-1, 1)),
            ytest=self.enc.transform(ytest.reshape(-1, 1)),
            ycorr=self.enc.transform(ycorr.reshape(-1, 1)),
        )

        self.corrsel = tf.constant(corrsel, name="corrsel")

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        C = tf.reduce_sum(self.ytest[:, self.corrupted_class])
        if exact:
            Q = tf.reduce_sum(
                tf.cast(manager.predict(self.Xtest) == self.corrupted_class, tf.float32)
            )
        else:
            # The relaxed count of the target class is the sum of the relevant probabilities
            Q = tf.reduce_sum(
                manager.predict_proba(self.Xtest)[:, self.corrupted_class]
            )

        return ComplaintRet(AC=tf.reshape(C, [1]), AQ=tf.reshape(Q, [1]))

    def ambiguity(self, manager: ModelManager) -> Optional[float]:
        AC, AQ, _, _ = self.complain(manager, exact=True)
        return multi_ambiguity_count(AC.numpy()[0], AQ.numpy()[0], len(self.Xtest), 10)

    # Returns the correction of the count based on how-to-provenance
    def tiresias(self, manager: ModelManager, n: int = 1) -> TiresiasOutput:
        # The predictions as a numpy array
        predictions_value = manager.predict(self.Xtest).numpy()

        # The count that needs to be enforced
        count_true = np.sum(self.ytest.numpy()[:, self.corrupted_class])

        # Fix the test predictions based on complaint
        predictions_fixed = minimal_single_count_fix_multi(
            predictions_value, 10, self.corrupted_class, count_true, n
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
