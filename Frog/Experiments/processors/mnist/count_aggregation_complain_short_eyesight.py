# Standard imports
# Import for typing stuff
from typing import Optional, Tuple

import numpy as np

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

# To fix the single-class-count complaints we can use tiresias
from mlsql import ModelManager, minimal_single_count_fix_multi, multi_ambiguity_count
from mlsql.processor import ComplaintRet, Processor

from .utils import load_dataset, blur_annotator_corruption
from sklearn.model_selection import train_test_split


class MNISTCountAggregationComplaintShortEyeSightProcessor(Processor):

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
    Corruption: Short eyesight annotator
    """

    def __init__(
        self,
        seed: int,
        corrupted_class: int,
        corrupted_to_class: int,
        sigma: float,
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
        sigma : float
            Kernel size of the gaussian blur
        """

        # First things first. Let us do some sanity checks
        assert corrupted_class in range(10)
        assert corrupted_to_class in range(10)
        assert corrupted_class != corrupted_to_class
        assert sigma > 0

        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__(
            seed=seed,
            corrupted_class=corrupted_class,
            corrupted_to_class=corrupted_to_class,
            sigma=sigma,
            size=size,
        )

        Xpre, ypre, X, y = load_dataset(self.auto_seed, train_size=size, test_size=size)

        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X,
            y,
            train_size=3000,
            test_size=3000,
            stratify=y,
            random_state=self.auto_seed,
        )

        ycorr, corrsel = blur_annotator_corruption(
            Xpre, ypre, Xtrain, ytrain, sigma, self.auto_seed
        )

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
        return complain_impl(
            self.Xtest, self.ytest, self.corrupted_class, manager, exact
        )

    def ambiguity(self, manager: ModelManager) -> Optional[float]:
        C, Q, _, _ = self.complain(manager, exact=True)
        return multi_ambiguity_count(
            int(C.numpy()[0]), Q.numpy()[0], len(self.Xtest), 10
        )

    # Returns the correction of the count based on how-to-provenance
    def tiresias(self, manager: ModelManager) -> Tuple[tf.Tensor, tf.Tensor]:
        # The predictions as a numpy array
        predictions_value = manager.predict(self.Xtest).numpy()

        # The count that needs to be enforced
        count_true = np.sum(self.ytest.numpy()[:, self.corrupted_class])

        # Fix the test predictions based on complaint
        predictions_fixed = minimal_single_count_fix_multi(
            predictions_value, 10, self.corrupted_class, count_true
        )

        # Detect the label differences between the fix and the predictions
        changed_mask = predictions_fixed != predictions_value
        changed_X = tf.boolean_mask(self.Xtest, changed_mask)
        changed_y = tf.boolean_mask(predictions_fixed, changed_mask)
        changed_y = tf.constant(
            self.enc.transform(changed_y.numpy().reshape(-1, 1)),
            dtype=tf.float32,
            name="tsy",
        )

        return changed_X, changed_y


@tf.function
def complain_impl(
    Xtest: tf.Tensor,
    ytest: tf.Tensor,
    corrupted_class: int,
    model: ModelManager,
    exact: bool = False,
) -> ComplaintRet:
    C = tf.reduce_sum(ytest[:, corrupted_class])
    if exact:
        Q = tf.reduce_sum(tf.cast(model.predict(Xtest) == corrupted_class, tf.float32))
    else:
        # The relaxed count of the target class is the sum of the relevant probabilities
        Q = tf.reduce_sum(model.predict_proba(Xtest)[:, corrupted_class])

    return ComplaintRet(AC=tf.reshape(C, [1]), AQ=tf.reshape(Q, [1]))
