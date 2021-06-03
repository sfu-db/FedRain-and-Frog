# Standard imports
# Import for typing stuff
from typing import List, Tuple, Union

# Library that loads mnist
import mnist
import numpy as np

# Library for converting mnist images to floats
from skimage import img_as_float

# MNIST has a prespecified train/test split
# But is a "relatively" big dataset (at least for now)
# We use train_test_split for subsampling
from sklearn.model_selection import train_test_split

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

# To fix the single-class-count complaints we can use tiresias
from mlsql import ModelManager, minimal_mutli_count_fix_multi
from mlsql.processor import ComplaintRet, Processor, TiresiasOutput
from mlsql.utils import safe_transform
from .utils import load_dataset, single_digit_corruption


class MNISTGroupbyCountProcessor(Processor):
    """
    This processor relates to the query
    SELECT COUNT(*), predict(D.img) FROM D GROUP BY predict(D.img)
    The corruption is corrupt $from_digit to $to_digit
    The complaint is on the count on some groups, 
    they can be either related to $from_digit or $to_digit, or totally not.
    """

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
    complaint_classes: tf.Tensor
    size: int

    def __init__(
        self,
        seed: int,
        corrupted_class: int,
        corrupted_to_class: int,
        corruption_rate: float,
        complaint_classes: List[int],
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
        complaint_classes: List[int]
            classes against which we issue a count complaint
        """

        # First things first. Let us do some sanity checks
        assert corrupted_class in range(10)
        assert corrupted_to_class in range(10)
        assert corrupted_class != corrupted_to_class
        assert 0 < corruption_rate <= 1
        assert set(complaint_classes).issubset(range(10))

        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__(
            seed=seed,
            corrupted_class=corrupted_class,
            corrupted_to_class=corrupted_to_class,
            corruption_rate=corruption_rate,
            complaint_classes=complaint_classes,
            size=size,
        )

        self.complaint_classes_tensor = tf.constant(
            complaint_classes, name="complaint_classes"
        )

        Xtrain, ytrain, Xtest, ytest = load_dataset(
            self.auto_seed, train_size=size, test_size=size
        )

        # It is now time to prepare the corruption
        # Let us set the numpy seed
        np.random.seed(seed)

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
        self.set_tensor_variables(dtype=tf.bool, corrsel=corrsel)

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        return complain_impl(
            self.Xtest, self.ytest, self.complaint_classes_tensor, manager, exact
        )

    # Returns the correction of the count based on how-to-provenance
    def tiresias(self, manager: ModelManager) -> TiresiasOutput:
        # The predictions as a numpy array
        predictions_value = manager.predict(self.Xtest).numpy()

        count_dict = dict()
        # The counts that needs to be enforced
        for complaint_class in self.complaint_classes_tensor.numpy():
            count_true = self.ytest.numpy()[:, complaint_class].sum()
            count_dict[complaint_class] = count_true

        # Fix the test predictions based on complaint
        predictions_fixed = minimal_mutli_count_fix_multi(
            predictions_value, 10, count_dict
        )

        # Detect the label differences between the fix and the predictions
        changed_idx, = np.where(predictions_fixed != predictions_value)
        changed_y = predictions_fixed[changed_idx]
        changed_y = safe_transform(self.enc, changed_y)
        changed_y = tf.constant(changed_y, dtype=tf.float32, name="tsY")
        changed_X = tf.gather(self.Xtest, changed_idx, name="tsX")

        return changed_X, changed_y


@tf.function
def complain_impl(
    Xtest: tf.Tensor,
    ytest: tf.Tensor,
    complaint_classes: tf.Tensor,
    manager: ModelManager,
    exact: bool = False,
) -> ComplaintRet:
    assert complaint_classes.shape.rank == 1

    # Model predictions
    if exact:
        ypred = tf.one_hot(manager.predict(Xtest), 10, dtype=tf.float32)
    else:
        ypred = manager.predict_proba(Xtest)

    Cs = tf.reduce_sum(tf.gather(ytest, complaint_classes, axis=1), axis=0)
    Qs = tf.reduce_sum(tf.gather(ypred, complaint_classes, axis=1), axis=0)

    return ComplaintRet(AC=Cs, AQ=Qs)
