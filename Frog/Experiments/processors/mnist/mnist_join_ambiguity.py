# Standard imports
# Library that loads mnist
# Import for typing stuff
from typing import List, Tuple, Union

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

# To fix the join-count complaints we can use tiresias
from mlsql import ModelManager, Processor, minimal_join_rows_fix_multi
from mlsql.processor import ComplaintRet
from mlsql.utils import safe_transform

from .utils import single_digit_corruption, load_dataset


class MNISTJoinAmbiguityProcessor(Processor):

    Xtrain: tf.Tensor
    Xtest: tf.Tensor
    corrsel: tf.Tensor
    ytrain: tf.Tensor
    ycorr: tf.Tensor
    ytest: tf.Tensor
    Xleft: tf.Tensor
    Xright: tf.Tensor
    ycommon: tf.Tensor
    Xdirect: tf.Tensor
    ydirect: tf.Tensor

    enc: OneHotEncoder

    seed: int
    corrupted_class: int
    corrupted_to_class: int
    corruption_rate: float
    direct_rate: float
    size: int

    def __init__(
        self,
        seed: int,
        corrupted_class: int,
        corrupted_to_class: int,
        corruption_rate: float,
        direct_rate: float,
        size: int = 3000,
    ):
        """
        Parameters
        ----------
        seed : int
            randomization seed for corruption
        corrupted_class : iint
            class of MINST that is corrupted a.k.a. its training count is reduced
            It is also used as the target class of the count complaint
        corrupted_to_class : int
            corruptions are flipped to this MNIST label
        corruption_rate : float
            percentage of corrupted_class that will be flipped
        direct_rate : float
            percentage of complaints that are specified directly and not through joins.
        """

        # First things first. Let us do some sanity checks
        assert corrupted_class in range(10)
        assert corrupted_to_class in range(10)
        assert corrupted_class != corrupted_to_class
        assert 0 < corruption_rate <= 1.0
        assert 0 < direct_rate <= 1.0

        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__(
            seed=seed,
            corrupted_class=corrupted_class,
            corrupted_to_class=corrupted_to_class,
            corruption_rate=corruption_rate,
            direct_rate=direct_rate,
            size=size,
        )

        Xtrain, ytrain, Xtest, ytest = load_dataset(1, train_size=size, test_size=10000)

        ycorr, corrsel = single_digit_corruption(
            ytrain, corrupted_class, corrupted_to_class, corruption_rate, self.auto_seed
        )

        # We are not ready yet
        # We need to convert all labels to one-hot representations

        # Here is our one-hot encoder
        self.enc = OneHotEncoder(categories=[np.arange(10)], sparse=False)
        # To initialize it we have
        self.enc.fit(ytrain.reshape(-1, 1))

        self.set_tensor_variables(
            Xtrain=Xtrain,
            Xtest=Xtest,
            ytrain=self.enc.transform(ytrain.reshape(-1, 1)),
            # The test labels not be one-hot encoded
            # See post_init for more details
            ytest=ytest,
            ycorr=self.enc.transform(ycorr.reshape(-1, 1)),
        )

        self.set_tensor_variables(tf.bool, corrsel=corrsel)

    def post_init(self, manager: ModelManager) -> None:

        # Split the test set (L vs R)
        Xleft = self.Xtest[self.ytest == self.corrupted_class]
        Xright = self.Xtest[self.ytest == self.corrupted_to_class]

        # Find the classifications of Xleft and Xright
        yleft_pred, yright_pred = manager.predict(Xleft), manager.predict(Xright)

        # Match wrong classifications of X_left with correct classifications of X_right
        Xleft_wrong = Xleft[yleft_pred == self.corrupted_to_class]
        Xright_correct = Xright[yright_pred == self.corrupted_to_class]

        # Partition the complaints on the join rows
        Xdirect_one, ydirect_one, Xleft_wrong, Xright_correct = direct_join_partition(
            Xleft_wrong.numpy(),
            Xright_correct.numpy(),
            self.corrupted_class,
            self.corrupted_to_class,
            self.direct_rate,
        )

        # Match wrong classifications of Xright with correct classifications of Xleft
        Xleft_correct = Xleft[yleft_pred == self.corrupted_class]
        Xright_wrong = Xright[yright_pred == self.corrupted_class]

        # Partition the complaints on the join rows
        Xdirect_two, ydirect_two, Xleft_correct, Xright_wrong = direct_join_partition(
            Xleft_correct.numpy(),
            Xright_wrong.numpy(),
            self.corrupted_class,
            self.corrupted_to_class,
            self.direct_rate,
        )

        # Build the final join rows complaint

        # Stack the two cases together to create the new X_left, X_right
        self.Xleft = tf.concat([Xleft_wrong, Xleft_correct], axis=0, name="Xleft")
        self.Xright = tf.concat([Xright_correct, Xright_wrong], axis=0, name="Xright")

        # Generate the predicted labels for Tiresias
        part_one = np.full(Xleft_wrong.shape[0], self.corrupted_to_class, dtype=int)
        part_two = np.full(Xleft_correct.shape[0], self.corrupted_class, dtype=int)

        # The final labels are here (No need for one-hot encoding)
        self.ycommon = tf.concat([part_one, part_two], axis=0, name="ycommon")

        # Build the final direct complaint
        self.Xdirect = tf.concat([Xdirect_one, Xdirect_two], name="Xdirect", axis=0)
        ydirect = np.concatenate([ydirect_one, ydirect_two])

        self.ydirect = tf.constant(
            safe_transform(self.enc, ydirect), dtype=tf.float32, name="ydirect"
        )

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        return complain_impl(
            self.Xleft, self.Xright, self.Xdirect, self.ydirect, manager, exact
        )

    def tiresias(self, manager: ModelManager) -> Tuple[tf.Tensor, tf.Tensor]:
        ycommon = self.ycommon.numpy()
        left_new, right_new = minimal_join_rows_fix_multi(ycommon, 10)

        # Find the changes
        changed_left_idx, = np.where(left_new != ycommon)
        changed_right_idx, = np.where(right_new != ycommon)

        changed_X_left = self.Xleft.numpy()[changed_left_idx, :]
        changed_X_right = self.Xright.numpy()[changed_right_idx, :]

        changed_y_left = left_new[changed_left_idx]
        changed_y_right = right_new[changed_right_idx]

        # Let us one-hot encode again
        changed_y_left = safe_transform(self.enc, changed_y_left)
        changed_y_right = safe_transform(self.enc, changed_y_right)

        #  Now it is time to stack the two resulting sets
        changed_X = tf.concat([changed_X_left, changed_X_right, self.Xdirect], axis=0)
        changed_Y = tf.concat([changed_y_left, changed_y_right, self.ydirect], axis=0)
        changed_X = tf.cast(changed_X, tf.float32, name="tsX")
        changed_Y = tf.cast(changed_Y, tf.float32, name="tsY")

        return changed_X, changed_Y


def direct_join_partition(
    left_array: np.ndarray,
    right_array: np.ndarray,
    left_label: int,
    right_label: int,
    rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Find the minimum size
    min_size = min(left_array.shape[0], right_array.shape[0])
    left_array = left_array[:min_size, :]
    right_array = right_array[:min_size, :]

    # Create a sample and the relevant mask
    count = int(rate * min_size)
    sample = np.random.choice(min_size, size=count, replace=False)
    mask = np.zeros(min_size, dtype=bool)
    mask[sample] = True

    # Create the direct complaints
    Xdirect = np.vstack([left_array[mask], right_array[mask]])
    ydirect = np.concatenate([np.full(count, left_label), np.full(count, right_label)])

    # The join complaints
    left_array = left_array[~mask]
    right_array = right_array[~mask]

    return (Xdirect, ydirect, left_array, right_array)


@tf.function
def complain_impl(
    Xleft: tf.Tensor,
    Xright: tf.Tensor,
    Xdirect: tf.Tensor,
    ydirect: tf.Tensor,
    manager: ModelManager,
    exact: bool = False,
) -> ComplaintRet:

    if exact:
        left_ = tf.one_hot(manager.predict(Xleft), 10)
        right_ = tf.one_hot(manager.predict(Xright), 10)
        direct_ = tf.one_hot(manager.predict(Xdirect), 10)
    else:
        # The predictions on the left and right table
        left_ = manager.predict_proba(Xleft)
        right_ = manager.predict_proba(Xright)
        direct_ = manager.predict_proba(Xdirect)

    # Now the relaxed/exact join count becomes
    Q_join = tf.reduce_sum(left_ * right_)
    C_join = tf.constant(0, dtype=tf.float32)

    # Now for the exact labels complaint
    # Minus sign because it corresponds to decrease

    Q_exact = tf.reduce_sum(direct_ * ydirect)
    C_exact = tf.constant(ydirect.shape[0], dtype=tf.float32)

    return ComplaintRet(AC=tf.stack([C_join, C_exact]), AQ=tf.stack([Q_join, Q_exact]))