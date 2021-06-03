# Standard imports
# Library that loads mnist
from typing import Optional, Tuple, Union

import mnist
import numpy as np
import tensorflow as tf

# To fix the join-count complaints we can use tiresias
from mlsql import (
    ModelManager,
    Processor,
    minimal_join_rows_fix_multi,
    multi_ambiguity_join_rows,
)
from mlsql.processor import ComplaintRet
from mlsql.utils import safe_transform

# Library for converting mnist images to floats
from skimage import img_as_float

# MNIST has a prespecified train/test split
# But is a "relatively" big dataset (at least for now)
# We use train_test_split for subsampling
from sklearn.model_selection import train_test_split

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder

from .utils import load_dataset, single_digit_corruption


class MNISTJoinRowsProcessor(Processor):

    # Parameters
    # seed : randomization seed for corruption
    # corrupted_class: class of MINST that is corrupted a.k.a. its training count is reduced
    #                  It is also used as the target class of the count complaint
    # corrupted_to_class : corruptions are flipped to this MNIST label
    # corruption_rate : percentage of corrupted_class that will be flipped

    Xtrain: tf.Tensor
    Xtest: tf.Tensor
    corrsel: tf.Tensor
    ytrain: tf.Tensor
    ycorr: tf.Tensor
    ytest: tf.Tensor
    Xleft: tf.Tensor
    Xright: tf.Tensor
    ycommon: tf.Tensor

    enc: OneHotEncoder

    seed: int
    corrupted_class: int
    corrupted_to_class: int
    corruption_rate: float
    size: int

    def __init__(
        self,
        seed: int,
        corrupted_class: int,
        corrupted_to_class: int,
        corruption_rate: float,
        size: int = 3000,
    ):

        # First things first. Let us do some sanity checks
        assert corrupted_class in range(10)
        assert corrupted_to_class in range(10)
        assert corrupted_class != corrupted_to_class
        assert corruption_rate > 0
        assert corruption_rate <= 1.0

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
            seed=1, train_size=size, test_size=10000
        )

        ycorr, corrsel = single_digit_corruption(
            ytrain, corrupted_class, corrupted_to_class, corruption_rate, self.auto_seed
        )

        # Here is our one-hot encoder
        self.enc = OneHotEncoder(categories=[np.arange(10)], sparse=False)
        # To initialize it we have
        self.enc.fit(ytrain.reshape(-1, 1))

        # Get the one-hot encoded labels
        self.set_tensor_variables(
            Xtrain=Xtrain,
            Xtest=Xtest,
            ytrain=self.enc.transform(ytrain.reshape(-1, 1)),
            ycorr=self.enc.transform(ycorr.reshape(-1, 1)),
            # The test labels not not be one-hot encoded
            # See generate_partitions for more details
            ytest=ytest,
        )

        self.set_tensor_variables(tf.bool, corrsel=corrsel)

    def post_init(self, manager: ModelManager):
        Xtest = self.Xtest.numpy()
        ytest = self.ytest.numpy()

        # Split the test set (L vs R)
        Xleft = Xtest[ytest == self.corrupted_class, :]
        Xright = Xtest[ytest == self.corrupted_to_class, :]

        # Find the classifications of Xleft and Xright
        yleft_pred = manager.predict(Xleft).numpy()
        yright_pred = manager.predict(Xright).numpy()

        # Match wrong classifications of X_left with correct classifications of X_right
        Xleft_wrong = Xleft[yleft_pred == self.corrupted_to_class]
        Xright_correct = Xright[yright_pred == self.corrupted_to_class]
        size_one = min(Xleft_wrong.shape[0], Xright_correct.shape[0])

        # Match wrong classifications of Xright with correct classifications of Xleft
        Xleft_correct = Xleft[yleft_pred == self.corrupted_class]
        Xright_wrong = Xright[yright_pred == self.corrupted_class]
        size_two = min(Xright_wrong.shape[0], Xleft_correct.shape[0])

        # Stack the two cases together to create the new X_left, X_right
        self.Xleft = tf.concat(
            [Xleft_wrong[:size_one, :], Xleft_correct[:size_two, :]],
            axis=0,
            name="Xleft",
        )
        self.Xright = tf.concat(
            [Xright_correct[:size_one, :], Xright_wrong[:size_two, :]],
            axis=0,
            name="Xright",
        )

        # Generate the predicted labels for Tiresias
        part_one = np.full(size_one, self.corrupted_to_class, dtype=int)
        part_two = np.full(size_two, self.corrupted_class, dtype=int)

        # The final labels are here (No need for one-hot encoding)
        self.ycommon = tf.concat([part_one, part_two], axis=0, name="ycommon")

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        return complain_impl(self.Xleft, self.Xright, manager, exact)

    # def ambiguity(self, manager: ModelManager) -> float:
    #     multi_ambiguity_join_rows()  # TODO

    def tiresias(self, manager: ModelManager) -> Tuple[tf.Tensor, tf.Tensor]:
        ycommon = self.ycommon.numpy()

        # Let us now apply the ILP fix.
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
        changed_X = np.vstack([changed_X_left, changed_X_right])
        changed_Y = np.vstack([changed_y_left, changed_y_right])
        changed_X = tf.constant(changed_X, dtype=tf.float32, name="tsX")
        changed_Y = tf.constant(changed_Y, dtype=tf.float32, name="tsY")

        return changed_X, changed_Y


@tf.function
def complain_impl(
    Xleft: tf.Tensor, Xright: tf.Tensor, manager: ModelManager, exact: bool = False
) -> ComplaintRet:
    if exact:
        left_proba = tf.one_hot(manager.predict(Xleft), 10)
        right_proba = tf.one_hot(manager.predict(Xright), 10)
    else:
        # The predictions on the left and right table
        left_proba = manager.predict_proba(Xleft)
        right_proba = manager.predict_proba(Xright)

    # Now the relaxed/exact join count becomes
    Q = tf.reduce_sum(left_proba * right_proba)

    C = tf.constant(0, dtype=tf.float32)

    return ComplaintRet(AC=tf.reshape(C, [1]), AQ=tf.reshape(Q, [1]))
