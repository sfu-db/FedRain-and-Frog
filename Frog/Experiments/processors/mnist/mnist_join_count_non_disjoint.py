from typing import List, Tuple, Union

# Library that loads mnist
import mnist
import numpy as np
import tensorflow as tf

# To fix the join-count complaints we can use tiresias
from mlsql import ModelManager, minimal_fix_join_count_multi
from mlsql.processor import ComplaintRet, Processor
from mlsql.utils import safe_transform

# Library for converting mnist images to floats
from skimage import img_as_float

# MNIST has a prespecified train/test split
# But is a "relatively" big dataset (at least for now)
# We use train_test_split for subsampling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .utils import load_dataset, single_digit_corruption


class MNISTNonDisjointJoinAggregationProcessor(Processor):
    """
        The query is `SELECT COUNT(*) FROM L JOIN R ON M(L) == M(R)`
        The expected output is zero.
    """

    Xtrain: tf.Tensor
    Xtest: tf.Tensor
    corrsel: tf.Tensor
    ytrain: tf.Tensor
    ycorr: tf.Tensor
    ytest: tf.Tensor
    yleft: tf.Tensor
    yright: tf.Tensor

    enc: OneHotEncoder

    seed: int
    corrupted_class: int
    corrupted_to_class: int
    corruption_rate: float
    partition: Tuple[List[int], List[int]]
    size: int

    # The argumenets are : the seed controlling the label corruption
    # The corrupted class that gets its labels flipped
    # The target class of the flips
    # The corruption rate
    # The partition of the test data in left and right
    # There are some rules on the partitions captured by the assertions
    def __init__(
        self,
        seed: int,
        corrupted_class: int,
        corrupted_to_class: int,
        corruption_rate: float,
        partition: Tuple[List[int], List[int]],
        mix_rate: float = 0.5,
        size: int = 3000,
    ):

        # First things first. Let us do some sanity checks
        assert corrupted_class in range(10), f"corrupted_class must in 0..10, got {corrupted_class}"
        assert corrupted_to_class in range(
            10
        ), f"corrupted_to_class must in 0..10, got {corrupted_to_class}"
        assert (
            corrupted_class != corrupted_to_class
        ), f"corrupted_class should not be same as corrupted_to_class got {corrupted_class} and {corrupted_to_class}"
        assert 0 < corruption_rate <= 1

        # Let us also do some sanity checks on the partitions
        assert len(partition[0]) > 0
        assert len(partition[1]) > 0
        assert set(partition[0]).issubset(range(10))
        assert set(partition[1]).issubset(range(10))
        assert set(partition[1]).isdisjoint(partition[0])

        # Check that the corruption makes sense
        # We could allow the other way as well
        # But let us keep it simple
        assert corrupted_class in partition[0]
        assert corrupted_to_class in partition[1]

        # To make sure there is no form of bias on any side
        # Let us keep the number of classes in each side equal
        assert len(set(partition[0])) == len(set(partition[1]))

        # Save class parameters
        self.left_partition = partition[0]
        self.right_partition = partition[1]

        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__(
            seed=seed,
            corrupted_class=corrupted_class,
            corrupted_to_class=corrupted_to_class,
            corruption_rate=corruption_rate,
            partition=partition,
            mix_rate=mix_rate,
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

        # migrate some corrupted_class to the right table
        corrupted_class_idx, = np.where(ytest == corrupted_class)
        migration = np.random.choice(corrupted_class_idx, int(mix_rate * len(corrupted_class_idx)))
        migration_mask = np.full((ytest.shape[0],), False)
        migration_mask[migration] = True

        # What about testing?
        # Split the test set (L vs R)

        test_in_left = np.isin(ytest, partition[0])
        yleft = ytest[
            (test_in_left & (ytest != corrupted_class)) | (test_in_left & ~migration_mask)
        ]

        test_in_right = np.isin(ytest, partition[1])
        yright = ytest[test_in_right | migration_mask]

        # We are not ready yet
        # We need to convert all labels to one-hot representations

        # Here is our one-hot encoder
        self.enc = OneHotEncoder(categories=[np.arange(10)], sparse=False)
        # To initialize it we have
        self.enc.fit(ytrain.reshape(-1, 1))

        # Now we have the training data part ready
        self.set_tensor_variables(
            Xtrain=Xtrain,
            Xleft=Xtest[
                (test_in_left & (ytest != corrupted_class)) | (test_in_left & ~migration_mask), :
            ],
            Xright=Xtest[test_in_right | migration_mask, :],
            ytrain=self.enc.transform(ytrain.reshape(-1, 1)),
            yleft=self.enc.transform(yleft.reshape(-1, 1)),
            yright=self.enc.transform(yright.reshape(-1, 1)),
            ycorr=self.enc.transform(ycorr.reshape(-1, 1)),
        )
        self.set_tensor_variables(tf.bool, corrsel=corrsel)

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        return complain_impl(self.Xleft, self.yleft, self.Xright, self.yright, manager, exact)

    def tiresias(self, manager: ModelManager) -> Tuple[tf.Tensor, tf.Tensor]:
        left = manager.predict(self.Xleft).numpy()
        right = manager.predict(self.Xright).numpy()

        # The true join count based on test labels
        left_class_true_sums = tf.reduce_sum(self.yleft, axis=0)
        right_class_true_sums = tf.reduce_sum(self.yright, axis=0)
        C = tf.reduce_sum(left_class_true_sums * right_class_true_sums).numpy()

        # Let us now apply the ILP fix.
        left_new, right_new = minimal_fix_join_count_multi(left, right, C, 10)

        # Find the changes
        changed_left_idx, = np.where(left_new != left)
        changed_right_idx, = np.where(right_new != right)

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
    Xleft: tf.Tensor,
    yleft: tf.Tensor,
    Xright: tf.Tensor,
    yright: tf.Tensor,
    manager: ModelManager,
    exact: bool = False,
) -> ComplaintRet:

    if exact:
        left_ = tf.one_hot(manager.predict(Xleft), 10, dtype=tf.float32)
        right_ = tf.one_hot(manager.predict(Xright), 10, dtype=tf.float32)
    else:
        # The predictions on the left and right table
        left_ = manager.predict_proba(Xleft)
        right_ = manager.predict_proba(Xright)

    # Now we want to group the probabilities/predictions by predicted class
    left_class_sums = tf.reduce_sum(left_, axis=0)
    right_class_sums = tf.reduce_sum(right_, axis=0)

    # Now the relaxed/exact join count becomes
    Q = tf.reduce_sum(left_class_sums * right_class_sums)

    left_class_true_sums = tf.reduce_sum(yleft, axis=0)
    right_class_true_sums = tf.reduce_sum(yright, axis=0)
    C = tf.reduce_sum(left_class_true_sums * right_class_true_sums)

    return ComplaintRet(AC=tf.reshape(C, [1]), AQ=tf.reshape(Q, [1]))
