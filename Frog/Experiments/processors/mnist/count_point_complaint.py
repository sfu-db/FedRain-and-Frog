# Standard imports
# Import for typing stuff
from typing import Optional, Tuple
from numbers import Number

import numpy as np

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

# To fix the single-class-count complaints we can use tiresias
from mlsql import ModelManager, multi_ambiguity_count
from mlsql.processor import ComplaintRet, Processor

from .utils import load_dataset, single_digit_corruption


class MNISTCountPointComplaintProcessor(Processor):

    Xtrain: tf.Tensor
    Xtest: tf.Tensor
    corrsel: tf.Tensor
    ytrain: tf.Tensor
    ycorr: tf.Tensor
    ytest: tf.Tensor
    complaint_mask: tf.Tensor

    enc: OneHotEncoder

    seed: int
    corrupted_class: int
    corrupted_to_class: int
    corruption_rate: float
    complaint_percent: float
    complaint_mode: str
    size: int

    """
    This processor relates to the query
    Query `SELECT COUNT(*) FROM D WHERE predict(D) = :digit`,
    Complaint `* == true_label`
    Corruption: Corrupt :digit to :another_digit
    """

    def __init__(
        self,
        seed: int,
        corrupted_class: int,
        corrupted_to_class: int,
        corruption_rate: float,
        complaint_percent: float,
        complaint_mode: str = "both",
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
        complaint_percent : float
            complaint percent of naive complaint
        complaint_mode : str
            "from", "to", "both", "random"
        """

        # First things first. Let us do some sanity checks
        assert corrupted_class in range(10)
        assert corrupted_to_class in range(10)
        assert corrupted_class != corrupted_to_class
        assert 0 < corruption_rate <= 1
        assert isinstance(complaint_percent, Number)
        assert complaint_mode in ["from", "to", "others", "all", "both"]

        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__(
            seed=seed,
            corrupted_class=corrupted_class,
            corrupted_to_class=corrupted_to_class,
            corruption_rate=corruption_rate,
            complaint_percent=complaint_percent,
            complaint_mode=complaint_mode,
            size=size,
        )

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

        # Here is our one-hot encoder
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
        """
        We pre select a bunch of test points as if the user will pick them up and manually label them
        """
        ypred = manager.predict(self.Xtest)

        if self.complaint_mode == "from":
            from_mask = ypred == self.corrupted_class
            mask = from_mask
        elif self.complaint_mode == "to":
            to_mask = ypred == self.corrupted_to_class
            mask = to_mask
        elif self.complaint_mode == "others":
            others_mask = ypred != self.corrupted_class
            mask = others_mask
        elif self.complaint_mode == "all":
            mask = tf.ones((len(ypred),), dtype=tf.bool)
        elif self.complaint_mode == "both":
            from_mask = ypred == self.corrupted_class
            to_mask = ypred == self.corrupted_to_class
            mask = from_mask | to_mask
        else:
            raise RuntimeError("Unreachable")

        complaint_indices = tf.reshape(tf.where(mask), [-1])
        n = tf.cast(tf.size(complaint_indices), tf.float32)
        nreport = tf.cast(tf.math.round(n * self.complaint_percent), dtype=tf.int32)

        sel = np.zeros(tf.shape(self.ytest)[0].numpy(), dtype=np.bool)

        np.random.seed(self.auto_seed)
        sel[np.random.choice(complaint_indices.numpy(), nreport.numpy())] = True

        self.complaint_mask = tf.constant(sel, name="complaint_mask")

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        return complain_impl(
            self.Xtest, self.ytest, self.complaint_mask, manager, exact
        )

    # Returns the correction of the count based on how-to-provenance
    def tiresias(self, manager: ModelManager) -> Tuple[tf.Tensor, tf.Tensor]:
        raise RuntimeError("No way to run tiresias")
        # # The predictions as a numpy array
        # tsX = self.Xtest[self.complaint_mask]
        # tsy = self.ytest[self.complaint_mask]

        # return tsX, tsy

    def ambiguity(self, manager: ModelManager) -> Optional[float]:
        C = tf.reduce_sum(self.ytest[:, self.corrupted_class])
        # The model predictions
        test_probas = manager.predict_proba(self.Xtest)
        predictions = tf.argmax(test_probas, axis=1)
        Q = tf.reduce_sum(tf.cast(predictions == self.corrupted_class, tf.float32))

        return multi_ambiguity_count(
            int(C.numpy()), int(Q.numpy()), len(self.Xtest), 10
        )


@tf.function
def complain_impl(
    Xtest: tf.Tensor,
    ytest: tf.Tensor,
    complaint_mask: tf.Tensor,
    manager: ModelManager,
    exact: bool = False,
) -> ComplaintRet:

    # ypred = manager.predict(Xtest[complaint_mask])
    # mispredict_mask = ypred != tf.argmax(ytest[complaint_mask], axis=1)

    # within the complaint selections, narrow down to these mis predictions
    Cs = ytest[complaint_mask]

    if exact:
        ypred = manager.predict(Xtest[complaint_mask])
        Qs = tf.one_hot(ypred, depth=10)
    else:
        # Return logits if we directly complain on labels
        Qs = manager.predict_logit(Xtest[complaint_mask])

    return ComplaintRet(PC=Cs, PQ=Qs)
