from abc import ABC, abstractmethod

import tensorflow as tf


class Ranker(ABC):
    @abstractmethod
    def get_rank(self) -> tf.Tensor:
        """
            Return a Tensor that has same length as the training set, indicating whether elements
            should be deleted or not.
        """

    @abstractmethod
    def name(self) -> str:
        pass

    def predict(self) -> tf.Tensor:
        return tf.reshape(self.get_rank(), (-1,))
