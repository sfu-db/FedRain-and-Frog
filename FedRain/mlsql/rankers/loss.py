import tensorflow as tf
from typing import Optional

from .ranker import Ranker
from ..manager import ModelManager


class LossRanker(Ranker):
    manager: ModelManager
    batch_size: tf.Tensor

    def __init__(
        self, manager: ModelManager, batch_size: Optional[tf.Tensor] = None
    ) -> None:
        self.manager = manager
        if batch_size is not None:
            self.batch_size = tf.constant(batch_size, dtype=tf.int32)
        else:
            self.batch_size = batch_size

    def get_rank(self) -> tf.Tensor:
        manager = self.manager
        batch_size = self.batch_size
        if batch_size is not None:
            influences = tf.TensorArray(size=manager.ntrain, dtype=tf.float32)

            for i in tf.range(0, manager.ntrain, batch_size):
                upper = tf.math.minimum(manager.ntrain, i + batch_size)
                eloss = manager.eloss(range=(i, upper))
                influences = influences.scatter(tf.range(i, upper), eloss)
            return influences.stack(name="influence")
        else:
            return manager.eloss()

    def name(self) -> str:
        return "Loss"
