import tensorflow as tf

# from ..manager import ModelManager


class LossRanker():
    def __init__(
        self, manager, batch_size=None):
        self.manager = manager
        if batch_size is not None:
            self.batch_size = tf.constant(batch_size, dtype=tf.int32)
        else:
            self.batch_size = batch_size

    def get_rank(self):
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
        
    def predict(self):
        return self.get_rank()

    def name(self):
        return "Loss"