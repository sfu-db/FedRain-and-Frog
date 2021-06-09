import tensorflow as tf
from ..processor import Processor


class LogReg(tf.keras.Model):
    def __init__(self, proc: Processor, nclass: int, C: float = 1.0):
        super().__init__()
        # regularizer = tf.keras.regularizers.l2(1 / C)
        self.dense = tf.keras.layers.Dense(
            nclass,
            # kernel_regularizer=regularizer,
            dtype=tf.float32,
            kernel_initializer=tf.keras.initializers.glorot_uniform(proc.auto_seed),
            bias_initializer=tf.keras.initializers.glorot_uniform(proc.auto_seed),
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.dense(inputs)
