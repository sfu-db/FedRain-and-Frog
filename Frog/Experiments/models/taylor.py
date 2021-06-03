import tensorflow as tf
import math


class TaylorAprx(tf.keras.Model):
  def __init__(self, nclass):
    super().__init__()
    auto_seed = int(tf.random.uniform([], maxval=2**10, seed=1))
    self.dense = tf.keras.layers.Dense(
      nclass,
      dtype=tf.float32,
      kernel_initializer=tf.keras.initializers.glorot_uniform(auto_seed),
      bias_initializer=tf.keras.initializers.glorot_uniform(auto_seed),
    )
  
  def value(self, inputs):
    return self.dense(inputs)
  
  def call(self, inputs):
    value = self.value(inputs)
    return 1 / 2 + value / 4 - value ** 3 / 48 + value ** 5 / 480
  
  def eloss(self, inputs, labels):
    value = self.value(inputs)
    labels = tf.expand_dims(labels, axis=1)
    l = -labels * value / 2 + value ** 2 / 8 - value ** 4 / 192
    return math.log(2) + l
  
  def loss(self, inputs, labels):
    value = self.value(inputs)
    labels = tf.expand_dims(labels, axis=1)
    l = -labels * value / 2 + value ** 2 / 8 - value ** 4 / 192
    return math.log(2) + tf.reduce_mean(l)
  
  def egrad(self, inputs, labels):
    return
  
  def grad(self, inputs, labels):
    labels = tf.expand_dims(labels, axis=1)
    value = self.value(inputs)
    factor = value / 4 - labels / 2
    w_grad = tf.reduce_mean(factor * inputs, axis=0)
    b_grad = tf.reduce_mean(factor, axis=0)
    w_grad = tf.expand_dims(w_grad, axis=1)
    return [w_grad, b_grad]
  
  def hessian(self, inputs, labels):
    return
