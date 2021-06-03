import tensorflow as tf


class LogReg(tf.keras.Model):
  def __init__(self, nclass):
    super().__init__()
    auto_seed = int(tf.random.uniform([], maxval=2**10, seed=1))
    self.dense = tf.keras.layers.Dense(
      nclass,
      dtype=tf.float32,
      kernel_initializer=tf.keras.initializers.glorot_uniform(auto_seed),
      bias_initializer=tf.keras.initializers.glorot_uniform(auto_seed),
    )
  
  def call(self, inputs):
    return tf.keras.activations.sigmoid(self.dense(inputs))

  def eloss(self, inputs, labels):
    logits = self.call(inputs)
    return -(
      tf.math.multiply(labels, tf.squeeze(tf.math.log(logits)))
      + tf.math.multiply(1 - labels, tf.squeeze(tf.math.log(1 - logits)))
    )

  def loss(self, inputs, labels):
    logits = self.call(inputs)
    return -tf.reduce_mean(
      (tf.math.multiply(labels, tf.squeeze(tf.math.log(logits)))
       + tf.math.multiply(1 - labels, tf.squeeze(tf.math.log(1 - logits))))
    )

  def predict(self, inputs):
    results = tf.squeeze(self.call(inputs) > 0.5)
    return tf.cast(results, tf.int32)

  def model_name(self):
    return "LogReg"
