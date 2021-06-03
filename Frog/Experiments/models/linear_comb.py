import tensorflow as tf


class LinearComb(tf.keras.Model):
    def __init__(self, nclass):
        super().__init__()
        auto_seed = int(tf.random.uniform([], maxval=2**10, seed=1))
        self.dense = tf.keras.layers.Dense(
          nclass,
          dtype=tf.float32,
          bias_initializer=tf.keras.initializers.glorot_uniform(auto_seed),
          kernel_initializer=tf.keras.initializers.glorot_uniform(auto_seed),
        )
        self.c = tf.Variable((tf.random.uniform([1], seed=1) + 0.5) / 2, name='c', trainable=False)
#         self.c = tf.Variable((tf.random.uniform([1], seed=1) + 0.5) / 2, name='c', trainable=True) # initial [0.25, 0.75]

    def call(self, inputs):
        return self.c * tf.keras.activations.sigmoid(self.dense(inputs))
  
    def f(self, inputs):
        return tf.keras.activations.sigmoid(self.dense(inputs))
  
    def eloss(self, inputs, labels):
        return 0.5 * (tf.squeeze(self.call(inputs)) - labels) ** 2
  
    def loss(self, inputs, labels):
        return 0.5 * tf.reduce_mean((tf.squeeze(self.call(inputs)) - labels) ** 2)

    def hessian(self, inputs, labels=None, other_party=0):
        f = tf.squeeze(self.f(inputs))
        factor1 = tf.reshape((self.c * f * (1 - f)) ** 2, [-1, 1, 1])
        value_diff = self.c * f + other_party - labels if labels is not None else f + other_party
        factor2 = tf.reshape(value_diff * self.c * f * (1 - f) * (1 - 2 * f), [-1, 1, 1])
        ex_inputs = tf.concat((tf.ones((len(inputs), 1)), inputs), axis=-1)
        hess_inputs = tf.expand_dims(ex_inputs, axis=2) * tf.expand_dims(ex_inputs, axis=1)
        hess = tf.reduce_mean(hess_inputs * (factor1 + factor2), axis=0)
        return hess
    
#         c_factor = tf.expand_dims(f * (1 - f), axis=-1)
#         c_hess = tf.expand_dims(tf.reduce_mean(c_factor * ex_inputs, axis=0), axis=-1)
#         ex_hess = tf.concat([tf.concat([hess, c_hess], 1), tf.concat([tf.transpose(c_hess), [[0]]], 1)], 0)
#         return ex_hess
    
    def egrads(self, inputs, labels, hess=False):
        f = tf.squeeze(self.f(inputs))
        factor = tf.expand_dims(self.c * f * (1 - f),
                                axis=-1)
        return self.pack([factor, factor * inputs])
#         if hess:
#             return self.pack([factor, factor * inputs, tf.expand_dims([0.0] * len(f), axis=-1)])
#         else:    
#             return self.pack([factor, factor * inputs, tf.expand_dims(f, axis=-1)])

    def pack(self, list_to_pack):
        return tf.concat(list_to_pack, axis=-1)

    def split(self, packed_tensor):
        return [tf.expand_dims(packed_tensor[..., 1:], axis=-1),
                tf.convert_to_tensor(packed_tensor[..., :1])]
#         return [tf.expand_dims(packed_tensor[..., 1:-1], axis=-1),
#                 tf.convert_to_tensor(packed_tensor[..., :1]),
#                 tf.convert_to_tensor(packed_tensor[..., -1:])]

    def model_name(self):
        return "LinearComb"