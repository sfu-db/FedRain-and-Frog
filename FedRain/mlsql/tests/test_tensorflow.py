import tensorflow as tf
import numpy as np
from ..utils import Hvp, iHvp, setdiff1d


def test_hvp():
    def compute(x):
        val = (x[0] ** 3) * (x[1] ** -1)
        return tf.reduce_sum(tf.math.log_sigmoid(val))

    x = [tf.random.normal((4, 1)), tf.random.normal((1, 1))]
    v = [tf.random.normal((4, 1)), tf.random.normal((1, 1))]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = compute(x)
        G = tf.concat(tape.gradient(y, x), axis=0)
    H = tf.squeeze(tf.concat(tape.jacobian(G, x), axis=-2))

    Hv0 = H @ tf.concat(v, axis=0)

    Hv1 = tf.concat(Hvp(lambda: compute(x), x, v), axis=0)

    assert np.allclose(Hv0, Hv1)


def test_ihvp():
    def compute(x):
        val = (x[0] ** 3) * (x[1] ** -1)
        return tf.reduce_sum(tf.math.log_sigmoid(val))

    x = [tf.random.normal((4, 1)), tf.random.normal((1, 1))]
    v = [tf.random.normal((4, 1)), tf.random.normal((1, 1))]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = compute(x)
        G = tf.concat(tape.gradient(y, x), axis=0)
    H = tf.squeeze(tf.concat(tape.jacobian(G, x), axis=-2))

    iHv0 = tf.linalg.inv(H) @ tf.concat(v, axis=0)

    iHv1 = tf.concat(iHvp(lambda: compute(x), x, v), axis=0)

    assert np.allclose(iHv0, iHv1)


def test_setdiff1d():
    b = tf.TensorArray(tf.int32, 10)
    b.write(0, 3)
    b.write(1, 5)

    a = tf.constant([5, 3, 4, 1, 2])

    assert setdiff1d(a, b.stack()[:2]).numpy() == np.asarray([4, 1, 2])
