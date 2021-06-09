import sys
from collections import defaultdict
from time import time
from typing import Callable, DefaultDict, List, TypeVar

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

from .conjugate_gradient import conjugate_gradient
from .ihvp import Hvp, iHvp, iHvpB
from .pack import pack

# from .scipy_optimizer import train_scipy
# from .tfp_optimizer import train_tfp

RT = TypeVar("RT")


@tf.function
def setdiff1d(a: tf.Tensor, b: tf.Tensor):
    """
    Computes set diff as a - b, preserve the elem order for a
    Parameters
    __________
    a : tf.Tensor
    b : tf.Tensor
    Returns
    -------
    tf.Tensor
    """
    assert a.shape.rank == 1, "Tensor a should have rank 1"
    assert b.shape.rank == 1, "Tensor b should have rank 1"

    unequals = a != tf.reshape(b, (-1, 1))

    mask = tf.reshape(tf.math.reduce_all(unequals, axis=0), [-1])

    return tf.boolean_mask(a, mask)


def no_tf_variable(func: Callable[..., RT]) -> Callable[..., RT]:
    """
    Translate all tf.Variable in the function input 
    into tf.Tensor since @tf.function has bug for tf.Variable
    """

    def imp(*args, **kwargs):
        args = tuple(
            tf.identity(arg) if isinstance(arg, tf.Variable) else arg for arg in args
        )
        kwargs = {
            k: (tf.identity(arg) if isinstance(arg, tf.Variable) else arg)
            for k, arg in kwargs
        }
        return func(*args, **kwargs)

    return imp


def safe_transform(enc: OneHotEncoder, array: np.ndarray) -> np.ndarray:
    if len(array.shape) == 1:
        array = array.reshape(-1, 1)
    assert len(array.shape) == 2

    if len(array) != 0:
        encoded = enc.transform(array)
    else:
        print("The empty label array passed to one hot encoder", file=sys.stderr)
        nout = len(enc.categories[0])
        encoded = np.empty(shape=(0, nout))

    return encoded


T = TypeVar("T")


class Timer:
    timer: DefaultDict[str, List[float]]

    def __init__(self) -> None:
        self.timer = defaultdict(list)

    def measure(self, name: str, func: Callable[[], T]) -> T:
        then = time()
        ret = func()
        self.timer[name].append(time() - then)
        return ret


def recall_k(truth: np.ndarray, rank: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(truth))
    flags[rank] = 1 + np.arange(len(rank))[::-1]

    _, recalls, _ = precision_recall_curve(truth + 0, flags)
    return recalls[-2:0:-1]


def add_constant(X: tf.Tensor) -> tf.Tensor:
    assert len(X.shape) == 2
    return tf.pad(X, [[0, 0], [0, 1]], "CONSTANT")
