from typing import Callable, List, Union, Tuple

import tensorflow_probability as tfp

import tensorflow as tf
import numpy as np


def train_tfp(
    loss: Callable[[], tf.Tensor], variables: List[tf.Variable], max_iter: int = 20
) -> None:
    def loss_grad_func(W) -> Tuple[tf.Tensor, tf.Tensor]:
        Ws = reshape_like(W, variables)
        for W, mW in zip(Ws, variables):
            mW.assign(W)
        with tf.GradientTape() as tape:
            regloss = loss()
        grad = flatten(tape.gradient(regloss, variables))
        return (regloss, grad)

    result = tfp.optimizer.lbfgs_minimize(
        loss_grad_func, flatten(variables), max_iterations=max_iter
    )

    W = result.position
    Ws = reshape_like(W, variables)
    for W, var in zip(Ws, variables):
        var.assign(W)


TensorOrShape = Union[tf.Tensor, tf.TensorShape]


def reshape_like(x: tf.Tensor, shapes_or_tensors: List[TensorOrShape]) -> List[tf.Tensor]:
    """
    Reshape a flattened tensor into a list of tensors defined by shapes_or_tensors
    Parameters
    ----------
    x : tf.Tensor
        The flattened tensor
    shapes_or_tensors : List[tf.Tensor]
        The second parameter.
    Returns
    -------
    List[tf.Tensor]
        HVP, with shape same as xs
    """
    assert x.shape.rank == 1

    shapes = []
    for shape_or_tensor in shapes_or_tensors:
        if isinstance(shape_or_tensor, tf.Tensor) or isinstance(shape_or_tensor, tf.Variable):
            shapes.append(shape_or_tensor.shape.as_list())
        else:
            shapes.append(shape_or_tensor)

    sizes = [np.prod(shape) for shape in shapes]
    assert sum(sizes) == np.prod(x.shape.as_list())

    indices = np.cumsum([0] + sizes)
    return [
        tf.reshape(tf.slice(x, (start,), (length,)), shape)
        for start, length, shape in zip(indices[:-1], sizes, shapes)
    ]


def flatten(inputs: List[tf.Tensor]) -> tf.Tensor:
    return tf.concat([tf.reshape(i, (-1,)) for i in inputs], axis=0)
