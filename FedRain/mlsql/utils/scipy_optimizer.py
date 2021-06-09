import itertools
from typing import Callable, List, Tuple

import numpy as np
import scipy

import tensorflow as tf

from .pack import pack


# Get shape of tensor as a tuple
def get_shape_tuple(tensor: tf.Tensor) -> Tuple[int]:
    return tuple(tensor.get_shape().as_list())


# Sometimes a gradient may be missing because
# a variable did not participate in the computation (e.g. unused layer)
# we replace its gradient with zeros
def zero_default(grads: List[tf.Tensor], var_list: tf.Variable) -> List[tf.Tensor]:
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


# This function actually implements the training
def train_scipy(loss: Callable[[], tf.Tensor], variables: List[tf.Variable]):

    # Get dimension of each variable in weights
    dims = [tf.size(var) for var in variables]

    # Get list of partial sums of such dims
    accumulated_dims = [0] + list(itertools.accumulate(dims))

    # Find the slice of each variable in the packed vector
    packing_slices = [
        slice(start, end) for start, end in zip(accumulated_dims[:-1], accumulated_dims[1:])
    ]

    def unpack_assign(x: np.ndarray):
        # Assign the values of x to the relevant variables of the model
        for var, packing_slice in zip(variables, packing_slices):
            var.assign(x[packing_slice].reshape(get_shape_tuple(var)))

    # Function that takes as input the packed variables x
    # and returns the loss and the gradient evaluated at x
    def loss_and_gradient(x: np.ndarray) -> Tuple[np.float32, np.ndarray]:

        unpack_assign(x)

        # Run the actual losses
        with tf.GradientTape() as tape:
            loss_value = loss()

        # Calculate the actual gradients
        grads = tape.gradient(loss_value, variables)

        # Account for potentially missing variables
        grads = zero_default(grads, variables)

        # Pack the grads to return the value
        return loss_value, pack(grads).numpy().astype(np.float32)

    # The initial value of the optimization
    initial_packed_var_val = pack(variables).numpy()

    minimize_kwargs = {"jac": True, "method": "L-BFGS-B"}

    result = scipy.optimize.minimize(loss_and_gradient, initial_packed_var_val, **minimize_kwargs)

    # Save the result of training back in the model
    unpack_assign(result["x"])
