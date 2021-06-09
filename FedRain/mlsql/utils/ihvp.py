import itertools
from typing import Callable, List, Tuple

import tensorflow as tf

from .conjugate_gradient import conjugate_gradient
from .pack import pack


# Sometimes a gradient may be missing because
# a variable did not participate in the computation (e.g. unused layer)
# we replace its gradient with zeros
def zero_default(grads: List[tf.Tensor], var_list: tf.Tensor) -> List[tf.Tensor]:
    return [
        grad if grad is not None else tf.zeros_like(var)
        for var, grad in zip(var_list, grads)
    ]


def Hvp(
    loss: Callable[[], tf.Tensor],
    variables: List[tf.Variable],
    vector: tf.Tensor,
    mode: str = "fb",
) -> tf.Tensor:
    """
        mode: fb = forward backward
              bb = backward backward
    """
    # for v in variables:
    #     assert isinstance(v, tf.Tensor), "All the variables should be tf.Tensor"

    # Get dimension of each variable in weights
    dims = [tf.size(var) for var in variables]

    # Get list of partial sums of such dims
    accumulated_dims = [0] + list(itertools.accumulate(dims))

    # Find the slice of each variable in the packed vector
    packing_slices = [
        slice(start, end)
        for start, end in zip(accumulated_dims[:-1], accumulated_dims[1:])
    ]

    vector_slices = []
    for var, packing_slice in zip(variables, packing_slices):
        vector_slice = tf.reshape(vector[packing_slice], var.shape)
        vector_slices.append(vector_slice)

    if mode == "bb":
        # Run the actual losses
        with tf.GradientTape(persistent=True) as tape:
            loss_value = loss()
            # Calculate the actual gradients
            grads = tape.gradient(loss_value, variables)
            # Account for missing variables
            grads = zero_default(grads, variables)
            # Gradient vector products
            gvp = [grad * vec for (grad, vec) in zip(grads, vector_slices)]

        # The hessian vector products
        hvps = tape.gradient(gvp, variables)
        # Account again for missing variables
        hvps = zero_default(hvps, variables)
    elif mode == "fb":
        with tf.autodiff.ForwardAccumulator(variables, vector_slices) as acc:
            with tf.GradientTape() as tape:
                loss_value = loss()
                # Calculate the actual gradients
                grads = tape.gradient(loss_value, variables)
        hvps = acc.jvp(grads)

    else:
        raise NotImplemented
    # Pack the list of tensors
    return pack(hvps)


class CGOperator:
    def __init__(
        self, N: int, loss: Callable[[], tf.Tensor], variables: List[tf.Tensor]
    ) -> None:
        self.shape = (N, N)
        self.dtype = tf.float32
        self.loss = loss
        self.variables = variables

    def apply(self, v: tf.Tensor) -> tf.Tensor:
        return tf.expand_dims(Hvp(self.loss, self.variables, v), -1)


def iHvp(
    loss: Callable[[], tf.Tensor],
    variables: List[tf.Variable],
    vectors: List[tf.Tensor],
) -> List[tf.Tensor]:

    # for v in variables:
    #     assert isinstance(v, tf.Tensor), "All the variables should be tf.Tensor"

    packed_vector = pack(vectors)
    N = tf.size(packed_vector)

    cg_out = conjugate_gradient(CGOperator(N, loss, variables), packed_vector)
    cg_out = cg_out.x

    # Get dimension of each variable in weights
    dims = [tf.size(var) for var in variables]

    # Get list of partial sums of such dims
    accumulated_dims = [0] + list(itertools.accumulate(dims))

    # Find the slice of each variable in the packed vector
    packing_slices = [
        slice(start, end)
        for start, end in zip(accumulated_dims[:-1], accumulated_dims[1:])
    ]

    vector_slices = []
    for var, packing_slice in zip(variables, packing_slices):
        vector_slice = tf.reshape(cg_out[packing_slice], var.shape)
        vector_slices.append(vector_slice)

    return vector_slices


def iHvpB(
    y: Callable[[], tf.Tensor],
    xs: List[tf.Tensor],
    vs: List[tf.Tensor],
    method: str = "unstack",
) -> List[tf.Tensor]:
    """
    Compute Inverse Hessian Product (IHVP), batched, each 'vs' is in (n, ...) shape
    Parameters
    ----------
    y : Callable[[], tf.Tensor]
        The first parameter.
    xs : List[tf.Tensor]
        The second parameter.
    vs : List[tf.Tensor]
        The vectors, which should have same length as xs
    Returns
    -------
    List[tf.Tensor]
        IHVP, with shape same as xs
    """
    if method == "unstack":
        return iHvpB_unstack(y, xs, vs)
    elif method == "loop":
        return iHvpB_loop(y, xs, vs)
    else:
        raise RuntimeError(f"Unknown iHvpB method {method}")


def iHvpB_unstack(
    y: Callable[[], tf.Tensor], xs: List[tf.Tensor], vs: List[tf.Tensor]
) -> List[tf.Tensor]:
    """
    Compute Inverse Hessian Product (IHVP), batched, each 'vs' is in (n, ...) shape
    Parameters
    ----------
    y : tf.Tensor
        The first parameter.
    xs : List[tf.Tensor]
        The second parameter.
    vs : List[tf.Tensor]
        The vectors, which should have same length as xs
    Returns
    -------
    List[tf.Tensor]
        IHVP, with shape same as xs
    """
    assert isinstance(xs, list)
    assert isinstance(vs, list)
    assert len(xs) == len(vs)

    for x, v in zip(xs, vs):
        assert x.shape.is_compatible_with(
            v.shape[1:]
        ), f"The shape of x and v[0] should be same, got {x.shape}, {v.shape}"
    vs_unstacked = [tf.unstack(v) for v in vs]
    ihvps = [iHvp(y, xs, list(v)) for v in zip(*vs_unstacked)]
    ihvps = [tf.stack(ihvp) for ihvp in zip(*ihvps)]
    return ihvps


def iHvpB_loop(
    y: Callable[[], tf.Tensor], xs: List[tf.Tensor], vs: List[tf.Tensor]
) -> List[tf.Tensor]:
    """
    Compute Inverse Hessian Product (IHVP), but each vs is in (n, ...) shape
    Parameters
    ----------
    y : Callable[[], tf.Tensor]
        The first parameter.
    xs : List[tf.Tensor]
        The second parameter.
    vs : List[tf.Tensor]
        The vectors, which should have same length as xs
    Returns
    -------
    List[tf.Tensor]
        IHVP, with shape same as xs
    """
    assert isinstance(xs, list)
    assert isinstance(vs, list)
    assert len(xs) == len(vs)

    for x, v in zip(xs, vs):
        assert x.shape.is_compatible_with(
            v.shape[1:]
        ), f"The shape of x and v[0] should be same, got {x.shape}, {v.shape}"
    n = tf.shape(vs[0])[0]
    outputs = [tf.TensorArray(dtype=tf.float32, size=n) for _ in xs]

    for i in tf.range(n):
        per_sample_vectors = [v[i] for v in vs]
        iHvps = iHvp(y, xs, per_sample_vectors)
        outputs = [a.write(i, ihvp) for a, ihvp in zip(outputs, iHvps)]

    ihvps = [ihvp.stack() for ihvp in outputs]
    return ihvps
