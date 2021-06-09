# Pack list of tensors into a single tensor
from typing import List

import tensorflow as tf


def pack(tensors: List[tf.Tensor]) -> tf.Tensor:
    if not tensors:
        return None
    elif len(tensors) == 1:
        return tf.reshape(tensors[0], [-1])
    else:
        flattened = [tf.reshape(tensor, [-1]) for tensor in tensors]
        return tf.concat(flattened, 0)
