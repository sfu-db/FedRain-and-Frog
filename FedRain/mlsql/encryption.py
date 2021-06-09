from typing import Any

import numpy as np
import tensorflow as tf
from phe import PaillierPrivateKey, PaillierPublicKey, paillier


def encrypt_from_tensor(tensor: tf.Tensor, key: PaillierPublicKey):
    return encrypt(tensor.numpy(), key)


def encrypt(value: Any, key: PaillierPublicKey):
    assert isinstance(key, PaillierPublicKey)

    if isinstance(value, (np.ndarray, list)):
        return np.asarray([encrypt(v, key) for v in value])
    elif isinstance(value, (float, np.float32, np.float64)):

        return key.encrypt(float(value))
    else:
        raise TypeError(f"Cannot encrypt {type(value)}: {value}")


def decrypt_to_tensor(value: tf.Tensor, key: PaillierPrivateKey):
    decrypt_value = decrypt(value, key)
    return tf.convert_to_tensor(decrypt_value, dtype=tf.float32)


def decrypt(value: Any, key: PaillierPrivateKey):
    assert isinstance(key, PaillierPrivateKey)
    if isinstance(value, (np.ndarray, list)):
        return np.asarray([decrypt(v, key) for v in value])
    elif isinstance(value, paillier.EncryptedNumber):
        return key.decrypt(value)
    else:
        raise TypeError(f"Cannot decrypt {type(value)}: {value}")
