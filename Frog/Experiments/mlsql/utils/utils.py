import numpy as np
import tensorflow as tf

from collections import defaultdict
from time import time
from sklearn.metrics import precision_recall_curve

# Pack list of tensors into a single tensor
def pack(tensors):
  if not tensors:
    return None
  elif len(tensors) == 1:
    return tf.reshape(tensors[0], [-1])
  else:
    flattened = [tf.reshape(tensor, [-1]) for tensor in tensors]
    return tf.concat(flattened, 0)


def recall_k(truth, rank):
  flags = np.zeros(len(truth))
  flags[rank] = 1 + np.arange(len(rank))[::-1]
  _, recalls, _ = precision_recall_curve(truth + 0, flags)
  return recalls[-2:0:-1]


class Timer:
  def __init__(self):
    self.timer = defaultdict(list)
  
  def measure(self, name, func):
    then = time()
    ret = func()
    self.timer[name].append(time() - then)
    return ret