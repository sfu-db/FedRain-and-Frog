import tensorflow as tf
from collections import namedtuple
from abc import abstractmethod


# AC = None  # Aggregation Complaint, Shape: (n,)
# AQ = None  # Aggregation Result, Shape: (n,)
# PC = None  # Point Complaint, Shape: (n, m)
# PQ = None  # Point Query Result, Shape: (n, m)
fields = ('AC', 'AQ', 'PC', 'PQ', 'Ids', 'Agg', 'Groupby', 'Groupbyagg')
ComplaintRet = namedtuple('ComplaintRet', fields)
ComplaintRet.__new__.__defaults__ = (None,) * len(ComplaintRet._fields)

  # def sanity_checked(self) -> "ComplaintRet":
  #   AC, AQ, PC, PQ = self
  #
  #   assert (AC is None) == (AQ is None)
  #   assert (PC is None) == (PQ is None)
  #
  #   assert AC is not None or PC is not None
  #
  #   if AC is not None:
  #     assert AC.shape == AQ.shape
  #
  #   if PC is not None:
  #     assert PC.shape == PQ.shape
  #
  #   return self


class Processor:
  
  def __init__(self, **kwargs):
    tf.random.set_seed(10)
    self.xtrain, self.ytrain, self.corrsel = None, None, None
    self.params = dict(kwargs)
    for k, v in kwargs.items():
      setattr(self, k, v)
  
  def post_init(self, manager):
    """
    Do post init with a trained model.
    """
  
  def get_clean(self):
    return self.xtrain, self.ytrain
  
  def get_corrupted(self):
    return self.xtrain, self.ytrain, self.corrsel
  
  def set_tensor_variables(self, dtype=tf.float32, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, tf.constant(value, dtype=dtype, name=key))
  
  @abstractmethod
  def complain(self, manager, exact=False):
    pass
