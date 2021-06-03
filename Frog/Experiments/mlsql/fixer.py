import numpy as np
import tensorflow as tf
from abc import abstractmethod

from .utils.utils import recall_k


class Fixer():
  def __init__(self, manager, truth, K):
    self.manager = manager
    self.deletions = tf.Variable(np.full((K,), -1), dtype=tf.int32)
    self.next_i = tf.Variable(0, dtype=tf.int32)
    self.truth = truth
  
  @abstractmethod
  def fix(self, rank, n=1):
    pass
  
  def get_candidate_removal(self, rank, n):
    next_i = self.next_i.value()
    
    removes = tf.argsort(-rank, name="removes")
    removes = self.manager.to_original_index(removes)
    to_remove = removes[:n]
    
    self.deletions.scatter_nd_update(tf.range(next_i, next_i + n)[:, None],
                                     to_remove, name="UpdateDeletion")
    self.next_i.assign_add(n)
    
    return to_remove
  
  def get_deletions(self):
    return self.deletions[: self.next_i]
  
  def recall_k(self):
    return recall_k(self.truth.numpy(), self.deletions.numpy())


class AutoFixer(Fixer):
  def fix(self, rank, n=1):
    to_remove = self.get_candidate_removal(rank, n)
    self.manager.set_delta(to_remove)


class HumanFixer(Fixer):
  def fix(self, rank, n=1):
    to_remove = self.get_candidate_removal(rank, n)
    if self.truth[to_remove]:
      self.manager.set_delta(to_remove)


class OracleFixer(Fixer):
  def fix(self, rank, n=1):
    to_remove = self.get_candidate_removal(rank, n)
    if self.truth[to_remove]:
      self.manager.set_y(to_remove, self.y[to_remove])
