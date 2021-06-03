from time import time
import tensorflow as tf

from .utils.utils import Timer


class InfluenceRanker():
  def __init__(self, manager, on, batch_size=None):
    self.manager = manager
    self.on = on
    if batch_size is not None:
      self.batch_size = tf.constant(batch_size, dtype=tf.int32)
    else:
      self.batch_size = batch_size
    self.timer = Timer()
  
  def get_rank(self):
    manager = self.manager
    
    with tf.GradientTape() as tape:
      # then = time()
      AC, AQ, PC, PQ, _, _, _, _ = self.on(manager, exact=False)
      # self.timer.timer["query"].append(time() - then)
      
      assert (AC is None) == (AQ is None)
      if AC is not None and AQ is not None:
        assert AC.shape.is_compatible_with(AQ.shape), \
          f"Shape of AC and AQ should be equal, got {AC.shape}, {AQ.shape}"
        assert AC.shape.rank == 1
      
      assert (PC is None) == (PQ is None)
      if PC is not None and PQ is not None:
        assert PC.shape.is_compatible_with(PQ.shape), \
          f"Shape of PC and PQ should be equal, got {PC.shape}, {PQ.shape}"
        assert PC.shape.rank == 2
      
      # then = time()
      
      query_loss = 0
      if AC is not None:
        query_loss += tf.norm(AC - AQ, 1)
      if PC is not None:
        query_loss += tf.nn.softmax_cross_entropy_with_logits(labels=PC,
                                                              logits=PQ)
    # self.timer.timer["query_loss"].append(time() - then)
    
    # then = time()
    qgrads = tape.gradient(query_loss, manager.variables)
    # self.timer.timer["ihvp"].append(time() - then)
    
    ihvps = manager.iHvp(qgrads)
    return -manager.egvp(ihvps, self.batch_size)

  def predict(self):
    return self.get_rank()
