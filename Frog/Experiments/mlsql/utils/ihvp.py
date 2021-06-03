import itertools
import tensorflow as tf

from .conjugate_gradient import conjugate_gradient
from .utils import pack


# Sometimes a gradient may be missing because
# a variable did not participate in the computation (e.g. unused layer)
# we replace its gradient with zeros
def zero_default(grads, var_list):
  return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]


def Hvp(loss, variables, vector, mode="fb"):
  """
      mode: fb = forward backward
            bb = backward backward
  """
  # Get dimension of each variable in weights
  dims = [tf.size(var) for var in variables]
  
  # Get list of partial sums of such dims
  accumulated_dims = [0] + list(itertools.accumulate(dims))
  
  # Find the slice of each variable in the packed vector
  packing_slices = [slice(start, end) for start, end in
                    zip(accumulated_dims[:-1], accumulated_dims[1:])]
  
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
  def __init__(self, N, loss, variables):
    self.shape = (N, N)
    self.dtype = tf.float32
    self.loss = loss
    self.variables = variables
  
  def apply(self, v):
    return tf.expand_dims(Hvp(self.loss, self.variables, v), -1)


def iHvp(loss, variables, vectors):
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


'''
This replace loss with a explicit H.
'''


class CGOperator_exp:
  def __init__(self, N, H):
    self.shape = (N, N)
    self.dtype = tf.float32
    self.H = H
  
  def apply(self, v):
    return tf.matmul(self.H, v)

def iHvp_hess(hess, vectors):
  N = tf.size(vectors)
  cg_out = conjugate_gradient(CGOperator_exp(N, hess), vectors)
  return cg_out.x
  
def hessian(loss, variables, **kwargs):
  with tf.GradientTape(persistent=True) as tape:
    loss_value = loss(**kwargs)
    grads = tape.gradient(loss_value, variables)
    hess = tf.map_fn(lambda grad: pack(tape.gradient(grad, variables)),
                     pack(grads))
  #         hess_list = [pack(tape.gradient(grad, variables)) for grad in pack(grads)]
  #         for grad in grads:
  #             hess = tape.gradient(grad, variables)
  #             hess_list.append(pack(hess))
  return hess


def iHvp_exp(loss, variables, vectors, **kwargs):
  packed_vector = pack(vectors)
  N = tf.size(packed_vector)
  
  H = hessian(loss, variables, **kwargs)
  cg_out = conjugate_gradient(CGOperator_exp(N, H), packed_vector)
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
