import numpy as np
import tensorflow as tf
import socket
import pickle
from sklearn.metrics import classification_report
from phe import paillier
import time
from time import sleep

from .utils.ihvp import iHvp, iHvp_exp
from .utils.utils import pack

RECEIVE_SIZE = 4096


# def get_sgd_fn():
#   @tf.function
#   def sgd_fn(manager, last_loss, opt, tol=1e-5):
#     with tf.GradientTape() as tape:
#       current_loss = manager.loss()
#     grads = tape.gradient(current_loss, manager.variables)
#     opt.apply_gradients(zip(grads, manager.variables))
#     cont = tf.math.abs(last_loss - current_loss) > tol
#     return current_loss, cont
  
#   return sgd_fn

class ModelManagerLM:
  def __init__(self, x, y, model, n_length, m_host="partyA", m_port=12345, s_host="partyB", s_port=12345):
    self.x = tf.Variable(x, name="x")
    self.y = tf.Variable(y, name="y")
    self.delta = tf.Variable(
      tf.ones(shape=self.x.shape[0], dtype=tf.float32, name="delta"))
    self.master_party = (m_host, m_port)
    self.slave_party = (s_host, s_port)
    self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=n_length)
    self.model = model
    
    # Initialize
    self.model(x)
  
  def report(self, x_train, y_train, x_test, y_test):
    print("Model name:", self.model.model_name())
    print("On Training\n", classification_report(y_train.numpy(), self.model.predict(x_train).numpy()))
    print("On Testing\n", classification_report(y_test.numpy(), self.model.predict(x_test).numpy()))
    
  def get_remaining(self):
    """Get the remaining training data"""
    mask = self.delta.value() == 1
    return tf.boolean_mask(self.x, mask), tf.boolean_mask(self.y, mask)
  
  # todo: 1. master send hyperparams; 2. encrypt pub key of b?
  def master_init_socket(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind(self.master_party)
    print("Party B connecting")
    self.master_send(self.public_key)
    self.public_key_b, _ = self.master_recv()
    print("Party B connected")
  
  def slave_init_socket(self):
    print("Party A connecting")
    self.public_key_a, _ = self.slave_recv()
    self.slave_send(self.public_key)
    print("Party A connected")
    
  def master_recv(self):
    self.socket.listen(1)
    conn, addr = self.socket.accept()
    data = []
    while True:
      packet=conn.recv(4096)
      if not packet:
        break
      data.append(packet)
    recv_time = time.time()
    conn.close()
    return pickle.loads(b"".join(data)), recv_time

  def master_send(self, data):
    self.socket.listen(1)
    conn, addr = self.socket.accept()
    data = pickle.dumps(data)
    send_time = time.time()
    conn.send(data)
    conn.close()
    return send_time

  def slave_recv(self):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while s.connect_ex(self.master_party) != 0:
      sleep(0.1)
    data = []
    while True:
      packet = s.recv(4096)
      if not packet:
          break
      data.append(packet)
    recv_time = time.time()
    s.close()
    return pickle.loads(b"".join(data)), recv_time
  
  def slave_send(self, data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while s.connect_ex(self.master_party) != 0:
      sleep(0.1)
    data = pickle.dumps(data)
    send_time = time.time()
    s.sendto(data, self.master_party)
    s.close()
    return send_time
    
  def master_evaluate(self):
    data, labels = self.get_remaining()
    return tf.squeeze(self.model(data)) - labels
  
  def slave_evaluate(self):
    data, labels = self.get_remaining()
    return tf.squeeze(self.model(data))
  
  def encrypt_from_tensor(self, tensor, key):
    value = tensor.numpy()
    return self.encrypt(value, key)
  
  def encrypt(self, value, key):
    if type(value) == np.ndarray or type(value) == list:
      return np.asarray([self.encrypt(v, key) for v in value])
    elif type(value) == float or type(value) == np.float32 \
      or type(value) == np.float64:
      return key.encrypt(float(value))
    else:
      raise TypeError("Not Supported Type!")
  
  def decrypt_to_tensor(self, value):
    decrypt_value = self.decrypt(value)
    return tf.convert_to_tensor(decrypt_value, dtype=tf.float32)
  
  def decrypt(self, value):
    if type(value) == np.ndarray or type(value) == list:
      return np.asarray([self.decrypt(v) for v in value])
    elif type(value) == paillier.EncryptedNumber:
      return self.private_key.decrypt(value)
    else:
      raise TypeError("Not Supported Type!")
  
  def update_gradient(self, opt, grads):
    grads = self.model.split(np.float32(grads))
    opt.apply_gradients(zip(grads, self.variables))
    
  def sgd_fn(self, data, labels, last_loss, opt, tol):
    with tf.GradientTape() as tape:
      current_loss = self.model.loss(data, labels)
    op_grad = getattr(self.model, "grad", None)
    if callable(op_grad):
      grads = op_grad(data, labels)
    else:
      grads = tape.gradient(current_loss, self.variables)
    opt.apply_gradients(zip(grads, self.variables))
    cont = tf.math.abs(last_loss - current_loss) > tol
    return current_loss, cont
  
  def fit(self, max_iter=100, print_value=False, tol=1e-5, lr=0.1):
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    last_loss = -np.inf
    for iteration in range(max_iter):
      data, labels = self.get_remaining()
      last_loss, cont = self.sgd_fn(data, labels, last_loss, opt, tol)
#       print(last_loss)
      if not cont:
        break
    if print_value:
      print("SGD loss:", last_loss)
      print("SGD steps:", iteration)
  
  def to_original_index(self, idx):
    """Deletions happen against the remaining indexes"""
    indices = tf.reshape(tf.where(self.delta), (-1,))
    indices = tf.cast(indices, tf.int32)
    return tf.gather(indices, idx, name="to_original_index")
  
  def set_delta(self, idx):
    """Usually delete a training point"""
    self.delta.scatter_nd_update(
      idx[:, None], tf.zeros_like(idx, dtype=tf.float32), name="set_delta"
    )
  
  def set_y(self, idx, value):
    """Change the label of a training point"""
    self.y.scatter_nd_update([[idx]], [value], name="set_y")
  
  def loss(self):
    """Total loss of training"""
    eloss = self.eloss()
    return tf.math.reduce_mean(eloss)
  
  def eloss(self):
    """
    Tensor of element-wise loss
    Returns
    -------
    tf.Tensor
        elementary loss, with the shape (n,)
    """
    data, labels = self.get_remaining()
    return self.model.eloss(data, labels)
  
  def egrads(self, **kwargs):
    op_egrads = getattr(self.model, "egrads", None)
    if callable(op_egrads):
        data, labels = self.get_remaining()
        return op_egrads(data, labels, **kwargs)
    else:
        print("Egrads")
        with tf.GradientTape(persistent=True) as tape:
          eloss = self.eloss()
        return tape.jacobian(eloss, self.variables, experimental_use_pfor=False)
  
  def hessian(self, is_master, **kwargs):
    op_hess = getattr(self.model, "hessian", None)
    if callable(op_hess):
        data, labels = self.get_remaining()
        if is_master:
            return op_hess(data, labels=labels, **kwargs)
        else:
            return op_hess(data, **kwargs)
    else:
        print("Hessian")
        with tf.GradientTape(persistent=True) as tape:
            loss_value = self.loss()
            grads = tape.gradient(loss_value, self.variables)
            hess = tf.map_fn(lambda grad: pack(tape.gradient(grad, self.variables)),
                             pack(grads))
        return hess
  
  @property
  def variables(self):
    return self.model.trainable_variables

  @property
  def ntrain(self):
    return tf.cast(tf.reduce_sum(self.delta), dtype=tf.int32)
  
  def iHvp(self, vectors):
    """Get inverse hessian vector product"""
    return iHvp_exp(self.loss, self.variables, vectors)
  
  def egvp(self, vs, batch_size=None):
    """
    Calculates egrads @ vs, conceptually (n x m) @ (m x 1)
    """
    if batch_size is not None:
      influences = tf.TensorArray(size=self.ntrain, dtype=tf.float32)
      
      for i in tf.range(0, self.ntrain, batch_size):
        upper = tf.math.minimum(self.ntrain, i + batch_size)
        
        with tf.autodiff.ForwardAccumulator(self.variables, vs) as acc:
          eloss = self.eloss(range=(i, upper))
        
        influence = acc.jvp(eloss)
        influences = influences.scatter(tf.range(i, upper), influence)
      influences = influences.stack(name="influence")
    else:
      with tf.autodiff.ForwardAccumulator(self.variables, vs) as acc:
        eloss = self.eloss()
      
      influences = acc.jvp(eloss)
    
    return influences
  
  def egmul(self, vs):
    data, labels = self.get_remaining()
    egrads = self.model.egrads(data, labels)
    return tf.matmul(egrads, tf.expand_dims(vs, -1))
