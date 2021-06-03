import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
  Conv1D,
  Conv2D,
  Dense,
  Dropout,
  Flatten,
  MaxPooling1D,
  MaxPooling2D,
  Reshape,
)


class SimpleCNN(tf.keras.Model):
  def __init__(self, nclass, input_shape=None):
    super().__init__()
    
    model = Sequential()
    if input_shape:
      model.add(Reshape(input_shape))
    model.add(Conv2D(28, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    # model.add(Dropout(0.2))
    model.add(Dense(nclass))
    
    self.model = model
  
  def call(self, inputs):
    return self.model(inputs)


class SimpleCNN1D(tf.keras.Model):
  def __init__(self, nclass, input_shape=None):
    super().__init__()
    
    model = Sequential()
    if input_shape:
      model.add(Reshape(input_shape))
    model.add(Conv1D(8, kernel_size=3))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(16, activation=tf.nn.relu))
    model.add(Dense(nclass))
    
    self.model = model
  
  def call(self, inputs):
    return tf.keras.activations.sigmoid(self.model(inputs))
  
  def eloss(self, inputs, labels):
    logits = tf.keras.activations.sigmoid(self.model(inputs))
    return -(
      tf.math.multiply(labels, tf.squeeze(tf.math.log(logits)))
      + tf.math.multiply(1 - labels, tf.squeeze(tf.math.log(1 - logits)))
    )
  
  def loss(self, inputs, labels):
    logits = tf.keras.activations.sigmoid(self.model(inputs))
    return -tf.reduce_mean(
      tf.math.multiply(labels, tf.squeeze(tf.math.log(logits)))
      + tf.math.multiply(1 - labels, tf.squeeze(tf.math.log(1 - logits)))
    )


class SimpleCNN1D_Linear(tf.keras.Model):
  def __init__(self, nclass, input_shape=None):
    super().__init__()
    
    model = Sequential()
    if input_shape:
      model.add(Reshape(input_shape))
    model.add(Conv1D(8, kernel_size=3))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(16, activation=tf.nn.relu))
    model.add(Dense(nclass))
    
    self.model = model
    self.model = model
    self.c = tf.Variable(tf.random.uniform([1], seed=1), name='c')
  
  def call(self, inputs):
    return self.c * self.f(inputs)
  
  def f(self, inputs):
    return tf.keras.activations.sigmoid(self.model(inputs))
  
  def eloss(self, inputs, labels):
    f = self.c * self.f1(inputs)
    return 0.5 * (tf.squeeze(f) - labels) ** 2
  
  def loss(self, inputs, labels):
    f = self.c * self.f1(inputs)
    return 0.5 * tf.reduce_mean((tf.squeeze(f) - labels)) ** 2
