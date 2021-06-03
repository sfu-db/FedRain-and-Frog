import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random


class MLP(tf.keras.Model):
  def __init__(self, nclass):
    super().__init__()
    
    model = Sequential()
    model.add(Dense(8, activation=tf.nn.relu))
    model.add(Dense(4, activation=tf.nn.relu))
    model.add(Dense(4, activation=tf.nn.relu))
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
      (tf.math.multiply(labels, tf.squeeze(tf.math.log(logits)))
       + tf.math.multiply(1 - labels, tf.squeeze(tf.math.log(1 - logits))))
    )


class MLP_Linear(tf.keras.Model):
  def __init__(self, nclass):
    super().__init__()
    
    model = Sequential()
    model.add(Dense(8, activation=tf.nn.relu))
    model.add(Dense(4, activation=tf.nn.relu))
    model.add(Dense(4, activation=tf.nn.relu))
    model.add(Dense(nclass))
    
    self.model = model
    self.c = tf.Variable(random.random(), name='c')
  
  def call(self, inputs):
    return self.c * self.f(inputs)
  
  def f(self, inputs):
    return tf.keras.activations.sigmoid(self.model(inputs))
  
  def eloss(self, inputs, labels):
    f = self.c * self.f(inputs)
    return 0.5 * (tf.squeeze(f) - labels) ** 2
  
  def loss(self, inputs, labels):
    f = self.c * self.f(inputs)
    return 0.5 * tf.reduce_mean((tf.squeeze(f) - labels) ** 2)
