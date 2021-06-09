import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    Reshape,
)
from typing import Optional, Sequence
from ..processor import Processor


class SimpleCNN(tf.keras.Model):
    def __init__(
        self, proc: Processor, nclass: int, input_shape: Optional[Sequence[int]] = None
    ):
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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.model(inputs)
