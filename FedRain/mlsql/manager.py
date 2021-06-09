from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple, cast

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from .utils import iHvp, iHvpB, no_tf_variable
from .utils.tfp_optimizer import train_tfp
from tqdm.notebook import tqdm


class ModelManager:
    X: tf.Variable
    y: tf.Variable
    delta: tf.Variable

    model: tf.keras.Model

    def __init__(self, X: tf.Tensor, y: tf.Tensor, model: tf.keras.Model):
        """Assuming the y is onehot encoded"""

        assert (
            isinstance(y, tf.Tensor) and y.shape.rank == 2 and y.shape[1] > 1
        ), f"labels should be onehot encoded but got shape {y.shape}"
        assert isinstance(X, tf.Tensor)

        super().__setattr__("X", tf.Variable(X, name="X"))
        super().__setattr__("y", tf.Variable(y, name="y"))

        super().__setattr__("model", model)
        self.model.build(self.X.shape)

        super().__setattr__(
            "delta",
            tf.Variable(tf.ones(shape=self.X.shape[0], dtype=tf.float32, name="delta")),
        )

    # def __setattr__(self, key, value):  # ! tf graph will cache the input signature
    #     raise RuntimeError(f"Immutable class, cannot set {key} as {value}")

    def get_remaining(self):
        """Get the remaining training data"""
        mask = self.delta.value() == 1
        return (
            tf.boolean_mask(self.X, mask),
            tf.boolean_mask(self.y, mask),
        )

    def fit(self, method: str = "tfp", tol: float = 0.001, max_iter: int = 1000):
        if method == "scipy":
            train_scipy(self.loss, self.variables)
        elif method == "tfp":
            train_tfp(self.loss, self.variables)
        elif method == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=0.1)
            last_loss = self.loss()
            opt.minimize(self.loss, self.variables)
            for _ in tqdm(range(max_iter)):
                if tf.math.abs(last_loss - self.loss()) <= tol:
                    break
                opt.minimize(self.loss, self.variables)
        else:
            raise RuntimeError(f"Unknown method {method}")

    def predict_logit(self, X: tf.Tensor) -> tf.Tensor:
        logits = self.model(X)
        return logits

    def predict_proba(self, X: tf.Tensor) -> tf.Tensor:
        logits = self.model(X)
        return tf.nn.softmax(logits)

    def predict(self, X: tf.Tensor) -> tf.Tensor:
        logits = self.model(X)
        probas = tf.nn.softmax(logits)
        return tf.argmax(probas, axis=1)

    def to_original_index(self, idx: tf.Tensor) -> tf.Tensor:
        """Deletions happen against the remaining indexes"""
        indices = tf.reshape(tf.where(self.delta), (-1,))
        indices = tf.cast(indices, tf.int32)
        return tf.gather(indices, idx, name="to_original_index")

    def set_delta(self, idx: tf.Tensor):
        """Usually delete a training point"""
        self.delta.scatter_nd_update(
            idx[:, None], tf.zeros_like(idx, dtype=tf.float32), name="SetDelta"
        )

    def set_y(self, idx: int, value: np.ndarray):
        """Change the label of a training point"""
        self.y.scatter_nd_update([[idx]], [value], name="SetY")

    def loss(self) -> tf.Tensor:
        """Total loss of training"""
        eloss = self.eloss()

        if self.model.losses:
            return tf.math.reduce_mean(eloss) + tf.add_n(self.model.losses) / tf.cast(
                tf.shape(eloss)[0], dtype=tf.float32
            )
        else:
            return tf.math.reduce_mean(eloss)

    def eloss(self, range: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> tf.Tensor:
        """
        Tensor of element-wise loss
        Returns
        -------
        tf.Tensor
            elementary loss, with the shape (n,)
        """
        data, labels = self.get_remaining()
        if range is not None:
            data = data[range[0] : range[1]]
            labels = labels[range[0] : range[1]]
        logits = self.model(data)
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name="elem_loss"
        )

    def egrads(self, range: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> tf.Tensor:
        with tf.GradientTape() as tape:
            eloss = self.eloss(range)
        return tape.jacobian(eloss, self.variables)

    @property
    def variables(self) -> List[tf.Variable]:
        return cast(List[tf.Variable], self.model.variables)

    @property
    def ntrain(self) -> tf.Tensor:
        return tf.cast(tf.reduce_sum(self.delta), dtype=tf.int32)

    def iHvp(self, vectors: List[tf.Tensor]) -> List[tf.Tensor]:
        """Get inverse hessian vector product"""
        return iHvp(self.loss, self.variables, vectors)

    def iHvpB(self, vectors: List[tf.Tensor], method: str = "loop") -> List[tf.Tensor]:
        """Get inverse hessian vector product"""
        return iHvpB(self.loss, self.variables, vectors, method=method)

    def egvp(self, vs: List[tf.Tensor], batch_size: Optional[int] = None) -> tf.Tensor:
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
