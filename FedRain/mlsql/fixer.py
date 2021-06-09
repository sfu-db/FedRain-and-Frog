from abc import ABC, abstractmethod

import numpy as np

import tensorflow as tf

from .manager import ModelManager
from .utils import recall_k


class Fixer(ABC):
    manager: ModelManager
    deletions: tf.Variable
    next_i: tf.Variable
    truth: tf.Tensor

    def __init__(self, manager: ModelManager, truth: tf.Tensor, K: int):
        self.manager = manager
        self.deletions = tf.Variable(np.full((K,), -1), dtype=tf.int32)
        self.next_i = tf.Variable(0, dtype=tf.int32)
        self.truth = truth

    @abstractmethod
    def fix(self, rank: tf.Tensor, n: int = 1):
        pass

    def get_candidate_removal(self, rank: tf.Tensor, n: int) -> tf.Tensor:
        next_i = self.next_i.value()

        removes = tf.argsort(-rank, name="removes")
        removes = self.manager.to_original_index(removes)
        to_remove = removes[:n]

        self.deletions.scatter_nd_update(
            tf.range(next_i, next_i + n)[:, None], to_remove, name="UpdateDeletion"
        )
        self.next_i.assign_add(n)

        return to_remove

    def get_deletions(self) -> tf.Tensor:
        return self.deletions[: self.next_i]

    def name(self) -> str:
        return self.__class__.__name__

    def recall_k(self) -> np.ndarray:
        return recall_k(self.truth.numpy(), self.deletions.numpy())


class AutoFixer(Fixer):
    def fix(self, rank: tf.Tensor, n: int = 1) -> None:
        to_remove = self.get_candidate_removal(rank, n)
        self.manager.set_delta(to_remove)


class HumanFixer(Fixer):
    def fix(self, rank: tf.Tensor, n: int = 1) -> None:
        to_remove = self.get_candidate_removal(rank, n)
        if self.truth[to_remove]:
            self.manager.set_delta(to_remove)


class OracleFixer(Fixer):
    def fix(self, rank: tf.Tensor, n: int = 1) -> None:
        to_remove = self.get_candidate_removal(rank, n)
        if self.truth[to_remove]:
            self.manager.set_y(to_remove, self.y[to_remove])
