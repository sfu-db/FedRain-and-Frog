import tensorflow as tf

from ..manager import ModelManager
from .ranker import Ranker


class SelfLossInfluenceRanker(Ranker):
    manager: ModelManager
    method: str

    def __init__(self, manager: ModelManager, method: str = "loop") -> None:
        self.manager = manager
        self.method = method

    def get_rank(self) -> tf.Tensor:
        rank = get_rank_impl(self.manager, self.method)
        return rank

    def name(self) -> str:
        return "SelfLoss"


@tf.function
def get_rank_impl(manager: ModelManager, method: str) -> tf.Tensor:
    egrads = manager.egrads()
    egrads_ = [-egrad for egrad in egrads]

    ihvps = manager.iHvpB(egrads_, method)
    influences = []
    for ihvp, egrad in zip(ihvps, egrads):
        egrad_2d = tf.reshape(egrad, (manager.ntrain, -1))  # n x m
        ihvp_2d = tf.reshape(ihvp, (manager.ntrain, -1))  # n x m
        influences.append(-tf.reduce_sum(egrad_2d * ihvp_2d, axis=1))
    return sum(influences)
