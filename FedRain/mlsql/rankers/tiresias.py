from typing import List, Optional, Tuple

import numpy as np

import tensorflow as tf

from ..manager import ModelManager
from ..processor import Processor
from .ranker import Ranker


class TiresiasRanker(Ranker):

    proc: Processor
    manager: ModelManager
    ts_results: List[Tuple[tf.Tensor, tf.Tensor]]
    batch_size: Optional[tf.Tensor]
    nsamples: int

    def __init__(
        self,
        manager: ModelManager,
        proc: Processor,
        batch_size: Optional[int] = None,
        nsamples: int = 1,
    ) -> None:
        self.proc = proc
        self.manager = manager
        self.ts_results = self.proc.tiresias(self.manager, n=nsamples)
        self.nsamples = nsamples

        if batch_size is not None:
            self.batch_size = tf.constant(batch_size, dtype=tf.int32)
        else:
            self.batch_size = None

    def get_rank(self) -> tf.Tensor:
        manager = self.manager
        with tf.GradientTape() as tape:
            tsloss = 0
            for tsX, tsy in self.ts_results:
                tsprobs = manager.predict_proba(tsX)
                # tsy should be one-hot encoded for this to work
                tsloss += tf.reduce_sum(tsprobs * tsy)
            tsloss /= len(self.ts_results)

        # Add a minus here to force increase (instead of decrease)
        qgrads = [-grad for grad in tape.gradient(tsloss, manager.variables)]

        ihvps = manager.iHvp(qgrads)

        return -manager.egvp(ihvps, self.batch_size)

    def name(self) -> str:
        return "Tiresias"
