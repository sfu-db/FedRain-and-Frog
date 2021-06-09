from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import tensorflow as tf

from .fixer import Fixer
from .manager import ModelManager
from .models import LogReg
from .processor import Processor


@tf.function
def query(proc, manager):
    return proc.complain(manager, True).AQ, proc.complain(manager, False).AQ


@tf.function
def train(manager):
    manager.fit()


@dataclass
class Statistics:
    """
    Currently this is only implemented for aggregation queries
    """

    name: str
    gt: float  # ground truth AQ
    fixer: Fixer
    proc: Processor
    manaager: ModelManager = field(init=False)

    nfixes: List[int] = field(default_factory=list)

    # 0 is for the retrained model and 1 is for the model passed-in
    qvales0: List[float] = field(default_factory=list)
    qvalrs0: List[float] = field(default_factory=list)
    qvales1: List[float] = field(default_factory=list)
    qvalrs1: List[float] = field(default_factory=list)

    f1s0: List[float] = field(default_factory=list)
    f1s1: List[float] = field(default_factory=list)

    Xtest: tf.Tensor = field(init=False)
    ytest: np.ndarray = field(init=False)

    def __post_init__(self):
        _, C = self.proc.ytrain.shape
        Xcorr, ycorr, _ = self.proc.get_corrupted()
        super().__setattr__("manager", ModelManager(Xcorr, ycorr, LogReg(self.proc, C)))

        super().__setattr__("Xtest", self.proc.Xtest)
        super().__setattr__("ytest", tf.argmax(self.proc.ytest, axis=1).numpy())

    def record(self, manager: ModelManager, nfix: int):
        """
        Will copy the $\delta$ from the manager to fit a new model for reporting.

        Parameters
        ----------
        manager
            The model manager
        nfix
            How many points do you fix in this round?
        """

        self.nfixes.append(nfix)

        self.manager.model.set_weights(manager.model.get_weights())
        self.manager.delta.assign(manager.delta)
        train(self.manager)

        qvale, qvalr = query(self.proc, self.manager)
        self.qvales0.append(qvale.numpy()[0])
        self.qvalrs0.append(qvalr.numpy()[0])

        qvale, qvalr = query(self.proc, manager)
        self.qvales1.append(qvale.numpy()[0])
        self.qvalrs1.append(qvalr.numpy()[0])

        ypred = self.manager.predict(self.Xtest).numpy()
        f1 = f1_score(self.ytest, ypred, average="micro")
        self.f1s0.append(f1)

        ypred = manager.predict(self.Xtest).numpy()
        f1 = f1_score(self.ytest, ypred, average="micro")
        self.f1s1.append(f1)

    def get_qvals(self):
        df = pd.DataFrame(
            {
                "QExact0": self.qvales0,
                "QRelax0": self.qvalrs0,
                "QExact1": self.qvales1,
                "QRelax1": self.qvalrs1,
                "k": np.cumsum(self.nfixes),
            }
        )
        df = pd.melt(
            df,
            id_vars="k",
            value_vars=["QRelax0", "QExact0", "QRelax1", "QExact1"],
            var_name="QType",
            value_name="Value",
        )
        df["Method"] = self.name

        return df

    def get_f1s(self):
        df = pd.DataFrame(
            {"f10": self.f1s0, "f11": self.f1s1, "k": np.cumsum(self.nfixes)}
        )
        df = pd.melt(
            df,
            id_vars="k",
            value_vars=["f10", "f11"],
            var_name="F1Type",
            value_name="f1",
        )
        df["Method"] = self.name

        return df

    def get_rk(self):
        recalls = self.fixer.recall_k()
        rk = pd.DataFrame({"Recall": recalls, "K": np.arange(len(recalls)) + 1,})
        rk["Method"] = self.name

        return rk
