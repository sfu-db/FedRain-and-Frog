from __future__ import annotations

from abc import abstractmethod
from json import JSONEncoder, dumps
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.sql import text

import tensorflow as tf
from tensorflow.python.framework.dtypes import DType

from .manager import ModelManager

if TYPE_CHECKING:
    from .fixer import Fixer
    from .rankers import Ranker

DB_RAW: str = str((Path(__file__).parent.parent / "migrations" / ".env").resolve())
with open(DB_RAW) as f:
    CONN_STR = f.read().strip()

_, DBURL = CONN_STR.split("=")


DB = create_engine(DBURL)
INSERT_STMT: str = f"""
    INSERT INTO experiments(
        proc, seed, params, bootstrap_time, elapses, deletions, truth, ambiguity, NA, NP
    ) VALUES(
        :proc, :seed, :params, :bootstrapT, :elapses, :deletions, :truth, :ambiguity, :NA, :NP
    )
"""


class ComplaintRet(NamedTuple):
    AC: Optional[tf.Tensor] = None  # Aggregation Complaint, Shape: (n,)
    AQ: Optional[tf.Tensor] = None  # Aggregation Result, Shape: (n,)
    PC: Optional[tf.Tensor] = None  # Point Complaint, Shape: (n, m)
    PQ: Optional[tf.Tensor] = None  # Point Query Result, Shape: (n, m)

    def sanity_checked(self) -> "ComplaintRet":
        AC, AQ, PC, PQ = self

        assert (AC is None) == (AQ is None)
        assert (PC is None) == (PQ is None)

        assert AC is not None or PC is not None

        if AC is not None:
            assert AC.shape == AQ.shape

        if PC is not None:
            assert PC.shape == PQ.shape

        return self


TiresiasOutput = List[Tuple[tf.Tensor, tf.Tensor]]  # tsX and tsY


class Processor:
    params: Dict[str, Any]
    seed: int

    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def post_init(self, manager: ModelManager) -> None:
        """
        Do post init with a trained model.
        """

    def get_clean(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.Xtrain, self.ytrain

    def get_corrupted(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.Xtrain, self.ycorr, self.corrsel

    @abstractmethod
    def complain(self, qprob: tf.Tensor, exact: bool = False) -> ComplaintRet:
        pass

    @property
    def auto_seed(self) -> int:
        """
        Return a random seed and plus one to it
        """
        self.seed += 1
        return self.seed - 1

    def set_tensor_variables(self, dtype: DType = tf.float32, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, tf.constant(value, dtype=dtype, name=key))

    def insert(
        self,
        fixer: Fixer,
        ranker: Ranker,
        model: tf.keras.Model,
        elapses: np.ndarray,
        ambiguity: Optional[float],
        bootstrapT: float,
        AC: Optional[np.ndarray],
        AQ: Optional[np.ndarray],
        PC: Optional[np.ndarray],
        PQ: Optional[np.ndarray],
    ):
        assert (
            isinstance(elapses, np.ndarray)
            and len(elapses.shape) == 2
            and elapses.shape[1] == 3
        ), f"Elapses Check Error, got {elapses.shape}"

        assert ambiguity is None or isinstance(
            ambiguity, Number
        ), f"got ambiguity {ambiguity}: {type(ambiguity)}"
        if ambiguity is not None:
            ambiguity = float(ambiguity)
        if AC is None:
            AC = np.zeros((0, 0), dtype=float)
        if PC is None:
            PC = np.zeros((0, 0), dtype=float)
        if AQ is None:
            AQ = np.zeros((0, 0), dtype=float)
        if PQ is None:
            PQ = np.zeros((0, 0), dtype=float)

        assert (
            isinstance(AC, np.ndarray)
            and isinstance(AQ, np.ndarray)
            and isinstance(PC, np.ndarray)
            and isinstance(PQ, np.ndarray)
        )
        AC, AQ, PC, PQ = (
            AC.astype(float),
            AQ.astype(float),
            PC.astype(float),
            PQ.astype(float),
        )

        record = {
            "proc": self.__class__.__name__,
            "seed": int(self.seed),
            "params": dumps(
                {
                    "ranker": ranker.name(),
                    "fixer": fixer.name(),
                    "model": model.__class__.__name__,
                    **self.params,
                },
                cls=NumpyEncoder,
            ),
            "elapses": elapses.tolist(),
            "deletions": fixer.get_deletions().numpy().astype(int).tolist(),
            "truth": fixer.truth.numpy().tolist(),
            "ambiguity": ambiguity,
            "bootstrapT": bootstrapT,
            "NA": AC.shape[1],
            "NP": PC.shape[1],
        }

        DB.execute(text(INSERT_STMT), record)


class NumpyEncoder(JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, o):
        if isinstance(
            o,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(o)
        elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.ndarray,)):  #### This is the fix
            return o.tolist()
        return JSONEncoder.default(self, o)
