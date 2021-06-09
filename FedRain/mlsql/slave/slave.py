from dataclasses import dataclass
from pickle import dumps, loads
from typing import Optional

import numpy as np
from phe import PaillierPrivateKey, PaillierPublicKey, paillier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import transpose as T

from tensorflow.compat.v1.train import GradientDescentOptimizer, Optimizer

from ..encryption import encrypt_from_tensor
from ..logger import LogFile
from ..processor import Processor
from ..utils import add_constant, pack
from .slave_pb2 import Data, Empty
from .slave_pb2_grpc import SlaveServicer


@dataclass
class DebugState:
    D: tf.Tensor
    z: tf.Tensor
    p: Optional[tf.Tensor] = None
    r: Optional[tf.Tensor] = None
    Hvp: Optional[tf.Tensor] = None
    q: Optional[tf.Tensor] = None


class Slave(SlaveServicer):
    lf: LogFile
    Xtrain: tf.Tensor
    Xtest: tf.Tensor
    Xquery: tf.Tensor

    delta: Optional[tf.Tensor]

    weights: tf.Variable
    opt: Optimizer

    public_key: PaillierPublicKey
    private_key: PaillierPrivateKey

    debug_state: Optional[DebugState] = None

    def __init__(self, processor: Processor, lf: LogFile, n_length: int = 1024) -> None:
        self.Xtrain = processor.Xtrain  # n x (m )
        self.Xtest = processor.Xtest  # n x (m )
        self.Xquery = processor.Xquery  # n x (m )

        self.lf = lf
        self.weights = tf.Variable(
            tf.random.uniform((self.Xtrain.shape[1], 1)), name="weights"
        )

        self.delta = None  # wait for A to set

        self.opt = None

        self.secure = False
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=n_length
        )

    async def SetSecure(self, request: Data, _context) -> Empty:
        stage_name = Data.Stage.Name(request.stage)
        with self.lf.log(stage_name, "deserialize", "SetSecure"):
            secure = loads(request.data)

        self.secure = secure

        with self.lf.log(stage_name, "serialize", "SetSecure"):
            return Empty()

    async def SetWeights(self, request: Data, _context) -> Empty:
        stage_name = Data.Stage.Name(request.stage)
        with self.lf.log(stage_name, "deserialize", "SetWeights"):
            weights = loads(request.data)

        assert len(weights.shape) == 1
        assert weights.shape[0] == self.Xtrain.shape[1]

        self.weights.assign(weights[:, None])

        with self.lf.log(stage_name, "serialize", "SetWeights"):
            return Empty()

    async def GetWeights(self, request: Data, _context) -> Data:
        stage_name = Data.Stage.Name(request.stage)
        with self.lf.log(stage_name, "deserialize", "GetWeights"):
            ...

        with self.lf.log(stage_name, "serialize", "GetWeights"):
            return Data(data=dumps(self.weights))

    async def UpdateDelta(self, request: Data, _context) -> Empty:
        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "UpdateDelta"):
            delta = loads(request.data)

        with self.lf.log(stage, "computation", "UpdateDelta"):
            assert delta.shape[0] == self.Xtrain.shape[0]
            self.delta = delta

        with self.lf.log(stage, "serialize", "UpdateDelta"):
            return Empty()

    async def GetThetaX(self, request: Data, _context) -> Data:
        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "GetThetaX"):
            ...

        with self.lf.log(stage, "computation", "GetThetaX"):
            X = self.get_remaining()
            ThetaX = X @ self.weights

        with self.lf.log(stage, "serialize", "GetThetaX"):
            return Data(data=dumps(ThetaX))

    async def GetThetaXForTest(self, request: Data, _context) -> Data:
        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "GetThetaXForText"):
            ...

        with self.lf.log(stage, "computation", "GetThetaXForTest"):
            ThetaX = self.Xtest @ self.weights

        with self.lf.log(stage, "serialize", "GetThetaXForTest"):
            return Data(data=dumps(ThetaX))

    async def GetThetaXForQuery(self, request: Data, _context) -> Data:
        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "GetThetaXForQuery"):
            ...

        with self.lf.log(stage, "computation", "GetThetaXForQuery"):
            ThetaX = self.Xquery @ self.weights

        with self.lf.log(stage, "serialize", "GetThetaXForQuery"):
            return Data(data=dumps(ThetaX))

    async def ComputeEncryptedGradient(self, request: Data, _context) -> Data:
        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "ComputeEncryptedGradient"):
            residual = loads(request.data)

        with self.lf.log(stage, "computation", "ComputeEncryptedGradient"):
            data = self.get_remaining()
            assert residual.shape[0] == data.shape[0]
            assert residual.shape[1] == 1

            if self.secure:
                gradient = 0 - np.mean(residual * data.numpy(), axis=0, keepdims=True).T
            else:
                gradient = T(-tf.reduce_mean(residual * data, axis=0, keepdims=True))

        with self.lf.log(stage, "serialize", "ComputeEncryptedGradient"):
            return Data(data=dumps(gradient))

    async def ApplyGradient(self, request: Data, _context) -> Empty:
        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "ApplyGradient"):
            gradient = loads(request.data)

        with self.lf.log(stage, "computation", "ApplyGradient"):
            self.opt.apply_gradients([(gradient, self.weights)])

        with self.lf.log(stage, "serialize", "ApplyGradient"):
            return Empty()

    async def SetOpt(self, request: Data, _context) -> Empty:
        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "SetLR"):
            opt = loads(request.data)

        self.opt = opt

        with self.lf.log(stage, "serialize", "SetLR"):
            return Empty()

    async def StartDebug(self, request: Data, _context) -> Empty:
        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "StartDebug"):
            D = loads(request.data)

        with self.lf.log(stage, "computation", "StartDebug"):
            self.debug_state = DebugState(
                D=D, z=tf.random.uniform((self.Xtrain.shape[1], 1)),
            )

        with self.lf.log(stage, "serialize", "StartDebug"):
            return Empty()

    async def QGradStep1(self, request: Empty, _context) -> Data:
        assert self.debug_state is not None
        assert self.debug_state.q is None

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "QGradStep1"):
            ...

        with self.lf.log(stage, "computation", "QGradStep1"):
            X_B = self.Xquery
            Theta_B = self.weights

        with self.lf.log(stage, "serialize", "QGradStep1"):
            return Data(data=dumps(X_B @ Theta_B))

    async def QGradStep2(self, request: Data, _context) -> Empty:
        assert self.debug_state is not None
        assert self.debug_state.q is None

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "QGradStep2"):
            denom = loads(request.data)

        with self.lf.log(stage, "computation", "QGradStep2"):
            self.debug_state.q = T(
                tf.reduce_sum(self.Xquery / denom, axis=0, keepdims=True)
            )
        with self.lf.log(stage, "serialize", "QGradStep2"):
            return Empty()

    async def HvpStep1(self, request: Data, _context) -> Data:
        assert self.debug_state is not None

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "HvpStep1"):
            vname = loads(request.data)

        with self.lf.log(stage, "computation", "HvpStep1"):
            X_B = self.get_remaining()

            if vname == "z":
                v_B = self.debug_state.z
            elif vname == "p":
                v_B = self.debug_state.p
            else:
                raise NotImplementedError
            assert v_B.shape[0] == X_B.shape[1]
            X_B_v_B = X_B @ v_B

        with self.lf.log(stage, "serialize", "HvpStep1"):
            return Data(data=dumps(X_B_v_B))

    async def HvpStep2(self, request: Data, _context) -> Data:
        assert self.debug_state is not None

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "HvpStep2"):
            DX_A_v_A, vname = loads(request.data)

        with self.lf.log(stage, "computation", "HvpStep2"):

            assert vname in {"z", "p"}

            X_B = self.get_remaining()

            if vname == "z":
                v_B = self.debug_state.z
            elif vname == "p":
                v_B = self.debug_state.p
            else:
                raise NotImplementedError

            assert DX_A_v_A.shape[0] == X_B.shape[0]

            if self.secure:
                mu_b = T(X_B).numpy() @ (
                    DX_A_v_A + self.debug_state.D * (X_B @ v_B).numpy()
                )
            else:
                mu_b = T(X_B) @ DX_A_v_A + T(X_B) @ (self.debug_state.D * X_B @ v_B)

        with self.lf.log(stage, "serialize", "HvpStep2"):
            return Data(data=dumps(mu_b))

    async def HvpStep3(self, request: Data, _context) -> Data:
        assert self.debug_state is not None

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "HvpStep3"):
            mu_a, mu_b = loads(request.data)

        with self.lf.log(stage, "computation", "HvpStep3"):
            assert mu_b.shape[0] == self.Xtrain.shape[1]

            self.debug_state.Hvp = mu_b

        with self.lf.log(stage, "serialize", "HvpStep3"):
            return Data(data=dumps(mu_a))

    async def GetDebugState(self, request: Data, _context) -> Data:
        return Data(data=dumps(self.debug_state))

    async def CGInit(self, request: Empty, _context) -> Empty:

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "CGInit"):
            ...

        with self.lf.log(stage, "computation", "CGInit"):
            self.debug_state.p = self.debug_state.q - self.debug_state.Hvp
            self.debug_state.r = self.debug_state.p
        return Empty()

    async def CGGetR2(self, request: Empty, _context) -> Data:

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "CGGetR2"):
            ...

        with self.lf.log(stage, "computation", "CGGetR2"):
            r2 = T(self.debug_state.r) @ self.debug_state.r

        with self.lf.log(stage, "serialize", "CGGetR2"):
            return Data(data=dumps(r2))

    async def CGpHvp(self, request: Empty, _context) -> Data:

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "CGpHvp"):
            ...

        with self.lf.log(stage, "computation", "CGpHvp"):
            pHvp = T(self.debug_state.p) @ self.debug_state.Hvp

        with self.lf.log(stage, "serialize", "CGpHvp"):
            return Data(data=dumps(pHvp))

    async def UpdateZandR(self, request: Data, _context) -> Empty:

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "UpdateZandR"):
            alpha = loads(request.data)

        with self.lf.log(stage, "computation", "UpdateZandR"):
            assert tf.rank(alpha).numpy() == 0

            self.debug_state.z += alpha * self.debug_state.p
            self.debug_state.r -= alpha * self.debug_state.Hvp

        with self.lf.log(stage, "serialize", "UpdateZandR"):
            return Empty()

    async def UpdateP(self, request: Data, _context) -> Empty:

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "UpdateP"):
            beta = loads(request.data)

        with self.lf.log(stage, "computation", "UpdateP"):
            assert tf.rank(beta).numpy() == 0

            self.debug_state.p = self.debug_state.r + beta * self.debug_state.p

        with self.lf.log(stage, "serialize", "UpdateP"):
            return Empty()

    async def CGGetRNorm(self, request: Empty, _context) -> Data:

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "CGGetRNorm"):
            ...

        with self.lf.log(stage, "computation", "CGGetRNorm"):
            rnorm = tf.norm(self.debug_state.r)

        with self.lf.log(stage, "serialize", "CGGetRNorm"):
            return Data(data=dumps(rnorm))

    async def Influence(self, request: Data, _context) -> Data:

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "Influence"):
            residual = loads(request.data)

        with self.lf.log(stage, "computation", "Influence"):
            X_B = self.get_remaining()

            if self.secure:
                inf = residual * (X_B @ self.debug_state.z).numpy()
            else:
                inf = residual * X_B @ self.debug_state.z

        with self.lf.log(stage, "serialize", "Influence"):
            return Data(data=dumps(inf))

    async def EndDebug(self, request: Empty, _context) -> Empty:

        stage = Data.Stage.Name(request.stage)

        with self.lf.log(stage, "deserialize", "EndDebug"):
            ...

        with self.lf.log(stage, "computation", "EndDebug"):
            self.debug_state = None

        with self.lf.log(stage, "serialize", "EndDebug"):
            return Empty()

    def get_remaining(self) -> tf.Tensor:
        """Get the remaining training data"""
        mask = self.delta.value() == 1
        return tf.boolean_mask(self.Xtrain, mask)

    def to_original_index(self, idx):
        """Deletions happen against the remaining indexes"""
        indices = tf.reshape(tf.where(self.delta), (-1,))
        indices = tf.cast(indices, tf.int32)
        return tf.gather(indices, idx, name="to_original_index")
