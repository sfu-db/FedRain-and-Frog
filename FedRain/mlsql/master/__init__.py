import logging
from collections import defaultdict
from dataclasses import dataclass
from pickle import dumps, loads
from typing import Any, NamedTuple, Tuple, Union

import grpc
import numpy as np
import tensorflow as tf
from contexttimer import Timer
from phe import PaillierPrivateKey, PaillierPublicKey, paillier
from sklearn.metrics import classification_report
from tensorflow import transpose as T
from tensorflow.compat.v1.train import GradientDescentOptimizer, Optimizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from ..utils import iHvp
from ..logger import LogFile
from ..encryption import decrypt_to_tensor, encrypt_from_tensor
from ..processor import Processor
from ..slave import SlaveStub
from ..slave.slave_pb2 import Data, Empty
from ..utils import add_constant, pack
from ..utils.ihvp import iHvp

LOGGER = logging.getLogger(__name__)


@dataclass
class CGState:
    D: tf.Tensor
    z: tf.Tensor
    p: tf.Tensor
    r: tf.Tensor
    alpha: tf.Tensor
    beta: tf.Tensor
    Hvp: tf.Tensor


class Master:
    lf: LogFile
    slave_comm: SlaveStub
    processor: Processor

    Xtrain: tf.Tensor
    Xtest: tf.Tensor
    Xquery: tf.Tensor
    ytrain: tf.Tensor

    delta: tf.Variable
    weights: tf.Variable
    bias: tf.Variable

    public_key: PaillierPublicKey
    private_key: PaillierPrivateKey

    opt: tf.keras.optimizers.Optimizer

    n: int
    m: int

    def __init__(
        self,
        processor: Processor,
        lf: LogFile,
        n_length: int = 1024,
        opt: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(learning_rate=0.1),
        secure: bool = False,
        slave_conn: str = "localhost:50051",
    ) -> None:
        self.slave_comm = SlaveStub(
            grpc.insecure_channel(
                slave_conn,
                options=[
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ],
            )
        )
        self.processor = processor
        self.lf = lf

        self.n, self.m = processor.Xtrain.shape
        # self.Xtrain_full = processor.Xtrain_full
        self.Xtrain = processor.Xtrain  # n x (m )
        self.Xtest = processor.Xtest  # n x (m )
        self.Xquery = processor.Xquery  # n x (m )
        # self.Xquery_full = processor.Xquery_full  # n x (m )

        self.ytrain = tf.expand_dims(
            processor.ycorr, axis=-1
        )  # expand y to 2D for easy multiplication and transposing (n x 1)

        self.delta = tf.Variable(
            tf.ones(shape=self.Xtrain.shape[0], dtype=tf.float32, name="delta")
        )  # (n,)
        self.weights = tf.Variable(
            tf.random.uniform((self.Xtrain.shape[1], 1)), name="weights"
        )
        self.bias = tf.Variable(tf.random.uniform((1, 1)), name="bias")

        self.secure = secure
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=n_length
        )
        self.n_length = n_length

        self.opt = opt

    def get_remaining(self, full=False) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get the remaining training data"""
        mask = self.delta.value() == 1
        if not full:
            return (
                tf.boolean_mask(self.Xtrain, mask),
                tf.boolean_mask(self.ytrain, mask),
            )
        else:
            return (
                tf.boolean_mask(self.Xtrain_full, mask),
                tf.boolean_mask(self.ytrain, mask),
            )

    def fit(self, max_iter: int = 100, tol=1e-5, retrain=False, local=False):

        if retrain:
            stage = Data.Stage.RETRAIN
        else:
            stage = Data.Stage.TRAIN
        stage_name = Data.Stage.Name(stage)

        if local:
            clf = LogisticRegression(penalty="none", max_iter=max_iter, tol=tol)
            Xtrain, ytrain = self.get_remaining(full=True)
            clf.fit(Xtrain, tf.reshape(ytrain, (-1,)))

            self.weights.assign(clf.coef_[0, : self.m].reshape(-1, 1))
            self.bias.assign(clf.intercept_.reshape(1, 1))
            rpc_call(
                self.slave_comm.SetWeights, clf.coef_[0, self.m :], stage=stage,
            )

            return

        with self.lf.log(stage_name, "communication", "SetOpt"):
            rpc_call(self.slave_comm.SetOpt, self.opt, stage=stage)

        with self.lf.log(stage_name, "communication", "SetSecure"):
            rpc_call(self.slave_comm.SetSecure, self.secure, stage=stage)

        last_loss = -np.inf
        pbar = tqdm(range(max_iter), desc=stage_name)
        for i in pbar:
            with self.lf.log(stage_name, "communication", "UpdateDelta"):
                rpc_call(self.slave_comm.UpdateDelta, self.delta, stage=stage)

            with self.lf.log(stage_name, "communication", "GetThetaX"):
                Theta_B_X_B = rpc_call(self.slave_comm.GetThetaX, stage=stage)

            with self.lf.log(stage_name, "computation", "fit"):
                assert Theta_B_X_B.shape[1] == 1

                X, labels = self.get_remaining()
                Theta_A_X_A = X @ self.weights
                assert Theta_A_X_A.shape[1] == 1
                assert Theta_A_X_A.shape[0] == Theta_B_X_B.shape[0]
                Theta_X = Theta_A_X_A + Theta_B_X_B + self.bias

                yhat = tf.sigmoid(Theta_X)
                residual = labels - yhat
                gradient_A = (
                    T(-tf.reduce_mean(residual * X, axis=0, keepdims=True)),
                    -tf.reduce_mean(residual, axis=0, keepdims=True),
                )

            with self.lf.log(stage_name, "encrypt", "fit"):
                encrypted_residual = self.encrypt_opt(residual)

            with self.lf.log(stage_name, "communication", "ComputeEncryptedGradient"):
                gradient_B = rpc_call(
                    self.slave_comm.ComputeEncryptedGradient,
                    encrypted_residual,
                    stage=stage,
                )

            with self.lf.log(stage_name, "encrypt", "fit"):
                gradient_B = self.decrypt_opt(gradient_B)

            with self.lf.log(stage_name, "communication", "ApplyGradient"):
                rpc_call(self.slave_comm.ApplyGradient, gradient_B, stage=stage)

            with self.lf.log(stage_name, "computation", "fit"):
                self.opt.apply_gradients(
                    list(zip(gradient_A, (self.weights, self.bias)))
                )

                eloss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=Theta_X, name="elem_loss"
                )

                loss = tf.math.reduce_mean(eloss)
                # LOGGER.info(f"loss is {loss}")
                cont = tf.math.abs(last_loss - loss) > tol
            pbar.set_postfix({"loss": loss.numpy()})
            # if not cont.numpy():
            #     break

    def predict(self, target: str, prob: bool = False) -> tf.Tensor:
        if target == "train":
            Theta_B_X_B = rpc_call(self.slave_comm.GetThetaX, stage=Data.Stage.RETRAIN)
            assert Theta_B_X_B.shape[1] == 1

            X, _ = self.get_remaining()
            Theta_A_X_A = X @ self.weights

            Theta_X = Theta_A_X_A + Theta_B_X_B + self.bias
        elif target == "test":
            Theta_B_X_B = rpc_call(
                self.slave_comm.GetThetaXForTest, stage=Data.Stage.TEST
            )
            assert Theta_B_X_B.shape[1] == 1
            Theta_A_X_A = self.Xtest @ self.weights

            Theta_X = Theta_A_X_A + Theta_B_X_B + self.bias
        elif target == "query":
            Theta_B_X_B = rpc_call(
                self.slave_comm.GetThetaXForQuery, stage=Data.Stage.QUERY
            )
            assert Theta_B_X_B.shape[1] == 1
            Theta_A_X_A = self.Xquery @ self.weights

            Theta_X = Theta_A_X_A + Theta_B_X_B + self.bias
        else:
            raise NotImplementedError

        pred = tf.reshape(tf.math.sigmoid(Theta_X), (-1,))

        if prob:
            return pred
        else:
            return tf.cast(tf.math.greater(pred, 0.5), tf.float64)

    def qgrad(self) -> tf.Tensor:
        stage = Data.Stage.DEBUG
        stage_name = Data.Stage.Name(stage)

        Theta_A = self.weights

        with self.lf.log(stage_name, "communication", "QGradStep1"):
            Theta_B_X_B = rpc_call(self.slave_comm.QGradStep1, stage=stage)

        with self.lf.log(stage_name, "computation", "qgrad"):
            Theta_A_X_A = self.Xquery @ Theta_A
            Theta_X = Theta_A_X_A + Theta_B_X_B + self.bias

            denom = tf.exp(Theta_X) + 2 + tf.exp(-Theta_X)

            q_A = T(tf.reduce_sum(self.Xquery / denom, axis=0, keepdims=True))

        with self.lf.log(stage_name, "communication", "QGradStep2"):
            rpc_call(self.slave_comm.QGradStep2, denom, stage=stage)

        return q_A

    def hvp(self, cg_state: CGState, vname: str) -> tf.Tensor:
        assert vname in {"z", "p"}

        stage = Data.Stage.DEBUG
        stage_name = Data.Stage.Name(stage)

        with self.lf.log(stage_name, "communication", "HvpStep1"):
            X_B_v_B = rpc_call(self.slave_comm.HvpStep1, vname, stage=stage)

        with self.lf.log(stage_name, "computation", "hvp"):
            X_A, _ = self.get_remaining()

            if vname == "z":
                v_A = cg_state.z
            elif vname == "p":
                v_A = cg_state.p
            else:
                raise NotImplementedError(vname)

            if self.secure:
                mu_a = (T(X_A) @ (cg_state.D * X_A) @ v_A).numpy() + T(X_A).numpy() @ (
                    cg_state.D.numpy() * X_B_v_B
                )
            else:
                mu_a = T(X_A) @ (cg_state.D * X_A) @ v_A + T(X_A) @ (
                    cg_state.D * X_B_v_B
                )

            DX_A_v_A = cg_state.D * X_A @ v_A

        with self.lf.log(stage_name, "encrypt", "hvp"):
            encrypted_DX_A_v_A = self.encrypt_opt(DX_A_v_A)

        with self.lf.log(stage_name, "communication", "HvpStep2"):
            mu_b = rpc_call(
                self.slave_comm.HvpStep2, (encrypted_DX_A_v_A, vname), stage=stage
            )

        with self.lf.log(stage_name, "encrypt", "hvp"):
            mu_b = self.decrypt_opt(mu_b)

        with self.lf.log(stage_name, "communication", "HvpStep3"):
            mu_a = rpc_call(self.slave_comm.HvpStep3, (mu_a, mu_b), stage=stage)

        return mu_a

    def residual(self) -> tf.Tensor:
        """Return residual on train"""

        stage = Data.Stage.DEBUG
        stage_name = Data.Stage.Name(stage)

        with self.lf.log(stage_name, "communication", "residual"):
            Theta_B_X_B = rpc_call(self.slave_comm.GetThetaX, stage=stage)

        with self.lf.log(stage_name, "computation", "residual"):
            assert Theta_B_X_B.shape[1] == 1

            data, labels = self.get_remaining()
            Theta_A_X_A = data @ self.weights
            assert Theta_A_X_A.shape[1] == 1
            assert Theta_A_X_A.shape[0] == Theta_B_X_B.shape[0]
            Theta_X = Theta_A_X_A + Theta_B_X_B + self.bias

            yhat = tf.sigmoid(Theta_X)
            residual = labels - yhat
        return residual

    def debug(self) -> tf.Tensor:
        stage = Data.Stage.DEBUG
        stage_name = Data.Stage.Name(stage)

        rpc_call(self.slave_comm.SetSecure, self.secure, stage=stage)
        rpc_call(self.slave_comm.UpdateDelta, self.delta, stage=stage)

        residual = self.residual()

        with self.lf.log(stage_name, "computation", "debug"):
            D = (1 - residual) * (residual)

        with self.lf.log(stage_name, "encrypt", "debug"):
            encrypted_D = self.encrypt_opt(D)

        with self.lf.log(stage_name, "communication", "StartDebug"):
            rpc_call(self.slave_comm.StartDebug, encrypted_D, stage=stage)

        q = self.qgrad()

        z = self.cg(q, D)
        inf = self.influence(z)
        inf = tf.reshape(inf, (-1,))

        with self.lf.log(stage_name, "communication", "EndDebug"):
            rpc_call(self.slave_comm.EndDebug, stage=stage)

        return inf

    def influence(self, z: tf.Tensor) -> tf.Tensor:
        stage = Data.Stage.DEBUG
        stage_name = Data.Stage.Name(stage)

        residual = self.residual()
        with self.lf.log(stage_name, "computation", "influence"):
            X_A, _ = self.get_remaining()
            inf_A = residual * X_A @ z

        with self.lf.log(stage_name, "encrypt", "influence"):
            encrypted_residual = self.encrypt_opt(residual)

        with self.lf.log(stage_name, "communication", "Influence"):
            inf_B = rpc_call(self.slave_comm.Influence, encrypted_residual, stage=stage)

        with self.lf.log(stage_name, "encrypt", "influence"):
            inf_B = self.decrypt_opt(inf_B)

        with self.lf.log(stage_name, "computation", "influence"):
            return inf_A + inf_B

    def cg(
        self, q: tf.Tensor, D: tf.Tensor, tol: float = 1e-3, max_iter=100
    ) -> tf.Tensor:

        stage = Data.Stage.DEBUG
        stage_name = Data.Stage.Name(stage)

        cg_state = CGState(
            D=D,
            z=tf.random.uniform((self.Xtrain.shape[1], 1)),
            alpha=None,
            beta=None,
            p=None,
            r=None,
            Hvp=None,
        )

        cg_state.Hvp = self.hvp(cg_state, "z")

        rpc_call(self.slave_comm.CGInit, stage=stage)

        with self.lf.log(stage_name, "computation", "cg"):
            cg_state.p = cg_state.r = q - cg_state.Hvp

        tol *= self.rnorm(cg_state)

        for _ in range(0, max_iter):
            cg_state.Hvp = self.hvp(cg_state, "p")

            with self.lf.log(stage_name, "communication", "cg"):
                phvp = rpc_call(self.slave_comm.CGpHvp, stage=stage)
                r2 = rpc_call(self.slave_comm.CGGetR2, stage=stage)

            with self.lf.log(stage_name, "computation", "cg"):
                phvp += T(cg_state.p) @ cg_state.Hvp
                r2 += T(cg_state.r) @ cg_state.r

                cg_state.alpha = tf.reshape(r2 / phvp, ())

                cg_state.z += cg_state.alpha * cg_state.p
                cg_state.r -= cg_state.alpha * cg_state.Hvp

            with self.lf.log(stage_name, "communication", "cg"):
                rpc_call(self.slave_comm.UpdateZandR, cg_state.alpha, stage=stage)

            if self.rnorm(cg_state) < tol:
                break

            with self.lf.log(stage_name, "communication", "cg"):
                new_r2 = rpc_call(self.slave_comm.CGGetR2, stage=stage)

            with self.lf.log(stage_name, "computation", "cg"):
                cg_state.beta = tf.reshape(
                    (new_r2 + T(cg_state.r) @ cg_state.r) / r2, (),
                )

                cg_state.p = cg_state.r + cg_state.beta * cg_state.p

            with self.lf.log(stage_name, "communication", "cg"):
                rpc_call(self.slave_comm.UpdateP, cg_state.beta, stage=stage)

        return cg_state.z

    def rnorm(self, cg_state: CGState) -> tf.Tensor:
        stage = Data.Stage.DEBUG
        stage_name = Data.Stage.Name(stage)

        with self.lf.log(stage_name, "communication", "rnorm"):
            rnorm_part = rpc_call(self.slave_comm.CGGetRNorm, stage=stage)

        with self.lf.log(stage_name, "computation", "rnorm"):
            return rnorm_part + tf.norm(cg_state.r)

    def to_original_index(self, idx: tf.Tensor):
        """Deletions happen against the remaining indexes"""
        indices = tf.reshape(tf.where(self.delta), (-1,))
        indices = tf.cast(indices, tf.int32)
        return tf.gather(indices, idx, name="to_original_index")

    def set_delta(self, idx: tf.Tensor):
        """Usually delete a training point"""
        self.delta.scatter_nd_update(
            idx[:, None], tf.zeros_like(idx, dtype=tf.float32), name="set_delta"
        )

    @property
    def ntrain(self):
        return tf.cast(tf.reduce_sum(self.delta), dtype=tf.int32)

    def encrypt_opt(self, t: tf.Tensor) -> Union[tf.Tensor, np.ndarray]:
        if self.secure:
            return encrypt_from_tensor(t, self.public_key)
        else:
            return t

    def decrypt_opt(self, t: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        if self.secure:
            return decrypt_to_tensor(t, self.private_key)
        else:
            return t


def rpc_call(fn, data=None, *, stage):
    resp = fn(Data(data=dumps(data), stage=stage))

    if isinstance(resp, Data):
        return loads(resp.data)
    elif isinstance(resp, Empty):
        return None
    else:
        raise NotImplementedError(f"Unknown type {type(resp)}")
