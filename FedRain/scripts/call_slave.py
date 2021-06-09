"""Exp.

Usage:
  exp.py <n> <m> <k>

Options:
  --help     Show this screen.
  --version     Show version.
"""
import asyncio
import logging
from pickle import dump

import numpy as np
import tensorflow as tf
from contexttimer import Timer
from docopt import docopt
from mlsql.logger import LogFile
from mlsql.master import Master
from mlsql.utils import recall_k
from processors.breastCancer import BreastCancerProcessor
from processors.diabetes import DiabetesProcessor
from processors.mnist import MNISTProcessor
from sklearn.metrics import classification_report
from tqdm import tqdm


async def run(n: int, m: int, k: float) -> None:
    lf = LogFile(
        "master",
        f"postgresql://postgres:postgres@172.17.0.1:15432/rain",
        f"exp-{n}-{m}-{k}",
        enabled=False,
    )

    processor = DiabetesProcessor("A", n, m, corr_rate=0.3)
    # processor = MNISTProcessor("A", corr_rate=0.4)
    # processor = BreastCancerProcessor("A", corr_rate=0.0)

    manager = Master(
        processor,
        lf,
        n_length=256,
        opt=tf.keras.optimizers.SGD(lr=0.1),
        secure=True,
        slave_conn="PA5:50051",
    )
    with Timer() as t:
        manager.fit(1000, local=False)
    print(f"train time: {t}")

    y = manager.predict("test", prob=False).numpy()
    print(classification_report(processor.ytest.numpy(), y))

    delete_per_round = 10
    deleted = np.zeros((0,), "i8")

    print(np.sum(processor.corrsel), "corruptions")

    inftime = 0
    retraintime = 0
    pbar = tqdm(range(int(np.ceil(k / delete_per_round))), desc="DEBUG")
    for _ in pbar:
        Q = tf.reduce_sum(manager.predict("query", prob=True)).numpy()
        C = tf.reduce_sum(processor.yquery).numpy()
        pbar.set_postfix({"Q": Q, "C": C})

        with Timer() as t:
            inf = manager.debug().numpy()

            indices = tf.argsort(
                inf, direction="ASCENDING"
            )  # ASCENDING: top indices increases the query value
            indices = manager.to_original_index(indices[:delete_per_round])
            # assert (manager.delta.numpy()[indices.numpy()] == 1).all()

            manager.set_delta(indices)

            # assert (manager.delta.numpy()[indices.numpy()] == 0).all()

            deleted = np.append(deleted, indices[:delete_per_round].numpy())
        inftime += t.elapsed

        print(
            "Recall @ K",
            recall_k(processor.corrsel, deleted)[-1],
            "GT",
            np.minimum(len(deleted) / np.sum(processor.corrsel), 1),
            "RD",
            len(deleted) / len(processor.corrsel),
        )

        with Timer() as t:
            manager.fit(100, retrain=True)
        retraintime += t.elapsed

        if len(deleted) >= k:
            break

    print(f"influence time: {inftime}")
    print(f"retraining time: {retraintime}")

    np.save("MNIST-deletions.npy", deleted)
    y = manager.predict("test", prob=False).numpy()
    print(classification_report(processor.ytest.numpy(), y))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = docopt(__doc__, version="Exp")

    n = int(args["<n>"])
    m = int(args["<m>"])
    k = int(args["<k>"])

    asyncio.run(run(n, m, k))

