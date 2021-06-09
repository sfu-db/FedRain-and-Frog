"""Exp.

Usage:
  exp.py <n> <m> <k>

Options:
  --help     Show this screen.
  --version     Show version.
"""

import asyncio
import logging

from docopt import docopt
from grpc import aio
from mlsql.logger import LogFile
from mlsql.slave import Slave, add_SlaveServicer_to_server
from processors.breastCancer import BreastCancerProcessor
from processors.diabetes import DiabetesProcessor
from processors.mnist import MNISTProcessor


async def serve(n: int, m: int, k: int) -> None:
    server = aio.server()

    lf = LogFile(
        "slave",
        f"postgresql://postgres:postgres@172.17.0.1:15432/rain",
        f"exp-{n}-{m}-{k}",
        enabled=False,
    )
    processor = DiabetesProcessor("B", n, m)
    # processor = MNISTProcessor("B", corr_rate=0.7)
    # processor = BreastCancerProcessor("B", corr_rate=00)
    add_SlaveServicer_to_server(Slave(processor, lf, n_length=256), server)

    listen_addr = "[::]:50051"

    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        # Shuts down the server with 0 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = docopt(__doc__, version="Exp")
    print(args)
    n = int(args["<n>"])
    m = int(args["<m>"])
    k = int(args["<k>"])

    asyncio.run(serve(n, m, k))
