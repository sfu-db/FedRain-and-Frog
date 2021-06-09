from time import time
from typing import Optional
from contextlib import contextmanager
from sqlalchemy.sql import column, table, insert
from sqlalchemy import create_engine


class LogFile:

    who: str
    enabled: bool

    table: str

    def __init__(self, who: str, pgurl: str, table: str, enabled: bool = True) -> None:
        self.conn = create_engine(pgurl)
        self.table = table
        self.who = who
        self.enabled = enabled
        if enabled:
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{table}" (
                    id SERIAL PRIMARY KEY,
                    who TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    event TEXT NOT NULL,
                    msg TEXT NOT NULL,
                    start FLOAT8 NOT NULL,
                    "end" FLOAT8 NOT NULL
                )"""
            )

    def _log(
        self, stage: str, event: str, msg: Optional[str], start: int, end: int
    ) -> None:
        assert stage in ["TRAIN", "DEBUG", "RETRAIN", "TEST", "QUERY"], stage

        self.conn.execute(
            insert(
                table(
                    self.table,
                    column("who"),
                    column("stage"),
                    column("event"),
                    column("msg"),
                    column("start"),
                    column("end"),
                )
            ).values(
                [
                    {
                        "who": self.who,
                        "stage": stage,
                        "event": event,
                        "msg": msg,
                        "start": start,
                        "end": end,
                    }
                ]
            )
        )

    def close(self) -> None:
        self.f.close()

    @contextmanager
    def log(self, stage: str, event: str, msg: Optional[str] = None) -> None:
        start = time()
        yield
        end = time()

        if self.enabled:
            self._log(stage, event, msg, start, end)
