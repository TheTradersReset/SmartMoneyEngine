"""
Background SQLite writer for the realtime paper signal pipeline.

Queues persistence operations so the hot path never blocks on disk I/O.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.logger import logger
from src.storage.sqlite import DEFAULT_DB_PATH, PaperSignalDatabase


@dataclass(frozen=True)
class DbWriteJob:
    """One queued persistence operation."""

    op: str
    payload: dict[str, Any]


class AsyncDbWriter:
    """
    Background thread that drains a queue of SQLite write jobs.

    Parameters
    ----------
    db_path : Path | str
        SQLite database path.
    max_queue_size : int
        Maximum queued jobs before ``enqueue`` raises ``queue.Full``.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        *,
        max_queue_size: int = 10_000,
    ) -> None:
        self._db_path = Path(db_path)
        self._queue: queue.Queue[DbWriteJob | None] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, name="async-db-writer", daemon=True)
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._last_write_ms: float = 0.0
        self._worker.start()

    @property
    def last_write_ms(self) -> float:
        return self._last_write_ms

    @property
    def pending_jobs(self) -> int:
        with self._pending_lock:
            return self._pending

    def enqueue(self, op: str, payload: dict[str, Any]) -> float:
        """
        Queue a write job.

        Returns
        -------
        float
            Time spent queueing (milliseconds).
        """
        started = time.perf_counter()
        with self._pending_lock:
            self._pending += 1
        self._queue.put(DbWriteJob(op=op, payload=payload), block=True)
        return (time.perf_counter() - started) * 1000.0

    def flush(self, *, timeout_seconds: float = 30.0) -> None:
        """Wait until all queued jobs have been written."""
        deadline = time.perf_counter() + timeout_seconds
        while self.pending_jobs > 0 and time.perf_counter() < deadline:
            time.sleep(0.05)

    def close(self, *, timeout_seconds: float = 30.0) -> None:
        """Stop the worker after draining the queue."""
        self.flush(timeout_seconds=timeout_seconds)
        self._stop_event.set()
        self._queue.put(None, block=True)
        self._worker.join(timeout=timeout_seconds)

    def _run(self) -> None:
        db = PaperSignalDatabase(self._db_path)
        try:
            while not self._stop_event.is_set():
                job = self._queue.get()
                if job is None:
                    break
                started = time.perf_counter()
                try:
                    self._execute(db, job)
                except Exception as exc:  # noqa: BLE001 — worker must stay alive
                    logger.exception("Async DB write failed op=%s: %s", job.op, exc)
                finally:
                    self._last_write_ms = (time.perf_counter() - started) * 1000.0
                    with self._pending_lock:
                        self._pending = max(0, self._pending - 1)
                    self._queue.task_done()
        finally:
            db.close()

    @staticmethod
    def _execute(db: PaperSignalDatabase, job: DbWriteJob) -> None:
        op = job.op
        payload = job.payload
        if op == "candle":
            db.insert_candle(**payload)
        elif op == "signal_decision":
            db.insert_signal_decision(payload)
        elif op == "signal":
            db.insert_signal(payload)
        elif op == "event":
            db.log_event(
                signal_id=payload.get("signal_id"),
                event_type=payload["event_type"],
                details=payload["details"],
            )
        elif op == "signal_outcome":
            db.update_signal_outcome(**payload)
        else:
            raise ValueError(f"Unsupported async DB op: {op}")
