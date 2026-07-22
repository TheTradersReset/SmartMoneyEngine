"""
Live evaluation worker (Phase 2).

Owns a dedicated thread that consumes ``LiveCloseQueue`` and invokes the
pipeline-provided evaluate callback. Never aggregates candles, never imports
or calls BUY_V3 / SELL_V6 directly, and never creates signals itself.
"""

from __future__ import annotations

import threading
from typing import Callable

from src.core.logger import logger
from src.live_paper.runtime.live_close_queue import ClosedCandleEvent, LiveCloseQueue

EvaluateFn = Callable[[ClosedCandleEvent], None]


class LiveEvalWorker:
    """
    Single-threaded consumer of closed-candle events for pipeline v2.

    Public API
    ----------
    start()
        Start the worker thread.
    request_stop()
        Signal the loop to exit after the queue is closed and drained.
    join(timeout=None) -> bool
        Wait for the worker thread to finish. Returns True if joined.
    is_alive / is_running / evaluated_count
    """

    def __init__(
        self,
        queue: LiveCloseQueue,
        *,
        evaluate_fn: EvaluateFn,
        name: str = "live-eval-worker",
        poll_timeout_sec: float = 0.2,
    ) -> None:
        if not callable(evaluate_fn):
            raise TypeError("evaluate_fn must be callable")
        self._queue = queue
        self._evaluate_fn = evaluate_fn
        self._name = name
        self._poll_timeout_sec = float(poll_timeout_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._evaluated_count = 0
        self._lock = threading.Lock()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def evaluated_count(self) -> int:
        with self._lock:
            return self._evaluated_count

    def start(self) -> None:
        """Start the worker thread. Idempotent if already alive."""
        if self.is_alive:
            return
        self._stop.clear()
        with self._lock:
            self._running = True
            self._evaluated_count = 0
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        """Request exit after in-flight / queued work completes."""
        self._stop.set()

    def join(self, timeout: float | None = None) -> bool:
        """Join the worker thread. Returns True when the thread has stopped."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _loop(self) -> None:
        try:
            while True:
                event = self._queue.get(timeout=self._poll_timeout_sec)
                if event is None:
                    if self._should_exit():
                        break
                    continue
                try:
                    self._evaluate_fn(event)
                except Exception:
                    logger.exception(
                        "LiveEvalWorker evaluate failed; continuing with remaining queue items"
                    )
                    continue
                with self._lock:
                    self._evaluated_count += 1
        finally:
            with self._lock:
                self._running = False

    def _should_exit(self) -> bool:
        if not self._stop.is_set():
            return False
        if not self._queue.is_closed:
            return False
        return self._queue.qsize() == 0