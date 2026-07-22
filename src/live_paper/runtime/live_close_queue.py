"""
Bounded live closed-candle queue (Phase 1).

Thread-safe handoff from the candle-close producer to a future LiveEvalWorker.
Drop-oldest overflow; never blocks the producer for more than a brief lock hold.
Not wired into the live service until later phases enable ``enable_pipeline_v2``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Iterator


@dataclass(frozen=True)
class ClosedCandleEvent:
    """Payload for one closed candle awaiting live evaluation."""

    symbol: str
    timestamp: datetime
    candle: dict[str, Any] = field(default_factory=dict)
    enqueued_at_monotonic: float = field(default_factory=time.monotonic)


class QueueClosedError(RuntimeError):
    """Raised when putting into a closed queue."""


class LiveCloseQueue:
    """
    Bounded, thread-safe closed-candle queue with drop-oldest overflow.

    Public API
    ----------
    put(event) -> bool
        Enqueue. Returns True if accepted without drop. Returns False when an
        older item was dropped to make room (overflow). Raises QueueClosedError
        when the queue is closed.
    get(*, timeout=None) -> ClosedCandleEvent | None
        Blocking get. Returns None on timeout. Raises Empty behaviour via None.
    get_nowait() -> ClosedCandleEvent | None
        Non-blocking get; None if empty.
    qsize() / maxsize / overflow_count / dropped_total
    close()
        Stop accepting puts; existing items remain gettable.
    is_closed
    reopen()
        Test/lifecycle helper to accept again after close (not for production
        mid-run use).
    drain(*, max_items=None) -> list
        Remove and return up to max_items (or all) currently queued events.
    clear()
        Discard all queued events (does not reset overflow counters).
    snapshot() -> dict
    __len__ / __iter__ (non-destructive copy iteration)
    """

    def __init__(self, maxsize: int = 32) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = int(maxsize)
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._items: Deque[ClosedCandleEvent] = deque()
        self._closed = False
        self._overflow_count = 0
        self._accepted_count = 0
        self._dropped_total = 0

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def overflow_count(self) -> int:
        with self._lock:
            return self._overflow_count

    @property
    def dropped_total(self) -> int:
        with self._lock:
            return self._dropped_total

    @property
    def accepted_count(self) -> int:
        with self._lock:
            return self._accepted_count

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)

    def __len__(self) -> int:
        return self.qsize()

    def put(self, event: ClosedCandleEvent) -> bool:
        """
        Enqueue ``event``.

        Returns True when no overflow occurred. Returns False when the oldest
        event was dropped to accept this one.
        """
        if not isinstance(event, ClosedCandleEvent):
            raise TypeError(f"event must be ClosedCandleEvent, got {type(event)!r}")
        with self._not_empty:
            if self._closed:
                raise QueueClosedError("LiveCloseQueue is closed")
            dropped = False
            if len(self._items) >= self._maxsize:
                self._items.popleft()
                self._overflow_count += 1
                self._dropped_total += 1
                dropped = True
            self._items.append(event)
            self._accepted_count += 1
            self._not_empty.notify()
            return not dropped

    def get(self, *, timeout: float | None = None) -> ClosedCandleEvent | None:
        """Block until an item is available or ``timeout`` elapses (None = forever)."""
        with self._not_empty:
            if timeout is None:
                while not self._items:
                    if self._closed:
                        return None
                    self._not_empty.wait()
            else:
                deadline = time.monotonic() + float(timeout)
                while not self._items:
                    if self._closed:
                        return None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(remaining)
            return self._items.popleft()

    def get_nowait(self) -> ClosedCandleEvent | None:
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft()

    def close(self) -> None:
        """Stop accepting new events; wake blocked getters."""
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    def reopen(self) -> None:
        """Allow puts again (lifecycle / tests)."""
        with self._lock:
            self._closed = False

    def drain(self, *, max_items: int | None = None) -> list[ClosedCandleEvent]:
        with self._lock:
            if max_items is None or max_items >= len(self._items):
                out = list(self._items)
                self._items.clear()
                return out
            out: list[ClosedCandleEvent] = []
            for _ in range(max(0, int(max_items))):
                if not self._items:
                    break
                out.append(self._items.popleft())
            return out

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "qsize": len(self._items),
                "maxsize": self._maxsize,
                "closed": self._closed,
                "overflow_count": self._overflow_count,
                "dropped_total": self._dropped_total,
                "accepted_count": self._accepted_count,
            }

    def __iter__(self) -> Iterator[ClosedCandleEvent]:
        with self._lock:
            return iter(list(self._items))
