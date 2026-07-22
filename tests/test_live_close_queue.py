"""Unit tests for LiveCloseQueue (Phase 1)."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.live_paper.runtime.live_close_queue import (
    ClosedCandleEvent,
    LiveCloseQueue,
    QueueClosedError,
)

IST = ZoneInfo("Asia/Kolkata")


def _event(minute: int) -> ClosedCandleEvent:
    return ClosedCandleEvent(
        symbol="NSE:NIFTY50-INDEX",
        timestamp=datetime(2026, 7, 22, 10, minute, tzinfo=IST),
        candle={"close": 100.0 + minute},
    )


def test_put_get_fifo() -> None:
    q = LiveCloseQueue(maxsize=8)
    assert q.put(_event(0)) is True
    assert q.put(_event(5)) is True
    first = q.get_nowait()
    second = q.get_nowait()
    assert first is not None and first.timestamp.minute == 0
    assert second is not None and second.timestamp.minute == 5
    assert q.get_nowait() is None


def test_drop_oldest_overflow() -> None:
    q = LiveCloseQueue(maxsize=2)
    assert q.put(_event(0)) is True
    assert q.put(_event(5)) is True
    assert q.put(_event(10)) is False
    assert q.overflow_count == 1
    assert q.qsize() == 2
    assert q.get_nowait().timestamp.minute == 5  # type: ignore[union-attr]
    assert q.get_nowait().timestamp.minute == 10  # type: ignore[union-attr]


def test_rejects_invalid_maxsize() -> None:
    with pytest.raises(ValueError):
        LiveCloseQueue(maxsize=0)


def test_close_rejects_put_and_wakes_getter() -> None:
    q = LiveCloseQueue(maxsize=4)
    result: list[object] = []

    def waiter() -> None:
        result.append(q.get(timeout=2.0))

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    q.close()
    t.join(timeout=2.0)
    assert q.is_closed
    assert result == [None]
    with pytest.raises(QueueClosedError):
        q.put(_event(0))


def test_drain_and_clear() -> None:
    q = LiveCloseQueue(maxsize=8)
    q.put(_event(0))
    q.put(_event(5))
    drained = q.drain(max_items=1)
    assert len(drained) == 1
    assert q.qsize() == 1
    q.clear()
    assert q.qsize() == 0


def test_reopen_allows_put() -> None:
    q = LiveCloseQueue(maxsize=2)
    q.close()
    q.reopen()
    assert q.put(_event(0)) is True


def test_snapshot() -> None:
    q = LiveCloseQueue(maxsize=3)
    q.put(_event(0))
    snap = q.snapshot()
    assert snap["qsize"] == 1
    assert snap["maxsize"] == 3
    assert snap["closed"] is False


def test_concurrent_producers() -> None:
    q = LiveCloseQueue(maxsize=32)

    def producer(start: int) -> None:
        for i in range(20):
            q.put(_event((start + i) % 60))

    threads = [threading.Thread(target=producer, args=(i * 3,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert q.qsize() == 32
    assert q.overflow_count == 48  # 80 puts - 32 capacity
    assert q.accepted_count == 80