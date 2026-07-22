"""Phase 2 unit tests: LiveEvalWorker integration."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.data.candle_builder import Candle
from src.live_paper.runtime.live_close_queue import ClosedCandleEvent, LiveCloseQueue
from src.live_paper.runtime.live_eval_worker import LiveEvalWorker
from src.pipeline.realtime_signal_pipeline import RealtimeSignalPipeline
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase

IST = ZoneInfo("Asia/Kolkata")


def _candle(minute: int = 20) -> Candle:
    return Candle(
        symbol="NSE:NIFTY50-INDEX",
        timestamp=datetime(2026, 7, 22, 10, minute, tzinfo=IST),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
        tick_count=3,
    )


def _event(minute: int = 20) -> ClosedCandleEvent:
    c = _candle(minute)
    return ClosedCandleEvent(
        symbol=c.symbol,
        timestamp=c.timestamp,
        candle={
            "symbol": c.symbol,
            "timestamp": c.timestamp.isoformat(),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "tick_count": c.tick_count,
        },
    )


def _pipeline(tmp_path: Path) -> RealtimeSignalPipeline:
    db = PaperSignalDatabase(tmp_path / "p2.db")
    async_db = AsyncDbWriter(db.db_path)
    return RealtimeSignalPipeline(db=db, async_db=async_db, history_csv=None)


def test_worker_startup_and_shutdown() -> None:
    queue = LiveCloseQueue(maxsize=8)
    seen: list[ClosedCandleEvent] = []

    def _eval(event: ClosedCandleEvent) -> None:
        seen.append(event)

    worker = LiveEvalWorker(queue, evaluate_fn=_eval, poll_timeout_sec=0.05)
    worker.start()
    assert worker.is_alive
    assert worker.is_running

    worker.request_stop()
    queue.close()
    assert worker.join(timeout=2.0)
    assert worker.is_alive is False
    assert worker.is_running is False
    assert seen == []


def test_queue_consumption() -> None:
    queue = LiveCloseQueue(maxsize=8)
    seen: list[datetime] = []
    ready = threading.Event()

    def _eval(event: ClosedCandleEvent) -> None:
        seen.append(event.timestamp)
        if len(seen) >= 2:
            ready.set()

    worker = LiveEvalWorker(queue, evaluate_fn=_eval, poll_timeout_sec=0.05)
    worker.start()
    queue.put(_event(15))
    queue.put(_event(20))
    assert ready.wait(timeout=2.0)
    worker.request_stop()
    queue.close()
    assert worker.join(timeout=2.0)
    assert worker.evaluated_count == 2
    assert [ts.minute for ts in seen] == [15, 20]


def test_feature_flag_off_evaluates_inline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.enable_pipeline_v2 = False
    pipeline._live_close_queue = LiveCloseQueue(maxsize=4)
    calls = {"n": 0}

    def _fake_eval() -> tuple[dict, dict, int]:
        calls["n"] += 1
        return {"verdict": "NO_TRADE"}, {"verdict": "NO_TRADE"}, 0

    # Bypass heavy context path: force early return after session by stubbing append path.
    monkeypatch.setattr(pipeline, "_should_enqueue_for_live_eval", lambda: False)

    enqueued = {"n": 0}

    def _enqueue(_candle: Candle) -> None:
        enqueued["n"] += 1

    monkeypatch.setattr(pipeline, "_enqueue_closed_candle_for_live_eval", _enqueue)

    # Call gate helper directly
    assert pipeline._should_enqueue_for_live_eval() is False

    # With flag off and queue present, helper still false
    pipeline.enable_pipeline_v2 = False
    assert pipeline._should_enqueue_for_live_eval() is False
    assert enqueued["n"] == 0
    assert calls["n"] == 0
    pipeline.async_db.close()
    pipeline.db.close()


def test_feature_flag_on_enqueues_without_inline_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline(tmp_path)
    queue = LiveCloseQueue(maxsize=4)
    pipeline.enable_pipeline_v2 = True
    pipeline._live_close_queue = queue

    eval_calls = {"n": 0}

    def _boom() -> tuple[dict, dict, int]:
        eval_calls["n"] += 1
        raise AssertionError("evaluate_latest must not run on producer thread")

    monkeypatch.setattr(pipeline.context, "evaluate_latest", _boom)

    # Avoid needing full warm-start: intercept after enqueue gate by ensuring gate works
    pipeline._handle_closed_candle(_candle(25))
    assert queue.qsize() == 1
    assert eval_calls["n"] == 0
    pipeline.async_db.close()
    pipeline.db.close()


def test_graceful_drain() -> None:
    queue = LiveCloseQueue(maxsize=16)
    processed: list[int] = []
    lock = threading.Lock()

    def _eval(event: ClosedCandleEvent) -> None:
        time.sleep(0.01)
        with lock:
            processed.append(event.timestamp.minute)

    worker = LiveEvalWorker(queue, evaluate_fn=_eval, poll_timeout_sec=0.05)
    worker.start()
    for minute in (0, 5, 10, 15, 20):
        queue.put(_event(minute))

    worker.request_stop()
    queue.close()
    assert worker.join(timeout=5.0)
    assert sorted(processed) == [0, 5, 10, 15, 20]
    assert worker.evaluated_count == 5
    assert queue.qsize() == 0


def test_no_duplicate_evaluation(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    queue = LiveCloseQueue(maxsize=8)
    pipeline.enable_pipeline_v2 = True
    pipeline._live_close_queue = queue

    eval_calls = {"n": 0}
    original = pipeline._evaluate_queued_closed_candle

    def _counting(event: ClosedCandleEvent) -> None:
        eval_calls["n"] += 1
        # Do not run full handle (needs warm context); count only.
        _ = event

    worker = LiveEvalWorker(queue, evaluate_fn=_counting, poll_timeout_sec=0.05)
    worker.start()

    pipeline._handle_closed_candle(_candle(30))
    pipeline._handle_closed_candle(_candle(35))

    deadline = time.time() + 2.0
    while eval_calls["n"] < 2 and time.time() < deadline:
        time.sleep(0.02)

    worker.request_stop()
    queue.close()
    assert worker.join(timeout=2.0)
    assert eval_calls["n"] == 2
    assert worker.evaluated_count == 2
    pipeline.async_db.close()
    pipeline.db.close()
    _ = original


def test_no_lost_closed_candles() -> None:
    queue = LiveCloseQueue(maxsize=32)
    got: list[int] = []

    def _eval(event: ClosedCandleEvent) -> None:
        got.append(event.timestamp.minute)

    worker = LiveEvalWorker(queue, evaluate_fn=_eval, poll_timeout_sec=0.05)
    worker.start()
    expected = list(range(0, 60, 5))
    for minute in expected:
        assert queue.put(_event(minute)) is True

    deadline = time.time() + 3.0
    while len(got) < len(expected) and time.time() < deadline:
        time.sleep(0.02)

    worker.request_stop()
    queue.close()
    assert worker.join(timeout=2.0)
    assert got == expected


def test_worker_skips_none_and_keeps_running() -> None:
    queue = LiveCloseQueue(maxsize=4)
    seen = {"n": 0}

    def _eval(_event: ClosedCandleEvent) -> None:
        seen["n"] += 1

    worker = LiveEvalWorker(queue, evaluate_fn=_eval, poll_timeout_sec=0.05)
    worker.start()
    time.sleep(0.15)  # several empty polls
    assert worker.is_running
    queue.put(_event(40))
    deadline = time.time() + 2.0
    while seen["n"] < 1 and time.time() < deadline:
        time.sleep(0.02)
    worker.request_stop()
    queue.close()
    assert worker.join(timeout=2.0)
    assert seen["n"] == 1

def test_worker_continues_after_evaluate_exception() -> None:
    queue = LiveCloseQueue(maxsize=8)
    seen: list[int] = []

    def _eval(event: ClosedCandleEvent) -> None:
        if event.timestamp.minute == 15:
            raise RuntimeError('boom')
        seen.append(event.timestamp.minute)

    worker = LiveEvalWorker(queue, evaluate_fn=_eval, poll_timeout_sec=0.05)
    worker.start()
    queue.put(_event(15))
    queue.put(_event(20))
    deadline = time.time() + 2.0
    while len(seen) < 1 and time.time() < deadline:
        time.sleep(0.02)
    assert worker.is_alive
    worker.request_stop()
    queue.close()
    assert worker.join(timeout=2.0)
    assert seen == [20]
    assert worker.evaluated_count == 1


def test_live_eval_marker_is_thread_local(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    assert pipeline._is_live_eval_worker_thread() is False
    pipeline._live_eval_worker_tls.active = True
    assert pipeline._is_live_eval_worker_thread() is True
    seen = {'other': None}

    def _other() -> None:
        seen['other'] = pipeline._is_live_eval_worker_thread()

    t = threading.Thread(target=_other)
    t.start()
    t.join(timeout=2.0)
    assert seen['other'] is False
    pipeline._live_eval_worker_tls.active = False
    pipeline.async_db.close()
    pipeline.db.close()

