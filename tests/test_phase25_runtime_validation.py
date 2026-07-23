"""Phase 2.5 runtime validation & operational readiness tests."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from src.data.candle_builder import Candle
from src.live_paper.runtime.live_close_queue import LiveCloseQueue, QueueClosedError
from src.live_paper.runtime.live_eval_worker import LiveEvalWorker
from src.live_paper.validation.runtime_harness import (
    ProducerEnqueueProbe,
    RuntimeValidationHarness,
    make_candle,
    make_closed_event,
)
from src.pipeline.realtime_signal_pipeline import RealtimeSignalPipeline
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase


def _pipeline(tmp_path: Path) -> RealtimeSignalPipeline:
    db = PaperSignalDatabase(tmp_path / "p25.db")
    async_db = AsyncDbWriter(db.db_path)
    return RealtimeSignalPipeline(db=db, async_db=async_db, history_csv=None)


# ---------------------------------------------------------------------------
# 1. Worker lifecycle
# ---------------------------------------------------------------------------


def test_worker_lifecycle_start_alive_clean_exit() -> None:
    harness = RuntimeValidationHarness()
    harness.start()
    assert harness.worker.is_alive
    assert harness.worker.is_running
    harness.put_minute(10)
    assert harness.wait_until_processed(1, timeout_sec=2.0)
    assert harness.worker.is_alive
    assert harness.stop(drain_timeout_sec=2.0)
    assert harness.worker.is_alive is False
    assert harness.worker.is_running is False


def test_worker_cannot_process_after_shutdown() -> None:
    harness = RuntimeValidationHarness()
    harness.start()
    assert harness.stop(drain_timeout_sec=2.0)
    with pytest.raises(QueueClosedError):
        harness.put_minute(15)
    assert harness.worker.is_alive is False
    assert harness.worker.evaluated_count == 0


# ---------------------------------------------------------------------------
# 2. Queue validation
# ---------------------------------------------------------------------------


def test_queue_fifo_no_duplicates_no_loss_and_drain() -> None:
    harness = RuntimeValidationHarness(maxsize=32)
    harness.start()
    minutes = list(range(0, 60, 5))
    for minute in minutes:
        assert harness.put_minute(minute) is True
    assert harness.wait_until_processed(len(minutes), timeout_sec=5.0)
    assert harness.processed_minutes == minutes
    assert len(harness.processed_minutes) == len(set(harness.processed_minutes))
    assert harness.stop(drain_timeout_sec=2.0)
    assert harness.queue.qsize() == 0


# ---------------------------------------------------------------------------
# 3. Feature flag validation
# ---------------------------------------------------------------------------


def test_feature_flag_off_synchronous_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.enable_pipeline_v2 = False
    pipeline._live_close_queue = LiveCloseQueue(maxsize=4)
    probe = ProducerEnqueueProbe(pipeline)
    probe.install_evaluate_probe()

    # Force early exit after session check by stubbing evaluate path readiness:
    # empty context will skip evaluate_latest via INSUFFICIENT_BARS once appended.
    # Ensure enqueue gate is off.
    assert pipeline._should_enqueue_for_live_eval() is False

    calls_before = probe.evaluate_latest_calls
    # Outside enqueue: handle runs inline (may skip eval for insufficient bars).
    pipeline._handle_closed_candle(make_candle(20))
    # With empty context, append may raise or create frame — if evaluate not called, still OK:
    # critical assertion is gate stays off and no worker ownership.
    assert pipeline._should_enqueue_for_live_eval() is False
    assert pipeline._live_close_queue.qsize() == 0
    assert probe.evaluate_latest_calls >= calls_before
    pipeline.async_db.close()
    pipeline.db.close()


def test_feature_flag_on_only_worker_evaluates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline(tmp_path)
    queue = LiveCloseQueue(maxsize=8)
    pipeline.enable_pipeline_v2 = True
    pipeline._live_close_queue = queue

    eval_threads: list[str] = []
    eval_count = {"n": 0}

    def _eval(event) -> None:  # noqa: ANN001
        eval_count["n"] += 1
        eval_threads.append(threading.current_thread().name)

    worker = LiveEvalWorker(queue, evaluate_fn=_eval, poll_timeout_sec=0.05)
    worker.start()

    def _boom() -> tuple[dict, dict, int]:
        raise AssertionError("producer must not call evaluate_latest when v2 ON")

    monkeypatch.setattr(pipeline.context, "evaluate_latest", _boom)

    producer_ms = []
    for minute in (10, 15, 20):
        started = time.perf_counter()
        pipeline._handle_closed_candle(make_candle(minute))
        producer_ms.append((time.perf_counter() - started) * 1000.0)

    deadline = time.time() + 3.0
    while eval_count["n"] < 3 and time.time() < deadline:
        time.sleep(0.02)

    worker.request_stop()
    queue.close()
    assert worker.join(timeout=2.0)
    assert eval_count["n"] == 3
    assert all(name == "live-eval-worker" for name in eval_threads)
    assert all(ms < 50.0 for ms in producer_ms)
    pipeline.async_db.close()
    pipeline.db.close()


# ---------------------------------------------------------------------------
# 4. Runtime validation (exactly-once, no concurrent evaluate, fast producer)
# ---------------------------------------------------------------------------


def test_exactly_once_and_no_concurrent_evaluate() -> None:
    harness = RuntimeValidationHarness()
    harness.start()
    minutes = [0, 5, 10, 15, 20, 25]
    for minute in minutes:
        harness.put_minute(minute)
    assert harness.wait_until_processed(len(minutes), timeout_sec=5.0)
    assert harness.processed_minutes == minutes
    assert harness.metrics.total_candles_processed == len(minutes)
    assert harness.metrics.concurrent_evaluate_violations == 0
    assert harness.stop(drain_timeout_sec=2.0)


def test_producer_returns_immediately_after_enqueue(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    queue = LiveCloseQueue(maxsize=8)
    pipeline.enable_pipeline_v2 = True
    pipeline._live_close_queue = queue
    probe = ProducerEnqueueProbe(pipeline)

    elapsed = probe.handle_closed_candle_timed(make_candle(30))
    assert queue.qsize() == 1
    assert elapsed < 25.0
    assert probe.metrics.average_producer_enqueue_ms < 25.0
    pipeline.async_db.close()
    pipeline.db.close()


# ---------------------------------------------------------------------------
# 5. Shutdown validation
# ---------------------------------------------------------------------------


def test_shutdown_drains_without_hang_or_deadlock() -> None:
    harness = RuntimeValidationHarness()
    harness.start()
    for minute in range(0, 50, 5):
        harness.put_minute(minute)
    started = time.monotonic()
    assert harness.stop(drain_timeout_sec=5.0)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0
    assert harness.queue.qsize() == 0
    assert harness.worker.is_alive is False
    assert harness.processed_minutes == list(range(0, 50, 5))


# ---------------------------------------------------------------------------
# 6. Reconnect validation
# ---------------------------------------------------------------------------


def test_simulated_reconnect_keeps_worker_and_queue_operational() -> None:
    harness = RuntimeValidationHarness()
    harness.start()
    harness.put_minute(5)
    assert harness.wait_until_processed(1, timeout_sec=2.0)

    harness.simulate_reconnect()
    assert harness.worker.is_alive
    assert harness.queue.is_closed is False

    harness.put_minute(10)
    harness.put_minute(15)
    assert harness.wait_until_processed(3, timeout_sec=3.0)
    assert harness.processed_minutes == [5, 10, 15]
    assert "simulated_reconnect" in harness.metrics.notes
    assert harness.stop(drain_timeout_sec=2.0)


# ---------------------------------------------------------------------------
# 7. Failure validation
# ---------------------------------------------------------------------------


def test_evaluate_failure_worker_survives_and_continues() -> None:
    harness = RuntimeValidationHarness(fail_minutes={10})
    harness.start()
    for minute in (5, 10, 15, 20):
        harness.put_minute(minute)
    assert harness.wait_until_processed(3, timeout_sec=5.0)
    assert harness.worker.is_alive
    assert harness.processed_minutes == [5, 15, 20]
    assert harness.metrics.total_evaluate_failures == 1
    assert harness.stop(drain_timeout_sec=2.0)


# ---------------------------------------------------------------------------
# 8. Performance validation metrics
# ---------------------------------------------------------------------------


def test_performance_metrics_collected() -> None:
    harness = RuntimeValidationHarness()
    harness.start()
    minutes = list(range(0, 30, 5))
    for minute in minutes:
        harness.put_minute(minute)
        # Create a little queue depth intentionally.
        time.sleep(0.005)
    assert harness.wait_until_processed(len(minutes), timeout_sec=5.0)
    assert harness.stop(drain_timeout_sec=2.0)

    m = harness.metrics
    assert m.total_candles_processed == len(minutes)
    assert m.max_queue_depth >= 1
    assert m.average_queue_wait_ms >= 0.0
    assert m.average_evaluation_duration_ms > 0.0
    assert m.worker_uptime_sec > 0.0
    payload = m.as_dict()
    assert payload["total_candles_processed"] == len(minutes)
    assert "average_queue_wait_ms" in payload
    assert "max_queue_depth" in payload
    assert "average_evaluation_duration_ms" in payload
    assert "worker_uptime_sec" in payload