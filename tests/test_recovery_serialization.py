"""PR-1: MissedCandleRecovery single-flight serialization."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.data.candle_builder import Candle
from src.live_paper.metrics import LiveMetrics
from src.live_paper.recovery import MissedCandleRecovery
from src.storage.sqlite import PaperSignalDatabase

IST = ZoneInfo("Asia/Kolkata")
SYMBOL = "NSE:NIFTY50-INDEX"

# Weekday session window with more than one missed 5m bar between last and now.
_LAST = datetime(2026, 7, 22, 10, 0, tzinfo=IST)
_NOW = datetime(2026, 7, 22, 10, 20, tzinfo=IST)
_BAR_10_05 = datetime(2026, 7, 22, 10, 5, tzinfo=IST)
_BAR_10_10 = datetime(2026, 7, 22, 10, 10, tzinfo=IST)


class _FakePipeline:
    def __init__(self, db: PaperSignalDatabase) -> None:
        self.symbol = SYMBOL
        self.db = db
        self.ingested: list[Candle] = []

    def ingest_closed_candle(self, candle: Candle) -> None:
        self.ingested.append(candle)


def _history_candles(*timestamps: datetime) -> dict[str, Any]:
    rows = []
    for i, ts in enumerate(timestamps):
        rows.append(
            [
                ts.timestamp(),
                100.0 + i,
                101.0 + i,
                99.0 + i,
                100.5 + i,
                10.0 + i,
            ]
        )
    return {"candles": rows}


def _recovery(
    tmp_path: Path,
    *,
    client_factory: Any,
) -> tuple[MissedCandleRecovery, _FakePipeline]:
    db = PaperSignalDatabase(tmp_path / "recovery.db")
    pipeline = _FakePipeline(db)
    recovery = MissedCandleRecovery(
        pipeline,
        LiveMetrics(),
        client_factory=client_factory,
    )
    return recovery, pipeline


def test_normal_recovery_still_works(tmp_path: Path) -> None:
    """Stale/heartbeat-style maybe_recover feeds missed bars when gap exists."""
    calls = {"n": 0}

    class _Client:
        def get_history(self, **_kwargs: Any) -> dict[str, Any]:
            calls["n"] += 1
            return _history_candles(_BAR_10_05, _BAR_10_10)

    recovery, pipeline = _recovery(tmp_path, client_factory=_Client)
    fed = recovery.maybe_recover(last_candle_ts=_LAST, now=_NOW)

    assert calls["n"] == 1
    assert fed == 2
    assert [c.timestamp for c in pipeline.ingested] == [_BAR_10_05, _BAR_10_10]


def test_reconnect_recovery_still_works(tmp_path: Path) -> None:
    """Reconnect-style maybe_recover (same entry point) still ingests history."""

    class _Client:
        def get_history(self, **_kwargs: Any) -> dict[str, Any]:
            return _history_candles(_BAR_10_05)

    recovery, pipeline = _recovery(tmp_path, client_factory=_Client)
    # Same call site used by pipeline_ext._trigger_recovery("reconnect")
    fed = recovery.maybe_recover(last_candle_ts=_LAST, now=_NOW)

    assert fed == 1
    assert len(pipeline.ingested) == 1
    assert pipeline.ingested[0].timestamp == _BAR_10_05


def test_heartbeat_and_reconnect_cannot_run_simultaneously(tmp_path: Path) -> None:
    """Second trigger while recovery holds the lock is skipped."""
    entered = threading.Event()
    release = threading.Event()
    fetch_count = {"n": 0}
    fetch_lock = threading.Lock()

    class _BlockingClient:
        def get_history(self, **_kwargs: Any) -> dict[str, Any]:
            with fetch_lock:
                fetch_count["n"] += 1
            entered.set()
            assert release.wait(timeout=5.0), "first recovery was not released"
            return _history_candles(_BAR_10_05, _BAR_10_10)

    recovery, pipeline = _recovery(tmp_path, client_factory=_BlockingClient)

    results: dict[str, int] = {}

    def _heartbeat() -> None:
        # Mimics health.on_stale -> _trigger_recovery("stale_ticks")
        results["heartbeat"] = recovery.maybe_recover(last_candle_ts=_LAST, now=_NOW)

    def _reconnect() -> None:
        assert entered.wait(timeout=5.0), "first recovery never entered get_history"
        # Mimics WS reconnect -> _trigger_recovery("reconnect") while first still running
        results["reconnect"] = recovery.maybe_recover(last_candle_ts=_LAST, now=_NOW)
        release.set()

    t1 = threading.Thread(target=_heartbeat, name="stale-recovery")
    t2 = threading.Thread(target=_reconnect, name="reconnect-recovery")
    t1.start()
    assert entered.wait(timeout=5.0), "heartbeat recovery did not start"
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert not t1.is_alive() and not t2.is_alive()

    assert results["heartbeat"] == 2
    assert results["reconnect"] == 0
    assert fetch_count["n"] == 1
    assert len(pipeline.ingested) == 2


def test_recovery_allowed_again_after_prior_completes(tmp_path: Path) -> None:
    """After the lock is released, a later trigger can recover normally."""
    fetch_count = {"n": 0}

    class _Client:
        def get_history(self, **_kwargs: Any) -> dict[str, Any]:
            fetch_count["n"] += 1
            return _history_candles(_BAR_10_05)

    recovery, pipeline = _recovery(tmp_path, client_factory=_Client)

    first = recovery.maybe_recover(last_candle_ts=_LAST, now=_NOW)
    second = recovery.maybe_recover(last_candle_ts=_LAST, now=_NOW)

    assert first == 1
    # Fake pipeline does not persist candles; second run is not blocked by the lock.
    assert second == 1
    assert fetch_count["n"] == 2
    assert len(pipeline.ingested) == 2
