"""PR-2: WatermarkStore wired into closed-candle apply path."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.candle_builder import Candle
from src.live_paper.metrics import LiveMetrics
from src.live_paper.recovery import MissedCandleRecovery
from src.pipeline.realtime_signal_pipeline import RealtimeSignalPipeline
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase

IST = ZoneInfo("Asia/Kolkata")
SYMBOL = "NSE:NIFTY50-INDEX"


def _candle(ts: datetime, *, close: float = 100.0) -> Candle:
    return Candle(
        symbol=SYMBOL,
        timestamp=ts,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=10.0,
        tick_count=2,
    )


def _pipeline(tmp_path: Path) -> RealtimeSignalPipeline:
    db = PaperSignalDatabase(tmp_path / "wm.db")
    async_db = AsyncDbWriter(db.db_path)
    return RealtimeSignalPipeline(db=db, async_db=async_db, history_csv=None)


def _seed_watermark(pipeline: RealtimeSignalPipeline, ts: datetime) -> None:
    now = datetime(2026, 7, 22, 15, 0, tzinfo=IST)
    assert pipeline._watermark.try_advance(ts, now=now) is True


def test_duplicate_timestamp_is_rejected(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    ts = datetime(2026, 7, 22, 10, 15, tzinfo=IST)
    _seed_watermark(pipeline, ts)
    bars_before = pipeline.context.memory.bar_count

    pipeline.ingest_closed_candle(_candle(ts, close=111.0))

    assert pipeline.context.memory.bar_count == bars_before
    assert pipeline._watermark.get() == ts
    pipeline.async_db.close()
    pipeline.db.close()


def test_newer_timestamp_is_accepted(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    older = datetime(2026, 7, 22, 10, 10, tzinfo=IST)
    newer = datetime(2026, 7, 22, 10, 15, tzinfo=IST)
    _seed_watermark(pipeline, older)
    bars_before = pipeline.context.memory.bar_count

    pipeline.ingest_closed_candle(_candle(newer, close=120.0))

    assert pipeline.context.memory.bar_count == bars_before + 1
    assert pipeline._watermark.get() == newer
    latest = pipeline.context.memory.latest_row()
    assert latest is not None
    pipeline.async_db.close()
    pipeline.db.close()


def test_watermark_survives_normal_runtime_flow(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    t1 = datetime(2026, 7, 22, 10, 5, tzinfo=IST)
    t2 = datetime(2026, 7, 22, 10, 10, tzinfo=IST)
    t3 = datetime(2026, 7, 22, 10, 15, tzinfo=IST)

    pipeline.ingest_closed_candle(_candle(t1))
    pipeline.ingest_closed_candle(_candle(t2))
    pipeline.ingest_closed_candle(_candle(t3))

    assert pipeline._watermark.get() == t3
    assert pipeline.context.memory.bar_count == 3

    # Duplicate of t2 must not move watermark backward or append.
    pipeline.ingest_closed_candle(_candle(t2, close=99.0))
    assert pipeline._watermark.get() == t3
    assert pipeline.context.memory.bar_count == 3

    pipeline.async_db.close()
    pipeline.db.close()


def test_warm_start_seeds_watermark(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    last = datetime(2026, 7, 22, 10, 0, tzinfo=IST)
    frame = pd.DataFrame(
        [
            {
                "Date": last.isoformat(),
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "Volume": 1.0,
            }
        ]
    )
    assert pipeline.warm_start_from_frame(frame) == 1
    assert pipeline._watermark.get() == last

    # Same timestamp rejected after warm-start seed.
    bars = pipeline.context.memory.bar_count
    pipeline.ingest_closed_candle(_candle(last, close=105.0))
    assert pipeline.context.memory.bar_count == bars

    pipeline.async_db.close()
    pipeline.db.close()


def test_recovery_still_functions(tmp_path: Path) -> None:
    """Recovery ingest still applies newer bars; watermark advances; duplicate blocked."""
    pipeline = _pipeline(tmp_path)
    last = datetime(2026, 7, 22, 10, 0, tzinfo=IST)
    now = datetime(2026, 7, 22, 10, 20, tzinfo=IST)
    bar_10_05 = datetime(2026, 7, 22, 10, 5, tzinfo=IST)
    bar_10_10 = datetime(2026, 7, 22, 10, 10, tzinfo=IST)

    _seed_watermark(pipeline, last)

    class _Client:
        def get_history(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "candles": [
                    [bar_10_05.timestamp(), 100.0, 101.0, 99.0, 100.5, 10.0],
                    [bar_10_10.timestamp(), 101.0, 102.0, 100.0, 101.5, 11.0],
                ]
            }

    recovery = MissedCandleRecovery(
        pipeline,
        LiveMetrics(),
        client_factory=_Client,
    )
    fed = recovery.maybe_recover(last_candle_ts=last, now=now)
    assert fed == 2
    assert pipeline._watermark.get() == bar_10_10
    assert pipeline.context.memory.bar_count == 2

    # Re-applying a recovered timestamp via ingest must be rejected by watermark.
    bars = pipeline.context.memory.bar_count
    pipeline.ingest_closed_candle(_candle(bar_10_10, close=999.0))
    assert pipeline.context.memory.bar_count == bars
    assert pipeline._watermark.get() == bar_10_10

    pipeline.async_db.close()
    pipeline.db.close()
