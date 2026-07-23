"""PR-3: FiveMinuteCandleBuilder sync after committed watermark / recovery."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.data.candle_builder import FiveMinuteCandleBuilder, Tick
from src.live_paper.config import LivePaperConfig
from src.live_paper.health import ConnectionHealthMonitor
from src.live_paper.metrics import LiveMetrics
from src.live_paper.pipeline_ext import LivePaperPipeline
from src.notifications.email import EmailNotifier
from src.paper_trading.trade_manager import PaperTradeManager
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase

IST = ZoneInfo("Asia/Kolkata")
SYMBOL = "NSE:NIFTY50-INDEX"


def _tick(ts: datetime, price: float = 100.0) -> Tick:
    return Tick(SYMBOL, price, ts)


def test_normal_live_session_unchanged() -> None:
    closed: list = []
    builder = FiveMinuteCandleBuilder(on_candle_close=lambda c: closed.append(c))
    t1 = datetime(2026, 7, 22, 10, 15, 10, tzinfo=IST)
    t2 = datetime(2026, 7, 22, 10, 19, 50, tzinfo=IST)
    t3 = datetime(2026, 7, 22, 10, 20, 5, tzinfo=IST)

    assert builder.ingest_tick(_tick(t1, 100.0)) is None
    assert builder.ingest_tick(_tick(t2, 102.0)) is None
    candle = builder.ingest_tick(_tick(t3, 101.0))
    assert candle is not None
    assert candle.timestamp == datetime(2026, 7, 22, 10, 15, tzinfo=IST)
    assert len(closed) == 1


def test_market_open_first_buckets() -> None:
    closed: list = []
    builder = FiveMinuteCandleBuilder(on_candle_close=lambda c: closed.append(c))
    open_tick = datetime(2026, 7, 22, 9, 15, 1, tzinfo=IST)
    next_bucket = datetime(2026, 7, 22, 9, 20, 0, tzinfo=IST)

    assert builder.ingest_tick(_tick(open_tick, 100.0)) is None
    assert builder._active_bucket == datetime(2026, 7, 22, 9, 15, tzinfo=IST)
    candle = builder.ingest_tick(_tick(next_bucket, 101.0))
    assert candle is not None
    assert candle.timestamp == datetime(2026, 7, 22, 9, 15, tzinfo=IST)
    assert len(closed) == 1


def test_reconnect_during_five_minute_candle_discards_active() -> None:
    """Mid-bucket reconnect/recovery: active obsolete bucket is discarded, not emitted."""
    closed: list = []
    builder = FiveMinuteCandleBuilder(on_candle_close=lambda c: closed.append(c))
    mid = datetime(2026, 7, 22, 10, 7, 30, tzinfo=IST)
    committed = datetime(2026, 7, 22, 10, 5, tzinfo=IST)

    assert builder.ingest_tick(_tick(mid, 100.0)) is None
    assert builder._active_bucket == committed

    builder.sync_after_committed(committed)

    assert builder._active_bucket is None
    assert closed == []

    # Same-bucket ticks after sync must not restart a committed bucket.
    assert builder.ingest_tick(_tick(mid, 101.0)) is None
    assert builder._active_bucket is None
    assert closed == []


def test_recovery_followed_by_live_ticks_emits_only_newer() -> None:
    closed: list = []
    builder = FiveMinuteCandleBuilder(on_candle_close=lambda c: closed.append(c))
    committed = datetime(2026, 7, 22, 10, 10, tzinfo=IST)

    # Simulate forming the recovered bar when disconnect happened.
    builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 12, tzinfo=IST), 100.0))
    builder.sync_after_committed(committed)
    assert builder._active_bucket is None

    # Live ticks in the next bar after recovery.
    assert builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 16, tzinfo=IST), 110.0)) is None
    assert builder._active_bucket == datetime(2026, 7, 22, 10, 15, tzinfo=IST)

    candle = builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 20, tzinfo=IST), 111.0))
    assert candle is not None
    assert candle.timestamp == datetime(2026, 7, 22, 10, 15, tzinfo=IST)
    assert [c.timestamp for c in closed] == [datetime(2026, 7, 22, 10, 15, tzinfo=IST)]


def test_multiple_reconnects_resync_each_time() -> None:
    closed: list = []
    builder = FiveMinuteCandleBuilder(on_candle_close=lambda c: closed.append(c))

    builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 3, tzinfo=IST), 100.0))
    builder.sync_after_committed(datetime(2026, 7, 22, 10, 0, tzinfo=IST))
    # Active 10:00 bucket discarded if committed through 10:00.
    # Mid 10:03 floors to 10:00; sync to 10:00 clears it.
    assert builder._active_bucket is None

    builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 8, tzinfo=IST), 101.0))
    assert builder._active_bucket == datetime(2026, 7, 22, 10, 5, tzinfo=IST)
    builder.sync_after_committed(datetime(2026, 7, 22, 10, 5, tzinfo=IST))
    assert builder._active_bucket is None

    builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 12, tzinfo=IST), 102.0))
    builder.sync_after_committed(datetime(2026, 7, 22, 10, 10, tzinfo=IST))
    assert builder._active_bucket is None
    assert closed == []

    candle = builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 15, tzinfo=IST), 103.0))
    assert candle is None
    assert builder._active_bucket == datetime(2026, 7, 22, 10, 15, tzinfo=IST)


def test_suppress_emit_when_closing_already_committed_bucket() -> None:
    """Emit guard: closing a committed bucket must not invoke on_candle_close."""
    closed: list = []
    builder = FiveMinuteCandleBuilder(on_candle_close=lambda c: closed.append(c))
    builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 6, tzinfo=IST), 100.0))
    # Simulate committed watermark without going through sync reset path.
    builder._committed_through = datetime(2026, 7, 22, 10, 5, tzinfo=IST)

    result = builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 10, tzinfo=IST), 101.0))
    assert result is None
    assert closed == []
    # Newer bucket may start after suppress.
    assert builder._active_bucket == datetime(2026, 7, 22, 10, 10, tzinfo=IST)


def test_pipeline_recovery_syncs_builder(tmp_path: Path) -> None:
    """LivePaperPipeline._trigger_recovery syncs builder to watermark after recovery."""
    db = PaperSignalDatabase(tmp_path / "sync.db")
    async_db = AsyncDbWriter(db.db_path)
    metrics = LiveMetrics()
    health = ConnectionHealthMonitor(metrics, stale_tick_seconds=30.0, heartbeat_seconds=60.0)
    config = LivePaperConfig(
        symbol=SYMBOL,
        history_csv=None,
        enable_pipeline_v2=False,
    )
    pipeline = LivePaperPipeline(
        config=config,
        metrics=metrics,
        health=health,
        email=EmailNotifier(enabled=False),
        trade_manager=PaperTradeManager(db),
        db=db,
        async_db=async_db,
        history_csv=None,
    )

    last = datetime(2026, 7, 22, 10, 0, tzinfo=IST)
    bar_10_05 = datetime(2026, 7, 22, 10, 5, tzinfo=IST)
    bar_10_10 = datetime(2026, 7, 22, 10, 10, tzinfo=IST)
    now = datetime(2026, 7, 22, 10, 20, tzinfo=IST)
    metrics.last_candle_ts = last.isoformat()

    # Forming mid-candle for a bar that recovery will commit.
    pipeline._candle_builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 7, tzinfo=IST), 100.0))
    assert pipeline._candle_builder._active_bucket == bar_10_05

    class _Client:
        def get_history(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "candles": [
                    [bar_10_05.timestamp(), 100.0, 101.0, 99.0, 100.5, 10.0],
                    [bar_10_10.timestamp(), 101.0, 102.0, 100.0, 101.5, 11.0],
                ]
            }

    pipeline.recovery._client_factory = _Client

    # Force now for gap detection via maybe_recover; _trigger_recovery uses metrics last.
    # Patch maybe_recover to pass now by wrapping.
    original = pipeline.recovery.maybe_recover

    def _recover_with_now(**kwargs: Any) -> int:
        kwargs.setdefault("now", now)
        return original(**kwargs)

    pipeline.recovery.maybe_recover = _recover_with_now  # type: ignore[method-assign]

    emitted: list = []
    pipeline._candle_builder.on_candle_close = lambda c: emitted.append(c)

    pipeline._trigger_recovery("reconnect")

    assert pipeline._watermark.get() == bar_10_10
    assert pipeline._candle_builder._committed_through == bar_10_10
    assert pipeline._candle_builder._active_bucket is None

    # Live ticks must not re-emit recovered timestamps.
    pipeline._candle_builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 12, tzinfo=IST), 200.0))
    assert pipeline._candle_builder._active_bucket is None
    pipeline._candle_builder.ingest_tick(_tick(datetime(2026, 7, 22, 10, 16, tzinfo=IST), 201.0))
    assert pipeline._candle_builder._active_bucket == datetime(2026, 7, 22, 10, 15, tzinfo=IST)
    assert emitted == []

    pipeline.async_db.close()
    pipeline.db.close()
