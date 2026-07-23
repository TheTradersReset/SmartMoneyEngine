"""Tests for Historical Replay Engine."""

from __future__ import annotations

import ast
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.data.candle_builder import Candle
from src.pipeline.realtime_signal_pipeline import RealtimeSignalPipeline
from src.replay.controller import ReplayController, ReplayState, parse_speed
from src.replay.data_feed import (
    HistoricalDataFeed,
    window_for_day,
    window_for_month,
    window_for_range,
    window_for_week,
)
from src.replay.engine import ReplayEngine
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase

IST = ZoneInfo("Asia/Kolkata")


def _write_sample_csv(path: Path, *, days: int = 3, bars_per_day: int = 5) -> None:
    rows: list[dict] = []
    base = date(2026, 3, 10)
    price = 25000.0
    for day_offset in range(days):
        day = date(base.year, base.month, base.day + day_offset)
        if day.weekday() >= 5:
            continue
        for bar in range(bars_per_day):
            minute = 15 + bar * 5
            ts = datetime(day.year, day.month, day.day, 9, minute, tzinfo=IST)
            open_ = price
            close = price + 1.0
            rows.append(
                {
                    "Date": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "Open": open_,
                    "High": close + 2.0,
                    "Low": open_ - 2.0,
                    "Close": close,
                    "Volume": 1000.0 + bar,
                },
            )
            price = close
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_speed_aliases() -> None:
    assert parse_speed("1x") == 1.0
    assert parse_speed("5x") == 5.0
    assert parse_speed("10x") == 10.0
    assert parse_speed("100x") == 100.0
    assert parse_speed("unlimited") == float("inf")
    with pytest.raises(ValueError):
        parse_speed("2x")


def test_replay_windows() -> None:
    assert window_for_day(date(2026, 3, 10)).start == date(2026, 3, 10)
    week = window_for_week(2026, 11)
    assert week.start == date(2026, 3, 9)
    assert week.end == date(2026, 3, 15)
    month = window_for_month(2026, 3)
    assert month.start == date(2026, 3, 1)
    assert month.end == date(2026, 3, 31)
    custom = window_for_range(date(2026, 3, 1), date(2026, 3, 5))
    assert custom.end == date(2026, 3, 5)


def test_historical_data_feed_filters(tmp_path: Path) -> None:
    csv_path = tmp_path / "hist.csv"
    _write_sample_csv(csv_path, days=3, bars_per_day=4)
    feed = HistoricalDataFeed(csv_path=csv_path, warm_start_bars=10)
    feed.load()
    window = window_for_day(date(2026, 3, 11))
    warm = feed.warm_start_frame(window)
    candles = feed.replay_candles(window)
    assert len(candles) == 4
    assert all(c.timestamp.date() == date(2026, 3, 11) for c in candles)
    assert len(warm) == 4
    assert "Date" in warm.columns


def test_controller_pause_resume_stop() -> None:
    ctrl = ReplayController(speed="unlimited")
    ctrl.start(total_candles=10)
    assert ctrl.state == ReplayState.RUNNING
    ctrl.pause()
    assert ctrl.state == ReplayState.PAUSED
    ctrl.resume()
    assert ctrl.state == ReplayState.RUNNING
    ctrl.stop()
    assert ctrl.state == ReplayState.STOPPED
    assert ctrl.should_stop() is True


def test_ingest_closed_candle_is_live_path(tmp_path: Path) -> None:
    """Public ingest_closed_candle must invoke the same handler as live closes."""
    db = PaperSignalDatabase(tmp_path / "sig.db")
    async_db = AsyncDbWriter(tmp_path / "sig.db")
    pipeline = RealtimeSignalPipeline(db=db, async_db=async_db, history_csv=None)
    seen: list[str] = []

    def _spy(candle: Candle) -> None:
        seen.append(candle.timestamp.isoformat())

    pipeline._handle_closed_candle = _spy  # type: ignore[method-assign]
    candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timestamp=datetime(2026, 3, 10, 10, 0, tzinfo=IST),
        open=25000.0,
        high=25010.0,
        low=24990.0,
        close=25005.0,
        volume=100.0,
        tick_count=1,
    )
    pipeline.ingest_closed_candle(candle)
    assert seen == [candle.timestamp.isoformat()]
    async_db.close()
    db.close()


def test_replay_engine_feeds_pipeline(tmp_path: Path) -> None:
    csv_path = tmp_path / "hist.csv"
    _write_sample_csv(csv_path, days=3, bars_per_day=5)
    signal_db = tmp_path / "replay_signals.db"
    validation_db = tmp_path / "replay_validation.db"

    engine = ReplayEngine(
        window=window_for_day(date(2026, 3, 11)),
        speed="unlimited",
        csv_path=csv_path,
        signal_db_path=signal_db,
        validation_db_path=validation_db,
        run_trade_validation=True,
    )
    try:
        result = engine.run()
    finally:
        engine.close()

    assert result.state == ReplayState.COMPLETED
    assert result.candles_fed == 5
    assert result.warm_start_bars == 5
    assert result.decisions >= 5


def test_replay_does_not_import_buy_sell_engines_directly() -> None:
    """Replay package must not call BUY_V3 / SELL_V6 — only the pipeline may."""
    replay_root = Path(__file__).resolve().parents[1] / "src" / "replay"
    for path in replay_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.endswith("buy_v3") or node.module.endswith("sell_v6"):
                    pytest.fail(f"{path.name} imports signal engines directly: {node.module}")
            if isinstance(node, ast.Name) and node.id in ("BuyV3Engine", "SellV6Engine"):
                pytest.fail(f"{path.name} references {node.id}")
