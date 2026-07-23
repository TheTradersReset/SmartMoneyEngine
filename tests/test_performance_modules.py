"""Tests for production performance modules."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.incremental_indicators import IncrementalIndicatorEngine
from src.pipeline.market_memory_cache import MarketMemoryCache
from src.research.filter_research_engine import FilterContextBuilder
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase


def _sample_frame(n: int = 30) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-07-16 09:15:00", tz="Asia/Kolkata")
    for i in range(n):
        ts = base + pd.Timedelta(minutes=5 * i)
        rows.append(
            {
                "Date": ts.isoformat(),
                "Open": 25000.0 + i,
                "High": 25005.0 + i,
                "Low": 24995.0 + i,
                "Close": 25002.0 + i,
                "Volume": 1000.0 + i,
            }
        )
    return pd.DataFrame(rows)


def test_market_memory_cache_append_and_dataframe() -> None:
    cache = MarketMemoryCache(max_bars=10)
    row = {"Date": "2026-07-16 09:15:00+0530", "Open": 1, "High": 2, "Low": 0.5, "Close": 1.5, "Volume": 10}
    index = cache.append_closed_row(row)
    assert index == 0
    frame = cache.as_dataframe()
    assert len(frame) == 1
    assert frame.iloc[0]["Close"] == 1.5


def test_incremental_indicators_match_full_enrich_tail() -> None:
    frame = _sample_frame(40)
    builder = FilterContextBuilder()
    full = builder.enrich(frame)

    engine = IncrementalIndicatorEngine()
    engine.seed_from_frame(frame.iloc[:25].reset_index(drop=True))
    incremental = engine.append_bar(frame.reset_index(drop=True), 25)

    for col in ("_ema_20", "_ema_50", "_ema_200", "_atr", "_vwap"):
        full_value = full.iloc[25][col]
        inc_value = incremental.iloc[25][col]
        if pd.isna(full_value):
            continue
        assert float(inc_value) == pytest.approx(float(full_value), rel=1e-4, abs=1e-4)

    assert len(incremental) == 26


def test_async_db_writer_persists_decision(tmp_path: Path) -> None:
    db_path = tmp_path / "async.db"
    writer = AsyncDbWriter(db_path)
    writer.enqueue(
        "signal_decision",
        {
            "timestamp": "2026-07-16T10:00:00+05:30",
            "symbol": "NSE:NIFTY50-INDEX",
            "open": 25000.0,
            "high": 25010.0,
            "low": 24990.0,
            "close": 25005.0,
            "volume": 1000.0,
            "trend": "Neutral",
            "market_regime": "range|low_vol|no_gap|mid_range",
            "buy_score": 0.0,
            "sell_score": 0.0,
            "final_signal": "NO_TRADE",
            "decision": "NO_TRADE",
            "reason_codes": ["NO_SIGNAL"],
            "evaluation_time_ms": 1.0,
        },
    )
    writer.close()

    db = PaperSignalDatabase(db_path)
    rows = db.recent_decisions(limit=1)
    db.close()
    assert rows[0]["decision"] == "NO_TRADE"
