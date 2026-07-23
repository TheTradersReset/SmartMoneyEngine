"""Tests for real-time candle builder."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.data.candle_builder import (
    FiveMinuteCandleBuilder,
    Tick,
    parse_fyers_tick,
)

IST = ZoneInfo("Asia/Kolkata")


def test_parse_fyers_tick_dict() -> None:
    tick = parse_fyers_tick({"symbol": "NSE:NIFTY50-INDEX", "ltp": 25001.5})
    assert tick is not None
    assert tick.price == 25001.5
    assert tick.symbol == "NSE:NIFTY50-INDEX"


def test_candle_builder_closes_bucket_on_roll() -> None:
    closed: list = []
    builder = FiveMinuteCandleBuilder(on_candle_close=lambda candle: closed.append(candle))
    t1 = datetime(2026, 7, 16, 9, 15, 10, tzinfo=IST)
    t2 = datetime(2026, 7, 16, 9, 19, 50, tzinfo=IST)
    t3 = datetime(2026, 7, 16, 9, 20, 5, tzinfo=IST)

    assert builder.ingest_tick(Tick("NSE:NIFTY50-INDEX", 100.0, t1)) is None
    assert builder.ingest_tick(Tick("NSE:NIFTY50-INDEX", 102.0, t2)) is None
    candle = builder.ingest_tick(Tick("NSE:NIFTY50-INDEX", 101.0, t3))
    assert candle is not None
    assert candle.open == 100.0
    assert candle.close == 102.0
    assert len(closed) == 1
