"""Tests for realtime signal pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.signals.regime_throttle import RegimeThrottle
from src.storage.sqlite import PaperSignalDatabase


def test_regime_throttle_block_rejects() -> None:
    composite = "range|unknown_vol|no_gap|mid_range"
    throttle = RegimeThrottle(
        buy_throttle_map={composite: "BLOCK"},
        sell_throttle_map={},
    )
    evaluation = {"layer2": {"htf_trend": "Neutral"}}
    decision = throttle.apply(direction="BUY", evaluation=evaluation)
    assert decision.composite_regime == composite
    assert decision.accepted is False
    assert decision.throttle_level == "BLOCK"


def test_sqlite_insert_signal(tmp_path: Path) -> None:
    db = PaperSignalDatabase(tmp_path / "test.db")
    signal_id = db.insert_signal(
        {
            "timestamp": "2026-07-16 10:00:00+05:30",
            "direction": "BUY",
            "engine_version": "BUY_V3",
            "entry": 25000.0,
            "stop": 24990.0,
            "target1": 25060.0,
            "target2": 25100.0,
            "target_structure": "60/100/Runner",
            "confidence": 1.0,
            "regime": "trending|low_vol|no_gap|unknown",
            "throttle_level": "FULL",
            "accepted": True,
            "rejection_reason": None,
        }
    )
    assert signal_id == 1
    rows = db.recent_signals(limit=1)
    assert rows[0]["direction"] == "BUY"
    db.close()


def test_sqlite_insert_signal_decision(tmp_path: Path) -> None:
    db = PaperSignalDatabase(tmp_path / "test.db")
    decision_id = db.insert_signal_decision(
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
            "buy_score": 40.0,
            "sell_score": 14.3,
            "final_signal": "NO_TRADE",
            "decision": "NO_TRADE",
            "reason_codes": ["FORMULA_INCOMPLETE", "NO_SIGNAL", "VWAP_MISMATCH"],
            "evaluation_time_ms": 12.34,
        }
    )
    assert decision_id == 1
    rows = db.recent_decisions(symbol="NSE:NIFTY50-INDEX", limit=1)
    assert rows[0]["decision"] == "NO_TRADE"
    assert rows[0]["reason_codes"] == ["FORMULA_INCOMPLETE", "NO_SIGNAL", "VWAP_MISMATCH"]
    db.close()