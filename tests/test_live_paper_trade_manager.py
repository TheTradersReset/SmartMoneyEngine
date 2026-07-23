"""Unit tests for PaperTradeManager."""

from __future__ import annotations

from pathlib import Path

from src.paper_trading.trade_manager import PaperTradeManager
from src.storage.sqlite import PaperSignalDatabase


def _db(tmp_path: Path) -> PaperSignalDatabase:
    return PaperSignalDatabase(tmp_path / "paper_test.db")


def test_dedupe_and_on_signal(tmp_path: Path) -> None:
    db = _db(tmp_path)
    sid = db.insert_signal(
        {
            "timestamp": "2026-07-21 10:00:00+05:30",
            "direction": "BUY",
            "engine_version": "BUY_V3",
            "entry": 25000.0,
            "stop": 24990.0,
            "target1": 25060.0,
            "target2": 25100.0,
            "target_structure": "60/100/Runner",
            "confidence": 1.0,
            "regime": "trending",
            "throttle_level": "FULL",
            "accepted": True,
            "rejection_reason": None,
            "risk": 10.0,
            "outcome": "PENDING",
        }
    )
    mgr = PaperTradeManager(db)
    trade = mgr.on_signal(
        {
            "timestamp": "2026-07-21 10:00:00+05:30",
            "direction": "BUY",
            "engine_version": "BUY_V3",
            "entry": 25000.0,
            "stop": 24990.0,
            "target1": 25060.0,
            "target2": 25100.0,
            "risk": 10.0,
            "accepted": True,
        }
    )
    assert trade is not None
    assert trade.signal_id == sid
    assert len(mgr.list_open()) == 1
    # dedupe
    again = mgr.on_signal(
        {
            "timestamp": "2026-07-21 10:00:00+05:30",
            "direction": "BUY",
            "engine_version": "BUY_V3",
            "entry": 25000.0,
            "stop": 24990.0,
            "target1": 25060.0,
            "target2": 25100.0,
            "accepted": True,
        }
    )
    assert again is trade
    assert len(mgr.list_open()) == 1


def test_on_outcome_and_stats(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.insert_signal(
        {
            "timestamp": "2026-07-21 10:00:00+05:30",
            "direction": "BUY",
            "engine_version": "BUY_V3",
            "entry": 25000.0,
            "stop": 24990.0,
            "target1": 25060.0,
            "target2": 25100.0,
            "target_structure": "60/100/Runner",
            "confidence": 1.0,
            "regime": "trending",
            "throttle_level": "FULL",
            "accepted": True,
            "rejection_reason": None,
            "risk": 10.0,
            "reward": 60.0,
            "outcome": "WIN",
            "holding_bars": 10,
            "outcome_timestamp": "2026-07-21 11:00:00+05:30",
        }
    )
    mgr = PaperTradeManager(db)
    mgr.on_signal(
        {
            "timestamp": "2026-07-21 10:00:00+05:30",
            "direction": "BUY",
            "engine_version": "BUY_V3",
            "entry": 25000.0,
            "stop": 24990.0,
            "target1": 25060.0,
            "target2": 25100.0,
            "accepted": True,
        }
    )
    closed = mgr.on_outcome(
        timestamp="2026-07-21 10:00:00+05:30",
        direction="BUY",
        outcome="WIN",
        reward=60.0,
        holding_bars=10,
        outcome_timestamp="2026-07-21 11:00:00+05:30",
    )
    assert closed is not None
    assert closed.status == "CLOSED"
    assert len(mgr.list_open()) == 0
    stats = mgr.stats()
    assert stats["running_pnl"] == 60.0
    assert stats["win_rate"] == 1.0
    assert stats["equity_curve"]
