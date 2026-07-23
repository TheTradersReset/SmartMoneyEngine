"""Tests for Trade Validation Engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.sqlite import PaperSignalDatabase
from src.trade_validation import (
    CandleBar,
    SignalRecord,
    TradeValidationConfig,
    TradeValidationEngine,
    evaluate_signal,
)
from src.trade_validation.storage import TradeValidationDatabase


def _insert_signal(db: PaperSignalDatabase, *, direction: str = "BUY", entry: float = 25000.0) -> int:
    return db.insert_signal(
        {
            "timestamp": "2026-07-16 10:00:00+05:30",
            "direction": direction,
            "engine_version": "BUY_V3" if direction == "BUY" else "SELL_V6",
            "entry": entry,
            "stop": entry - 10 if direction == "BUY" else entry + 10,
            "target1": entry + 60 if direction == "BUY" else entry - 60,
            "target2": entry + 100 if direction == "BUY" else entry - 100,
            "target_structure": "60/100/Runner",
            "confidence": 1.0,
            "regime": "trending|low_vol|no_gap|unknown",
            "throttle_level": "FULL",
            "accepted": True,
            "rejection_reason": None,
            "evaluation": {"layer5": {"reason_codes": ["FORMULA_COMPLETE"], "formula_completion_pct": 85.0}},
        },
    )


def _insert_decision(db: PaperSignalDatabase) -> None:
    db.insert_signal_decision(
        {
            "timestamp": "2026-07-16T10:00:00+05:30",
            "symbol": "NSE:NIFTY50-INDEX",
            "open": 24995.0,
            "high": 25005.0,
            "low": 24990.0,
            "close": 25000.0,
            "volume": 1000.0,
            "trend": "Bullish",
            "market_regime": "trending|low_vol|no_gap|unknown",
            "buy_score": 85.0,
            "sell_score": 10.0,
            "final_signal": "BUY",
            "decision": "BUY",
            "reason_codes": ["FORMULA_COMPLETE"],
            "evaluation_time_ms": 5.0,
        },
    )


def _insert_candles(db: PaperSignalDatabase, bars: list[tuple[float, float, float, float]]) -> None:
    for index, (open_, high, low, close) in enumerate(bars, start=1):
        minute = 5 + index * 5
        db.insert_candle(
            symbol="NSE:NIFTY50-INDEX",
            timestamp=f"2026-07-16 10:{minute:02d}:00+05:30",
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=1000.0,
            tick_count=10,
        )


def test_buy_win_on_target_hit() -> None:
    signal = SignalRecord(
        id=1,
        timestamp="2026-07-16 10:00:00+05:30",
        direction="BUY",
        entry=25000.0,
        engine_version="BUY_V3",
        accepted=True,
        symbol="NSE:NIFTY50-INDEX",
        signal_score=85.0,
        reason_codes=("FORMULA_COMPLETE",),
        raw_payload={},
    )
    cfg = TradeValidationConfig(target_pct=0.24, stop_pct=0.04, evaluation_window_bars=5)
    candles = [
        CandleBar("2026-07-16 10:05:00+05:30", 25000.0, 25070.0, 24995.0, 25065.0),
    ]
    result = evaluate_signal(signal, candles, config=cfg)
    assert result.outcome == "WIN"
    assert result.exit_reason == "TARGET_HIT"
    assert result.target_hit is True
    assert result.pnl == pytest.approx(60.0, rel=1e-3)
    assert result.mfe == pytest.approx(70.0)
    assert result.next_candle_close == 25065.0


def test_buy_loss_on_stop_hit() -> None:
    signal = SignalRecord(
        id=2,
        timestamp="2026-07-16 10:00:00+05:30",
        direction="BUY",
        entry=25000.0,
        engine_version="BUY_V3",
        accepted=True,
        symbol="NSE:NIFTY50-INDEX",
        signal_score=85.0,
        reason_codes=("FORMULA_COMPLETE",),
        raw_payload={},
    )
    cfg = TradeValidationConfig(target_pct=0.24, stop_pct=0.04, evaluation_window_bars=5)
    candles = [
        CandleBar("2026-07-16 10:05:00+05:30", 25000.0, 25005.0, 24980.0, 24985.0),
    ]
    result = evaluate_signal(signal, candles, config=cfg)
    assert result.outcome == "LOSS"
    assert result.exit_reason == "STOP_HIT"
    assert result.stop_hit is True
    assert result.pnl == pytest.approx(-10.0, rel=1e-3)


def test_open_trade_insufficient_window() -> None:
    signal = SignalRecord(
        id=3,
        timestamp="2026-07-16 10:00:00+05:30",
        direction="BUY",
        entry=25000.0,
        engine_version="BUY_V3",
        accepted=True,
        symbol="NSE:NIFTY50-INDEX",
        signal_score=85.0,
        reason_codes=("FORMULA_COMPLETE",),
        raw_payload={},
    )
    cfg = TradeValidationConfig(evaluation_window_bars=20)
    candles = [
        CandleBar("2026-07-16 10:05:00+05:30", 25000.0, 25010.0, 24995.0, 25005.0),
        CandleBar("2026-07-16 10:10:00+05:30", 25005.0, 25015.0, 25000.0, 25010.0),
    ]
    result = evaluate_signal(signal, candles, config=cfg)
    assert result.outcome == "OPEN"
    assert result.exit_reason == "OPEN"
    assert result.next_3_candle_close is None
    assert result.next_5_candle_close is None
    assert result.pnl is None


def test_engine_end_to_end(tmp_path: Path) -> None:
    signal_db = tmp_path / "signals.db"
    validation_db = tmp_path / "validation.db"
    paper_db = PaperSignalDatabase(signal_db)
    _insert_decision(paper_db)
    signal_id = _insert_signal(paper_db)
    _insert_candles(
        paper_db,
        [
            (25000.0, 25070.0, 24995.0, 25065.0),
            (25065.0, 25075.0, 25060.0, 25070.0),
        ],
    )
    paper_db.close()

    cfg = TradeValidationConfig(
        signal_db_path=signal_db,
        validation_db_path=validation_db,
        evaluation_window_bars=5,
        target_pct=0.24,
        stop_pct=0.04,
    )
    engine = TradeValidationEngine(cfg)
    results = engine.run_once()
    engine.close()

    assert len(results) == 1
    assert results[0].outcome == "WIN"
    assert results[0].source_signal_id == signal_id

    store = TradeValidationDatabase(validation_db)
    rows = store.recent_validations(limit=1)
    store.close()
    assert rows[0]["direction"] == "BUY"
    assert rows[0]["signal_score"] == 85.0
    assert rows[0]["reason_codes"] == ["FORMULA_COMPLETE"]
