"""Unit tests: signal decision vs forward outcome separation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.liquidity_move_reconstruction_research import FORWARD_BARS
from src.research.smartmoneyengine_v3_implementation_validation_research import (
    SmartMoneyEngineV3Engine,
)
from src.signals.sell_v6 import SellV6Engine
from src.signals.signal_outcome import (
    classify_trade_outcome,
    evaluate_post_signal_outcome,
    build_realtime_layer4_plan,
)
from src.storage.sqlite import PaperSignalDatabase


def _ohlc_frame(n: int = 10, *, start: float = 100.0) -> pd.DataFrame:
    rows = []
    price = start
    for i in range(n):
        rows.append(
            {
                "Date": f"2026-03-05 09:{i:02d}:00",
                "Open": price,
                "High": price + 5,
                "Low": price - 5,
                "Close": price + 1,
                "Volume": 1000,
            }
        )
        price += 1
    return pd.DataFrame(rows)


def test_classify_trade_outcome() -> None:
    assert classify_trade_outcome({"realized_pnl_points": 10}) == "WIN"
    assert classify_trade_outcome({"realized_pnl_points": -3}) == "LOSS"
    assert classify_trade_outcome({"realized_pnl_points": 0}) == "BREAKEVEN"
    assert classify_trade_outcome({}) == "INCOMPLETE"


def test_layer4_sell_does_not_require_forward_bars() -> None:
    engine = SmartMoneyEngineV3Engine()
    frame = _ohlc_frame(5)
    bar = len(frame) - 1  # latest bar — no forward candles
    assert engine._trade_outcome(frame, bar, "bearish") == {}

    layer4 = engine._layer4_execution(
        frame,
        bar,
        layer1={"events_detected": ["Failed Breakout"], "failed_breakout_present": True},
        layer2={"htf_trend": "Bearish", "vwap_state": "Below", "ema_structure": "Bear Context"},
        layer3={"confirmation_candle": "Marubozu", "volume_bucket": "Normal"},
        context={"location": "Near Resistance"},
    )
    assert layer4 is not None
    assert layer4["direction"] == "SELL"
    assert layer4["entry"] > 0
    assert layer4["risk_points"] > 0
    assert layer4["forward_outcome"] is None
    assert layer4["outcome_pending"] is True


def test_evaluate_bar_sets_sell_without_forward() -> None:
    """Layer5 pass + realtime Layer4 ⇒ verdict SELL even with empty forward window."""
    engine = SellV6Engine()
    frame = _ohlc_frame(5)
    bar = len(frame) - 1

    # Force Layer5 pass by patching layer builders on the candidate engine.
    cand = engine._engine

    def _l1(_events):
        return {
            "active": True,
            "events_detected": ["Failed Breakout"],
            "primary_event": "Failed Breakout",
            "failed_breakout_present": True,
        }

    def _l2(_context):
        return {
            "direction": "SELL",
            "htf_trend": "Bearish",
            "vwap_state": "Below",
            "vwap_gate_rule": "VWAP Below only",
            "vwap_gate_passes": True,
            "ema_structure": "Bear Context",
            "v4_ema_rule": "x",
            "v4_ema_bearish": True,
            "aligned": True,
        }

    def _l3(_context):
        return {
            "confirmation_candle": "Marubozu",
            "volume_bucket": "Normal",
            "confirmed": True,
        }

    cand._detect_events_at_bar = lambda *_a, **_k: ("Failed Breakout",)  # type: ignore[method-assign]
    cand._context_at_bar = lambda **_k: {  # type: ignore[method-assign]
        "htf_trend": "Bearish",
        "vwap": "Below",
        "v4_ema_bearish": "True",
        "v4_ema_structure": "Bear Context",
        "confirmation_candle": "Marubozu",
        "volume": "Normal",
        "location": "Near Resistance",
    }
    cand._layer1_early_warning = _l1  # type: ignore[method-assign]
    cand._layer2_directional_filter = _l2  # type: ignore[method-assign]
    cand._layer3_confirmation = _l3  # type: ignore[method-assign]

    evaluation = engine.evaluate_bar(
        frame=frame,
        enriched=frame,
        calendar=frame,
        intel_frames={},
        bar=bar,
    )
    assert evaluation["layer5"]["pass"] is True
    assert evaluation["verdict"] == "SELL"
    assert evaluation.get("layer4") is not None
    signal = engine.to_signal(evaluation)
    assert signal is not None
    assert signal.direction == "SELL"


def test_post_signal_outcome_after_forward_bars() -> None:
    engine = SmartMoneyEngineV3Engine()
    # Signal at bar 0; need FORWARD_BARS+1 rows so bar 0 has forward data when evaluated later
    frame = _ohlc_frame(FORWARD_BARS + 5, start=20000.0)
    signal_bar = 0
    # Decline after entry → LOSS for bearish? bearish wins when price falls
    # Our synthetic frame drifts up → bearish realized < 0 → LOSS
    update = evaluate_post_signal_outcome(
        engine,
        frame=frame,
        signal_bar=signal_bar,
        direction="SELL",
        decision_timestamp=str(frame.iloc[signal_bar]["Date"]),
        outcome_timestamp=str(frame.iloc[signal_bar + FORWARD_BARS]["Date"]),
        forward_bars=FORWARD_BARS,
    )
    assert update is not None
    assert update.outcome in {"WIN", "LOSS", "BREAKEVEN"}
    assert update.holding_bars == FORWARD_BARS
    assert update.decision_timestamp == str(frame.iloc[signal_bar]["Date"])
    assert update.risk > 0


def test_normalize_timestamp_key_equates_offset_formats() -> None:
    from src.signals.signal_outcome import normalize_timestamp_key, timestamps_equivalent

    a = "2026-03-05 09:35:00+0530"
    b = "2026-03-05 09:35:00+05:30"
    assert normalize_timestamp_key(a) is not None
    assert normalize_timestamp_key(a) == normalize_timestamp_key(b)
    assert timestamps_equivalent(a, b) is True
    assert timestamps_equivalent(a, "2026-03-05 09:40:00+05:30") is False


def test_sqlite_update_signal_outcome_preserves_decision_timestamp(tmp_path: Path) -> None:
    db = PaperSignalDatabase(tmp_path / "outcome.db")
    decision_ts = "2026-03-05 10:25:00+05:30"
    db.insert_signal(
        {
            "timestamp": decision_ts,
            "direction": "SELL",
            "engine_version": "SELL_V6",
            "entry": 24500.0,
            "stop": 24510.0,
            "target1": 24440.0,
            "target2": 24400.0,
            "target_structure": "60/100/Runner",
            "confidence": 1.0,
            "regime": "trending|high_vol|no_gap|near_resistance",
            "throttle_level": "FULL",
            "accepted": True,
            "rejection_reason": None,
            "outcome": "PENDING",
        }
    )
    outcome_ts = "2026-03-05 17:05:00+05:30"
    updated = db.update_signal_outcome(
        timestamp=decision_ts,
        direction="SELL",
        entry=24500.0,
        stop=24600.0,
        target=24200.0,
        risk=100.0,
        reward=25.5,
        outcome="WIN",
        holding_bars=FORWARD_BARS,
        outcome_timestamp=outcome_ts,
        forward_outcome={"realized_pnl_points": 25.5},
    )
    assert updated == 1
    row = db.recent_signals(limit=1)[0]
    assert row["timestamp"] == decision_ts
    assert row["outcome"] == "WIN"
    assert row["outcome_timestamp"] == outcome_ts
    assert float(row["reward"]) == 25.5
    assert float(row["risk"]) == 100.0
    assert int(row["holding_bars"]) == FORWARD_BARS
    db.close()


def test_build_realtime_layer4_plan_buy_targets() -> None:
    plan = build_realtime_layer4_plan(
        model_id="TEST",
        direction="BUY",
        entry=100.0,
        stop_loss=90.0,
        risk_points=10.0,
        liquidity_target=None,
        signal_reason_stack={},
        forward_outcome=None,
    )
    assert plan["target_1"] == 110.0
    assert plan["target_2"] == 120.0
    assert plan["outcome_pending"] is True
