"""Tests for per-candle diagnostics reporting."""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.candle_builder import Candle
from src.pipeline.candle_diagnostics import build_candle_report, format_candle_report
from src.signals.regime_throttle import ThrottleDecision


def _sample_candle() -> Candle:
    return Candle(
        symbol="NSE:NIFTY50-INDEX",
        timestamp=datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc),
        open=25000.0,
        high=25010.0,
        low=24990.0,
        close=25005.0,
        volume=1000,
        tick_count=42,
    )


def test_build_candle_report_no_trade_lists_failed_conditions() -> None:
    buy_eval = {
        "verdict": "NO_TRADE",
        "layer1": {
            "active": False,
            "formula_events_matched": ["Gap Reversal"],
            "formula_events_missing": ["PDL Sweep", "Failed Breakdown"],
        },
        "layer2": {
            "htf_trend": "Neutral",
            "vwap_state": "Above",
            "ema_structure": "Mixed",
            "location": "Mid Range",
            "location_ok": False,
            "aligned": False,
        },
        "layer3": {"confirmed": True, "volume_bucket": "Normal", "confirmation_candle": "None"},
        "layer5": {"pass": False, "reason_codes": ["FORMULA_INCOMPLETE", "LOCATION_MISMATCH"]},
        "context": {
            "htf_trend": "Neutral",
            "vwap": "Above",
            "bos": "Absent",
            "choch": "Absent",
            "rsi": "40-60",
            "location": "Mid Range",
        },
    }
    sell_eval = {
        "verdict": "NO_TRADE",
        "layer1": {"active": False, "failed_breakout_present": False, "events_detected": []},
        "layer2": {
            "htf_trend": "Neutral",
            "vwap_state": "Above",
            "vwap_gate_passes": False,
            "vwap_gate_rule": "VWAP Below only",
            "ema_structure": "Mixed",
            "v4_ema_bearish": False,
            "aligned": False,
        },
        "layer3": {"confirmed": True, "volume_bucket": "Normal", "confirmation_candle": "None"},
        "layer5": {"pass": False, "reason_codes": ["NO_EARLY_WARNING", "VWAP_MISMATCH"]},
        "context": {"htf_trend": "Neutral", "vwap": "Above", "location": "Mid Range"},
    }

    report = build_candle_report(
        candle=_sample_candle(),
        bar=150,
        buy_eval=buy_eval,
        sell_eval=sell_eval,
        eval_ms=12.34,
        context_snapshot={
            "rsi": 52.1,
            "atr": 18.5,
            "ema20": 24980.0,
            "ema50": 24950.0,
            "ema200": 24800.0,
            "support_zone": 24900.0,
            "resistance_zone": 25100.0,
            "bar_events": {"Gap Reversal"},
            "lookback_events": {"Gap Reversal"},
            "regime_composite": "range|low_vol|no_gap|mid_range",
        },
    )

    assert report["final_signal"] == "NO_TRADE"
    assert report["buy_decision"] == "NO_TRADE"
    assert report["sell_decision"] == "NO_TRADE"
    assert "FORMULA_INCOMPLETE" in report["reason_codes"]
    assert "VWAP_MISMATCH" in report["reason_codes"]
    assert report["eval_ms"] == 12.34
    assert len(report["buy_failed_conditions"]) >= 1
    assert len(report["sell_failed_conditions"]) >= 1

    text = format_candle_report(report)
    assert "CANDLE REPORT" in text
    assert "Buy Score" in text
    assert "Eval Time (ms)" in text


def test_build_candle_report_accepted_buy() -> None:
    buy_eval = {
        "verdict": "BUY",
        "layer1": {
            "active": True,
            "formula_events_matched": ["Gap Reversal", "PDL Sweep", "Failed Breakdown"],
            "formula_events_missing": [],
        },
        "layer2": {
            "htf_trend": "Bullish",
            "vwap_state": "Above",
            "ema_structure": "Bull Stack",
            "location": "Near Support",
            "location_ok": True,
            "aligned": True,
        },
        "layer3": {"confirmed": True, "volume_bucket": "Normal", "confirmation_candle": "Hammer"},
        "layer5": {"pass": True, "reason_codes": []},
        "context": {"htf_trend": "Bullish", "vwap": "Above", "location": "Near Support"},
    }
    sell_eval = {"verdict": "NO_TRADE", "layer1": {}, "layer2": {}, "layer3": {}, "layer5": {"pass": False, "reason_codes": []}, "context": {}}

    throttle = ThrottleDecision(
        composite_regime="trending|low_vol|no_gap|near_support",
        throttle_level="FULL",
        weight=1.0,
        accepted=True,
        rejection_reason=None,
    )

    report = build_candle_report(
        candle=_sample_candle(),
        bar=200,
        buy_eval=buy_eval,
        sell_eval=sell_eval,
        eval_ms=8.5,
        context_snapshot={},
        buy_throttle=throttle,
        buy_accepted=True,
    )

    assert report["final_signal"] == "BUY"
    assert report["buy_decision"] == "ACCEPTED"
    assert all(c["passed"] for c in report["buy_passed_conditions"])
