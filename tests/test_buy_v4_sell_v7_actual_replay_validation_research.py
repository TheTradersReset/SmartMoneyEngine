"""Tests for BUY_V4 / SELL_V7 actual replay validation research."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.research.buy_v4_sell_v7_actual_replay_validation_research import (
    DEFAULT_BUY_V4_REJECT_PATTERNS,
    DEFAULT_SELL_V7_REJECT_PATTERNS,
    FALLBACK_TRADING_DAYS,
    PREFERRED_TRADING_DAYS,
    BuyV4CandidateEngine,
    BuyV4SellV7ActualReplayValidationError,
    BuyV4SellV7ActualReplayValidationReport,
    BuyV4SellV7ActualReplayValidationResearch,
    SellV7CandidateEngine,
    _build_final_answer,
    _compare_engines,
    _engine_should_emit,
    _load_blueprint_filters,
    _patterns_blocked,
    _patterns_use_forward_path,
    _resolve_trading_days,
    generate_buy_v4_sell_v7_actual_replay_validation_report,
)


def _signal(*, side: str = "BUY", mfe: float = 80.0, mae: float = 20.0, **overrides: object) -> dict:
    if side == "BUY":
        base = {
            "timestamp": "2026-01-06 09:30:00+05:30",
            "bar": 100,
            "direction": "BUY",
            "entry": 23500.0,
            "stop_loss": 23490.0,
            "mfe_points": mfe,
            "mae_points": mae,
            "realized_pnl_points": 40.0,
            "win": True,
            "classification": "Real Reversal",
            "bars_before_expansion": 4,
            "layers": {
                "layer1": {
                    "events_detected": [
                        "Failed Breakdown",
                        "Gap Reversal",
                        "Liquidity Grab",
                        "PDL Sweep",
                    ],
                },
                "layer2": {
                    "htf_trend": "Bullish",
                    "vwap_state": "Reclaimed",
                    "location": "Near Support",
                },
            },
        }
    else:
        base = {
            "timestamp": "2026-01-05 10:25:00+05:30",
            "bar": 90,
            "direction": "SELL",
            "entry": 23600.0,
            "stop_loss": 23610.0,
            "mfe_points": mfe,
            "mae_points": mae,
            "realized_pnl_points": 50.0,
            "win": True,
            "classification": "Winner",
            "bars_before_expansion": 3,
            "layers": {
                "layer1": {"events_detected": ["Failed Breakout"], "primary_event": "Failed Breakout"},
                "layer2": {"htf_trend": "Bearish", "vwap_state": "Below", "aligned": True},
            },
        }
    base.update(overrides)
    return base


def test_resolve_trading_days_prefers_300() -> None:
    assert _resolve_trading_days(400) == PREFERRED_TRADING_DAYS
    assert _resolve_trading_days(280) == FALLBACK_TRADING_DAYS
    assert _resolve_trading_days(100) == 100


def test_blueprint_defaults_and_load(tmp_path: Path) -> None:
    missing = _load_blueprint_filters(tmp_path / "missing.json")
    assert missing["buy_v4_reject_patterns"] == list(DEFAULT_BUY_V4_REJECT_PATTERNS)
    assert missing["sell_v7_reject_patterns"] == list(DEFAULT_SELL_V7_REJECT_PATTERNS)

    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "buy_v4_design": {"selected_patterns": ["Liquidity Sweep Failure", "Gap Continuation"]},
                "sell_v7_design": {"selected_patterns": ["Liquidity Sweep Failure", "Volatility Collapse"]},
            },
        ),
        encoding="utf-8",
    )
    loaded = _load_blueprint_filters(path)
    assert loaded["blueprint_loaded"] is True
    assert "Liquidity Sweep Failure" in loaded["buy_v4_reject_patterns"]
    assert "Volatility Collapse" in loaded["sell_v7_reject_patterns"]


def test_buy_v4_engine_rejects_liquidity_sweep_failure() -> None:
    engine = BuyV4CandidateEngine()
    # mae > mfe + sweep events → Liquidity Sweep Failure (+ Gap Continuation)
    loser = _signal(side="BUY", mfe=10.0, mae=50.0)
    blocked = _patterns_blocked(loser, side="BUY", reject_patterns=engine.reject_patterns)
    assert "Liquidity Sweep Failure" in blocked
    emit, patterns = engine.should_emit_signal(loser)
    assert emit is False
    assert patterns

    winner = _signal(side="BUY", mfe=120.0, mae=15.0)
    emit_ok, blocked_ok = engine.should_emit_signal(winner)
    assert emit_ok is True
    assert blocked_ok == []


def test_sell_v7_engine_rejects_volatility_collapse() -> None:
    engine = SellV7CandidateEngine()
    collapse = _signal(side="SELL", mfe=20.0, mae=150.0)
    emit, blocked = engine.should_emit_signal(collapse)
    assert emit is False
    assert "Volatility Collapse" in blocked

    clean = _signal(side="SELL", mfe=100.0, mae=20.0)
    # Failed Breakout + mae < mfe → no Liquidity Sweep Failure / Volatility Collapse
    emit_ok, blocked_ok = engine.should_emit_signal(clean)
    assert emit_ok is True
    assert blocked_ok == []


def test_engine_should_emit_helper() -> None:
    ok, blocked = _engine_should_emit(
        _signal(side="BUY", mfe=100.0, mae=10.0),
        side="BUY",
        reject_patterns=DEFAULT_BUY_V4_REJECT_PATTERNS,
    )
    assert ok is True
    assert blocked == []


def test_forward_path_dependency_flags() -> None:
    assert _patterns_use_forward_path(["Liquidity Sweep Failure"]) is True
    assert _patterns_use_forward_path(["Counter Trend Entry"]) is False


def test_compare_and_final_answer_structure() -> None:
    baseline = {
        "signals": 100,
        "profit_factor": 1.5,
        "win_rate_pct": 30.0,
        "expectancy": 20.0,
    }
    candidate = {
        "signals": 50,
        "profit_factor": 4.0,
        "win_rate_pct": 60.0,
        "expectancy": 100.0,
    }
    compare = _compare_engines(
        baseline, candidate, baseline_name="BUY_V3", candidate_name="BUY_V4",
    )
    assert compare["metric_outperform_10pct_pf"] is True
    assert compare["signal_reduction_pct"] == 50.0

    final = _build_final_answer(
        buy_v3=baseline,
        buy_v4=candidate,
        sell_v6={**baseline, "profit_factor": 2.4, "win_rate_pct": 60.0},
        sell_v7={**candidate, "profit_factor": 7.0, "win_rate_pct": 80.0},
        buy_compare=compare,
        sell_compare=_compare_engines(
            {**baseline, "profit_factor": 2.4, "win_rate_pct": 60.0, "expectancy": 70.0},
            {**candidate, "profit_factor": 7.0, "win_rate_pct": 80.0, "expectancy": 150.0},
            baseline_name="SELL_V6",
            candidate_name="SELL_V7",
        ),
        buy_patterns=list(DEFAULT_BUY_V4_REJECT_PATTERNS),
        sell_patterns=list(DEFAULT_SELL_V7_REJECT_PATTERNS),
        trading_days=300,
        throttle={"profit_factor": 3.0},
    )
    # Forward-path gated filters → NO replace despite metric lift
    assert final["should_buy_v4_replace_buy_v3"] == "NO"
    assert final["should_sell_v7_replace_sell_v6"] == "NO"
    assert final["does_buy_v4_genuinely_outperform_buy_v3"] == "NO"
    assert final["best_buy_engine"] == "BUY_V3"
    assert final["best_sell_engine"] == "SELL_V6"
    assert final["evidence_strength"] == "WEAK"
    assert final["overfitting_risk"] == "HIGH"
    assert "pf_wr_comparison" in final
    assert final["best_stop"] == "fixed_10"
    assert final["best_exit_structure"] == "60/100/Runner"


def test_generate_report_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text('{"end_date": "2026-07-03"}', encoding="utf-8")
    destination = tmp_path / "buy_v4_sell_v7_actual_replay_validation.json"

    def _fake_run(self, metadata: dict, **kwargs: object) -> BuyV4SellV7ActualReplayValidationReport:
        del metadata, kwargs
        return BuyV4SellV7ActualReplayValidationReport(
            report_type="BUY_V4 & SELL_V7 Actual Replay Validation",
            engines=["BUY_V3", "BUY_V4", "SELL_V6", "SELL_V7"],
            symbol="NIFTY50",
            timeframe="5M",
            trading_days_used=300,
            preferred_trading_days=300,
            fallback_trading_days=240,
            available_trading_days=420,
            replay_start_date="2025-01-01",
            replay_end_date="2026-07-03",
            methodology={"actual_bar_replay": True},
            approved_filters={},
            engine_definitions={},
            core_metrics={
                "buy_v3": {"signals": 200, "profit_factor": 1.5, "win_rate_pct": 30.0},
                "buy_v4": {"signals": 100, "profit_factor": 4.0, "win_rate_pct": 60.0},
                "sell_v6": {"signals": 500, "profit_factor": 2.4, "win_rate_pct": 64.0},
                "sell_v7": {"signals": 350, "profit_factor": 5.0, "win_rate_pct": 80.0},
            },
            target_achievement_matrix={},
            trade_lifecycle={},
            entry_timing={},
            reward_risk={},
            capture_metrics={},
            regime_throttle={},
            engine_comparison={},
            per_signal_details={"buy_v3": [], "buy_v4": [], "sell_v6": [], "sell_v7": []},
            final_answer={
                "should_buy_v4_replace_buy_v3": "NO",
                "should_sell_v7_replace_sell_v6": "NO",
                "pf_wr_comparison": {},
            },
            conclusions=["test"],
            execution_time_seconds=1.0,
        )

    monkeypatch.setattr(BuyV4SellV7ActualReplayValidationResearch, "run", _fake_run)

    report = generate_buy_v4_sell_v7_actual_replay_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.trading_days_used == 300
    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["engines"] == ["BUY_V3", "BUY_V4", "SELL_V6", "SELL_V7"]
    assert payload["final_answer"]["should_buy_v4_replace_buy_v3"] == "NO"


def test_generate_report_missing_filter() -> None:
    with pytest.raises(BuyV4SellV7ActualReplayValidationError):
        generate_buy_v4_sell_v7_actual_replay_validation_report(
            filter_report_path=Path("missing_filter_report.json"),
        )
