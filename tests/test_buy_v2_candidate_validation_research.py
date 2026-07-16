"""Tests for BUY_V2 candidate validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.buy_v2_candidate_validation_research import (
    BUY_V1_FORMULA_TEXT,
    BUY_V2_FORMULA_TEXT,
    BUY_V2_COMPONENTS,
    BuyV1CandidateEngine,
    BuyV2CandidateEngine,
    BuyV2CandidateValidationError,
    BuyV2CandidateValidationReport,
    BuyV2CandidateValidationResearch,
    _bullish_point_capture,
    _classify_failed_buy_signal,
    _events_in_lookback,
    _passes_production_gates,
    _split_trading_day_sets,
    generate_buy_v2_candidate_validation_report,
)


def test_formula_constants() -> None:
    assert "Failed Breakdown" in BUY_V2_FORMULA_TEXT
    assert "Gap Reversal" in BUY_V2_FORMULA_TEXT
    assert "Liquidity Grab" in BUY_V1_FORMULA_TEXT
    assert BUY_V2_COMPONENTS == ("Failed Breakdown", "Gap Reversal")


def test_buy_v2_engine_required_events() -> None:
    engine = BuyV2CandidateEngine()
    assert engine.MODEL_ID == "LDM-BUY-V2"
    assert engine.REQUIRED_EVENTS == ("Failed Breakdown", "Gap Reversal")
    assert engine.REQUIRED_LOCATION is None


def test_buy_v1_engine_required_events() -> None:
    engine = BuyV1CandidateEngine()
    assert engine.MODEL_ID == "LDM-BUY-V1"
    assert engine.REQUIRED_LOCATION == "Near Support"


def test_classify_failed_buy_signal_no_expansion() -> None:
    label = _classify_failed_buy_signal({"mfe_points": 20, "mae_points": 10, "win": False}, context={})
    assert label == "No Expansion"


def test_classify_failed_buy_signal_real_reversal() -> None:
    label = _classify_failed_buy_signal(
        {"mfe_points": 250, "mae_points": 30, "win": True},
        context={"htf_trend": "Bullish"},
    )
    assert label == "Real Reversal"


def test_split_trading_day_sets() -> None:
    from datetime import date, timedelta

    start = date(2026, 1, 1)
    parsed = {start + timedelta(days=offset) for offset in range(120)}
    train, validate = _split_trading_day_sets(parsed)
    assert len(train) == 80
    assert len(validate) == 40


def test_passes_production_gates() -> None:
    result = _passes_production_gates(
        {"win_rate_pct": 70.0, "profit_factor": 2.5, "signals_per_month": 25.0},
        {"40": {"capture_rate_pct": 5.0}},
    )
    assert result["all_pass"] is True


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text('{"end_date": "2026-07-03"}', encoding="utf-8")
    destination = tmp_path / "buy_v2_candidate_validation.json"

    def _fake_run(self, metadata: dict) -> BuyV2CandidateValidationReport:
        del metadata
        return BuyV2CandidateValidationReport(
            report_type="BUY_V2 Candidate Validation",
            engines_compared=["BUY_V1", "BUY_V2"],
            buy_v1_formula=["Liquidity Grab", "Failed Breakdown", "Near Support"],
            buy_v2_formula=["Failed Breakdown", "Gap Reversal"],
            symbol="NIFTY50",
            timeframe="5M",
            trading_days_replayed=120,
            replay_start_date="2026-01-01",
            replay_end_date="2026-07-03",
            walk_forward={},
            methodology={"actual_replay": True},
            replay_rules={},
            comparison={
                "buy_v1": {"overall_statistics": {"signals_emitted": 12}, "signals_emitted_count": 12},
                "buy_v2": {"overall_statistics": {"signals_emitted": 45}, "signals_emitted_count": 45},
            },
            per_signal_details={"buy_v1": [], "buy_v2": []},
            missed_reversal_recovery={"recovered_by_buy_v2": 18, "cohort_size_used": 47},
            failed_signal_classification={},
            condition_attribution={},
            sell_v5_benchmark={},
            production_safety_check={},
            final_verdicts={
                "replay_validated_vs_synthesis_only": {"answer": "REPLAY-VALID"},
                "buy_v2_classification": {"verdict": "Dry Run Candidate"},
                "buy_v2_sell_v5_equivalent": {"answer": "PARTIAL"},
            },
            conclusions=[],
            execution_time_seconds=1.0,
        )

    monkeypatch.setattr(BuyV2CandidateValidationResearch, "run", _fake_run)

    report = generate_buy_v2_candidate_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.comparison["buy_v2"]["overall_statistics"]["signals_emitted"] == 45
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(BuyV2CandidateValidationError):
        generate_buy_v2_candidate_validation_report(filter_report_path=Path("missing.json"))
