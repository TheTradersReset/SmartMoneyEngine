"""Tests for BUY_V3 candidate validation research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.buy_v3_candidate_validation_research import (
    ABLATION_VARIANTS,
    BUY_V3_EVENTS,
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
    BuyV3CandidateEngine,
    BuyV3CandidateValidationError,
    BuyV3CandidateValidationReport,
    BuyV3CandidateValidationResearch,
    _ablation_contribution_ranking,
    _classification_summary,
    _load_v2_false_reversal_cohort,
    _passes_production_gates,
    _signal_timing_analysis,
    _split_trading_day_sets,
    generate_buy_v3_candidate_validation_report,
)


def test_formula_constants() -> None:
    assert "Failed Breakdown" in BUY_V3_FORMULA_TEXT
    assert "Gap Reversal" in BUY_V3_FORMULA_TEXT
    assert "Liquidity Grab" in BUY_V3_FORMULA_TEXT
    assert "Near Support" in BUY_V3_FORMULA_TEXT
    assert "PDL Sweep" in BUY_V3_FORMULA_TEXT
    assert BUY_V3_EVENTS == ("Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep")


def test_buy_v3_engine_required_events() -> None:
    engine = BuyV3CandidateEngine()
    assert engine.MODEL_ID == BUY_V3_MODEL_ID
    assert engine.REQUIRED_EVENTS == BUY_V3_EVENTS
    assert engine.REQUIRED_LOCATION == "Near Support"


def test_ablation_variants_cover_required_removals() -> None:
    removed = {meta["removed"] for meta in ABLATION_VARIANTS.values() if meta["removed"]}
    assert removed == {"Liquidity Grab", "Near Support", "PDL Sweep", "Gap Reversal"}
    assert "full_buy_v3" in ABLATION_VARIANTS


def test_classification_summary_false_rate() -> None:
    signals = [
        {"classification": "Real Reversal", "win": True},
        {"classification": "False Reversal", "win": False},
        {"classification": "Dead Cat Bounce", "win": False},
    ]
    summary = _classification_summary(signals)
    assert summary["false_reversal_rate_pct"] == pytest.approx(66.67, abs=0.1)


def test_signal_timing_analysis() -> None:
    signals = [
        {"bars_before_expansion": 5, "points_before_expansion": 12.0},
        {"bars_before_expansion": 0},
        {"bars_before_expansion": -2},
        {"bars_before_expansion": None},
    ]
    result = _signal_timing_analysis(signals, engine_key="BUY_V3")
    assert result["before_expansion_count"] == 1
    assert result["during_expansion_count"] == 1
    assert result["after_expansion_count"] == 1
    assert result["no_linked_move_count"] == 1


def test_load_v2_false_reversal_cohort() -> None:
    payload = {
        "per_signal_details": {
            "buy_v1": [{"bar": 10, "timestamp": "2026-01-01 10:00:00"}],
            "buy_v2": [
                {"bar": 500, "classification": "False Reversal"},
                {"bar": 600, "classification": "Real Reversal"},
                {"bar": 10, "classification": "Dead Cat Bounce"},
            ],
        },
    }
    cohort = _load_v2_false_reversal_cohort(payload)
    assert len(cohort) == 1
    assert cohort[0]["bar"] == 500


def test_ablation_contribution_ranking() -> None:
    ablation = {
        "full_buy_v3": {
            "overall_statistics": {"win_rate_pct": 70.0, "profit_factor": 3.0, "signals_per_month": 30.0},
            "false_reversal_rate_pct": 5.0,
            "point_capture": {"40": {"capture_rate_pct": 10.0}},
        },
        "minus_liquidity_grab": {
            "removed_condition": "Liquidity Grab",
            "overall_statistics": {"win_rate_pct": 50.0, "profit_factor": 1.5, "signals_per_month": 80.0},
            "false_reversal_rate_pct": 40.0,
            "point_capture": {"40": {"capture_rate_pct": 5.0}},
        },
    }
    ranking = _ablation_contribution_ranking(ablation)
    assert ranking["most_quality_contribution"] == "Liquidity Grab"
    assert ranking["most_frequency_contribution"] == "Liquidity Grab"


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
    destination = tmp_path / "buy_v3_candidate_validation.json"

    def _fake_run(self, metadata: dict) -> BuyV3CandidateValidationReport:
        del metadata
        return BuyV3CandidateValidationReport(
            report_type="BUY_V3 Candidate Validation",
            engines_compared=["BUY_V1", "BUY_V2", "BUY_V3", "SELL_V5"],
            buy_v1_formula=["Liquidity Grab", "Failed Breakdown", "Near Support"],
            buy_v2_formula=["Failed Breakdown", "Gap Reversal"],
            buy_v3_formula=["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "Near Support", "PDL Sweep"],
            symbol="NIFTY50",
            timeframe="5M",
            trading_days_replayed=120,
            replay_start_date="2026-01-01",
            replay_end_date="2026-07-03",
            walk_forward={},
            methodology={"actual_replay": True},
            replay_rules={},
            comparison={
                "buy_v3": {"overall_statistics": {"signals_emitted": 86}, "signals_emitted_count": 86},
            },
            false_reversal_removal={"removed_by_buy_v3": 900, "baseline_false_reversal_count": 947},
            ablation_analysis={},
            signal_timing={},
            tradeability={},
            per_signal_details={"buy_v3": []},
            failed_signal_classification={},
            sell_v5_benchmark={},
            production_safety_check={},
            final_verdict={"classification": "Dry Run Candidate"},
            conclusions=[],
            execution_time_seconds=1.0,
        )

    monkeypatch.setattr(BuyV3CandidateValidationResearch, "run", _fake_run)

    report = generate_buy_v3_candidate_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.comparison["buy_v3"]["overall_statistics"]["signals_emitted"] == 86
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(BuyV3CandidateValidationError):
        generate_buy_v3_candidate_validation_report(filter_report_path=Path("missing.json"))
