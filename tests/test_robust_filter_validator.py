"""Tests for robust filter validator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.filter_research_engine import (
    FilterState,
    FilteredTradeRecord,
)
from src.research.robust_filter_validator import (
    MIN_TRADES,
    RobustFilterValidator,
    ValidatedCombination,
)
from src.signals.setup_classifier import SetupType


def _trade(
    index: int,
    outcome: str,
    pnl: float,
    setup_type: str = SetupType.CONTINUATION_BOS.value,
    session: str = "Midday",
) -> FilteredTradeRecord:
    base = pd.Timestamp("2026-01-06 09:15:00+05:30")
    return FilteredTradeRecord(
        setup_type=setup_type,
        direction="bullish",
        timeframe="5M",
        trigger_bar=index,
        trigger_timestamp=(base + pd.Timedelta(minutes=5 * index)).isoformat(),
        entry_hit=True,
        outcome=outcome,
        realized_pnl_points=pnl,
        realized_rr=1.0 if pnl > 0 else -1.0,
        filters=FilterState(
            ema_alignment="Mixed",
            vwap_position="Above VWAP",
            rsi_band="50-60",
            session=session,
            atr_percentile="Mid (34-66)",
            volume_spike="No",
        ),
    )


def test_max_drawdown() -> None:
    drawdown = RobustFilterValidator._max_drawdown([10.0, -5.0, -8.0, 15.0])
    assert drawdown == 13.0


def test_train_validation_split_ratio() -> None:
    trades = [_trade(index, "Win", 10.0) for index in range(100)]
    validator = RobustFilterValidator()
    train, validation = validator._split_trades(trades)
    assert len(train) == 70
    assert len(validation) == 30


def test_validation_passes_when_criteria_met() -> None:
    validator = RobustFilterValidator()
    train = validator._period_metrics([_trade(1, "Win", 10.0)] * 70)
    validation = validator._period_metrics([_trade(80, "Win", 8.0)] * 30)
    passed, degradation = validator._validation_result(train, validation)
    assert passed is True
    assert degradation == 0.0


def test_validation_fails_on_win_rate_degradation() -> None:
    validator = RobustFilterValidator()
    train = validator._period_metrics([_trade(1, "Win", 10.0)] * 70)
    validation = validator._period_metrics(
        [_trade(80, "Loss", -5.0)] * 30
    )
    passed, degradation = validator._validation_result(train, validation)
    assert passed is False
    assert degradation >= 10.0


def test_evaluate_candidate_rejects_low_sample() -> None:
    trades = [_trade(index, "Win", 10.0) for index in range(50)]
    validator = RobustFilterValidator(min_trades=100)
    result = validator._evaluate_candidate(
        trades,
        SetupType.CONTINUATION_BOS.value,
        {"session": "Midday"},
    )
    assert result is None


def test_evaluate_candidate_accepts_qualifying_combo() -> None:
    wins = [_trade(index, "Win", 10.0) for index in range(80)]
    losses = [_trade(index + 80, "Loss", -4.0) for index in range(20)]
    trades = wins + losses
    validator = RobustFilterValidator(min_trades=100, min_profit_factor=1.1)
    result = validator._evaluate_candidate(
        trades,
        SetupType.CONTINUATION_BOS.value,
        {"session": "Midday"},
    )
    assert result is not None
    assert result.trades == 100
    assert (result.profit_factor or 0) > 1.1


def test_stability_score_zero_when_invalid() -> None:
    validator = RobustFilterValidator()
    train = validator._period_metrics([_trade(1, "Win", 10.0)] * 70)
    validation = validator._period_metrics([_trade(80, "Loss", -5.0)] * 30)
    score = validator._stability_score(train, validation, False, 100.0)
    assert score == 0.0


def test_candidates_from_input_report(tmp_path: Path) -> None:
    payload = {
        "top_20_combinations": [
            {
                "label": "Continuation BOS: session=Closing",
                "filters": {"session": "Closing"},
            }
        ],
        "single_filter_analysis": {},
    }
    report_path = tmp_path / "filter_research_report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validator = RobustFilterValidator(input_report_path=report_path)
    candidates = validator._candidates_from_input(payload)
    assert len(candidates) == 1
    assert candidates[0][0] == SetupType.CONTINUATION_BOS.value


def test_report_structure_fields() -> None:
    from src.research.robust_filter_validator import RobustFilterReport

    report = RobustFilterReport(
        input_report_path="input.json",
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        timeframes_analyzed=["5M"],
        setups_analyzed=["Continuation BOS"],
        train_split_pct=70.0,
        validation_split_pct=30.0,
        min_trades=100,
        min_profit_factor=1.1,
        candidates_from_input=5,
        candidates_evaluated=10,
        candidates_after_trade_filter=3,
        removed_low_sample_count=7,
        top_10_robust_combinations=[],
        top_10_overfit_combinations=[],
        best_production_ready_filter_stack=None,
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert "top_10_robust_combinations" in payload
    assert payload["min_trades"] == 100


def test_min_trades_constant() -> None:
    assert MIN_TRADES == 100


@pytest.mark.integration
def test_full_robust_validation_if_input_exists() -> None:
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "outputs" / "research" / "filter_research_report.json"
    if not input_path.exists():
        pytest.skip("Filter research report not available.")

    validator = RobustFilterValidator(input_report_path=input_path)
    report = validator.run()
    assert report.candidates_evaluated > 0
