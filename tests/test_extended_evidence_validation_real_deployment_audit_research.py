"""Tests for extended evidence validation real deployment audit research."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.research.extended_evidence_validation_real_deployment_audit_research import (
    REPLAY_WINDOWS,
    ExtendedEvidenceValidationRealDeploymentAuditResearch,
    _combined_throttled_metrics,
    _evidence_score_from_breakdown,
    _evidence_strength_audit,
    _final_verdict,
    _outcome_distribution,
    _sell_v6_component_ranking,
    _split_trading_day_sets_70_30,
    _synthesize_prior_exports,
    _tier_capture_extended,
    _unknown_risk_audit,
    generate_extended_evidence_validation_real_deployment_audit_report,
)


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-06 09:30:00+05:30",
        "bar": 100,
        "direction": "BUY",
        "entry": 23500.0,
        "stop_loss": 23450.0,
        "realized_pnl_points": 40.0,
        "mfe_points": 80.0,
        "mae_points": 15.0,
        "win": True,
        "classification": "Real Reversal",
        "bars_before_expansion": 3,
        "layers": {
            "layer1": {
                "events_detected": ["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep"],
            },
            "layer2": {"htf_trend": "Bullish", "vwap_state": "Reclaimed", "location": "Near Support"},
        },
    }
    base.update(overrides)
    return base


def _sell_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 10:25:00+05:30",
        "bar": 90,
        "direction": "SELL",
        "entry": 23600.0,
        "stop_loss": 23650.0,
        "realized_pnl_points": 60.0,
        "mfe_points": 120.0,
        "mae_points": 25.0,
        "win": True,
        "classification": "Winner",
        "bars_before_expansion": 4,
        "layers": {
            "layer1": {"events_detected": ["Failed Breakout"]},
            "layer2": {
                "htf_trend": "Bearish",
                "vwap_state": "Below",
                "vwap_gate_passes": True,
                "v4_ema_bearish": True,
                "location": "Near Resistance",
            },
        },
    }
    base.update(overrides)
    return base


def test_replay_windows_constant() -> None:
    assert REPLAY_WINDOWS == (120, 250, 500)


def test_split_trading_day_sets_70_30() -> None:
    dates = {date(2026, 1, d) for d in range(1, 11)}
    train, validate = _split_trading_day_sets_70_30(dates)
    assert len(train) == 7
    assert len(validate) == 3
    assert max(train) < min(validate)


def test_tier_capture_extended() -> None:
    signals = [
        _buy_signal(mfe_points=45),
        _sell_signal(mfe_points=150),
        _buy_signal(mfe_points=310, bar=101),
    ]
    tiers = _tier_capture_extended(signals)
    assert tiers["40"]["signals_hitting_tier"] == 3
    assert tiers["300"]["signals_hitting_tier"] == 1


def test_outcome_distribution() -> None:
    signals = [_buy_signal(), _sell_signal(realized_pnl_points=-20.0, win=False)]
    dist = _outcome_distribution(signals, win_fn=lambda s: bool(s.get("win")))
    assert dist["trade_count"] == 2
    assert dist["win_count"] == 1
    assert dist["loss_count"] == 1


def test_evidence_score_from_breakdown() -> None:
    score = _evidence_score_from_breakdown(replay_pct=70.0, synthesis_pct=20.0, assumption_pct=10.0)
    assert 80.0 <= score <= 100.0


def test_sell_v6_component_ranking() -> None:
    ranking = _sell_v6_component_ranking([_sell_signal(), _sell_signal(bar=91, timestamp="2026-01-05 11:00:00")])
    assert ranking["top_contributor"] is not None
    assert len(ranking["ranking"]) == 5


def test_combined_throttled_metrics() -> None:
    buy = [_buy_signal()]
    sell = [_sell_signal()]
    throttle_maps = {
        "buy_v3": {"trending|low_vol|no_gap": "FULL"},
        "sell_v6": {"trending|low_vol|no_gap": "HALF"},
    }
    result = _combined_throttled_metrics(buy, sell, throttle_maps, trading_days=120)
    assert result["signals_emitted"] >= 1
    assert "profit_factor" in result


def test_synthesize_prior_exports_empty() -> None:
    synthesis = _synthesize_prior_exports({})
    assert synthesis["aligned_with_prior_audits"] is None
    assert len(synthesis["gaps"]) >= 3


def test_unknown_risk_audit() -> None:
    audit = _unknown_risk_audit(
        window_results={"500": {"combined": {"walk_forward": {"stable": False}}}},
        prior_synthesis={"gaps": []},
    )
    assert len(audit["unknowns"]) == 10
    assert audit["thesis_invalidating_count"] >= 1


def test_final_verdict_paper() -> None:
    window_results = {
        "120": {"combined": {"profit_factor": 2.2}, "combined_regime_throttle": {"profit_factor": 2.4}},
        "250": {"combined": {"profit_factor": 2.0}, "combined_regime_throttle": {"profit_factor": 2.1}},
        "500": {
            "combined": {"profit_factor": 2.1, "walk_forward": {"stable": True}},
            "combined_regime_throttle": {"profit_factor": 2.3},
        },
    }
    final = _final_verdict(
        window_results=window_results,
        scores={"production_readiness_score": 72, "confidence_score": 68, "production_risk_score": 60},
        evidence_audit={"aggregate_evidence_score": 70},
        unknown_risks={"thesis_invalidating_count": 0},
        prior_synthesis={"aligned_with_prior_audits": True},
        ablation={},
    )
    assert final["definitive_verdict"] in {"Paper", "Small Capital", "Full Capital"}
    assert final["should_research_buy_v4"] == "NO"


def test_evidence_strength_audit() -> None:
    audit = _evidence_strength_audit(
        window_results={
            "500": {
                "combined": {"profit_factor": 2.5},
                "combined_regime_throttle": {"profit_factor": 2.6},
            },
        },
        ablation={"contribution_ranking": {"most_quality_contribution": "Liquidity Grab"}},
        prior_synthesis={"aligned_with_prior_audits": True},
    )
    assert audit["aggregate_evidence_score"] > 0
    assert len(audit["recommendations"]) >= 8


def test_mocked_full_run_exports_json(tmp_path: Path) -> None:
    metadata = {"end_date": "2026-07-02"}
    filter_path = tmp_path / "filter.json"
    filter_path.write_text(json.dumps(metadata), encoding="utf-8")

    buy = _buy_signal(bar=5)
    sell = _sell_signal(bar=3)
    frame = pd.DataFrame(
        {
            "Date": [f"2026-01-{d:02d} 09:15:00+05:30" for d in range(1, 11)],
            "Open": [23500.0] * 10,
            "High": [23550.0] * 10,
            "Low": [23450.0] * 10,
            "Close": [23520.0] * 10,
            "Volume": [1000] * 10,
        },
    )

    mock_window = {
        "trading_days": 120,
        "replay_start_date": "2026-01-01",
        "replay_end_date": "2026-01-10",
        "buy_v3_only": {"profit_factor": 2.5, "win_rate_pct": 70.0, "signals_per_month": 30.0},
        "sell_v6_only": {"profit_factor": 3.0, "win_rate_pct": 72.0, "signals_per_month": 80.0},
        "combined": {
            "profit_factor": 2.8,
            "win_rate_pct": 71.0,
            "signals_per_month": 110.0,
            "walk_forward": {"stable": True, "train": {}, "validate": {}},
        },
        "combined_regime_throttle": {"profit_factor": 3.1, "win_rate_pct": 73.0},
        "signal_classification": {},
        "signal_timing": {},
        "target_achievement": {},
        "mfe_distribution": {},
    }

    report_path = tmp_path / "extended_audit.json"
    research = ExtendedEvidenceValidationRealDeploymentAuditResearch()

    with (
        patch.object(research, "_replay_production", return_value={"buy_v3": [buy], "sell_v6": [sell]}),
        patch(
            "src.research.extended_evidence_validation_real_deployment_audit_research.FilterResearchEngine",
        ) as mock_filter_cls,
        patch(
            "src.research.extended_evidence_validation_real_deployment_audit_research._last_n_trading_day_set",
            return_value={date(2026, 1, d) for d in range(1, 11)},
        ),
        patch.object(research, "_analyze_window", return_value=mock_window),
        patch.object(research, "_run_ablation_analysis", return_value={"contribution_ranking": {}}),
        patch.object(
            research,
            "_run_execution_analysis",
            return_value={"runner_optimization": {"buy_v3": {}, "sell_v6": {}}},
        ),
    ):
        mock_filter = MagicMock()
        mock_filter._ensure_pipeline.return_value = tmp_path / "data.csv"
        mock_filter_cls.return_value = mock_filter
        (tmp_path / "data.csv").write_text(frame.to_csv(index=False), encoding="utf-8")

        report = research.run(metadata, windows=(120,), run_ablation=False)
        research.export(report, report_path)

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Extended Evidence Validation & Real Deployment Audit"
    assert "final_answer" in payload
    assert payload["replay_windows"] == [120]


def test_generate_helper_raises_without_filter(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        generate_extended_evidence_validation_real_deployment_audit_report(
            report_path=tmp_path / "out.json",
            filter_report_path=tmp_path / "missing.json",
        )
