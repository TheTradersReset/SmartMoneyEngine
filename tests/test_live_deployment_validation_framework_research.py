"""Tests for live deployment validation framework research (mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.live_deployment_validation_framework_research import (
    LOCKED_STACK,
    REPLAY_BASELINES_240D,
    LiveDeploymentValidationFrameworkResearch,
    _checklist_20_session,
    _checklist_40_session,
    _measurement_definitions,
    _promotion_criteria,
    _risk_controls,
)


def test_locked_stack_and_baselines() -> None:
    assert LOCKED_STACK["buy_engine"] == "BUY_V3"
    assert LOCKED_STACK["sell_engine"] == "SELL_V6"
    assert LOCKED_STACK["stop"] == "fixed_10"
    assert LOCKED_STACK["targets"] == "60/100/Runner"
    assert LOCKED_STACK["regime_throttle"] is True
    assert LOCKED_STACK["buy_v4_sell_v7_status"] == "DO_NOT_PROMOTE"
    throttled = REPLAY_BASELINES_240D["combined_regime_throttle"]
    assert throttled["profit_factor"] == 5.58
    assert throttled["win_rate_pct"] == 69.01


def test_measurement_definitions_cover_required_metrics() -> None:
    defs = _measurement_definitions(baselines=REPLAY_BASELINES_240D, closure={})
    required = {
        "slippage",
        "execution_delay",
        "missed_entries",
        "same_bar_conflicts",
        "partial_fills",
        "target_execution_accuracy",
        "stop_execution_accuracy",
        "capture_efficiency",
    }
    assert required.issubset(defs.keys())
    for key in required:
        row = defs[key]
        assert "definition" in row
        assert "formula" in row
        assert "data_source" in row
        assert "pass_threshold" in row


def test_checklists_and_promotion_gates() -> None:
    c20 = _checklist_20_session()
    c40 = _checklist_40_session()
    assert c20["duration_sessions"] == 20
    assert len(c20["session_by_session"]) == 20
    assert len(c20["end_of_phase_gates"]) >= 6
    assert c40["duration_sessions"] == 40
    assert len(c40["itemized_gates"]) >= 10

    promo = _promotion_criteria(baselines=REPLAY_BASELINES_240D)
    for key in ("paper_to_inr_50k", "inr_50k_to_inr_1l", "inr_1l_to_inr_2l"):
        tier = promo[key]
        assert tier["min_sessions"] >= 20
        assert tier["min_profit_factor"] >= 1.5
        assert tier["min_win_rate_pct"] >= 55.0
        assert tier["max_median_slippage_pts"] <= 5.0
        assert tier["current_verdict"] == "NO"
        assert len(tier["evidence_required"]) >= 5


def test_risk_controls_kill_switches() -> None:
    risk = _risk_controls(closure={}, live={})
    assert len(risk["kill_switch_conditions"]) >= 10
    assert risk["maximum_allowed_slippage"]["median_pts_pass"] == 5.0
    assert risk["maximum_allowed_slippage"]["kill_median_pts"] == 10.0
    assert risk["maximum_consecutive_losses"]["paper"] == 7
    ids = {ks["id"] for ks in risk["kill_switch_conditions"]}
    assert "KS-1" in ids and "KS-3" in ids and "KS-10" in ids


def test_run_export_with_mocked_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)

    (research_dir / "deployment_readiness_validation.json").write_text(
        json.dumps(
            {
                "production_scores": {
                    "production_readiness_score": 72.0,
                    "confidence_score": 66.2,
                    "production_risk_score": 68.5,
                    "evidence_score": 84.9,
                    "deployment_tier": "Production Candidate",
                },
                "final_answer": {
                    "can_paper_trading_start_now": {"answer": "YES", "evidence": "mock"},
                    "can_inr_50k_deployment_start_now": {"answer": "NO", "evidence": "mock"},
                    "can_inr_1l_deployment_start_now": {"answer": "NO", "evidence": "mock"},
                    "can_inr_2l_deployment_start_now": {"answer": "NO", "evidence": "mock"},
                },
                "evidence_still_required_before_real_capital": [
                    "Live slippage and fill quality on NIFTY50 5M",
                ],
                "small_capital_deployment": {
                    "tiers": {
                        "inr_50k": {"readiness": "CONDITIONAL", "deployment_verdict": "NO"},
                        "inr_1l": {"readiness": "CONDITIONAL", "deployment_verdict": "NO"},
                        "inr_2l": {"readiness": "CONDITIONAL", "deployment_verdict": "NO"},
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "extended_trade_level_truth_audit.json").write_text(
        json.dumps(
            {
                "replay_start_date": "2025-07-11",
                "replay_end_date": "2026-07-02",
                "core_metrics_by_window": {
                    "240": {
                        "buy_v3": {
                            "win_rate_pct": 48.29,
                            "profit_factor": 1.51,
                            "expectancy": 38.53,
                            "max_drawdown_points": 10996.65,
                        },
                        "sell_v6": {
                            "win_rate_pct": 63.85,
                            "profit_factor": 2.47,
                            "expectancy": 73.1,
                            "max_drawdown_points": 7208.7,
                        },
                        "combined": {
                            "win_rate_pct": 59.63,
                            "profit_factor": 2.13,
                            "expectancy": 63.72,
                            "max_drawdown_points": 17151.4,
                        },
                        "combined_regime_throttle": {
                            "win_rate_pct": 69.01,
                            "profit_factor": 5.58,
                            "expectancy": 108.05,
                            "max_drawdown_points": 2424.26,
                            "signals_per_month": 63.89,
                        },
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "buy_v4_sell_v7_actual_replay_validation.json").write_text(
        json.dumps(
            {
                "final_answer": {
                    "should_buy_v4_replace_buy_v3": "NO",
                    "should_sell_v7_replace_sell_v6": "NO",
                    "best_buy_engine": "BUY_V3",
                    "best_sell_engine": "SELL_V6",
                    "best_stop": "fixed_10",
                    "best_exit_structure": "60/100/Runner",
                },
            },
        ),
        encoding="utf-8",
    )

    import src.research.live_deployment_validation_framework_research as mod

    monkeypatch.setattr(mod, "RESEARCH_DIR", research_dir)
    monkeypatch.setattr(
        mod,
        "SOURCE_EXPORTS",
        {name: research_dir / path.name for name, path in mod.SOURCE_EXPORTS.items()},
    )

    out_path = research_dir / "live_deployment_validation_framework.json"
    research = LiveDeploymentValidationFrameworkResearch(report_path=out_path)
    report = research.run()
    exported = research.export(report)

    assert exported.exists()
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "live_deployment_validation_framework"
    assert payload["stack_locked"]["buy_engine"] == "BUY_V3"
    assert payload["final_answer"]["real_capital_deployment_ready"] == "NO"
    assert payload["final_answer"]["paper_trading_verdict"] == "CONDITIONAL"
    assert payload["capital_tier_readiness"]["overall"].startswith("NO")
    assert len(payload["evidence_required_before_real_capital"]) >= 8
    assert len(payload["risk_controls"]["kill_switch_conditions"]) >= 10
    assert payload["checklist_20_session"]["duration_sessions"] == 20
    assert payload["checklist_40_session"]["duration_sessions"] == 40
    assert "slippage" in payload["measurement_definitions"]
    assert payload["replay_baselines"]["combined_regime_throttle"]["profit_factor"] == 5.58
