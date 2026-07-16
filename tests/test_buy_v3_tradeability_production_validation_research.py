"""Tests for BUY_V3 tradeability production validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.buy_v3_tradeability_production_validation_research import (
    BUY_V3_MODEL_ID,
    EXIT_TARGET_TIERS,
    TRADEABILITY_TIERS,
    BuyV3TradeabilityProductionValidationResearch,
    _exit_target_optimization,
    _fixed_target_pnl,
    _pre_expansion_tradeable,
    _tier_hit,
)


def test_constants() -> None:
    assert BUY_V3_MODEL_ID == "LDM-BUY-V3"
    assert 40 in TRADEABILITY_TIERS
    assert 60 in EXIT_TARGET_TIERS


def test_tier_hit_and_pre_expansion() -> None:
    signal = {"mfe_points": 45.0, "bars_before_expansion": 5}
    assert _tier_hit(signal, 40) is True
    assert _tier_hit(signal, 60) is False
    assert _pre_expansion_tradeable(signal, 40) is True
    assert _pre_expansion_tradeable({"mfe_points": 45.0, "bars_before_expansion": -1}, 40) is False


def test_fixed_target_pnl() -> None:
    win, pnl = _fixed_target_pnl({"mfe_points": 50.0, "mae_points": 20.0}, 40)
    assert win is True
    assert pnl == 40.0
    loss_win, loss_pnl = _fixed_target_pnl({"mfe_points": 10.0, "mae_points": 25.0}, 40)
    assert loss_win is False
    assert loss_pnl == -25.0


def test_exit_target_optimization() -> None:
    signals = [
        {"mfe_points": 80.0, "mae_points": 20.0, "trade_duration_bars": 40},
        {"mfe_points": 30.0, "mae_points": 15.0, "trade_duration_bars": 40},
        {"mfe_points": 100.0, "mae_points": 10.0, "trade_duration_bars": 30},
    ]
    result = _exit_target_optimization(signals)
    assert result["optimal_target_points"] in EXIT_TARGET_TIERS
    assert "by_target" in result


def test_generate_report(tmp_path: Path) -> None:
    v3_export = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "trading_days_replayed": 120,
        "replay_start_date": "2026-01-05",
        "replay_end_date": "2026-07-02",
        "methodology": {"no_lookahead": True, "actual_replay": True},
        "comparison": {
            "buy_v1": {
                "overall_statistics": {
                    "signals_emitted": 10,
                    "signals_per_month": 2.0,
                    "win_rate_pct": 60.0,
                    "profit_factor": 1.5,
                    "expectancy": 10.0,
                },
                "point_capture": {"40": {"capture_rate_pct": 5.0}},
                "classification_summary": {"real_reversal_rate_pct": 40.0, "false_reversal_rate_pct": 30.0},
            },
            "buy_v3": {
                "overall_statistics": {
                    "signals_emitted": 2,
                    "signals_per_month": 22.0,
                    "win_rate_pct": 70.0,
                    "profit_factor": 3.0,
                    "expectancy": 50.0,
                    "average_mfe": 80.0,
                    "average_mae": 20.0,
                },
                "point_capture": {"40": {"capture_rate_pct": 10.0}, "60": {"capture_rate_pct": 8.0}},
                "classification_summary": {
                    "real_reversal_rate_pct": 50.0,
                    "false_reversal_rate_pct": 20.0,
                },
            },
            "sell_v5_benchmark": {
                "signals_emitted": 100,
                "signals_per_month": 50.0,
                "win_rate_pct": 68.0,
                "profit_factor": 3.0,
                "expectancy": 100.0,
                "capture_40_plus_pct": 55.0,
            },
        },
        "production_safety_check": {
            "buy_v3": {
                "win_rate_above_65_pct": True,
                "profit_factor_above_2": True,
                "signals_per_month_20_plus": True,
                "capture_40_plus": True,
                "all_pass": True,
            },
        },
        "walk_forward": {
            "train": {"buy_v3": {"overall_statistics": {"win_rate_pct": 70.0, "profit_factor": 3.0}}},
            "validate": {"buy_v3": {"overall_statistics": {"win_rate_pct": 65.0, "profit_factor": 2.5}}},
        },
        "signal_timing": {
            "buy_v3": {"before_expansion_pct": 95.0, "lead_time_bars": {"avg": 20.0}},
        },
        "tradeability": {
            "buy_v3": {
                "by_threshold": {
                    "40": {"horizons": {"2_trading_days": {"captured_moves": 5}}},
                },
            },
        },
        "false_reversal_removal": {"removed_by_buy_v3": 10, "baseline_false_reversal_count": 10},
        "per_signal_details": {
            "buy_v3": [
                {
                    "timestamp": "2026-01-10 10:00:00+05:30",
                    "move_start_time": "2026-01-10 12:00:00+05:30",
                    "bars_before_expansion": 10,
                    "points_before_expansion": 15.0,
                    "signal_before_expansion": True,
                    "mfe_points": 80.0,
                    "mae_points": 20.0,
                    "trade_duration_bars": 40,
                    "classification": "Real Reversal",
                    "win": True,
                    "realized_pnl_points": 60.0,
                },
                {
                    "timestamp": "2026-01-11 10:00:00+05:30",
                    "move_start_time": "2026-01-11 11:00:00+05:30",
                    "bars_before_expansion": 5,
                    "points_before_expansion": 8.0,
                    "signal_before_expansion": True,
                    "mfe_points": 35.0,
                    "mae_points": 15.0,
                    "trade_duration_bars": 30,
                    "classification": "Bull Trap",
                    "win": False,
                    "realized_pnl_points": -15.0,
                },
            ],
        },
    }
    v5_export = {
        "comparison": {
            "v5_candidate": {
                "overall_statistics": {
                    "signals_emitted": 100,
                    "signals_per_month": 50.0,
                    "win_rate_pct": 68.0,
                    "profit_factor": 3.0,
                    "expectancy": 100.0,
                },
            },
        },
        "missed_move_recovery": {"recovered_move_details": []},
    }

    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)
    (research_dir / "buy_v3_candidate_validation.json").write_text(
        json.dumps(v3_export),
        encoding="utf-8",
    )
    (research_dir / "smartmoneyengine_v5_candidate_validation.json").write_text(
        json.dumps(v5_export),
        encoding="utf-8",
    )

    report_path = research_dir / "buy_v3_tradeability_production_validation.json"
    research = BuyV3TradeabilityProductionValidationResearch(report_path=report_path)

    import src.research.buy_v3_tradeability_production_validation_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "buy_v2_candidate_validation": research_dir / "buy_v2_candidate_validation.json",
            "smartmoneyengine_v5_candidate_validation": research_dir
            / "smartmoneyengine_v5_candidate_validation.json",
            "tradeable_move_validation": research_dir / "tradeable_move_validation.json",
            "buy_winner_vs_false_reversal_analysis": research_dir
            / "buy_winner_vs_false_reversal_analysis.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["model_id"] == "LDM-BUY-V3"
    assert len(payload["per_signal_tradeability"]) == 2
    assert payload["final_answers"]["buy_v3_suitable_for_practical_intraday"] in {"YES", "NO", "PARTIAL"}
    assert payload["final_answers"]["optimal_target_tier_points"] in EXIT_TARGET_TIERS
    assert payload["production_gates_validation"]["checks"]["replay_validated"] is True
    assert "tradeability_tier_metrics" in payload
    assert "combined_engine_simulation" in payload


def test_generate_report_missing_v3_export(tmp_path: Path) -> None:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)
    report_path = research_dir / "buy_v3_tradeability_production_validation.json"
    research = BuyV3TradeabilityProductionValidationResearch(report_path=report_path)

    import src.research.buy_v3_tradeability_production_validation_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "buy_v3_candidate_validation": research_dir / "missing.json",
            "buy_v2_candidate_validation": research_dir / "buy_v2_candidate_validation.json",
            "smartmoneyengine_v5_candidate_validation": research_dir
            / "smartmoneyengine_v5_candidate_validation.json",
            "tradeable_move_validation": research_dir / "tradeable_move_validation.json",
            "buy_winner_vs_false_reversal_analysis": research_dir
            / "buy_winner_vs_false_reversal_analysis.json",
        }
        with pytest.raises(Exception):
            research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports
