"""Tests for trade level truth audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.trade_level_truth_audit_research import (
    CONDITIONAL_TIERS,
    LIFECYCLE_OUTCOMES,
    MFE_TIERS,
    PF_IMPROVEMENT_THRESHOLD_PCT,
    TradeLevelTruthAuditResearch,
    _classify_lifecycle_outcome,
    _conditional_probability_analysis,
    _entry_precision_audit,
    _classify_buy_loser,
    _per_signal_record,
    _tier_reached,
    _trade_level_target_matrix,
    _trade_lifecycle_analysis,
)


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-06 09:30:00+05:30",
        "move_start_time": "2026-01-06 09:15:00+05:30",
        "direction": "BUY",
        "entry": 23500.0,
        "stop_loss": 23450.0,
        "target_1": 23550.0,
        "target_2": 23600.0,
        "target_3": 23650.0,
        "bars_before_expansion": 10,
        "points_before_expansion": 12.5,
        "mfe_points": 80.0,
        "mae_points": 20.0,
        "trade_duration_bars": 40,
        "hit_1r": True,
        "hit_2r": True,
        "hit_3r": False,
        "win": True,
        "classification": "Real Reversal",
        "realized_pnl_points": 60.0,
        "signal_reason_stack": {
            "layer1": ["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep"],
            "layer2": {"htf_trend": "Bullish", "vwap": "Reclaimed", "location": "Near Support"},
        },
        "layers": {
            "layer1": {
                "events_detected": ["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep"],
            },
            "layer2": {
                "htf_trend": "Bullish",
                "vwap_state": "Reclaimed",
                "location": "Near Support",
                "aligned": True,
            },
        },
    }
    base.update(overrides)
    return base


def _sell_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 10:25:00+05:30",
        "move_start_time": "2026-01-05 10:10:00+05:30",
        "direction": "SELL",
        "entry": 23600.0,
        "stop_loss": 23650.0,
        "target_1": 23550.0,
        "target_2": 23500.0,
        "target_3": 23450.0,
        "bars_before_expansion": 5,
        "mfe_points": 120.0,
        "mae_points": 40.0,
        "trade_duration_bars": 35,
        "mfe_capture_tiers": {"40": True, "60": True, "100": True, "200": False},
        "hit_1r": True,
        "hit_2r": True,
        "hit_3r": True,
        "win": True,
        "realized_pnl_points": 80.0,
        "classification": "Winner",
        "layers": {
            "layer1": {"events_detected": ["Failed Breakout"], "primary_event": "Failed Breakout"},
            "layer2": {
                "htf_trend": "Bearish",
                "vwap_state": "Below",
                "vwap_gate_rule": "VWAP Below only",
                "aligned": True,
            },
        },
        "signal_reason_stack": {
            "layer1": ["Failed Breakout"],
            "layer2": {"htf_trend": "Bearish", "vwap": "Below"},
        },
    }
    base.update(overrides)
    return base


def test_constants() -> None:
    assert 40 in MFE_TIERS
    assert 40 in CONDITIONAL_TIERS
    assert "Stopped Out" in LIFECYCLE_OUTCOMES
    assert PF_IMPROVEMENT_THRESHOLD_PCT == 10.0


def test_tier_reached() -> None:
    assert _tier_reached(_buy_signal(mfe_points=80.0), 60) is True
    assert _tier_reached(_buy_signal(mfe_points=30.0), 60) is False
    assert _tier_reached(_sell_signal(), 40) is True
    assert _tier_reached(_sell_signal(mfe_capture_tiers={"40": False, "60": False}), 40) is False


def test_classify_lifecycle_outcome() -> None:
    from src.research.production_reality_audit_research import RUNNER_STRATEGIES

    structure = RUNNER_STRATEGIES["60_100_runner"]
    stopped = _classify_lifecycle_outcome(
        _buy_signal(mfe_points=30.0, mae_points=15.0),
        structure=structure,
        stop_pts=10.0,
        pnl=-10.0,
    )
    assert stopped == "Stopped Out"
    runner = _classify_lifecycle_outcome(
        _sell_signal(mfe_points=150.0, mae_points=20.0),
        structure=structure,
        stop_pts=10.0,
        pnl=80.0,
    )
    assert runner in {"Runner", "Full Trend Capture", "T2 Only"}


def test_trade_level_target_matrix() -> None:
    from src.research.production_reality_audit_research import RUNNER_STRATEGIES

    result = _trade_level_target_matrix(
        [_buy_signal(), _buy_signal(mfe_points=150.0)],
        side="BUY",
        structure=RUNNER_STRATEGIES["60_100_runner"],
        stop_variant="fixed_10",
    )
    assert result["sample_size"] == 2
    assert result["by_tier"]["40"]["count"] == 2
    assert "avg_failure_rate_pct" in result["by_tier"]["40"]


def test_conditional_probability_analysis() -> None:
    from src.research.production_reality_audit_research import RUNNER_STRATEGIES

    result = _conditional_probability_analysis(
        [_buy_signal(), _buy_signal(mfe_points=20.0, mae_points=50.0)],
        side="BUY",
        structure=RUNNER_STRATEGIES["60_100_runner"],
        stop_variant="fixed_10",
    )
    assert "40" in result["tiers"]
    assert result["summary"]["p_40_plus"] > 0


def test_trade_lifecycle_analysis() -> None:
    from src.research.production_edge_enhancement_audit_research import _is_buy_winner
    from src.research.production_reality_audit_research import RUNNER_STRATEGIES

    result = _trade_lifecycle_analysis(
        [_buy_signal(), _buy_signal(mfe_points=30.0, mae_points=15.0)],
        side="BUY",
        structure=RUNNER_STRATEGIES["60_100_runner"],
        stop_variant="fixed_10",
        win_fn=_is_buy_winner,
    )
    assert result["sample_size"] == 2
    assert "Stopped Out" in result["by_outcome"]
    assert result["aggregate"]["capture_efficiency_pct"] >= 0


def test_entry_precision_audit() -> None:
    from src.research.production_edge_enhancement_audit_research import _is_buy_winner

    result = _entry_precision_audit(
        [_buy_signal(), _buy_signal(bars_before_expansion=-1)],
        side="BUY",
        win_fn=_is_buy_winner,
    )
    assert result["timing_class_metrics"]["Very Early"]["count"] == 1
    assert result["timing_class_metrics"]["Late"]["count"] == 1
    assert "predictive_vs_reactive" in result


def test_per_signal_record() -> None:
    from src.research.production_edge_enhancement_audit_research import _is_buy_winner
    from src.research.production_reality_audit_research import RUNNER_STRATEGIES

    record = _per_signal_record(
        _buy_signal(),
        side="BUY",
        structure=RUNNER_STRATEGIES["60_100_runner"],
        stop_variant="fixed_10",
        cohort_mae_median=20.0,
        win_fn=_is_buy_winner,
        classify_fn=_classify_buy_loser,
    )
    assert record["signal_timestamp"] is not None
    assert record["entry"] == 23500.0
    assert record["mfe"] == 80.0
    assert record["timing_class"] == "Very Early"


@pytest.fixture
def tmp_research_dir(tmp_path: Path) -> Path:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)

    buy_signals = [_buy_signal(), _buy_signal(mfe_points=150.0, bars_before_expansion=3)]
    sell_signals = [_sell_signal(), _sell_signal(mfe_points=200.0, bars_before_expansion=8)]

    (research_dir / "buy_v3_candidate_validation.json").write_text(
        json.dumps(
            {
                "symbol": "NIFTY50",
                "timeframe": "5M",
                "trading_days_replayed": 120,
                "replay_start_date": "2026-01-05",
                "replay_end_date": "2026-07-02",
                "per_signal_details": {"buy_v3": buy_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "sell_v6_replay_validation.json").write_text(
        json.dumps(
            {
                "symbol": "NIFTY50",
                "timeframe": "5M",
                "trading_days_replayed": 120,
                "replay_start_date": "2026-01-05",
                "replay_end_date": "2026-07-02",
                "per_signal_details": {"sell_v6": sell_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "live_trade_management_execution_efficiency_audit.json").write_text(
        json.dumps(
            {
                "methodology": {"synthesis_only": True},
                "capture_leakage": {
                    "buy_v3": {"miss_reason_ranking": [{"reason": "timing", "count": 1, "rank": 1}]},
                    "sell_v6": {"miss_reason_ranking": [{"reason": "runner", "count": 1, "rank": 1}]},
                },
                "final_answer": {
                    "optimal_stops": {"buy_v3": "fixed_10", "sell_v6": "fixed_10"},
                    "optimal_exit_structures": {"buy_v3": "60/100/Runner", "sell_v6": "60/100/Runner"},
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "production_reality_audit.json").write_text(
        json.dumps(
            {
                "methodology": {"synthesis_only": True},
                "signal_reality": {
                    "buy_v3": {"predictive_vs_reactive": {"verdict": "PREDICTIVE"}},
                    "sell_v6": {"predictive_vs_reactive": {"verdict": "PREDICTIVE"}},
                },
                "target_achievement_matrix": {},
            },
        ),
        encoding="utf-8",
    )
    return research_dir


def test_export_synthetic(tmp_research_dir: Path, tmp_path: Path) -> None:
    report_path = tmp_research_dir / "trade_level_truth_audit.json"
    research = TradeLevelTruthAuditResearch(report_path=report_path)

    import src.research.trade_level_truth_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = tmp_research_dir
        module.SOURCE_EXPORTS = {
            "buy_v3_candidate_validation": tmp_research_dir / "buy_v3_candidate_validation.json",
            "sell_v6_replay_validation": tmp_research_dir / "sell_v6_replay_validation.json",
            "live_trade_management_execution_efficiency_audit": tmp_research_dir
            / "live_trade_management_execution_efficiency_audit.json",
            "production_reality_audit": tmp_research_dir / "production_reality_audit.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Trade Level Truth Audit"
    assert payload["methodology"]["research_only"] is True
    assert "per_signal_records" in payload
    assert "target_achievement_matrix" in payload
    assert "conditional_probability" in payload
    assert "trade_lifecycle_analysis" in payload
    assert "entry_precision_audit" in payload
    assert "buy_v4_sell_v7_potential" in payload
    assert "uncaptured_edge" in payload
    final = payload["final_answer"]
    assert final["buy_v4_recommendation"] in {"YES", "NO"}
    assert final["sell_v7_recommendation"] in {"YES", "NO"}
    assert "probability_matrix" in final
    assert "trade_lifecycle_matrix" in final
    assert "entry_quality_matrix" in final
    assert "maximum_theoretical_improvement" in final
    assert "deployment_readiness" not in json.dumps(payload).lower()
    assert len(payload["conclusions"]) >= 5


@pytest.mark.skipif(
    not Path("outputs/research/buy_v3_candidate_validation.json").exists(),
    reason="Requires completed replay exports",
)
def test_export_real_data() -> None:
    research = TradeLevelTruthAuditResearch()
    path = research.export()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["per_signal_records"]["buy_v3"]["sample_size"] >= 100
    assert payload["per_signal_records"]["sell_v6"]["sample_size"] >= 100
    assert payload["final_answer"]["probability_matrix"]["buy_v3"]["p_40_plus"] > 0
