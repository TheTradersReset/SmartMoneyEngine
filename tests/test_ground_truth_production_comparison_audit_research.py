"""Tests for ground truth production comparison audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.ground_truth_production_comparison_audit_research import (
    GroundTruthProductionComparisonAuditError,
    GroundTruthProductionComparisonAuditResearch,
    generate_ground_truth_production_comparison_audit_report,
)


def _minimal_sources() -> dict:
    return {
        "extended_trade_level_truth_audit": {
            "replay_windows": [240],
            "available_trading_days": 247,
            "replay_start_date": "2025-07-11",
            "replay_end_date": "2026-07-02",
            "core_metrics_by_window": {
                "240": {
                    "buy_v3": {
                        "signals_emitted": 234,
                        "win_rate_pct": 48.29,
                        "profit_factor": 1.51,
                        "expectancy": 38.53,
                        "capture_efficiency_pct": 37.37,
                        "max_drawdown_points": 10996.65,
                        "recovery_factor": 0.82,
                    },
                    "sell_v6": {
                        "signals_emitted": 628,
                        "win_rate_pct": 63.85,
                        "profit_factor": 2.47,
                        "expectancy": 73.1,
                        "capture_efficiency_pct": 37.77,
                        "max_drawdown_points": 7208.7,
                        "recovery_factor": 6.37,
                    },
                }
            },
            "target_achievement_matrix": {
                "240": {
                    "buy_v3": {
                        "by_tier": {
                            "20": {"count": 200, "probability_pct": 85.0},
                            "40": {"count": 180, "probability_pct": 77.0},
                            "60": {"count": 170, "probability_pct": 73.0},
                            "80": {"count": 140, "probability_pct": 60.0},
                            "100": {"count": 120, "probability_pct": 51.0},
                            "150": {"count": 100, "probability_pct": 43.0},
                            "200": {"count": 70, "probability_pct": 30.0},
                            "300": {"count": 40, "probability_pct": 17.0},
                        }
                    },
                    "sell_v6": {
                        "by_tier": {
                            "20": {"count": 590, "probability_pct": 94.0},
                            "40": {"count": 560, "probability_pct": 89.0},
                            "60": {"count": 520, "probability_pct": 83.0},
                            "80": {"count": 460, "probability_pct": 73.0},
                            "100": {"count": 410, "probability_pct": 65.0},
                            "150": {"count": 310, "probability_pct": 49.0},
                            "200": {"count": 210, "probability_pct": 33.0},
                            "300": {"count": 100, "probability_pct": 16.0},
                        }
                    },
                }
            },
            "trade_lifecycle_analysis": {
                "240": {
                    "buy_v3": {
                        "by_outcome": {
                            "Stopped Out": {"percentage_pct": 26.5},
                            "T1 Only": {"percentage_pct": 18.0},
                        }
                    },
                    "sell_v6": {
                        "by_outcome": {
                            "Stopped Out": {"percentage_pct": 17.0},
                            "T1 Only": {"percentage_pct": 18.0},
                        }
                    },
                }
            },
            "entry_precision_audit": {
                "240": {
                    "buy_v3": {
                        "timing_class_summary": {
                            "Very Early": {"count": 100, "pct": 50.0, "avg_lead_bars": 20},
                            "Early": {"count": 20, "pct": 10.0, "avg_lead_bars": 3},
                            "Same": {"count": 10, "pct": 5.0, "avg_lead_bars": 1},
                            "Late": {"count": 0, "pct": 0.0},
                        },
                        "predictive_vs_reactive": {"predictive_pct": 60.0, "reactive_pct": 40.0},
                    },
                    "sell_v6": {
                        "timing_class_summary": {
                            "Very Early": {"count": 400, "pct": 64.0, "avg_lead_bars": 25},
                        },
                        "predictive_vs_reactive": {"predictive_pct": 70.0, "reactive_pct": 30.0},
                    },
                }
            },
            "uncaptured_edge": {
                "max_window": {
                    "buy_v3": {
                        "current": {"expectancy": 61.0, "capture_efficiency_pct": 37.0},
                        "theoretical_maximum": {"uncaptured_points": 25000.0},
                    },
                    "sell_v6": {
                        "current": {"expectancy": 75.0, "capture_efficiency_pct": 38.0},
                        "theoretical_maximum": {"uncaptured_points": 50000.0},
                    },
                }
            },
            "final_answer": {
                "stop_loss_validation": {"buy_v3": {"best_stop_variant": "fixed_10"}},
                "runner_validation": {"production_strategy": "60_100_runner"},
            },
        },
        "buy_v3_candidate_validation": {
            "trading_days_replayed": 120,
            "replay_start_date": "2026-01-05",
            "replay_end_date": "2026-07-02",
        },
        "sell_v6_replay_validation": {
            "trading_days_replayed": 120,
            "replay_start_date": "2026-01-05",
            "replay_end_date": "2026-07-02",
        },
        "buy_v4_sell_v7_design_blueprint_audit": {
            "buy_v4_design": {"selected_patterns": ["Liquidity Sweep Failure"]},
            "sell_v7_design": {"selected_patterns": ["Volatility Collapse"]},
        },
        "buy_v4_sell_v7_final_production_validation": {
            "methodology": {"signal_source": "filtered V3/V6 signals"},
            "core_metrics_by_window": {
                "240": {
                    "buy_v3": {
                        "signals_emitted": 234,
                        "win_rate_pct": 30.34,
                        "profit_factor": 1.51,
                        "expectancy": 38.53,
                        "capture_pct": 66.41,
                        "max_drawdown_points": 10996.65,
                        "recovery_factor": 0.82,
                        "average_mfe": 171.14,
                        "average_mae": 140.68,
                        "median_mfe": 119.53,
                        "median_mae": 120.67,
                    },
                    "buy_v4": {
                        "signals_emitted": 118,
                        "win_rate_pct": 60.17,
                        "profit_factor": 93.16,
                        "expectancy": 216.54,
                        "capture_pct": 77.26,
                        "max_drawdown_points": 117.05,
                        "recovery_factor": 218.29,
                        "average_mfe": 283.33,
                        "average_mae": 53.14,
                        "median_mfe": 215.43,
                        "median_mae": 45.0,
                    },
                    "sell_v6": {
                        "signals_emitted": 628,
                        "win_rate_pct": 63.85,
                        "profit_factor": 2.47,
                        "expectancy": 73.1,
                        "capture_pct": 59.7,
                        "max_drawdown_points": 7208.7,
                        "recovery_factor": 6.37,
                        "average_mfe": 205.48,
                        "average_mae": 97.71,
                        "median_mfe": 151.77,
                        "median_mae": 60.42,
                    },
                    "sell_v7": {
                        "signals_emitted": 434,
                        "win_rate_pct": 86.41,
                        "profit_factor": 23.99,
                        "expectancy": 167.07,
                        "capture_pct": 64.14,
                        "max_drawdown_points": 802.2,
                        "recovery_factor": 90.39,
                        "average_mfe": 271.8,
                        "average_mae": 44.42,
                        "median_mfe": 195.32,
                        "median_mae": 36.88,
                    },
                }
            },
            "trade_outcome_distribution": {
                "240": {
                    "buy_v4": {
                        "by_tier": {
                            "20": {"count": 118, "probability_pct": 100.0},
                            "40": {"count": 118, "probability_pct": 100.0},
                            "60": {"count": 118, "probability_pct": 100.0},
                            "80": {"count": 118, "probability_pct": 100.0},
                            "100": {"count": 114, "probability_pct": 96.6},
                            "150": {"count": 96, "probability_pct": 81.4},
                            "200": {"count": 70, "probability_pct": 59.3},
                            "300": {"count": 40, "probability_pct": 33.9},
                        }
                    },
                    "sell_v7": {
                        "by_tier": {
                            "20": {"count": 434, "probability_pct": 100.0},
                            "40": {"count": 434, "probability_pct": 100.0},
                            "60": {"count": 432, "probability_pct": 99.5},
                            "80": {"count": 420, "probability_pct": 96.8},
                            "100": {"count": 400, "probability_pct": 92.2},
                            "150": {"count": 300, "probability_pct": 69.1},
                            "200": {"count": 200, "probability_pct": 46.1},
                            "300": {"count": 100, "probability_pct": 23.0},
                        }
                    },
                }
            },
            "target_path_analysis": {
                "buy_v3": {"target_path_tree": {"probabilities_pct": {"stop": 26.5, "t1": 73.5}}},
                "buy_v4": {"target_path_tree": {"probabilities_pct": {"stop": 9.3, "t1": 100.0}}},
                "sell_v6": {"target_path_tree": {"probabilities_pct": {"stop": 17.0, "t1": 83.0}}},
                "sell_v7": {"target_path_tree": {"probabilities_pct": {"stop": 5.0, "t1": 99.0}}},
                "target_matrices": {},
            },
            "trade_lifecycle_audit": {
                "buy_v3": {"hit_probabilities_pct": {"Stopped Out": 26.5, "Hit T1": 73.5}},
                "buy_v4": {"hit_probabilities_pct": {"Stopped Out": 0.0, "Hit T1": 100.0}},
                "sell_v6": {"hit_probabilities_pct": {"Stopped Out": 17.0, "Hit T1": 83.0}},
                "sell_v7": {"hit_probabilities_pct": {"Stopped Out": 5.0, "Hit T1": 99.0}},
            },
            "signal_timing_reality": {
                "buy_v3": {
                    "timing_class_metrics": {"Very Early": {"count": 100, "pct": 50.0}},
                    "average_lead_bars": 26.0,
                    "median_lead_bars": 19.0,
                    "average_lead_minutes": 130.0,
                    "median_lead_minutes": 95.0,
                    "predictive_signal_share_pct": 62.0,
                    "predictive_vs_reactive": "predictive",
                },
                "buy_v4": {
                    "timing_class_metrics": {"Very Early": {"count": 80, "pct": 70.0}},
                    "average_lead_bars": 30.0,
                    "average_lead_minutes": 150.0,
                    "predictive_signal_share_pct": 80.0,
                },
                "sell_v6": {
                    "timing_class_metrics": {"Very Early": {"count": 400, "pct": 64.0}},
                    "average_lead_bars": 28.0,
                    "average_lead_minutes": 140.0,
                    "predictive_signal_share_pct": 70.0,
                },
                "sell_v7": {
                    "timing_class_metrics": {"Very Early": {"count": 350, "pct": 80.0}},
                    "average_lead_bars": 32.0,
                    "average_lead_minutes": 160.0,
                    "predictive_signal_share_pct": 85.0,
                },
            },
            "entry_quality_analysis": {
                "buy_v3": {"average_entry_loss_points": 80.0},
                "buy_v4": {"average_entry_loss_points": 40.0},
                "sell_v6": {"average_entry_loss_points": 50.0},
                "sell_v7": {"average_entry_loss_points": 30.0},
            },
            "reward_risk_reality": {
                "buy_v3": {
                    "average_stop_points": 10.0,
                    "average_rr": 2.69,
                    "median_rr": 1.23,
                    "rr_probability": {"1_to_1": 57.0, "1_to_2": 33.0, "1_to_3": 19.0, "1_to_5": 8.0},
                },
                "buy_v4": {
                    "average_stop_points": 10.0,
                    "average_rr": 5.0,
                    "median_rr": 3.0,
                    "rr_probability": {"1_to_1": 90.0, "1_to_2": 70.0, "1_to_3": 50.0, "1_to_5": 30.0},
                },
                "sell_v6": {
                    "average_stop_points": 10.0,
                    "average_rr": 3.1,
                    "median_rr": 1.5,
                    "rr_probability": {"1_to_1": 65.0, "1_to_2": 40.0, "1_to_3": 25.0, "1_to_5": 12.0},
                },
                "sell_v7": {
                    "average_stop_points": 10.0,
                    "average_rr": 6.0,
                    "median_rr": 4.0,
                    "rr_probability": {"1_to_1": 95.0, "1_to_2": 80.0, "1_to_3": 60.0, "1_to_5": 35.0},
                },
            },
            "final_production_decision": {
                "best_buy_engine": "BUY_V4",
                "best_sell_engine": "SELL_V7",
                "best_stop_structure": "fixed_10",
                "best_target_structure": "60/100/Runner",
            },
            "final_answer": {
                "should_buy_v4_replace_buy_v3": "YES",
                "should_sell_v7_replace_sell_v6": "YES",
                "readiness": {
                    "paper_trading_readiness": "YES",
                    "small_capital_readiness": "CONDITIONAL",
                    "full_production_readiness": "NO",
                },
            },
        },
        "research_integrity_ground_truth_validation_audit": {
            "buy_v4_validation_audit": {"validation_method": "B) Signal Filtering", "method_code": "B"},
            "sell_v7_validation_audit": {"validation_method": "B) Signal Filtering", "method_code": "B"},
            "replacement_sufficiency": {
                "sufficient_to_replace_buy_v3_with_buy_v4": False,
                "sufficient_to_replace_sell_v6_with_sell_v7": False,
            },
            "production_evidence_audit": {
                "conclusions": [
                    {
                        "conclusion": "BUY_V4 should replace BUY_V3 in production",
                        "status": "UNPROVEN",
                        "basis": "filter only",
                    }
                ]
            },
            "final_answer": {
                "can_buy_v4_replace_buy_v3": "NO",
                "buy_v4_confidence_pct": 45.0,
                "buy_v4_evidence_pct": 55.0,
                "can_sell_v7_replace_sell_v6": "NO",
                "sell_v7_confidence_pct": 45.0,
                "sell_v7_evidence_pct": 55.0,
                "exact_evidence_still_missing": [
                    "Dedicated BUY_V4 bar-by-bar replay with filters inside the emission path",
                    "Dedicated SELL_V7 bar-by-bar replay with filters inside the emission path",
                ],
                "exact_replay_still_required": [
                    "BUY_V3 vs BUY_V4 head-to-head replay",
                    "SELL_V6 vs SELL_V7 head-to-head replay",
                ],
            },
        },
        "extended_evidence_validation_real_deployment_audit": {
            "final_answer": {"definitive_verdict": "Paper", "evidence_score": 81.1},
            "production_config": {
                "buy_engine": "LDM-BUY-V3",
                "sell_engine": "LDM-SELL-V6",
                "exit_structure": "60/100/Runner",
                "buy_stop": "fixed_10",
                "sell_stop": "fixed_10",
                "regime_throttle": {"enabled": True},
            },
        },
    }


def test_comparison_audit_mocked(tmp_path: Path) -> None:
    research = GroundTruthProductionComparisonAuditResearch()
    report = research.run(_minimal_sources())
    out = tmp_path / "ground_truth_production_comparison_audit.json"
    research.export(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["report_type"] == "Ground Truth Production Comparison Audit"
    assert payload["final_answer"]["can_buy_v4_replace_buy_v3"] == "NO"
    assert payload["final_answer"]["can_sell_v7_replace_sell_v6"] == "NO"
    assert payload["real_replay_evidence_check"]["buy_v4_dedicated_replay"] == "NO"
    assert payload["real_replay_evidence_check"]["sell_v7_dedicated_replay"] == "NO"
    assert payload["engine_comparison"]["buy_v3"]["profit_factor"]["provenance"] == "Measured"
    assert payload["engine_comparison"]["buy_v4"]["profit_factor"]["provenance"] == "UNPROVEN"
    assert payload["engine_comparison"]["sell_v6"]["profit_factor"]["provenance"] == "Measured"
    assert payload["engine_comparison"]["sell_v7"]["profit_factor"]["provenance"] == "UNPROVEN"
    assert payload["best_production_picks"]["best_buy_engine"] == "BUY_V3"
    assert payload["best_production_picks"]["best_sell_engine"] == "SELL_V6"
    assert "60" in payload["target_achievement_matrix"]["buy_v3"]["by_tier"]
    assert payload["target_achievement_matrix"]["buy_v4"]["provenance"] == "UNPROVEN"
    assert "Paper Trading" in payload["capital_verdicts"]
    assert payload["capital_verdicts"]["Full Production"]["verdict"] == "NO"
    assert payload["reward_risk_analysis"]["buy_v3"]["probability_1_to_1"]["provenance"] == "Measured"
    assert any(
        c.get("status") == "UNPROVEN"
        for c in payload["conclusion_classifications"]
        if "BUY_V4" in str(c.get("conclusion", ""))
    )


def test_generate_raises_without_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import ground_truth_production_comparison_audit_research as mod

    monkeypatch.setattr(
        mod,
        "REQUIRED_EXPORTS",
        {"extended_trade_level_truth_audit": tmp_path / "missing.json"},
    )
    with pytest.raises(GroundTruthProductionComparisonAuditError):
        generate_ground_truth_production_comparison_audit_report(report_path=tmp_path / "out.json")
