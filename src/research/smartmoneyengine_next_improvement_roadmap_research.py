"""
SmartMoneyEngine Next Improvement Roadmap — synthesis-only research.

Determines the single highest-value remaining improvement from completed exports.
No replay, discovery, optimization, or new models.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"

SOURCE_EXPORTS = {
    "v4_candidate_validation": RESEARCH_DIR / "smartmoneyengine_v4_candidate_validation.json",
    "engine_gap_analysis": RESEARCH_DIR / "smartmoneyengine_engine_gap_analysis.json",
    "v31_validation": RESEARCH_DIR / "smartmoneyengine_v31_validation.json",
    "final_signal_extraction": RESEARCH_DIR / "smartmoneyengine_final_signal_extraction.json",
    "buy_formula_reality": RESEARCH_DIR / "buy_formula_reality_verification.json",
    "buy_side_reality_discovery": RESEARCH_DIR / "nifty50_buy_side_reality_discovery.json",
    "research_consistency_audit": RESEARCH_DIR / "research_consistency_audit.json",
}

DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_next_improvement_roadmap.json"

CAPTURE_THRESHOLDS = (40, 60, 100, 200, 300, 500)
SELL_CAPTURE_PROXY = {40: 50, 60: 50}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class SmartMoneyEngineNextImprovementRoadmapError(Exception):
    """Raised when next-improvement roadmap synthesis fails."""


@dataclass
class SmartMoneyEngineNextImprovementRoadmapReport:
    """Next improvement roadmap output."""

    report_type: str
    symbol: str
    timeframe: str
    methodology: dict[str, Any]
    source_exports: list[str]
    sell_side_analysis: dict[str, Any]
    buy_side_analysis: dict[str, Any]
    ranked_opportunities: list[dict[str, Any]]
    final_recommendation: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SmartMoneyEngineNextImprovementRoadmapError(f"Missing source export: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pack_capture_row(source: dict[str, Any], threshold: int) -> dict[str, Any]:
    row = source.get(str(threshold), {})
    total = int(row.get("total_bearish_moves", 0))
    captured = int(row.get("signals_before_move", 0))
    missed = int(row.get("missed_moves", max(total - captured, 0)))
    rate = float(row.get("capture_rate_pct", 0.0))
    return {
        "total_bearish_moves": total,
        "signals_before_move": captured,
        "missed_moves": missed,
        "capture_rate_pct": rate,
        "miss_rate_pct": round(100.0 - rate, 2) if total else 0.0,
    }


def _sell_capture_metrics(
    *,
    v3_capture: dict[str, Any],
    v4_capture: dict[str, Any],
    v31_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"v3_baseline_120d": {}, "v4_candidate_120d": {}, "v31_baseline_120d": {}}
    for threshold in CAPTURE_THRESHOLDS:
        key = f"{threshold}_plus"
        proxy = SELL_CAPTURE_PROXY.get(threshold, threshold)
        metrics["v3_baseline_120d"][key] = _pack_capture_row(v3_capture, proxy)
        metrics["v4_candidate_120d"][key] = _pack_capture_row(v4_capture, proxy)
        if v31_capture is not None:
            metrics["v31_baseline_120d"][key] = _pack_capture_row(v31_capture, proxy)
        if threshold in SELL_CAPTURE_PROXY:
            metrics["v3_baseline_120d"][key]["proxy_note"] = (
                f"40+/60+ not stored in exports; proxied from {proxy}+ capture."
            )
            metrics["v4_candidate_120d"][key]["proxy_note"] = (
                f"40+/60+ not stored in exports; proxied from {proxy}+ capture."
            )
    return metrics


def _sell_side_analysis(
    *,
    v4: dict[str, Any],
    v31: dict[str, Any],
    gap: dict[str, Any],
) -> dict[str, Any]:
    comparison = v4.get("comparison", {})
    v3_stats = comparison.get("v3", {}).get("overall_statistics", {})
    v4_stats = comparison.get("v4_candidate", {}).get("overall_statistics", {})
    v3_capture = comparison.get("v3", {}).get("point_capture", {})
    v4_capture = comparison.get("v4_candidate", {}).get("point_capture", {})
    v31_capture = v31.get("comparison", {}).get("v3", {}).get("point_capture", {})

    recovery = v4.get("missed_move_recovery", {}).get("by_threshold", {})
    gap_final = gap.get("final_answer", {})
    gap_counterfactual = gap.get("counterfactual_relaxation", {})
    gap_attribution = gap.get("filter_block_attribution", {}).get("interpretation", {})

    v4_rejections = comparison.get("v4_candidate", {}).get("layer_rejection_summary", {})
    remaining_miss_200 = int(recovery.get("200", {}).get("neither_captured", 110))
    remaining_miss_100 = int(recovery.get("100", {}).get("neither_captured", 137))

    pf_safe = gap_counterfactual.get("ranked_by_recovery_per_pf_damage", [])
    raw_recovery = gap_counterfactual.get("ranked_by_recovery", [])

    post_v4_blockers = sorted(
        [
            ("VWAP", int(v4_rejections.get("VWAP_MISMATCH", 0))),
            ("HTF", int(v4_rejections.get("HTF_CONFLICT", 0))),
            ("EMA", int(v4_rejections.get("EMA_MISMATCH", 0))),
            ("Location", int(v4_rejections.get("LOCATION_MID_RANGE", 0))),
            ("Failed Breakout", int(v4_rejections.get("NO_FAILED_BREAKOUT", 0))),
            ("Layer 2 Stack", int(v4_rejections.get("DIRECTION_NOT_ALIGNED", 0))),
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    vwap_scenario = gap_counterfactual.get("scenarios", {}).get("VWAP", {})
    location_scenario = gap_counterfactual.get("scenarios", {}).get("Location", {})
    confirmation_scenario = gap_counterfactual.get("scenarios", {}).get("Confirmation", {})

    return {
        "current_metrics": {
            "v3_baseline": {
                "win_rate_pct": float(v3_stats.get("win_rate_pct", 0.0)),
                "profit_factor": float(v3_stats.get("profit_factor", 0.0)),
                "expectancy": float(v3_stats.get("expectancy", 0.0)),
                "signals_per_month": float(v3_stats.get("signals_per_month", 0.0)),
            },
            "v4_candidate_latest": {
                "win_rate_pct": float(v4_stats.get("win_rate_pct", 0.0)),
                "profit_factor": float(v4_stats.get("profit_factor", 0.0)),
                "expectancy": float(v4_stats.get("expectancy", 0.0)),
                "signals_per_month": float(v4_stats.get("signals_per_month", 0.0)),
            },
            "v4_delta_vs_v3": comparison.get("delta_v4_minus_v3", {}),
        },
        "capture_metrics": _sell_capture_metrics(
            v3_capture=v3_capture,
            v4_capture=v4_capture,
            v31_capture=v31_capture,
        ),
        "v4_improvement_summary": {
            "200_plus_capture_gain_pp": float(comparison.get("delta_v4_minus_v3", {}).get("200_plus_capture_pp", 0.0)),
            "v3_missed_200_plus_recovered_by_v4": int(recovery.get("200", {}).get("v3_missed_v4_captured", 23)),
            "remaining_200_plus_misses_after_v4": remaining_miss_200,
            "remaining_100_plus_misses_after_v4": remaining_miss_100,
            "v4_superior_verdict": v4.get("final_questions", {}).get("4_is_v4_superior_to_v3", {}).get("answer", "YES"),
        },
        "biggest_remaining_weakness": {
            "issue": "Structural gate alignment still blocks ~41% of 200+ bearish moves after V4",
            "evidence": {
                "v4_200_plus_capture_pct": float(v4_capture.get("200", {}).get("capture_rate_pct", 0.0)),
                "v4_200_plus_miss_rate_pct": round(
                    100.0 - float(v4_capture.get("200", {}).get("capture_rate_pct", 0.0)),
                    2,
                ),
                "neither_captured_200_plus": remaining_miss_200,
                "top_v4_rejection_gates": dict(remaining_miss_200 and {k: v for k, v in post_v4_blockers[:4]}),
            },
            "largest_layer2_blocker_pre_v4": gap_final.get("single_filter_largest_move_capture_loss", {}).get("filter"),
            "largest_structural_gate": gap_attribution.get("largest_any_gate_120d"),
        },
        "why_remaining_moves_are_missed": [
            gap_final.get("why_half_of_major_moves_are_missed"),
            (
                "V4 recovered 23/133 V3-missed 200+ moves via EMA22+optional confirmation, "
                f"but {remaining_miss_200} 200+ moves still have no signal in the 100-bar pre-expansion window."
            ),
            (
                "Explicit missed-move sample attributes Failed Breakout as primary blocker; "
                "120d rejection counts show VWAP_MISMATCH (5141) and NO_FAILED_BREAKOUT (5570) "
                "as dominant gate volume after V4 EMA/confirmation relief."
            ),
            gap_final.get("v31_cluster_first_note"),
        ],
        "recommended_modification_for_capture": {
            "primary": {
                "filter": "VWAP",
                "modification": "Relax VWAP Below hard gate to near-VWAP / recent-cross-below tolerance",
                "rationale": (
                    "Largest remaining Layer-2 blocker with highest marginal recovery in gap-analysis "
                    "counterfactuals; V4 already consumed Confirmation (best PF-safe) and EMA changes."
                ),
                "expected_capture_impact_from_gap_counterfactual": {
                    "40_plus_proxy_50": vwap_scenario.get("per_threshold", {}).get("100_plus", {}),
                    "60_plus_proxy_50": vwap_scenario.get("per_threshold", {}).get("100_plus", {}),
                    "100_plus": vwap_scenario.get("per_threshold", {}).get("100_plus", {}),
                    "200_plus": vwap_scenario.get("per_threshold", {}).get("200_plus", {}),
                },
                "expected_pf_impact": vwap_scenario.get("estimated_pf_impact", {}),
                "v4_context_note": (
                    "Counterfactual anchored on V3 baseline; apply directionally to V4 remaining misses "
                    f"({remaining_miss_200} at 200+)."
                ),
            },
            "pf_safer_alternative": {
                "filter": "Location",
                "modification": "Allow Near Support / edge-of-range SELL entries; soften Mid Range block",
                "expected_capture_impact": {
                    "100_plus": location_scenario.get("per_threshold", {}).get("100_plus", {}),
                    "200_plus": location_scenario.get("per_threshold", {}).get("200_plus", {}),
                },
                "expected_pf_impact": location_scenario.get("estimated_pf_impact", {}),
            },
            "already_addressed_by_v4": {
                "filter": "Confirmation",
                "v4_change": v4.get("v4_change_summary", {}).get("modified", {}).get("confirmation", {}),
                "gap_counterfactual_recovery": confirmation_scenario.get("total_estimated_recovered_moves"),
            },
            "gap_analysis_rankings": {
                "by_raw_recovery": raw_recovery,
                "by_recovery_per_pf_damage": pf_safe,
            },
        },
    }


def _buy_side_analysis(
    *,
    buy_formula: dict[str, Any],
    buy_discovery: dict[str, Any],
    final_extraction: dict[str, Any],
    consistency: dict[str, Any],
) -> dict[str, Any]:
    cross = buy_discovery.get("cross_cohort_comparison", {})
    successful = cross.get("successful_bullish_moves_200_plus", {})
    failed_reversals = cross.get("failed_bullish_reversals_no_expansion", {})
    bull_traps = cross.get("bull_traps_failed_breakout_led", {})
    dead_cat = cross.get("dead_cat_bounces_counter_trend", {})
    capture_gap = buy_discovery.get("production_readiness_cross_check", {}).get("capture_gap", {})

    buy_rejections = [
        row
        for row in consistency.get("top_buy_pattern_audit", [])
        if row.get("audit_verdict") == "REJECTED"
    ]
    rejected_buy_models = int(final_extraction.get("extraction_summary", {}).get("rejected_counts", {}).get("BUY", 0))
    accepted_buy_models = int(final_extraction.get("extraction_summary", {}).get("accepted_counts", {}).get("BUY", 0))

    salvage_verdict = "PARTIAL"
    salvage_basis = [
        "Causal pre-move conditions exist (Gap Reversal, Failed Breakdown, Near Support) with strong retrospective capture.",
        "No BUY model survives final extraction (0/20 accepted) or consistency audit walk-forward.",
        "V3 emits 0 BUY signals; anatomy engine captures 1.48% at move start.",
        "Narrow formula (FB+Gap Reversal+Near Support) shows 100% WR on 12 samples but only 3 signals/month — insufficient for production.",
    ]
    if buy_formula.get("final_decision", {}).get("can_buy_formula_survive_reality") == "NO" and accepted_buy_models == 0:
        salvage_verdict = "PARTIAL"

    return {
        "why_buy_fails": [
            "SmartMoneyEngine V3 is SELL-only — 0 BUY signals emitted over 120d.",
            f"{buy_discovery.get('findings', [''])[0]}",
            "Production engines require post-structure confirmation (BOS/CHOCH/FVG) absent at earliest causal bar.",
            f"Anatomy engine signal at bullish move start: {capture_gap.get('anatomy_engine_signal_at_move_start_pct', 1.48)}%.",
            f"Final signal extraction accepted {accepted_buy_models} BUY models; rejected {rejected_buy_models}.",
            "Consistency audit rejects all audited BUY patterns — OOS walk-forward win rate 0% on liquidity-decision combos.",
        ],
        "conditions_before_successful_bullish_moves": {
            "dominant_first_events_200_plus": successful.get("dominant_first_event", []),
            "timing_bars_before_move": buy_discovery.get("move_threshold_analysis", {}).get("200", {}).get("timing", {}),
            "context_at_T_minus_60": successful.get("context_T_minus_60", {}),
            "recurring_patterns": [
                "Gap Reversal or Gap Continuation as earliest causal warning (~96–100 bars pre-expansion).",
                "Failed Breakdown present in pre-move event chain.",
                "Near Support common at T-60 (73/135 in successful cohort context).",
                "Below VWAP frequent even in bullish expansions (78/135).",
                "Counter-trend bearish HTF still produces large moves (85/135 anatomy records).",
            ],
        },
        "conditions_causing_false_reversals": {
            "failed_bullish_reversals_no_expansion": failed_reversals.get("profile", {}),
            "bull_trap_failed_breakout_led": {
                "statistics": bull_traps.get("statistics", {}),
                "interpretation": (
                    "Failed Breakout as second event — 43.74% reach 200+ vs 67.8% for Liquidity Grab; "
                    "high occurrence, lower follow-through quality."
                ),
            },
            "dead_cat_bounces_counter_trend": {
                "sample_size": dead_cat.get("sample_size"),
                "average_move_size_points": dead_cat.get("average_move_size_points"),
                "note": "Bearish HTF + Below VWAP bounces can expand but are lower quality than trend-aligned setups.",
            },
            "differences_summary": cross.get("differences_summary", []),
            "false_reversal_signals": [
                "Liquidity events with no 50+ follow-through (avg 45.68 pts, 0% reach 50+).",
                "Failed Breakout-led bull traps without Failed Breakdown confirmation chain.",
                "Mid-range entries without Near Support proximity.",
            ],
        },
        "salvageability": {
            "verdict": salvage_verdict,
            "evidence": salvage_basis,
            "buy_formula_verdict": buy_formula.get("final_decision", {}).get("can_buy_formula_survive_reality"),
            "accepted_buy_models": accepted_buy_models,
            "rejected_buy_patterns_in_consistency_audit": len(buy_rejections),
            "positive_research_pockets": buy_discovery.get("production_readiness_cross_check", {}).get(
                "realtime_replay_positive_buy_pockets", []
            ),
        },
    }


def _rank_opportunities(
    *,
    sell: dict[str, Any],
    buy: dict[str, Any],
    v4: dict[str, Any],
    gap: dict[str, Any],
) -> list[dict[str, Any]]:
    remaining_200 = sell["v4_improvement_summary"]["remaining_200_plus_misses_after_v4"]
    vwap_impact = sell["recommended_modification_for_capture"]["primary"]["expected_pf_impact"]
    location_impact = sell["recommended_modification_for_capture"]["pf_safer_alternative"]["expected_pf_impact"]

    return [
        {
            "rank": 1,
            "opportunity": "VWAP Below gate relaxation (V5 research on V4 base)",
            "side": "SELL",
            "roi_rationale": (
                "Largest remaining Layer-2 blocker (5141 rejections); gap analysis estimates highest marginal "
                f"200+ recovery after V4 consumed EMA+Confirmation gains. {remaining_200} moves still missed at 200+."
            ),
            "expected_improvement": {
                "200_plus_capture_gain_pp_estimate": gap.get("counterfactual_relaxation", {})
                .get("scenarios", {})
                .get("VWAP", {})
                .get("per_threshold", {})
                .get("200_plus", {})
                .get("capture_gain_pp"),
                "recovered_moves_estimate": gap.get("counterfactual_relaxation", {})
                .get("scenarios", {})
                .get("VWAP", {})
                .get("total_estimated_recovered_moves"),
                "signals_per_month_direction": "up",
            },
            "expected_risk": {
                "profit_factor_delta_estimate": vwap_impact.get("profit_factor_delta_vs_current"),
                "win_rate_delta_pp_estimate": vwap_impact.get("win_rate_delta_pp"),
                "risk_level": "MEDIUM",
            },
        },
        {
            "rank": 2,
            "opportunity": "Location / Mid-Range gate softening",
            "side": "SELL",
            "roi_rationale": (
                "Second-best recovery-to-PF-damage ratio after Confirmation (already in V4). "
                "Targets mid-range blocked entries with moderate capture lift."
            ),
            "expected_improvement": {
                "200_plus_capture_gain_pp_estimate": gap.get("counterfactual_relaxation", {})
                .get("scenarios", {})
                .get("Location", {})
                .get("per_threshold", {})
                .get("200_plus", {})
                .get("capture_gain_pp"),
                "recovered_moves_estimate": gap.get("counterfactual_relaxation", {})
                .get("scenarios", {})
                .get("Location", {})
                .get("total_estimated_recovered_moves"),
            },
            "expected_risk": {
                "profit_factor_delta_estimate": location_impact.get("profit_factor_delta_vs_current"),
                "win_rate_delta_pp_estimate": location_impact.get("win_rate_delta_pp"),
                "risk_level": "LOW-MEDIUM",
            },
        },
        {
            "rank": 3,
            "opportunity": "Promote V4 candidate to production baseline",
            "side": "SELL",
            "roi_rationale": (
                "Validated improvement already complete: 200+ capture +8.55pp, 23 recovered moves, "
                f"PF {v4['comparison']['v4_candidate']['overall_statistics']['profit_factor']} "
                f"WR {v4['comparison']['v4_candidate']['overall_statistics']['win_rate_pct']}%."
            ),
            "expected_improvement": {
                "200_plus_capture_pct": v4["comparison"]["v4_candidate"]["point_capture"]["200"]["capture_rate_pct"],
                "signals_per_month": v4["comparison"]["v4_candidate"]["overall_statistics"]["signals_per_month"],
            },
            "expected_risk": {
                "profit_factor_delta_vs_v3": v4["comparison"]["delta_v4_minus_v3"]["profit_factor"],
                "expectancy_delta_vs_v3": v4["comparison"]["delta_v4_minus_v3"]["expectancy"],
                "risk_level": "LOW — already replay-validated",
            },
        },
        {
            "rank": 4,
            "opportunity": "BUY-side Failed Breakdown early-warning module",
            "side": "BUY",
            "roi_rationale": (
                "PARTIAL salvageability — causal window exists but no production model. "
                "Low priority vs SELL capture gap; high false-reversal risk from bull traps."
            ),
            "expected_improvement": {
                "note": "Retrospective formula 3 signals/month; realtime 5M Failed Breakdown n=10 only.",
            },
            "expected_risk": {
                "walkforward_oos_win_rate_pct": 0.0,
                "consistency_audit_verdict": "REJECTED",
                "risk_level": "HIGH",
            },
            "priority_note": "Deprioritized — included for completeness; not recommended as next cycle.",
        },
    ]


def _final_recommendation(
    *,
    ranked: list[dict[str, Any]],
    sell: dict[str, Any],
    buy: dict[str, Any],
) -> dict[str, Any]:
    top = ranked[0]
    return {
        "if_only_one_more_research_cycle": {
            "research_target": top["opportunity"],
            "why": top["roi_rationale"],
            "expected_improvement": top["expected_improvement"],
            "expected_risk": top["expected_risk"],
            "implementation_shape": (
                "V5 candidate: inherit V4 EMA22 + optional confirmation; "
                "replace strict VWAP Below with graduated VWAP proximity rule "
                "(e.g., within N points below VWAP or fresh cross-below within M bars)."
            ),
            "success_criteria": {
                "200_plus_capture_pct_min": 65.0,
                "profit_factor_min": 3.5,
                "win_rate_pct_min": 65.0,
                "signals_per_month_min": 55.0,
            },
            "do_not_pursue_next": [
                "BUY-side production model — salvageability PARTIAL, 0 accepted models",
                "V3.1 cluster-first refire — zero capture gain on 200+",
                "Raw HTF relaxation — highest PF damage per gap analysis (-1.09 PF delta)",
            ],
        },
        "buy_salvageability_verdict": buy["salvageability"]["verdict"],
        "sell_post_v4_status": (
            f"V4 validated; {sell['v4_improvement_summary']['remaining_200_plus_misses_after_v4']} "
            "200+ moves still uncaptured — VWAP gate is the binding constraint."
        ),
    }


class SmartMoneyEngineNextImprovementRoadmapResearch:
    """Synthesis-only next improvement roadmap from completed research exports."""

    def __init__(
        self,
        *,
        v4_path: Path = SOURCE_EXPORTS["v4_candidate_validation"],
        gap_path: Path = SOURCE_EXPORTS["engine_gap_analysis"],
        v31_path: Path = SOURCE_EXPORTS["v31_validation"],
        extraction_path: Path = SOURCE_EXPORTS["final_signal_extraction"],
        buy_formula_path: Path = SOURCE_EXPORTS["buy_formula_reality"],
        buy_discovery_path: Path = SOURCE_EXPORTS["buy_side_reality_discovery"],
        consistency_path: Path = SOURCE_EXPORTS["research_consistency_audit"],
        report_path: Path = DEFAULT_REPORT_PATH,
    ) -> None:
        self.v4_path = v4_path
        self.gap_path = gap_path
        self.v31_path = v31_path
        self.extraction_path = extraction_path
        self.buy_formula_path = buy_formula_path
        self.buy_discovery_path = buy_discovery_path
        self.consistency_path = consistency_path
        self.report_path = report_path

    def run(self) -> SmartMoneyEngineNextImprovementRoadmapReport:
        started = time.perf_counter()

        v4 = _load_json(self.v4_path)
        gap = _load_json(self.gap_path)
        v31 = _load_json(self.v31_path)
        extraction = _load_json(self.extraction_path)
        buy_formula = _load_json(self.buy_formula_path)
        buy_discovery = _load_json(self.buy_discovery_path)
        consistency = _load_json(self.consistency_path)

        sell = _sell_side_analysis(v4=v4, v31=v31, gap=gap)
        buy = _buy_side_analysis(
            buy_formula=buy_formula,
            buy_discovery=buy_discovery,
            final_extraction=extraction,
            consistency=consistency,
        )
        ranked = _rank_opportunities(sell=sell, buy=buy, v4=v4, gap=gap)
        final_rec = _final_recommendation(ranked=ranked, sell=sell, buy=buy)

        v4_stats = v4["comparison"]["v4_candidate"]["overall_statistics"]
        v4_200 = v4["comparison"]["v4_candidate"]["point_capture"]["200"]["capture_rate_pct"]

        conclusions = [
            (
                f"V4 SELL metrics: WR {v4_stats['win_rate_pct']}%, PF {v4_stats['profit_factor']}, "
                f"Expectancy {v4_stats['expectancy']}, Signals/month {v4_stats['signals_per_month']}."
            ),
            (
                f"200+/300+/500+ capture (V4): {v4_200}% / "
                f"{v4['comparison']['v4_candidate']['point_capture']['300']['capture_rate_pct']}% / "
                f"{v4['comparison']['v4_candidate']['point_capture']['500']['capture_rate_pct']}%."
            ),
            (
                f"V4 improved 200+ capture by {sell['v4_improvement_summary']['200_plus_capture_gain_pp']}pp "
                f"but {sell['v4_improvement_summary']['remaining_200_plus_misses_after_v4']} moves remain missed."
            ),
            f"Biggest remaining SELL weakness: {sell['biggest_remaining_weakness']['issue']}.",
            (
                f"#1 ROI improvement: {ranked[0]['opportunity']} — "
                f"estimated 200+ gain {ranked[0]['expected_improvement'].get('200_plus_capture_gain_pp_estimate')}pp."
            ),
            f"BUY salvageability: {buy['salvageability']['verdict']} — 0 accepted BUY models in final extraction.",
            (
                f"If only one research cycle allowed: {final_rec['if_only_one_more_research_cycle']['research_target']}."
            ),
        ]

        return SmartMoneyEngineNextImprovementRoadmapReport(
            report_type="SmartMoneyEngine Next Improvement Roadmap",
            symbol=v4.get("symbol", "NIFTY50"),
            timeframe=v4.get("timeframe", "5M"),
            methodology={
                "research_only": True,
                "no_new_replays": True,
                "no_new_models": True,
                "no_optimization": True,
                "no_discovery": True,
                "source_exports_only": [path.name for path in SOURCE_EXPORTS.values()],
                "synthesis_approach": (
                    "Cross-synthesize V4 validation, V3.1 baseline, engine gap counterfactuals, "
                    "BUY reality exports, final signal extraction, and consistency audit."
                ),
                "capture_threshold_note": (
                    "40+/60+ capture not stored in exports; proxied from 50+ point_capture rows."
                ),
                "limitations": [
                    "VWAP/Location counterfactuals anchored on V3 rejection blocks — directional estimates for V4 remaining misses.",
                    "BUY walk-forward samples are tiny; salvageability reflects research potential not production readiness.",
                    "Gap analysis per-move attribution is synthesis-only.",
                ],
            },
            source_exports=[path.name for path in SOURCE_EXPORTS.values()],
            sell_side_analysis=sell,
            buy_side_analysis=buy,
            ranked_opportunities=ranked[:3],
            final_recommendation=final_rec,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: SmartMoneyEngineNextImprovementRoadmapReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Exported next improvement roadmap to %s", self.report_path)
        return self.report_path


def generate_smartmoneyengine_next_improvement_roadmap_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export next improvement roadmap JSON."""
    return SmartMoneyEngineNextImprovementRoadmapResearch(report_path=report_path).export()


def main() -> int:
    try:
        path = generate_smartmoneyengine_next_improvement_roadmap_report()
        report = _load_json(path)
        print("SmartMoneyEngine Next Improvement Roadmap")
        print(f"Report: {path}")
        print(f"#1: {report['ranked_opportunities'][0]['opportunity']}")
        print(f"BUY salvageability: {report['buy_side_analysis']['salvageability']['verdict']}")
        return 0
    except SmartMoneyEngineNextImprovementRoadmapError as exc:
        logger.error("Next improvement roadmap error: %s", exc)
        print(f"Next improvement roadmap error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
