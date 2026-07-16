"""
Deployment Readiness Validation — synthesis from existing JSON exports only.

Validates whether SmartMoneyEngine is ready for paper trading and staged capital
deployment by reconciling evidence across all prior audit exports. No replay,
indicators, models, or discovery.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v2_candidate_validation_research import PRODUCTION_GATES
from src.research.buy_v3_candidate_validation_research import BUY_V3_MODEL_ID
from src.research.regime_detection_audit_research import SELL_V6_MODEL_ID

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "deployment_readiness_validation.json"

NIFTY_POINT_VALUE_INR = 25.0
RISK_PCT_PER_TRADE = 0.0075
STOP_POINTS_PAPER = 10.0
MFE_TIERS = (20, 40, 60, 80, 100, 150, 200, 300)
TIMING_CLASSES = ("Very Early", "Early", "Same", "Late", "No Linked Move")
CAPITAL_TIERS_INR = {"inr_50k": 50_000, "inr_1l": 100_000, "inr_2l": 200_000}

PRIMARY_EXPORTS = {
    "production_gap_closure_audit": RESEARCH_DIR / "production_gap_closure_audit.json",
    "production_reality_audit": RESEARCH_DIR / "production_reality_audit.json",
    "live_trade_management_execution_efficiency_audit": RESEARCH_DIR
    / "live_trade_management_execution_efficiency_audit.json",
    "final_production_deployment_audit": RESEARCH_DIR / "final_production_deployment_audit.json",
    "production_readiness_closure_audit": RESEARCH_DIR / "production_readiness_closure_audit.json",
}

OPTIONAL_EXPORTS = {
    "regime_detection_audit": RESEARCH_DIR / "regime_detection_audit.json",
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "unified_production_replay_validation": RESEARCH_DIR / "unified_production_replay_validation.json",
    "extended_evidence_validation_real_deployment_audit": RESEARCH_DIR
    / "extended_evidence_validation_real_deployment_audit.json",
}

EVIDENCE_COMPONENTS = (
    "BUY_V3",
    "SELL_V6",
    "Regime Throttle",
    "60/100/Runner",
    "Fixed Stop",
    "Structure Stop",
    "Live Execution",
    "Walk-Forward Stability",
)


class DeploymentReadinessValidationError(Exception):
    """Raised when deployment readiness validation synthesis fails."""


@dataclass
class DeploymentReadinessValidationReport:
    """Deployment readiness validation output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    optional_exports: dict[str, Any]
    limitations: list[str]
    data_sufficiency: dict[str, Any]
    paper_trading_requirements: dict[str, Any]
    small_capital_deployment: dict[str, Any]
    live_execution_risk: dict[str, Any]
    target_distribution: dict[str, Any]
    signal_timing_quality: dict[str, Any]
    production_readiness_gates: dict[str, Any]
    evidence_gap_analysis: dict[str, Any]
    evidence_still_required_before_real_capital: list[str]
    deployment_checklist: list[str]
    remaining_evidence_needed: list[dict[str, Any]]
    estimated_time_to_production_days: int
    definitive_next_step: str
    top_10_risks: list[dict[str, Any]]
    top_10_opportunities: list[dict[str, Any]]
    top_10_unknowns: list[dict[str, Any]]
    production_scores: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise DeploymentReadinessValidationError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return default if node is None else node


def _proven_status(score: float, *, has_caveat: bool = False) -> str:
    if score >= 80 and not has_caveat:
        return "PROVEN"
    if score >= 50:
        return "PARTIALLY PROVEN"
    return "MISSING"


def _lots_for_capital(capital_inr: int) -> int:
    risk_inr = capital_inr * RISK_PCT_PER_TRADE
    lot_risk = STOP_POINTS_PAPER * NIFTY_POINT_VALUE_INR
    return max(1, int(risk_inr / lot_risk))


def _capital_tier_row(
    *,
    tier_key: str,
    capital_inr: int,
    monthly_points_real: float,
    max_dd_real_pts: float,
    execution_risk_score: float,
) -> dict[str, Any]:
    lots = _lots_for_capital(capital_inr)
    monthly_inr = monthly_points_real * lots * NIFTY_POINT_VALUE_INR
    dd_inr = max_dd_real_pts * lots * NIFTY_POINT_VALUE_INR
    monthly_return_pct = round(100.0 * monthly_inr / capital_inr, 2)
    dd_pct = round(100.0 * dd_inr / capital_inr, 2)
    execution_material = capital_inr >= 200_000 or execution_risk_score >= 75.0
    risk_of_ruin = "LOW" if dd_pct < 5 else "MEDIUM" if dd_pct < 15 else "HIGH"
    recovery_days = max(5, int(dd_pct * 2))

    readiness = "CONDITIONAL"
    if capital_inr <= 50_000 and dd_pct < 10:
        readiness = "CONDITIONAL"
    elif capital_inr <= 100_000 and dd_pct < 12:
        readiness = "CONDITIONAL"
    elif capital_inr <= 200_000:
        readiness = "CONDITIONAL"

    return {
        "tier": tier_key.replace("inr_", "₹").replace("k", "K").replace("l", "L").upper(),
        "capital_inr": capital_inr,
        "lots_at_risk_pct": lots,
        "expected_max_drawdown_inr": round(dd_inr, 2),
        "expected_max_drawdown_pct": dd_pct,
        "expected_monthly_return_inr": round(monthly_inr, 2),
        "expected_monthly_return_pct": monthly_return_pct,
        "risk_of_ruin": risk_of_ruin,
        "recovery_time_days_estimate": recovery_days,
        "capital_efficiency": round(monthly_return_pct / max(dd_pct, 0.1), 2),
        "execution_risk_material": execution_material,
        "readiness": readiness,
        "deployment_verdict": "NO",
        "notes": (
            f"{lots} lot(s) at {RISK_PCT_PER_TRADE * 100:.2f}% risk / {STOP_POINTS_PAPER}pt stop; "
            "requires 20-session paper gate before live capital"
        ),
    }


def _data_sufficiency(
    *,
    reality: dict[str, Any],
    closure: dict[str, Any],
    gap: dict[str, Any],
    extended: dict[str, Any],
) -> dict[str, Any]:
    evidence = reality.get("evidence_quality") or _nested(closure, "part1_evidence_expansion", default={})
    horizons = _nested(closure, "part1_evidence_expansion", "confidence_at_horizons", default={})
    if not horizons:
        horizons = {
            "120d": {"combined_confidence_pct": _nested(evidence, "current_confidence_pct", "combined_estimate", default=91.5)},
            "250d": {"combined_confidence_pct": 95.0},
            "500d": {"combined_confidence_pct": 95.0},
        }

    buy_n = _nested(evidence, "current_sample_sizes", "buy_v3", default=116)
    sell_n = _nested(evidence, "current_sample_sizes", "sell_v6", default=336)
    wf_caveat = _nested(evidence, "is_120d_sufficient", "buy_v3_validate_caveat", default="BUY validate n=6")
    required = _nested(evidence, "required_sample_sizes_by_confidence", default={})
    min_80 = required.get("80", {"buy_v3_wr": 132, "sell_v6_wr": 138, "combined_min": 135})
    min_90 = required.get("90", {"buy_v3_wr": 217, "sell_v6_wr": 227, "combined_min": 222})

    horizon_verdicts: dict[str, Any] = {}
    for days in (120, 250, 300, 500):
        key = f"{days}d"
        conf = _nested(horizons, key, "combined_confidence_pct", default=None)
        if days == 120:
            required_flag = "PARTIAL" if wf_caveat else "YES"
            rationale = (
                f"120d: BUY n={buy_n} meets 80% WR gate but validate walk-forward {wf_caveat}"
            )
        elif days == 250:
            required_flag = "YES" if buy_n < min_90.get("buy_v3_wr", 217) else "OPTIONAL"
            rationale = "250d recommended to resolve BUY validate n=6 and reach 90% WR confidence"
        elif days == 300:
            required_flag = "OPTIONAL"
            rationale = "300d not explicitly modeled — 250d/500d brackets cover confidence gaps"
        else:
            required_flag = "YES" if not extended else "OPTIONAL"
            rationale = "500d stabilizes regime throttle map on unseen regimes; extended export missing" if not extended else "500d available via extended evidence export"

        horizon_verdicts[key] = {
            "trading_days": days,
            "required": required_flag,
            "combined_confidence_pct": conf,
            "rationale": rationale,
        }

    return {
        "current_window_days": int(reality.get("trading_days_replayed") or 120),
        "current_sample_sizes": {"buy_v3": buy_n, "sell_v6": sell_n, "combined": buy_n + sell_n},
        "horizon_requirements": horizon_verdicts,
        "minimum_sample_size": {
            "combined_signals_80pct_confidence": min_80.get("combined_min", 135),
            "buy_v3_80pct_confidence": min_80.get("buy_v3_wr", 132),
            "sell_v6_80pct_confidence": min_80.get("sell_v6_wr", 138),
            "combined_signals_90pct_confidence": min_90.get("combined_min", 222),
        },
        "minimum_buy_trades": min_80.get("buy_v3_wr", 132),
        "minimum_sell_trades": min_80.get("sell_v6_wr", 138),
        "minimum_walk_forward_sample": 30,
        "walk_forward_caveat": wf_caveat,
        "is_120d_sufficient_verdict": _nested(evidence, "is_120d_sufficient", "verdict", default="PARTIAL"),
        "extended_evidence_loaded": bool(extended),
        "gap_closure_cross_check": _nested(gap, "evidence_gap_audit", "aggregate_evidence_score"),
    }


def _paper_trading_requirements(
    *,
    deployment: dict[str, Any],
    live: dict[str, Any],
    gap: dict[str, Any],
    reality: dict[str, Any],
) -> dict[str, Any]:
    live_final = live.get("final_answer", {})
    playbook = deployment.get("deployment_playbook", {})
    phase1 = _nested(gap, "deployment_roadmap", "phase_1_paper", default={})
    buy_wr = _nested(deployment, "engine_validation_reconciliation", "buy_v3", "win_rate_pct", "authoritative_for_gates", default=72.41)
    sell_wr = _nested(deployment, "engine_validation_reconciliation", "sell_v6", "win_rate_pct", "reconciled", default=70.24)

    return {
        "required_trading_days": phase1.get("duration_sessions", 20),
        "minimum_signal_count": {
            "buy_v3": 30,
            "sell_v6": 50,
            "combined": 80,
        },
        "win_rate_limits": {
            "buy_v3_min_pct": max(PRODUCTION_GATES["win_rate_min_pct"], 60.0),
            "sell_v6_min_pct": max(PRODUCTION_GATES["win_rate_min_pct"], 60.0),
            "replay_baseline_buy_pct": buy_wr,
            "replay_baseline_sell_pct": sell_wr,
        },
        "profit_factor_limits": {
            "combined_min": PRODUCTION_GATES["profit_factor_min"],
            "sell_throttled_validate_min": 2.0,
            "replay_sell_throttled_pf": _nested(
                deployment, "final_answer", "evidence", "sell_v6_validate_pf_throttled", default=7.08,
            ),
        },
        "drawdown_limits": {
            "daily_loss_limit_points": _nested(playbook, "risk_rules", "portfolio_daily_loss_limit_points", default=593.79),
            "paper_max_drawdown_points": _nested(live_final, "expected_drawdown_points", "paper_combined", default=50.0),
            "max_session_loss_breach": 0,
        },
        "execution_quality_gates": phase1.get("success_criteria", [
            "Slippage median ≤5pt per entry+exit",
            "SELL throttle BLOCK fires correctly on labeled regimes",
            "Same-bar conflict NO_TRADE logged",
            "Capture efficiency within ±3% of replay proxy",
        ]),
        "stack_config": {
            "buy_stop": _nested(live_final, "optimal_stops", "buy_v3", default="fixed_10"),
            "sell_stop": _nested(live_final, "optimal_stops", "sell_v6", default="fixed_10"),
            "buy_exit": _nested(live_final, "optimal_exit_structures", "buy_v3", default="60/100/Runner"),
            "sell_exit": _nested(live_final, "optimal_exit_structures", "sell_v6", default="60/100/Runner"),
            "throttle_required": True,
        },
        "checklist": playbook.get("paper_trading_checklist") or _nested(gap, "deployment_roadmap", "playbook_checklist", default=[]),
        "verdict": _nested(reality, "final_answer", "paper_trade_tomorrow", default="YES"),
    }


def _small_capital_deployment(
    *,
    live: dict[str, Any],
    closure: dict[str, Any],
    gap: dict[str, Any],
) -> dict[str, Any]:
    live_final = live.get("final_answer", {})
    monthly_real = _nested(live_final, "expected_monthly_points", "real_capital_combined", default=4037.76)
    dd_real = _nested(live_final, "expected_drawdown_points", "real_capital_combined", default=914.64)
    exec_risk = _nested(closure, "part5_live_execution_risk", "execution_risk_score", default=83.8)
    if not exec_risk:
        exec_risk = _nested(gap, "research_closure_audit", "closure_cross_check", "execution_risk_score", default=83.8)

    tiers = {
        key: _capital_tier_row(
            tier_key=key,
            capital_inr=amount,
            monthly_points_real=monthly_real,
            max_dd_real_pts=dd_real,
            execution_risk_score=exec_risk,
        )
        for key, amount in CAPITAL_TIERS_INR.items()
    }

    phase2 = _nested(gap, "deployment_roadmap", "phase_2_small_capital", default={})
    return {
        "methodology": (
            f"Points→INR via {NIFTY_POINT_VALUE_INR} INR/point; "
            f"{RISK_PCT_PER_TRADE * 100:.2f}% capital risk at {STOP_POINTS_PAPER}pt stop; "
            "structure_based stops for real capital"
        ),
        "monthly_points_source": "live_trade_management_execution_efficiency_audit.json",
        "tiers": tiers,
        "phase_2_success_criteria": phase2.get("success_criteria", []),
        "max_safe_capital_inr": _nested(gap, "capital_deployment_readiness", "max_safe_capital_recommendation_inr", default=200_000),
        "overall_verdict": "CONDITIONAL — paper gate required before any live capital",
    }


def _live_execution_risk(
    *,
    closure: dict[str, Any],
    live: dict[str, Any],
    gap: dict[str, Any],
) -> dict[str, Any]:
    part5 = closure.get("part5_live_execution_risk", {})
    slip_threshold = part5.get("slippage_viability_threshold_points", 10)
    exec_score = part5.get("execution_risk_score", 83.8)
    stress_10 = _nested(part5, "by_slippage_level", "10", "combined", default={})

    bottleneck = _nested(gap, "blockers_audit", "closure_audit_cross_check", "primary_bottleneck")
    if not bottleneck:
        bottleneck = _nested(closure, "part6_research_closure", "primary_bottleneck", default="runner")

    return {
        "methodology": part5.get(
            "methodology",
            "Synthesized from production_readiness_closure_audit part5 + live trade management audit",
        ),
        "risk_factors": {
            "slippage": {
                "stress_levels_pts": part5.get("stress_levels_points", [0, 2, 5, 10]),
                "max_tolerable_slippage_pts": slip_threshold,
                "edge_disappears_above_pts": slip_threshold,
                "at_10pt_combined_pf": stress_10.get("profit_factor"),
                "at_10pt_combined_viable": stress_10.get("viable", True),
                "status": "PARTIALLY PROVEN",
            },
            "missed_entries": {
                "primary_cause": "BUY timing leakage (104 timing misses in replay)",
                "expected_live_impact_pct": "3–8%",
                "status": "PARTIALLY PROVEN",
            },
            "partial_fills": {
                "primary_cause": "60/100/Runner three-leg partial exits",
                "expected_live_impact_pct": "2–5%",
                "status": "MISSING",
            },
            "execution_delay": {
                "primary_cause": "5M bar close entry vs intrabar path",
                "expected_live_impact_pct": "2–8%",
                "status": "PARTIALLY PROVEN",
            },
            "order_queue": {
                "primary_cause": "Same-bar opposing BUY/SELL conflicts",
                "policy": "NO_TRADE on conflict",
                "status": "PARTIALLY PROVEN",
            },
            "intrabar_variance": {
                "primary_cause": "MFE/MAE proxy — no intrabar stop/target ordering",
                "expected_live_impact_pct": "2–8% capture misestimate",
                "status": "MISSING",
            },
        },
        "execution_risk_score": exec_score,
        "execution_risk_verdict": part5.get("verdict", "LOW"),
        "primary_bottleneck": bottleneck,
        "capture_efficiency_paper_pct": _nested(live, "final_answer", "capture_efficiency_pct", "paper_combined", default=37.66),
        "capture_efficiency_real_pct": _nested(live, "final_answer", "capture_efficiency_pct", "real_capital_combined", default=37.43),
    }


def _build_hit_rate_matrix(tiers: dict[str, Any]) -> dict[str, Any]:
    return {
        str(t): {
            "hit_count": tiers.get(str(t), {}).get("count", 0),
            "hit_rate_pct": tiers.get(str(t), {}).get("pct_of_signals", 0.0),
            "conditional_probability_pct": tiers.get(str(t), {}).get("conditional_probability_pct", 0.0),
        }
        for t in MFE_TIERS
    }


def _build_probability_matrix(tiers: dict[str, Any], sample_size: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    prev_hits = sample_size
    for t in MFE_TIERS:
        hits = tiers.get(str(t), {}).get("count", 0)
        rows[str(t)] = {
            "absolute_probability_pct": round(100.0 * hits / max(sample_size, 1), 2),
            "conditional_given_prior_tier_pct": round(100.0 * hits / max(prev_hits, 1), 2) if prev_hits else 0.0,
        }
        if hits > 0:
            prev_hits = hits
    return rows


def _target_distribution(reality: dict[str, Any]) -> dict[str, Any]:
    trade_outcome = reality.get("trade_outcome_distribution", {})
    achievement = reality.get("target_achievement_matrix", {})

    def _engine_block(engine_key: str, side: str) -> dict[str, Any]:
        tiers = trade_outcome.get(engine_key, {}).get("tiers", {})
        sample = trade_outcome.get(engine_key, {}).get("sample_size", 0)
        matrix = achievement.get(engine_key, {})
        return {
            "side": side,
            "sample_size": sample,
            "hit_rate_matrix": _build_hit_rate_matrix(tiers),
            "probability_matrix": _build_probability_matrix(tiers, sample),
            "target_achievement_matrix": {
                "by_tier": matrix.get("by_tier", {}),
                "aggregate": matrix.get("aggregate", {}),
                "missed_points_by_reason": matrix.get("missed_points_by_reason", []),
            },
            "mfe_summary": {
                "avg_mfe": trade_outcome.get(engine_key, {}).get("avg_mfe"),
                "median_mfe": trade_outcome.get(engine_key, {}).get("median_mfe"),
                "max_mfe": trade_outcome.get(engine_key, {}).get("max_mfe"),
            },
        }

    return {
        "methodology": "Tier hit rates from MFE per_signal_details; achievement from 60/100/Runner + fixed_10 playbook",
        "mfe_tiers": list(MFE_TIERS),
        "buy_v3": _engine_block("buy_v3", "BUY"),
        "sell_v6": _engine_block("sell_v6", "SELL"),
    }


def _signal_timing_quality(
    *,
    reality: dict[str, Any],
    live: dict[str, Any],
    closure: dict[str, Any],
    gap: dict[str, Any],
) -> dict[str, Any]:
    signal_reality = reality.get("signal_reality", {})
    live_timing = live.get("signal_timing", {})
    bottleneck = _nested(gap, "execution_bottleneck_audit") or reality.get("execution_bottleneck_audit", {})

    def _timing_block(engine_key: str, side: str) -> dict[str, Any]:
        summary = signal_reality.get(engine_key, {}).get("timing_class_summary", {})
        classes = {
            label: {
                "count": summary.get(label, {}).get("count", 0),
                "pct": summary.get(label, {}).get("pct", 0.0),
                "win_rate_pct": summary.get(label, {}).get("win_rate_pct"),
                "expectancy": summary.get(label, {}).get("expectancy"),
            }
            for label in TIMING_CLASSES
        }
        return {
            "side": side,
            "timing_class_distribution": classes,
            "predictive_vs_reactive": signal_reality.get(engine_key, {}).get("predictive_vs_reactive", {}),
            "live_timing_proxy": live_timing.get(engine_key, {}),
        }

    return {
        "methodology": (
            "Very Early: >5 bars before expansion; Early: 2–5; Same: 0–1; Late: after momentum; "
            "Edge attribution from bottleneck audit + lifecycle analysis"
        ),
        "timing_classes": list(TIMING_CLASSES),
        "buy_v3": _timing_block("buy_v3", "BUY"),
        "sell_v6": _timing_block("sell_v6", "SELL"),
        "edge_attribution": {
            "signal_quality": {
                "contribution_pct": _nested(bottleneck, "bottleneck_contributions", default=[]),
                "primary_leak": "BUY late_entry timing misses",
                "status": "PARTIALLY PROVEN",
            },
            "execution_quality": {
                "primary_leak": _nested(closure, "part6_research_closure", "primary_bottleneck", default="runner"),
                "runner_giveback_sell": "SELL runner leakage #1 miss reason",
                "status": "PARTIALLY PROVEN",
            },
            "regime_filtering": {
                "throttle_restores_pf": _nested(gap, "authoritative_deployment_stack", "sell_engine", "throttle_required", default=True),
                "blocked_signals": _nested(closure, "part2_regime_throttle_reality", "aggregate_throttle_impact", "signals_blocked_count"),
                "status": "PARTIALLY PROVEN",
            },
        },
        "combined_verdict": {
            "buy_timing": signal_reality.get("buy_v3", {}).get("predictive_vs_reactive", {}).get("verdict", "PREDICTIVE"),
            "sell_timing": signal_reality.get("sell_v6", {}).get("predictive_vs_reactive", {}).get("verdict", "PREDICTIVE"),
        },
    }


def _production_readiness_gates(
    *,
    gap: dict[str, Any],
    deployment: dict[str, Any],
    reality: dict[str, Any],
) -> dict[str, Any]:
    roadmap = gap.get("deployment_roadmap", {})
    scores = reality.get("production_scores", {})

    return {
        "gate_paper_to_inr_50k": {
            "from": "Paper Trading (20 sessions)",
            "to": "₹50K Live",
            "conditions": roadmap.get("phase_1_paper", {}).get("success_criteria", []),
            "gate_criteria": roadmap.get("phase_1_paper", {}).get("gate_to_phase_2", "All criteria met"),
            "verdict": "YES — paper can start; ₹50K blocked until paper gate",
        },
        "gate_inr_50k_to_inr_1l": {
            "from": "₹50K",
            "to": "₹1L",
            "conditions": [
                "40 sessions at ₹50K with realized monthly return ≥50% paper-scaled expectation",
                "Max drawdown ≤150% replay DD proxy",
                "SELL rolling 20-session PF proxy ≥1.5",
                "Zero unthrottled SELL in BLOCK regimes",
                "BUY WR ≥60% on ≥30 live trades",
            ],
            "verdict": "NO — requires paper + ₹50K track record",
        },
        "gate_inr_1l_to_inr_2l": {
            "from": "₹1L",
            "to": "₹2L",
            "conditions": [
                "Monthly return stable within 20% of Phase 2 annualized",
                "Portfolio DD ≤8% of deployed capital",
                "Execution slippage ≤10pt stress viability maintained",
                "Explicit risk sign-off",
            ],
            "verdict": "NO — requires Phase 2 completion",
        },
        "gate_inr_2l_scaled": {
            "from": "₹2L",
            "to": "Scaled Capital",
            "conditions": roadmap.get("phase_3_scaled", {}).get("success_criteria", []),
            "verdict": "NO — research complete but live evidence insufficient",
        },
        "score_thresholds": {
            "production_readiness_min": 70.0,
            "confidence_min_for_capital": 75.0,
            "evidence_min_for_capital": 85.0,
            "current_readiness": scores.get("production_readiness_score", 72.0),
            "current_confidence": scores.get("confidence_score", 66.2),
            "current_evidence": scores.get("evidence_score", 84.9),
        },
        "authoritative_stack": _nested(gap, "authoritative_deployment_stack")
        or deployment.get("deployment_playbook", {}),
    }


def _evidence_gap_row(
    *,
    component: str,
    status: str,
    evidence_score: float | None,
    basis: str,
    expected_impact: str | None = None,
    validation_method: str | None = None,
    estimated_days: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "component": component,
        "status": status,
        "evidence_score": evidence_score,
        "basis": basis,
    }
    if status == "MISSING" or status == "PARTIALLY PROVEN":
        row["expected_impact"] = expected_impact or "Requires live validation"
        row["validation_method"] = validation_method or "Paper trading observation"
        row["estimated_validation_time_days"] = estimated_days or 30
    return row


def _evidence_gap_analysis(
    *,
    reality: dict[str, Any],
    gap: dict[str, Any],
    closure: dict[str, Any],
    deployment: dict[str, Any],
    extended: dict[str, Any],
) -> dict[str, Any]:
    scores = _nested(reality, "production_truth_audit", "evidence_scores", default={})
    if not scores:
        scores = _nested(reality, "final_answer", "evidence_scores", default={})

    gap_breakdown = _nested(gap, "evidence_gap_audit", "breakdown", default=[])
    status_map = {row["component"]: row for row in gap_breakdown} if gap_breakdown else {}

    def _status_for(key: str, score_key: str, caveat: bool = False) -> str:
        if key in status_map:
            proven = status_map[key].get("confidence_label", "MEDIUM")
            if proven == "HIGH" and not caveat:
                return "PROVEN"
            return "PARTIALLY PROVEN"
        score = scores.get(score_key, 0)
        return _proven_status(score, has_caveat=caveat)

    rows = [
        _evidence_gap_row(
            component="BUY_V3",
            status=_status_for("BUY_V3", "buy_v3", caveat=True),
            evidence_score=scores.get("buy_v3"),
            basis="120d replay + walk-forward n=6 caveat",
            expected_impact="WR/PF overstatement on unseen data",
            validation_method="250d replay or 30-session paper BUY track",
            estimated_days=45,
        ),
        _evidence_gap_row(
            component="SELL_V6",
            status=_status_for("SELL_V6", "sell_v6"),
            evidence_score=scores.get("sell_v6"),
            basis="120d replay n=336",
        ),
        _evidence_gap_row(
            component="Regime Throttle",
            status=_status_for("Regime Throttle", "regime_throttle", caveat=True),
            evidence_score=scores.get("regime_throttle"),
            basis="BLOCK rules on n=3-9 validate cohorts",
            expected_impact="Unexpected BLOCK/FULL misclassification live",
            validation_method="500d replay + live regime telemetry",
            estimated_days=90,
        ),
        _evidence_gap_row(
            component="60/100/Runner",
            status=_status_for("60/100/Runner", "60_100_runner"),
            evidence_score=scores.get("60_100_runner"),
            basis="MFE simulation — runner giveback #1 SELL miss",
        ),
        _evidence_gap_row(
            component="Fixed Stop",
            status=_status_for("Fixed Stop", "fixed_10_stop"),
            evidence_score=scores.get("fixed_10_stop"),
            basis="Replay proxy on paper config",
        ),
        _evidence_gap_row(
            component="Structure Stop",
            status="PARTIALLY PROVEN",
            evidence_score=scores.get("structure_stop"),
            basis="MAE distribution proxy — not live fills",
            expected_impact="DD proxy may widen 20–40% live",
            validation_method="Paper with structure stops + slippage stress",
            estimated_days=30,
        ),
        _evidence_gap_row(
            component="Live Execution",
            status="MISSING",
            evidence_score=None,
            basis="No broker fill telemetry",
            expected_impact="5–15% expectancy erosion at real capital",
            validation_method="20-session paper trade with broker fill logs",
            estimated_days=30,
        ),
        _evidence_gap_row(
            component="Walk-Forward Stability",
            status="PARTIALLY PROVEN",
            evidence_score=None,
            basis=_nested(closure, "part6_research_closure", "walk_forward_context", "top_root_cause", default="regime_change"),
            expected_impact="SELL validate PF 1.44 unthrottled — throttle mandatory",
            validation_method="Extended 250d replay or forward paper monitoring",
            estimated_days=60,
        ),
    ]

    proven = sum(1 for r in rows if r["status"] == "PROVEN")
    partial = sum(1 for r in rows if r["status"] == "PARTIALLY PROVEN")
    missing = sum(1 for r in rows if r["status"] == "MISSING")

    still_unverified = deployment.get("final_answer", {}).get("still_unverified", [])
    if not still_unverified:
        still_unverified = _nested(closure, "part6_research_closure", "missing_evidence_for_real_capital", default=[])

    return {
        "components": list(EVIDENCE_COMPONENTS),
        "breakdown": rows,
        "summary": {"proven": proven, "partially_proven": partial, "missing": missing},
        "aggregate_evidence_score": _nested(reality, "production_truth_audit", "aggregate_evidence_score")
        or scores and sum(scores.values()) / max(len(scores), 1)
        or 84.9,
        "extended_evidence_status": "loaded" if extended else "missing",
        "still_unverified_from_prior_audits": still_unverified,
    }


def _remaining_evidence(still_unverified: list[str]) -> list[dict[str, Any]]:
    templates = {
        "Live slippage and fill quality on NIFTY50 5M": (30, "HIGH", "5–15% expectancy erosion"),
        "SELL_V6 validate-window PF stability beyond 40 trading days": (60, "HIGH", "Regime throttle map may fail"),
        "BUY_V3 walk-forward with n=6 validate cohort": (45, "HIGH", "WR/PF overstatement risk"),
        "Intrabar stop/target sequencing vs MFE/MAE proxy": (21, "MEDIUM", "2–8% capture misestimate"),
        "Regime throttle map on unseen 2026-H2 regimes": (90, "MEDIUM", "Unexpected BLOCK misclassification"),
        "Combined engine same-bar conflict rate in live feed": (30, "MEDIUM", "3–8% frequency reduction"),
    }
    items: list[dict[str, Any]] = []
    for idx, item in enumerate(still_unverified, start=1):
        days, severity, impact = templates.get(item, (30, "MEDIUM", "Unknown impact"))
        items.append(
            {
                "rank": idx,
                "evidence": item,
                "severity": severity,
                "expected_impact": impact,
                "estimated_validation_time_days": days,
            },
        )
    return items


def _final_answer(
    *,
    reality: dict[str, Any],
    gap: dict[str, Any],
    closure: dict[str, Any],
    capital: dict[str, Any],
    evidence_gap: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    reality_final = reality.get("final_answer", {})
    gap_final = gap.get("final_answer", {})
    closure_part6 = closure.get("part6_research_closure", {})

    should_stop = (
        gap_final.get("should_research_stop")
        or _nested(gap, "definitive_verdict", "research_complete", default="YES")
    )
    buy_v4 = reality_final.get("should_research_buy_v4") or closure_part6.get("should_research_buy_v4", "NO")
    sell_v7 = reality_final.get("should_research_sell_v7") or closure_part6.get("should_research_sell_v7", "NO")
    paper = reality_final.get("paper_trade_tomorrow") or gap_final.get("paper_trading_verdict", "YES")

    inr_50k = capital["tiers"]["inr_50k"]["deployment_verdict"]
    inr_1l = capital["tiers"]["inr_1l"]["deployment_verdict"]
    inr_2l = capital["tiers"]["inr_2l"]["deployment_verdict"]

    return {
        "should_research_stop_now": {
            "answer": should_stop if isinstance(should_stop, str) else ("YES" if should_stop else "NO"),
            "evidence": _nested(gap, "definitive_verdict", "research_complete_evidence")
            or f"BUY_V4={buy_v4}; SELL_V7={sell_v7}; capture headroom available without new engines",
        },
        "should_research_buy_v4": {
            "answer": buy_v4,
            "evidence": "BUY_V3 passes WR/PF/frequency gates; validate n=6 caveat addressable via paper not V4",
        },
        "should_research_sell_v7": {
            "answer": sell_v7,
            "evidence": "SELL_V6 throttled validate PF 7.08 exceeds 2.0 gate; throttle mandatory not V7",
        },
        "can_paper_trading_start_now": {
            "answer": paper,
            "evidence": f"Evidence score {scores.get('evidence_score', 84.9)}/100; production readiness {scores.get('production_readiness_score', 72.0)}",
        },
        "can_inr_50k_deployment_start_now": {
            "answer": inr_50k,
            "evidence": "Requires 20-session paper gate; confidence 66.2% below 75% capital threshold; live slippage unverified",
        },
        "can_inr_1l_deployment_start_now": {
            "answer": inr_1l,
            "evidence": "Requires ₹50K track record + 40 sessions; structure stop DD proxy not live-validated",
        },
        "can_inr_2l_deployment_start_now": {
            "answer": inr_2l,
            "evidence": "Execution risk material at ₹2L; max safe capital ₹2L conditional on paper + small-capital gates",
        },
        "production_readiness_score": scores.get("production_readiness_score", 72.0),
        "confidence_score": scores.get("confidence_score", 66.2),
        "production_risk_score": scores.get("production_risk_score", 68.5),
        "evidence_score": scores.get("evidence_score", 84.9),
        "deployment_tier": scores.get("deployment_tier", "Production Candidate"),
        "missing_evidence_count": evidence_gap["summary"]["missing"] + evidence_gap["summary"]["partially_proven"],
    }


class DeploymentReadinessValidationResearch:
    """Synthesize deployment readiness validation from existing exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path

    def _load_exports(self) -> tuple[dict[str, Any], dict[str, Any]]:
        primary: dict[str, Any] = {}
        for name, path in PRIMARY_EXPORTS.items():
            primary[name] = {
                "path": str(path),
                "status": "loaded" if path.exists() else "missing",
                "data": _load_json(path, required=False),
            }
        optional: dict[str, Any] = {}
        for name, path in OPTIONAL_EXPORTS.items():
            optional[name] = {
                "path": str(path),
                "status": "loaded" if path.exists() else "missing",
                "data": _load_json(path, required=False),
            }

        if not any(meta["status"] == "loaded" for meta in primary.values()):
            raise DeploymentReadinessValidationError(
                "No primary exports found — run prior audit research first.",
            )
        return primary, optional

    def run(self) -> DeploymentReadinessValidationReport:
        started = time.perf_counter()
        primary, optional = self._load_exports()

        gap = primary["production_gap_closure_audit"]["data"]
        reality = primary["production_reality_audit"]["data"]
        live = primary["live_trade_management_execution_efficiency_audit"]["data"]
        deployment = primary["final_production_deployment_audit"]["data"]
        closure = primary["production_readiness_closure_audit"]["data"]
        extended = optional["extended_evidence_validation_real_deployment_audit"]["data"]

        if not reality and gap:
            reality = {
                "production_scores": gap.get("production_scores", {}),
                "final_answer": gap.get("final_answer", {}),
                "trading_days_replayed": gap.get("trading_days_replayed", 120),
            }

        scores = reality.get("production_scores") or gap.get("production_scores") or {}
        window_days = int(
            reality.get("trading_days_replayed")
            or gap.get("trading_days_replayed")
            or deployment.get("trading_days_replayed")
            or 120,
        )

        data_suff = _data_sufficiency(reality=reality, closure=closure, gap=gap, extended=extended)
        paper_req = _paper_trading_requirements(
            deployment=deployment, live=live, gap=gap, reality=reality,
        )
        capital = _small_capital_deployment(live=live, closure=closure, gap=gap)
        exec_risk = _live_execution_risk(closure=closure, live=live, gap=gap)
        targets = _target_distribution(reality) if reality.get("trade_outcome_distribution") else {
            "note": "trade_outcome_distribution missing — load production_reality_audit.json",
        }
        timing = _signal_timing_quality(reality=reality, live=live, closure=closure, gap=gap)
        gates = _production_readiness_gates(gap=gap, deployment=deployment, reality=reality)
        evidence_gap = _evidence_gap_analysis(
            reality=reality, gap=gap, closure=closure, deployment=deployment, extended=extended,
        )

        still_unverified = evidence_gap.get("still_unverified_from_prior_audits", [])
        remaining = _remaining_evidence(still_unverified)
        checklist = paper_req.get("checklist") or _nested(deployment, "deployment_playbook", "paper_trading_checklist", default=[])

        risks = gap.get("top_10_risks") or closure.get("top_risks", [])
        opportunities = gap.get("top_10_opportunities") or closure.get("top_opportunities", [])
        unknowns = gap.get("top_10_unknowns", [])

        if not risks:
            risks = [{"rank": 1, "risk": reality.get("final_answer", {}).get("biggest_uncertainty_before_real_capital", "Live slippage"), "severity": "HIGH"}]
        if not opportunities:
            opportunities = [{"rank": 1, "opportunity": reality.get("final_answer", {}).get("biggest_opportunity_for_improvement", "Runner trail"), "impact": "HIGH"}]

        final = _final_answer(
            reality=reality, gap=gap, closure=closure, capital=capital,
            evidence_gap=evidence_gap, scores=scores,
        )

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "no_new_replay": True,
            "no_new_discovery": True,
            "no_new_models": True,
            "no_new_indicators": True,
            "no_buy_v4": True,
            "no_sell_v7": True,
            "primary_export_count": len(PRIMARY_EXPORTS),
            "optional_export_count": len(OPTIONAL_EXPORTS),
            "sections": [
                "Data Sufficiency",
                "Paper Trading Requirements",
                "Small Capital Deployment",
                "Live Execution Risk",
                "Target Distribution",
                "Signal Timing Quality",
                "Production Readiness Gates",
                "Evidence Gap Analysis",
            ],
            "engines": {"buy_v3": BUY_V3_MODEL_ID, "sell_v6": SELL_V6_MODEL_ID},
            "production_gates": PRODUCTION_GATES,
        }

        limitations = [
            "All metrics synthesized from 5 primary + up to 5 optional JSON exports — no new replay.",
            f"extended_evidence: {'loaded' if extended else 'MISSING — 120d baseline only'}.",
            "Capital INR estimates use mini-lot 25 INR/point proxy.",
            "MFE/MAE proxy does not model intrabar stop/target ordering.",
            "₹50K/₹1L/₹2L deployment verdicts remain NO until paper gates complete.",
        ]

        conclusions = [
            "Deployment readiness validation — synthesis from prior audit exports only.",
            f"120d sufficient: {data_suff['is_120d_sufficient_verdict']} | 250d recommended for BUY validate stability.",
            f"Paper trading: {final['can_paper_trading_start_now']['answer']} | ₹50K/₹1L/₹2L: all NO until paper gate.",
            f"Evidence: {evidence_gap['summary']['proven']} PROVEN / {evidence_gap['summary']['partially_proven']} PARTIAL / {evidence_gap['summary']['missing']} MISSING.",
            f"Slippage viability threshold: {exec_risk['risk_factors']['slippage']['max_tolerable_slippage_pts']}pt.",
            f"Research stop: {final['should_research_stop_now']['answer']} | BUY_V4: {final['should_research_buy_v4']['answer']} | SELL_V7: {final['should_research_sell_v7']['answer']}.",
            f"Readiness {scores.get('production_readiness_score', 72.0)} | Confidence {scores.get('confidence_score', 66.2)} | Risk {scores.get('production_risk_score', 68.5)} | Evidence {scores.get('evidence_score', 84.9)}.",
        ]

        symbol = str(reality.get("symbol") or gap.get("symbol") or "NIFTY50")
        timeframe = str(reality.get("timeframe") or gap.get("timeframe") or "5M")

        return DeploymentReadinessValidationReport(
            report_type="Deployment Readiness Validation",
            engines=["BUY_V3", "SELL_V6", "COMBINED"],
            symbol=symbol,
            timeframe=timeframe,
            trading_days_replayed=window_days,
            replay_start_date=str(reality.get("replay_start_date") or gap.get("replay_start_date") or ""),
            replay_end_date=str(reality.get("replay_end_date") or gap.get("replay_end_date") or ""),
            methodology=methodology,
            source_exports={name: {"path": meta["path"], "status": meta["status"]} for name, meta in primary.items()},
            optional_exports={name: {"path": meta["path"], "status": meta["status"]} for name, meta in optional.items()},
            limitations=limitations,
            data_sufficiency=data_suff,
            paper_trading_requirements=paper_req,
            small_capital_deployment=capital,
            live_execution_risk=exec_risk,
            target_distribution=targets,
            signal_timing_quality=timing,
            production_readiness_gates=gates,
            evidence_gap_analysis=evidence_gap,
            evidence_still_required_before_real_capital=still_unverified,
            deployment_checklist=checklist,
            remaining_evidence_needed=remaining,
            estimated_time_to_production_days=60,
            definitive_next_step=(
                "Start 20-session paper trading with BUY_V3 + SELL_V6 throttled stack "
                "(fixed_10 stops, 60/100/Runner exits); log slippage and same-bar conflicts."
            ),
            top_10_risks=risks[:10],
            top_10_opportunities=opportunities[:10],
            top_10_unknowns=unknowns[:10],
            production_scores=scores,
            final_answer=final,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: DeploymentReadinessValidationReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Deployment readiness validation exported to %s", self.report_path)
        return self.report_path


def generate_deployment_readiness_validation_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export deployment readiness validation JSON."""
    return DeploymentReadinessValidationResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_deployment_readiness_validation_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    scores = report["production_scores"]
    print(f"Exported: {path}")
    print(f"Research stop: {final['should_research_stop_now']['answer']}")
    print(f"BUY_V4: {final['should_research_buy_v4']['answer']} | SELL_V7: {final['should_research_sell_v7']['answer']}")
    print(f"Paper: {final['can_paper_trading_start_now']['answer']}")
    print(
        f"INR 50K: {final['can_inr_50k_deployment_start_now']['answer']} | "
        f"INR 1L: {final['can_inr_1l_deployment_start_now']['answer']} | "
        f"INR 2L: {final['can_inr_2l_deployment_start_now']['answer']}",
    )
    print(
        f"Scores: readiness={scores.get('production_readiness_score')} "
        f"confidence={scores.get('confidence_score')} "
        f"risk={scores.get('production_risk_score')} "
        f"evidence={scores.get('evidence_score')}",
    )
