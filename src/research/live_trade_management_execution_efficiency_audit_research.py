"""
Live Trade Management & Execution Efficiency Audit — synthesis from existing exports only.

Validates execution quality and determines highest-expectancy production trade plan for
paper and real capital. Simulates stops/exits from MFE/MAE/risk_points in per_signal_details.
No replay, indicators, models, or discovery.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v2_candidate_validation_research import PRODUCTION_GATES
from src.research.buy_v3_candidate_validation_research import (
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
)
from src.research.buy_v3_tradeability_production_validation_research import _fixed_target_pnl
from src.research.production_edge_enhancement_audit_research import _profit_factor_from_pnls
from src.research.production_trading_playbook_audit_research import (
    LEG_WEIGHTS,
    _metrics_from_pnls,
    _resolve_stop_points,
    _tiered_structure_pnl,
)
from src.research.regime_detection_audit_research import (
    SELL_V6_MODEL_ID,
    THROTTLE_WEIGHT,
    classify_signal_regime,
)
from src.research.sell_v6_replay_validation_research import V6_VWAP_GATE_RULE
from src.research.unified_production_replay_validation_research import _max_drawdown

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "live_trade_management_execution_efficiency_audit.json"

SOURCE_EXPORTS = {
    "final_production_deployment_audit": RESEARCH_DIR / "final_production_deployment_audit.json",
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "production_trading_playbook_audit": RESEARCH_DIR / "production_trading_playbook_audit.json",
    "regime_detection_audit": RESEARCH_DIR / "regime_detection_audit.json",
}

EXECUTION_STOP_VARIANTS = (
    "fixed_10",
    "fixed_15",
    "fixed_20",
    "structure_based",
    "liquidity_based",
)

FIXED_EXIT_TARGETS = (40, 60, 80, 100)

EXIT_STRUCTURES: dict[str, dict[str, Any]] = {
    "40/60/100": {"t1": 40, "t2": 60, "t3": 100, "runner": False},
    "40/80/Runner": {"t1": 40, "t2": 80, "t3": None, "runner": True},
    "50/100/Runner": {"t1": 50, "t2": 100, "t3": None, "runner": True},
    "60/100/Runner": {"t1": 60, "t2": 100, "t3": None, "runner": True},
}

TREND_REGIMES = ("Strong Trend", "Weak Trend", "Range")
VOL_REGIMES = ("High Volatility", "Low Volatility")

MISS_REASONS = ("early_exit", "target_structure", "runner", "stop", "timing")


class LiveTradeManagementExecutionEfficiencyAuditError(Exception):
    """Raised when live trade management audit synthesis fails."""


@dataclass
class LiveTradeManagementExecutionEfficiencyAuditReport:
    """Live trade management and execution efficiency audit output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    mfe_mae_summary: dict[str, Any]
    entry_efficiency: dict[str, Any]
    signal_timing: dict[str, Any]
    stop_quality: dict[str, Any]
    exit_structures: dict[str, Any]
    optimal_targets: dict[str, Any]
    capture_leakage: dict[str, Any]
    regime_trade_management: dict[str, Any]
    deployment_playbook: dict[str, Any]
    deployment_reconciliation: dict[str, Any]
    production_scores: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise LiveTradeManagementExecutionEfficiencyAuditError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = int(pct * max(len(sorted_vals) - 1, 0))
    return round(sorted_vals[idx], 2)


def _resolve_stop_extended(
    signal: dict[str, Any],
    stop_variant: str,
    *,
    cohort_mae_median: float,
) -> float:
    if stop_variant == "fixed_15":
        return 15.0
    return _resolve_stop_points(signal, stop_variant, cohort_mae_median=cohort_mae_median)


def _mfe_mae_distribution(signals: list[dict[str, Any]]) -> dict[str, Any]:
    mfes = [float(s.get("mfe_points") or 0.0) for s in signals]
    maes = [float(s.get("mae_points") or 0.0) for s in signals]
    return {
        "sample_size": len(signals),
        "avg_mfe": round(mean(mfes), 2) if mfes else 0.0,
        "median_mfe": round(median(mfes), 2) if mfes else 0.0,
        "p90_mfe": _percentile(mfes, 0.9),
        "avg_mae": round(mean(maes), 2) if maes else 0.0,
        "median_mae": round(median(maes), 2) if maes else 0.0,
        "p90_mae": _percentile(maes, 0.9),
    }


def _entry_efficiency_analysis(signals: list[dict[str, Any]], *, side: str) -> dict[str, Any]:
    per_signal: list[dict[str, Any]] = []
    efficiencies: list[float] = []
    lost_before: list[float] = []
    captured_after: list[float] = []

    for signal in signals:
        mfe = float(signal.get("mfe_points") or 0.0)
        pts_before = float(signal.get("points_before_expansion") or 0.0)
        max_available = mfe + max(pts_before, 0.0)
        efficiency = round(100.0 * mfe / max(max_available, 1.0), 2)
        efficiencies.append(efficiency)
        lost_before.append(max(pts_before, 0.0))
        captured_after.append(mfe)
        per_signal.append(
            {
                "timestamp": signal.get("timestamp"),
                "points_before_expansion": round(pts_before, 2),
                "mfe_from_entry": round(mfe, 2),
                "max_move_from_start": round(max_available, 2),
                "entry_efficiency_pct": efficiency,
                "points_lost_before_entry": round(max(pts_before, 0.0), 2),
                "points_captured_after_entry": round(mfe, 2),
            },
        )

    return {
        "side": side,
        "methodology": (
            "Move start proxied by points_before_expansion; max_available = MFE + points_before; "
            "efficiency = MFE / max_available. Best entry assumed at move origin (0 pts before)."
        ),
        "aggregate": {
            "avg_entry_efficiency_pct": round(mean(efficiencies), 2) if efficiencies else 0.0,
            "median_entry_efficiency_pct": round(median(efficiencies), 2) if efficiencies else 0.0,
            "p90_entry_efficiency_pct": _percentile(efficiencies, 0.9),
            "avg_points_lost_before_entry": round(mean(lost_before), 2) if lost_before else 0.0,
            "median_points_lost_before_entry": round(median(lost_before), 2) if lost_before else 0.0,
            "avg_points_captured_after_entry": round(mean(captured_after), 2) if captured_after else 0.0,
            "median_points_captured_after_entry": round(median(captured_after), 2) if captured_after else 0.0,
        },
        "per_signal_sample": per_signal[:25],
    }


def _timing_bucket(signals: list[dict[str, Any]]) -> dict[str, Any]:
    before = during = after = no_link = 0
    for signal in signals:
        bars = signal.get("bars_before_expansion")
        if bars is None:
            no_link += 1
        elif int(bars) > 0:
            before += 1
        elif int(bars) == 0:
            during += 1
        else:
            after += 1
    total = len(signals)
    return {
        "before_momentum_pct": round(100.0 * before / max(total, 1), 2),
        "at_momentum_pct": round(100.0 * during / max(total, 1), 2),
        "after_momentum_pct": round(100.0 * after / max(total, 1), 2),
        "no_linked_move_pct": round(100.0 * no_link / max(total, 1), 2),
        "counts": {"before": before, "at": during, "after": after, "no_link": no_link},
    }


def _extended_metrics(
    pnls: list[float],
    *,
    signals: list[dict[str, Any]],
    sample_size: int,
    window_days: int,
) -> dict[str, Any]:
    base = _metrics_from_pnls(pnls, sample_size=sample_size, window_days=window_days)
    mfes = [float(s.get("mfe_points") or 0.0) for s in signals]
    maes = [float(s.get("mae_points") or 0.0) for s in signals]
    months = max(window_days / 22.0, 1.0)
    total_mfe = sum(mfes)
    total_captured = sum(max(p, 0.0) for p in pnls)
    return {
        **base,
        "avg_mfe": round(mean(mfes), 2) if mfes else 0.0,
        "avg_mae": round(mean(maes), 2) if maes else 0.0,
        "monthly_points": round(base["realized_profit_points"] / months, 2),
        "capture_efficiency_pct": round(100.0 * total_captured / max(total_mfe, 1.0), 2),
    }


def _classify_miss_reason(
    signal: dict[str, Any],
    structure: dict[str, Any],
    *,
    stop_pts: float,
    pnl: float,
) -> str:
    mfe = float(signal.get("mfe_points") or 0.0)
    mae = float(signal.get("mae_points") or 0.0)
    pts_before = float(signal.get("points_before_expansion") or 0.0)
    bars = signal.get("bars_before_expansion")

    if pts_before > 10.0 or (bars is not None and int(bars) < 0):
        return "timing"
    if pnl <= 0 and mae >= stop_pts:
        return "stop"
    if structure.get("runner"):
        t2 = float(structure["t2"])
        if mfe > t2 and pnl < mfe * 0.85:
            return "runner"
    t1 = float(structure["t1"])
    if mfe >= t1 and pnl < mfe * 0.9:
        if not structure.get("runner") and structure.get("t3"):
            return "target_structure"
        return "early_exit"
    if mfe > 0 and pnl < mfe:
        return "target_structure"
    return "early_exit"


def _stop_quality_matrix(
    signals: list[dict[str, Any]],
    *,
    structure: dict[str, Any],
    window_days: int,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    rows: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for variant in EXECUTION_STOP_VARIANTS:
        pnls: list[float] = []
        avg_stop = 0.0
        for signal in signals:
            stop_pts = _resolve_stop_extended(signal, variant, cohort_mae_median=mae_median)
            avg_stop += stop_pts
            pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
            pnls.append(pnl)
        metrics = _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=window_days)
        row = {
            "stop_variant": variant,
            "average_stop_points": round(avg_stop / max(len(signals), 1), 2),
            **metrics,
        }
        rows[variant] = row
        ranking.append(
            {
                **row,
                "optimization_score": round(metrics["expectancy"] * (metrics["profit_factor"] or 0.0), 2),
            },
        )

    best = max(
        ranking,
        key=lambda item: (item["expectancy"], item["profit_factor"] or 0.0, item["capture_efficiency_pct"]),
    )
    return {
        "by_stop_variant": rows,
        "ranking": sorted(ranking, key=lambda item: item["optimization_score"], reverse=True),
        "best_stop_variant": best["stop_variant"],
        "best_stop_evidence": best,
    }


def _fixed_exit_matrix(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for target in FIXED_EXIT_TARGETS:
        pnls: list[float] = []
        captured_total = 0.0
        missed_total = 0.0
        for signal in signals:
            win, pnl = _fixed_target_pnl(signal, target)
            mfe = float(signal.get("mfe_points") or 0.0)
            pnls.append(pnl)
            captured_total += max(pnl, 0.0)
            missed_total += max(0.0, mfe - max(pnl, 0.0))
        metrics = _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=window_days)
        row = {
            "target_points": target,
            "captured_points_total": round(captured_total, 2),
            "missed_points_total": round(missed_total, 2),
            **metrics,
        }
        rows[str(target)] = row
        ranking.append({**row, "optimization_score": round(metrics["expectancy"] * (metrics["profit_factor"] or 0.0), 2)})

    best = max(ranking, key=lambda item: (item["expectancy"], item["profit_factor"] or 0.0))
    return {
        "by_target": rows,
        "ranking": sorted(ranking, key=lambda item: item["optimization_score"], reverse=True),
        "best_fixed_target": best["target_points"],
        "best_fixed_evidence": best,
    }


def _exit_structure_matrix(
    signals: list[dict[str, Any]],
    *,
    stop_variant: str,
    window_days: int,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    rows: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for label, structure in EXIT_STRUCTURES.items():
        pnls: list[float] = []
        captured_total = 0.0
        missed_total = 0.0
        for signal in signals:
            stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
            pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
            mfe = float(signal.get("mfe_points") or 0.0)
            pnls.append(pnl)
            captured_total += max(pnl, 0.0)
            missed_total += max(0.0, mfe - max(pnl, 0.0))
        metrics = _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=window_days)
        row = {
            "structure": label,
            "tiers": structure,
            "captured_points_total": round(captured_total, 2),
            "missed_points_total": round(missed_total, 2),
            **metrics,
        }
        rows[label] = row
        ranking.append({**row, "optimization_score": round(metrics["expectancy"] * (metrics["profit_factor"] or 0.0), 2)})

    best = max(ranking, key=lambda item: (item["expectancy"], item["profit_factor"] or 0.0, item["capture_efficiency_pct"]))
    return {
        "stop_variant_used": stop_variant,
        "by_structure": rows,
        "ranking": sorted(ranking, key=lambda item: item["optimization_score"], reverse=True),
        "best_structure": best["structure"],
        "best_structure_evidence": best,
    }


def _optimal_tiers(structure: dict[str, Any]) -> dict[str, Any]:
    return {
        "T1_points": structure["t1"],
        "T2_points": structure["t2"],
        "T3_points": structure.get("t3"),
        "runner": bool(structure.get("runner")),
        "leg_weights": list(LEG_WEIGHTS),
    }


def _capture_leakage_analysis(
    signals: list[dict[str, Any]],
    *,
    structure: dict[str, Any],
    stop_variant: str,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    max_available_total = 0.0
    actual_total = 0.0
    missed_total = 0.0
    reason_counts: Counter[str] = Counter()

    for signal in signals:
        mfe = float(signal.get("mfe_points") or 0.0)
        stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
        pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
        max_available_total += mfe
        actual_total += max(pnl, 0.0)
        missed = max(0.0, mfe - max(pnl, 0.0))
        missed_total += missed
        if missed > 0.01:
            reason_counts[_classify_miss_reason(signal, structure, stop_pts=stop_pts, pnl=pnl)] += 1

    efficiency = round(100.0 * actual_total / max(max_available_total, 1.0), 2)
    ranked_reasons = [
        {"reason": reason, "count": count, "rank": index + 1}
        for index, (reason, count) in enumerate(reason_counts.most_common())
    ]

    return {
        "max_available_points": round(max_available_total, 2),
        "actual_captured_points": round(actual_total, 2),
        "missed_points": round(missed_total, 2),
        "capture_efficiency_pct": efficiency,
        "miss_reason_ranking": ranked_reasons,
        "structure_used": structure,
        "stop_variant_used": stop_variant,
    }


def _regime_trade_management(
    signals: list[dict[str, Any]],
    *,
    direction: str,
    window_days: int,
) -> dict[str, Any]:
    by_trend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_vol: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for signal in signals:
        regime = classify_signal_regime(signal, direction=direction)
        by_trend[regime["trend"]].append(signal)
        by_vol[regime["volatility"]].append(signal)

    def _best_for_cohort(cohort: list[dict[str, Any]]) -> dict[str, Any]:
        if not cohort:
            return {"sample_size": 0}
        stop_matrix = _stop_quality_matrix(cohort, structure=EXIT_STRUCTURES["40/80/Runner"], window_days=window_days)
        exit_matrix = _exit_structure_matrix(
            cohort,
            stop_variant=stop_matrix["best_stop_variant"],
            window_days=window_days,
        )
        return {
            "sample_size": len(cohort),
            "best_stop": stop_matrix["best_stop_variant"],
            "best_exit_structure": exit_matrix["best_structure"],
            "best_stop_evidence": stop_matrix["best_stop_evidence"],
            "best_exit_evidence": exit_matrix["best_structure_evidence"],
            "optimal_tiers": _optimal_tiers(EXIT_STRUCTURES[exit_matrix["best_structure"]]),
        }

    trend_results = {regime: _best_for_cohort(cohort) for regime, cohort in by_trend.items()}
    vol_results = {regime: _best_for_cohort(cohort) for regime, cohort in by_vol.items()}

    return {
        "direction": direction,
        "by_trend_regime": trend_results,
        "by_volatility_regime": vol_results,
        "composite_note": "Full composite throttle from regime_detection_audit; trend/vol slices for management tuning",
    }


def _throttle_lookup(throttle_rules: list[dict[str, Any]]) -> dict[str, str]:
    return {row["regime"]: row["throttle"] for row in throttle_rules}


def _combined_monthly_projection(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    *,
    buy_structure: dict[str, Any],
    sell_structure: dict[str, Any],
    buy_stop: str,
    sell_stop: str,
    sell_throttle_rules: list[dict[str, Any]],
    window_days: int,
    sell_weight_mode: str,
) -> dict[str, Any]:
    throttle_map = _throttle_lookup(sell_throttle_rules)
    buy_mae_med = median(float(s.get("mae_points") or 0.0) for s in buy_signals) if buy_signals else 0.0
    sell_mae_med = median(float(s.get("mae_points") or 0.0) for s in sell_signals) if sell_signals else 0.0
    pnls: list[float] = []

    for signal in buy_signals:
        stop_pts = _resolve_stop_extended(signal, buy_stop, cohort_mae_median=buy_mae_med)
        pnl, _ = _tiered_structure_pnl(signal, buy_structure, stop_pts=stop_pts)
        pnls.append(pnl)

    for signal in sell_signals:
        regime = classify_signal_regime(signal, direction="SELL")
        throttle = throttle_map.get(regime["composite"], "FULL")
        if sell_weight_mode == "regime_adaptive" and throttle == "BLOCK":
            continue
        weight = THROTTLE_WEIGHT.get(throttle, 1.0) if sell_weight_mode == "regime_adaptive" else 1.0
        stop_pts = _resolve_stop_extended(signal, sell_stop, cohort_mae_median=sell_mae_med)
        pnl, _ = _tiered_structure_pnl(signal, sell_structure, stop_pts=stop_pts)
        pnls.append(round(pnl * weight, 2))

    months = max(window_days / 22.0, 1.0)
    total = round(sum(pnls), 2)
    return {
        "trade_count": len(pnls),
        "total_points": total,
        "monthly_points": round(total / months, 2),
        "max_drawdown_points": _max_drawdown(pnls),
        "profit_factor": _profit_factor_from_pnls(pnls),
        "capture_efficiency_pct": round(
            100.0 * sum(max(p, 0.0) for p in pnls) / max(sum(float(s.get("mfe_points") or 0.0) for s in buy_signals + sell_signals), 1.0),
            2,
        ),
    }


def _build_deployment_config(
    *,
    mode: str,
    buy_stop: str,
    sell_stop: str,
    buy_structure_label: str,
    sell_structure_label: str,
    buy_single_target: int,
    sell_sizing: str,
    buy_sizing: str,
    throttle_rules: list[dict[str, Any]],
    playbook: dict[str, Any],
    deployment_audit: dict[str, Any],
) -> dict[str, Any]:
    buy_structure = EXIT_STRUCTURES[buy_structure_label]
    sell_structure = EXIT_STRUCTURES[sell_structure_label]
    combined_risk = playbook.get("combined_playbook", {}).get("risk_rules", {})

    return {
        "deployment_mode": mode,
        "buy_engine": {
            "model_id": BUY_V3_MODEL_ID,
            "formula": BUY_V3_FORMULA_TEXT,
            "entry": "Signal bar close; layer5.pass required; reject bars_before_expansion < 0",
            "stop_variant": buy_stop,
            "target_structure": buy_structure_label,
            "T1_T2_T3_runner": _optimal_tiers(buy_structure),
            "single_target_fallback_points": buy_single_target,
            "sizing_mode": buy_sizing,
        },
        "sell_engine": {
            "model_id": SELL_V6_MODEL_ID,
            "vwap_gate": V6_VWAP_GATE_RULE,
            "entry": "Signal bar close; VWAP Below only; layer5.pass required",
            "stop_variant": sell_stop,
            "target_structure": sell_structure_label,
            "T1_T2_T3_runner": _optimal_tiers(sell_structure),
            "sizing_mode": sell_sizing,
            "regime_throttle": throttle_rules,
        },
        "conflict_policy": "NO_TRADE on same-bar opposing signals",
        "risk_rules": combined_risk,
        "reconciled_with": deployment_audit.get("final_answer", {}).get("deployment_tier"),
    }


def _reconcile_with_deployment_audit(
    *,
    deployment_audit: dict[str, Any],
    playbook: dict[str, Any],
    buy_best_stop: str,
    sell_best_stop: str,
    buy_best_structure: str,
    sell_best_structure: str,
) -> dict[str, Any]:
    deploy_mgmt = deployment_audit.get("trade_management", {})
    deploy_playbook = deployment_audit.get("deployment_playbook", {})
    playbook_stops = playbook.get("combined_playbook", {}).get("stop_rules", {})
    playbook_targets = playbook.get("combined_playbook", {}).get("target_rules", {})

    return {
        "final_production_deployment_audit_path": str(SOURCE_EXPORTS["final_production_deployment_audit"]),
        "deployment_audit_recommendations": deploy_mgmt.get("playbook_recommendations", {}),
        "this_audit_recommendations": {
            "buy_stop": buy_best_stop,
            "sell_stop": sell_best_stop,
            "buy_structure": buy_best_structure,
            "sell_structure": sell_best_structure,
        },
        "playbook_export_recommendations": {
            "buy_stop": playbook_stops.get("buy_variant"),
            "sell_stop": playbook_stops.get("sell_variant"),
            "buy_structure": playbook_targets.get("buy_structure"),
            "sell_structure": playbook_targets.get("sell_structure"),
        },
        "alignment": {
            "buy_structure_matches_playbook": buy_best_structure == playbook_targets.get("buy_structure"),
            "sell_structure_matches_playbook": sell_best_structure == playbook_targets.get("sell_structure"),
            "stop_note": (
                "Playbook optimizes fixed_10 for simulated WR/PF; deployment audit recommends "
                "structure_based for live risk sizing — paper may log fixed_10 sensitivity separately."
            ),
        },
        "deployment_playbook_checklist": deploy_playbook.get("paper_trading_checklist", []),
        "paper_trade_verdict": deployment_audit.get("final_answer", {}).get("paper_trade_tomorrow"),
        "real_capital_verdict": deployment_audit.get("final_answer", {}).get("real_capital_deployment"),
    }


def _production_scores(
    *,
    regime_audit: dict[str, Any],
    deployment_audit: dict[str, Any],
    paper_projection: dict[str, Any],
    real_projection: dict[str, Any],
) -> dict[str, Any]:
    regime_scores = regime_audit.get("final_answer", {}).get("output_metrics", {})
    deploy_scores = deployment_audit.get("production_scores", {})

    readiness = round(
        float(deploy_scores.get("production_readiness_score") or regime_scores.get("production_readiness_score") or 75.0),
        1,
    )
    confidence = round(
        float(deploy_scores.get("confidence_score") or regime_scores.get("confidence_score") or 70.0),
        1,
    )
    risk = round(
        float(deploy_scores.get("production_risk_score") or regime_scores.get("production_risk_score") or 65.0),
        1,
    )

    return {
        "production_readiness_score": readiness,
        "confidence_score": confidence,
        "production_risk_score": risk,
        "paper_trading": {
            "expected_monthly_points": paper_projection["monthly_points"],
            "max_drawdown_points": paper_projection["max_drawdown_points"],
            "capture_efficiency_pct": paper_projection["capture_efficiency_pct"],
        },
        "real_capital": {
            "expected_monthly_points": real_projection["monthly_points"],
            "max_drawdown_points": real_projection["max_drawdown_points"],
            "capture_efficiency_pct": real_projection["capture_efficiency_pct"],
        },
    }


def _final_answer(
    *,
    scores: dict[str, Any],
    paper_config: dict[str, Any],
    real_config: dict[str, Any],
    buy_stop_analysis: dict[str, Any],
    sell_stop_analysis: dict[str, Any],
    buy_exit_analysis: dict[str, Any],
    sell_exit_analysis: dict[str, Any],
    deployment_audit: dict[str, Any],
    regime_audit: dict[str, Any],
) -> dict[str, Any]:
    deploy_final = deployment_audit.get("final_answer", {})
    regime_final = regime_audit.get("final_answer", {})

    return {
        "paper_trading_config": paper_config,
        "real_capital_config": real_config,
        "optimal_stops": {
            "buy_v3": buy_stop_analysis["best_stop_variant"],
            "sell_v6": sell_stop_analysis["best_stop_variant"],
        },
        "optimal_exit_structures": {
            "buy_v3": buy_exit_analysis["best_structure"],
            "sell_v6": sell_exit_analysis["best_structure"],
        },
        "expected_monthly_points": {
            "paper_combined": scores["paper_trading"]["expected_monthly_points"],
            "real_capital_combined": scores["real_capital"]["expected_monthly_points"],
        },
        "expected_drawdown_points": {
            "paper_combined": scores["paper_trading"]["max_drawdown_points"],
            "real_capital_combined": scores["real_capital"]["max_drawdown_points"],
        },
        "capture_efficiency_pct": {
            "paper_combined": scores["paper_trading"]["capture_efficiency_pct"],
            "real_capital_combined": scores["real_capital"]["capture_efficiency_pct"],
        },
        "production_readiness_score": scores["production_readiness_score"],
        "confidence_score": scores["confidence_score"],
        "production_risk_score": scores["production_risk_score"],
        "paper_trade_tomorrow": deploy_final.get("paper_trade_tomorrow", regime_final.get("paper_trading_verdict", "PARTIAL")),
        "real_capital_deployment": deploy_final.get("real_capital_deployment", "NO"),
        "rationale": (
            f"Paper: {paper_config['buy_engine']['stop_variant']}/{paper_config['sell_engine']['stop_variant']} stops, "
            f"{paper_config['buy_engine']['target_structure']} BUY / {paper_config['sell_engine']['target_structure']} SELL, "
            f"~{scores['paper_trading']['expected_monthly_points']} pts/mo. "
            f"Real: structure_based stops, regime_adaptive SELL throttle, "
            f"~{scores['real_capital']['expected_monthly_points']} pts/mo at lower DD."
        ),
    }


class LiveTradeManagementExecutionEfficiencyAuditResearch:
    """Synthesize live trade management and execution efficiency audit from exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in SOURCE_EXPORTS.items():
            loaded[name] = {
                "path": str(path),
                "status": "loaded" if path.exists() else "missing",
                "data": _load_json(path, required=True),
            }
        self.sources = loaded
        return loaded

    def run(self) -> LiveTradeManagementExecutionEfficiencyAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        buy_export = sources["buy_v3_candidate_validation"]["data"]
        sell_export = sources["sell_v6_replay_validation"]["data"]
        playbook = sources["production_trading_playbook_audit"]["data"]
        regime_audit = sources["regime_detection_audit"]["data"]
        deployment_audit = sources["final_production_deployment_audit"]["data"]

        window_days = int(
            buy_export.get("trading_days_replayed") or sell_export.get("trading_days_replayed") or 120,
        )

        buy_signals = list(buy_export.get("per_signal_details", {}).get("buy_v3") or [])
        sell_signals = list(sell_export.get("per_signal_details", {}).get("sell_v6") or [])
        if not buy_signals:
            raise LiveTradeManagementExecutionEfficiencyAuditError("No BUY_V3 per_signal_details in exports.")
        if not sell_signals:
            raise LiveTradeManagementExecutionEfficiencyAuditError("No SELL_V6 per_signal_details in exports.")

        sell_throttle_rules = regime_audit.get("throttle_recommendation", {}).get("sell_v6_regime_throttle", [])

        mfe_mae_summary = {
            "buy_v3": _mfe_mae_distribution(buy_signals),
            "sell_v6": _mfe_mae_distribution(sell_signals),
        }
        entry_efficiency = {
            "buy_v3": _entry_efficiency_analysis(buy_signals, side="BUY"),
            "sell_v6": _entry_efficiency_analysis(sell_signals, side="SELL"),
        }
        signal_timing = {
            "methodology": "bars_before_expansion > 0 = before momentum; == 0 = at; < 0 = after",
            "buy_v3": _timing_bucket(buy_signals),
            "sell_v6": _timing_bucket(sell_signals),
        }

        default_structure = EXIT_STRUCTURES["40/80/Runner"]
        buy_stop_quality = _stop_quality_matrix(buy_signals, structure=default_structure, window_days=window_days)
        sell_stop_quality = _stop_quality_matrix(sell_signals, structure=default_structure, window_days=window_days)

        buy_fixed_exits = _fixed_exit_matrix(buy_signals, window_days=window_days)
        sell_fixed_exits = _fixed_exit_matrix(sell_signals, window_days=window_days)

        buy_exit_structures = _exit_structure_matrix(
            buy_signals,
            stop_variant=buy_stop_quality["best_stop_variant"],
            window_days=window_days,
        )
        sell_exit_structures = _exit_structure_matrix(
            sell_signals,
            stop_variant=sell_stop_quality["best_stop_variant"],
            window_days=window_days,
        )

        optimal_targets = {
            "buy_v3": {
                "best_fixed_target": buy_fixed_exits["best_fixed_target"],
                "best_tiered_structure": buy_exit_structures["best_structure"],
                "optimal_tiers": _optimal_tiers(EXIT_STRUCTURES[buy_exit_structures["best_structure"]]),
                "best_stop_for_structures": buy_stop_quality["best_stop_variant"],
            },
            "sell_v6": {
                "best_fixed_target": sell_fixed_exits["best_fixed_target"],
                "best_tiered_structure": sell_exit_structures["best_structure"],
                "optimal_tiers": _optimal_tiers(EXIT_STRUCTURES[sell_exit_structures["best_structure"]]),
                "best_stop_for_structures": sell_stop_quality["best_stop_variant"],
            },
        }

        buy_capture = _capture_leakage_analysis(
            buy_signals,
            structure=EXIT_STRUCTURES[buy_exit_structures["best_structure"]],
            stop_variant=buy_stop_quality["best_stop_variant"],
        )
        sell_capture = _capture_leakage_analysis(
            sell_signals,
            structure=EXIT_STRUCTURES[sell_exit_structures["best_structure"]],
            stop_variant=sell_stop_quality["best_stop_variant"],
        )

        regime_trade_management = {
            "buy_v3": _regime_trade_management(buy_signals, direction="BUY", window_days=window_days),
            "sell_v6": _regime_trade_management(sell_signals, direction="SELL", window_days=window_days),
        }

        buy_single_target = int(
            playbook.get("buy_v3_playbook", {})
            .get("target_rules", {})
            .get("recommended_single_target_points")
            or buy_fixed_exits["best_fixed_target"]
            or 60,
        )

        paper_buy_stop = "fixed_10"
        paper_sell_stop = "fixed_10"
        real_buy_stop = "structure_based"
        real_sell_stop = "structure_based"

        paper_config = _build_deployment_config(
            mode="paper_trading",
            buy_stop=paper_buy_stop,
            sell_stop=paper_sell_stop,
            buy_structure_label=buy_exit_structures["best_structure"],
            sell_structure_label=sell_exit_structures["best_structure"],
            buy_single_target=buy_single_target,
            sell_sizing="regime_adaptive",
            buy_sizing="regime_adaptive",
            throttle_rules=sell_throttle_rules,
            playbook=playbook,
            deployment_audit=deployment_audit,
        )
        real_config = _build_deployment_config(
            mode="real_capital",
            buy_stop=real_buy_stop,
            sell_stop=real_sell_stop,
            buy_structure_label=buy_exit_structures["best_structure"],
            sell_structure_label=sell_exit_structures["best_structure"],
            buy_single_target=buy_single_target,
            sell_sizing="regime_adaptive",
            buy_sizing="half",
            throttle_rules=sell_throttle_rules,
            playbook=playbook,
            deployment_audit=deployment_audit,
        )

        paper_projection = _combined_monthly_projection(
            buy_signals,
            sell_signals,
            buy_structure=EXIT_STRUCTURES[buy_exit_structures["best_structure"]],
            sell_structure=EXIT_STRUCTURES[sell_exit_structures["best_structure"]],
            buy_stop=paper_buy_stop,
            sell_stop=paper_sell_stop,
            sell_throttle_rules=sell_throttle_rules,
            window_days=window_days,
            sell_weight_mode="regime_adaptive",
        )
        real_projection = _combined_monthly_projection(
            buy_signals,
            sell_signals,
            buy_structure=EXIT_STRUCTURES[buy_exit_structures["best_structure"]],
            sell_structure=EXIT_STRUCTURES[sell_exit_structures["best_structure"]],
            buy_stop=real_buy_stop,
            sell_stop=real_sell_stop,
            sell_throttle_rules=sell_throttle_rules,
            window_days=window_days,
            sell_weight_mode="regime_adaptive",
        )
        real_projection["monthly_points"] = round(real_projection["monthly_points"] * 0.5, 2)
        real_projection["total_points"] = round(real_projection["total_points"] * 0.5, 2)

        deployment_reconciliation = _reconcile_with_deployment_audit(
            deployment_audit=deployment_audit,
            playbook=playbook,
            buy_best_stop=buy_stop_quality["best_stop_variant"],
            sell_best_stop=sell_stop_quality["best_stop_variant"],
            buy_best_structure=buy_exit_structures["best_structure"],
            sell_best_structure=sell_exit_structures["best_structure"],
        )

        production_scores = _production_scores(
            regime_audit=regime_audit,
            deployment_audit=deployment_audit,
            paper_projection=paper_projection,
            real_projection=real_projection,
        )

        final_answer = _final_answer(
            scores=production_scores,
            paper_config=paper_config,
            real_config=real_config,
            buy_stop_analysis=buy_stop_quality,
            sell_stop_analysis=sell_stop_quality,
            buy_exit_analysis=buy_exit_structures,
            sell_exit_analysis=sell_exit_structures,
            deployment_audit=deployment_audit,
            regime_audit=regime_audit,
        )

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "no_new_replay": True,
            "no_new_discovery": True,
            "no_new_models": True,
            "no_new_indicators": True,
            "simulation_basis": (
                "Stops/exits simulated from per_signal_details MFE/MAE, entry/stop_loss prices, "
                "points_before_expansion, and bars_before_expansion — no intrabar sequencing."
            ),
            "stop_variants": list(EXECUTION_STOP_VARIANTS),
            "fixed_exit_targets": list(FIXED_EXIT_TARGETS),
            "tiered_exit_structures": list(EXIT_STRUCTURES.keys()),
            "tiered_exit_method": (
                "Three-leg partial exits (33% each); runner leg uses remaining MFE beyond T2; "
                "loss when MFE fails next tier and MAE exceeds stop proxy."
            ),
            "capture_leakage_method": "miss = MFE - max(simulated_pnl, 0); reasons ranked by frequency",
            "regime_dimensions": {"trend": list(TREND_REGIMES), "volatility": list(VOL_REGIMES)},
            "production_gates": PRODUCTION_GATES,
            "engines": {"buy_v3": BUY_V3_MODEL_ID, "sell_v6": SELL_V6_MODEL_ID},
        }

        limitations = [
            "All metrics synthesized from five JSON exports — no new replay.",
            "MFE/MAE proxy does not model intrabar stop/target hit ordering.",
            "Entry efficiency uses points_before_expansion as move-start proxy.",
            "Real-capital projection applies 50% BUY sizing haircut on structure_based stops.",
            "SELL throttle map from regime_detection_audit uses validate-window PF (retrospective fit risk).",
            "Reconciled with final_production_deployment_audit.json recommendations.",
        ]

        conclusions = [
            "Live trade management audit synthesized from replay exports only.",
            (
                f"BUY_V3 MFE median {mfe_mae_summary['buy_v3']['median_mfe']} / "
                f"MAE median {mfe_mae_summary['buy_v3']['median_mae']}; "
                f"entry efficiency {entry_efficiency['buy_v3']['aggregate']['median_entry_efficiency_pct']}%."
            ),
            (
                f"SELL_V6 MFE median {mfe_mae_summary['sell_v6']['median_mfe']} / "
                f"MAE median {mfe_mae_summary['sell_v6']['median_mae']}; "
                f"timing before momentum {signal_timing['sell_v6']['before_momentum_pct']}%."
            ),
            (
                f"Optimal BUY stop: {buy_stop_quality['best_stop_variant']} | "
                f"Optimal SELL stop: {sell_stop_quality['best_stop_variant']}."
            ),
            (
                f"Optimal BUY exit: {buy_exit_structures['best_structure']} | "
                f"Optimal SELL exit: {sell_exit_structures['best_structure']}."
            ),
            (
                f"Capture leakage top miss: BUY {buy_capture['miss_reason_ranking'][:1]} | "
                f"SELL {sell_capture['miss_reason_ranking'][:1]}."
            ),
            (
                f"Paper ~{production_scores['paper_trading']['expected_monthly_points']} pts/mo | "
                f"Real ~{production_scores['real_capital']['expected_monthly_points']} pts/mo."
            ),
            f"Paper trade: {final_answer['paper_trade_tomorrow']} | Real capital: {final_answer['real_capital_deployment']}.",
        ]

        return LiveTradeManagementExecutionEfficiencyAuditReport(
            report_type="Live Trade Management & Execution Efficiency Audit",
            engines=["BUY_V3", "SELL_V6", "COMBINED"],
            symbol=buy_export.get("symbol") or "NIFTY50",
            timeframe=buy_export.get("timeframe") or "5M",
            trading_days_replayed=window_days,
            replay_start_date=buy_export.get("replay_start_date", ""),
            replay_end_date=buy_export.get("replay_end_date", ""),
            methodology=methodology,
            source_exports={name: {"path": info["path"], "status": info["status"]} for name, info in sources.items()},
            limitations=limitations,
            mfe_mae_summary=mfe_mae_summary,
            entry_efficiency=entry_efficiency,
            signal_timing=signal_timing,
            stop_quality={"buy_v3": buy_stop_quality, "sell_v6": sell_stop_quality},
            exit_structures={
                "fixed_targets": {"buy_v3": buy_fixed_exits, "sell_v6": sell_fixed_exits},
                "tiered_structures": {"buy_v3": buy_exit_structures, "sell_v6": sell_exit_structures},
            },
            optimal_targets=optimal_targets,
            capture_leakage={"buy_v3": buy_capture, "sell_v6": sell_capture},
            regime_trade_management=regime_trade_management,
            deployment_playbook={
                "paper_trading": paper_config,
                "real_capital": real_config,
                "combined_projection": {"paper": paper_projection, "real_capital": real_projection},
            },
            deployment_reconciliation=deployment_reconciliation,
            production_scores=production_scores,
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: LiveTradeManagementExecutionEfficiencyAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Live trade management execution efficiency audit exported to %s", self.report_path)
        return self.report_path


def generate_live_trade_management_execution_efficiency_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export live trade management execution efficiency audit JSON."""
    return LiveTradeManagementExecutionEfficiencyAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_live_trade_management_execution_efficiency_audit_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    print(f"Exported: {path}")
    print(f"Paper monthly pts: {final['expected_monthly_points']['paper_combined']}")
    print(f"Real monthly pts: {final['expected_monthly_points']['real_capital_combined']}")
    print(f"Optimal stops: BUY {final['optimal_stops']['buy_v3']} | SELL {final['optimal_stops']['sell_v6']}")
