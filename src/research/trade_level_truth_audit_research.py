"""
Trade Level Truth Audit — synthesis from existing replay JSON exports only.

Trade-level evidence audit for BUY_V3 and SELL_V6 per-signal records, target
achievement, conditional probabilities, lifecycle, entry precision, and V4/V7
potential. Research-only; no replay, indicators, models, or discovery.
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
from src.research.buy_v3_candidate_validation_research import BAR_MINUTES, BUY_V3_MODEL_ID
from src.research.live_trade_management_execution_efficiency_audit_research import (
    _classify_miss_reason,
    _resolve_stop_extended,
)
from src.research.production_edge_enhancement_audit_research import (
    LOSER_CLASSIFICATIONS,
    TRAP_CLASSIFICATIONS,
    _classify_sell_signal,
    _is_buy_winner,
    _is_sell_winner,
    _map_buy_audit_classification,
    _profit_factor_from_pnls,
)
from src.research.production_reality_audit_research import (
    MFE_TIERS,
    RUNNER_STRATEGIES,
    _mfe_tier_distribution,
    _runner_exit_optimization,
    _signal_reality_analysis,
    _target_achievement_matrix,
    _timing_class,
)
from src.research.production_trading_playbook_audit_research import (
    LEG_WEIGHTS,
    _metrics_from_pnls,
    _tiered_structure_pnl,
)
from src.research.regime_detection_audit_research import SELL_V6_MODEL_ID

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "trade_level_truth_audit.json"

CONDITIONAL_TIERS = (40, 60, 100, 150, 200)
TIMING_CLASSES = ("Very Early", "Early", "Same", "Late", "No Linked Move")
LIFECYCLE_OUTCOMES = (
    "Stopped Out",
    "T1 Only",
    "T2 Only",
    "T3",
    "Runner",
    "Full Trend Capture",
)
PF_IMPROVEMENT_THRESHOLD_PCT = 10.0
FULL_CAPTURE_RATIO = 0.85

SOURCE_EXPORTS = {
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "live_trade_management_execution_efficiency_audit": RESEARCH_DIR
    / "live_trade_management_execution_efficiency_audit.json",
    "production_reality_audit": RESEARCH_DIR / "production_reality_audit.json",
}

OPTIONAL_EXPORTS = {
    "extended_evidence_validation_real_deployment_audit": RESEARCH_DIR
    / "extended_evidence_validation_real_deployment_audit.json",
}


class TradeLevelTruthAuditError(Exception):
    """Raised when trade level truth audit synthesis fails."""


@dataclass
class TradeLevelTruthAuditReport:
    """Trade level truth audit output."""

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
    per_signal_records: dict[str, Any]
    target_achievement_matrix: dict[str, Any]
    conditional_probability: dict[str, Any]
    trade_lifecycle_analysis: dict[str, Any]
    entry_precision_audit: dict[str, Any]
    buy_v4_sell_v7_potential: dict[str, Any]
    uncaptured_edge: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise TradeLevelTruthAuditError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _tier_reached(signal: dict[str, Any], threshold: int) -> bool:
    tiers = signal.get("mfe_capture_tiers")
    if isinstance(tiers, dict) and str(threshold) in tiers:
        return bool(tiers[str(threshold)])
    return float(signal.get("mfe_points") or 0.0) >= threshold


def _estimate_time_to_tier(signal: dict[str, Any], threshold: int) -> float | None:
    if not _tier_reached(signal, threshold):
        return None
    duration = signal.get("trade_duration_bars")
    mfe = float(signal.get("mfe_points") or 0.0)
    if duration is not None and mfe > 0:
        return round(float(duration) * min(1.0, threshold / mfe) * BAR_MINUTES, 2)
    return round(BAR_MINUTES * 3, 2)


def _derive_exit_points(signal: dict[str, Any], *, pnl: float, side: str) -> float:
    entry = float(signal.get("entry") or 0.0)
    if side == "BUY":
        return round(entry + pnl, 2)
    return round(entry - pnl, 2)


def _classify_buy_loser(signal: dict[str, Any]) -> str:
    if _is_buy_winner(signal):
        return "Winner"
    return _map_buy_audit_classification(str(signal.get("classification") or "Unknown"))


def _classify_lifecycle_outcome(
    signal: dict[str, Any],
    *,
    structure: dict[str, Any],
    stop_pts: float,
    pnl: float,
) -> str:
    mfe = float(signal.get("mfe_points") or 0.0)
    t1 = float(structure["t1"])
    t2 = float(structure["t2"])
    t3 = structure.get("t3")
    captured = max(pnl, 0.0)

    if mfe < t1 or pnl < 0:
        return "Stopped Out"
    if mfe >= t1 and captured >= mfe * FULL_CAPTURE_RATIO:
        return "Full Trend Capture"
    if structure.get("runner") and mfe > t2:
        if captured >= mfe * FULL_CAPTURE_RATIO:
            return "Full Trend Capture"
        return "Runner"
    if t3 is not None and mfe >= float(t3):
        return "T3"
    if mfe >= t2:
        return "T2 Only"
    if mfe >= t1:
        return "T1 Only"
    return "Stopped Out"


def _per_signal_record(
    signal: dict[str, Any],
    *,
    side: str,
    structure: dict[str, Any],
    stop_variant: str,
    cohort_mae_median: float,
    win_fn: Any,
    classify_fn: Any,
) -> dict[str, Any]:
    stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=cohort_mae_median)
    pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
    mfe = float(signal.get("mfe_points") or 0.0)
    mae = float(signal.get("mae_points") or 0.0)
    bars = signal.get("bars_before_expansion")
    bars_int = int(bars) if bars is not None else None

    return {
        "signal_timestamp": signal.get("timestamp"),
        "move_start_timestamp": signal.get("move_start_time"),
        "entry": signal.get("entry"),
        "stop": signal.get("stop_loss"),
        "exit": _derive_exit_points(signal, pnl=pnl, side=side),
        "exit_pnl_points": round(pnl, 2),
        "mfe": round(mfe, 2),
        "mae": round(mae, 2),
        "final_outcome": classify_fn(signal) if not win_fn(signal) else "Winner",
        "lifecycle_outcome": _classify_lifecycle_outcome(signal, structure=structure, stop_pts=stop_pts, pnl=pnl),
        "is_winner": win_fn(signal),
        "bars_before_expansion": bars_int,
        "lead_time_minutes": round(bars_int * BAR_MINUTES, 2) if bars_int is not None and bars_int > 0 else 0,
        "timing_class": _timing_class(bars_int),
        "classification": signal.get("classification"),
    }


def _build_per_signal_records(
    signals: list[dict[str, Any]],
    *,
    side: str,
    structure: dict[str, Any],
    stop_variant: str,
    win_fn: Any,
    classify_fn: Any,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    records = [
        _per_signal_record(
            signal,
            side=side,
            structure=structure,
            stop_variant=stop_variant,
            cohort_mae_median=mae_median,
            win_fn=win_fn,
            classify_fn=classify_fn,
        )
        for signal in signals
    ]
    return {
        "side": side,
        "sample_size": len(records),
        "playbook_structure": structure,
        "stop_variant": stop_variant,
        "records": records,
    }


def _trade_level_target_matrix(
    signals: list[dict[str, Any]],
    *,
    side: str,
    structure: dict[str, Any],
    stop_variant: str,
) -> dict[str, Any]:
    total = len(signals)
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    rows: dict[str, Any] = {}

    for threshold in MFE_TIERS:
        hits = [s for s in signals if _tier_reached(s, threshold)]
        count = len(hits)
        times = [_estimate_time_to_tier(s, threshold) for s in hits]
        times_valid = [t for t in times if t is not None]
        failures = 0
        for signal in hits:
            stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
            pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
            if pnl <= 0:
                failures += 1

        rows[str(threshold)] = {
            "count": count,
            "percentage_pct": round(100.0 * count / max(total, 1), 2),
            "probability_pct": round(100.0 * count / max(total, 1), 2),
            "avg_time_to_reach_minutes": round(mean(times_valid), 2) if times_valid else None,
            "avg_failure_rate_pct": round(100.0 * failures / max(count, 1), 2) if count else 0.0,
        }

    return {
        "side": side,
        "sample_size": total,
        "playbook_structure": structure,
        "stop_variant": stop_variant,
        "by_tier": rows,
        "methodology": (
            "Count/percentage from mfe_capture_tiers when present else mfe_points >= tier. "
            "Avg time proxied from trade_duration_bars * (tier/mfe). "
            "Failure rate = tier hits with non-positive playbook PnL."
        ),
    }


def _reached_before_stop(
    signal: dict[str, Any],
    threshold: int,
    *,
    stop_pts: float,
    cohort_mae_median: float,
    stop_variant: str,
) -> bool:
    if not _tier_reached(signal, threshold):
        return False
    resolved_stop = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=cohort_mae_median)
    mae = float(signal.get("mae_points") or 0.0)
    return mae < resolved_stop or float(signal.get("mfe_points") or 0.0) >= threshold


def _conditional_probability_analysis(
    signals: list[dict[str, Any]],
    *,
    side: str,
    structure: dict[str, Any],
    stop_variant: str,
) -> dict[str, Any]:
    total = len(signals)
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    stop_pts = 10.0
    tiers: dict[str, Any] = {}

    for threshold in CONDITIONAL_TIERS:
        reached = sum(
            1
            for signal in signals
            if _reached_before_stop(
                signal,
                threshold,
                stop_pts=stop_pts,
                cohort_mae_median=mae_median,
                stop_variant=stop_variant,
            )
        )
        tiers[str(threshold)] = {
            "count_reached_before_stop": reached,
            "probability_pct": round(100.0 * reached / max(total, 1), 2),
            "label": f"P(reach {threshold}+ before stop | signal fires)",
        }

    return {
        "side": side,
        "sample_size": total,
        "stop_variant": stop_variant,
        "methodology": (
            "P(reach tier before stop | signal fires): tier reached when mfe >= tier "
            "and MAE stayed below resolved stop (fixed_10/structure proxy)."
        ),
        "tiers": tiers,
        "summary": {
            "p_40_plus": tiers["40"]["probability_pct"],
            "p_60_plus": tiers["60"]["probability_pct"],
            "p_100_plus": tiers["100"]["probability_pct"],
            "p_150_plus": tiers["150"]["probability_pct"],
            "p_200_plus": tiers["200"]["probability_pct"],
        },
    }


def _trade_lifecycle_analysis(
    signals: list[dict[str, Any]],
    *,
    side: str,
    structure: dict[str, Any],
    stop_variant: str,
    win_fn: Any,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_signal: list[dict[str, Any]] = []

    for signal in signals:
        stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
        pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
        mfe = float(signal.get("mfe_points") or 0.0)
        captured = max(pnl, 0.0)
        missed = max(0.0, mfe - captured)
        outcome = _classify_lifecycle_outcome(signal, structure=structure, stop_pts=stop_pts, pnl=pnl)
        row = {
            "timestamp": signal.get("timestamp"),
            "lifecycle_outcome": outcome,
            "captured_points": round(captured, 2),
            "missed_points": round(missed, 2),
            "max_possible_points": round(mfe, 2),
            "capture_efficiency_pct": round(100.0 * captured / max(mfe, 1.0), 2),
            "is_winner": win_fn(signal),
        }
        by_outcome[outcome].append(row)
        per_signal.append(row)

    outcome_summary: dict[str, Any] = {}
    for label in LIFECYCLE_OUTCOMES:
        cohort = by_outcome.get(label, [])
        if not cohort:
            outcome_summary[label] = {
                "count": 0,
                "percentage_pct": 0.0,
                "avg_captured_points": 0.0,
                "avg_missed_points": 0.0,
                "avg_max_possible_points": 0.0,
                "avg_capture_efficiency_pct": 0.0,
            }
            continue
        outcome_summary[label] = {
            "count": len(cohort),
            "percentage_pct": round(100.0 * len(cohort) / max(len(signals), 1), 2),
            "avg_captured_points": round(mean(r["captured_points"] for r in cohort), 2),
            "avg_missed_points": round(mean(r["missed_points"] for r in cohort), 2),
            "avg_max_possible_points": round(mean(r["max_possible_points"] for r in cohort), 2),
            "avg_capture_efficiency_pct": round(mean(r["capture_efficiency_pct"] for r in cohort), 2),
        }

    total_mfe = sum(float(s.get("mfe_points") or 0.0) for s in signals)
    total_captured = sum(r["captured_points"] for r in per_signal)

    return {
        "side": side,
        "sample_size": len(signals),
        "playbook_structure": structure,
        "stop_variant": stop_variant,
        "by_outcome": outcome_summary,
        "aggregate": {
            "avg_captured_points": round(mean(r["captured_points"] for r in per_signal), 2) if per_signal else 0.0,
            "avg_missed_points": round(mean(r["missed_points"] for r in per_signal), 2) if per_signal else 0.0,
            "avg_max_possible_points": round(mean(r["max_possible_points"] for r in per_signal), 2) if per_signal else 0.0,
            "capture_efficiency_pct": round(100.0 * total_captured / max(total_mfe, 1.0), 2),
        },
        "per_signal_details": per_signal,
    }


def _entry_precision_audit(
    signals: list[dict[str, Any]],
    *,
    side: str,
    win_fn: Any,
) -> dict[str, Any]:
    reality = _signal_reality_analysis(signals, side=side, win_fn=win_fn, window_days=120)
    per_signal: list[dict[str, Any]] = []

    for signal in signals:
        bars = signal.get("bars_before_expansion")
        bars_int = int(bars) if bars is not None else None
        timing = _timing_class(bars_int)
        pnl = float(signal.get("realized_pnl_points") or 0.0)
        per_signal.append(
            {
                "signal_timestamp": signal.get("timestamp"),
                "move_start_timestamp": signal.get("move_start_time"),
                "lead_time_bars": bars_int,
                "lead_time_minutes": round(bars_int * BAR_MINUTES, 2) if bars_int is not None and bars_int > 0 else 0,
                "timing_class": timing,
                "is_winner": win_fn(signal),
                "pnl_points": round(pnl, 2),
                "predictive_vs_reactive": "predictive" if timing in {"Very Early", "Early"} else "reactive",
            },
        )

    timing_summary = reality["timing_class_summary"]
    timing_metrics: dict[str, Any] = {}
    for label in ("Very Early", "Early", "Same", "Late"):
        cohort = [s for s in signals if _timing_class(
            int(s["bars_before_expansion"]) if s.get("bars_before_expansion") is not None else None,
        ) == label]
        if not cohort:
            timing_metrics[label] = {
                "count": 0,
                "win_rate_pct": 0.0,
                "profit_factor": None,
                "expectancy": 0.0,
            }
            continue
        pnls = [float(s.get("realized_pnl_points") or 0.0) for s in cohort]
        timing_metrics[label] = {
            "count": len(cohort),
            "win_rate_pct": round(100.0 * sum(1 for s in cohort if win_fn(s)) / len(cohort), 2),
            "profit_factor": _profit_factor_from_pnls(pnls),
            "expectancy": round(mean(pnls), 2),
        }

    return {
        "side": side,
        "methodology": reality["methodology"],
        "timing_class_summary": timing_summary,
        "timing_class_metrics": timing_metrics,
        "predictive_vs_reactive": reality["predictive_vs_reactive"],
        "per_signal_details": per_signal,
    }


def _pf_if_class_removed(
    signals: list[dict[str, Any]],
    *,
    class_label: str,
    classify_fn: Any,
) -> tuple[float | None, float | None]:
    baseline_pnls = [float(s.get("realized_pnl_points") or 0.0) for s in signals]
    baseline_pf = _profit_factor_from_pnls(baseline_pnls)
    filtered_pnls = [
        float(s.get("realized_pnl_points") or 0.0)
        for s in signals
        if classify_fn(s) != class_label
    ]
    filtered_pf = _profit_factor_from_pnls(filtered_pnls)
    if baseline_pf is None or filtered_pf is None or baseline_pf == 0:
        return baseline_pf, None
    improvement_pct = round(100.0 * (filtered_pf - baseline_pf) / baseline_pf, 2)
    return baseline_pf, improvement_pct


def _losing_class_analysis(
    signals: list[dict[str, Any]],
    *,
    side: str,
    classify_fn: Any,
    is_winner_fn: Any,
) -> dict[str, Any]:
    losers = [s for s in signals if not is_winner_fn(s)]
    baseline_pnls = [float(s.get("realized_pnl_points") or 0.0) for s in signals]
    baseline_pf = _profit_factor_from_pnls(baseline_pnls)
    total = len(signals)

    classes: list[dict[str, Any]] = []
    for label in LOSER_CLASSIFICATIONS:
        if label == "Winner":
            continue
        cohort = [s for s in signals if classify_fn(s) == label]
        if not cohort:
            continue
        pnls = [float(s.get("realized_pnl_points") or 0.0) for s in cohort]
        pnl_impact = round(sum(pnls), 2)
        _, pf_impact_pct = _pf_if_class_removed(signals, class_label=label, classify_fn=classify_fn)
        classes.append(
            {
                "class": label,
                "count": len(cohort),
                "frequency_pct": round(100.0 * len(cohort) / max(total, 1), 2),
                "pnl_impact_points": pnl_impact,
                "pf_impact_if_removed_pct": pf_impact_pct,
                "avg_loss_points": round(mean(pnls), 2),
                "is_trap": label in TRAP_CLASSIFICATIONS,
            },
        )

    classes.sort(key=lambda row: (row["pf_impact_if_removed_pct"] or 0.0, -row["pnl_impact_points"]), reverse=True)
    best_improvement = max((row["pf_impact_if_removed_pct"] or 0.0) for row in classes) if classes else 0.0
    can_improve = best_improvement >= PF_IMPROVEMENT_THRESHOLD_PCT

    return {
        "side": side,
        "baseline_profit_factor": baseline_pf,
        "loser_count": len(losers),
        "unexplained_losing_classes": classes,
        "best_pf_improvement_if_class_removed_pct": round(best_improvement, 2),
        "can_improve_pf_over_10pct": can_improve,
        "recommendation": "YES" if can_improve else "NO",
        "evidence": (
            f"Removing top loser class improves PF by {best_improvement}% (threshold {PF_IMPROVEMENT_THRESHOLD_PCT}%)."
            if can_improve
            else (
                f"No single loser class removal improves PF by >= {PF_IMPROVEMENT_THRESHOLD_PCT}% "
                f"(best: {best_improvement}%). Execution/lifecycle optimization preferred."
            )
        ),
        "top_classes_if_yes": [row for row in classes if (row["pf_impact_if_removed_pct"] or 0) >= PF_IMPROVEMENT_THRESHOLD_PCT],
    }


def _buy_v4_sell_v7_potential(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    buy_analysis = _losing_class_analysis(
        buy_signals,
        side="BUY_V3",
        classify_fn=_classify_buy_loser,
        is_winner_fn=_is_buy_winner,
    )
    sell_analysis = _losing_class_analysis(
        sell_signals,
        side="SELL_V6",
        classify_fn=_classify_sell_signal,
        is_winner_fn=_is_sell_winner,
    )
    return {
        "methodology": (
            "Unexplained losing trade classes ranked by PF impact if class removed. "
            f"YES if any class removal improves PF >= {PF_IMPROVEMENT_THRESHOLD_PCT}%."
        ),
        "buy_v4": buy_analysis,
        "sell_v7": sell_analysis,
    }


def _uncaptured_edge(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    *,
    buy_structure: dict[str, Any],
    sell_structure: dict[str, Any],
    buy_stop: str,
    sell_stop: str,
    window_days: int,
) -> dict[str, Any]:
    def _side_edge(
        signals: list[dict[str, Any]],
        *,
        side: str,
        structure: dict[str, Any],
        stop_variant: str,
    ) -> dict[str, Any]:
        mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
        pnls: list[float] = []
        mfes: list[float] = []
        captured: list[float] = []

        for signal in signals:
            stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
            pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
            mfe = float(signal.get("mfe_points") or 0.0)
            pnls.append(pnl)
            mfes.append(mfe)
            captured.append(max(pnl, 0.0))

        metrics = _metrics_from_pnls(pnls, sample_size=len(signals), window_days=window_days)
        total_mfe = sum(mfes)
        total_captured = sum(captured)
        theoretical_captured = total_mfe
        wr = metrics["win_rate_pct"]
        pf = metrics["profit_factor"]
        capture_pct = round(100.0 * total_captured / max(total_mfe, 1.0), 2)

        runner_opt = _runner_exit_optimization(signals, side=side, stop_variant=stop_variant, window_days=window_days)
        best = runner_opt["by_strategy"].get(runner_opt["best_strategy"], {})
        best_pf = best.get("profit_factor")
        best_wr = best.get("win_rate_pct")
        best_capture = best.get("capture_efficiency_pct")

        return {
            "side": side,
            "current": {
                "profit_factor": pf,
                "win_rate_pct": wr,
                "capture_efficiency_pct": capture_pct,
                "expectancy": metrics["expectancy"],
            },
            "theoretical_maximum": {
                "profit_factor_if_full_mfe_capture": _profit_factor_from_pnls(mfes),
                "win_rate_pct_if_all_positive_mfe": round(
                    100.0 * sum(1 for m in mfes if m > 0) / max(len(mfes), 1),
                    2,
                ),
                "capture_efficiency_pct": 100.0,
                "uncaptured_points": round(theoretical_captured - total_captured, 2),
            },
            "best_runner_strategy": {
                "strategy": runner_opt["best_strategy"],
                "profit_factor": best_pf,
                "win_rate_pct": best_wr,
                "capture_efficiency_pct": best_capture,
                "expectancy": best.get("expectancy"),
            },
            "additional_available": {
                "pf_delta_vs_current": round((best_pf or 0) - (pf or 0), 2) if best_pf and pf else None,
                "wr_delta_pp": round((best_wr or 0) - wr, 2) if best_wr else None,
                "capture_delta_pct": round((best_capture or 0) - capture_pct, 2) if best_capture else None,
            },
        }

    buy_edge = _side_edge(buy_signals, side="BUY", structure=buy_structure, stop_variant=buy_stop)
    sell_edge = _side_edge(sell_signals, side="SELL", structure=sell_structure, stop_variant=sell_stop)

    return {
        "methodology": (
            "Uncaptured edge = gap between current 60/100/Runner playbook capture and "
            "best runner strategy / full MFE theoretical ceiling."
        ),
        "buy_v3": buy_edge,
        "sell_v6": sell_edge,
        "combined": {
            "avg_current_capture_pct": round(
                mean([buy_edge["current"]["capture_efficiency_pct"], sell_edge["current"]["capture_efficiency_pct"]]),
                2,
            ),
            "avg_best_strategy_capture_pct": round(
                mean(
                    [
                        buy_edge["best_runner_strategy"]["capture_efficiency_pct"] or 0,
                        sell_edge["best_runner_strategy"]["capture_efficiency_pct"] or 0,
                    ],
                ),
                2,
            ),
        },
    }


def _build_final_answer(
    *,
    buy_cond: dict[str, Any],
    sell_cond: dict[str, Any],
    buy_lifecycle: dict[str, Any],
    sell_lifecycle: dict[str, Any],
    buy_entry: dict[str, Any],
    sell_entry: dict[str, Any],
    v4_v7: dict[str, Any],
    uncaptured: dict[str, Any],
) -> dict[str, Any]:
    probability_matrix = {
        "buy_v3": buy_cond["summary"],
        "sell_v6": sell_cond["summary"],
    }

    trade_lifecycle_matrix = {
        "buy_v3": {
            label: buy_lifecycle["by_outcome"][label]
            for label in LIFECYCLE_OUTCOMES
        },
        "sell_v6": {
            label: sell_lifecycle["by_outcome"][label]
            for label in LIFECYCLE_OUTCOMES
        },
    }

    entry_quality_matrix = {
        "buy_v3": {
            label: buy_entry["timing_class_metrics"].get(label, {})
            for label in ("Very Early", "Early", "Same", "Late")
        },
        "sell_v6": {
            label: sell_entry["timing_class_metrics"].get(label, {})
            for label in ("Very Early", "Early", "Same", "Late")
        },
        "buy_predictive_verdict": buy_entry["predictive_vs_reactive"]["verdict"],
        "sell_predictive_verdict": sell_entry["predictive_vs_reactive"]["verdict"],
    }

    buy_v4_rec = v4_v7["buy_v4"]["recommendation"]
    sell_v7_rec = v4_v7["sell_v7"]["recommendation"]

    max_improvement = {
        "buy_v3": {
            "profit_factor_delta": uncaptured["buy_v3"]["additional_available"]["pf_delta_vs_current"],
            "win_rate_delta_pp": uncaptured["buy_v3"]["additional_available"]["wr_delta_pp"],
            "capture_delta_pct": uncaptured["buy_v3"]["additional_available"]["capture_delta_pct"],
            "best_strategy": uncaptured["buy_v3"]["best_runner_strategy"]["strategy"],
        },
        "sell_v6": {
            "profit_factor_delta": uncaptured["sell_v6"]["additional_available"]["pf_delta_vs_current"],
            "win_rate_delta_pp": uncaptured["sell_v6"]["additional_available"]["wr_delta_pp"],
            "capture_delta_pct": uncaptured["sell_v6"]["additional_available"]["capture_delta_pct"],
            "best_strategy": uncaptured["sell_v6"]["best_runner_strategy"]["strategy"],
        },
        "combined_capture_headroom_pct": round(
            uncaptured["combined"]["avg_best_strategy_capture_pct"]
            - uncaptured["combined"]["avg_current_capture_pct"],
            2,
        ),
    }

    return {
        "probability_matrix": probability_matrix,
        "trade_lifecycle_matrix": trade_lifecycle_matrix,
        "entry_quality_matrix": entry_quality_matrix,
        "buy_v4_recommendation": buy_v4_rec,
        "sell_v7_recommendation": sell_v7_rec,
        "maximum_theoretical_improvement": max_improvement,
        "rationale": (
            f"BUY_V4={buy_v4_rec}: {v4_v7['buy_v4']['evidence']} "
            f"SELL_V7={sell_v7_rec}: {v4_v7['sell_v7']['evidence']}"
        ),
    }


class TradeLevelTruthAuditResearch:
    """Synthesize trade level truth audit from existing exports."""

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
        for name, path in OPTIONAL_EXPORTS.items():
            loaded[name] = {
                "path": str(path),
                "status": "loaded" if path.exists() else "missing",
                "data": _load_json(path, required=False),
            }
        self.sources = loaded
        return loaded

    def run(self) -> TradeLevelTruthAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        buy_export = sources["buy_v3_candidate_validation"]["data"]
        sell_export = sources["sell_v6_replay_validation"]["data"]
        live_audit = sources["live_trade_management_execution_efficiency_audit"]["data"]
        reality_audit = sources["production_reality_audit"]["data"]

        window_days = int(
            buy_export.get("trading_days_replayed") or sell_export.get("trading_days_replayed") or 120,
        )

        buy_signals = list(buy_export.get("per_signal_details", {}).get("buy_v3") or [])
        sell_signals = list(sell_export.get("per_signal_details", {}).get("sell_v6") or [])
        if not buy_signals:
            raise TradeLevelTruthAuditError("No BUY_V3 per_signal_details in exports.")
        if not sell_signals:
            raise TradeLevelTruthAuditError("No SELL_V6 per_signal_details in exports.")

        live_final = live_audit.get("final_answer", {})
        buy_stop = live_final.get("optimal_stops", {}).get("buy_v3", "fixed_10")
        sell_stop = live_final.get("optimal_stops", {}).get("sell_v6", "fixed_10")
        buy_structure = RUNNER_STRATEGIES["60_100_runner"]
        sell_structure = RUNNER_STRATEGIES["60_100_runner"]

        per_signal_records = {
            "buy_v3": _build_per_signal_records(
                buy_signals,
                side="BUY",
                structure=buy_structure,
                stop_variant=buy_stop,
                win_fn=_is_buy_winner,
                classify_fn=_classify_buy_loser,
            ),
            "sell_v6": _build_per_signal_records(
                sell_signals,
                side="SELL",
                structure=sell_structure,
                stop_variant=sell_stop,
                win_fn=_is_sell_winner,
                classify_fn=_classify_sell_signal,
            ),
        }

        target_achievement_matrix = {
            "buy_v3": _trade_level_target_matrix(
                buy_signals,
                side="BUY",
                structure=buy_structure,
                stop_variant=buy_stop,
            ),
            "sell_v6": _trade_level_target_matrix(
                sell_signals,
                side="SELL",
                structure=sell_structure,
                stop_variant=sell_stop,
            ),
            "playbook_capture_matrix": {
                "buy_v3": _target_achievement_matrix(
                    buy_signals,
                    structure=buy_structure,
                    stop_variant=buy_stop,
                    window_days=window_days,
                    side="BUY",
                ),
                "sell_v6": _target_achievement_matrix(
                    sell_signals,
                    structure=sell_structure,
                    stop_variant=sell_stop,
                    window_days=window_days,
                    side="SELL",
                ),
            },
            "mfe_tier_distribution": {
                "buy_v3": _mfe_tier_distribution(buy_signals),
                "sell_v6": _mfe_tier_distribution(sell_signals),
            },
        }

        conditional_probability = {
            "buy_v3": _conditional_probability_analysis(
                buy_signals,
                side="BUY",
                structure=buy_structure,
                stop_variant=buy_stop,
            ),
            "sell_v6": _conditional_probability_analysis(
                sell_signals,
                side="SELL",
                structure=sell_structure,
                stop_variant=sell_stop,
            ),
        }

        trade_lifecycle_analysis = {
            "buy_v3": _trade_lifecycle_analysis(
                buy_signals,
                side="BUY",
                structure=buy_structure,
                stop_variant=buy_stop,
                win_fn=_is_buy_winner,
            ),
            "sell_v6": _trade_lifecycle_analysis(
                sell_signals,
                side="SELL",
                structure=sell_structure,
                stop_variant=sell_stop,
                win_fn=_is_sell_winner,
            ),
        }

        entry_precision_audit = {
            "buy_v3": _entry_precision_audit(buy_signals, side="BUY", win_fn=_is_buy_winner),
            "sell_v6": _entry_precision_audit(sell_signals, side="SELL", win_fn=_is_sell_winner),
        }

        buy_v4_sell_v7_potential = _buy_v4_sell_v7_potential(buy_signals, sell_signals)

        uncaptured_edge = _uncaptured_edge(
            buy_signals,
            sell_signals,
            buy_structure=buy_structure,
            sell_structure=sell_structure,
            buy_stop=buy_stop,
            sell_stop=sell_stop,
            window_days=window_days,
        )

        final_answer = _build_final_answer(
            buy_cond=conditional_probability["buy_v3"],
            sell_cond=conditional_probability["sell_v6"],
            buy_lifecycle=trade_lifecycle_analysis["buy_v3"],
            sell_lifecycle=trade_lifecycle_analysis["sell_v6"],
            buy_entry=entry_precision_audit["buy_v3"],
            sell_entry=entry_precision_audit["sell_v6"],
            v4_v7=buy_v4_sell_v7_potential,
            uncaptured=uncaptured_edge,
        )

        extended_status = sources["extended_evidence_validation_real_deployment_audit"]["status"]
        limitations = [
            "Synthesis-only: no new replay; all metrics from completed JSON exports.",
            "Avg time-to-tier proxied from trade_duration_bars — intrabar sequencing not modeled.",
            "Conditional P(tier before stop) uses MFE/MAE proxy; live fill path may differ.",
            f"extended_evidence_validation_real_deployment_audit: {extended_status}.",
            "BUY_V3 lacks mfe_capture_tiers in export — tier hits derived from mfe_points.",
        ]

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "engines": [BUY_V3_MODEL_ID, SELL_V6_MODEL_ID],
            "playbook": "60/100/Runner + optimal stops from live_trade_management audit",
            "per_signal_source": "buy_v3_candidate_validation.per_signal_details.buy_v3, "
            "sell_v6_replay_validation.per_signal_details.sell_v6",
            "excluded": "deployment readiness scores, capital gates, research closure verdicts",
        }

        conclusions = [
            (
                f"BUY_V3 n={len(buy_signals)} | SELL_V6 n={len(sell_signals)} trade-level records synthesized."
            ),
            (
                f"P(40+ before stop): BUY {conditional_probability['buy_v3']['summary']['p_40_plus']}% | "
                f"SELL {conditional_probability['sell_v6']['summary']['p_40_plus']}%."
            ),
            (
                f"P(100+ before stop): BUY {conditional_probability['buy_v3']['summary']['p_100_plus']}% | "
                f"SELL {conditional_probability['sell_v6']['summary']['p_100_plus']}%."
            ),
            (
                f"Lifecycle capture: BUY {trade_lifecycle_analysis['buy_v3']['aggregate']['capture_efficiency_pct']}% | "
                f"SELL {trade_lifecycle_analysis['sell_v6']['aggregate']['capture_efficiency_pct']}%."
            ),
            (
                f"Entry timing: BUY {entry_precision_audit['buy_v3']['predictive_vs_reactive']['verdict']} | "
                f"SELL {entry_precision_audit['sell_v6']['predictive_vs_reactive']['verdict']}."
            ),
            (
                f"BUY_V4 recommendation: {final_answer['buy_v4_recommendation']} | "
                f"SELL_V7 recommendation: {final_answer['sell_v7_recommendation']}."
            ),
            (
                f"Uncaptured capture headroom: {uncaptured_edge['combined']['avg_best_strategy_capture_pct']}% best strategy "
                f"vs {uncaptured_edge['combined']['avg_current_capture_pct']}% current."
            ),
        ]

        source_exports = {
            name: {"path": meta["path"], "status": meta["status"]}
            for name, meta in sources.items()
        }

        return TradeLevelTruthAuditReport(
            report_type="Trade Level Truth Audit",
            engines=["BUY_V3", "SELL_V6"],
            symbol=str(buy_export.get("symbol") or "NIFTY50"),
            timeframe=str(buy_export.get("timeframe") or "5M"),
            trading_days_replayed=window_days,
            replay_start_date=str(buy_export.get("replay_start_date") or ""),
            replay_end_date=str(buy_export.get("replay_end_date") or ""),
            methodology=methodology,
            source_exports=source_exports,
            limitations=limitations,
            per_signal_records=per_signal_records,
            target_achievement_matrix=target_achievement_matrix,
            conditional_probability=conditional_probability,
            trade_lifecycle_analysis=trade_lifecycle_analysis,
            entry_precision_audit=entry_precision_audit,
            buy_v4_sell_v7_potential=buy_v4_sell_v7_potential,
            uncaptured_edge=uncaptured_edge,
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: TradeLevelTruthAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Trade level truth audit exported to %s", self.report_path)
        return self.report_path


def generate_trade_level_truth_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export trade level truth audit JSON."""
    return TradeLevelTruthAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_trade_level_truth_audit_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    prob = final["probability_matrix"]
    print(f"Exported: {path}")
    print(
        f"BUY P(40+/60+/100+): {prob['buy_v3']['p_40_plus']}% / "
        f"{prob['buy_v3']['p_60_plus']}% / {prob['buy_v3']['p_100_plus']}%"
    )
    print(
        f"SELL P(40+/60+/100+): {prob['sell_v6']['p_40_plus']}% / "
        f"{prob['sell_v6']['p_60_plus']}% / {prob['sell_v6']['p_100_plus']}%"
    )
    print(f"BUY_V4: {final['buy_v4_recommendation']} | SELL_V7: {final['sell_v7_recommendation']}")
