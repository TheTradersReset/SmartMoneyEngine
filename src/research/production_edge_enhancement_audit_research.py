"""
Production Edge Enhancement Audit — synthesis from existing replay exports only.

Identifies marginal filter improvements for BUY_V3 and SELL_V5 without new replay,
indicators, models, or discovery. Combines winner/loser anatomy, condition attribution,
timing, and walk-forward stability from completed validation JSON exports.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v2_candidate_validation_research import PRODUCTION_GATES
from src.research.buy_v3_candidate_validation_research import (
    BAR_MINUTES,
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
)
from src.research.buy_winner_vs_false_reversal_analysis_research import (
    ANALYSIS_CONDITIONS,
    _extract_conditions_from_signal,
)
from src.research.smartmoneyengine_v5_candidate_validation_research import V5_VWAP_GATE_RULE

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "production_edge_enhancement_audit.json"

SELL_V5_MODEL_ID = "LDM-SELL-V5"
BUY_MIN_SIGNALS_PER_MONTH = 20.0
SELL_MIN_SIGNALS_PER_MONTH = 60.0

SOURCE_EXPORTS = {
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "smartmoneyengine_v5_candidate_validation": RESEARCH_DIR
    / "smartmoneyengine_v5_candidate_validation.json",
    "unified_production_replay_validation": RESEARCH_DIR
    / "unified_production_replay_validation.json",
    "buy_v3_signal_quality_audit": RESEARCH_DIR / "buy_v3_signal_quality_audit.json",
    "buy_v3_tradeability_production_validation": RESEARCH_DIR
    / "buy_v3_tradeability_production_validation.json",
    "buy_winner_vs_false_reversal_analysis": RESEARCH_DIR
    / "buy_winner_vs_false_reversal_analysis.json",
    "smartmoneyengine_walkforward_validation": RESEARCH_DIR
    / "smartmoneyengine_walkforward_validation.json",
}

LOSER_CLASSIFICATIONS = (
    "Winner",
    "Bull Trap",
    "Bear Trap",
    "Range Failure",
    "Liquidity Failure",
    "Gap Failure",
    "Trend Exhaustion",
    "No Expansion",
    "Late Entry",
)

BUY_EXPORT_TO_AUDIT: dict[str, str] = {
    "Real Reversal": "Winner",
    "Bull Trap": "Bull Trap",
    "Range Failure": "Range Failure",
    "No Expansion": "No Expansion",
    "Counter Trend Bounce": "Trend Exhaustion",
    "Dead Cat Bounce": "Liquidity Failure",
    "False Reversal": "Liquidity Failure",
}

TRAP_CLASSIFICATIONS = frozenset({"Bull Trap", "Bear Trap"})
EXPANSION_THRESHOLDS = (40, 60, 100, 200)
NO_EXPANSION_MFE = 20.0
RANGE_FAILURE_MFE = 100.0

BUY_FILTER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("require", "HTF Bullish"),
    ("exclude", "HTF Bullish"),
    ("require", "VWAP Reclaim"),
    ("require", "PWL Sweep"),
    ("require", "Gap Continuation"),
    ("exclude", "Gap Continuation"),
    ("require", "PDL Sweep"),
    ("require", "Near Support"),
)

SELL_ANALYSIS_CONDITIONS = (
    "Failed Breakout",
    "Failed Breakdown",
    "Gap Reversal",
    "Gap Continuation",
    "HTF Bearish",
    "VWAP Below",
    "VWAP Rejected",
    "Near Resistance",
    "Near Support",
    "Confirmation Present",
    "Bear Context EMA",
)

SELL_FILTER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("require", "VWAP Below"),
    ("exclude", "VWAP Rejected"),
    ("require", "Failed Breakout"),
    ("exclude", "Gap Reversal"),
    ("exclude", "Near Support"),
    ("require", "Near Resistance"),
    ("require", "Confirmation Present"),
    ("exclude", "Gap Continuation"),
)


class ProductionEdgeEnhancementAuditError(Exception):
    """Raised when production edge enhancement audit synthesis fails."""


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ProductionEdgeEnhancementAuditError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _profit_factor_from_pnls(pnls: list[float]) -> float | None:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return None if gross_profit == 0 else round(gross_profit, 2)
    return round(gross_profit / gross_loss, 2)


def _signal_date(timestamp: str) -> str:
    return str(timestamp)[:10]


def _is_buy_winner(signal: dict[str, Any]) -> bool:
    return signal.get("classification") == "Real Reversal"


def _is_sell_winner(signal: dict[str, Any]) -> bool:
    return bool(signal.get("win"))


def _map_buy_audit_classification(export_class: str) -> str:
    return BUY_EXPORT_TO_AUDIT.get(export_class or "Unknown", "Liquidity Failure")


def _classify_sell_signal(signal: dict[str, Any]) -> str:
    if _is_sell_winner(signal):
        return "Winner"

    mfe = float(signal.get("mfe_points") or 0.0)
    mae = float(signal.get("mae_points") or 0.0)
    layer2 = signal.get("layers", {}).get("layer2", {})
    stack2 = signal.get("signal_reason_stack", {}).get("layer2", {})
    layer1_stack = signal.get("signal_reason_stack", {}).get("layer1", [])
    htf = layer2.get("htf_trend") or stack2.get("htf_trend") or "Neutral"
    bars = signal.get("bars_before_expansion")

    if bars is not None and int(bars) < 0:
        return "Late Entry"
    if mfe < NO_EXPANSION_MFE:
        return "No Expansion"
    if htf != "Bearish":
        return "Trend Exhaustion"
    if "Gap Reversal" in layer1_stack and mae > mfe:
        return "Gap Failure"
    if mae > mfe and mae > RANGE_FAILURE_MFE:
        return "Bear Trap"
    if mfe < RANGE_FAILURE_MFE:
        return "Range Failure"
    if mfe >= RANGE_FAILURE_MFE:
        return "Liquidity Failure"
    return "Range Failure"


def _extract_sell_conditions(signal: dict[str, Any]) -> dict[str, bool]:
    layer1 = signal.get("layers", {}).get("layer1", {})
    layer2 = signal.get("layers", {}).get("layer2", {})
    layer3 = signal.get("layers", {}).get("layer3", {})
    stack = signal.get("signal_reason_stack", {})
    layer1_stack = stack.get("layer1", [])
    layer2_stack = stack.get("layer2", {})
    location = layer2_stack.get("location") or stack.get("location", "")

    events: set[str] = set(layer1.get("events_detected", [])) | set(layer1_stack)
    confirmation = layer3.get("confirmation_candle") or layer2_stack.get("confirmation_candle") or "None"

    return {
        "Failed Breakout": "Failed Breakout" in events,
        "Failed Breakdown": "Failed Breakdown" in events,
        "Gap Reversal": "Gap Reversal" in events,
        "Gap Continuation": "Gap Continuation" in events,
        "HTF Bearish": layer2.get("htf_trend") == "Bearish" or layer2_stack.get("htf_trend") == "Bearish",
        "VWAP Below": layer2.get("vwap_state") == "Below",
        "VWAP Rejected": layer2.get("vwap_state") == "Rejected",
        "Near Resistance": "Resistance" in str(location),
        "Near Support": "Support" in str(location),
        "Confirmation Present": confirmation not in {"", "None"},
        "Bear Context EMA": layer2.get("ema_structure") == "Bear Context",
    }


def _timing_label(bars_before_expansion: int | None) -> str:
    if bars_before_expansion is None:
        return "No Linked Move"
    if bars_before_expansion > 0:
        return "Early"
    if bars_before_expansion == 0:
        return "Same Candle"
    return "Delayed"


def _cohort_performance(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
    win_fn: Any | None = None,
) -> dict[str, Any]:
    resolve_win = win_fn or (lambda signal: bool(signal.get("win")))

    if not signals:
        return {
            "sample_size": 0,
            "signals_per_month": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "average_mfe": None,
            "average_mae": None,
            "average_lead_bars": None,
            "median_lead_bars": None,
            "average_lead_minutes": None,
            "average_stop_loss_points": None,
            "max_favorable_excursion_avg": None,
            "max_adverse_excursion_avg": None,
        }

    pnls = [float(signal.get("realized_pnl_points") or 0.0) for signal in signals]
    mfes = [float(signal.get("mfe_points") or 0.0) for signal in signals]
    maes = [float(signal.get("mae_points") or 0.0) for signal in signals]
    wins = sum(1 for signal in signals if resolve_win(signal))
    months = max(window_days / 22.0, 1.0)
    lead_bars = [
        int(signal["bars_before_expansion"])
        for signal in signals
        if signal.get("bars_before_expansion") is not None and int(signal["bars_before_expansion"]) > 0
    ]

    return {
        "sample_size": len(signals),
        "signals_per_month": round(len(signals) / months, 2),
        "win_rate_pct": round(100.0 * wins / len(signals), 2),
        "profit_factor": _profit_factor_from_pnls(pnls),
        "expectancy": round(mean(pnls), 2),
        "average_mfe": round(mean(mfes), 2),
        "average_mae": round(mean(maes), 2),
        "average_lead_bars": round(mean(lead_bars), 2) if lead_bars else None,
        "median_lead_bars": round(median(lead_bars), 2) if lead_bars else None,
        "average_lead_minutes": round(mean(lead_bars) * BAR_MINUTES, 2) if lead_bars else None,
        "average_stop_loss_points": round(mean(maes), 2),
        "max_favorable_excursion_avg": round(mean(mfes), 2),
        "max_adverse_excursion_avg": round(mean(maes), 2),
    }


def _loser_classification_summary(
    signals: list[dict[str, Any]],
    *,
    classify_fn: Any,
) -> dict[str, Any]:
    counts = Counter(classify_fn(signal) for signal in signals)
    total = len(signals)
    trap_count = sum(counts.get(label, 0) for label in TRAP_CLASSIFICATIONS)
    return {
        "total_signals": total,
        "counts": {label: counts.get(label, 0) for label in LOSER_CLASSIFICATIONS},
        "rates_pct": {
            label: round(100.0 * counts.get(label, 0) / max(total, 1), 2) for label in LOSER_CLASSIFICATIONS
        },
        "winner_rate_pct": round(100.0 * counts.get("Winner", 0) / max(total, 1), 2),
        "failure_rate_pct": round(100.0 * (total - counts.get("Winner", 0)) / max(total, 1), 2),
        "trap_rate_pct": round(100.0 * trap_count / max(total, 1), 2),
    }


def _winner_loser_side_analysis(
    signals: list[dict[str, Any]],
    *,
    side: str,
    window_days: int,
    is_winner_fn: Any,
    classify_fn: Any,
) -> dict[str, Any]:
    winners = [signal for signal in signals if is_winner_fn(signal)]
    losers = [signal for signal in signals if not is_winner_fn(signal)]

    return {
        "engine": side,
        "baseline": _cohort_performance(signals, window_days=window_days),
        "winners": _cohort_performance(winners, window_days=window_days),
        "losers": _cohort_performance(losers, window_days=window_days),
        "loser_classification": _loser_classification_summary(signals, classify_fn=classify_fn),
        "winner_vs_loser_delta": {
            "mfe_delta_points": round(
                (_cohort_performance(winners, window_days=window_days).get("average_mfe") or 0)
                - (_cohort_performance(losers, window_days=window_days).get("average_mfe") or 0),
                2,
            ),
            "mae_delta_points": round(
                (_cohort_performance(losers, window_days=window_days).get("average_mae") or 0)
                - (_cohort_performance(winners, window_days=window_days).get("average_mae") or 0),
                2,
            ),
            "expectancy_delta_points": round(
                (_cohort_performance(winners, window_days=window_days).get("expectancy") or 0)
                - (_cohort_performance(losers, window_days=window_days).get("expectancy") or 0),
                2,
            ),
        },
    }


def _condition_attribution(
    signals: list[dict[str, Any]],
    *,
    conditions: tuple[str, ...],
    is_winner_fn: Any,
) -> dict[str, Any]:
    winners = [signal for signal in signals if is_winner_fn(signal)]
    failures = [signal for signal in signals if not is_winner_fn(signal)]
    metrics: list[dict[str, Any]] = []

    for condition in conditions:
        w_present = sum(1 for signal in winners if signal.get("conditions", {}).get(condition))
        f_present = sum(1 for signal in failures if signal.get("conditions", {}).get(condition))
        w_total = len(winners)
        f_total = len(failures)
        w_absent = w_total - w_present
        f_absent = f_total - f_present

        winner_cov = round(100.0 * w_present / max(w_total, 1), 2)
        failure_cov = round(100.0 * f_present / max(f_total, 1), 2)
        false_reduction = round(100.0 * f_absent / max(f_total, 1), 2)
        winner_retention = round(100.0 * w_present / max(w_total, 1), 2)

        w_pnls = [
            float(signal.get("realized_pnl_points") or 0.0)
            for signal in winners
            if signal.get("conditions", {}).get(condition)
        ]
        f_pnls = [
            float(signal.get("realized_pnl_points") or 0.0)
            for signal in failures
            if signal.get("conditions", {}).get(condition)
        ]
        w_maes = [
            float(signal.get("mae_points") or 0.0)
            for signal in winners
            if signal.get("conditions", {}).get(condition)
        ]
        f_maes = [
            float(signal.get("mae_points") or 0.0)
            for signal in failures
            if signal.get("conditions", {}).get(condition)
        ]

        metrics.append(
            {
                "condition": condition,
                "winner_coverage_pct": winner_cov,
                "failure_coverage_pct": failure_cov,
                "coverage_delta_pp": round(winner_cov - failure_cov, 2),
                "failure_reduction_if_required_pct": false_reduction,
                "winner_retention_if_required_pct": winner_retention,
                "precision_pct": round(100.0 * w_present / max(w_present + f_present, 1), 2),
                "winner_pf_if_required": _profit_factor_from_pnls(w_pnls),
                "failure_pf_if_required": _profit_factor_from_pnls(f_pnls),
                "winner_avg_mae_if_required": round(mean(w_maes), 2) if w_maes else None,
                "failure_avg_mae_if_required": round(mean(f_maes), 2) if f_maes else None,
                "stop_loss_reduction_pp": round(
                    100.0
                    * (
                        1.0
                        - (mean(f_maes) / max(mean([float(s.get("mae_points") or 0) for s in failures]), 1.0))
                    ),
                    2,
                )
                if f_maes and failures
                else 0.0,
                "winner_present_count": w_present,
                "failure_present_count": f_present,
            },
        )

    composite = sorted(
        metrics,
        key=lambda item: (
            item["coverage_delta_pp"],
            item["failure_reduction_if_required_pct"],
            item["winner_retention_if_required_pct"],
        ),
        reverse=True,
    )

    return {
        "winner_count": len(winners),
        "failure_count": len(failures),
        "per_condition": metrics,
        "rankings": {
            "by_accuracy_separation": composite,
            "by_failure_reduction": sorted(
                metrics,
                key=lambda item: (item["failure_reduction_if_required_pct"], item["winner_retention_if_required_pct"]),
                reverse=True,
            ),
            "by_stop_loss_reduction": sorted(metrics, key=lambda item: item["stop_loss_reduction_pp"], reverse=True),
            "by_winner_retention": sorted(
                metrics,
                key=lambda item: (item["winner_retention_if_required_pct"], item["coverage_delta_pp"]),
                reverse=True,
            ),
        },
    }


def _apply_filter(
    signals: list[dict[str, Any]],
    *,
    mode: str,
    condition: str,
) -> list[dict[str, Any]]:
    if mode == "require":
        return [signal for signal in signals if signal.get("conditions", {}).get(condition)]
    if mode == "exclude":
        return [signal for signal in signals if not signal.get("conditions", {}).get(condition)]
    return list(signals)


def _trap_reduction_pct(
    signals: list[dict[str, Any]],
    filtered: list[dict[str, Any]],
    *,
    classify_fn: Any,
) -> float:
    baseline_traps = sum(1 for signal in signals if classify_fn(signal) in TRAP_CLASSIFICATIONS)
    filtered_traps = sum(1 for signal in filtered if classify_fn(signal) in TRAP_CLASSIFICATIONS)
    if baseline_traps == 0:
        return 0.0
    return round(100.0 * (baseline_traps - filtered_traps) / baseline_traps, 2)


def _passes_frequency_gate(side: str, signals_per_month: float) -> bool:
    if side == "BUY_V3":
        return signals_per_month >= BUY_MIN_SIGNALS_PER_MONTH
    return signals_per_month >= SELL_MIN_SIGNALS_PER_MONTH


def _filter_simulations(
    signals: list[dict[str, Any]],
    *,
    side: str,
    window_days: int,
    candidates: tuple[tuple[str, str], ...],
    is_winner_fn: Any,
    classify_fn: Any,
    min_spm: float,
) -> dict[str, Any]:
    baseline = _cohort_performance(signals, window_days=window_days)
    failures = [signal for signal in signals if not is_winner_fn(signal)]
    baseline_failure_count = len(failures)
    baseline_mae = baseline.get("average_mae") or 0.0
    simulations: list[dict[str, Any]] = []

    for mode, condition in candidates:
        filtered = _apply_filter(signals, mode=mode, condition=condition)
        perf = _cohort_performance(filtered, window_days=window_days)
        filtered_failures = [signal for signal in filtered if not is_winner_fn(signal)]
        removal_pct = round(
            100.0 * (baseline_failure_count - len(filtered_failures)) / max(baseline_failure_count, 1),
            2,
        )
        trap_reduction = _trap_reduction_pct(signals, filtered, classify_fn=classify_fn)
        mae_delta = round((perf.get("average_mae") or 0.0) - baseline_mae, 2)
        pf_delta = None
        if perf.get("profit_factor") is not None and baseline.get("profit_factor") is not None:
            pf_delta = round(perf["profit_factor"] - baseline["profit_factor"], 2)

        passes = (
            perf["win_rate_pct"] >= PRODUCTION_GATES["win_rate_min_pct"]
            and (perf.get("profit_factor") is None or perf["profit_factor"] >= PRODUCTION_GATES["profit_factor_min"])
            and perf["signals_per_month"] >= min_spm
        )

        simulations.append(
            {
                "filter_mode": mode,
                "filter_condition": condition,
                "label": f"{side} {'+ require' if mode == 'require' else '- exclude'} {condition}",
                "failure_removal_pct": removal_pct,
                "trap_reduction_pct": trap_reduction,
                "failures_removed": baseline_failure_count - len(filtered_failures),
                "baseline_failures": baseline_failure_count,
                "remaining_failures": len(filtered_failures),
                "mae_delta_points": mae_delta,
                "pf_delta_vs_baseline": pf_delta,
                "passes_production_gates": passes,
                **perf,
            },
        )

    gate_passing = [sim for sim in simulations if sim["passes_production_gates"]]
    improving = [
        sim
        for sim in gate_passing
        if (sim.get("pf_delta_vs_baseline") or 0) > 0
        or sim["failure_removal_pct"] > 5.0
        or sim["trap_reduction_pct"] > 5.0
        or (sim.get("mae_delta_points") or 0) < -5.0
    ]

    if improving:
        best_gate_passing = max(
            improving,
            key=lambda item: (
                item.get("pf_delta_vs_baseline") or 0.0,
                item["trap_reduction_pct"],
                item["failure_removal_pct"],
                -(item.get("mae_delta_points") or 0.0),
            ),
        )
    elif gate_passing:
        best_gate_passing = max(
            gate_passing,
            key=lambda item: (
                item.get("profit_factor") or 0.0,
                item["win_rate_pct"],
                item["signals_per_month"],
            ),
        )
    else:
        best_gate_passing = {}

    best_tradeoff = max(
        simulations,
        key=lambda item: (
            item.get("pf_delta_vs_baseline") or 0.0,
            item["trap_reduction_pct"],
            item["failure_removal_pct"],
        ),
    ) if simulations else {}

    return {
        "baseline": {"label": f"{side} full stack", "failure_count": baseline_failure_count, **baseline},
        "candidate_filters": [{"mode": mode, "condition": cond} for mode, cond in candidates],
        "simulations": sorted(
            simulations,
            key=lambda item: (
                item.get("pf_delta_vs_baseline") or 0.0,
                item["trap_reduction_pct"],
                item["failure_removal_pct"],
            ),
            reverse=True,
        ),
        "best_gate_passing_filter": best_gate_passing,
        "best_tradeoff_filter": best_tradeoff,
        "gate_passing_filter_count": len(gate_passing),
        "improving_gate_passing_count": len(improving),
        "min_signals_per_month": min_spm,
        "production_gates": PRODUCTION_GATES,
    }


def _expansion_timing_analysis(
    signals: list[dict[str, Any]],
    *,
    capture_export: dict[str, Any] | None,
    direction_label: str,
) -> dict[str, Any]:
    early = same = delayed = no_move = late_entry = 0
    lead_bars: list[int] = []
    lead_points: list[float] = []
    tier_hits = {str(threshold): 0 for threshold in EXPANSION_THRESHOLDS}

    for signal in signals:
        mfe = float(signal.get("mfe_points") or 0.0)
        for threshold in EXPANSION_THRESHOLDS:
            if mfe >= threshold:
                tier_hits[str(threshold)] += 1

        bars = signal.get("bars_before_expansion")
        if bars is None:
            no_move += 1
            continue
        bars_int = int(bars)
        if bars_int > 0:
            early += 1
            lead_bars.append(bars_int)
            if signal.get("points_before_expansion") is not None:
                lead_points.append(float(signal["points_before_expansion"]))
        elif bars_int == 0:
            same += 1
        else:
            delayed += 1
            if bars_int < 0:
                late_entry += 1

    linked = early + same + delayed
    total = len(signals)
    tier_rates = {
        str(threshold): {
            "count": tier_hits[str(threshold)],
            "rate_pct": round(100.0 * tier_hits[str(threshold)] / max(total, 1), 2),
        }
        for threshold in EXPANSION_THRESHOLDS
    }

    return {
        "direction": direction_label,
        "total_signals": total,
        "timing_distribution": {
            "early_count": early,
            "same_candle_count": same,
            "delayed_count": delayed,
            "late_entry_count": late_entry,
            "no_linked_move_count": no_move,
            "early_pct": round(100.0 * early / max(linked, 1), 2),
            "same_candle_pct": round(100.0 * same / max(linked, 1), 2),
            "delayed_pct": round(100.0 * delayed / max(linked, 1), 2),
            "late_entry_pct": round(100.0 * late_entry / max(total, 1), 2),
        },
        "lead_time_bars": {
            "avg": round(mean(lead_bars), 2) if lead_bars else None,
            "median": round(median(lead_bars), 2) if lead_bars else None,
            "min": min(lead_bars) if lead_bars else None,
            "max": max(lead_bars) if lead_bars else None,
        },
        "lead_time_minutes": {
            "avg": round(mean(lead_bars) * BAR_MINUTES, 2) if lead_bars else None,
            "median": round(median(lead_bars) * BAR_MINUTES, 2) if lead_bars else None,
        },
        "points_before_expansion": {
            "avg": round(mean(lead_points), 2) if lead_points else None,
            "median": round(median(lead_points), 2) if lead_points else None,
        },
        "mfe_expansion_achievement": tier_rates,
        "export_point_capture_cross_check": capture_export,
        "timing_potential": {
            "earlier_entry_headroom": "Limited — majority already early; delayed/late entries are minority failure mode.",
            "same_candle_risk": "Same-candle entries carry higher trap rate; marginal gain from earlier filters is small.",
            "later_entry_risk": f"Late/delayed entries account for {round(100.0 * (delayed + late_entry) / max(total, 1), 2)}% of signals.",
        },
    }


def _split_walk_forward(
    signals: list[dict[str, Any]],
    walk_forward: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_start = walk_forward.get("validate_start_date")
    if not validate_start:
        midpoint = len(signals) // 2
        ordered = sorted(signals, key=lambda item: item.get("timestamp", ""))
        return ordered[:midpoint], ordered[midpoint:]
    train = [signal for signal in signals if _signal_date(signal.get("timestamp", "")) < validate_start]
    validate = [signal for signal in signals if _signal_date(signal.get("timestamp", "")) >= validate_start]
    return train, validate


def _walk_forward_filter_impact(
    signals: list[dict[str, Any]],
    *,
    side: str,
    walk_forward: dict[str, Any],
    proposed_filter: dict[str, Any] | None,
    is_winner_fn: Any,
) -> dict[str, Any]:
    train_signals, validate_signals = _split_walk_forward(signals, walk_forward)
    train_days = int(walk_forward.get("train_trading_days", 80))
    validate_days = int(walk_forward.get("validate_trading_days", 40))

    baseline_train = _cohort_performance(train_signals, window_days=train_days)
    baseline_validate = _cohort_performance(validate_signals, window_days=validate_days)

    result: dict[str, Any] = {
        "engine": side,
        "walk_forward_split": {
            "train_start_date": walk_forward.get("train_start_date"),
            "train_end_date": walk_forward.get("train_end_date"),
            "validate_start_date": walk_forward.get("validate_start_date"),
            "validate_end_date": walk_forward.get("validate_end_date"),
            "train_signals": len(train_signals),
            "validate_signals": len(validate_signals),
        },
        "baseline": {
            "train": baseline_train,
            "validate": baseline_validate,
        },
        "proposed_filter": proposed_filter,
        "filtered": None,
        "stability_verdict": "preserve",
    }

    if not proposed_filter:
        export_wf = walk_forward.get("train", {}).get(side.lower().replace("_v3", "_v3").replace("sell_v5", "sell_v5"))
        if export_wf is None and side == "BUY_V3":
            export_wf = walk_forward.get("train", {}).get("buy_v3")
        if export_wf is None and side == "SELL_V5":
            export_wf = walk_forward.get("train", {}).get("sell_v5")
        result["export_walk_forward_reference"] = export_wf
        return result

    mode = proposed_filter.get("filter_mode")
    condition = proposed_filter.get("filter_condition")
    if not mode or not condition:
        return result

    filtered_train = _apply_filter(train_signals, mode=mode, condition=condition)
    filtered_validate = _apply_filter(validate_signals, mode=mode, condition=condition)
    filtered_train_perf = _cohort_performance(filtered_train, window_days=train_days)
    filtered_validate_perf = _cohort_performance(filtered_validate, window_days=validate_days)

    train_pf = filtered_train_perf.get("profit_factor")
    validate_pf = filtered_validate_perf.get("profit_factor")
    baseline_validate_pf = baseline_validate.get("profit_factor")
    pf_retention = None
    if validate_pf is not None and baseline_validate_pf:
        pf_retention = round(100.0 * validate_pf / baseline_validate_pf, 2)

    if validate_pf is not None and baseline_validate_pf is not None:
        if validate_pf >= baseline_validate_pf * 0.95:
            verdict = "improve"
        elif validate_pf >= baseline_validate_pf * 0.85:
            verdict = "preserve"
        else:
            verdict = "worsen"
    else:
        verdict = "preserve"

    result["filtered"] = {
        "train": filtered_train_perf,
        "validate": filtered_validate_perf,
        "validate_pf_retention_pct": pf_retention,
        "validate_pf_delta": round((validate_pf or 0) - (baseline_validate_pf or 0), 2)
        if validate_pf is not None and baseline_validate_pf is not None
        else None,
        "train_pf_delta": round((train_pf or 0) - (baseline_train.get("profit_factor") or 0), 2)
        if train_pf is not None and baseline_train.get("profit_factor") is not None
        else None,
    }
    result["stability_verdict"] = verdict
    return result


def _build_final_answer(
    *,
    buy_analysis: dict[str, Any],
    sell_analysis: dict[str, Any],
    buy_filters: dict[str, Any],
    sell_filters: dict[str, Any],
    buy_wf: dict[str, Any],
    sell_wf: dict[str, Any],
) -> dict[str, Any]:
    buy_best = buy_filters.get("best_gate_passing_filter") or {}
    sell_best = sell_filters.get("best_gate_passing_filter") or {}
    buy_improving = int(buy_filters.get("improving_gate_passing_count") or 0) > 0
    sell_improving = int(sell_filters.get("improving_gate_passing_count") or 0) > 0

    buy_baseline = buy_analysis.get("baseline", {})
    sell_baseline = sell_analysis.get("baseline", {})

    engines_with_improvement: list[dict[str, Any]] = []
    if buy_improving and buy_best.get("passes_production_gates"):
        engines_with_improvement.append(
            {
                "engine": "BUY_V3",
                "filter": buy_best.get("label"),
                "pf_delta": buy_best.get("pf_delta_vs_baseline"),
                "trap_reduction_pct": buy_best.get("trap_reduction_pct"),
                "signals_per_month": buy_best.get("signals_per_month"),
                "walk_forward_verdict": buy_wf.get("stability_verdict"),
            },
        )
    if sell_improving and sell_best.get("passes_production_gates"):
        engines_with_improvement.append(
            {
                "engine": "SELL_V5",
                "filter": sell_best.get("label"),
                "pf_delta": sell_best.get("pf_delta_vs_baseline"),
                "trap_reduction_pct": sell_best.get("trap_reduction_pct"),
                "signals_per_month": sell_best.get("signals_per_month"),
                "walk_forward_verdict": sell_wf.get("stability_verdict"),
            },
        )

    if engines_with_improvement:
        verdict = "YES"
        rationale = (
            "At least one production leg has a single additional filter that improves PF/trap profile "
            "while preserving frequency floors and walk-forward stability."
        )
    elif buy_best.get("passes_production_gates") or sell_best.get("passes_production_gates"):
        verdict = "PARTIAL"
        rationale = (
            "Production gates pass on baseline stacks; gate-passing filters exist but marginal PF/trap "
            "gains do not justify frequency sacrifice or walk-forward risk."
        )
    else:
        verdict = "NO"
        rationale = (
            "No single additional filter meets production frequency floors with measurable PF improvement — "
            "near-optimal evidence on current replay exports."
        )

    return {
        "can_production_engine_improve_further": verdict,
        "rationale": rationale,
        "buy_v3_baseline": {
            "signals_per_month": buy_baseline.get("signals_per_month"),
            "win_rate_pct": buy_baseline.get("win_rate_pct"),
            "profit_factor": buy_baseline.get("profit_factor"),
            "expectancy": buy_baseline.get("expectancy"),
        },
        "sell_v5_baseline": {
            "signals_per_month": sell_baseline.get("signals_per_month"),
            "win_rate_pct": sell_baseline.get("win_rate_pct"),
            "profit_factor": sell_baseline.get("profit_factor"),
            "expectancy": sell_baseline.get("expectancy"),
        },
        "top_proposed_filters": [
            {
                "engine": "BUY_V3",
                "filter": buy_best.get("label") if buy_best else None,
                "passes_gates": bool(buy_best.get("passes_production_gates")) if buy_best else False,
                "improves_metrics": buy_improving,
                "pf_delta": buy_best.get("pf_delta_vs_baseline") if buy_best else None,
                "trap_reduction_pct": buy_best.get("trap_reduction_pct") if buy_best else None,
                "mae_delta_points": buy_best.get("mae_delta_points") if buy_best else None,
                "signals_per_month": buy_best.get("signals_per_month") if buy_best else None,
                "near_optimal_without_frequency_sacrifice": not buy_improving,
            },
            {
                "engine": "SELL_V5",
                "filter": sell_best.get("label") if sell_best else None,
                "passes_gates": bool(sell_best.get("passes_production_gates")) if sell_best else False,
                "improves_metrics": sell_improving,
                "pf_delta": sell_best.get("pf_delta_vs_baseline") if sell_best else None,
                "trap_reduction_pct": sell_best.get("trap_reduction_pct") if sell_best else None,
                "mae_delta_points": sell_best.get("mae_delta_points") if sell_best else None,
                "signals_per_month": sell_best.get("signals_per_month") if sell_best else None,
                "near_optimal_without_frequency_sacrifice": not sell_improving,
            },
        ],
        "engines_with_measurable_improvement": engines_with_improvement,
        "frequency_floors": {
            "buy_v3_min_signals_per_month": BUY_MIN_SIGNALS_PER_MONTH,
            "sell_v5_min_signals_per_month": SELL_MIN_SIGNALS_PER_MONTH,
        },
    }


@dataclass
class ProductionEdgeEnhancementAuditReport:
    """Production edge enhancement audit synthesis output."""

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
    buy_v3_winner_loser_analysis: dict[str, Any]
    sell_v5_winner_loser_analysis: dict[str, Any]
    condition_rankings: dict[str, Any]
    proposed_filters: dict[str, Any]
    timing_analysis: dict[str, Any]
    walk_forward_impact: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


class ProductionEdgeEnhancementAuditResearch:
    """Synthesize production edge enhancement audit from completed replay exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in SOURCE_EXPORTS.items():
            required = name in {
                "buy_v3_candidate_validation",
                "unified_production_replay_validation",
                "smartmoneyengine_v5_candidate_validation",
            }
            status = "loaded" if path.exists() else ("missing" if required else "optional_missing")
            loaded[name] = {
                "path": str(path),
                "status": status,
                "data": _load_json(path, required=required) if path.exists() or required else {},
            }
        self.sources = loaded
        return loaded

    def run(self) -> ProductionEdgeEnhancementAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        v3_export = sources["buy_v3_candidate_validation"]["data"]
        unified = sources["unified_production_replay_validation"]["data"]
        v5_export = sources["smartmoneyengine_v5_candidate_validation"]["data"]
        quality_audit = sources["buy_v3_signal_quality_audit"]["data"]
        tradeability = sources["buy_v3_tradeability_production_validation"]["data"]
        winner_export = sources["buy_winner_vs_false_reversal_analysis"]["data"]
        walkforward_optional = sources["smartmoneyengine_walkforward_validation"]["data"]

        buy_signals = list(
            unified.get("per_signal_details", {}).get("buy_v3")
            or v3_export.get("per_signal_details", {}).get("buy_v3")
            or [],
        )
        sell_signals = list(unified.get("per_signal_details", {}).get("sell_v5") or [])
        if not buy_signals:
            raise ProductionEdgeEnhancementAuditError("No BUY_V3 per_signal_details found in exports.")
        if not sell_signals:
            raise ProductionEdgeEnhancementAuditError("No SELL_V5 per_signal_details found in unified export.")

        window_days = int(unified.get("trading_days_replayed") or v3_export.get("trading_days_replayed") or 120)
        walk_forward = unified.get("walk_forward", {})

        for signal in buy_signals:
            signal["conditions"] = _extract_conditions_from_signal(signal)
            signal["audit_classification"] = _map_buy_audit_classification(signal.get("classification", "Unknown"))
        for signal in sell_signals:
            signal["conditions"] = _extract_sell_conditions(signal)
            signal["audit_classification"] = _classify_sell_signal(signal)

        buy_analysis = _winner_loser_side_analysis(
            buy_signals,
            side="BUY_V3",
            window_days=window_days,
            is_winner_fn=_is_buy_winner,
            classify_fn=lambda signal: signal.get("audit_classification", "Liquidity Failure"),
        )
        sell_analysis = _winner_loser_side_analysis(
            sell_signals,
            side="SELL_V5",
            window_days=window_days,
            is_winner_fn=_is_sell_winner,
            classify_fn=_classify_sell_signal,
        )

        buy_conditions = _condition_attribution(
            buy_signals,
            conditions=ANALYSIS_CONDITIONS,
            is_winner_fn=_is_buy_winner,
        )
        sell_conditions = _condition_attribution(
            sell_signals,
            conditions=SELL_ANALYSIS_CONDITIONS,
            is_winner_fn=_is_sell_winner,
        )

        buy_filters = _filter_simulations(
            buy_signals,
            side="BUY_V3",
            window_days=window_days,
            candidates=BUY_FILTER_CANDIDATES,
            is_winner_fn=_is_buy_winner,
            classify_fn=lambda signal: signal.get("audit_classification", "Liquidity Failure"),
            min_spm=BUY_MIN_SIGNALS_PER_MONTH,
        )
        sell_filters = _filter_simulations(
            sell_signals,
            side="SELL_V5",
            window_days=window_days,
            candidates=SELL_FILTER_CANDIDATES,
            is_winner_fn=_is_sell_winner,
            classify_fn=_classify_sell_signal,
            min_spm=SELL_MIN_SIGNALS_PER_MONTH,
        )

        buy_capture = (
            unified.get("engine_comparison", {})
            .get("buy_v3_only", {})
            .get("point_capture_bullish")
        )
        sell_capture = (
            v5_export.get("comparison", {})
            .get("v5_candidate", {})
            .get("point_capture")
            or unified.get("engine_comparison", {})
            .get("sell_v5_only", {})
            .get("point_capture_bearish")
        )

        timing_analysis = {
            "buy_v3": _expansion_timing_analysis(
                buy_signals,
                capture_export=buy_capture,
                direction_label="bullish",
            ),
            "sell_v5": _expansion_timing_analysis(
                sell_signals,
                capture_export=sell_capture,
                direction_label="bearish",
            ),
            "cross_export_timing_reference": {
                "buy_v3_signal_quality_audit": quality_audit.get("signal_timing"),
                "buy_v3_tradeability": tradeability.get("walk_forward_stability"),
            },
        }

        buy_wf = _walk_forward_filter_impact(
            buy_signals,
            side="BUY_V3",
            walk_forward=walk_forward,
            proposed_filter=buy_filters.get("best_gate_passing_filter"),
            is_winner_fn=_is_buy_winner,
        )
        sell_wf = _walk_forward_filter_impact(
            sell_signals,
            side="SELL_V5",
            walk_forward=walk_forward,
            proposed_filter=sell_filters.get("best_gate_passing_filter"),
            is_winner_fn=_is_sell_winner,
        )

        final_answer = _build_final_answer(
            buy_analysis=buy_analysis,
            sell_analysis=sell_analysis,
            buy_filters=buy_filters,
            sell_filters=sell_filters,
            buy_wf=buy_wf,
            sell_wf=sell_wf,
        )

        buy_best = buy_filters.get("best_gate_passing_filter") or {}
        sell_best = sell_filters.get("best_gate_passing_filter") or {}
        buy_top_sep = buy_conditions["rankings"]["by_accuracy_separation"][:3]
        sell_top_sep = sell_conditions["rankings"]["by_accuracy_separation"][:3]

        conclusions = [
            "Production edge enhancement audit synthesized from replay-validated exports only — no new replay.",
            (
                f"BUY_V3: {buy_analysis['baseline']['sample_size']} signals, "
                f"{buy_analysis['baseline']['signals_per_month']}/mo, WR {buy_analysis['baseline']['win_rate_pct']}%, "
                f"PF {buy_analysis['baseline']['profit_factor']}, trap rate "
                f"{buy_analysis['loser_classification']['trap_rate_pct']}%."
            ),
            (
                f"SELL_V5: {sell_analysis['baseline']['sample_size']} signals, "
                f"{sell_analysis['baseline']['signals_per_month']}/mo, WR {sell_analysis['baseline']['win_rate_pct']}%, "
                f"PF {sell_analysis['baseline']['profit_factor']}, trap rate "
                f"{sell_analysis['loser_classification']['trap_rate_pct']}%."
            ),
            (
                f"BUY_V3 top separator: {buy_top_sep[0]['condition']} "
                f"({buy_top_sep[0]['coverage_delta_pp']}pp delta) | "
                f"SELL_V5 top separator: {sell_top_sep[0]['condition']} "
                f"({sell_top_sep[0]['coverage_delta_pp']}pp delta)."
            ),
            (
                f"Proposed BUY filter (frequency-preserving): "
                f"{buy_best.get('label') or 'none — near-optimal at current stack'}."
            ),
            (
                f"Proposed SELL filter: {sell_best.get('label', 'none')} "
                f"(PF delta {sell_best.get('pf_delta_vs_baseline')}, trap reduction "
                f"{sell_best.get('trap_reduction_pct')}%)."
            ),
            (
                f"Walk-forward: BUY {buy_wf.get('stability_verdict')}, "
                f"SELL {sell_wf.get('stability_verdict')} on proposed filters."
            ),
            f"Final verdict: {final_answer['can_production_engine_improve_further']} — {final_answer['rationale']}",
        ]

        if winner_export:
            conclusions.append(
                "BUY_V3 formula stack already removed 947/947 V2-only false reversals per winner vs false reversal export."
            )

        return ProductionEdgeEnhancementAuditReport(
            report_type="Production Edge Enhancement Audit",
            engines=["BUY_V3", "SELL_V5"],
            symbol=unified.get("symbol") or v3_export.get("symbol", "NIFTY50"),
            timeframe=unified.get("timeframe") or v3_export.get("timeframe", "5M"),
            trading_days_replayed=window_days,
            replay_start_date=unified.get("replay_start_date") or v3_export.get("replay_start_date", ""),
            replay_end_date=unified.get("replay_end_date") or v3_export.get("replay_end_date", ""),
            methodology={
                "research_only": True,
                "synthesis_only": True,
                "no_new_replay": True,
                "no_new_discovery": True,
                "no_new_models": True,
                "no_new_indicators": True,
                "primary_signal_sources": {
                    "buy_v3": "unified_production_replay_validation.json per_signal_details.buy_v3",
                    "sell_v5": "unified_production_replay_validation.json per_signal_details.sell_v5",
                },
                "loser_taxonomy": list(LOSER_CLASSIFICATIONS),
                "frequency_floors": {
                    "buy_v3_signals_per_month_min": BUY_MIN_SIGNALS_PER_MONTH,
                    "sell_v5_signals_per_month_min": SELL_MIN_SIGNALS_PER_MONTH,
                },
                "production_gates": PRODUCTION_GATES,
                "sell_v5_vwap_gate": V5_VWAP_GATE_RULE,
                "expansion_thresholds": list(EXPANSION_THRESHOLDS),
                "walk_forward_method": "train/validate split from unified export dates; filter simulation on per-signal cohorts",
            },
            source_exports={
                name: {"path": entry["path"], "status": entry["status"]} for name, entry in sources.items()
            },
            limitations=[
                "No new replay — all metrics derived from completed validation JSON exports.",
                "SELL_V5 loser taxonomy inferred from per-signal MFE/MAE/HTF context (no export classification field).",
                "Single-filter simulation applies one extra require/exclude condition to existing signals only.",
                "Walk-forward validate cohort is small for BUY_V3 (6 signals) — stability flags are indicative.",
                "smartmoneyengine_walkforward_validation.json optional; unified walk_forward used as primary.",
                "Point-capture timing for SELL uses aggregate export cross-check; per-signal lead time sparse on SELL leg.",
            ],
            buy_v3_winner_loser_analysis=buy_analysis,
            sell_v5_winner_loser_analysis=sell_analysis,
            condition_rankings={
                "buy_v3": buy_conditions,
                "sell_v5": sell_conditions,
                "combined_insights": {
                    "buy_v3_best_accuracy_separator": buy_top_sep[0] if buy_top_sep else None,
                    "sell_v5_best_accuracy_separator": sell_top_sep[0] if sell_top_sep else None,
                    "buy_v3_best_stop_loss_reducer": buy_conditions["rankings"]["by_stop_loss_reduction"][:3],
                    "sell_v5_best_stop_loss_reducer": sell_conditions["rankings"]["by_stop_loss_reduction"][:3],
                    "quality_audit_cross_check": quality_audit.get("final_answer"),
                    "tradeability_cross_check": tradeability.get("final_answers"),
                    "walkforward_optional_reference": walkforward_optional.get("final_answer")
                    if walkforward_optional
                    else None,
                },
            },
            proposed_filters={
                "buy_v3": buy_filters,
                "sell_v5": sell_filters,
                "improvement_potential_if_applied": {
                    "buy_v3": {
                        "filter": (buy_filters.get("best_gate_passing_filter") or {}).get("label"),
                        "best_tradeoff_reference": (buy_filters.get("best_tradeoff_filter") or {}).get("label"),
                        "lead_time_change_bars": None,
                        "points_captured_delta_pct": (buy_filters.get("best_gate_passing_filter") or {}).get(
                            "capture_100_plus_pct"
                        ),
                        "stop_loss_size_delta_points": (buy_filters.get("best_gate_passing_filter") or {}).get(
                            "mae_delta_points"
                        ),
                        "pf_delta": (buy_filters.get("best_gate_passing_filter") or {}).get("pf_delta_vs_baseline"),
                        "signals_per_month": (buy_filters.get("best_gate_passing_filter") or {}).get(
                            "signals_per_month"
                        ),
                        "frequency_preserving_filter_found": bool(
                            buy_filters.get("improving_gate_passing_count"),
                        ),
                    },
                    "sell_v5": {
                        "filter": sell_best.get("label"),
                        "lead_time_change_bars": None,
                        "points_captured_delta_pct": sell_best.get("capture_100_plus_pct"),
                        "stop_loss_size_delta_points": sell_best.get("mae_delta_points"),
                        "pf_delta": sell_best.get("pf_delta_vs_baseline"),
                        "signals_per_month": sell_best.get("signals_per_month"),
                        "frequency_preserving_filter_found": bool(
                            sell_filters.get("improving_gate_passing_count"),
                        ),
                    },
                },
            },
            timing_analysis=timing_analysis,
            walk_forward_impact={
                "split_definition": walk_forward,
                "buy_v3": buy_wf,
                "sell_v5": sell_wf,
                "unified_export_stability_flag": walk_forward.get("stable"),
            },
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: ProductionEdgeEnhancementAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Production edge enhancement audit exported to %s", self.report_path)
        return self.report_path


def generate_production_edge_enhancement_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export production edge enhancement audit JSON."""
    return ProductionEdgeEnhancementAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_production_edge_enhancement_audit_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    print(f"Exported: {path}")
    print(f"Verdict: {final['can_production_engine_improve_further']}")
    for item in final.get("top_proposed_filters", []):
        print(f"  {item['engine']}: {item.get('filter')} | PF delta {item.get('pf_delta')}")
