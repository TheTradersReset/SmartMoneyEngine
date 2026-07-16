"""
Regime Aware Execution Validation — synthesis from existing replay exports only.

Determines whether execution rules should change by regime and validates the
deployment playbook against signal-level evidence. No replay, indicators, models,
or discovery.
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
from src.research.live_trade_management_execution_efficiency_audit_research import (
    _capture_leakage_analysis,
    _classify_miss_reason,
    _resolve_stop_extended,
)
from src.research.production_edge_enhancement_audit_research import (
    _is_buy_winner,
    _is_sell_winner,
    _timing_label,
)
from src.research.production_trading_playbook_audit_research import (
    _metrics_from_pnls,
    _tiered_structure_pnl,
)
from src.research.regime_detection_audit_research import (
    SELL_V6_MODEL_ID,
    THROTTLE_WEIGHT,
    classify_signal_regime,
)
from src.research.sell_v6_replay_validation_research import V6_VWAP_GATE_RULE
logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "regime_aware_execution_validation.json"

SOURCE_EXPORTS = {
    "live_trade_management_execution_efficiency_audit": RESEARCH_DIR
    / "live_trade_management_execution_efficiency_audit.json",
    "regime_detection_audit": RESEARCH_DIR / "regime_detection_audit.json",
    "final_production_deployment_audit": RESEARCH_DIR / "final_production_deployment_audit.json",
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
}

EXECUTION_STOP_VARIANTS = ("fixed_10", "fixed_20", "structure_based", "liquidity_based")
FIXED_EXIT_TARGETS = (40, 60, 80, 100)
EXIT_STRUCTURES: dict[str, dict[str, Any]] = {
    "40/60/100": {"t1": 40, "t2": 60, "t3": 100, "runner": False},
    "40/80/Runner": {"t1": 40, "t2": 80, "t3": None, "runner": True},
    "50/100/Runner": {"t1": 50, "t2": 100, "t3": None, "runner": True},
    "60/100/Runner": {"t1": 60, "t2": 100, "t3": None, "runner": True},
}

WINNER_PRESERVATION_PCTS = (0.7, 0.8, 0.9)
ENTRY_WAIT_BUCKETS = (
    ("immediate", lambda bars: bars is not None and int(bars) >= 0),
    ("wait_1_candle", lambda bars: bars is not None and int(bars) >= 1),
    ("wait_2_candles", lambda bars: bars is not None and int(bars) >= 2),
)

LOSS_CAUSES = (
    "signal_quality",
    "execution",
    "sizing",
    "regime",
    "target",
    "stop",
)


class RegimeAwareExecutionValidationError(Exception):
    """Raised when regime-aware execution validation synthesis fails."""


@dataclass
class RegimeAwareExecutionValidationReport:
    """Regime-aware execution validation audit output."""

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
    regime_classification: dict[str, Any]
    per_regime_performance: dict[str, Any]
    per_regime_stop_exit_comparison: dict[str, Any]
    execution_failure_audit: dict[str, Any]
    entry_precision: dict[str, Any]
    capture_leakage: dict[str, Any]
    regime_aware_playbook: dict[str, Any]
    paper_vs_real_configs: dict[str, Any]
    loss_root_cause: dict[str, Any]
    production_scores: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise RegimeAwareExecutionValidationError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = int(pct * max(len(sorted_vals) - 1, 0))
    return round(sorted_vals[idx], 2)


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


def _group_signals_by_regime(
    signals: list[dict[str, Any]],
    *,
    direction: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        regime = classify_signal_regime(signal, direction=direction)
        grouped[regime["composite"]].append(signal)
    return dict(grouped)


def _regime_dimension_counts(signals: list[dict[str, Any]], *, direction: str) -> dict[str, Any]:
    trend: Counter[str] = Counter()
    vol: Counter[str] = Counter()
    gap: Counter[str] = Counter()
    liq: Counter[str] = Counter()
    composite: Counter[str] = Counter()
    export_tags = 0

    for signal in signals:
        regime = classify_signal_regime(signal, direction=direction)
        trend[regime["trend"]] += 1
        vol[regime["volatility"]] += 1
        gap[regime["gap"]] += 1
        liq[regime["liquidity"]] += 1
        composite[regime["composite"]] += 1
        if regime.get("export_regime_present"):
            export_tags += 1

    return {
        "direction": direction,
        "total_signals": len(signals),
        "export_regime_tagged_count": export_tags,
        "by_trend": dict(trend),
        "by_volatility": dict(vol),
        "by_gap": dict(gap),
        "by_liquidity": dict(liq),
        "by_composite": dict(composite),
    }


def _regime_performance_row(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
    win_fn: Any,
    structure: dict[str, Any],
    stop_variant: str,
) -> dict[str, Any]:
    if not signals:
        return {
            "signal_count": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "avg_mfe": 0.0,
            "avg_mae": 0.0,
            "capture_efficiency_pct": 0.0,
            "monthly_points": 0.0,
        }

    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals)
    pnls: list[float] = []
    for signal in signals:
        stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
        pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
        pnls.append(pnl)

    metrics = _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=window_days)
    wins = sum(1 for signal in signals if win_fn(signal))
    return {
        "signal_count": len(signals),
        "win_rate_pct": round(100.0 * wins / len(signals), 2),
        "profit_factor": metrics["profit_factor"],
        "expectancy": metrics["expectancy"],
        "avg_mfe": metrics["avg_mfe"],
        "avg_mae": metrics["avg_mae"],
        "capture_efficiency_pct": metrics["capture_efficiency_pct"],
        "monthly_points": metrics["monthly_points"],
        "simulated_win_rate_pct": metrics["win_rate_pct"],
    }


def _per_regime_performance_table(
    signals: list[dict[str, Any]],
    *,
    direction: str,
    window_days: int,
    win_fn: Any,
    structure: dict[str, Any],
    stop_variant: str,
) -> dict[str, Any]:
    grouped = _group_signals_by_regime(signals, direction=direction)
    table: dict[str, Any] = {}
    for regime, cohort in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        table[regime] = _regime_performance_row(
            cohort,
            window_days=window_days,
            win_fn=win_fn,
            structure=structure,
            stop_variant=stop_variant,
        )
    return table


def _tier_hit_rates_for_structure(
    signals: list[dict[str, Any]],
    structure: dict[str, Any],
) -> dict[str, Any]:
    total = len(signals)
    t1 = float(structure["t1"])
    t2 = float(structure["t2"])
    t1_hits = sum(1 for s in signals if float(s.get("mfe_points") or 0.0) >= t1)
    t2_hits = sum(1 for s in signals if float(s.get("mfe_points") or 0.0) >= t2)
    runner_hits = 0
    if structure.get("runner"):
        runner_hits = sum(1 for s in signals if float(s.get("mfe_points") or 0.0) > t2)
    elif structure.get("t3"):
        t3 = float(structure["t3"])
        runner_hits = sum(1 for s in signals if float(s.get("mfe_points") or 0.0) >= t3)
    return {
        "T1_hit_pct": round(100.0 * t1_hits / max(total, 1), 2),
        "T2_hit_pct": round(100.0 * t2_hits / max(total, 1), 2),
        "runner_or_T3_hit_pct": round(100.0 * runner_hits / max(total, 1), 2),
    }


def _winner_preservation_thresholds(
    signals: list[dict[str, Any]],
    *,
    win_fn: Any,
) -> dict[str, float | None]:
    winners = [s for s in signals if win_fn(s)]
    maes = sorted(float(s.get("mae_points") or 0.0) for s in winners)
    result: dict[str, float | None] = {}
    for pct in WINNER_PRESERVATION_PCTS:
        label = f"{int(pct * 100)}pct"
        if not maes:
            result[label] = None
        else:
            idx = int(pct * max(len(maes) - 1, 0))
            result[label] = round(maes[idx], 2)
    return result


def _execution_failure_for_stop(
    signals: list[dict[str, Any]],
    *,
    structure: dict[str, Any],
    stop_variant: str,
    win_fn: Any,
    window_days: int,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    pnls: list[float] = []
    stop_hits = 0
    premature = 0
    rr_values: list[float] = []
    avg_stop = 0.0

    t1 = float(structure["t1"])
    for signal in signals:
        stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
        avg_stop += stop_pts
        mfe = float(signal.get("mfe_points") or 0.0)
        mae = float(signal.get("mae_points") or 0.0)
        pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
        pnls.append(pnl)

        if mae >= stop_pts and pnl <= 0:
            stop_hits += 1
        if win_fn(signal) and mfe >= t1 and pnl <= 0 and mae >= stop_pts:
            premature += 1
        if pnl > 0 and stop_pts > 0:
            rr_values.append(round(pnl / stop_pts, 2))

    metrics = _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=window_days)
    tier_hits = _tier_hit_rates_for_structure(signals, structure)

    return {
        "stop_variant": stop_variant,
        "average_stop_points": round(avg_stop / max(len(signals), 1), 2),
        "stop_hit_pct": round(100.0 * stop_hits / max(len(signals), 1), 2),
        **tier_hits,
        "avg_rr": round(mean(rr_values), 2) if rr_values else 0.0,
        "net_expectancy": metrics["expectancy"],
        "profit_factor": metrics["profit_factor"],
        "premature_stop_outs": premature,
        "premature_stop_out_pct": round(100.0 * premature / max(len(signals), 1), 2),
        "min_stop_for_winner_preservation": _winner_preservation_thresholds(signals, win_fn=win_fn),
    }


def _execution_failure_audit(
    signals: list[dict[str, Any]],
    *,
    structure: dict[str, Any],
    win_fn: Any,
    window_days: int,
) -> dict[str, Any]:
    by_stop: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    for variant in EXECUTION_STOP_VARIANTS:
        row = _execution_failure_for_stop(
            signals,
            structure=structure,
            stop_variant=variant,
            win_fn=win_fn,
            window_days=window_days,
        )
        by_stop[variant] = row
        ranking.append(
            {
                **row,
                "optimization_score": round(row["net_expectancy"] * (row["profit_factor"] or 0.0), 2),
            },
        )
    best = max(ranking, key=lambda item: (item["net_expectancy"], item["profit_factor"] or 0.0))
    return {
        "structure_used": structure,
        "by_stop_variant": by_stop,
        "ranking": sorted(ranking, key=lambda item: item["optimization_score"], reverse=True),
        "best_stop_variant": best["stop_variant"],
    }


def _fixed_exit_row(
    signals: list[dict[str, Any]],
    *,
    target: int,
    stop_variant: str,
    window_days: int,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    pnls: list[float] = []
    for signal in signals:
        stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
        _, pnl = _fixed_target_pnl(signal, target)
        mae = float(signal.get("mae_points") or 0.0)
        if mae >= stop_pts and float(signal.get("mfe_points") or 0.0) < target:
            pnl = -min(mae, stop_pts)
        pnls.append(pnl)
    return _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=window_days)


def _exit_structure_row(
    signals: list[dict[str, Any]],
    *,
    label: str,
    structure: dict[str, Any],
    stop_variant: str,
    window_days: int,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    pnls: list[float] = []
    for signal in signals:
        stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
        pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
        pnls.append(pnl)
    metrics = _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=window_days)
    return {"structure": label, "tiers": structure, **metrics}


def _best_stop_exit_for_cohort(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
) -> dict[str, Any]:
    if not signals:
        return {"sample_size": 0}

    best_stop = None
    best_stop_score = -1.0
    stop_evidence: dict[str, Any] = {}

    default_structure = EXIT_STRUCTURES["60/100/Runner"]
    for variant in EXECUTION_STOP_VARIANTS:
        row = _execution_failure_for_stop(
            signals,
            structure=default_structure,
            stop_variant=variant,
            win_fn=lambda s: bool(s.get("win")),
            window_days=window_days,
        )
        score = row["net_expectancy"] * (row["profit_factor"] or 0.0)
        if score > best_stop_score:
            best_stop_score = score
            best_stop = variant
            stop_evidence = row

    assert best_stop is not None
    exit_ranking: list[dict[str, Any]] = []
    for label, structure in EXIT_STRUCTURES.items():
        metrics = _exit_structure_row(
            signals,
            label=label,
            structure=structure,
            stop_variant=best_stop,
            window_days=window_days,
        )
        exit_ranking.append(
            {
                **metrics,
                "optimization_score": round(metrics["expectancy"] * (metrics["profit_factor"] or 0.0), 2),
            },
        )

    fixed_ranking: list[dict[str, Any]] = []
    for target in FIXED_EXIT_TARGETS:
        metrics = _fixed_exit_row(signals, target=target, stop_variant=best_stop, window_days=window_days)
        fixed_ranking.append(
            {
                "target_points": target,
                **metrics,
                "optimization_score": round(metrics["expectancy"] * (metrics["profit_factor"] or 0.0), 2),
            },
        )

    best_tiered = max(exit_ranking, key=lambda item: (item["expectancy"], item["profit_factor"] or 0.0))
    best_fixed = max(fixed_ranking, key=lambda item: (item["expectancy"], item["profit_factor"] or 0.0))

    return {
        "sample_size": len(signals),
        "best_stop_variant": best_stop,
        "best_stop_evidence": stop_evidence,
        "best_tiered_exit": best_tiered["structure"],
        "best_tiered_evidence": best_tiered,
        "best_fixed_exit_target": best_fixed["target_points"],
        "best_fixed_evidence": best_fixed,
        "tiered_ranking": sorted(exit_ranking, key=lambda item: item["optimization_score"], reverse=True),
        "fixed_ranking": sorted(fixed_ranking, key=lambda item: item["optimization_score"], reverse=True),
    }


def _per_regime_stop_exit_comparison(
    signals: list[dict[str, Any]],
    *,
    direction: str,
    window_days: int,
) -> dict[str, Any]:
    grouped = _group_signals_by_regime(signals, direction=direction)
    by_regime: dict[str, Any] = {}
    for regime, cohort in grouped.items():
        by_regime[regime] = _best_stop_exit_for_cohort(cohort, window_days=window_days)
    return {
        "direction": direction,
        "regime_count": len(by_regime),
        "by_regime": by_regime,
    }


def _entry_precision_analysis(
    signals: list[dict[str, Any]],
    *,
    direction: str,
) -> dict[str, Any]:
    per_signal: list[dict[str, Any]] = []
    timing_counts: Counter[str] = Counter()
    timing_mfe: dict[str, list[float]] = defaultdict(list)
    wait_capture: dict[str, list[float]] = defaultdict(list)

    for signal in signals:
        bars = signal.get("bars_before_expansion")
        pts_before = float(signal.get("points_before_expansion") or 0.0)
        mfe = float(signal.get("mfe_points") or 0.0)
        entry = float(signal.get("entry") or 0.0)
        timing = _timing_label(bars if bars is not None else None)
        timing_counts[timing] += 1
        timing_mfe[timing].append(mfe)

        optimal_entry_proxy = entry - pts_before if direction == "BUY" else entry + pts_before
        slippage_proxy = max(pts_before, 0.0)
        max_available = mfe + slippage_proxy
        capture_pct = round(100.0 * mfe / max(max_available, 1.0), 2)

        per_signal.append(
            {
                "timestamp": signal.get("timestamp"),
                "signal_entry_price": entry,
                "optimal_entry_price_proxy": round(optimal_entry_proxy, 2),
                "slippage_points_proxy": round(slippage_proxy, 2),
                "points_lost_before_entry": round(slippage_proxy, 2),
                "timing_class": timing,
                "bars_before_expansion": bars,
                "mfe_from_signal_entry": round(mfe, 2),
                "capture_pct_of_max_move": capture_pct,
            },
        )

        for label, predicate in ENTRY_WAIT_BUCKETS:
            if predicate(bars):
                wait_capture[label].append(capture_pct)

    timing_summary = {
        label: {
            "count": timing_counts[label],
            "pct": round(100.0 * timing_counts[label] / max(len(signals), 1), 2),
            "avg_mfe": round(mean(timing_mfe[label]), 2) if timing_mfe[label] else 0.0,
        }
        for label in ("Early", "Same Candle", "Delayed", "No Linked Move")
    }

    wait_summary = {
        label: {
            "eligible_signals": len(wait_capture[label]),
            "avg_capture_pct": round(mean(wait_capture[label]), 2) if wait_capture[label] else 0.0,
        }
        for label, _ in ENTRY_WAIT_BUCKETS
    }

    return {
        "direction": direction,
        "methodology": (
            "Optimal entry proxied by move origin (entry ± points_before_expansion). "
            "Slippage = points_before_expansion. Timing: Early (>0 bars before), "
            "Same Candle (=0), Delayed (<0)."
        ),
        "aggregate": {
            "avg_slippage_points": round(
                mean(float(s.get("points_before_expansion") or 0.0) for s in signals),
                2,
            )
            if signals
            else 0.0,
            "median_slippage_points": round(
                median(float(s.get("points_before_expansion") or 0.0) for s in signals),
                2,
            )
            if signals
            else 0.0,
            "avg_points_lost_before_entry": round(
                mean(max(float(s.get("points_before_expansion") or 0.0), 0.0) for s in signals),
                2,
            )
            if signals
            else 0.0,
        },
        "timing_class_summary": timing_summary,
        "entry_wait_capture": wait_summary,
        "per_signal_sample": per_signal[:20],
    }


def _capture_leakage_with_regime(
    signals: list[dict[str, Any]],
    *,
    structure: dict[str, Any],
    stop_variant: str,
    direction: str,
    throttle_map: dict[str, str],
) -> dict[str, Any]:
    base = _capture_leakage_analysis(signals, structure=structure, stop_variant=stop_variant)
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    reason_counts: Counter[str] = Counter()
    regime_blocked_mfe = 0.0

    for signal in signals:
        regime = classify_signal_regime(signal, direction=direction)
        throttle = throttle_map.get(regime["composite"], "FULL")
        mfe = float(signal.get("mfe_points") or 0.0)
        stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
        pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
        missed = max(0.0, mfe - max(pnl, 0.0))

        if throttle == "BLOCK" and direction == "SELL":
            regime_blocked_mfe += mfe
            if missed > 0.01:
                reason_counts["regime_filter"] += 1
            continue

        if missed > 0.01:
            reason = _classify_miss_reason(signal, structure, stop_pts=stop_pts, pnl=pnl)
            if reason == "timing":
                reason_counts["late_entry"] += 1
            elif reason == "stop":
                reason_counts["premature_stop"] += 1
            elif reason == "runner":
                reason_counts["runner"] += 1
            else:
                reason_counts["target"] += 1

    ranked = [
        {"source": source, "count": count, "rank": index + 1}
        for index, (source, count) in enumerate(reason_counts.most_common())
    ]

    return {
        **base,
        "leakage_source_ranking": ranked,
        "regime_filter_blocked_mfe_points": round(regime_blocked_mfe, 2),
        "direction": direction,
    }


def _simulate_execution(
    signals: list[dict[str, Any]],
    *,
    direction: str,
    stop_variant: str,
    structure: dict[str, Any],
    per_regime_overrides: dict[str, dict[str, Any]] | None,
    throttle_map: dict[str, str],
    window_days: int,
    sizing_mode: str,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    pnls: list[float] = []
    used: list[dict[str, Any]] = []

    for signal in signals:
        regime = classify_signal_regime(signal, direction=direction)
        composite = regime["composite"]
        if direction == "SELL" and throttle_map.get(composite) == "BLOCK":
            continue

        override = (per_regime_overrides or {}).get(composite, {})
        stop = override.get("stop_variant", stop_variant)
        struct_label = override.get("exit_structure")
        struct = EXIT_STRUCTURES[struct_label] if struct_label else structure

        stop_pts = _resolve_stop_extended(signal, stop, cohort_mae_median=mae_median)
        pnl, _ = _tiered_structure_pnl(signal, struct, stop_pts=stop_pts)

        weight = 1.0
        if sizing_mode == "regime_adaptive" and direction == "SELL":
            throttle = throttle_map.get(composite, "FULL")
            weight = THROTTLE_WEIGHT.get(throttle, 1.0)
        elif sizing_mode == "half":
            weight = 0.5

        pnls.append(round(pnl * weight, 2))
        used.append(signal)

    metrics = _extended_metrics(pnls, signals=used, sample_size=len(used), window_days=window_days)
    metrics["pnls"] = pnls
    return metrics


def _build_regime_playbook_rules(
    per_regime_comparison: dict[str, Any],
    throttle_rules: list[dict[str, Any]],
    *,
    direction: str,
) -> list[dict[str, Any]]:
    throttle_map = {row["regime"]: row["throttle"] for row in throttle_rules}
    rules: list[dict[str, Any]] = []

    for regime, evidence in per_regime_comparison.get("by_regime", {}).items():
        if evidence.get("sample_size", 0) == 0:
            continue
        throttle = throttle_map.get(regime, "FULL") if direction == "SELL" else "FULL"
        sizing = "BLOCK" if throttle == "BLOCK" else ("HALF" if throttle in {"HALF", "QUARTER"} else "FULL")
        rules.append(
            {
                "regime": regime,
                "direction": direction,
                "throttle": throttle,
                "sizing": sizing,
                "stop_variant": evidence.get("best_stop_variant"),
                "tiered_exit": evidence.get("best_tiered_exit"),
                "fixed_exit_fallback": evidence.get("best_fixed_exit_target"),
                "expectancy": evidence.get("best_tiered_evidence", {}).get("expectancy"),
                "profit_factor": evidence.get("best_tiered_evidence", {}).get("profit_factor"),
            },
        )

    rules.sort(key=lambda row: row.get("expectancy") or 0.0, reverse=True)
    return rules


def _loss_root_cause_analysis(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    *,
    buy_structure: dict[str, Any],
    sell_structure: dict[str, Any],
    buy_stop: str,
    sell_stop: str,
    throttle_map: dict[str, str],
) -> dict[str, Any]:
    cause_counts: Counter[str] = Counter()

    def _attribute(signal: dict[str, Any], *, direction: str, structure: dict[str, Any], stop: str) -> None:
        win_fn = _is_buy_winner if direction == "BUY" else _is_sell_winner
        if win_fn(signal):
            return

        regime = classify_signal_regime(signal, direction=direction)
        if direction == "SELL" and throttle_map.get(regime["composite"]) == "BLOCK":
            cause_counts["regime"] += 1
            return

        mfe = float(signal.get("mfe_points") or 0.0)
        mae = float(signal.get("mae_points") or 0.0)
        pts_before = float(signal.get("points_before_expansion") or 0.0)
        bars = signal.get("bars_before_expansion")
        mae_median = float(signal.get("mae_points") or 0.0)
        stop_pts = _resolve_stop_extended(signal, stop, cohort_mae_median=mae_median)
        pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)

        if mfe < float(structure["t1"]) * 0.5:
            cause_counts["signal_quality"] += 1
        elif pts_before > 15.0 or (bars is not None and int(bars) < 0):
            cause_counts["execution"] += 1
        elif mae >= stop_pts and pnl <= 0:
            cause_counts["stop"] += 1
        elif mfe >= float(structure["t2"]) and pnl < mfe * 0.7:
            cause_counts["target"] += 1
        elif direction == "SELL" and throttle_map.get(regime["composite"]) in {"HALF", "QUARTER"}:
            cause_counts["sizing"] += 1
        else:
            cause_counts["regime"] += 1

    for signal in buy_signals:
        _attribute(signal, direction="BUY", structure=buy_structure, stop=buy_stop)
    for signal in sell_signals:
        _attribute(signal, direction="SELL", structure=sell_structure, stop=sell_stop)

    total = sum(cause_counts.values()) or 1
    ranking = [
        {
            "cause": cause,
            "count": cause_counts[cause],
            "pct": round(100.0 * cause_counts[cause] / total, 2),
            "rank": index + 1,
        }
        for index, cause in enumerate(
            sorted(cause_counts, key=lambda key: cause_counts[key], reverse=True),
        )
    ]

    return {
        "methodology": "Losses attributed by MFE/MAE/timing/throttle heuristics on losing signals only.",
        "total_loss_signals_analyzed": total,
        "cause_ranking": ranking,
        "primary_cause": ranking[0]["cause"] if ranking else None,
    }


def _throttle_lookup(throttle_rules: list[dict[str, Any]]) -> dict[str, str]:
    return {row["regime"]: row["throttle"] for row in throttle_rules}


def _build_paper_real_configs(
    *,
    buy_best: dict[str, Any],
    sell_best: dict[str, Any],
    throttle_rules: list[dict[str, Any]],
    regime_playbook: dict[str, Any],
    deployment_audit: dict[str, Any],
) -> dict[str, Any]:
    deploy_playbook = deployment_audit.get("deployment_playbook", {})
    risk_rules = deploy_playbook.get("risk_rules", {})

    paper = {
        "deployment_mode": "paper_trading",
        "buy_engine": {
            "model_id": BUY_V3_MODEL_ID,
            "formula": BUY_V3_FORMULA_TEXT,
            "stop_variant": "fixed_10",
            "target_structure": buy_best.get("best_tiered_exit", "60/100/Runner"),
            "sizing_mode": "regime_adaptive",
        },
        "sell_engine": {
            "model_id": SELL_V6_MODEL_ID,
            "vwap_gate": V6_VWAP_GATE_RULE,
            "stop_variant": "fixed_10",
            "target_structure": sell_best.get("best_tiered_exit", "60/100/Runner"),
            "sizing_mode": "regime_adaptive",
            "regime_throttle": throttle_rules,
        },
        "regime_execution_overrides": regime_playbook,
        "risk_rules": risk_rules,
    }

    real = {
        "deployment_mode": "real_capital",
        "buy_engine": {
            "model_id": BUY_V3_MODEL_ID,
            "formula": BUY_V3_FORMULA_TEXT,
            "stop_variant": "structure_based",
            "target_structure": buy_best.get("best_tiered_exit", "60/100/Runner"),
            "sizing_mode": "half",
        },
        "sell_engine": {
            "model_id": SELL_V6_MODEL_ID,
            "vwap_gate": V6_VWAP_GATE_RULE,
            "stop_variant": "structure_based",
            "target_structure": sell_best.get("best_tiered_exit", "60/100/Runner"),
            "sizing_mode": "regime_adaptive",
            "regime_throttle": throttle_rules,
        },
        "regime_execution_overrides": regime_playbook,
        "risk_rules": risk_rules,
        "size_discount_pct": 50,
    }

    return {"paper_trading": paper, "real_capital": real}


def _regime_override_map(per_regime_comparison: dict[str, Any]) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for regime, evidence in per_regime_comparison.get("by_regime", {}).items():
        if evidence.get("sample_size", 0) < 3:
            continue
        overrides[regime] = {
            "stop_variant": evidence.get("best_stop_variant"),
            "exit_structure": evidence.get("best_tiered_exit"),
        }
    return overrides


def _near_optimal_verdict(
    *,
    capture_efficiency: float,
    uniform_expectancy: float,
    optimal_expectancy: float,
    win_rate: float,
) -> str:
    gap = abs(optimal_expectancy - uniform_expectancy)
    relative_gap = gap / max(abs(uniform_expectancy), 1.0)
    if capture_efficiency >= 75.0 and relative_gap <= 0.12 and win_rate >= 65.0:
        return "YES"
    if capture_efficiency >= 60.0 and relative_gap <= 0.25:
        return "PARTIAL"
    return "NO"


def _regime_execution_improvement_verdict(
    uniform: dict[str, Any],
    regime_aware: dict[str, Any],
) -> str:
    pf_u = uniform.get("profit_factor") or 0.0
    pf_r = regime_aware.get("profit_factor") or 0.0
    exp_u = uniform.get("expectancy") or 0.0
    exp_r = regime_aware.get("expectancy") or 0.0

    pf_gain = (pf_r - pf_u) / max(pf_u, 0.01)
    exp_gain = (exp_r - exp_u) / max(abs(exp_u), 0.01)

    if pf_gain >= 0.08 and exp_gain >= 0.08:
        return "YES"
    if pf_gain >= 0.03 or exp_gain >= 0.05:
        return "PARTIAL"
    return "NO"


def _production_scores(
    *,
    deployment_audit: dict[str, Any],
    regime_audit: dict[str, Any],
    live_audit: dict[str, Any],
    uniform_combined: dict[str, Any],
    regime_combined: dict[str, Any],
) -> dict[str, Any]:
    deploy_scores = deployment_audit.get("production_scores", {})
    regime_scores = regime_audit.get("final_answer", {}).get("output_metrics", {})
    live_final = live_audit.get("final_answer", {})

    readiness = round(float(deploy_scores.get("production_readiness_score") or 72.0), 1)
    confidence = round(float(deploy_scores.get("confidence_score") or 66.0), 1)
    risk = round(float(deploy_scores.get("production_risk_score") or 68.5), 1)

    improvement = _regime_execution_improvement_verdict(uniform_combined, regime_combined)
    if improvement == "YES":
        confidence = min(95.0, confidence + 4.0)
    elif improvement == "PARTIAL":
        confidence = min(95.0, confidence + 2.0)

    return {
        "production_readiness_score": readiness,
        "confidence_score": confidence,
        "production_risk_score": risk,
        "regime_execution_improvement": improvement,
        "paper_monthly_points": live_final.get("expected_monthly_points", {}).get("paper_combined"),
        "real_monthly_points": live_final.get("expected_monthly_points", {}).get("real_capital_combined"),
        "regime_audit_throttle_pf": regime_audit.get("final_answer", {}).get(
            "throttled_sell_v6_validate_pf",
        ),
    }


def _final_answer(
    *,
    scores: dict[str, Any],
    configs: dict[str, Any],
    buy_uniform: dict[str, Any],
    buy_regime: dict[str, Any],
    sell_uniform: dict[str, Any],
    sell_regime: dict[str, Any],
    combined_uniform: dict[str, Any],
    combined_regime: dict[str, Any],
    buy_near_optimal: str,
    sell_near_optimal: str,
    loss_root_cause: dict[str, Any],
    deployment_audit: dict[str, Any],
    highest_impact: str,
) -> dict[str, Any]:
    regime_improvement = _regime_execution_improvement_verdict(combined_uniform, combined_regime)
    primary_loss = loss_root_cause.get("primary_cause", "execution")

    buy_v4 = "YES" if primary_loss in {"signal_quality", "regime"} and buy_near_optimal in {"NO", "PARTIAL"} else "NO"
    sell_v7 = "YES" if primary_loss in {"signal_quality", "regime"} and sell_near_optimal in {"NO", "PARTIAL"} else "NO"

    return {
        "regime_aware_execution_improves_pf_expectancy": regime_improvement,
        "buy_v3_near_optimal": buy_near_optimal,
        "sell_v6_near_optimal": sell_near_optimal,
        "should_research_buy_v4": buy_v4,
        "should_research_sell_v7": sell_v7,
        "highest_impact_remaining_improvement": highest_impact,
        "paper_trading_config": configs["paper_trading"],
        "real_capital_config": configs["real_capital"],
        "uniform_vs_regime_aware": {
            "buy_v3": {"uniform": buy_uniform, "regime_aware": buy_regime},
            "sell_v6": {"uniform": sell_uniform, "regime_aware": sell_regime},
            "combined": {"uniform": combined_uniform, "regime_aware": combined_regime},
        },
        "production_readiness_score": scores["production_readiness_score"],
        "confidence_score": scores["confidence_score"],
        "production_risk_score": scores["production_risk_score"],
        "paper_trade_tomorrow": deployment_audit.get("final_answer", {}).get("paper_trade_tomorrow", "YES"),
        "real_capital_deployment": deployment_audit.get("final_answer", {}).get("real_capital_deployment", "NO"),
        "rationale": (
            f"Regime-aware execution: {regime_improvement}. "
            f"BUY near-optimal: {buy_near_optimal}; SELL near-optimal: {sell_near_optimal}. "
            f"Primary loss driver: {primary_loss}. Highest impact: {highest_impact}."
        ),
    }


class RegimeAwareExecutionValidationResearch:
    """Synthesize regime-aware execution validation from existing exports."""

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

    def run(self) -> RegimeAwareExecutionValidationReport:
        started = time.perf_counter()
        sources = self._load_sources()

        buy_export = sources["buy_v3_candidate_validation"]["data"]
        sell_export = sources["sell_v6_replay_validation"]["data"]
        regime_audit = sources["regime_detection_audit"]["data"]
        deployment_audit = sources["final_production_deployment_audit"]["data"]
        live_audit = sources["live_trade_management_execution_efficiency_audit"]["data"]

        window_days = int(
            buy_export.get("trading_days_replayed") or sell_export.get("trading_days_replayed") or 120,
        )

        buy_signals = list(buy_export.get("per_signal_details", {}).get("buy_v3") or [])
        sell_signals = list(sell_export.get("per_signal_details", {}).get("sell_v6") or [])
        if not buy_signals:
            raise RegimeAwareExecutionValidationError("No BUY_V3 per_signal_details in exports.")
        if not sell_signals:
            raise RegimeAwareExecutionValidationError("No SELL_V6 per_signal_details in exports.")

        throttle_rules = regime_audit.get("throttle_recommendation", {}).get("sell_v6_regime_throttle", [])
        throttle_map = _throttle_lookup(throttle_rules)

        live_buy_exit = live_audit.get("final_answer", {}).get("optimal_exit_structures", {}).get(
            "buy_v3",
            "60/100/Runner",
        )
        live_sell_exit = live_audit.get("final_answer", {}).get("optimal_exit_structures", {}).get(
            "sell_v6",
            "60/100/Runner",
        )
        live_buy_stop = live_audit.get("final_answer", {}).get("optimal_stops", {}).get("buy_v3", "fixed_10")
        live_sell_stop = live_audit.get("final_answer", {}).get("optimal_stops", {}).get("sell_v6", "fixed_10")

        buy_structure = EXIT_STRUCTURES.get(live_buy_exit, EXIT_STRUCTURES["60/100/Runner"])
        sell_structure = EXIT_STRUCTURES.get(live_sell_exit, EXIT_STRUCTURES["60/100/Runner"])

        regime_classification = {
            "methodology": (
                "SELL uses embedded regime tags where present; BUY inferred from layers/context "
                "via classify_signal_regime (trend/volatility/gap/liquidity composite)."
            ),
            "dimensions": [
                "Strong/Weak Trend",
                "Range",
                "High/Low Volatility",
                "Liquidity Expansion/Compression",
                "Gap Expansion/Compression",
            ],
            "buy_v3": _regime_dimension_counts(buy_signals, direction="BUY"),
            "sell_v6": _regime_dimension_counts(sell_signals, direction="SELL"),
        }

        per_regime_performance = {
            "buy_v3": _per_regime_performance_table(
                buy_signals,
                direction="BUY",
                window_days=window_days,
                win_fn=_is_buy_winner,
                structure=buy_structure,
                stop_variant=live_buy_stop,
            ),
            "sell_v6": _per_regime_performance_table(
                sell_signals,
                direction="SELL",
                window_days=window_days,
                win_fn=_is_sell_winner,
                structure=sell_structure,
                stop_variant=live_sell_stop,
            ),
        }

        buy_regime_stop_exit = _per_regime_stop_exit_comparison(
            buy_signals,
            direction="BUY",
            window_days=window_days,
        )
        sell_regime_stop_exit = _per_regime_stop_exit_comparison(
            sell_signals,
            direction="SELL",
            window_days=window_days,
        )
        per_regime_stop_exit_comparison = {
            "buy_v3": buy_regime_stop_exit,
            "sell_v6": sell_regime_stop_exit,
        }

        execution_failure_audit = {
            "buy_v3": _execution_failure_audit(
                buy_signals,
                structure=buy_structure,
                win_fn=_is_buy_winner,
                window_days=window_days,
            ),
            "sell_v6": _execution_failure_audit(
                sell_signals,
                structure=sell_structure,
                win_fn=_is_sell_winner,
                window_days=window_days,
            ),
        }

        entry_precision = {
            "buy_v3": _entry_precision_analysis(buy_signals, direction="BUY"),
            "sell_v6": _entry_precision_analysis(sell_signals, direction="SELL"),
        }

        capture_leakage = {
            "buy_v3": _capture_leakage_with_regime(
                buy_signals,
                structure=buy_structure,
                stop_variant=live_buy_stop,
                direction="BUY",
                throttle_map=throttle_map,
            ),
            "sell_v6": _capture_leakage_with_regime(
                sell_signals,
                structure=sell_structure,
                stop_variant=live_sell_stop,
                direction="SELL",
                throttle_map=throttle_map,
            ),
        }

        buy_global_best = _best_stop_exit_for_cohort(buy_signals, window_days=window_days)
        sell_global_best = _best_stop_exit_for_cohort(sell_signals, window_days=window_days)

        regime_aware_playbook = {
            "buy_v3_rules_by_regime": _build_regime_playbook_rules(
                buy_regime_stop_exit,
                throttle_rules,
                direction="BUY",
            ),
            "sell_v6_rules_by_regime": _build_regime_playbook_rules(
                sell_regime_stop_exit,
                throttle_rules,
                direction="SELL",
            ),
            "global_defaults": {
                "buy_v3": buy_global_best,
                "sell_v6": sell_global_best,
            },
            "conflict_policy": "NO_TRADE on same-bar opposing signals",
            "reconciled_with": deployment_audit.get("final_answer", {}).get("deployment_tier"),
        }

        paper_vs_real_configs = _build_paper_real_configs(
            buy_best=buy_global_best,
            sell_best=sell_global_best,
            throttle_rules=throttle_rules,
            regime_playbook=regime_aware_playbook,
            deployment_audit=deployment_audit,
        )

        loss_root_cause = _loss_root_cause_analysis(
            buy_signals,
            sell_signals,
            buy_structure=buy_structure,
            sell_structure=sell_structure,
            buy_stop=live_buy_stop,
            sell_stop=live_sell_stop,
            throttle_map=throttle_map,
        )

        buy_overrides = _regime_override_map(buy_regime_stop_exit)
        sell_overrides = _regime_override_map(sell_regime_stop_exit)

        buy_uniform = _simulate_execution(
            buy_signals,
            direction="BUY",
            stop_variant=live_buy_stop,
            structure=buy_structure,
            per_regime_overrides=None,
            throttle_map=throttle_map,
            window_days=window_days,
            sizing_mode="full",
        )
        buy_regime_aware = _simulate_execution(
            buy_signals,
            direction="BUY",
            stop_variant=live_buy_stop,
            structure=buy_structure,
            per_regime_overrides=buy_overrides,
            throttle_map=throttle_map,
            window_days=window_days,
            sizing_mode="full",
        )
        sell_uniform = _simulate_execution(
            sell_signals,
            direction="SELL",
            stop_variant=live_sell_stop,
            structure=sell_structure,
            per_regime_overrides=None,
            throttle_map=throttle_map,
            window_days=window_days,
            sizing_mode="regime_adaptive",
        )
        sell_regime_aware = _simulate_execution(
            sell_signals,
            direction="SELL",
            stop_variant=live_sell_stop,
            structure=sell_structure,
            per_regime_overrides=sell_overrides,
            throttle_map=throttle_map,
            window_days=window_days,
            sizing_mode="regime_adaptive",
        )

        uniform_pnls = buy_uniform.pop("pnls", []) + sell_uniform.pop("pnls", [])
        regime_pnls = buy_regime_aware.pop("pnls", []) + sell_regime_aware.pop("pnls", [])
        combined_uniform = _extended_metrics(
            uniform_pnls,
            signals=buy_signals + sell_signals,
            sample_size=len(uniform_pnls),
            window_days=window_days,
        )
        combined_regime = _extended_metrics(
            regime_pnls,
            signals=buy_signals + sell_signals,
            sample_size=len(regime_pnls),
            window_days=window_days,
        )

        buy_near = _near_optimal_verdict(
            capture_efficiency=buy_uniform["capture_efficiency_pct"],
            uniform_expectancy=buy_uniform["expectancy"],
            optimal_expectancy=buy_regime_aware["expectancy"],
            win_rate=float(buy_export.get("walk_forward", {}).get("validate", {}).get("buy_v3", {}).get(
                "overall_statistics", {},
            ).get("win_rate_pct") or 70.0),
        )
        sell_near = _near_optimal_verdict(
            capture_efficiency=sell_uniform["capture_efficiency_pct"],
            uniform_expectancy=sell_uniform["expectancy"],
            optimal_expectancy=sell_regime_aware["expectancy"],
            win_rate=float(sell_export.get("comparison_table", {}).get("sell_v6", {}).get("win_rate_pct") or 70.0),
        )

        leakage_rank = (
            capture_leakage["sell_v6"].get("leakage_source_ranking")
            or capture_leakage["buy_v3"].get("leakage_source_ranking")
            or []
        )
        top_leak = leakage_rank[0]["source"] if leakage_rank else "runner"
        highest_impact = {
            "late_entry": "Tighten entry timing filter (reject bars_before_expansion < 0)",
            "premature_stop": "Widen structure_based stop in high-vol regimes",
            "target": "Extend T2 runner leg in strong-trend regimes",
            "runner": "Improve runner trail giveback policy beyond T2",
            "regime_filter": "Expand SELL_V6 regime BLOCK map from validate deterioration",
        }.get(top_leak, "Apply regime-adaptive stop/exit overrides per composite regime")

        production_scores = _production_scores(
            deployment_audit=deployment_audit,
            regime_audit=regime_audit,
            live_audit=live_audit,
            uniform_combined=combined_uniform,
            regime_combined=combined_regime,
        )

        final_answer = _final_answer(
            scores=production_scores,
            configs=paper_vs_real_configs,
            buy_uniform=buy_uniform,
            buy_regime=buy_regime_aware,
            sell_uniform=sell_uniform,
            sell_regime=sell_regime_aware,
            combined_uniform=combined_uniform,
            combined_regime=combined_regime,
            buy_near_optimal=buy_near,
            sell_near_optimal=sell_near,
            loss_root_cause=loss_root_cause,
            deployment_audit=deployment_audit,
            highest_impact=highest_impact,
        )

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "no_new_replay": True,
            "no_new_discovery": True,
            "no_new_models": True,
            "no_new_indicators": True,
            "regime_classification": regime_classification["methodology"],
            "stop_variants": list(EXECUTION_STOP_VARIANTS),
            "fixed_exit_targets": list(FIXED_EXIT_TARGETS),
            "tiered_exit_structures": list(EXIT_STRUCTURES.keys()),
            "simulation_basis": (
                "Stops/exits simulated from per_signal_details MFE/MAE — no intrabar sequencing."
            ),
            "production_gates": PRODUCTION_GATES,
            "engines": {"buy_v3": BUY_V3_MODEL_ID, "sell_v6": SELL_V6_MODEL_ID},
        }

        limitations = [
            "No intrabar stop/target sequencing — MFE/MAE proxy only.",
            "BUY regime tags inferred from layers; SELL uses embedded regime where present.",
            "Per-regime stop/exit optima require n≥3 signals per composite bucket.",
            "Entry precision uses points_before_expansion proxy, not tick-level fills.",
            "Regime-aware uplift measured vs uniform playbook, not live broker fills.",
        ]

        conclusions = [
            "Regime-aware execution validation synthesized from 5 replay exports only.",
            f"BUY_V3 classified into {regime_classification['buy_v3']['by_composite'].__len__()} composite regimes.",
            f"SELL_V6 classified into {regime_classification['sell_v6']['by_composite'].__len__()} composite regimes.",
            f"Regime-aware execution improves PF/expectancy: {final_answer['regime_aware_execution_improves_pf_expectancy']}.",
            f"BUY_V3 near-optimal: {final_answer['buy_v3_near_optimal']} | SELL_V6 near-optimal: {final_answer['sell_v6_near_optimal']}.",
            f"Primary loss driver: {loss_root_cause.get('primary_cause')}.",
            f"Highest-impact improvement: {highest_impact}.",
            f"Paper trade: {final_answer['paper_trade_tomorrow']} | Real capital: {final_answer['real_capital_deployment']}.",
        ]

        return RegimeAwareExecutionValidationReport(
            report_type="Regime Aware Execution Validation",
            engines=["BUY_V3", "SELL_V6", "COMBINED"],
            symbol=str(buy_export.get("symbol") or sell_export.get("symbol") or "NIFTY50"),
            timeframe=str(buy_export.get("timeframe") or sell_export.get("timeframe") or "5M"),
            trading_days_replayed=window_days,
            replay_start_date=str(buy_export.get("replay_start_date") or ""),
            replay_end_date=str(buy_export.get("replay_end_date") or ""),
            methodology=methodology,
            source_exports={name: {"path": meta["path"], "status": meta["status"]} for name, meta in sources.items()},
            limitations=limitations,
            regime_classification=regime_classification,
            per_regime_performance=per_regime_performance,
            per_regime_stop_exit_comparison=per_regime_stop_exit_comparison,
            execution_failure_audit=execution_failure_audit,
            entry_precision=entry_precision,
            capture_leakage=capture_leakage,
            regime_aware_playbook=regime_aware_playbook,
            paper_vs_real_configs=paper_vs_real_configs,
            loss_root_cause=loss_root_cause,
            production_scores=production_scores,
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: RegimeAwareExecutionValidationReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Regime-aware execution validation exported to %s", self.report_path)
        return self.report_path


def generate_regime_aware_execution_validation_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export regime-aware execution validation JSON."""
    return RegimeAwareExecutionValidationResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_regime_aware_execution_validation_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    print(f"Exported: {path}")
    print(f"Regime-aware improves: {final['regime_aware_execution_improves_pf_expectancy']}")
    print(f"BUY near-optimal: {final['buy_v3_near_optimal']} | SELL near-optimal: {final['sell_v6_near_optimal']}")
    print(f"Highest impact: {final['highest_impact_remaining_improvement']}")
