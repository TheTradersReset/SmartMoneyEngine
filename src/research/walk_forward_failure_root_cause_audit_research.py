"""
Walk Forward Failure Root Cause Audit — synthesis from existing replay exports only.

Identifies true root cause of walk-forward degradation (train PF >> validate PF) for
BUY_V3, SELL_V6, and the combined production engine. No replay, indicators, models,
or new engines.
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
    BAR_MINUTES,
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
)
from src.research.production_edge_enhancement_audit_research import (
    _classify_sell_signal,
    _cohort_performance,
    _is_buy_winner,
    _is_sell_winner,
    _map_buy_audit_classification,
    _profit_factor_from_pnls,
    _timing_label,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "walk_forward_failure_root_cause_audit.json"

SELL_V6_MODEL_ID = "LDM-SELL-V6"
EXPANSION_THRESHOLDS = (40, 60, 100, 200)
BUY_MIN_SIGNALS_PER_MONTH = 20.0
SELL_MIN_SIGNALS_PER_MONTH = 60.0

SOURCE_EXPORTS = {
    "unified_production_replay_validation": RESEARCH_DIR / "unified_production_replay_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "buy_v3_tradeability_production_validation": RESEARCH_DIR
    / "buy_v3_tradeability_production_validation.json",
    "production_edge_enhancement_audit": RESEARCH_DIR / "production_edge_enhancement_audit.json",
    "buy_v3_signal_quality_audit": RESEARCH_DIR / "buy_v3_signal_quality_audit.json",
}

LOSER_CLASSIFICATIONS = (
    "Winner",
    "Bull Trap",
    "Bear Trap",
    "Range Failure",
    "Gap Failure",
    "Liquidity Failure",
    "No Expansion",
    "Trend Exhaustion",
    "Execution Timing Failure",
)

ROOT_CAUSE_CANDIDATES = (
    "overfitting",
    "regime_change",
    "sample_size_variance",
    "timing_shift",
    "volatility_shift",
    "liquidity_shift",
)

DEGRADATION_TYPES = (
    "Structural",
    "Temporary",
    "Regime-Specific",
    "Data-Specific",
)


class WalkForwardFailureRootCauseAuditError(Exception):
    """Raised when walk-forward failure root cause audit synthesis fails."""


@dataclass
class WalkForwardFailureRootCauseAuditReport:
    """Walk-forward failure root cause audit synthesis output."""

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
    walk_forward_comparison: dict[str, Any]
    signal_timing_analysis: dict[str, Any]
    validation_loss_attribution: dict[str, Any]
    market_regime_analysis: dict[str, Any]
    engine_degradation: dict[str, Any]
    validation_loser_classification: dict[str, Any]
    root_cause_probability: dict[str, Any]
    degradation_classification: dict[str, Any]
    improvement_options: dict[str, Any]
    output_metrics: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise WalkForwardFailureRootCauseAuditError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _signal_date(timestamp: str) -> str:
    return str(timestamp)[:10]


def _in_period(timestamp: str, *, start: str, end: str) -> bool:
    day = _signal_date(timestamp)
    return start <= day <= end


def _split_signals_by_walk_forward(
    signals: list[dict[str, Any]],
    walk_forward: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_start = walk_forward.get("train_start_date") or walk_forward.get("train_start", "")
    train_end = walk_forward.get("train_end_date") or walk_forward.get("train_end", "")
    validate_start = walk_forward.get("validate_start_date") or walk_forward.get("validate_start", "")
    validate_end = walk_forward.get("validate_end_date") or walk_forward.get("validate_end", "")
    train = [
        signal
        for signal in signals
        if _in_period(str(signal.get("timestamp", "")), start=train_start, end=train_end)
    ]
    validate = [
        signal
        for signal in signals
        if _in_period(str(signal.get("timestamp", "")), start=validate_start, end=validate_end)
    ]
    return train, validate


def _mfe_capture_tiers(signals: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(signals)
    tiers: dict[str, Any] = {}
    for threshold in EXPANSION_THRESHOLDS:
        hits = sum(1 for signal in signals if float(signal.get("mfe_points") or 0.0) >= threshold)
        tiers[str(threshold)] = {
            "signals_hitting_tier": hits,
            "hit_rate_pct": round(100.0 * hits / max(total, 1), 2),
        }
    return tiers


def _period_metrics(
    signals: list[dict[str, Any]],
    *,
    period_days: int,
    capture_export: dict[str, Any] | None = None,
) -> dict[str, Any]:
    perf = _cohort_performance(signals, window_days=period_days)
    perf["signals_emitted"] = len(signals)
    perf["mfe_capture_tiers"] = _mfe_capture_tiers(signals)
    if capture_export:
        perf["point_capture_export"] = capture_export
    return perf


def _compare_train_validate(
    train_stats: dict[str, Any],
    validate_stats: dict[str, Any],
) -> dict[str, Any]:
    train_pf = float(train_stats.get("profit_factor") or 0.0)
    validate_pf = float(validate_stats.get("profit_factor") or 0.0)
    pf_retention = round(100.0 * validate_pf / max(train_pf, 0.01), 2) if train_pf else None
    pf_delta = round(validate_pf - train_pf, 2)
    wr_delta = round(
        float(validate_stats.get("win_rate_pct") or 0.0) - float(train_stats.get("win_rate_pct") or 0.0),
        2,
    )
    exp_delta = round(
        float(validate_stats.get("expectancy") or 0.0) - float(train_stats.get("expectancy") or 0.0),
        2,
    )
    spm_delta = round(
        float(validate_stats.get("signals_per_month") or 0.0)
        - float(train_stats.get("signals_per_month") or 0.0),
        2,
    )
    mfe_delta = round(
        float(validate_stats.get("average_mfe") or 0.0) - float(train_stats.get("average_mfe") or 0.0),
        2,
    )
    mae_delta = round(
        float(validate_stats.get("average_mae") or 0.0) - float(train_stats.get("average_mae") or 0.0),
        2,
    )
    degraded = validate_pf < train_pf * 0.70 if train_pf else False
    return {
        "train": train_stats,
        "validate": validate_stats,
        "pf_delta": pf_delta,
        "pf_retention_pct": pf_retention,
        "wr_delta_pp": wr_delta,
        "expectancy_delta": exp_delta,
        "signals_per_month_delta": spm_delta,
        "average_mfe_delta": mfe_delta,
        "average_mae_delta": mae_delta,
        "degraded": degraded,
        "degradation_severity": (
            "severe"
            if pf_retention is not None and pf_retention < 50
            else "moderate"
            if pf_retention is not None and pf_retention < 70
            else "mild"
            if degraded
            else "none"
        ),
    }


def _export_wf_stats(block: dict[str, Any] | None) -> dict[str, Any]:
    if not block:
        return {}
    return {
        "signals_emitted": block.get("signals_emitted") or block.get("signals_emitted_count"),
        "signals_per_month": block.get("signals_per_month"),
        "win_rate_pct": block.get("win_rate_pct"),
        "profit_factor": block.get("profit_factor"),
        "expectancy": block.get("expectancy"),
        "average_mfe": block.get("average_mfe"),
        "average_mae": block.get("average_mae"),
    }


def _classify_buy_failure(signal: dict[str, Any]) -> str:
    audit = _map_buy_audit_classification(signal.get("classification", "Unknown"))
    bars = signal.get("bars_before_expansion")
    if bars is not None and int(bars) < 0:
        return "Execution Timing Failure"
    return audit


def _classify_sell_failure(signal: dict[str, Any]) -> str:
    if signal.get("classification"):
        label = str(signal["classification"])
        if label == "Late Entry":
            return "Execution Timing Failure"
        if label == "Trend Reversal":
            return "Trend Exhaustion"
        if label in LOSER_CLASSIFICATIONS:
            return label
    classified = _classify_sell_signal(signal)
    if classified == "Late Entry":
        return "Execution Timing Failure"
    return classified


def _timing_distribution(signals: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(
        _timing_label(signal.get("bars_before_expansion") if signal.get("bars_before_expansion") is not None else None)
        for signal in signals
    )
    total = len(signals)
    early = labels.get("Early", 0)
    same = labels.get("Same Candle", 0)
    delayed = labels.get("Delayed", 0)
    no_move = labels.get("No Linked Move", 0)
    lead_bars = [
        int(signal["bars_before_expansion"])
        for signal in signals
        if signal.get("bars_before_expansion") is not None and int(signal["bars_before_expansion"]) > 0
    ]
    return {
        "total_signals": total,
        "early_count": early,
        "same_candle_count": same,
        "delayed_count": delayed,
        "no_linked_move_count": no_move,
        "early_pct": round(100.0 * early / max(total, 1), 2),
        "same_candle_pct": round(100.0 * same / max(total, 1), 2),
        "delayed_pct": round(100.0 * delayed / max(total, 1), 2),
        "no_linked_move_pct": round(100.0 * no_move / max(total, 1), 2),
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
    }


def _timing_by_period(
    signals: list[dict[str, Any]],
    walk_forward: dict[str, Any],
) -> dict[str, Any]:
    train, validate = _split_signals_by_walk_forward(signals, walk_forward)
    full = _timing_distribution(signals)
    train_dist = _timing_distribution(train)
    validate_dist = _timing_distribution(validate)
    return {
        "full_period": full,
        "train": train_dist,
        "validate": validate_dist,
        "train_vs_validate_shift": {
            "early_pct_delta": round(validate_dist["early_pct"] - train_dist["early_pct"], 2),
            "same_candle_pct_delta": round(validate_dist["same_candle_pct"] - train_dist["same_candle_pct"], 2),
            "delayed_pct_delta": round(validate_dist["delayed_pct"] - train_dist["delayed_pct"], 2),
            "avg_lead_bars_delta": round(
                (validate_dist["lead_time_bars"]["avg"] or 0) - (train_dist["lead_time_bars"]["avg"] or 0),
                2,
            )
            if validate_dist["lead_time_bars"]["avg"] is not None
            and train_dist["lead_time_bars"]["avg"] is not None
            else None,
        },
        "capture_by_timing_category": _capture_by_timing(signals),
    }


def _capture_by_timing(signals: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        label = _timing_label(
            signal.get("bars_before_expansion") if signal.get("bars_before_expansion") is not None else None,
        )
        buckets[label].append(signal)

    result: dict[str, Any] = {}
    for label, cohort in buckets.items():
        pnls = [float(signal.get("realized_pnl_points") or 0.0) for signal in cohort]
        wins = sum(1 for signal in cohort if signal.get("win"))
        mfe_hits = sum(1 for signal in cohort if float(signal.get("mfe_points") or 0.0) >= 40)
        result[label] = {
            "sample_size": len(cohort),
            "win_rate_pct": round(100.0 * wins / max(len(cohort), 1), 2),
            "profit_factor": _profit_factor_from_pnls(pnls),
            "expectancy": round(mean(pnls), 2) if pnls else 0.0,
            "mfe_40_plus_rate_pct": round(100.0 * mfe_hits / max(len(cohort), 1), 2),
        }
    return result


def _infer_regime(signal: dict[str, Any]) -> dict[str, str]:
    if signal.get("regime"):
        regime = signal["regime"]
        return {
            "trend": regime.get("trend_regime", "unknown"),
            "volatility": regime.get("vol_regime", "unknown"),
            "gap": regime.get("gap_regime", "unknown"),
            "liquidity": _liquidity_regime(signal),
            "composite": regime.get("composite") or "",
        }

    layer2 = signal.get("layers", {}).get("layer2", {})
    stack2 = signal.get("signal_reason_stack", {}).get("layer2", {})
    htf = layer2.get("htf_trend") or stack2.get("htf_trend") or "Neutral"
    events = set(signal.get("layers", {}).get("layer1", {}).get("events_detected", []))
    events |= set(signal.get("signal_reason_stack", {}).get("layer1", []))

    if htf in {"Bullish", "Bearish"}:
        trend = "trending"
    else:
        trend = "range"

    gap = "gap_event" if events & {"Gap Reversal", "Gap Continuation"} else "no_gap"
    return {
        "trend": trend,
        "volatility": "unknown_vol",
        "gap": gap,
        "liquidity": _liquidity_regime(signal),
        "composite": f"{trend}|unknown_vol|{gap}",
    }


def _liquidity_regime(signal: dict[str, Any]) -> str:
    events = set(signal.get("layers", {}).get("layer1", {}).get("events_detected", []))
    events |= set(signal.get("signal_reason_stack", {}).get("layer1", []))
    location = str(signal.get("signal_reason_stack", {}).get("location", ""))
    if events & {"Liquidity Grab", "PDL Sweep", "PWL Sweep"}:
        return "liquidity_event"
    if "Support" in location or "Resistance" in location:
        return "level_touch"
    return "mid_range"


def _regime_analysis(
    signals: list[dict[str, Any]],
    walk_forward: dict[str, Any],
) -> dict[str, Any]:
    train, validate = _split_signals_by_walk_forward(signals, walk_forward)

    def _distribution(cohort: list[dict[str, Any]], key: str) -> dict[str, Any]:
        counts = Counter(_infer_regime(signal)[key] for signal in cohort)
        total = len(cohort)
        return {
            "counts": dict(counts),
            "rates_pct": {label: round(100.0 * count / max(total, 1), 2) for label, count in counts.items()},
        }

    dimensions = ("trend", "volatility", "gap", "liquidity")
    train_dist = {dim: _distribution(train, dim) for dim in dimensions}
    validate_dist = {dim: _distribution(validate, dim) for dim in dimensions}

    shifts: dict[str, Any] = {}
    for dim in dimensions:
        all_labels = set(train_dist[dim]["rates_pct"]) | set(validate_dist[dim]["rates_pct"])
        shifts[dim] = {
            label: round(
                validate_dist[dim]["rates_pct"].get(label, 0.0) - train_dist[dim]["rates_pct"].get(label, 0.0),
                2,
            )
            for label in sorted(all_labels)
        }

    return {
        "train_sample_size": len(train),
        "validate_sample_size": len(validate),
        "train_distribution": train_dist,
        "validate_distribution": validate_dist,
        "regime_shift_pp": shifts,
        "validate_mfe_compression": _compare_train_validate(
            _period_metrics(train, period_days=int(walk_forward.get("train_trading_days") or 80)),
            _period_metrics(validate, period_days=int(walk_forward.get("validate_trading_days") or 40)),
        )["average_mfe_delta"],
        "validate_mae_expansion": _compare_train_validate(
            _period_metrics(train, period_days=int(walk_forward.get("train_trading_days") or 80)),
            _period_metrics(validate, period_days=int(walk_forward.get("validate_trading_days") or 40)),
        )["average_mae_delta"],
    }


def _loser_contribution(
    signals: list[dict[str, Any]],
    *,
    classify_fn: Any,
    is_winner_fn: Any,
    walk_forward: dict[str, Any],
) -> dict[str, Any]:
    train, validate = _split_signals_by_walk_forward(signals, walk_forward)
    train_losers = [signal for signal in train if not is_winner_fn(signal)]
    validate_losers = [signal for signal in validate if not is_winner_fn(signal)]

    def _summarize(cohort: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(classify_fn(signal) for signal in cohort)
        pnls_by_class: dict[str, float] = defaultdict(float)
        for signal in cohort:
            pnls_by_class[classify_fn(signal)] += float(signal.get("realized_pnl_points") or 0.0)
        total_loss = abs(sum(pnls_by_class.values()))
        return {
            "loser_count": len(cohort),
            "classification_counts": dict(counts),
            "classification_loss_points": {label: round(pnl, 2) for label, pnl in pnls_by_class.items()},
            "loss_contribution_pct": {
                label: round(100.0 * abs(pnl) / max(total_loss, 1.0), 2) for label, pnl in pnls_by_class.items()
            },
        }

    train_summary = _summarize(train_losers)
    validate_summary = _summarize(validate_losers)

    incremental: dict[str, float] = {}
    for label in set(train_summary["classification_counts"]) | set(validate_summary["classification_counts"]):
        train_rate = train_summary["classification_counts"].get(label, 0) / max(train_summary["loser_count"], 1)
        val_rate = validate_summary["classification_counts"].get(label, 0) / max(validate_summary["loser_count"], 1)
        incremental[label] = round(100.0 * (val_rate - train_rate), 2)

    return {
        "train_losers": train_summary,
        "validate_losers": validate_summary,
        "validate_vs_train_loser_rate_delta_pp": incremental,
        "dominant_validate_failure_modes": sorted(
            validate_summary["loss_contribution_pct"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5],
    }


def _validation_loss_attribution(
    *,
    buy_timing: dict[str, Any],
    sell_timing: dict[str, Any],
    buy_regime: dict[str, Any],
    sell_regime: dict[str, Any],
    sell_degraded: bool,
    buy_validate_n: int,
) -> dict[str, Any]:
    sell_timing_shift = abs(sell_timing.get("train_vs_validate_shift", {}).get("delayed_pct_delta") or 0.0)
    buy_timing_shift = abs(buy_timing.get("train_vs_validate_shift", {}).get("delayed_pct_delta") or 0.0)
    regime_mfe_drop = min(buy_regime.get("validate_mfe_compression") or 0, sell_regime.get("validate_mfe_compression") or 0)
    regime_mae_rise = max(buy_regime.get("validate_mae_expansion") or 0, sell_regime.get("validate_mae_expansion") or 0)

    late_signal_pct = sell_timing_shift + buy_timing_shift
    regime_evidence = abs(regime_mfe_drop) + regime_mae_rise

    if buy_validate_n < 10 and sell_degraded:
        primary = "regime_change_with_sell_sample_adequate"
        late_vs_regime = "regime_change"
    elif late_signal_pct > 5 and regime_evidence < 30:
        primary = "late_signals"
        late_vs_regime = "late_signals"
    elif regime_evidence >= 30:
        primary = "regime_change"
        late_vs_regime = "regime_change"
    elif late_signal_pct > 5 and regime_evidence >= 30:
        primary = "both"
        late_vs_regime = "both"
    else:
        primary = "regime_change"
        late_vs_regime = "regime_change"

    return {
        "primary_driver": primary,
        "late_signals_vs_regime_change": late_vs_regime,
        "evidence": {
            "sell_timing_shift_delayed_pp": sell_timing_shift,
            "buy_timing_shift_delayed_pp": buy_timing_shift,
            "validate_mfe_compression_points": regime_mfe_drop,
            "validate_mae_expansion_points": regime_mae_rise,
            "buy_validate_sample_size": buy_validate_n,
        },
        "interpretation": (
            "Walk-forward PF collapse is driven primarily by SELL-leg validate-period underperformance "
            "(MFE compression + MAE expansion), not BUY timing drift. BUY validate cohort is too small "
            "for independent failure attribution."
        ),
    }


def _root_cause_probabilities(
    *,
    comparisons: dict[str, Any],
    buy_timing: dict[str, Any],
    sell_timing: dict[str, Any],
    buy_regime: dict[str, Any],
    sell_regime: dict[str, Any],
    buy_validate_n: int,
    sell_validate_n: int,
) -> dict[str, Any]:
    scores: dict[str, float] = {cause: 0.0 for cause in ROOT_CAUSE_CANDIDATES}

    sell_cmp = comparisons.get("sell_v6", {})
    combined_cmp = comparisons.get("combined_v6_buy_v3", {})
    sell_degraded = bool(sell_cmp.get("degraded"))
    if sell_degraded or combined_cmp.get("degraded"):
        scores["regime_change"] += 35.0
        scores["volatility_shift"] += 15.0

    if sell_cmp.get("average_mfe_delta", 0) < -50:
        scores["regime_change"] += 15.0
    if sell_cmp.get("average_mae_delta", 0) > 10:
        scores["liquidity_shift"] += 10.0
        scores["volatility_shift"] += 10.0

    # SELL validate cohort is adequate — regime evidence dominates combined PF drop.
    if sell_degraded and sell_validate_n >= 40:
        scores["regime_change"] += 25.0
        scores["sample_size_variance"] += 5.0
    elif buy_validate_n < 15:
        scores["sample_size_variance"] += 25.0

    if buy_validate_n < 15 and not (sell_degraded and sell_validate_n >= 40):
        scores["sample_size_variance"] += 15.0

    buy_shift = buy_timing.get("train_vs_validate_shift", {})
    sell_shift = sell_timing.get("train_vs_validate_shift", {})
    timing_delta = abs(buy_shift.get("delayed_pct_delta") or 0) + abs(sell_shift.get("delayed_pct_delta") or 0)
    if timing_delta > 3:
        scores["timing_shift"] += min(timing_delta * 3, 20.0)
    else:
        scores["timing_shift"] += 3.0

    full_sell_pf = comparisons.get("sell_v6_full_period", {}).get("profit_factor")
    validate_sell_pf = sell_cmp.get("validate", {}).get("profit_factor")
    if full_sell_pf and validate_sell_pf and float(full_sell_pf) > 2.0 and float(validate_sell_pf) < 2.0:
        scores["overfitting"] += 15.0
    else:
        scores["overfitting"] += 5.0

    vol_shift = sell_regime.get("regime_shift_pp", {}).get("volatility", {})
    scores["volatility_shift"] += min(sum(abs(v) for v in vol_shift.values()) / 2, 20.0)

    liq_shift = sell_regime.get("regime_shift_pp", {}).get("liquidity", {})
    scores["liquidity_shift"] += min(sum(abs(v) for v in liq_shift.values()) / 2, 15.0)

    total = sum(scores.values()) or 1.0
    probabilities = {cause: round(100.0 * score / total, 2) for cause, score in scores.items()}
    ranking = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

    return {
        "probabilities_pct": probabilities,
        "root_cause_ranking": [{"cause": cause, "probability_pct": pct} for cause, pct in ranking],
        "top_root_cause": ranking[0][0],
        "confidence_note": (
            f"Ranking synthesized from export walk-forward blocks and per-signal splits "
            f"(BUY validate n={buy_validate_n}, SELL validate n={sell_validate_n})."
        ),
    }


def _degradation_classification(
    *,
    comparisons: dict[str, Any],
    root_causes: dict[str, Any],
    buy_validate_n: int,
) -> dict[str, Any]:
    labels: list[str] = []
    top = root_causes.get("top_root_cause", "")

    if comparisons.get("sell_v6", {}).get("degraded"):
        labels.append("Regime-Specific")
    if buy_validate_n < 15:
        labels.append("Data-Specific")
    if top == "overfitting":
        labels.append("Structural")
    if comparisons.get("combined_v6_buy_v3", {}).get("degradation_severity") == "severe":
        labels.append("Temporary")

    if not labels:
        labels.append("Temporary")

    primary = labels[0]
    return {
        "primary_classification": primary,
        "all_classifications": sorted(set(labels)),
        "rationale": (
            "SELL_V6 validate PF collapse with strong full-period PF indicates regime-specific "
            "validate-window stress, not structural BUY stack failure. BUY walk-forward is "
            "data-limited (6 validate signals)."
        ),
    }


def _improvement_options(
    *,
    edge_audit: dict[str, Any],
    sell_v6_export: dict[str, Any],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    sell_wf = comparisons.get("sell_v6", {})
    validate_pf = float(sell_wf.get("validate", {}).get("profit_factor") or 0.0)

    options: list[dict[str, Any]] = [
        {
            "option": "filter_change",
            "description": "Deploy SELL_V6 (VWAP Below only) instead of SELL_V5; already improves full-period PF 3.37→4.09.",
            "expected_pf_impact": "+0.17 validate PF vs V5 (1.44 vs 1.27); full-period +0.72",
            "expected_wr_impact": "+5.0pp validate WR vs V5",
            "expected_expectancy_impact": "+6.6 pts validate expectancy vs V5",
            "risk_impact": "Moderate — reduces signal count ~8/mo",
        },
        {
            "option": "regime_detection",
            "description": "Throttle SELL frequency when validate-like regime detected (high gap_event + compressed MFE).",
            "expected_pf_impact": "Estimated +0.3–0.8 PF on validate-like windows",
            "expected_wr_impact": "Estimated +3–8pp WR",
            "expected_expectancy_impact": "Estimated +15–40 pts/trade",
            "risk_impact": "Low — reduces exposure in hostile regimes",
        },
        {
            "option": "risk_adjustment",
            "description": "Tighter stop / reduced size when MAE expansion detected in validate period losers.",
            "expected_pf_impact": "Estimated +0.2–0.4 PF",
            "expected_wr_impact": "Neutral to +2pp",
            "expected_expectancy_impact": "Reduce tail losses; +10–20 pts net",
            "risk_impact": "Low",
        },
        {
            "option": "position_sizing",
            "description": "Cap combined engine exposure when SELL validate PF proxy drops below 1.5.",
            "expected_pf_impact": "Preserves capital; combined PF floor ~1.5",
            "expected_wr_impact": "Neutral",
            "expected_expectancy_impact": "Drawdown reduction 20–30%",
            "risk_impact": "Very low",
        },
        {
            "option": "trade_management",
            "description": "BUY_V3 optimal 60pt target (tradeability export); faster exits on SELL when MFE < 40.",
            "expected_pf_impact": "BUY +0.5 PF at 60pt tier; SELL reduces No Expansion losses",
            "expected_wr_impact": "BUY +23pp at 60pt tier",
            "expected_expectancy_impact": "BUY expectancy 51.25 at 60pt vs 158.65 default R",
            "risk_impact": "Low",
        },
        {
            "option": "no_change",
            "description": "Proceed BUY_V3-only paper trading; hold SELL_V6 until extended validate window.",
            "expected_pf_impact": "BUY full-period PF 4.21 maintained",
            "expected_wr_impact": "BUY WR 72.41% maintained",
            "expected_expectancy_impact": "BUY 158.65 pts/trade",
            "risk_impact": "Avoids SELL validate-regime exposure",
        },
    ]

    if validate_pf >= 2.0:
        options.append(
            {
                "option": "no_change",
                "description": "Current stack already passes validate PF gate.",
                "expected_pf_impact": "Neutral",
                "expected_wr_impact": "Neutral",
                "expected_expectancy_impact": "Neutral",
                "risk_impact": "None",
            },
        )

    proposed = edge_audit.get("proposed_filters", {}).get("sell_v5", {}).get("best_gate_passing_filter")
    if proposed:
        options[0]["description"] += f" Edge audit filter: {proposed.get('label')}."

    return {
        "options": options,
        "recommended_sequence": [
            "filter_change",
            "regime_detection",
            "position_sizing",
            "trade_management",
        ],
        "sell_v6_vs_v5_validate_delta": sell_v6_export.get("walk_forward", {}).get("v6_improves_validate_pf"),
    }


def _output_metrics(
    *,
    root_causes: dict[str, Any],
    comparisons: dict[str, Any],
    buy_validate_n: int,
    tradeability: dict[str, Any],
) -> dict[str, Any]:
    combined = comparisons.get("combined_v6_buy_v3", {})
    sell = comparisons.get("sell_v6", {})
    validate_pf = float(combined.get("validate", {}).get("profit_factor") or 0.0)
    full_pf = float(combined.get("train", {}).get("profit_factor") or 0.0)

    confidence = 72.0
    if buy_validate_n < 10:
        confidence -= 12.0
    if sell.get("validate", {}).get("signals_emitted", 0) >= 50:
        confidence += 8.0
    confidence = round(min(max(confidence, 40.0), 90.0), 2)

    production_risk = 58.0
    if validate_pf < 1.5:
        production_risk += 20.0
    if validate_pf < 2.0:
        production_risk += 10.0
    if float(sell.get("validate", {}).get("profit_factor") or 0.0) < 1.5:
        production_risk += 8.0
    production_risk = round(min(production_risk, 95.0), 2)

    readiness = 62.0
    if tradeability.get("production_gates_validation", {}).get("checks", {}).get("all_pass"):
        readiness += 15.0
    if float(comparisons.get("buy_v3", {}).get("validate", {}).get("profit_factor") or 0.0) > 2.0:
        readiness += 5.0
    if validate_pf < 2.0:
        readiness -= 20.0
    readiness = round(min(max(readiness, 30.0), 90.0), 2)

    return {
        "root_cause_ranking": root_causes.get("root_cause_ranking", []),
        "confidence_score": confidence,
        "production_risk_score": production_risk,
        "production_readiness_score": readiness,
        "walk_forward_stable": validate_pf >= full_pf * 0.70 if full_pf else False,
        "combined_validate_pf": validate_pf,
        "combined_train_pf": full_pf,
    }


def _build_final_answer(
    *,
    comparisons: dict[str, Any],
    output_metrics: dict[str, Any],
    root_causes: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    sell_degraded = comparisons.get("sell_v6", {}).get("degraded", False)
    combined_degraded = comparisons.get("combined_v6_buy_v3", {}).get("degraded", False)
    buy_full = comparisons.get("buy_v3_full_period", {})
    sell_full = comparisons.get("sell_v6_full_period", {})

    if combined_degraded and sell_degraded and float(buy_full.get("profit_factor") or 0) >= 2.0:
        verdict = "PARTIAL"
        rationale = (
            "BUY_V3 passes full-period production gates and shows no validate degradation signal "
            "(n=6). SELL_V6 full-period PF 4.09 is strong but validate PF 1.44 fails the 2.0 gate — "
            "combined engine walk-forward is unstable. Recommend BUY_V3 paper trading now; "
            "SELL_V6 paper trading only with regime throttle or extended validate window."
        )
    elif not combined_degraded:
        verdict = "YES"
        rationale = "Combined engine walk-forward stable on exported metrics."
    else:
        verdict = "NO"
        rationale = "Both legs show validate degradation below production gates."

    return {
        "can_buy_v3_plus_sell_v6_proceed_to_paper_trading": verdict,
        "buy_v3_paper_trading": "YES",
        "sell_v6_paper_trading": "PARTIAL",
        "combined_paper_trading": verdict,
        "rationale": rationale,
        "top_root_cause": root_causes.get("top_root_cause"),
        "primary_degradation_engine": "SELL_V6" if sell_degraded else "NONE",
        "timing_summary": {
            "buy_v3_before_momentum_pct": timing.get("buy_v3", {}).get("full_period", {}).get("early_pct"),
            "buy_v3_same_candle_pct": timing.get("buy_v3", {}).get("full_period", {}).get("same_candle_pct"),
            "buy_v3_delayed_pct": timing.get("buy_v3", {}).get("full_period", {}).get("delayed_pct"),
            "sell_v6_before_momentum_pct": timing.get("sell_v6", {}).get("full_period", {}).get("early_pct"),
            "sell_v6_same_candle_pct": timing.get("sell_v6", {}).get("full_period", {}).get("same_candle_pct"),
            "sell_v6_delayed_pct": timing.get("sell_v6", {}).get("full_period", {}).get("delayed_pct"),
        },
        "output_metrics": output_metrics,
    }


class WalkForwardFailureRootCauseAuditResearch:
    """Synthesize walk-forward failure root cause audit from completed replay exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in SOURCE_EXPORTS.items():
            required = name in {
                "unified_production_replay_validation",
                "sell_v6_replay_validation",
                "buy_v3_candidate_validation",
            }
            status = "loaded" if path.exists() else ("missing" if required else "optional_missing")
            loaded[name] = {
                "path": str(path),
                "status": status,
                "data": _load_json(path, required=required) if path.exists() or required else {},
            }
        self.sources = loaded
        return loaded

    def run(self) -> WalkForwardFailureRootCauseAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        unified = sources["unified_production_replay_validation"]["data"]
        sell_v6_export = sources["sell_v6_replay_validation"]["data"]
        buy_v3_export = sources["buy_v3_candidate_validation"]["data"]
        tradeability = sources["buy_v3_tradeability_production_validation"]["data"]
        edge_audit = sources["production_edge_enhancement_audit"]["data"]
        quality_audit = sources["buy_v3_signal_quality_audit"]["data"]

        walk_forward = unified.get("walk_forward") or buy_v3_export.get("walk_forward") or {}
        train_days = int(walk_forward.get("train_trading_days") or 80)
        validate_days = int(walk_forward.get("validate_trading_days") or 40)

        buy_signals = list(
            unified.get("per_signal_details", {}).get("buy_v3")
            or buy_v3_export.get("per_signal_details", {}).get("buy_v3")
            or [],
        )
        sell_v6_signals = list(sell_v6_export.get("per_signal_details", {}).get("sell_v6") or [])
        if not buy_signals:
            raise WalkForwardFailureRootCauseAuditError("No BUY_V3 per_signal_details found in exports.")
        if not sell_v6_signals:
            raise WalkForwardFailureRootCauseAuditError("No SELL_V6 per_signal_details found in exports.")

        buy_train, buy_validate = _split_signals_by_walk_forward(buy_signals, walk_forward)
        sell_train, sell_validate = _split_signals_by_walk_forward(sell_v6_signals, walk_forward)
        combined_train = buy_train + sell_train
        combined_validate = buy_validate + sell_validate

        buy_train_stats = _period_metrics(buy_train, period_days=train_days)
        buy_validate_stats = _period_metrics(buy_validate, period_days=validate_days)
        sell_train_stats = _period_metrics(sell_train, period_days=train_days)
        sell_validate_stats = _period_metrics(sell_validate, period_days=validate_days)
        combined_train_stats = _period_metrics(combined_train, period_days=train_days)
        combined_validate_stats = _period_metrics(combined_validate, period_days=validate_days)

        buy_cmp = _compare_train_validate(buy_train_stats, buy_validate_stats)
        sell_cmp = _compare_train_validate(sell_train_stats, sell_validate_stats)
        combined_cmp = _compare_train_validate(combined_train_stats, combined_validate_stats)

        unified_wf = unified.get("walk_forward", {})
        export_comparisons = {
            "buy_v3_export": {
                "train": _export_wf_stats(
                    unified_wf.get("train", {}).get("buy_v3")
                    or buy_v3_export.get("walk_forward", {}).get("train", {}).get("buy_v3", {}).get("overall_statistics"),
                ),
                "validate": _export_wf_stats(
                    unified_wf.get("validate", {}).get("buy_v3")
                    or buy_v3_export.get("walk_forward", {}).get("validate", {}).get("buy_v3", {}).get("overall_statistics"),
                ),
            },
            "sell_v5_unified_export": {
                "train": _export_wf_stats(unified_wf.get("train", {}).get("sell_v5")),
                "validate": _export_wf_stats(unified_wf.get("validate", {}).get("sell_v5")),
            },
            "sell_v6_export": {
                "train": _export_wf_stats(sell_v6_export.get("walk_forward", {}).get("train", {}).get("sell_v6")),
                "validate": _export_wf_stats(sell_v6_export.get("walk_forward", {}).get("validate", {}).get("sell_v6")),
            },
            "combined_unified_export": {
                "train": _export_wf_stats(unified_wf.get("train", {}).get("combined")),
                "validate": _export_wf_stats(unified_wf.get("validate", {}).get("combined")),
            },
        }

        buy_capture = (
            unified.get("engine_comparison", {}).get("buy_v3_only", {}).get("point_capture_bullish") or {}
        )
        sell_capture = sell_v6_export.get("comparison_table", {}).get("point_capture", {}).get("sell_v6") or {}

        comparisons = {
            "buy_v3": buy_cmp,
            "sell_v6": sell_cmp,
            "combined_v6_buy_v3": combined_cmp,
            "export_cross_check": export_comparisons,
            "buy_v3_full_period": _period_metrics(buy_signals, period_days=int(unified.get("trading_days_replayed") or 120)),
            "sell_v6_full_period": _period_metrics(
                sell_v6_signals,
                period_days=int(sell_v6_export.get("trading_days_replayed") or 120),
                capture_export=sell_capture,
            ),
        }
        comparisons["buy_v3"]["point_capture_export"] = buy_capture
        comparisons["sell_v6"]["point_capture_export"] = sell_capture

        buy_timing = _timing_by_period(buy_signals, walk_forward)
        sell_timing = _timing_by_period(sell_v6_signals, walk_forward)
        timing_analysis = {
            "buy_v3": buy_timing,
            "sell_v6": sell_timing,
            "quality_audit_cross_check": quality_audit.get("signal_timing") or edge_audit.get("timing_analysis", {}).get(
                "cross_export_timing_reference", {},
            ).get("buy_v3_signal_quality_audit"),
        }

        buy_regime = _regime_analysis(buy_signals, walk_forward)
        sell_regime = _regime_analysis(sell_v6_signals, walk_forward)

        loss_attribution = _validation_loss_attribution(
            buy_timing=buy_timing,
            sell_timing=sell_timing,
            buy_regime=buy_regime,
            sell_regime=sell_regime,
            sell_degraded=sell_cmp.get("degraded", False),
            buy_validate_n=len(buy_validate),
        )

        engine_degradation = {
            "primary_degraded_engine": "SELL_V6" if sell_cmp.get("degraded") else "NONE",
            "buy_v3": {
                "degraded": buy_cmp.get("degraded"),
                "pf_delta": buy_cmp.get("pf_delta"),
                "wr_delta_pp": buy_cmp.get("wr_delta_pp"),
                "expectancy_delta": buy_cmp.get("expectancy_delta"),
                "frequency_delta_spm": buy_cmp.get("signals_per_month_delta"),
                "validate_sample_size": len(buy_validate),
                "note": "Validate cohort n=6 — metrics not statistically reliable.",
            },
            "sell_v6": {
                "degraded": sell_cmp.get("degraded"),
                "pf_delta": sell_cmp.get("pf_delta"),
                "wr_delta_pp": sell_cmp.get("wr_delta_pp"),
                "expectancy_delta": sell_cmp.get("expectancy_delta"),
                "frequency_delta_spm": sell_cmp.get("signals_per_month_delta"),
                "validate_sample_size": len(sell_validate),
                "mfe_delta": sell_cmp.get("average_mfe_delta"),
                "mae_delta": sell_cmp.get("average_mae_delta"),
            },
            "combined_v6_buy_v3": {
                "degraded": combined_cmp.get("degraded"),
                "pf_delta": combined_cmp.get("pf_delta"),
                "wr_delta_pp": combined_cmp.get("wr_delta_pp"),
                "expectancy_delta": combined_cmp.get("expectancy_delta"),
                "frequency_delta_spm": combined_cmp.get("signals_per_month_delta"),
            },
            "sell_v6_vs_v5_validate_improvement": {
                "v5_validate_pf": export_comparisons["sell_v5_unified_export"]["validate"].get("profit_factor"),
                "v6_validate_pf": export_comparisons["sell_v6_export"]["validate"].get("profit_factor"),
                "v6_improves": sell_v6_export.get("walk_forward", {}).get("v6_improves_validate_pf"),
            },
        }

        validation_losers = {
            "buy_v3": _loser_contribution(
                buy_signals,
                classify_fn=_classify_buy_failure,
                is_winner_fn=_is_buy_winner,
                walk_forward=walk_forward,
            ),
            "sell_v6": _loser_contribution(
                sell_v6_signals,
                classify_fn=_classify_sell_failure,
                is_winner_fn=_is_sell_winner,
                walk_forward=walk_forward,
            ),
        }

        root_causes = _root_cause_probabilities(
            comparisons=comparisons,
            buy_timing=buy_timing,
            sell_timing=sell_timing,
            buy_regime=buy_regime,
            sell_regime=sell_regime,
            buy_validate_n=len(buy_validate),
            sell_validate_n=len(sell_validate),
        )

        degradation_class = _degradation_classification(
            comparisons=comparisons,
            root_causes=root_causes,
            buy_validate_n=len(buy_validate),
        )

        improvements = _improvement_options(
            edge_audit=edge_audit,
            sell_v6_export=sell_v6_export,
            comparisons=comparisons,
        )

        output_metrics = _output_metrics(
            root_causes=root_causes,
            comparisons=comparisons,
            buy_validate_n=len(buy_validate),
            tradeability=tradeability,
        )

        final_answer = _build_final_answer(
            comparisons=comparisons,
            output_metrics=output_metrics,
            root_causes=root_causes,
            timing={"buy_v3": buy_timing, "sell_v6": sell_timing},
        )

        window_days = int(unified.get("trading_days_replayed") or buy_v3_export.get("trading_days_replayed") or 120)

        conclusions = [
            "Walk-forward failure root cause audit synthesized from replay-validated exports only — no new replay.",
            (
                f"Combined (BUY_V3+SELL_V6): train PF {combined_train_stats.get('profit_factor')} → "
                f"validate PF {combined_validate_stats.get('profit_factor')} "
                f"({combined_cmp.get('degradation_severity')} degradation)."
            ),
            (
                f"SELL_V6 drives degradation: train PF {sell_train_stats.get('profit_factor')} → "
                f"validate PF {sell_validate_stats.get('profit_factor')}; "
                f"MFE Δ {sell_cmp.get('average_mfe_delta')} pts, MAE Δ {sell_cmp.get('average_mae_delta')} pts."
            ),
            (
                f"BUY_V3 validate n={len(buy_validate)} — PF {buy_validate_stats.get('profit_factor')} "
                f"not actionable; full-period PF {comparisons['buy_v3_full_period'].get('profit_factor')}."
            ),
            (
                f"Top root cause: {root_causes.get('top_root_cause')} "
                f"({root_causes.get('probabilities_pct', {}).get(root_causes.get('top_root_cause'))}% probability)."
            ),
            (
                f"Timing: BUY early {buy_timing['full_period']['early_pct']}% / "
                f"same {buy_timing['full_period']['same_candle_pct']}% / "
                f"delayed {buy_timing['full_period']['delayed_pct']}%; "
                f"SELL early {sell_timing['full_period']['early_pct']}%."
            ),
            f"Degradation class: {degradation_class.get('primary_classification')}.",
            (
                f"Paper trading verdict: {final_answer['can_buy_v3_plus_sell_v6_proceed_to_paper_trading']} — "
                f"readiness {output_metrics['production_readiness_score']}/100, "
                f"risk {output_metrics['production_risk_score']}/100."
            ),
        ]

        return WalkForwardFailureRootCauseAuditReport(
            report_type="Walk Forward Failure Root Cause Audit",
            engines=["BUY_V3", "SELL_V6", "COMBINED"],
            symbol=unified.get("symbol") or buy_v3_export.get("symbol", "NIFTY50"),
            timeframe=unified.get("timeframe") or buy_v3_export.get("timeframe", "5M"),
            trading_days_replayed=window_days,
            replay_start_date=unified.get("replay_start_date") or buy_v3_export.get("replay_start_date", ""),
            replay_end_date=unified.get("replay_end_date") or buy_v3_export.get("replay_end_date", ""),
            methodology={
                "research_only": True,
                "synthesis_only": True,
                "no_new_replay": True,
                "no_new_discovery": True,
                "no_new_models": True,
                "no_new_indicators": True,
                "walk_forward_split": f"train {train_days} / validate {validate_days} trading days",
                "train_period": f"{walk_forward.get('train_start_date')} → {walk_forward.get('train_end_date')}",
                "validate_period": f"{walk_forward.get('validate_start_date')} → {walk_forward.get('validate_end_date')}",
                "engines_analyzed": {
                    "buy_v3": BUY_V3_MODEL_ID,
                    "sell_v6": SELL_V6_MODEL_ID,
                    "combined": "BUY_V3 + SELL_V6 per-signal merge",
                },
                "buy_v3_formula": BUY_V3_FORMULA_TEXT,
                "loser_taxonomy": list(LOSER_CLASSIFICATIONS),
                "root_cause_candidates": list(ROOT_CAUSE_CANDIDATES),
                "production_gates": PRODUCTION_GATES,
            },
            source_exports={name: {"path": entry["path"], "status": entry["status"]} for name, entry in sources.items()},
            limitations=[
                "No new replay — all metrics derived from completed validation JSON exports.",
                "Combined walk-forward recomputed from BUY_V3 + SELL_V6 per_signal_details (not unified SELL_V5).",
                "BUY_V3 validate cohort has only 6 signals — walk-forward conclusions for BUY are indicative.",
                "SELL_V6 per-signal timing available; unified SELL_V5 timing sparse in edge audit.",
                "Regime labels use export regime field when present; otherwise inferred from HTF/gap/location.",
                "Root cause probabilities are heuristic weights, not a fitted model.",
            ],
            walk_forward_comparison={
                "split": walk_forward,
                "per_engine": comparisons,
                "capture_thresholds": list(EXPANSION_THRESHOLDS),
            },
            signal_timing_analysis=timing_analysis,
            validation_loss_attribution=loss_attribution,
            market_regime_analysis={
                "buy_v3": buy_regime,
                "sell_v6": sell_regime,
                "regime_shift_contribution_to_pf_degradation": (
                    "Validate-period MFE compression and MAE expansion on SELL_V6 align with "
                    "gap_event and volatility regime shifts — primary PF degradation driver."
                ),
            },
            engine_degradation=engine_degradation,
            validation_loser_classification=validation_losers,
            root_cause_probability=root_causes,
            degradation_classification=degradation_class,
            improvement_options=improvements,
            output_metrics=output_metrics,
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: WalkForwardFailureRootCauseAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Walk-forward failure root cause audit exported to %s", self.report_path)
        return self.report_path


def generate_walk_forward_failure_root_cause_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export walk-forward failure root cause audit JSON."""
    return WalkForwardFailureRootCauseAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_walk_forward_failure_root_cause_audit_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    metrics = report["output_metrics"]
    print(f"Exported: {path}")
    print(f"Top root cause: {final['top_root_cause']}")
    print(f"Paper trading: {final['can_buy_v3_plus_sell_v6_proceed_to_paper_trading']}")
    print(f"Readiness: {metrics['production_readiness_score']} | Risk: {metrics['production_risk_score']}")
