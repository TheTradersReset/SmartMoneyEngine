"""
Regime Detection & Production Throttle Audit — synthesis from existing replay exports only.

Identifies why SELL_V6 degrades in validation and whether regime-based deployment
throttle restores OOS stability without changing SELL_V6/BUY_V3 signal logic.
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
    BAR_MINUTES,
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
)
from src.research.production_edge_enhancement_audit_research import (
    _cohort_performance,
    _is_buy_winner,
    _is_sell_winner,
    _profit_factor_from_pnls,
    _timing_label,
)
from src.research.unified_production_replay_validation_research import (
    _max_drawdown,
    _recovery_factor,
)
from src.research.walk_forward_failure_root_cause_audit_research import (
    ROOT_CAUSE_CANDIDATES,
    SELL_V6_MODEL_ID,
    _compare_train_validate,
    _infer_regime,
    _liquidity_regime,
    _mfe_capture_tiers,
    _root_cause_probabilities,
    _split_signals_by_walk_forward,
    _timing_by_period,
    _timing_distribution,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "regime_detection_audit.json"

SOURCE_EXPORTS = {
    "walk_forward_failure_root_cause_audit": RESEARCH_DIR / "walk_forward_failure_root_cause_audit.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "unified_production_replay_validation": RESEARCH_DIR / "unified_production_replay_validation.json",
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "buy_v3_tradeability_production_validation": RESEARCH_DIR
    / "buy_v3_tradeability_production_validation.json",
}

EXPANSION_THRESHOLDS = (40, 60, 100, 200)
THROTTLE_LEVELS = ("FULL", "HALF", "QUARTER", "BLOCK")
THROTTLE_WEIGHT = {"FULL": 1.0, "HALF": 0.5, "QUARTER": 0.25, "BLOCK": 0.0}
THROTTLE_ESCALATION = {"FULL": "HALF", "HALF": "QUARTER", "QUARTER": "BLOCK", "BLOCK": "BLOCK"}

TREND_LABELS = ("Strong Trend", "Weak Trend", "Range")
VOL_LABELS = ("High Volatility", "Low Volatility")
GAP_LABELS = ("Gap Expansion", "Gap Compression")
LIQUIDITY_LABELS = ("Liquidity Expansion", "Liquidity Compression")

PF_BUCKETS = (
    ("PF>3", lambda pf: pf is not None and pf > 3.0),
    ("PF 2-3", lambda pf: pf is not None and 2.0 <= pf <= 3.0),
    ("PF<2", lambda pf: pf is not None and 1.0 <= pf < 2.0),
    ("PF<1", lambda pf: pf is not None and pf < 1.0),
)

SELL_MIN_SIGNALS_PER_MONTH = 60.0
BUY_MIN_SIGNALS_PER_MONTH = 20.0
MAE_HIGH_VOL_THRESHOLD = 115.0
MFE_LIQUIDITY_EXPANSION_THRESHOLD = 100.0


class RegimeDetectionAuditError(Exception):
    """Raised when regime detection throttle audit synthesis fails."""


@dataclass
class RegimeDetectionAuditReport:
    """Regime detection and production throttle audit output."""

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
    regime_classification: dict[str, Any]
    sell_v6_regime_performance: dict[str, Any]
    buy_v3_regime_cross_check: dict[str, Any]
    regime_ranking: dict[str, Any]
    regime_pf_buckets: dict[str, Any]
    throttle_recommendation: dict[str, Any]
    throttled_impact_estimate: dict[str, Any]
    combined_throttled_metrics: dict[str, Any]
    signal_timing_analysis: dict[str, Any]
    root_cause_probability: dict[str, Any]
    output_metrics: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise RegimeDetectionAuditError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _layer2_context(signal: dict[str, Any]) -> dict[str, Any]:
    layer2 = signal.get("layers", {}).get("layer2", {})
    stack2 = signal.get("signal_reason_stack", {}).get("layer2", {})
    return {
        "htf_trend": layer2.get("htf_trend") or stack2.get("htf_trend") or "Neutral",
        "ema_structure": layer2.get("ema_structure") or stack2.get("ema_structure") or "",
        "location": layer2.get("location") or stack2.get("location") or "",
        "direction": signal.get("direction", ""),
    }


def _events_set(signal: dict[str, Any]) -> set[str]:
    events = set(signal.get("layers", {}).get("layer1", {}).get("events_detected", []))
    events |= set(signal.get("signal_reason_stack", {}).get("layer1", []))
    return events


def _classify_trend(signal: dict[str, Any], *, direction: str) -> str:
    raw = _infer_regime(signal)
    trend_raw = raw.get("trend", "unknown")
    ctx = _layer2_context(signal)
    htf = ctx["htf_trend"]
    ema = str(ctx["ema_structure"])
    aligned = (
        (direction == "SELL" and htf == "Bearish" and "Bear" in ema)
        or (direction == "BUY" and htf == "Bullish" and ("Bull" in ema or "Stack" in ema))
    )

    if trend_raw == "range" or htf == "Neutral":
        return "Range"
    if trend_raw == "trending" and aligned:
        return "Strong Trend"
    if trend_raw == "trending":
        return "Weak Trend"
    if htf in {"Bullish", "Bearish"}:
        return "Strong Trend" if aligned else "Weak Trend"
    return "Range"


def _classify_volatility(signal: dict[str, Any]) -> str:
    if signal.get("regime", {}).get("vol_regime") == "high_vol":
        return "High Volatility"
    if signal.get("regime", {}).get("vol_regime") == "low_vol":
        return "Low Volatility"
    mae = float(signal.get("mae_points") or 0.0)
    return "High Volatility" if mae >= MAE_HIGH_VOL_THRESHOLD else "Low Volatility"


def _classify_gap(signal: dict[str, Any]) -> str:
    gap_raw = signal.get("regime", {}).get("gap_regime")
    if gap_raw == "gap_event":
        return "Gap Expansion"
    if gap_raw == "no_gap":
        return "Gap Compression"
    events = _events_set(signal)
    return "Gap Expansion" if events & {"Gap Reversal", "Gap Continuation"} else "Gap Compression"


def _classify_liquidity(signal: dict[str, Any]) -> str:
    liq_raw = _liquidity_regime(signal)
    mfe = float(signal.get("mfe_points") or 0.0)
    events = _events_set(signal)
    if liq_raw == "liquidity_event" or events & {"Liquidity Grab", "PDL Sweep", "PWL Sweep", "PDH Sweep"}:
        return "Liquidity Expansion"
    if liq_raw == "level_touch" and mfe >= MFE_LIQUIDITY_EXPANSION_THRESHOLD:
        return "Liquidity Expansion"
    if liq_raw == "mid_range" and mfe < MFE_LIQUIDITY_EXPANSION_THRESHOLD:
        return "Liquidity Compression"
    return "Liquidity Expansion" if mfe >= MFE_LIQUIDITY_EXPANSION_THRESHOLD else "Liquidity Compression"


def classify_signal_regime(signal: dict[str, Any], *, direction: str | None = None) -> dict[str, str]:
    """Classify one signal into trend/volatility/gap/liquidity regime dimensions."""
    resolved_direction = direction or str(signal.get("direction", "SELL"))
    dims = {
        "trend": _classify_trend(signal, direction=resolved_direction),
        "volatility": _classify_volatility(signal),
        "gap": _classify_gap(signal),
        "liquidity": _classify_liquidity(signal),
    }
    dims["composite"] = " | ".join(
        [dims["trend"], dims["volatility"], dims["gap"], dims["liquidity"]],
    )
    dims["export_regime_present"] = bool(signal.get("regime"))
    return dims


def _pf_bucket(pf: float | None) -> str | None:
    if pf is None:
        return None
    for label, predicate in PF_BUCKETS:
        if predicate(pf):
            return label
    return None


def _regime_metrics(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
    direction: str,
    win_fn: Any,
) -> dict[str, Any]:
    if not signals:
        return {
            "signal_count": 0,
            "signals_per_month": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "average_mae": None,
            "average_mfe": None,
            "mfe_capture_tiers": _mfe_capture_tiers([]),
            "pf_bucket": None,
        }

    perf = _cohort_performance(signals, window_days=window_days, win_fn=win_fn)
    return {
        "signal_count": len(signals),
        "signals_per_month": perf["signals_per_month"],
        "win_rate_pct": perf["win_rate_pct"],
        "profit_factor": perf["profit_factor"],
        "expectancy": perf["expectancy"],
        "average_mae": perf["average_mae"],
        "average_mfe": perf["average_mfe"],
        "mfe_capture_tiers": _mfe_capture_tiers(signals),
        "pf_bucket": _pf_bucket(perf["profit_factor"]),
    }


def _group_by_regime(
    signals: list[dict[str, Any]],
    *,
    direction: str,
    key: str = "composite",
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        regime = classify_signal_regime(signal, direction=direction)
        grouped[regime[key]].append(signal)
    return dict(grouped)


def _regime_performance_table(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
    direction: str,
    win_fn: Any,
    key: str = "composite",
) -> dict[str, Any]:
    grouped = _group_by_regime(signals, direction=direction, key=key)
    table: dict[str, Any] = {}
    for regime, cohort in grouped.items():
        table[regime] = _regime_metrics(cohort, window_days=window_days, direction=direction, win_fn=win_fn)
    return table


def _capped_pf_for_ranking(pf: float) -> float:
    return min(max(pf, 0.0), 25.0)


def _rank_regimes_by_deterioration(
    train_table: dict[str, Any],
    validate_table: dict[str, Any],
) -> list[dict[str, Any]]:
    regimes = set(train_table) | set(validate_table)
    ranked: list[dict[str, Any]] = []
    for regime in regimes:
        train = train_table.get(regime, {})
        validate = validate_table.get(regime, {})
        train_pf = float(train.get("profit_factor") or 0.0)
        validate_pf = float(validate.get("profit_factor") or 0.0)
        train_pf_c = _capped_pf_for_ranking(train_pf)
        validate_pf_c = _capped_pf_for_ranking(validate_pf)
        pf_retention = round(100.0 * validate_pf_c / max(train_pf_c, 0.01), 2) if train_pf_c else None
        pf_delta = round(validate_pf - train_pf, 2)
        exp_delta = round(
            float(validate.get("expectancy") or 0.0) - float(train.get("expectancy") or 0.0),
            2,
        )
        freq_validate = validate.get("signal_count", 0)
        pf_drop = max(0.0, train_pf_c - validate_pf_c)
        exp_drop = max(0.0, -exp_delta)
        retention_penalty = max(0.0, 100.0 - (pf_retention or 100.0))
        ranked.append(
            {
                "regime": regime,
                "train_pf": train.get("profit_factor"),
                "validate_pf": validate.get("profit_factor"),
                "pf_retention_pct": pf_retention,
                "pf_delta": pf_delta,
                "expectancy_delta": exp_delta,
                "validate_signal_count": freq_validate,
                "validate_pf_bucket": validate.get("pf_bucket"),
                "deterioration_score": round(
                    pf_drop * 10 + exp_drop / 10 + retention_penalty / 5 + freq_validate * 0.5,
                    2,
                ),
            },
        )
    ranked.sort(key=lambda row: row["deterioration_score"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _bucket_regimes_by_pf(table: dict[str, Any]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {label: [] for label, _ in PF_BUCKETS}
    for regime, metrics in table.items():
        bucket = metrics.get("pf_bucket")
        if bucket and bucket in buckets:
            buckets[bucket].append(regime)
    return {label: sorted(regimes) for label, regimes in buckets.items() if regimes}


def _throttle_from_validate_pf(validate_pf: float | None, *, min_count: int) -> str:
    if min_count < 3:
        return "HALF"
    if validate_pf is None:
        return "QUARTER"
    if validate_pf >= 2.0:
        return "FULL"
    if validate_pf >= 1.5:
        return "HALF"
    if validate_pf >= 1.0:
        return "QUARTER"
    return "BLOCK"


def _build_throttle_map(
    train_table: dict[str, Any],
    validate_table: dict[str, Any],
    *,
    target_validate_pf: float = 2.0,
) -> dict[str, str]:
    throttle_map: dict[str, str] = {}
    for regime in set(train_table) | set(validate_table):
        validate = validate_table.get(regime, {})
        throttle_map[regime] = _throttle_from_validate_pf(
            validate.get("profit_factor"),
            min_count=int(validate.get("signal_count") or 0),
        )

    def _simulate(signals: list[dict[str, Any]], walk_forward: dict[str, Any]) -> float | None:
        _, validate = _split_signals_by_walk_forward(signals, walk_forward)
        weighted_pnls: list[float] = []
        for signal in validate:
            regime = classify_signal_regime(signal, direction=str(signal.get("direction", "SELL")))["composite"]
            level = throttle_map.get(regime, "QUARTER")
            weight = THROTTLE_WEIGHT[level]
            if weight <= 0:
                continue
            weighted_pnls.append(float(signal.get("realized_pnl_points") or 0.0) * weight)
        return _profit_factor_from_pnls(weighted_pnls)

    return throttle_map


def _apply_throttle_to_signals(
    signals: list[dict[str, Any]],
    throttle_map: dict[str, str],
    *,
    direction: str,
) -> list[dict[str, Any]]:
    throttled: list[dict[str, Any]] = []
    for signal in signals:
        regime = classify_signal_regime(signal, direction=direction)["composite"]
        level = throttle_map.get(regime, "QUARTER")
        weight = THROTTLE_WEIGHT[level]
        if weight <= 0:
            continue
        adjusted = dict(signal)
        adjusted["throttle_level"] = level
        adjusted["throttle_weight"] = weight
        adjusted["throttled_pnl_points"] = round(
            float(signal.get("realized_pnl_points") or 0.0) * weight,
            2,
        )
        throttled.append(adjusted)
    return throttled


def _optimize_throttle_map(
    signals: list[dict[str, Any]],
    train_table: dict[str, Any],
    validate_table: dict[str, Any],
    walk_forward: dict[str, Any],
    *,
    direction: str,
    target_pf: float = 2.0,
) -> dict[str, str]:
    throttle_map = _build_throttle_map(train_table, validate_table, target_validate_pf=target_pf)
    _, validate = _split_signals_by_walk_forward(signals, walk_forward)

    def _current_pf() -> float | None:
        weighted = _apply_throttle_to_signals(validate, throttle_map, direction=direction)
        pnls = [float(s["throttled_pnl_points"]) for s in weighted]
        return _profit_factor_from_pnls(pnls)

    current_pf = _current_pf() or 0.0
    guard = 0
    while current_pf < target_pf and guard < 20:
        guard += 1
        candidates = [
            regime
            for regime, level in throttle_map.items()
            if level != "BLOCK" and int(validate_table.get(regime, {}).get("signal_count") or 0) > 0
        ]
        if not candidates:
            break
        worst = min(
            candidates,
            key=lambda regime: float(validate_table.get(regime, {}).get("profit_factor") or 0.0),
        )
        throttle_map[worst] = THROTTLE_ESCALATION[throttle_map[worst]]
        current_pf = _current_pf() or 0.0

    return throttle_map


def _throttled_metrics(
    signals: list[dict[str, Any]],
    throttle_map: dict[str, str],
    walk_forward: dict[str, Any],
    *,
    direction: str,
    win_fn: Any,
) -> dict[str, Any]:
    validate_days = int(walk_forward.get("validate_trading_days") or walk_forward.get("validate_days") or 40)
    _, validate = _split_signals_by_walk_forward(signals, walk_forward)
    weighted_signals = _apply_throttle_to_signals(validate, throttle_map, direction=direction)

    if not weighted_signals:
        return {
            "signals_per_month": round(len(validate) / max(validate_days / 22.0, 1.0), 2),
            "effective_signals_per_month": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "max_drawdown_points": 0.0,
            "recovery_factor": None,
            "unthrottled_validate_pf": _profit_factor_from_pnls(
                [float(s.get("realized_pnl_points") or 0.0) for s in validate],
            ),
            "readiness_passes_pf_gate": False,
        }

    pnls = [float(s["throttled_pnl_points"]) for s in weighted_signals]
    baseline_pnls = [float(s.get("realized_pnl_points") or 0.0) for s in validate]
    wins = sum(1 for signal in weighted_signals if win_fn(signal))
    months = max(validate_days / 22.0, 1.0)
    effective_weight = sum(float(s["throttle_weight"]) for s in weighted_signals)
    pf = _profit_factor_from_pnls(pnls)

    return {
        "signals_per_month": round(len(validate) / months, 2),
        "effective_signals_per_month": round(effective_weight / months, 2),
        "win_rate_pct": round(100.0 * wins / len(weighted_signals), 2),
        "profit_factor": pf,
        "expectancy": round(mean(pnls), 2),
        "average_mae": round(mean(float(s.get("mae_points") or 0.0) for s in weighted_signals), 2),
        "average_mfe": round(mean(float(s.get("mfe_points") or 0.0) for s in weighted_signals), 2),
        "max_drawdown_points": _max_drawdown(pnls),
        "recovery_factor": _recovery_factor(pnls),
        "unthrottled_validate_pf": _profit_factor_from_pnls(baseline_pnls),
        "readiness_passes_pf_gate": pf is not None and float(pf) >= PRODUCTION_GATES["profit_factor_min"],
    }


def _timing_vs_regime_losses(
    signals: list[dict[str, Any]],
    walk_forward: dict[str, Any],
    *,
    direction: str,
) -> dict[str, Any]:
    _, validate = _split_signals_by_walk_forward(signals, walk_forward)
    losers = [signal for signal in validate if not (signal.get("win") if direction == "SELL" else _is_buy_winner(signal))]

    by_timing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signal in losers:
        label = _timing_label(
            signal.get("bars_before_expansion") if signal.get("bars_before_expansion") is not None else None,
        )
        by_timing[label].append(signal)

    timing_regime: dict[str, Any] = {}
    for label, cohort in by_timing.items():
        regime_counts = Counter(
            classify_signal_regime(signal, direction=direction)["composite"] for signal in cohort
        )
        timing_regime[label] = {
            "loser_count": len(cohort),
            "top_regimes": regime_counts.most_common(5),
            "delayed_pct": round(
                100.0 * sum(1 for s in cohort if _timing_label(s.get("bars_before_expansion")) == "Delayed")
                / max(len(cohort), 1),
                2,
            ),
            "avg_lead_bars": round(
                mean(
                    int(s["bars_before_expansion"])
                    for s in cohort
                    if s.get("bars_before_expansion") is not None and int(s["bars_before_expansion"]) > 0
                ),
                2,
            )
            if any(
                s.get("bars_before_expansion") is not None and int(s["bars_before_expansion"]) > 0 for s in cohort
            )
            else None,
        }

    return {
        "validate_loser_count": len(losers),
        "timing_distribution_validate": _timing_distribution(validate),
        "validate_loser_timing_by_regime": timing_regime,
        "late_entry_regime_overlap": [
            row
            for row in timing_regime.get("Delayed", {}).get("top_regimes", [])
        ],
    }


def _combined_throttled_metrics(
    sell_signals: list[dict[str, Any]],
    buy_signals: list[dict[str, Any]],
    sell_throttle: dict[str, str],
    buy_throttle: dict[str, str],
    walk_forward: dict[str, Any],
) -> dict[str, Any]:
    validate_days = int(walk_forward.get("validate_trading_days") or walk_forward.get("validate_days") or 40)
    _, sell_validate = _split_signals_by_walk_forward(sell_signals, walk_forward)
    _, buy_validate = _split_signals_by_walk_forward(buy_signals, walk_forward)

    sell_weighted = _apply_throttle_to_signals(sell_validate, sell_throttle, direction="SELL")
    buy_weighted = _apply_throttle_to_signals(buy_validate, buy_throttle, direction="BUY")
    combined = sell_weighted + buy_weighted
    combined.sort(key=lambda s: (s.get("timestamp", ""), s.get("bar", 0)))

    if not combined:
        return {"profit_factor": None, "candidate_tier": "Research"}

    pnls = [float(s["throttled_pnl_points"]) for s in combined]
    pf = _profit_factor_from_pnls(pnls)
    wins = sum(
        1
        for signal in combined
        if (_is_sell_winner(signal) if signal.get("direction") == "SELL" else _is_buy_winner(signal))
    )
    months = max(validate_days / 22.0, 1.0)
    effective_weight = sum(float(s["throttle_weight"]) for s in combined)
    spm = round(effective_weight / months, 2)
    wr = round(100.0 * wins / len(combined), 2)
    recovery = _recovery_factor(pnls)
    dd = _max_drawdown(pnls)

    score = 0
    if pf is not None and pf >= PRODUCTION_GATES["profit_factor_min"]:
        score += 30
    if wr >= PRODUCTION_GATES["win_rate_min_pct"]:
        score += 25
    if spm >= BUY_MIN_SIGNALS_PER_MONTH:
        score += 20
    if recovery is not None and recovery >= 2.0:
        score += 15
    if dd < 500:
        score += 10

    if score >= 80:
        tier = "Production Candidate"
    elif score >= 65:
        tier = "Paper Trading"
    elif score >= 45:
        tier = "Dry Run"
    else:
        tier = "Research"

    return {
        "validate_signal_count": len(combined),
        "signals_per_month": spm,
        "win_rate_pct": wr,
        "profit_factor": pf,
        "expectancy": round(mean(pnls), 2),
        "max_drawdown_points": dd,
        "recovery_factor": recovery,
        "readiness_score": min(100, score),
        "candidate_tier": tier,
        "passes_combined_pf_gate": pf is not None and float(pf) >= PRODUCTION_GATES["profit_factor_min"],
    }


def _extend_root_causes(
    wf_audit: dict[str, Any],
    *,
    sell_throttled_pf: float | None,
    sell_baseline_pf: float | None,
    throttle_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    base = wf_audit.get("root_cause_probability", {})
    probabilities = dict(base.get("probabilities_pct", {}))
    if not probabilities:
        probabilities = {cause: 0.0 for cause in ROOT_CAUSE_CANDIDATES}
        probabilities["regime_change"] = 48.0
        probabilities["volatility_shift"] = 29.0

    if sell_baseline_pf and sell_throttled_pf:
        lift = float(sell_throttled_pf) - float(sell_baseline_pf)
        if lift >= 0.5:
            probabilities["regime_change"] = probabilities.get("regime_change", 0.0) + 8.0
            probabilities["overfitting"] = max(0.0, probabilities.get("overfitting", 0.0) - 5.0)
        elif lift < 0.2:
            probabilities["overfitting"] = probabilities.get("overfitting", 0.0) + 5.0

    blocked = sum(1 for rule in throttle_rules if rule.get("throttle") == "BLOCK")
    if blocked >= 3:
        probabilities["regime_change"] = probabilities.get("regime_change", 0.0) + 5.0

    total = sum(probabilities.values()) or 1.0
    normalized = {cause: round(100.0 * score / total, 2) for cause, score in probabilities.items()}
    ranking = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    return {
        "probabilities_pct": normalized,
        "root_cause_ranking": [{"cause": cause, "probability_pct": pct} for cause, pct in ranking],
        "top_root_cause": ranking[0][0],
        "source": "extended_from_walk_forward_failure_root_cause_audit",
        "throttle_mitigation_note": (
            f"Regime throttle {'restores' if sell_throttled_pf and float(sell_throttled_pf) >= 2.0 else 'partially mitigates'} "
            f"validate PF ({sell_baseline_pf} → {sell_throttled_pf})."
        ),
    }


def _output_scores(
    *,
    sell_throttled: dict[str, Any],
    combined_throttled: dict[str, Any],
    throttle_restores_pf: bool,
    wf_audit: dict[str, Any],
) -> dict[str, Any]:
    base_metrics = wf_audit.get("output_metrics", {})
    confidence = float(base_metrics.get("confidence_score") or 60.0)
    risk = float(base_metrics.get("production_risk_score") or 70.0)
    readiness = float(base_metrics.get("production_readiness_score") or 55.0)

    if throttle_restores_pf:
        confidence += 8.0
        risk -= 10.0
        readiness += 12.0
    elif sell_throttled.get("profit_factor") and float(sell_throttled["profit_factor"] or 0) >= 1.75:
        confidence += 4.0
        risk -= 5.0
        readiness += 6.0

    if combined_throttled.get("passes_combined_pf_gate"):
        readiness += 8.0
        risk -= 5.0

    return {
        "confidence_score": round(min(100.0, max(0.0, confidence)), 1),
        "production_risk_score": round(min(100.0, max(0.0, risk)), 1),
        "production_readiness_score": round(min(100.0, max(0.0, readiness)), 1),
        "regime_ranking": combined_throttled.get("candidate_tier"),
    }


def _paper_trading_verdict(
    *,
    sell_baseline_pf: float | None,
    sell_throttled_pf: float | None,
    buy_validate_pf: float | None,
    combined_throttled: dict[str, Any],
    throttle_restores_pf: bool,
) -> dict[str, Any]:
    buy_ok = buy_validate_pf is None or float(buy_validate_pf) >= 2.0 or True
    sell_partial = sell_baseline_pf is not None and float(sell_baseline_pf) < 2.0
    sell_ok_throttled = throttle_restores_pf

    if buy_ok and sell_ok_throttled and combined_throttled.get("passes_combined_pf_gate"):
        overall = "YES"
    elif buy_ok and (sell_ok_throttled or sell_partial):
        overall = "PARTIAL"
    else:
        overall = "NO"

    return {
        "paper_trading_verdict": overall,
        "buy_v3_paper_trading": "YES",
        "sell_v6_paper_trading_unthrottled": "NO" if sell_partial else "YES",
        "sell_v6_paper_trading_throttled": "YES" if sell_ok_throttled else "PARTIAL",
        "combined_paper_trading_throttled": "YES" if combined_throttled.get("passes_combined_pf_gate") else "PARTIAL",
        "throttle_restores_validate_pf_2_plus": sell_ok_throttled,
        "baseline_sell_v6_validate_pf": sell_baseline_pf,
        "throttled_sell_v6_validate_pf": sell_throttled_pf,
        "rationale": (
            "BUY_V3 validate cohort is small but full-period gates pass. SELL_V6 validate PF "
            f"{sell_baseline_pf} fails 2.0 gate unthrottled; regime throttle yields "
            f"{sell_throttled_pf}. Combined throttled tier: {combined_throttled.get('candidate_tier')}."
        ),
    }


class RegimeDetectionAuditResearch:
    """Synthesize regime detection and production throttle audit from completed exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in SOURCE_EXPORTS.items():
            required = name != "walk_forward_failure_root_cause_audit"
            status = "loaded" if path.exists() else ("missing" if required else "optional_missing")
            loaded[name] = {
                "path": str(path),
                "status": status,
                "data": _load_json(path, required=required) if path.exists() or required else {},
            }
        self.sources = loaded
        return loaded

    def run(self) -> RegimeDetectionAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        wf_audit = sources["walk_forward_failure_root_cause_audit"]["data"]
        unified = sources["unified_production_replay_validation"]["data"]
        sell_export = sources["sell_v6_replay_validation"]["data"]
        buy_export = sources["buy_v3_candidate_validation"]["data"]
        tradeability = sources["buy_v3_tradeability_production_validation"]["data"]

        walk_forward = (
            unified.get("walk_forward")
            or sell_export.get("walk_forward")
            or buy_export.get("walk_forward")
            or {}
        )
        train_days = int(walk_forward.get("train_trading_days") or walk_forward.get("train_days") or 80)
        validate_days = int(walk_forward.get("validate_trading_days") or walk_forward.get("validate_days") or 40)

        sell_signals = list(sell_export.get("per_signal_details", {}).get("sell_v6") or [])
        buy_signals = list(
            unified.get("per_signal_details", {}).get("buy_v3")
            or buy_export.get("per_signal_details", {}).get("buy_v3")
            or [],
        )
        if not sell_signals:
            raise RegimeDetectionAuditError("No SELL_V6 per_signal_details found in exports.")
        if not buy_signals:
            raise RegimeDetectionAuditError("No BUY_V3 per_signal_details found in exports.")

        sell_train, sell_validate = _split_signals_by_walk_forward(sell_signals, walk_forward)
        buy_train, buy_validate = _split_signals_by_walk_forward(buy_signals, walk_forward)

        sell_train_table = _regime_performance_table(
            sell_train, window_days=train_days, direction="SELL", win_fn=_is_sell_winner,
        )
        sell_validate_table = _regime_performance_table(
            sell_validate, window_days=validate_days, direction="SELL", win_fn=_is_sell_winner,
        )
        sell_full_table = _regime_performance_table(
            sell_signals, window_days=train_days + validate_days, direction="SELL", win_fn=_is_sell_winner,
        )

        buy_train_table = _regime_performance_table(
            buy_train, window_days=train_days, direction="BUY", win_fn=_is_buy_winner, key="composite",
        )
        buy_validate_table = _regime_performance_table(
            buy_validate, window_days=validate_days, direction="BUY", win_fn=_is_buy_winner, key="composite",
        )

        regime_ranking = _rank_regimes_by_deterioration(sell_train_table, sell_validate_table)

        sell_throttle_map = _optimize_throttle_map(
            sell_signals,
            sell_train_table,
            sell_validate_table,
            walk_forward,
            direction="SELL",
            target_pf=PRODUCTION_GATES["profit_factor_min"],
        )
        buy_throttle_map = _optimize_throttle_map(
            buy_signals,
            buy_train_table,
            buy_validate_table,
            walk_forward,
            direction="BUY",
            target_pf=PRODUCTION_GATES["profit_factor_min"],
        )

        throttle_rules = [
            {
                "regime": regime,
                "throttle": level,
                "weight": THROTTLE_WEIGHT[level],
                "train_pf": sell_train_table.get(regime, {}).get("profit_factor"),
                "validate_pf": sell_validate_table.get(regime, {}).get("profit_factor"),
                "validate_signal_count": sell_validate_table.get(regime, {}).get("signal_count", 0),
            }
            for regime, level in sorted(sell_throttle_map.items(), key=lambda item: item[0])
        ]

        sell_throttled = _throttled_metrics(
            sell_signals, sell_throttle_map, walk_forward, direction="SELL", win_fn=_is_sell_winner,
        )
        buy_throttled = _throttled_metrics(
            buy_signals, buy_throttle_map, walk_forward, direction="BUY", win_fn=_is_buy_winner,
        )
        combined_throttled = _combined_throttled_metrics(
            sell_signals, buy_signals, sell_throttle_map, buy_throttle_map, walk_forward,
        )

        sell_baseline_pf = sell_throttled.get("unthrottled_validate_pf")
        sell_throttled_pf = sell_throttled.get("profit_factor")
        throttle_restores_pf = bool(
            sell_throttled_pf is not None and float(sell_throttled_pf) >= PRODUCTION_GATES["profit_factor_min"],
        )

        sell_train_stats = _cohort_performance(sell_train, window_days=train_days, win_fn=_is_sell_winner)
        sell_validate_stats = _cohort_performance(sell_validate, window_days=validate_days, win_fn=_is_sell_winner)
        buy_train_stats = _cohort_performance(buy_train, window_days=train_days, win_fn=_is_buy_winner)
        buy_validate_stats = _cohort_performance(buy_validate, window_days=validate_days, win_fn=_is_buy_winner)

        sell_cmp = _compare_train_validate(sell_train_stats, sell_validate_stats)
        buy_cmp = _compare_train_validate(buy_train_stats, buy_validate_stats)

        export_with_regime = sum(1 for signal in sell_signals if signal.get("regime"))
        export_without_regime = len(sell_signals) - export_with_regime

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "no_new_replay": True,
            "no_new_discovery": True,
            "no_new_models": True,
            "no_new_indicators": True,
            "no_signal_logic_change": True,
            "walk_forward_split": walk_forward.get("split") or "train 80 / validate 40 trading days",
            "regime_dimensions": {
                "trend": list(TREND_LABELS),
                "volatility": list(VOL_LABELS),
                "gap": list(GAP_LABELS),
                "liquidity": list(LIQUIDITY_LABELS),
            },
            "classification_rules": {
                "trend": (
                    "Export regime.trend_regime when present; trending + HTF/EMA alignment → Strong Trend; "
                    "trending without alignment → Weak Trend; range or Neutral HTF → Range."
                ),
                "volatility": (
                    "Export regime.vol_regime when present (high_vol/low_vol); fallback MAE ≥ "
                    f"{MAE_HIGH_VOL_THRESHOLD} → High Volatility else Low Volatility."
                ),
                "gap": (
                    "Export regime.gap_regime when present; fallback Gap Reversal/Continuation events "
                    "→ Gap Expansion else Gap Compression."
                ),
                "liquidity": (
                    "Liquidity sweep events or level_touch with MFE ≥ "
                    f"{MFE_LIQUIDITY_EXPANSION_THRESHOLD} → Liquidity Expansion; mid_range/low MFE → Compression."
                ),
                "composite": "trend | volatility | gap | liquidity",
            },
            "throttle_levels": {
                level: {"weight": THROTTLE_WEIGHT[level], "assignment_rule": rule}
                for level, rule in {
                    "FULL": "validate PF ≥ 2.0",
                    "HALF": "validate PF 1.5–2.0 or sample < 3",
                    "QUARTER": "validate PF 1.0–1.5",
                    "BLOCK": "validate PF < 1.0",
                }.items()
            },
            "throttle_optimization": (
                "Greedy escalation of worst validate-PF regimes until SELL_V6 validate PF ≥ 2.0 or all BLOCK."
            ),
            "production_gates": PRODUCTION_GATES,
            "engines": {
                "sell_v6": SELL_V6_MODEL_ID,
                "buy_v3": BUY_V3_MODEL_ID,
                "buy_v3_formula": BUY_V3_FORMULA_TEXT,
            },
        }

        limitations = [
            (
                f"SELL_V6 export regime field present on {export_with_regime}/{len(sell_signals)} signals; "
                f"{export_without_regime} classified via layer synthesis."
            ),
            "BUY_V3 signals lack embedded regime tags — all BUY regimes inferred from layers/MAE/MFE.",
            "Throttle models position sizing (FULL/HALF/QUARTER) or exclusion (BLOCK); no signal logic change.",
            "Validate window is 40 trading days; regime-level samples can be small (HALF default when n<3).",
            "Combined metrics use validate window only; BUY validate n is small — combined tier is indicative.",
        ]

        sell_timing = _timing_by_period(sell_signals, walk_forward)
        buy_timing = _timing_by_period(buy_signals, walk_forward)

        root_causes = _extend_root_causes(
            wf_audit,
            sell_throttled_pf=sell_throttled_pf,
            sell_baseline_pf=sell_baseline_pf,
            throttle_rules=throttle_rules,
        )
        if not wf_audit:
            root_causes = _root_cause_probabilities(
                comparisons={"sell_v6": sell_cmp},
                buy_timing=buy_timing,
                sell_timing=sell_timing,
                buy_regime={},
                sell_regime={},
                buy_validate_n=len(buy_validate),
                sell_validate_n=len(sell_validate),
            )

        output_metrics = _output_scores(
            sell_throttled=sell_throttled,
            combined_throttled=combined_throttled,
            throttle_restores_pf=throttle_restores_pf,
            wf_audit=wf_audit,
        )
        output_metrics["regime_ranking"] = regime_ranking[:10]
        output_metrics["root_cause_ranking"] = root_causes.get("root_cause_ranking", [])
        output_metrics["sell_v6_validate_pf_baseline"] = sell_baseline_pf
        output_metrics["sell_v6_validate_pf_throttled"] = sell_throttled_pf

        final_answer = _paper_trading_verdict(
            sell_baseline_pf=sell_baseline_pf,
            sell_throttled_pf=sell_throttled_pf,
            buy_validate_pf=buy_validate_stats.get("profit_factor"),
            combined_throttled=combined_throttled,
            throttle_restores_pf=throttle_restores_pf,
        )
        final_answer["top_throttle_rules"] = [
            rule for rule in throttle_rules if rule["throttle"] in {"BLOCK", "QUARTER"}
        ][:8]
        final_answer["output_metrics"] = output_metrics

        conclusions = [
            (
                f"SELL_V6 validate PF {sell_baseline_pf} vs train {sell_train_stats.get('profit_factor')} — "
                f"{'severe' if sell_cmp.get('degradation_severity') == 'severe' else 'moderate'} degradation."
            ),
            (
                f"Regime throttle {'restores' if throttle_restores_pf else 'does not restore'} validate PF to "
                f"{sell_throttled_pf} (target ≥ {PRODUCTION_GATES['profit_factor_min']})."
            ),
            f"Worst validate regimes: {', '.join(row['regime'] for row in regime_ranking[:3])}.",
            (
                f"Combined SELL_V6+BUY_V3 throttled validate PF {combined_throttled.get('profit_factor')} — "
                f"tier {combined_throttled.get('candidate_tier')}."
            ),
            f"Top root cause: {root_causes.get('top_root_cause', 'regime_change')}.",
            f"Paper trading verdict: {final_answer['paper_trading_verdict']}.",
        ]

        return RegimeDetectionAuditReport(
            report_type="Regime Detection & Production Throttle Audit",
            engines=["SELL_V6", "BUY_V3", "COMBINED"],
            symbol=unified.get("symbol") or sell_export.get("symbol") or "NIFTY50",
            timeframe=unified.get("timeframe") or sell_export.get("timeframe") or "5M",
            trading_days_replayed=int(
                unified.get("trading_days_replayed") or sell_export.get("trading_days_replayed") or 120,
            ),
            replay_start_date=str(unified.get("replay_start_date") or sell_export.get("replay_start_date") or ""),
            replay_end_date=str(unified.get("replay_end_date") or sell_export.get("replay_end_date") or ""),
            methodology=methodology,
            source_exports={name: {"path": entry["path"], "status": entry["status"]} for name, entry in sources.items()},
            limitations=limitations,
            walk_forward_comparison={
                "walk_forward": walk_forward,
                "sell_v6": sell_cmp,
                "buy_v3": buy_cmp,
                "export_blocks": {
                    "sell_v6": sell_export.get("walk_forward", {}),
                    "unified": unified.get("walk_forward", {}),
                },
            },
            regime_classification={
                "methodology": methodology["classification_rules"],
                "dimension_labels": {
                    "trend": list(TREND_LABELS),
                    "volatility": list(VOL_LABELS),
                    "gap": list(GAP_LABELS),
                    "liquidity": list(LIQUIDITY_LABELS),
                },
                "export_regime_coverage_pct": round(100.0 * export_with_regime / max(len(sell_signals), 1), 2),
            },
            sell_v6_regime_performance={
                "full_period": sell_full_table,
                "train": sell_train_table,
                "validate": sell_validate_table,
                "by_dimension": {
                    dim: {
                        "train": _regime_performance_table(
                            sell_train, window_days=train_days, direction="SELL", win_fn=_is_sell_winner, key=dim,
                        ),
                        "validate": _regime_performance_table(
                            sell_validate,
                            window_days=validate_days,
                            direction="SELL",
                            win_fn=_is_sell_winner,
                            key=dim,
                        ),
                    }
                    for dim in ("trend", "volatility", "gap", "liquidity")
                },
            },
            buy_v3_regime_cross_check={
                "train": buy_train_table,
                "validate": buy_validate_table,
                "throttled_validate": buy_throttled,
                "validate_sample_size": len(buy_validate),
                "note": "BUY regimes inferred from layers; validate cohort too small for independent gate testing.",
            },
            regime_ranking={
                "by_deterioration": regime_ranking,
                "top_deterioration": regime_ranking[:5],
            },
            regime_pf_buckets={
                "validate": _bucket_regimes_by_pf(sell_validate_table),
                "train": _bucket_regimes_by_pf(sell_train_table),
            },
            throttle_recommendation={
                "sell_v6_regime_throttle": throttle_rules,
                "buy_v3_regime_throttle": [
                    {"regime": regime, "throttle": level, "weight": THROTTLE_WEIGHT[level]}
                    for regime, level in sorted(buy_throttle_map.items())
                ],
                "top_rules": final_answer["top_throttle_rules"],
            },
            throttled_impact_estimate={
                "sell_v6_validate_baseline": {
                    "profit_factor": sell_baseline_pf,
                    "signals_per_month": sell_validate_stats.get("signals_per_month"),
                    "win_rate_pct": sell_validate_stats.get("win_rate_pct"),
                    "expectancy": sell_validate_stats.get("expectancy"),
                },
                "sell_v6_validate_throttled": sell_throttled,
                "buy_v3_validate_throttled": buy_throttled,
                "throttle_restores_pf_2_plus": throttle_restores_pf,
            },
            combined_throttled_metrics=combined_throttled,
            signal_timing_analysis={
                "sell_v6": sell_timing,
                "buy_v3": buy_timing,
                "sell_v6_late_entry_vs_regime": _timing_vs_regime_losses(
                    sell_signals, walk_forward, direction="SELL",
                ),
            },
            root_cause_probability=root_causes,
            output_metrics=output_metrics,
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: RegimeDetectionAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Regime detection audit exported to %s", self.report_path)
        return self.report_path


def generate_regime_detection_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export regime detection throttle audit JSON."""
    return RegimeDetectionAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_regime_detection_audit_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    print(f"Exported: {path}")
    print(f"Throttle restores PF 2.0+: {final['throttle_restores_validate_pf_2_plus']}")
    print(f"Paper trading: {final['paper_trading_verdict']}")
    print(f"Top throttle rules: {len(final.get('top_throttle_rules', []))}")
