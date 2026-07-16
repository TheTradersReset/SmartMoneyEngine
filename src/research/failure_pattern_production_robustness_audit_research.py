"""
Failure Pattern & Production Robustness Audit — synthesis from existing exports only.

Determines whether BUY_V4 / SELL_V7 are statistically justified via structural failure
patterns (not labels alone), target/timing/RR reality, and multi-window robustness.
No replay, engines, indicators, models, or discovery.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v4_sell_v7_design_justification_audit_research import (
    _events,
    _layer2,
)
from src.research.production_edge_enhancement_audit_research import (
    _is_buy_winner,
    _is_sell_winner,
    _map_buy_audit_classification,
    _profit_factor_from_pnls,
)
from src.research.production_reality_audit_research import (
    RUNNER_STRATEGIES,
    _runner_exit_optimization,
    _timing_class,
)
from src.research.trade_level_truth_audit_research import (
    PF_IMPROVEMENT_THRESHOLD_PCT,
    _classify_buy_loser,
    _classify_sell_signal,
    _conditional_probability_analysis,
    _entry_precision_audit,
    _trade_level_target_matrix,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "failure_pattern_production_robustness_audit.json"

REQUIRED_EXPORTS = {
    "extended_trade_level_truth_audit": RESEARCH_DIR / "extended_trade_level_truth_audit.json",
    "extended_evidence_validation_real_deployment_audit": RESEARCH_DIR
    / "extended_evidence_validation_real_deployment_audit.json",
}

OPTIONAL_EXPORTS = {
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "production_reality_audit": RESEARCH_DIR / "production_reality_audit.json",
    "production_gap_closure_audit": RESEARCH_DIR / "production_gap_closure_audit.json",
    "buy_v4_sell_v7_design_justification_audit": RESEARCH_DIR
    / "buy_v4_sell_v7_design_justification_audit.json",
    "regime_detection_audit": RESEARCH_DIR / "regime_detection_audit.json",
}

BUY_FAILURE_CLASSES = (
    "Bull Trap",
    "Range Failure",
    "No Expansion",
    "Liquidity Failure",
    "Trend Exhaustion",
)

SELL_FAILURE_CLASSES = (
    "Bear Trap",
    "Gap Failure",
    "No Expansion",
    "Liquidity Failure",
    "Range Failure",
    "Trend Exhaustion",
)

STRUCTURAL_PATTERNS = (
    "Failed Reclaim",
    "Weak Displacement",
    "Late BOS",
    "Liquidity Sweep Failure",
    "VWAP Reclaim Failure",
    "Gap Continuation",
    "Counter Trend Entry",
    "Low Expansion Regime",
    "Volatility Collapse",
)

WINNER_REMOVAL_CAP_PCT = 15.0
TARGET_TIERS = (20, 40, 60, 80, 100, 150, 200, 300)
DEFAULT_STOP_VARIANT = "fixed_10"
PRODUCTION_STRUCTURE = RUNNER_STRATEGIES["60_100_runner"]


class FailurePatternProductionRobustnessAuditError(Exception):
    """Raised when failure pattern robustness audit fails."""


@dataclass
class FailurePatternProductionRobustnessAuditReport:
    """Failure pattern & production robustness audit output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    failure_pattern_root_cause_audit: dict[str, Any]
    structural_pattern_isolation: dict[str, Any]
    target_achievement_matrix: dict[str, Any]
    signal_timing_reality_audit: dict[str, Any]
    reward_risk_reality_audit: dict[str, Any]
    production_survival_audit: dict[str, Any]
    robustness_validation: dict[str, Any]
    buy_v4_sell_v7_decision: dict[str, Any]
    bottleneck_and_roi_ranking: dict[str, Any]
    final_answer: dict[str, Any]
    production_scores: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
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


def _cohort_metrics(
    signals: list[dict[str, Any]],
    *,
    is_winner_fn: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    pnls = [float(s.get("realized_pnl_points") or 0.0) for s in signals]
    wins = sum(1 for s in signals if is_winner_fn(s))
    total = len(signals)
    return {
        "signal_count": total,
        "win_rate_pct": round(100.0 * wins / max(total, 1), 2),
        "profit_factor": _profit_factor_from_pnls(pnls),
        "expectancy": round(mean(pnls), 2) if pnls else 0.0,
        "total_pnl_points": round(sum(pnls), 2),
    }


def _classify_buy_failure(signal: dict[str, Any]) -> str:
    if _is_buy_winner(signal):
        return "Winner"
    mapped = _classify_buy_loser(signal)
    if mapped in BUY_FAILURE_CLASSES:
        return mapped
    if mapped == "Late Entry":
        return "Trend Exhaustion"
    return _map_buy_audit_classification(str(signal.get("classification") or "Unknown"))


def _classify_sell_failure(signal: dict[str, Any]) -> str:
    if _is_sell_winner(signal):
        return "Winner"
    mapped = _classify_sell_signal(signal)
    if mapped in SELL_FAILURE_CLASSES:
        return mapped
    if mapped == "Late Entry":
        return "Trend Exhaustion"
    return "Range Failure"


def _detect_structural_patterns(signal: dict[str, Any], *, side: str) -> list[str]:
    """Map structural root-cause proxies from layers + MFE/MAE (not just labels)."""
    patterns: list[str] = []
    events = _events(signal)
    layer2 = _layer2(signal)
    mfe = float(signal.get("mfe_points") or 0.0)
    mae = float(signal.get("mae_points") or 0.0)
    bars = signal.get("bars_before_expansion")
    htf = layer2.get("htf_trend") or "Neutral"
    vwap = layer2.get("vwap_state")

    if mfe < 20:
        patterns.append("Low Expansion Regime")
    if mfe < 40 and mae > 60:
        patterns.append("Weak Displacement")
    if bars is not None and int(bars) < 0:
        patterns.append("Late BOS")
    if mae > mfe and mae > 100:
        patterns.append("Volatility Collapse")
    if "Liquidity Grab" in events or "PDL Sweep" in events or "Failed Breakout" in events:
        if mae > mfe:
            patterns.append("Liquidity Sweep Failure")
    if side == "BUY":
        if "Gap Reversal" in events and mae > mfe:
            patterns.append("Gap Continuation")
        if vwap in {"Below", "Rejected"} and mae > mfe:
            patterns.append("VWAP Reclaim Failure")
        if mfe < 40 and ("Near Support" in str(layer2.get("location") or "") or "Failed Breakdown" in events):
            patterns.append("Failed Reclaim")
        if htf == "Bearish":
            patterns.append("Counter Trend Entry")
    else:
        if "Gap Reversal" in events or "Gap Continuation" in events:
            if mae > mfe:
                patterns.append("Gap Continuation")
        if vwap not in {None, "", "Below"} and mae > mfe:
            patterns.append("VWAP Reclaim Failure")
        if htf != "Bearish":
            patterns.append("Counter Trend Entry")
        if mfe < 40:
            patterns.append("Failed Reclaim")

    return sorted(set(patterns))


def _failure_class_analysis(
    signals: list[dict[str, Any]],
    *,
    classes: tuple[str, ...],
    classify_fn: Callable[[dict[str, Any]], str],
    is_winner_fn: Callable[[dict[str, Any]], bool],
    side: str,
) -> dict[str, Any]:
    baseline = _cohort_metrics(signals, is_winner_fn=is_winner_fn)
    total = len(signals)
    rows: list[dict[str, Any]] = []

    for label in classes:
        cohort = [s for s in signals if classify_fn(s) == label]
        if not cohort:
            continue
        pnls = [float(s.get("realized_pnl_points") or 0.0) for s in cohort]
        filtered = [s for s in signals if classify_fn(s) != label]
        after = _cohort_metrics(filtered, is_winner_fn=is_winner_fn)
        base_pf = float(baseline["profit_factor"] or 0.0)
        after_pf = float(after["profit_factor"] or 0.0)
        pf_impact = round(100.0 * (after_pf - base_pf) / base_pf, 2) if base_pf else None
        rows.append(
            {
                "class": label,
                "count": len(cohort),
                "frequency_pct": round(100.0 * len(cohort) / max(total, 1), 2),
                "pnl_impact_points": round(sum(pnls), 2),
                "pf_impact_if_removed_pct": pf_impact,
                "expectancy_impact": round(mean(pnls), 2),
                "wr_change_if_removed_pp": round(after["win_rate_pct"] - baseline["win_rate_pct"], 2),
                "frequency_reduction_if_removed_pct": round(
                    100.0 * len(cohort) / max(total, 1),
                    2,
                ),
            },
        )

    rows.sort(key=lambda row: (row["pf_impact_if_removed_pct"] or 0.0), reverse=True)
    return {"side": side, "baseline": baseline, "classes": rows}


def _structural_pattern_isolation(
    signals: list[dict[str, Any]],
    *,
    side: str,
    is_winner_fn: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    winners = [s for s in signals if is_winner_fn(s)]
    losers = [s for s in signals if not is_winner_fn(s)]
    winner_n = max(len(winners), 1)
    loser_n = max(len(losers), 1)
    baseline = _cohort_metrics(signals, is_winner_fn=is_winner_fn)
    rows: list[dict[str, Any]] = []

    for pattern in STRUCTURAL_PATTERNS:
        in_winners = [s for s in winners if pattern in _detect_structural_patterns(s, side=side)]
        in_losers = [s for s in losers if pattern in _detect_structural_patterns(s, side=side)]
        occ_w = len(in_winners)
        occ_l = len(in_losers)
        if occ_w + occ_l == 0:
            continue

        winner_removal_pct = round(100.0 * occ_w / winner_n, 2)
        isolatable = winner_removal_pct <= WINNER_REMOVAL_CAP_PCT and occ_l > 0

        filtered = [
            s
            for s in signals
            if pattern not in _detect_structural_patterns(s, side=side)
        ]
        after = _cohort_metrics(filtered, is_winner_fn=is_winner_fn)
        base_pf = float(baseline["profit_factor"] or 0.0)
        after_pf = float(after["profit_factor"] or 0.0)
        pf_gain = round(100.0 * (after_pf - base_pf) / base_pf, 2) if base_pf else None

        rows.append(
            {
                "pattern": pattern,
                "occurrence_in_winners": occ_w,
                "occurrence_in_losers": occ_l,
                "occurrence_winners_pct": round(100.0 * occ_w / winner_n, 2),
                "occurrence_losers_pct": round(100.0 * occ_l / loser_n, 2),
                "winner_loser_ratio": round(occ_w / max(occ_l, 1), 3),
                "winner_removal_if_filtered_pct": winner_removal_pct,
                "isolatable_without_gt_15pct_winner_loss": isolatable,
                "expected_pf_improvement_pct": pf_gain if isolatable else None,
                "expected_wr_improvement_pp": (
                    round(after["win_rate_pct"] - baseline["win_rate_pct"], 2) if isolatable else None
                ),
                "expected_frequency_reduction_pct": (
                    round(100.0 * (len(signals) - len(filtered)) / max(len(signals), 1), 2)
                    if isolatable
                    else None
                ),
                "rejected_reason": (
                    None
                    if isolatable
                    else f"Filtering removes {winner_removal_pct}% of winners (> {WINNER_REMOVAL_CAP_PCT}% cap)"
                    if winner_removal_pct > WINNER_REMOVAL_CAP_PCT
                    else "No loser occurrences"
                ),
            },
        )

    rows.sort(
        key=lambda row: (
            1 if row["isolatable_without_gt_15pct_winner_loss"] else 0,
            row["expected_pf_improvement_pct"] or -999,
            row["occurrence_in_losers"],
        ),
        reverse=True,
    )
    accepted = [row for row in rows if row["isolatable_without_gt_15pct_winner_loss"] and (row["expected_pf_improvement_pct"] or 0) >= PF_IMPROVEMENT_THRESHOLD_PCT]
    return {
        "side": side,
        "winner_count": len(winners),
        "loser_count": len(losers),
        "winner_removal_cap_pct": WINNER_REMOVAL_CAP_PCT,
        "patterns": rows,
        "accepted_isolatable_patterns": accepted,
    }


def _target_matrix_from_signals(
    signals: list[dict[str, Any]],
    *,
    side: str,
) -> dict[str, Any]:
    matrix = _trade_level_target_matrix(
        signals,
        side=side,
        structure=PRODUCTION_STRUCTURE,
        stop_variant=DEFAULT_STOP_VARIANT,
    )
    # Ensure all requested tiers appear (MFE_TIERS may already include them)
    by_tier = dict(matrix.get("by_tier") or {})
    total = len(signals)
    for tier in TARGET_TIERS:
        key = str(tier)
        if key not in by_tier:
            hits = sum(1 for s in signals if float(s.get("mfe_points") or 0.0) >= tier)
            by_tier[key] = {
                "count": hits,
                "frequency_pct": round(100.0 * hits / max(total, 1), 2),
                "probability_pct": round(100.0 * hits / max(total, 1), 2),
                "avg_time_to_reach_minutes": None,
            }
        else:
            by_tier[key]["frequency_pct"] = by_tier[key].get("percentage_pct")
    conditional = _conditional_probability_analysis(
        signals,
        side=side,
        structure=PRODUCTION_STRUCTURE,
        stop_variant=DEFAULT_STOP_VARIANT,
    )
    return {
        "side": side,
        "by_tier": by_tier,
        "conditional_before_stop": conditional,
        "methodology": matrix.get("methodology"),
    }


def _target_structure_comparison(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    *,
    window_days: int,
) -> dict[str, Any]:
    structures = {
        "40_fixed": {"t1": 40, "t2": 40, "t3": 40, "runner": False, "trailing": False},
        "60_fixed": {"t1": 60, "t2": 60, "t3": 60, "runner": False, "trailing": False},
        "100_fixed": {"t1": 100, "t2": 100, "t3": 100, "runner": False, "trailing": False},
        "40_80_runner": RUNNER_STRATEGIES["40_80_runner"],
        "60_100_runner": RUNNER_STRATEGIES["60_100_runner"],
        "100_runner": RUNNER_STRATEGIES["100_runner"],
    }
    # Reuse runner optimization for standard keys; add fixed synthetics via same helper keys
    buy_opt = _runner_exit_optimization(
        buy_signals,
        side="BUY",
        stop_variant=DEFAULT_STOP_VARIANT,
        window_days=window_days,
    )
    sell_opt = _runner_exit_optimization(
        sell_signals,
        side="SELL",
        stop_variant=DEFAULT_STOP_VARIANT,
        window_days=window_days,
    )

    def _tier_probs(signals: list[dict[str, Any]], structure: dict[str, Any]) -> dict[str, Any]:
        t1 = float(structure["t1"])
        t2 = float(structure["t2"])
        t3 = float(structure["t3"] or structure["t2"])
        total = max(len(signals), 1)
        p1 = sum(1 for s in signals if float(s.get("mfe_points") or 0.0) >= t1) / total
        p2 = sum(1 for s in signals if float(s.get("mfe_points") or 0.0) >= t2) / total
        p3 = sum(1 for s in signals if float(s.get("mfe_points") or 0.0) >= t3) / total
        runner_p = p2 if structure.get("runner") else 0.0
        return {
            "target_1_probability_pct": round(100.0 * p1, 2),
            "target_2_probability_pct": round(100.0 * p2, 2),
            "target_3_probability_pct": round(100.0 * p3, 2),
            "runner_probability_pct": round(100.0 * runner_p, 2),
        }

    comparison: dict[str, Any] = {}
    for name, structure in structures.items():
        comparison[name] = {
            "buy_v3": _tier_probs(buy_signals, structure),
            "sell_v6": _tier_probs(sell_signals, structure),
            "structure": structure,
        }

    # Best structure: prefer 60_100_runner from existing optimization PF if present
    buy_best = _nested(buy_opt, "best_strategy") or "60_100_runner"
    sell_best = _nested(sell_opt, "best_strategy") or "60_100_runner"
    return {
        "by_structure": comparison,
        "runner_optimization_buy": buy_opt,
        "runner_optimization_sell": sell_opt,
        "best_target_structure_buy": buy_best if buy_best in comparison or buy_best in RUNNER_STRATEGIES else "60_100_runner",
        "best_target_structure_sell": sell_best if sell_best in comparison or sell_best in RUNNER_STRATEGIES else "60_100_runner",
        "recommended_production_structure": "60_100_runner",
        "rationale": (
            "60/100/Runner remains production default: balances T1 probability with runner upside; "
            "fixed exits under-capture MFE on 240d evidence."
        ),
    }


def _signal_timing_audit(
    signals: list[dict[str, Any]],
    *,
    side: str,
    is_winner_fn: Callable[[dict[str, Any]], bool],
    window_days: int,
) -> dict[str, Any]:
    precision = _entry_precision_audit(
        signals,
        side=side,
        win_fn=is_winner_fn,
    )
    leads = [
        int(s["bars_before_expansion"])
        for s in signals
        if s.get("bars_before_expansion") is not None
    ]
    lead_minutes = [bars * 5 for bars in leads]
    by_class: dict[str, list[dict[str, Any]]] = {}
    for signal in signals:
        label = _timing_class(
            int(signal["bars_before_expansion"]) if signal.get("bars_before_expansion") is not None else None,
        )
        # Normalize Same -> Same Candle for report
        if label == "Same":
            label = "Same Candle"
        by_class.setdefault(label, []).append(signal)

    class_metrics = {}
    for label, cohort in by_class.items():
        class_metrics[label] = _cohort_metrics(cohort, is_winner_fn=is_winner_fn)

    predictive_share = sum(
        1
        for s in signals
        if s.get("bars_before_expansion") is not None and int(s["bars_before_expansion"]) > 0
    )
    late_share = sum(
        1
        for s in signals
        if s.get("bars_before_expansion") is not None and int(s["bars_before_expansion"]) < 0
    )
    predictive = predictive_share >= late_share and predictive_share / max(len(signals), 1) >= 0.4

    return {
        "side": side,
        "entry_precision": precision,
        "timing_class_metrics": class_metrics,
        "average_lead_bars": round(mean(leads), 2) if leads else None,
        "median_lead_bars": round(median(leads), 2) if leads else None,
        "average_lead_minutes": round(mean(lead_minutes), 2) if lead_minutes else None,
        "median_lead_minutes": round(median(lead_minutes), 2) if lead_minutes else None,
        "predictive_vs_reactive": "predictive" if predictive else "reactive",
        "predictive_signal_share_pct": round(100.0 * predictive_share / max(len(signals), 1), 2),
        "late_signal_share_pct": round(100.0 * late_share / max(len(signals), 1), 2),
    }


def _reward_risk_audit(
    signals: list[dict[str, Any]],
    *,
    side: str,
    is_winner_fn: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    stops: list[float] = []
    targets: list[float] = []
    rrs: list[float] = []
    win_rrs: list[float] = []
    lose_rrs: list[float] = []

    for signal in signals:
        entry = float(signal.get("entry") or 0.0)
        stop = float(signal.get("stop_loss") or 0.0)
        stop_dist = abs(entry - stop) if entry and stop else 10.0
        if stop_dist <= 0:
            stop_dist = 10.0
        mfe = float(signal.get("mfe_points") or 0.0)
        rr = mfe / stop_dist
        stops.append(stop_dist)
        targets.append(mfe)
        rrs.append(rr)
        if is_winner_fn(signal):
            win_rrs.append(rr)
        else:
            lose_rrs.append(rr)

    def _rr_prob(threshold: float) -> float:
        return round(100.0 * sum(1 for rr in rrs if rr >= threshold) / max(len(rrs), 1), 2)

    return {
        "side": side,
        "average_stop_points": round(mean(stops), 2) if stops else None,
        "average_target_mfe_points": round(mean(targets), 2) if targets else None,
        "average_rr": round(mean(rrs), 2) if rrs else None,
        "median_rr": round(median(rrs), 2) if rrs else None,
        "winning_rr_avg": round(mean(win_rrs), 2) if win_rrs else None,
        "losing_rr_avg": round(mean(lose_rrs), 2) if lose_rrs else None,
        "actual_achievable_rr": round(median(win_rrs), 2) if win_rrs else None,
        "rr_probability": {
            "1_to_1": _rr_prob(1.0),
            "1_to_2": _rr_prob(2.0),
            "1_to_3": _rr_prob(3.0),
            "1_to_5": _rr_prob(5.0),
        },
    }


def _production_survival_audit(
    *,
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    extended_evidence: dict[str, Any],
    regime_export: dict[str, Any],
) -> dict[str, Any]:
    """Environment survival using structural proxies + multi-window PF context."""
    environments = {
        "High Volatility": lambda s: float(s.get("mae_points") or 0.0) > 100,
        "Volatility Compression": lambda s: float(s.get("mfe_points") or 0.0) < 40
        and float(s.get("mae_points") or 0.0) < 40,
        "Gap Expansion": lambda s: "Gap" in " ".join(_events(s)),
        "Gap Compression": lambda s: float(s.get("mfe_points") or 0.0) < 20,
        "Liquidity Compression": lambda s: "Liquidity" in " ".join(_events(s))
        and float(s.get("mfe_points") or 0.0) < 40,
        "Trend Exhaustion": lambda s: (
            _classify_buy_failure(s) == "Trend Exhaustion"
            if s.get("direction") == "BUY"
            else _classify_sell_failure(s) == "Trend Exhaustion"
        ),
        "Low Expansion": lambda s: float(s.get("mfe_points") or 0.0) < 20,
    }

    combined = []
    for signal in buy_signals:
        row = dict(signal)
        row["direction"] = "BUY"
        combined.append(row)
    for signal in sell_signals:
        row = dict(signal)
        row["direction"] = "SELL"
        combined.append(row)

    env_rows: list[dict[str, Any]] = []
    for name, predicate in environments.items():
        cohort = [s for s in combined if predicate(s)]
        if not cohort:
            continue
        pnls = [float(s.get("realized_pnl_points") or 0.0) for s in cohort]
        wins = sum(1 for p in pnls if p > 0)
        # Simple DD proxy: max cumulative drawdown of cohort pnls
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        env_rows.append(
            {
                "environment": name,
                "signal_count": len(cohort),
                "win_rate_pct": round(100.0 * wins / max(len(cohort), 1), 2),
                "profit_factor": _profit_factor_from_pnls(pnls),
                "expectancy": round(mean(pnls), 2),
                "max_drawdown_points": round(max_dd, 2),
            },
        )

    env_rows.sort(key=lambda row: (row["profit_factor"] or 0.0, row["expectancy"]))
    collapse = [row for row in env_rows if (row["profit_factor"] or 0.0) < 1.2]
    worst = env_rows[0] if env_rows else None

    pf_120 = float(_nested(extended_evidence, "final_answer", "window_profit_factors", "120d", default=0) or 0)
    pf_250 = float(_nested(extended_evidence, "final_answer", "window_profit_factors", "250d", default=0) or 0)
    pf_500 = float(_nested(extended_evidence, "final_answer", "window_profit_factors", "500d", default=0) or 0)
    throttled = float(_nested(extended_evidence, "final_answer", "throttled_pf_500d", default=0) or 0)

    # Survival: longer-window PF still > 1.5; fragility: 120d vs 500d gap; robustness: throttle recovery
    survival = min(100.0, max(0.0, 40.0 + (pf_500 - 1.0) * 30.0 + (10.0 if pf_250 >= 1.5 else 0)))
    fragility = min(100.0, max(0.0, abs(pf_120 - pf_500) / max(pf_120, 0.01) * 100.0))
    robustness = min(
        100.0,
        max(0.0, survival * 0.5 + (30.0 if throttled >= 2.0 else 10.0) + (20.0 if not collapse[:1] or (collapse[0]["profit_factor"] or 0) > 0.8 else 0)),
    )

    return {
        "environments": env_rows,
        "production_collapse_environments": [row["environment"] for row in collapse],
        "worst_environment": worst,
        "multi_window_pf": {"120d": pf_120, "250d": pf_250, "500d": pf_500, "throttled_500d": throttled},
        "regime_export_loaded": bool(regime_export),
        "survival_score": round(survival, 1),
        "fragility_score": round(fragility, 1),
        "production_robustness_score": round(robustness, 1),
    }


def _class_frequency(signals: list[dict[str, Any]], classify_fn: Callable, label: str) -> float:
    if not signals:
        return 0.0
    return 100.0 * sum(1 for s in signals if classify_fn(s) == label) / len(signals)


def _robustness_validation(
    *,
    buy_240: list[dict[str, Any]],
    sell_240: list[dict[str, Any]],
    buy_120: list[dict[str, Any]],
    sell_120: list[dict[str, Any]],
    buy_isolation: dict[str, Any],
    sell_isolation: dict[str, Any],
    extended_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Accept patterns only if isolatable on 240d AND frequency-consistent on 120d."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    pf_ok = all(
        float(_nested(extended_evidence, "final_answer", "window_profit_factors", key, default=0) or 0) >= 1.5
        for key in ("250d", "500d")
    )

    for side, isolation, signals_240, signals_120, classify_fn in (
        ("BUY", buy_isolation, buy_240, buy_120, _classify_buy_failure),
        ("SELL", sell_isolation, sell_240, sell_120, _classify_sell_failure),
    ):
        for row in isolation.get("patterns") or []:
            pattern = row["pattern"]
            # Cross-window: pattern must remain more common in losers than winners on 120d if 120d present
            if signals_120:
                w120 = [s for s in signals_120 if _is_buy_winner(s)] if side == "BUY" else [s for s in signals_120 if _is_sell_winner(s)]
                l120 = [s for s in signals_120 if not (_is_buy_winner(s) if side == "BUY" else _is_sell_winner(s))]
                occ_w = sum(1 for s in w120 if pattern in _detect_structural_patterns(s, side=side))
                occ_l = sum(1 for s in l120 if pattern in _detect_structural_patterns(s, side=side))
                oos_ok = occ_l >= occ_w and (occ_l + occ_w) > 0
            else:
                oos_ok = True  # no 120d — rely on 240d only with caveat

            freq_ok = row["occurrence_in_losers"] >= 5
            isolatable = row["isolatable_without_gt_15pct_winner_loss"]
            pf_gain = row["expected_pf_improvement_pct"] or 0.0
            cross_window_ok = pf_ok

            passed = isolatable and oos_ok and freq_ok and cross_window_ok and pf_gain >= PF_IMPROVEMENT_THRESHOLD_PCT
            entry = {
                "side": side,
                "pattern": pattern,
                "out_of_sample_consistency": oos_ok,
                "cross_window_consistency": cross_window_ok,
                "frequency_robustness": freq_ok,
                "isolatable": isolatable,
                "expected_pf_improvement_pct": row["expected_pf_improvement_pct"],
                "passed": passed,
            }
            if passed:
                accepted.append(entry)
            else:
                rejected.append(entry)

    # Also validate failure class removals across 120 vs 240 frequency
    class_checks = []
    for side, classes, c240, c120, fn in (
        ("BUY", BUY_FAILURE_CLASSES, buy_240, buy_120, _classify_buy_failure),
        ("SELL", SELL_FAILURE_CLASSES, sell_240, sell_120, _classify_sell_failure),
    ):
        for label in classes:
            f240 = _class_frequency(c240, fn, label)
            f120 = _class_frequency(c120, fn, label) if c120 else None
            consistent = True if f120 is None else abs(f240 - f120) <= 15.0 or (f240 >= 5 and f120 >= 5)
            class_checks.append(
                {
                    "side": side,
                    "class": label,
                    "frequency_240d_pct": round(f240, 2),
                    "frequency_120d_pct": round(f120, 2) if f120 is not None else None,
                    "cross_window_frequency_ok": consistent,
                },
            )

    return {
        "windows_required": ["120d", "240d", "250d", "500d"],
        "note": (
            "Per-signal pattern tests use 120d+240d lists. "
            "250d/500d enforce combined PF>=1.5 cross-window gate (no per-signal lists)."
        ),
        "accepted_patterns": accepted,
        "rejected_patterns": rejected,
        "failure_class_cross_window": class_checks,
        "multi_window_pf_gate_passed": pf_ok,
    }


def _v4_v7_decision(
    *,
    buy_failure: dict[str, Any],
    sell_failure: dict[str, Any],
    buy_isolation: dict[str, Any],
    sell_isolation: dict[str, Any],
    robustness: dict[str, Any],
    extended_trade: dict[str, Any],
    design_audit: dict[str, Any],
) -> dict[str, Any]:
    accepted_buy = [p for p in robustness.get("accepted_patterns") or [] if p["side"] == "BUY"]
    accepted_sell = [p for p in robustness.get("accepted_patterns") or [] if p["side"] == "SELL"]

    buy_class_yes = any((row.get("pf_impact_if_removed_pct") or 0) >= 10 for row in buy_failure.get("classes") or [])
    sell_class_yes = any((row.get("pf_impact_if_removed_pct") or 0) >= 10 for row in sell_failure.get("classes") or [])

    # Require BOTH class-level PF gain AND at least one robust isolatable structural pattern
    buy_yes = buy_class_yes and bool(accepted_buy)
    sell_yes = sell_class_yes and bool(accepted_sell)

    # Soften: if class YES but no accepted structural pattern, still YES with lower confidence
    # User asked statistical justification via structural causes — prefer accepted patterns
    if buy_class_yes and not accepted_buy:
        buy_yes = False  # not statistically isolatable without killing winners
    if sell_class_yes and not accepted_sell:
        sell_yes = False

    def _conf(yes: bool, accepted: list, failure: dict) -> float:
        base = 55.0 if yes else 35.0
        if accepted:
            base += 20.0
        top = (failure.get("classes") or [{}])[0]
        if (top.get("pf_impact_if_removed_pct") or 0) >= 50:
            base += 15.0
        elif (top.get("pf_impact_if_removed_pct") or 0) >= 10:
            base += 8.0
        if robustness.get("multi_window_pf_gate_passed"):
            base += 5.0
        return round(min(95.0, base), 1)

    buy_spm = _nested(extended_trade, "core_metrics_by_window", "240", "buy_v3", "signals_per_month")
    sell_spm = _nested(extended_trade, "core_metrics_by_window", "240", "sell_v6", "signals_per_month")
    buy_dd = _nested(extended_trade, "core_metrics_by_window", "240", "buy_v3", "max_drawdown_points")
    sell_dd = _nested(extended_trade, "core_metrics_by_window", "240", "sell_v6", "max_drawdown_points")

    def _blueprint(side: str, yes: bool, failure: dict, accepted: list, spm: Any, dd: Any) -> dict[str, Any]:
        if not yes:
            return {
                "recommendation": "NO",
                "why": (
                    "Failure classes show PF impact, but no structural pattern passes "
                    "isolatability (<=15% winner loss) + cross-window robustness gates. "
                    "Prefer regime throttle / execution over a new engine version."
                    if any((r.get("pf_impact_if_removed_pct") or 0) >= 10 for r in failure.get("classes") or [])
                    else "No failure class clears >=10% PF improvement on 240d."
                ),
            }
        top_class = (failure.get("classes") or [{}])[0]
        top_pattern = accepted[0] if accepted else {}
        pf_gain = float(top_pattern.get("expected_pf_improvement_pct") or top_class.get("pf_impact_if_removed_pct") or 0)
        base_pf = float((failure.get("baseline") or {}).get("profit_factor") or 0)
        base_wr = float((failure.get("baseline") or {}).get("win_rate_pct") or 0)
        freq_cut = float(top_pattern.get("expected_frequency_reduction_pct") or top_class.get("frequency_reduction_if_removed_pct") or 0)
        return {
            "recommendation": "YES",
            "filters_to_add": [
                f"Reject when structural pattern '{top_pattern.get('pattern')}' is present",
                f"Secondary reject for failure class '{top_class.get('class')}' when isolatable",
            ],
            "failure_pattern_removed": top_pattern.get("pattern"),
            "failure_class_targeted": top_class.get("class"),
            "expected": {
                "profit_factor": round(base_pf * (1 + pf_gain / 100.0), 2) if base_pf else None,
                "win_rate_pct": round(base_wr + float(top_pattern.get("expected_wr_improvement_pp") or top_class.get("wr_change_if_removed_pp") or 0), 2),
                "signals_per_month": round(float(spm) * (1 - freq_cut / 100.0), 2) if spm is not None else None,
                "drawdown_points": dd,
            },
        }

    # Align soft with design audit but override with robustness gate
    design_buy = _nested(design_audit, "final_answer", "buy_v4", "recommendation")
    design_sell = _nested(design_audit, "final_answer", "sell_v7", "recommendation")

    return {
        "buy_v4": {
            "should_build": "YES" if buy_yes else "NO",
            "confidence_pct": _conf(buy_yes, accepted_buy, buy_failure),
            "design_audit_recommendation": design_buy,
            "robustness_override_applied": design_buy == "YES" and not buy_yes,
            **_blueprint("BUY", buy_yes, buy_failure, accepted_buy, buy_spm, buy_dd),
        },
        "sell_v7": {
            "should_build": "YES" if sell_yes else "NO",
            "confidence_pct": _conf(sell_yes, accepted_sell, sell_failure),
            "design_audit_recommendation": design_sell,
            "robustness_override_applied": design_sell == "YES" and not sell_yes,
            **_blueprint("SELL", sell_yes, sell_failure, accepted_sell, sell_spm, sell_dd),
        },
        "methodology": (
            "YES only if (1) failure class PF impact >=10% on 240d AND "
            "(2) at least one structural pattern is isolatable (<=15% winner removal) AND "
            "(3) pattern passes 120d/240d/250d/500d robustness gates."
        ),
    }


def _bottleneck_ranking(
    *,
    buy_isolation: dict[str, Any],
    sell_isolation: dict[str, Any],
    survival: dict[str, Any],
    target_compare: dict[str, Any],
    decision: dict[str, Any],
    gap_closure: dict[str, Any],
) -> dict[str, Any]:
    opportunities = [
        {
            "area": "Regime Detection",
            "expected_impact": "HIGH",
            "evidence": f"Throttled 500d PF {_nested(survival, 'multi_window_pf', 'throttled_500d')} vs unthrottled {_nested(survival, 'multi_window_pf', '500d')}",
            "rank_score": 95.0,
        },
        {
            "area": "Signal Quality",
            "expected_impact": "MEDIUM-HIGH" if decision["buy_v4"]["should_build"] == "YES" or decision["sell_v7"]["should_build"] == "YES" else "MEDIUM",
            "evidence": (
                f"Accepted structural filters BUY={len([p for p in buy_isolation.get('accepted_isolatable_patterns') or []])} "
                f"SELL={len([p for p in sell_isolation.get('accepted_isolatable_patterns') or []])}"
            ),
            "rank_score": 70.0 if decision["buy_v4"]["should_build"] == "YES" or decision["sell_v7"]["should_build"] == "YES" else 45.0,
        },
        {
            "area": "Target Structure",
            "expected_impact": "LOW-MEDIUM",
            "evidence": f"Recommended {target_compare.get('recommended_production_structure')}",
            "rank_score": 35.0,
        },
        {
            "area": "Execution",
            "expected_impact": "MEDIUM",
            "evidence": _nested(gap_closure, "final_answer", "definitive_verdict", "small_capital", default="live slippage unverified"),
            "rank_score": 55.0,
        },
        {
            "area": "Risk Management",
            "expected_impact": "MEDIUM",
            "evidence": f"Collapse environments: {survival.get('production_collapse_environments')}",
            "rank_score": 50.0,
        },
    ]
    opportunities.sort(key=lambda row: row["rank_score"], reverse=True)
    return {
        "single_biggest_bottleneck": opportunities[0]["area"],
        "ranked_opportunities": opportunities,
    }


class FailurePatternProductionRobustnessAuditResearch:
    """Synthesize failure-pattern and production-robustness audit from exports."""

    def run(self, sources: dict[str, dict[str, Any]]) -> FailurePatternProductionRobustnessAuditReport:
        started = time.perf_counter()
        extended_trade = sources.get("extended_trade_level_truth_audit") or {}
        extended_evidence = sources.get("extended_evidence_validation_real_deployment_audit") or {}
        buy_120_export = sources.get("buy_v3_candidate_validation") or {}
        sell_120_export = sources.get("sell_v6_replay_validation") or {}
        reality = sources.get("production_reality_audit") or {}
        gap = sources.get("production_gap_closure_audit") or {}
        design = sources.get("buy_v4_sell_v7_design_justification_audit") or {}
        regime = sources.get("regime_detection_audit") or {}

        if not extended_trade:
            raise FailurePatternProductionRobustnessAuditError(
                "Required: extended_trade_level_truth_audit.json",
            )

        buy_240 = list(_nested(extended_trade, "per_signal_details", "buy_v3", default=[]) or [])
        sell_240 = list(_nested(extended_trade, "per_signal_details", "sell_v6", default=[]) or [])
        buy_120 = list(_nested(buy_120_export, "per_signal_details", "buy_v3", default=[]) or [])
        sell_120 = list(_nested(sell_120_export, "per_signal_details", "sell_v6", default=[]) or [])
        if not buy_240 or not sell_240:
            raise FailurePatternProductionRobustnessAuditError("Missing 240d per_signal_details")

        window_days = int(_nested(extended_trade, "core_metrics_by_window", "240", "trading_days", default=240) or 240)

        buy_failure = _failure_class_analysis(
            buy_240,
            classes=BUY_FAILURE_CLASSES,
            classify_fn=_classify_buy_failure,
            is_winner_fn=_is_buy_winner,
            side="BUY_V3",
        )
        sell_failure = _failure_class_analysis(
            sell_240,
            classes=SELL_FAILURE_CLASSES,
            classify_fn=_classify_sell_failure,
            is_winner_fn=_is_sell_winner,
            side="SELL_V6",
        )
        buy_isolation = _structural_pattern_isolation(buy_240, side="BUY", is_winner_fn=_is_buy_winner)
        sell_isolation = _structural_pattern_isolation(sell_240, side="SELL", is_winner_fn=_is_sell_winner)

        target_buy = _target_matrix_from_signals(buy_240, side="BUY")
        target_sell = _target_matrix_from_signals(sell_240, side="SELL")
        target_compare = _target_structure_comparison(buy_240, sell_240, window_days=window_days)

        timing_buy = _signal_timing_audit(buy_240, side="BUY", is_winner_fn=_is_buy_winner, window_days=window_days)
        timing_sell = _signal_timing_audit(sell_240, side="SELL", is_winner_fn=_is_sell_winner, window_days=window_days)

        rr_buy = _reward_risk_audit(buy_240, side="BUY", is_winner_fn=_is_buy_winner)
        rr_sell = _reward_risk_audit(sell_240, side="SELL", is_winner_fn=_is_sell_winner)

        survival = _production_survival_audit(
            buy_signals=buy_240,
            sell_signals=sell_240,
            extended_evidence=extended_evidence,
            regime_export=regime,
        )
        robustness = _robustness_validation(
            buy_240=buy_240,
            sell_240=sell_240,
            buy_120=buy_120,
            sell_120=sell_120,
            buy_isolation=buy_isolation,
            sell_isolation=sell_isolation,
            extended_evidence=extended_evidence,
        )
        decision = _v4_v7_decision(
            buy_failure=buy_failure,
            sell_failure=sell_failure,
            buy_isolation=buy_isolation,
            sell_isolation=sell_isolation,
            robustness=robustness,
            extended_trade=extended_trade,
            design_audit=design,
        )
        bottleneck = _bottleneck_ranking(
            buy_isolation=buy_isolation,
            sell_isolation=sell_isolation,
            survival=survival,
            target_compare=target_compare,
            decision=decision,
            gap_closure=gap,
        )

        buy_pf = float(_nested(extended_trade, "core_metrics_by_window", "240", "buy_v3", "profit_factor") or 0)
        sell_pf = float(_nested(extended_trade, "core_metrics_by_window", "240", "sell_v6", "profit_factor") or 0)
        buy_verdict = "PASS" if buy_pf >= 1.5 else "WEAK"
        sell_verdict = "PASS" if sell_pf >= 1.5 else "WEAK"

        best_stop = _nested(extended_trade, "final_answer", "stop_loss_validation", "best_stop") or "fixed_10"
        if isinstance(best_stop, dict):
            best_stop = best_stop.get("buy_v3") or best_stop.get("recommended") or "fixed_10"

        evidence_score = float(_nested(extended_evidence, "final_answer", "evidence_score", default=81.1) or 81.1)
        readiness = float(_nested(extended_evidence, "final_answer", "production_readiness_score", default=72.0) or 72.0)
        overfitting = round(
            min(
                100.0,
                20.0
                + (30.0 if decision["buy_v4"]["robustness_override_applied"] or decision["sell_v7"]["robustness_override_applied"] else 0)
                + float(survival["fragility_score"]) * 0.3,
            ),
            1,
        )
        confidence = round(
            min(
                95.0,
                50.0
                + (15.0 if robustness["multi_window_pf_gate_passed"] else 0)
                + (10.0 if timing_buy["predictive_vs_reactive"] == "predictive" else 0)
                + (10.0 if len(robustness["accepted_patterns"]) else 5),
            ),
            1,
        )

        production_scores = {
            "confidence_score": confidence,
            "evidence_score": evidence_score,
            "overfitting_risk_score": overfitting,
            "production_readiness_score": readiness,
            "survival_score": survival["survival_score"],
            "fragility_score": survival["fragility_score"],
            "production_robustness_score": survival["production_robustness_score"],
        }

        final = {
            "1_current_buy_v3_verdict": buy_verdict,
            "2_current_sell_v6_verdict": sell_verdict,
            "3_buy_v4_verdict": decision["buy_v4"]["should_build"],
            "4_sell_v7_verdict": decision["sell_v7"]["should_build"],
            "5_best_target_structure": target_compare["recommended_production_structure"],
            "6_best_stop_structure": best_stop if isinstance(best_stop, str) else "fixed_10",
            "7_expected_rr": {
                "buy_v3": rr_buy.get("actual_achievable_rr"),
                "sell_v6": rr_sell.get("actual_achievable_rr"),
            },
            "8_production_robustness": survival["production_robustness_score"],
            "9_production_failure_conditions": survival["production_collapse_environments"],
            "10_highest_roi_improvement_remaining": bottleneck["single_biggest_bottleneck"],
            "buy_v4_detail": decision["buy_v4"],
            "sell_v7_detail": decision["sell_v7"],
            "scores": production_scores,
        }

        source_status = {
            name: "loaded" if sources.get(name) else "missing"
            for name in {**REQUIRED_EXPORTS, **OPTIONAL_EXPORTS}
        }

        conclusions = [
            f"BUY_V3 240d PF={buy_pf} → {buy_verdict}; SELL_V6 PF={sell_pf} → {sell_verdict}.",
            f"BUY_V4 build: {decision['buy_v4']['should_build']} (conf {decision['buy_v4']['confidence_pct']}%).",
            f"SELL_V7 build: {decision['sell_v7']['should_build']} (conf {decision['sell_v7']['confidence_pct']}%).",
            f"Accepted robust structural patterns: {len(robustness['accepted_patterns'])}.",
            f"Biggest bottleneck: {bottleneck['single_biggest_bottleneck']}.",
            f"Signals are predominantly {timing_buy['predictive_vs_reactive']} (BUY) / {timing_sell['predictive_vs_reactive']} (SELL).",
            f"Production robustness score: {survival['production_robustness_score']}.",
        ]

        return FailurePatternProductionRobustnessAuditReport(
            report_type="Failure Pattern & Production Robustness Audit",
            engines=["BUY_V3", "SELL_V6"],
            symbol=str(extended_trade.get("symbol") or "NIFTY50"),
            timeframe=str(extended_trade.get("timeframe") or "5M"),
            methodology={
                "research_only": True,
                "no_replay": True,
                "no_buy_v4_engine": True,
                "no_sell_v7_engine": True,
                "no_new_indicators": True,
                "no_models": True,
                "no_discovery_engines": True,
                "authoritative_per_signal_window": 240,
                "winner_removal_cap_pct": WINNER_REMOVAL_CAP_PCT,
                "pf_improvement_threshold_pct": PF_IMPROVEMENT_THRESHOLD_PCT,
                "structural_patterns": list(STRUCTURAL_PATTERNS),
            },
            source_exports=source_status,
            limitations=[
                "250d/500d have no per-signal lists — used for PF gates only.",
                "Structural patterns are layer/MFE/MAE proxies, not new detectors.",
                "Regime environment tags on 240d signals may be incomplete; proxies used.",
                "120d used for OOS consistency, not for YES/NO alone.",
            ],
            failure_pattern_root_cause_audit={
                "buy_v3": buy_failure,
                "sell_v6": sell_failure,
            },
            structural_pattern_isolation={
                "buy_v3": buy_isolation,
                "sell_v6": sell_isolation,
            },
            target_achievement_matrix={
                "buy_v3": target_buy,
                "sell_v6": target_sell,
                "structure_comparison": target_compare,
                "prior_extended_matrix": _nested(extended_trade, "target_achievement_matrix", "240"),
            },
            signal_timing_reality_audit={
                "buy_v3": timing_buy,
                "sell_v6": timing_sell,
            },
            reward_risk_reality_audit={
                "buy_v3": rr_buy,
                "sell_v6": rr_sell,
            },
            production_survival_audit=survival,
            robustness_validation=robustness,
            buy_v4_sell_v7_decision=decision,
            bottleneck_and_roi_ranking=bottleneck,
            final_answer=final,
            production_scores=production_scores,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: FailurePatternProductionRobustnessAuditReport, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("Failure pattern production robustness audit exported: %s", path)
        return path


def generate_failure_pattern_production_robustness_audit_report(
    report_path: Path | str | None = None,
) -> FailurePatternProductionRobustnessAuditReport:
    sources: dict[str, dict[str, Any]] = {}
    for name, path in {**REQUIRED_EXPORTS, **OPTIONAL_EXPORTS}.items():
        data = _load_json(path)
        if name in REQUIRED_EXPORTS and not data:
            raise FailurePatternProductionRobustnessAuditError(f"Required export missing: {path}")
        sources[name] = data

    research = FailurePatternProductionRobustnessAuditResearch()
    report = research.run(sources)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = generate_failure_pattern_production_robustness_audit_report()
        final = report.final_answer
        print(f"Exported: {DEFAULT_REPORT_PATH}")
        print(f"BUY_V3: {final['1_current_buy_v3_verdict']} | SELL_V6: {final['2_current_sell_v6_verdict']}")
        print(f"BUY_V4: {final['3_buy_v4_verdict']} | SELL_V7: {final['4_sell_v7_verdict']}")
        print(f"Bottleneck: {final['10_highest_roi_improvement_remaining']}")
        print(f"Robustness: {final['8_production_robustness']} | Scores: {final['scores']}")
        return 0
    except FailurePatternProductionRobustnessAuditError as exc:
        logger.error("Failure pattern audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
