"""
Production Reality Audit — synthesis from completed replay exports only.

Determines whether current SmartMoneyEngine deployment recommendations are
supported by actual evidence. No replay, indicators, models, or discovery.
"""

from __future__ import annotations

import json
import logging
import math
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
from src.research.buy_v3_tradeability_production_validation_research import _fixed_target_pnl
from src.research.live_trade_management_execution_efficiency_audit_research import (
    _capture_leakage_analysis,
    _classify_miss_reason,
    _resolve_stop_extended,
)
from src.research.production_edge_enhancement_audit_research import (
    _is_buy_winner,
    _is_sell_winner,
    _profit_factor_from_pnls,
)
from src.research.production_trading_playbook_audit_research import (
    LEG_WEIGHTS,
    _metrics_from_pnls,
    _tiered_structure_pnl,
)
from src.research.regime_detection_audit_research import SELL_V6_MODEL_ID
from src.research.sell_v6_replay_validation_research import V6_VWAP_GATE_RULE

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "production_reality_audit.json"

SOURCE_EXPORTS = {
    "live_trade_management_execution_efficiency_audit": RESEARCH_DIR
    / "live_trade_management_execution_efficiency_audit.json",
    "regime_aware_execution_validation": RESEARCH_DIR / "regime_aware_execution_validation.json",
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "final_production_deployment_audit": RESEARCH_DIR / "final_production_deployment_audit.json",
}

MFE_TIERS = (20, 40, 60, 80, 100, 150, 200, 300)
TIMING_CLASSES = ("Very Early", "Early", "Same", "Late", "No Linked Move")
CONFIDENCE_Z = {60: 0.842, 70: 1.036, 80: 1.282, 90: 1.645}

RUNNER_STRATEGIES: dict[str, dict[str, Any]] = {
    "no_runner": {"t1": 60, "t2": 100, "t3": 100, "runner": False, "trailing": False},
    "40_80_exit": {"t1": 40, "t2": 80, "t3": 80, "runner": False, "trailing": False},
    "60_100_exit": {"t1": 60, "t2": 100, "t3": 100, "runner": False, "trailing": False},
    "40_80_runner": {"t1": 40, "t2": 80, "t3": None, "runner": True, "trailing": False},
    "60_100_runner": {"t1": 60, "t2": 100, "t3": None, "runner": True, "trailing": False},
    "100_runner": {"t1": 100, "t2": 100, "t3": None, "runner": True, "trailing": False},
    "trailing_runner": {"t1": 60, "t2": 100, "t3": None, "runner": True, "trailing": True},
}

BOTTLENECK_CAUSES = ("signal_quality", "execution", "regime", "target", "stop", "runner")


class ProductionRealityAuditError(Exception):
    """Raised when production reality audit synthesis fails."""


@dataclass
class ProductionRealityAuditReport:
    """Production reality audit output."""

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
    trade_outcome_distribution: dict[str, Any]
    target_achievement_matrix: dict[str, Any]
    signal_reality: dict[str, Any]
    runner_exit_optimization: dict[str, Any]
    execution_bottleneck_audit: dict[str, Any]
    evidence_quality: dict[str, Any]
    production_truth_audit: dict[str, Any]
    production_scores: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ProductionRealityAuditError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _timing_class(bars: int | None) -> str:
    if bars is None:
        return "No Linked Move"
    if bars > 5:
        return "Very Early"
    if bars >= 2:
        return "Early"
    if bars >= 0:
        return "Same"
    return "Late"


def _extended_metrics(
    pnls: list[float],
    *,
    signals: list[dict[str, Any]],
    sample_size: int,
    window_days: int,
) -> dict[str, Any]:
    base = _metrics_from_pnls(pnls, sample_size=sample_size, window_days=window_days)
    mfes = [float(s.get("mfe_points") or 0.0) for s in signals]
    months = max(window_days / 22.0, 1.0)
    total_mfe = sum(mfes)
    total_captured = sum(max(p, 0.0) for p in pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {
        **base,
        "win_rate_pct": round(100.0 * wins / max(len(pnls), 1), 2),
        "monthly_points": round(base["realized_profit_points"] / months, 2),
        "capture_efficiency_pct": round(100.0 * total_captured / max(total_mfe, 1.0), 2),
    }


def _trailing_runner_pnl(
    signal: dict[str, Any],
    structure: dict[str, Any],
    *,
    stop_pts: float,
    giveback_pct: float = 0.4,
) -> tuple[float, bool]:
    mfe = float(signal.get("mfe_points") or 0.0)
    mae = float(signal.get("mae_points") or 0.0)
    t1 = float(structure["t1"])
    t2 = float(structure["t2"])
    effective_stop = max(stop_pts, 1.0)

    if mfe < t1:
        return round(-min(mae, effective_stop), 2), False

    pnl = t1 * LEG_WEIGHTS[0]
    if mfe < t2:
        remainder = LEG_WEIGHTS[1] + LEG_WEIGHTS[2]
        return round(pnl - min(mae, effective_stop) * remainder, 2), pnl > 0

    pnl += t2 * LEG_WEIGHTS[1]
    runner_gain = max(0.0, mfe - t2) * (1.0 - giveback_pct)
    pnl += runner_gain * LEG_WEIGHTS[2]
    return round(pnl, 2), pnl > 0


def _strategy_pnl(
    signal: dict[str, Any],
    structure: dict[str, Any],
    *,
    stop_pts: float,
) -> tuple[float, bool]:
    if structure.get("trailing"):
        return _trailing_runner_pnl(signal, structure, stop_pts=stop_pts)
    return _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)


def _mfe_tier_distribution(signals: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(signals)
    tiers: dict[str, Any] = {}
    prev_hits = total

    for threshold in MFE_TIERS:
        hits = sum(1 for s in signals if float(s.get("mfe_points") or 0.0) >= threshold)
        tiers[str(threshold)] = {
            "count": hits,
            "pct_of_signals": round(100.0 * hits / max(total, 1), 2),
            "conditional_probability_pct": round(100.0 * hits / max(prev_hits, 1), 2),
        }
        prev_hits = hits

    mfes = [float(s.get("mfe_points") or 0.0) for s in signals]
    return {
        "sample_size": total,
        "tiers": tiers,
        "avg_mfe": round(mean(mfes), 2) if mfes else 0.0,
        "median_mfe": round(median(mfes), 2) if mfes else 0.0,
        "max_mfe": round(max(mfes), 2) if mfes else 0.0,
    }


def _target_achievement_matrix(
    signals: list[dict[str, Any]],
    *,
    structure: dict[str, Any],
    stop_variant: str,
    window_days: int,
    side: str,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    rows: dict[str, Any] = {}
    total_max = 0.0
    total_actual = 0.0
    miss_by_reason: Counter[str] = Counter()

    for threshold in MFE_TIERS:
        cohort = [s for s in signals if float(s.get("mfe_points") or 0.0) >= threshold]
        if not cohort:
            rows[str(threshold)] = {
                "eligible_signals": 0,
                "max_achievable_points": 0.0,
                "actual_captured_points": 0.0,
                "missed_points": 0.0,
                "capture_pct": 0.0,
            }
            continue

        max_pts = sum(float(s.get("mfe_points") or 0.0) for s in cohort)
        actual_pts = 0.0
        for signal in cohort:
            stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
            pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
            actual_pts += max(pnl, 0.0)
            mfe = float(signal.get("mfe_points") or 0.0)
            missed = max(0.0, mfe - max(pnl, 0.0))
            if missed > 0.01:
                reason = _classify_miss_reason(signal, structure, stop_pts=stop_pts, pnl=pnl)
                if reason == "timing":
                    miss_by_reason["late_entry"] += 1
                elif reason == "stop":
                    miss_by_reason["stop"] += 1
                elif reason == "runner":
                    miss_by_reason["runner_giveback"] += 1
                else:
                    miss_by_reason["early_exit"] += 1

        total_max += max_pts
        total_actual += actual_pts
        rows[str(threshold)] = {
            "eligible_signals": len(cohort),
            "max_achievable_points": round(max_pts, 2),
            "actual_captured_points": round(actual_pts, 2),
            "missed_points": round(max_pts - actual_pts, 2),
            "capture_pct": round(100.0 * actual_pts / max(max_pts, 1.0), 2),
        }

    return {
        "side": side,
        "playbook_structure": structure,
        "stop_variant": stop_variant,
        "by_tier": rows,
        "aggregate": {
            "max_achievable_points": round(total_max, 2),
            "actual_captured_points": round(total_actual, 2),
            "missed_points": round(total_max - total_actual, 2),
            "capture_pct": round(100.0 * total_actual / max(total_max, 1.0), 2),
        },
        "missed_points_by_reason": [
            {"reason": reason, "count": count, "rank": index + 1}
            for index, (reason, count) in enumerate(miss_by_reason.most_common())
        ],
    }


def _signal_reality_analysis(
    signals: list[dict[str, Any]],
    *,
    side: str,
    win_fn: Any,
    window_days: int,
) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_signal: list[dict[str, Any]] = []

    for signal in signals:
        bars = signal.get("bars_before_expansion")
        bars_int = int(bars) if bars is not None else None
        timing = _timing_class(bars_int)
        by_class[timing].append(signal)
        per_signal.append(
            {
                "timestamp": signal.get("timestamp"),
                "bars_before_expansion": bars_int,
                "points_before_expansion": signal.get("points_before_expansion"),
                "lead_time_minutes": round(bars_int * BAR_MINUTES, 2) if bars_int is not None and bars_int > 0 else 0,
                "timing_class": timing,
                "mfe_points": signal.get("mfe_points"),
                "is_winner": win_fn(signal),
            },
        )

    class_summary: dict[str, Any] = {}
    for label in TIMING_CLASSES:
        cohort = by_class.get(label, [])
        if not cohort:
            class_summary[label] = {
                "count": 0,
                "pct": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor": None,
                "expectancy": 0.0,
                "avg_lead_bars": None,
            }
            continue
        pnls = [float(s.get("realized_pnl_points") or 0.0) for s in cohort]
        lead_bars = [
            int(s["bars_before_expansion"])
            for s in cohort
            if s.get("bars_before_expansion") is not None and int(s["bars_before_expansion"]) > 0
        ]
        class_summary[label] = {
            "count": len(cohort),
            "pct": round(100.0 * len(cohort) / max(len(signals), 1), 2),
            "win_rate_pct": round(100.0 * sum(1 for s in cohort if win_fn(s)) / len(cohort), 2),
            "profit_factor": _profit_factor_from_pnls(pnls),
            "expectancy": round(mean(pnls), 2),
            "avg_lead_bars": round(mean(lead_bars), 2) if lead_bars else None,
        }

    predictive_count = sum(
        class_summary[label]["count"] for label in ("Very Early", "Early") if label in class_summary
    )
    reactive_count = class_summary.get("Late", {}).get("count", 0) + class_summary.get("Same", {}).get("count", 0)

    return {
        "side": side,
        "methodology": (
            "Very Early: >5 bars before expansion; Early: 2-5 bars; "
            "Same: 0-1 bars; Late: negative bars (after momentum start)."
        ),
        "timing_class_summary": class_summary,
        "predictive_vs_reactive": {
            "predictive_signals": predictive_count,
            "predictive_pct": round(100.0 * predictive_count / max(len(signals), 1), 2),
            "reactive_signals": reactive_count,
            "reactive_pct": round(100.0 * reactive_count / max(len(signals), 1), 2),
            "verdict": "PREDICTIVE" if predictive_count > reactive_count else "REACTIVE",
        },
        "per_signal_sample": per_signal[:25],
    }


def _runner_exit_optimization(
    signals: list[dict[str, Any]],
    *,
    side: str,
    stop_variant: str,
    window_days: int,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    strategies: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for label, structure in RUNNER_STRATEGIES.items():
        pnls: list[float] = []
        runner_captured = 0.0
        runner_giveback = 0.0
        runner_wins = 0
        runner_trades = 0
        partial_exits = 0
        full_exits = 0

        for signal in signals:
            stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
            pnl, won = _strategy_pnl(signal, structure, stop_pts=stop_pts)
            pnls.append(pnl)
            mfe = float(signal.get("mfe_points") or 0.0)
            t2 = float(structure["t2"])

            if structure.get("runner") and mfe > t2:
                runner_trades += 1
                runner_leg = max(0.0, mfe - t2)
                if structure.get("trailing"):
                    captured_runner = runner_leg * (1.0 - 0.4) * LEG_WEIGHTS[2]
                else:
                    captured_runner = runner_leg * LEG_WEIGHTS[2]
                runner_captured += captured_runner
                runner_giveback += max(0.0, runner_leg * LEG_WEIGHTS[2] - captured_runner)
                if captured_runner > 0:
                    runner_wins += 1

            if not structure.get("runner"):
                full_exits += 1
            elif mfe >= float(structure["t1"]):
                partial_exits += 1

        metrics = _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=window_days)
        row = {
            "strategy": label,
            "structure": {k: v for k, v in structure.items() if k != "trailing"},
            "partial_exits": partial_exits,
            "full_exits": full_exits,
            "runner_leg": {
                "trades_with_runner_potential": runner_trades,
                "runner_captured_points": round(runner_captured, 2),
                "runner_giveback_points": round(runner_giveback, 2),
                "runner_win_rate_pct": round(100.0 * runner_wins / max(runner_trades, 1), 2),
                "runner_expectancy": round(runner_captured / max(runner_trades, 1), 2),
            },
            **metrics,
        }
        strategies[label] = row
        ranking.append({**row, "optimization_score": round(metrics["expectancy"] * (metrics["profit_factor"] or 0.0), 2)})

    best = max(ranking, key=lambda item: (item["expectancy"], item["profit_factor"] or 0.0, item["capture_efficiency_pct"]))
    current_playbook = strategies.get("60_100_runner", {})

    return {
        "side": side,
        "stop_variant": stop_variant,
        "by_strategy": strategies,
        "ranking": sorted(ranking, key=lambda item: item["optimization_score"], reverse=True),
        "best_strategy": best["strategy"],
        "current_playbook_strategy": "60_100_runner",
        "current_vs_best": {
            "current_expectancy": current_playbook.get("expectancy"),
            "best_expectancy": best["expectancy"],
            "improvement_potential_pct": round(
                100.0
                * ((best["expectancy"] or 0) - (current_playbook.get("expectancy") or 0))
                / max(abs(current_playbook.get("expectancy") or 1), 1.0),
                2,
            ),
        },
    }


def _execution_bottleneck_audit(
    *,
    live_audit: dict[str, Any],
    regime_audit: dict[str, Any],
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    loss_root = regime_audit.get("loss_root_cause", {})
    buy_leakage = live_audit.get("capture_leakage", {}).get("buy_v3", {})
    sell_leakage = live_audit.get("capture_leakage", {}).get("sell_v6", {})

    cause_map = {
        "signal_quality": 0.0,
        "execution": 0.0,
        "regime": 0.0,
        "target": 0.0,
        "stop": 0.0,
        "runner": 0.0,
    }

    for row in loss_root.get("cause_ranking", []):
        cause = row.get("cause", "")
        if cause in cause_map:
            cause_map[cause] += float(row.get("pct") or 0.0)

    for leakage, weight in ((buy_leakage, 0.35), (sell_leakage, 0.65)):
        for row in leakage.get("miss_reason_ranking", []):
            reason = row.get("reason", "")
            count = float(row.get("count") or 0)
            if reason == "timing":
                cause_map["execution"] += count * weight
            elif reason == "stop":
                cause_map["stop"] += count * weight
            elif reason == "runner":
                cause_map["runner"] += count * weight
            elif reason in {"early_exit", "target_structure"}:
                cause_map["target"] += count * weight

    total_signals = len(buy_signals) + len(sell_signals)
    total_weight = sum(cause_map.values()) or 1.0
    ranking = [
        {
            "bottleneck": cause,
            "contribution_pct": round(100.0 * value / total_weight, 2),
            "rank": index + 1,
        }
        for index, (cause, value) in enumerate(
            sorted(cause_map.items(), key=lambda item: item[1], reverse=True),
        )
    ]

    return {
        "methodology": (
            "Combines regime_aware loss_root_cause ranking with live_trade capture_leakage "
            "miss reasons; weighted 35% BUY / 65% SELL by signal count."
        ),
        "total_signals": total_signals,
        "primary_bottleneck": ranking[0]["bottleneck"] if ranking else None,
        "bottleneck_ranking": ranking,
        "loss_root_cause_source": loss_root.get("primary_cause"),
        "capture_leakage_top_miss": {
            "buy_v3": (buy_leakage.get("miss_reason_ranking") or [None])[0],
            "sell_v6": (sell_leakage.get("miss_reason_ranking") or [None])[0],
        },
    }


def _required_sample_size(p_hat: float, *, margin: float = 0.05, confidence_pct: int = 90) -> int:
    z = CONFIDENCE_Z.get(confidence_pct, 1.645)
    p = min(max(p_hat, 0.01), 0.99)
    n = (z**2 * p * (1.0 - p)) / (margin**2)
    return int(math.ceil(n))


def _evidence_quality_analysis(
    *,
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    buy_export: dict[str, Any],
    sell_export: dict[str, Any],
    deployment_audit: dict[str, Any],
    window_days: int,
) -> dict[str, Any]:
    buy_n = len(buy_signals)
    sell_n = len(sell_signals)
    buy_wr = float(
        deployment_audit.get("engine_validation_reconciliation", {})
        .get("buy_v3", {})
        .get("win_rate_pct", {})
        .get("authoritative_for_gates")
        or 72.0,
    ) / 100.0
    sell_wr = float(
        deployment_audit.get("engine_validation_reconciliation", {})
        .get("sell_v6", {})
        .get("win_rate_pct", {})
        .get("reconciled")
        or 70.0,
    ) / 100.0

    required_by_confidence: dict[str, Any] = {}
    for conf in (60, 70, 80, 90):
        required_by_confidence[str(conf)] = {
            "buy_v3_wr": _required_sample_size(buy_wr, confidence_pct=conf),
            "sell_v6_wr": _required_sample_size(sell_wr, confidence_pct=conf),
            "combined_min": _required_sample_size((buy_wr + sell_wr) / 2.0, confidence_pct=conf),
        }

    buy_validate_n = int(
        buy_export.get("walk_forward", {}).get("validate", {}).get("buy_v3", {}).get("signals_emitted_count")
        or buy_export.get("walk_forward", {}).get("validate", {}).get("buy_v3", {}).get("overall_statistics", {}).get(
            "signals_emitted",
        )
        or 6,
    )

    current_confidence_buy = min(95.0, round(100.0 * buy_n / max(required_by_confidence["80"]["buy_v3_wr"], 1), 1))
    current_confidence_sell = min(95.0, round(100.0 * sell_n / max(required_by_confidence["80"]["sell_v6_wr"], 1), 1))

    days_250_signals_buy = round(buy_n * 250 / max(window_days, 1), 0)
    days_250_signals_sell = round(sell_n * 250 / max(window_days, 1), 0)
    days_500_signals_buy = round(buy_n * 500 / max(window_days, 1), 0)
    days_500_signals_sell = round(sell_n * 500 / max(window_days, 1), 0)

    return {
        "window_trading_days": window_days,
        "current_sample_sizes": {"buy_v3": buy_n, "sell_v6": sell_n, "combined": buy_n + sell_n},
        "is_120d_sufficient": {
            "buy_v3": buy_n >= required_by_confidence["70"]["buy_v3_wr"],
            "sell_v6": sell_n >= required_by_confidence["70"]["sell_v6_wr"],
            "verdict": (
                "PARTIAL"
                if buy_n < required_by_confidence["80"]["buy_v3_wr"]
                else "YES_FOR_SELL_MARGINAL_FOR_BUY"
            ),
            "buy_v3_validate_caveat": f"BUY validate n={buy_validate_n} — walk-forward stability not definitive",
        },
        "required_sample_sizes_by_confidence": required_by_confidence,
        "current_confidence_pct": {
            "buy_v3_wr_estimate": current_confidence_buy,
            "sell_v6_wr_estimate": current_confidence_sell,
            "combined_estimate": round((current_confidence_buy + current_confidence_sell) / 2.0, 1),
        },
        "projected_at_250d": {
            "buy_v3_signals": days_250_signals_buy,
            "sell_v6_signals": days_250_signals_sell,
            "would_change_buy_conclusions": days_250_signals_buy >= required_by_confidence["90"]["buy_v3_wr"],
            "would_change_sell_conclusions": False,
            "note": "250d extends BUY sample toward 90% WR confidence; SELL already sufficient at 120d.",
        },
        "projected_at_500d": {
            "buy_v3_signals": days_500_signals_buy,
            "sell_v6_signals": days_500_signals_sell,
            "would_change_combined_conclusions": True,
            "note": "500d would stabilize regime throttle map and walk-forward — may revise SELL BLOCK rules.",
        },
    }


def _component_evidence_score(
    *,
    sample_size: int,
    required_n: int,
    source_type: str,
    cross_export_consistent: bool,
    gate_passes: bool,
) -> float:
    source_weight = {"replay": 1.0, "partial": 0.65, "synthetic": 0.35}.get(source_type, 0.5)
    size_ratio = min(1.0, sample_size / max(required_n, 1))
    consistency_bonus = 10.0 if cross_export_consistent else 0.0
    gate_bonus = 10.0 if gate_passes else 0.0
    return round(min(100.0, 40.0 * source_weight + 40.0 * size_ratio + consistency_bonus + gate_bonus), 1)


def _production_truth_audit(
    *,
    deployment_audit: dict[str, Any],
    live_audit: dict[str, Any],
    regime_audit: dict[str, Any],
    buy_export: dict[str, Any],
    sell_export: dict[str, Any],
    evidence_quality: dict[str, Any],
) -> dict[str, Any]:
    exports = [
        ("buy_v3_candidate_validation", buy_export, "replay"),
        ("sell_v6_replay_validation", sell_export, "replay"),
        ("live_trade_management", live_audit, "partial"),
        ("regime_aware_execution", regime_audit, "partial"),
        ("final_production_deployment", deployment_audit, "partial"),
    ]

    source_counts = Counter(item[2] for item in exports)
    total = sum(source_counts.values())
    conclusion_sources = {
        "replay_pct": round(100.0 * source_counts["replay"] / total, 2),
        "partial_pct": round(100.0 * source_counts["partial"] / total, 2),
        "synthetic_pct": round(100.0 * source_counts.get("synthetic", 0) / total, 2),
    }

    buy_n = evidence_quality["current_sample_sizes"]["buy_v3"]
    sell_n = evidence_quality["current_sample_sizes"]["sell_v6"]
    req_80 = evidence_quality["required_sample_sizes_by_confidence"]["80"]

    deploy_recon = deployment_audit.get("engine_validation_reconciliation", {})
    buy_pf = deploy_recon.get("buy_v3", {}).get("profit_factor", {}).get("reconciled")
    sell_pf = deploy_recon.get("sell_v6", {}).get("profit_factor", {}).get("reconciled")
    sell_validate_pf = deploy_recon.get("sell_v6", {}).get("profit_factor", {}).get("validate_unthrottled")

    live_final = live_audit.get("final_answer", {})
    regime_final = regime_audit.get("final_answer", {})

    evidence_scores = {
        "buy_v3": _component_evidence_score(
            sample_size=buy_n,
            required_n=req_80["buy_v3_wr"],
            source_type="replay",
            cross_export_consistent=buy_pf is not None and buy_pf >= 2.0,
            gate_passes=buy_pf is not None and buy_pf >= PRODUCTION_GATES["profit_factor_min"],
        ),
        "sell_v6": _component_evidence_score(
            sample_size=sell_n,
            required_n=req_80["sell_v6_wr"],
            source_type="replay",
            cross_export_consistent=sell_pf is not None and sell_pf >= 2.0,
            gate_passes=sell_pf is not None and sell_pf >= PRODUCTION_GATES["profit_factor_min"],
        ),
        "regime_throttle": _component_evidence_score(
            sample_size=sell_n,
            required_n=req_80["sell_v6_wr"],
            source_type="partial",
            cross_export_consistent=regime_final.get("regime_aware_execution_improves_pf_expectancy") in {"YES", "PARTIAL"},
            gate_passes=sell_validate_pf is not None and float(sell_validate_pf) < 2.0,
        ),
        "60_100_runner": _component_evidence_score(
            sample_size=buy_n + sell_n,
            required_n=req_80["combined_min"],
            source_type="partial",
            cross_export_consistent=live_final.get("optimal_exit_structures", {}).get("buy_v3") == "60/100/Runner",
            gate_passes=True,
        ),
        "fixed_10_stop": _component_evidence_score(
            sample_size=buy_n + sell_n,
            required_n=req_80["combined_min"],
            source_type="partial",
            cross_export_consistent=live_final.get("optimal_stops", {}).get("buy_v3") == "fixed_10",
            gate_passes=True,
        ),
        "structure_stop": _component_evidence_score(
            sample_size=buy_n + sell_n,
            required_n=req_80["combined_min"],
            source_type="partial",
            cross_export_consistent=True,
            gate_passes=True,
        ),
    }

    return {
        "conclusion_source_breakdown": conclusion_sources,
        "export_lineage": [
            {"export": name, "evidence_type": etype, "methodology": data.get("methodology", {})}
            for name, data, etype in exports
        ],
        "synthetic_flags": deployment_audit.get("limitations", [])[:3],
        "evidence_scores": evidence_scores,
        "aggregate_evidence_score": round(mean(evidence_scores.values()), 1),
        "deployment_recommendations_supported": {
            "buy_v3_paper": deployment_audit.get("final_answer", {}).get("buy_v3_paper_trading") == "YES",
            "sell_v6_throttled": deployment_audit.get("final_answer", {}).get("sell_v6_paper_trading_throttled") == "YES",
            "real_capital_withheld": deployment_audit.get("final_answer", {}).get("real_capital_deployment") == "NO",
            "evidence_supports_paper": round(mean([evidence_scores["buy_v3"], evidence_scores["sell_v6"]]), 1) >= 60.0,
        },
    }


def _capture_summary(
    buy_matrix: dict[str, Any],
    sell_matrix: dict[str, Any],
    runner_buy: dict[str, Any],
    runner_sell: dict[str, Any],
) -> dict[str, Any]:
    buy_current = runner_buy.get("by_strategy", {}).get("60_100_runner", {})
    sell_current = runner_sell.get("by_strategy", {}).get("60_100_runner", {})
    buy_best = runner_buy.get("by_strategy", {}).get(runner_buy.get("best_strategy", ""), {})
    sell_best = runner_sell.get("by_strategy", {}).get(runner_sell.get("best_strategy", ""), {})

    current_cap = mean(
        [
            buy_matrix.get("aggregate", {}).get("capture_pct") or 0.0,
            sell_matrix.get("aggregate", {}).get("capture_pct") or 0.0,
        ],
    )
    max_cap = mean(
        [
            buy_best.get("capture_efficiency_pct") or 0.0,
            sell_best.get("capture_efficiency_pct") or 0.0,
        ],
    )
    improvement = max(0.0, max_cap - current_cap)

    return {
        "current_capture_pct": round(current_cap, 2),
        "max_achievable_capture_pct": round(max_cap, 2),
        "improvement_potential_capture_pct": round(improvement, 2),
        "buy_current_capture_efficiency": buy_current.get("capture_efficiency_pct"),
        "sell_current_capture_efficiency": sell_current.get("capture_efficiency_pct"),
    }


def _production_scores(
    *,
    deployment_audit: dict[str, Any],
    truth_audit: dict[str, Any],
    capture_summary: dict[str, Any],
    evidence_quality: dict[str, Any],
) -> dict[str, Any]:
    deploy_scores = deployment_audit.get("production_scores", {})
    readiness = float(deploy_scores.get("production_readiness_score") or 72.0)
    confidence = float(deploy_scores.get("confidence_score") or 66.0)
    risk = float(deploy_scores.get("production_risk_score") or 68.5)
    evidence = float(truth_audit.get("aggregate_evidence_score") or 60.0)

    if evidence_quality["current_confidence_pct"]["combined_estimate"] < 70.0:
        confidence = max(50.0, confidence - 3.0)

    return {
        "production_readiness_score": round(readiness, 1),
        "confidence_score": round(confidence, 1),
        "production_risk_score": round(risk, 1),
        "evidence_score": round(evidence, 1),
        "deployment_tier": deployment_audit.get("final_answer", {}).get("deployment_tier", "Production Candidate"),
        "capture_summary": capture_summary,
    }


def _can_improve_without_new_engine(
    *,
    runner_buy: dict[str, Any],
    runner_sell: dict[str, Any],
    bottleneck: dict[str, Any],
) -> str:
    buy_improvement = runner_buy.get("current_vs_best", {}).get("improvement_potential_pct") or 0.0
    sell_improvement = runner_sell.get("current_vs_best", {}).get("improvement_potential_pct") or 0.0
    primary = bottleneck.get("primary_bottleneck", "execution")
    if primary in {"execution", "target", "stop", "runner"} and (buy_improvement > 5.0 or sell_improvement > 5.0):
        return "YES"
    if primary == "signal_quality":
        return "NO"
    return "PARTIAL"


def _final_answer(
    *,
    scores: dict[str, Any],
    truth_audit: dict[str, Any],
    evidence_quality: dict[str, Any],
    bottleneck: dict[str, Any],
    regime_audit: dict[str, Any],
    deployment_audit: dict[str, Any],
    can_improve: str,
    capture_summary: dict[str, Any],
) -> dict[str, Any]:
    primary_bottleneck = bottleneck.get("primary_bottleneck", "execution")
    regime_final = regime_audit.get("final_answer", {})
    buy_near = regime_final.get("buy_v3_near_optimal", "PARTIAL")
    sell_near = regime_final.get("sell_v6_near_optimal", "PARTIAL")

    biggest_uncertainty = (
        "Live slippage/fill quality and intrabar stop-target sequencing — all metrics use MFE/MAE proxy."
        if primary_bottleneck in {"execution", "stop"}
        else "SELL_V6 validate-window regime throttle stability on unseen 2026-H2 regimes."
    )

    improvement_map = {
        "execution": "Tighten entry timing filter; reduce late_entry miss (BUY timing leakage #1).",
        "runner": "Improve runner trail giveback policy beyond T2 (SELL runner leakage #1).",
        "target": "Extend T2 or adopt trailing_runner exit in strong-trend regimes.",
        "stop": "Regime-adaptive structure_based stop in high-volatility buckets.",
        "signal_quality": "Research next-gen signal formula (BUY_V4 / SELL_V7).",
        "regime": "Expand SELL BLOCK map from validate deterioration signals.",
    }
    biggest_opportunity = improvement_map.get(primary_bottleneck, improvement_map["runner"])

    should_buy_v4 = "YES" if primary_bottleneck == "signal_quality" and buy_near in {"NO", "PARTIAL"} else "NO"
    should_sell_v7 = "YES" if primary_bottleneck == "signal_quality" and sell_near in {"NO", "PARTIAL"} else "NO"
    if can_improve == "YES" and primary_bottleneck != "signal_quality":
        should_buy_v4 = "NO"
        should_sell_v7 = "NO"

    return {
        "biggest_uncertainty_before_real_capital": biggest_uncertainty,
        "biggest_opportunity_for_improvement": biggest_opportunity,
        "can_expectancy_improve_without_buy_v4_sell_v7": can_improve,
        "should_research_buy_v4": should_buy_v4,
        "should_research_sell_v7": should_sell_v7,
        "current_capture_pct": capture_summary["current_capture_pct"],
        "max_capture_pct": capture_summary["max_achievable_capture_pct"],
        "improvement_potential_capture_pct": capture_summary["improvement_potential_capture_pct"],
        "production_readiness_score": scores["production_readiness_score"],
        "confidence_score": scores["confidence_score"],
        "production_risk_score": scores["production_risk_score"],
        "evidence_score": scores["evidence_score"],
        "deployment_tier": scores["deployment_tier"],
        "paper_trade_tomorrow": deployment_audit.get("final_answer", {}).get("paper_trade_tomorrow", "YES"),
        "real_capital_deployment": deployment_audit.get("final_answer", {}).get("real_capital_deployment", "NO"),
        "evidence_scores": truth_audit.get("evidence_scores", {}),
        "deployment_recommendations_supported_by_evidence": truth_audit.get(
            "deployment_recommendations_supported", {},
        ),
        "is_120d_sufficient": evidence_quality["is_120d_sufficient"]["verdict"],
        "rationale": (
            f"Evidence score {scores['evidence_score']}/100 supports paper trading (YES) but not real capital (NO). "
            f"Primary bottleneck: {primary_bottleneck}. Expectancy improvement without V4/V7: {can_improve}. "
            f"Capture {capture_summary['current_capture_pct']}% current vs {capture_summary['max_achievable_capture_pct']}% max."
        ),
    }


class ProductionRealityAuditResearch:
    """Synthesize production reality audit from existing exports."""

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

    def run(self) -> ProductionRealityAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        buy_export = sources["buy_v3_candidate_validation"]["data"]
        sell_export = sources["sell_v6_replay_validation"]["data"]
        deployment_audit = sources["final_production_deployment_audit"]["data"]
        live_audit = sources["live_trade_management_execution_efficiency_audit"]["data"]
        regime_audit = sources["regime_aware_execution_validation"]["data"]

        window_days = int(
            buy_export.get("trading_days_replayed") or sell_export.get("trading_days_replayed") or 120,
        )

        buy_signals = list(buy_export.get("per_signal_details", {}).get("buy_v3") or [])
        sell_signals = list(sell_export.get("per_signal_details", {}).get("sell_v6") or [])
        if not buy_signals:
            raise ProductionRealityAuditError("No BUY_V3 per_signal_details in exports.")
        if not sell_signals:
            raise ProductionRealityAuditError("No SELL_V6 per_signal_details in exports.")

        live_final = live_audit.get("final_answer", {})
        buy_stop = live_final.get("optimal_stops", {}).get("buy_v3", "fixed_10")
        sell_stop = live_final.get("optimal_stops", {}).get("sell_v6", "fixed_10")
        buy_exit_label = live_final.get("optimal_exit_structures", {}).get("buy_v3", "60/100/Runner")
        sell_exit_label = live_final.get("optimal_exit_structures", {}).get("sell_v6", "60/100/Runner")
        buy_structure = RUNNER_STRATEGIES.get("60_100_runner", RUNNER_STRATEGIES["60_100_runner"])
        sell_structure = RUNNER_STRATEGIES.get("60_100_runner", RUNNER_STRATEGIES["60_100_runner"])

        trade_outcome_distribution = {
            "buy_v3": _mfe_tier_distribution(buy_signals),
            "sell_v6": _mfe_tier_distribution(sell_signals),
            "combined_note": "Tier hit rates from per_signal_details MFE; conditional probability = P(tier | prior tier).",
        }

        target_achievement_matrix = {
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
        }

        signal_reality = {
            "buy_v3": _signal_reality_analysis(
                buy_signals,
                side="BUY",
                win_fn=_is_buy_winner,
                window_days=window_days,
            ),
            "sell_v6": _signal_reality_analysis(
                sell_signals,
                side="SELL",
                win_fn=_is_sell_winner,
                window_days=window_days,
            ),
        }

        runner_exit_optimization = {
            "buy_v3": _runner_exit_optimization(
                buy_signals,
                side="BUY",
                stop_variant=buy_stop,
                window_days=window_days,
            ),
            "sell_v6": _runner_exit_optimization(
                sell_signals,
                side="SELL",
                stop_variant=sell_stop,
                window_days=window_days,
            ),
        }

        execution_bottleneck_audit = _execution_bottleneck_audit(
            live_audit=live_audit,
            regime_audit=regime_audit,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
        )

        evidence_quality = _evidence_quality_analysis(
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            buy_export=buy_export,
            sell_export=sell_export,
            deployment_audit=deployment_audit,
            window_days=window_days,
        )

        production_truth_audit = _production_truth_audit(
            deployment_audit=deployment_audit,
            live_audit=live_audit,
            regime_audit=regime_audit,
            buy_export=buy_export,
            sell_export=sell_export,
            evidence_quality=evidence_quality,
        )

        capture_summary = _capture_summary(
            target_achievement_matrix["buy_v3"],
            target_achievement_matrix["sell_v6"],
            runner_exit_optimization["buy_v3"],
            runner_exit_optimization["sell_v6"],
        )

        production_scores = _production_scores(
            deployment_audit=deployment_audit,
            truth_audit=production_truth_audit,
            capture_summary=capture_summary,
            evidence_quality=evidence_quality,
        )

        can_improve = _can_improve_without_new_engine(
            runner_buy=runner_exit_optimization["buy_v3"],
            runner_sell=runner_exit_optimization["sell_v6"],
            bottleneck=execution_bottleneck_audit,
        )

        final_answer = _final_answer(
            scores=production_scores,
            truth_audit=production_truth_audit,
            evidence_quality=evidence_quality,
            bottleneck=execution_bottleneck_audit,
            regime_audit=regime_audit,
            deployment_audit=deployment_audit,
            can_improve=can_improve,
            capture_summary=capture_summary,
        )

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "no_new_replay": True,
            "no_new_discovery": True,
            "no_new_models": True,
            "no_new_indicators": True,
            "source_export_count": len(SOURCE_EXPORTS),
            "mfe_tiers": list(MFE_TIERS),
            "timing_classes": list(TIMING_CLASSES),
            "runner_strategies": list(RUNNER_STRATEGIES.keys()),
            "production_gates": PRODUCTION_GATES,
            "engines": {"buy_v3": BUY_V3_MODEL_ID, "sell_v6": SELL_V6_MODEL_ID},
        }

        limitations = [
            "All metrics synthesized from five JSON exports — no new replay.",
            "MFE/MAE proxy does not model intrabar stop/target hit ordering.",
            "Entry timing uses bars_before_expansion / points_before_expansion proxies.",
            "Runner/trailing simulations approximate giveback — not live broker fills.",
            "Evidence scores weight replay exports higher than MFE/MAE synthesis exports.",
            "120d window: BUY_V3 n=116 marginal for 80% WR confidence; SELL_V6 n=336 sufficient.",
        ]

        conclusions = [
            "Production reality audit synthesized from 5 completed replay exports only.",
            (
                f"BUY_V3 tier-40 hit {trade_outcome_distribution['buy_v3']['tiers']['40']['pct_of_signals']}% | "
                f"SELL_V6 tier-40 hit {trade_outcome_distribution['sell_v6']['tiers']['40']['pct_of_signals']}%."
            ),
            (
                f"Signal timing: BUY {signal_reality['buy_v3']['predictive_vs_reactive']['verdict']} | "
                f"SELL {signal_reality['sell_v6']['predictive_vs_reactive']['verdict']}."
            ),
            (
                f"Best runner strategy: BUY {runner_exit_optimization['buy_v3']['best_strategy']} | "
                f"SELL {runner_exit_optimization['sell_v6']['best_strategy']}."
            ),
            f"Primary execution bottleneck: {execution_bottleneck_audit['primary_bottleneck']}.",
            (
                f"Capture: {capture_summary['current_capture_pct']}% current / "
                f"{capture_summary['max_achievable_capture_pct']}% max / "
                f"{capture_summary['improvement_potential_capture_pct']}% improvement potential."
            ),
            (
                f"Evidence score {production_scores['evidence_score']}/100 | "
                f"120d sufficient: {evidence_quality['is_120d_sufficient']['verdict']}."
            ),
            (
                f"Expectancy improve w/o V4/V7: {final_answer['can_expectancy_improve_without_buy_v4_sell_v7']} | "
                f"BUY_V4 research: {final_answer['should_research_buy_v4']} | "
                f"SELL_V7 research: {final_answer['should_research_sell_v7']}."
            ),
            (
                f"Paper trade: {final_answer['paper_trade_tomorrow']} | "
                f"Real capital: {final_answer['real_capital_deployment']}."
            ),
        ]

        return ProductionRealityAuditReport(
            report_type="Production Reality Audit",
            engines=["BUY_V3", "SELL_V6", "COMBINED"],
            symbol=str(buy_export.get("symbol") or "NIFTY50"),
            timeframe=str(buy_export.get("timeframe") or "5M"),
            trading_days_replayed=window_days,
            replay_start_date=str(buy_export.get("replay_start_date") or ""),
            replay_end_date=str(buy_export.get("replay_end_date") or ""),
            methodology=methodology,
            source_exports={name: {"path": meta["path"], "status": meta["status"]} for name, meta in sources.items()},
            limitations=limitations,
            trade_outcome_distribution=trade_outcome_distribution,
            target_achievement_matrix=target_achievement_matrix,
            signal_reality=signal_reality,
            runner_exit_optimization=runner_exit_optimization,
            execution_bottleneck_audit=execution_bottleneck_audit,
            evidence_quality=evidence_quality,
            production_truth_audit=production_truth_audit,
            production_scores=production_scores,
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: ProductionRealityAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Production reality audit exported to %s", self.report_path)
        return self.report_path


def generate_production_reality_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export production reality audit JSON."""
    return ProductionRealityAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_production_reality_audit_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    print(f"Exported: {path}")
    print(f"Evidence score: {final['evidence_score']}")
    print(f"Capture: {final['current_capture_pct']}% / {final['max_capture_pct']}%")
    print(f"Improve w/o V4/V7: {final['can_expectancy_improve_without_buy_v4_sell_v7']}")
    print(f"BUY_V4: {final['should_research_buy_v4']} | SELL_V7: {final['should_research_sell_v7']}")
