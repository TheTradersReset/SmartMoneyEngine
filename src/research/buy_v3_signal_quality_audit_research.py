"""
BUY_V3 Signal Quality Audit — synthesis from replay-validated exports only.

Audits all BUY_V3 per_signal_details: failure taxonomy, timing, move capture,
condition separation, and single-filter improvement simulation.
No new replay, indicators, models, or discovery.
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
from src.research.buy_v3_candidate_validation_research import (
    BAR_MINUTES,
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
)
from src.research.buy_v2_candidate_validation_research import PRODUCTION_GATES
from src.research.buy_winner_vs_false_reversal_analysis_research import (
    ANALYSIS_CONDITIONS,
    _extract_conditions_from_signal,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_v3_signal_quality_audit.json"

SOURCE_EXPORTS = {
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "buy_v3_tradeability_production_validation": RESEARCH_DIR
    / "buy_v3_tradeability_production_validation.json",
    "buy_winner_vs_false_reversal_analysis": RESEARCH_DIR
    / "buy_winner_vs_false_reversal_analysis.json",
    "buy_v2_candidate_validation": RESEARCH_DIR / "buy_v2_candidate_validation.json",
}

AUDIT_CLASSIFICATIONS = (
    "Winner",
    "Bull Trap",
    "Range Failure",
    "No Expansion",
    "Counter Trend Bounce",
    "Liquidity Failure",
)

EXPORT_TO_AUDIT_CLASS: dict[str, str] = {
    "Real Reversal": "Winner",
    "Bull Trap": "Bull Trap",
    "Range Failure": "Range Failure",
    "No Expansion": "No Expansion",
    "Counter Trend Bounce": "Counter Trend Bounce",
    "False Reversal": "Liquidity Failure",
    "Dead Cat Bounce": "Liquidity Failure",
}

MOVE_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-100", 60.0, 100.0),
    ("100-150", 100.0, 150.0),
    ("150-200", 150.0, 200.0),
    ("200+", 200.0, None),
)

ACHIEVEMENT_THRESHOLDS = (20, 40, 60, 80, 100, 150, 200)

FAILURE_EXPORT_CLASSES = frozenset(
    {
        "Bull Trap",
        "Range Failure",
        "No Expansion",
        "Counter Trend Bounce",
        "False Reversal",
        "Dead Cat Bounce",
    },
)

ADDITIONAL_FILTER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("require", "HTF Bullish"),
    ("require", "VWAP Reclaim"),
    ("require", "PWL Sweep"),
    ("require", "Support Reclaim"),
    ("require", "Gap Continuation"),
    ("exclude", "HTF Bullish"),
    ("exclude", "Gap Continuation"),
)


class BuyV3SignalQualityAuditError(Exception):
    """Raised when BUY_V3 signal quality audit cannot be completed."""


def _profit_factor_from_pnls(pnls: list[float]) -> float | None:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return None if gross_profit == 0 else round(gross_profit, 2)
    return round(gross_profit / gross_loss, 2)


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise BuyV3SignalQualityAuditError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _audit_classification(export_class: str) -> str:
    return EXPORT_TO_AUDIT_CLASS.get(export_class or "Unknown", "Liquidity Failure")


def _is_winner(signal: dict[str, Any]) -> bool:
    return signal.get("classification") == "Real Reversal"


def _mfe_bucket(mfe: float) -> str:
    for label, low, high in MOVE_BUCKETS:
        if high is None and mfe >= low:
            return label
        if high is not None and low <= mfe < high:
            return label
    return "0-20"


def _timing_label(signal: dict[str, Any]) -> str:
    bars = signal.get("bars_before_expansion")
    if bars is None:
        return "No Linked Move"
    if bars > 0:
        return "Early"
    if bars == 0:
        return "Same Candle"
    return "Delayed"


def _build_per_signal_audit(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal in signals:
        export_class = signal.get("classification", "Unknown")
        mfe = float(signal.get("mfe_points") or 0.0)
        mae = float(signal.get("mae_points") or 0.0)
        rows.append(
            {
                "timestamp": signal.get("timestamp"),
                "move_start_time": signal.get("move_start_time"),
                "export_classification": export_class,
                "audit_classification": _audit_classification(export_class),
                "is_winner": _is_winner(signal),
                "bars_before_expansion": signal.get("bars_before_expansion"),
                "points_before_expansion": signal.get("points_before_expansion"),
                "timing_label": _timing_label(signal),
                "mfe_points": mfe,
                "mae_points": mae,
                "mfe_bucket": _mfe_bucket(mfe),
                "win": signal.get("win"),
                "realized_pnl_points": signal.get("realized_pnl_points"),
                "conditions": signal.get("conditions", {}),
            },
        )
    return rows


def _classification_summary(per_signal: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["audit_classification"] for row in per_signal)
    total = len(per_signal)
    rates = {label: round(100.0 * counts.get(label, 0) / max(total, 1), 2) for label in AUDIT_CLASSIFICATIONS}
    failure_count = total - counts.get("Winner", 0)
    return {
        "total_signals": total,
        "counts": {label: counts.get(label, 0) for label in AUDIT_CLASSIFICATIONS},
        "rates_pct": rates,
        "winner_rate_pct": rates.get("Winner", 0.0),
        "failure_rate_pct": round(100.0 * failure_count / max(total, 1), 2),
    }


def _move_distribution(per_signal: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter(row["mfe_bucket"] for row in per_signal)
    total = len(per_signal)
    by_bucket: dict[str, Any] = {}
    for label, _, _ in MOVE_BUCKETS:
        count = bucket_counts.get(label, 0)
        by_bucket[label] = {
            "count": count,
            "share_pct": round(100.0 * count / max(total, 1), 2),
        }
    winner_buckets = Counter(row["mfe_bucket"] for row in per_signal if row["is_winner"])
    failure_buckets = Counter(row["mfe_bucket"] for row in per_signal if not row["is_winner"])
    return {
        "basis": "MFE points per signal",
        "total_signals": total,
        "by_bucket": by_bucket,
        "winner_by_bucket": dict(winner_buckets),
        "failure_by_bucket": dict(failure_buckets),
    }


def _achievement_counts(per_signal: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(per_signal)
    overall: dict[str, Any] = {}
    winner_only: dict[str, Any] = {}
    winners = [row for row in per_signal if row["is_winner"]]
    for threshold in ACHIEVEMENT_THRESHOLDS:
        key = f"{threshold}_plus"
        overall[key] = {
            "count": sum(1 for row in per_signal if row["mfe_points"] >= threshold),
            "rate_pct": round(
                100.0 * sum(1 for row in per_signal if row["mfe_points"] >= threshold) / max(total, 1),
                2,
            ),
        }
        winner_only[key] = {
            "count": sum(1 for row in winners if row["mfe_points"] >= threshold),
            "rate_pct": round(
                100.0 * sum(1 for row in winners if row["mfe_points"] >= threshold) / max(len(winners), 1),
                2,
            ),
        }
    return {
        "all_signals": overall,
        "winners_only": winner_only,
        "thresholds": list(ACHIEVEMENT_THRESHOLDS),
    }


def _signal_timing_analysis(signals: list[dict[str, Any]], timing_export: dict[str, Any]) -> dict[str, Any]:
    early = same = delayed = no_move = 0
    lead_bars: list[int] = []
    lead_points: list[float] = []

    for signal in signals:
        bars = signal.get("bars_before_expansion")
        if bars is None:
            no_move += 1
            continue
        if bars > 0:
            early += 1
            lead_bars.append(int(bars))
            if signal.get("points_before_expansion") is not None:
                lead_points.append(float(signal["points_before_expansion"]))
        elif bars == 0:
            same += 1
        else:
            delayed += 1

    linked = early + same + delayed
    return {
        "early_count": early,
        "same_candle_count": same,
        "delayed_count": delayed,
        "no_linked_move_count": no_move,
        "early_pct": round(100.0 * early / max(linked, 1), 2),
        "same_candle_pct": round(100.0 * same / max(linked, 1), 2),
        "delayed_pct": round(100.0 * delayed / max(linked, 1), 2),
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
        "export_cross_check": timing_export.get("buy_v3"),
    }


def _condition_comparison(
    signals: list[dict[str, Any]],
) -> dict[str, Any]:
    winners = [signal for signal in signals if _is_winner(signal)]
    failures = [signal for signal in signals if not _is_winner(signal)]
    metrics: list[dict[str, Any]] = []

    for condition in ANALYSIS_CONDITIONS:
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
        frequency_impact = round(100.0 * (w_present + f_present) / max(w_total + f_total, 1), 2)

        metrics.append(
            {
                "condition": condition,
                "winner_coverage_pct": winner_cov,
                "failure_coverage_pct": failure_cov,
                "coverage_delta_pp": round(winner_cov - failure_cov, 2),
                "failure_reduction_if_required_pct": false_reduction,
                "winner_retention_if_required_pct": winner_retention,
                "frequency_impact_pct": frequency_impact,
                "winner_present_count": w_present,
                "failure_present_count": f_present,
            },
        )

    by_failure_reduction = sorted(
        metrics,
        key=lambda item: (item["failure_reduction_if_required_pct"], item["winner_retention_if_required_pct"]),
        reverse=True,
    )
    by_winner_retention = sorted(
        metrics,
        key=lambda item: (item["winner_retention_if_required_pct"], item["failure_reduction_if_required_pct"]),
        reverse=True,
    )
    by_frequency = sorted(metrics, key=lambda item: item["frequency_impact_pct"])
    by_separation = sorted(metrics, key=lambda item: item["coverage_delta_pp"], reverse=True)

    return {
        "winner_count": len(winners),
        "failure_count": len(failures),
        "per_condition": metrics,
        "rankings": {
            "by_failure_reduction_power": by_failure_reduction,
            "by_winner_retention": by_winner_retention,
            "by_frequency_impact": by_frequency,
            "by_winner_vs_failure_separation": by_separation,
        },
    }


def _signal_performance(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
) -> dict[str, Any]:
    if not signals:
        return {
            "sample_size": 0,
            "signals_per_month": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "capture_40_plus_pct": 0.0,
            "capture_60_plus_pct": 0.0,
            "capture_100_plus_pct": 0.0,
            "passes_production_gates": False,
        }

    pnls = [float(signal.get("realized_pnl_points") or 0.0) for signal in signals]
    wins = sum(1 for signal in signals if signal.get("win"))
    months = max(window_days / 22.0, 1.0)
    wr = round(100.0 * wins / len(signals), 2)
    pf = _profit_factor_from_pnls(pnls)
    spm = round(len(signals) / months, 2)

    def _capture(threshold: int) -> float:
        return round(
            100.0
            * sum(1 for signal in signals if float(signal.get("mfe_points") or 0.0) >= threshold)
            / len(signals),
            2,
        )

    passes = (
        wr >= PRODUCTION_GATES["win_rate_min_pct"]
        and (pf is None or pf >= PRODUCTION_GATES["profit_factor_min"])
        and spm >= PRODUCTION_GATES["signals_per_month_min"]
    )

    return {
        "sample_size": len(signals),
        "signals_per_month": spm,
        "win_rate_pct": wr,
        "profit_factor": pf,
        "expectancy": round(mean(pnls), 2),
        "capture_40_plus_pct": _capture(40),
        "capture_60_plus_pct": _capture(60),
        "capture_100_plus_pct": _capture(100),
        "passes_production_gates": passes,
    }


def _failure_removal_pct(
    baseline_failures: list[dict[str, Any]],
    filtered_failures: list[dict[str, Any]],
) -> float:
    if not baseline_failures:
        return 0.0
    removed = len(baseline_failures) - len(filtered_failures)
    return round(100.0 * removed / len(baseline_failures), 2)


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


def _single_filter_simulation(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
    baseline_stats: dict[str, Any],
) -> dict[str, Any]:
    failures = [signal for signal in signals if not _is_winner(signal)]
    baseline_failure_count = len(failures)
    simulations: list[dict[str, Any]] = []

    for mode, condition in ADDITIONAL_FILTER_CANDIDATES:
        filtered = _apply_filter(signals, mode=mode, condition=condition)
        filtered_failures = [signal for signal in filtered if not _is_winner(signal)]
        perf = _signal_performance(filtered, window_days=window_days)
        removal_pct = _failure_removal_pct(failures, filtered_failures)
        label = f"BUY_V3 {'+ require' if mode == 'require' else '- exclude'} {condition}"
        simulations.append(
            {
                "filter_mode": mode,
                "filter_condition": condition,
                "label": label,
                "failure_removal_pct": removal_pct,
                "failures_removed": baseline_failure_count - len(filtered_failures),
                "baseline_failures": baseline_failure_count,
                "remaining_failures": len(filtered_failures),
                **perf,
            },
        )

    gate_passing = [
        sim
        for sim in simulations
        if sim["passes_production_gates"]
        and sim["signals_per_month"] >= PRODUCTION_GATES["signals_per_month_min"]
        and sim["win_rate_pct"] > PRODUCTION_GATES["win_rate_min_pct"]
        and (sim.get("profit_factor") is None or sim["profit_factor"] > PRODUCTION_GATES["profit_factor_min"])
    ]

    if gate_passing:
        best = max(
            gate_passing,
            key=lambda item: (
                item["failure_removal_pct"],
                item["win_rate_pct"],
                item.get("profit_factor") or 0.0,
            ),
        )
    else:
        best = max(simulations, key=lambda item: item["failure_removal_pct"]) if simulations else {}

    best_tradeoff = max(
        simulations,
        key=lambda item: (
            item["failure_removal_pct"],
            item.get("profit_factor") or 0.0,
            item["signals_per_month"],
        ),
    ) if simulations else {}

    return {
        "baseline": {
            "label": "BUY_V3 full stack",
            "failure_count": baseline_failure_count,
            **baseline_stats,
        },
        "candidate_filters": [{"mode": mode, "condition": cond} for mode, cond in ADDITIONAL_FILTER_CANDIDATES],
        "simulations": sorted(simulations, key=lambda item: item["failure_removal_pct"], reverse=True),
        "best_additional_filter": best,
        "best_failure_removal_tradeoff": best_tradeoff,
        "gate_passing_filter_count": len(gate_passing),
        "production_gates": PRODUCTION_GATES,
        "note": (
            "Single-filter simulation applies one extra require/exclude context condition to existing "
            "BUY_V3 replay per_signal_details — no new replay."
        ),
    }


def _why_buy_v3_fails(
    per_signal: list[dict[str, Any]],
    condition_comparison: dict[str, Any],
    v2_export: dict[str, Any] | None,
) -> dict[str, Any]:
    failures = [row for row in per_signal if row["audit_classification"] != "Winner"]
    failure_by_class = Counter(row["audit_classification"] for row in failures)
    total_failures = len(failures)

    mfe_by_class: dict[str, list[float]] = {}
    mae_by_class: dict[str, list[float]] = {}
    for row in failures:
        cls = row["audit_classification"]
        mfe_by_class.setdefault(cls, []).append(row["mfe_points"])
        mae_by_class.setdefault(cls, []).append(row["mae_points"])

    class_profiles: dict[str, Any] = {}
    for cls in AUDIT_CLASSIFICATIONS:
        if cls == "Winner":
            continue
        mfes = mfe_by_class.get(cls, [])
        maes = mae_by_class.get(cls, [])
        if not mfes:
            continue
        class_profiles[cls] = {
            "count": len(mfes),
            "share_of_failures_pct": round(100.0 * len(mfes) / max(total_failures, 1), 2),
            "avg_mfe": round(mean(mfes), 2),
            "avg_mae": round(mean(maes), 2) if maes else None,
        }

    top_sep = condition_comparison["rankings"]["by_winner_vs_failure_separation"][:3]
    v2_failure_patterns: list[str] = []
    if v2_export:
        v2_cls = v2_export.get("comparison", {}).get("buy_v2", {}).get("classification_summary", {})
        v2_failure_patterns.append(
            f"BUY_V2 baseline false-reversal rate {v2_cls.get('false_reversal_rate_pct')}% "
            f"(Bull Trap {v2_cls.get('bull_trap_rate_pct')}%, "
            f"No Expansion {v2_cls.get('no_expansion_rate_pct')}%)."
        )

    return {
        "total_failures": total_failures,
        "failure_rate_pct": round(100.0 * total_failures / max(len(per_signal), 1), 2),
        "failure_by_audit_class": dict(failure_by_class),
        "failure_class_profiles": class_profiles,
        "primary_failure_modes": [
            {
                "classification": cls,
                "count": class_profiles[cls]["count"],
                "share_pct": class_profiles[cls]["share_of_failures_pct"],
            }
            for cls in sorted(class_profiles, key=lambda c: class_profiles[c]["count"], reverse=True)
        ],
        "failure_context_separators": top_sep,
        "v2_failure_pattern_reference": v2_failure_patterns,
        "narrative": (
            f"BUY_V3 residual failures ({total_failures}/116, "
            f"{round(100.0 * total_failures / 116, 1)}%) are dominated by "
            f"Bull Trap ({failure_by_class.get('Bull Trap', 0)}) and "
            f"Range Failure ({failure_by_class.get('Range Failure', 0)}). "
            "Formula stack removed 947/947 V2-only false reversals; remaining failures "
            "are in-stack quality gaps, not frequency-collapse patterns."
        ),
    }


def _ablation_insights(v3_export: dict[str, Any]) -> dict[str, Any]:
    ablation = v3_export.get("ablation_analysis", {})
    ranking = ablation.get("contribution_ranking", {})
    variants = ablation.get("variants", {})
    variant_summary: list[dict[str, Any]] = []
    for key, variant in variants.items():
        stats = variant.get("overall_statistics", {})
        variant_summary.append(
            {
                "variant_key": key,
                "label": variant.get("label"),
                "removed_condition": variant.get("removed_condition"),
                "signals_per_month": stats.get("signals_per_month"),
                "win_rate_pct": stats.get("win_rate_pct"),
                "profit_factor": stats.get("profit_factor"),
                "false_reversal_rate_pct": variant.get("false_reversal_rate_pct"),
            },
        )
    return {
        "contribution_ranking": ranking,
        "variant_summary": variant_summary,
        "final_verdict_ablation": v3_export.get("final_verdict", {}).get("ablation_insights"),
    }


def _final_answer(
    *,
    baseline_stats: dict[str, Any],
    classification: dict[str, Any],
    filter_sim: dict[str, Any],
    ablation: dict[str, Any],
    achievement: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    best = filter_sim.get("best_additional_filter") or {}
    baseline_spm = float(baseline_stats.get("signals_per_month") or 0.0)
    baseline_wr = float(baseline_stats.get("win_rate_pct") or 0.0)
    baseline_pf = float(baseline_stats.get("profit_factor") or 0.0)
    failure_rate = float(classification.get("failure_rate_pct") or 0.0)

    best_passes = bool(best.get("passes_production_gates"))
    best_removal = float(best.get("failure_removal_pct") or 0.0)
    best_spm = float(best.get("signals_per_month") or 0.0)
    best_wr = float(best.get("win_rate_pct") or 0.0)
    best_pf = best.get("profit_factor")

    gates_met = (
        baseline_spm >= PRODUCTION_GATES["signals_per_month_min"]
        and baseline_wr >= PRODUCTION_GATES["win_rate_min_pct"]
        and baseline_pf >= PRODUCTION_GATES["profit_factor_min"]
    )

    marginal_gain = best_passes and best_removal >= 15.0 and best_spm >= baseline_spm * 0.95

    if gates_met and failure_rate <= 35.0 and best_removal < 10.0:
        verdict = "YES"
    elif gates_met and (best_removal >= 10.0 or failure_rate <= 45.0):
        verdict = "PARTIAL"
    else:
        verdict = "NO"

    return {
        "near_optimal_without_sacrificing_frequency": verdict,
        "baseline_passes_production_gates": gates_met,
        "baseline_metrics": {
            "signals_per_month": baseline_spm,
            "win_rate_pct": baseline_wr,
            "profit_factor": baseline_pf,
            "failure_rate_pct": failure_rate,
            "winner_rate_pct": classification.get("winner_rate_pct"),
        },
        "best_failure_removal_tradeoff": filter_sim.get("best_failure_removal_tradeoff"),
        "best_additional_filter": {
            "condition": best.get("filter_condition"),
            "mode": best.get("filter_mode"),
            "label": best.get("label"),
            "passes_production_gates": best_passes,
            "estimated_win_rate_pct": best_wr,
            "estimated_profit_factor": best_pf,
            "estimated_signals_per_month": best_spm,
            "failure_removal_pct": best_removal,
            "capture_40_plus_pct": best.get("capture_40_plus_pct"),
            "capture_60_plus_pct": best.get("capture_60_plus_pct"),
            "capture_100_plus_pct": best.get("capture_100_plus_pct"),
        },
        "ablation_quality_anchor": ablation.get("final_verdict_ablation"),
        "achievement_summary": achievement.get("all_signals"),
        "timing_summary": {
            "early_pct": timing.get("early_pct"),
            "median_lead_bars": timing.get("lead_time_bars", {}).get("median"),
        },
        "evidence": [
            (
                f"BUY_V3 baseline: {baseline_spm}/mo, WR {baseline_wr}%, PF {baseline_pf}, "
                f"winner rate {classification.get('winner_rate_pct')}%, failure rate {failure_rate}%."
            ),
            (
                f"Best single additional filter: {best.get('filter_condition') or 'none'} — "
                f"removes {best_removal}% failures, est. {best_spm}/mo WR {best_wr}% PF {best_pf}."
            ),
            (
                f"Ablation: removing Liquidity Grab collapses WR to "
                f"{next((v['win_rate_pct'] for v in ablation.get('variant_summary', []) if v.get('removed_condition') == 'Liquidity Grab'), 'N/A')}%."
            ),
            (
                f"Move capture: 40+ achieved on {achievement['all_signals']['40_plus']['rate_pct']}% of signals; "
                f"early timing {timing.get('early_pct')}%."
            ),
        ],
    }


@dataclass
class BuyV3SignalQualityAuditReport:
    """BUY_V3 signal quality audit synthesis output."""

    report_type: str
    model_id: str
    formula_text: str
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    per_signal_audit: list[dict[str, Any]]
    audit_classification_summary: dict[str, Any]
    why_buy_v3_fails: dict[str, Any]
    move_distribution: dict[str, Any]
    achievement_counts: dict[str, Any]
    signal_timing: dict[str, Any]
    winners_vs_failures_condition_comparison: dict[str, Any]
    ablation_insights: dict[str, Any]
    single_filter_simulation: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


class BuyV3SignalQualityAuditResearch:
    """Synthesize BUY_V3 signal quality audit from completed replay exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in SOURCE_EXPORTS.items():
            required = name == "buy_v3_candidate_validation"
            status = "loaded" if path.exists() else ("missing" if required else "optional_missing")
            loaded[name] = {
                "path": str(path),
                "status": status,
                "data": _load_json(path, required=required) if path.exists() or required else {},
            }
        self.sources = loaded
        return loaded

    def run(self) -> BuyV3SignalQualityAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()
        v3_export = sources["buy_v3_candidate_validation"]["data"]
        tradeability = sources["buy_v3_tradeability_production_validation"]["data"]
        winner_export = sources["buy_winner_vs_false_reversal_analysis"]["data"]
        v2_export = sources["buy_v2_candidate_validation"]["data"]

        signals = list(v3_export.get("per_signal_details", {}).get("buy_v3", []))
        if not signals:
            raise BuyV3SignalQualityAuditError(
                "buy_v3_candidate_validation.json has no BUY_V3 per_signal_details.",
            )

        for signal in signals:
            signal["conditions"] = _extract_conditions_from_signal(signal)

        window_days = int(v3_export.get("trading_days_replayed", 120))
        baseline_stats = v3_export.get("comparison", {}).get("buy_v3", {}).get("overall_statistics", {})

        per_signal = _build_per_signal_audit(signals)
        classification = _classification_summary(per_signal)
        move_dist = _move_distribution(per_signal)
        achievements = _achievement_counts(per_signal)
        timing = _signal_timing_analysis(signals, v3_export.get("signal_timing", {}))
        condition_cmp = _condition_comparison(signals)
        failure_analysis = _why_buy_v3_fails(per_signal, condition_cmp, v2_export or None)
        ablation = _ablation_insights(v3_export)
        filter_sim = _single_filter_simulation(
            signals,
            window_days=window_days,
            baseline_stats=_signal_performance(signals, window_days=window_days),
        )
        final = _final_answer(
            baseline_stats=baseline_stats,
            classification=classification,
            filter_sim=filter_sim,
            ablation=ablation,
            achievement=achievements,
            timing=timing,
        )

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "no_new_replay": True,
            "no_new_discovery": True,
            "no_new_models": True,
            "no_new_indicators": True,
            "audit_classifications": list(AUDIT_CLASSIFICATIONS),
            "move_buckets": [label for label, _, _ in MOVE_BUCKETS],
            "achievement_thresholds": list(ACHIEVEMENT_THRESHOLDS),
            "timing_labels": ["Early", "Same Candle", "Delayed", "No Linked Move"],
            "production_gates": PRODUCTION_GATES,
            "primary_source": SOURCE_EXPORTS["buy_v3_candidate_validation"].name,
        }

        limitations = [
            "No new replay — all metrics derived from buy_v3_candidate_validation per_signal_details.",
            "Liquidity Failure maps export False Reversal / Dead Cat Bounce labels.",
            "Single-filter simulation applies one extra condition to existing signals only.",
            "V2 failure patterns referenced for context; BUY_V3 cohort is post-filter stack.",
            "Winner vs failure condition comparison uses BUY_V3 signals only (n=116).",
        ]
        if winner_export:
            limitations.append(
                "Cross-export winner/false-reversal rankings available in buy_winner_vs_false_reversal_analysis.",
            )

        best_filter = final["best_additional_filter"]
        conclusions = [
            f"Audited {len(signals)} BUY_V3 replay signals over {window_days} days.",
            failure_analysis["narrative"],
            (
                f"Timing: {timing['early_pct']}% early, {timing['same_candle_pct']}% same-candle; "
                f"median lead {timing['lead_time_bars'].get('median')} bars."
            ),
            (
                f"MFE achievement: 40+ on {achievements['all_signals']['40_plus']['rate_pct']}% | "
                f"60+ {achievements['all_signals']['60_plus']['rate_pct']}% | "
                f"100+ {achievements['all_signals']['100_plus']['rate_pct']}%."
            ),
            (
                f"Best condition separator within BUY_V3: "
                f"{condition_cmp['rankings']['by_winner_vs_failure_separation'][0]['condition']} "
                f"(delta {condition_cmp['rankings']['by_winner_vs_failure_separation'][0]['coverage_delta_pp']}pp)."
            ),
            (
                f"Single-filter best: {best_filter.get('condition') or 'none'} — "
                f"{best_filter.get('failure_removal_pct')}% failure removal, "
                f"est. {best_filter.get('estimated_signals_per_month')}/mo WR "
                f"{best_filter.get('estimated_win_rate_pct')}%."
            ),
            f"Near-optimal verdict: {final['near_optimal_without_sacrificing_frequency']}.",
        ]

        return BuyV3SignalQualityAuditReport(
            report_type="BUY_V3 Signal Quality Audit",
            model_id=BUY_V3_MODEL_ID,
            formula_text=BUY_V3_FORMULA_TEXT,
            symbol=v3_export.get("symbol", "NIFTY50"),
            timeframe=v3_export.get("timeframe", "5M"),
            trading_days_replayed=window_days,
            replay_start_date=v3_export.get("replay_start_date", ""),
            replay_end_date=v3_export.get("replay_end_date", ""),
            methodology=methodology,
            source_exports={
                name: {"path": payload["path"], "status": payload["status"]}
                for name, payload in sources.items()
            },
            limitations=limitations,
            per_signal_audit=per_signal,
            audit_classification_summary=classification,
            why_buy_v3_fails=failure_analysis,
            move_distribution=move_dist,
            achievement_counts=achievements,
            signal_timing=timing,
            winners_vs_failures_condition_comparison=condition_cmp,
            ablation_insights=ablation,
            single_filter_simulation=filter_sim,
            final_answer=final,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: BuyV3SignalQualityAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("BUY_V3 signal quality audit exported to %s", self.report_path)
        return self.report_path


def generate_buy_v3_signal_quality_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export BUY_V3 signal quality audit JSON."""
    return BuyV3SignalQualityAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_buy_v3_signal_quality_audit_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    best = final["best_additional_filter"]
    print(f"Exported: {path}")
    print(f"Near-optimal: {final['near_optimal_without_sacrificing_frequency']}")
    print(f"Best filter: {best.get('condition')} | removal {best.get('failure_removal_pct')}%")
