"""
BUY_V1 Missed Reversal Analysis — synthesis from existing exports only.

Analyzes Real Reversal moves that BUY_V1 failed to capture, maps condition stacks,
causal-event lead times, and evaluates recovery-stack candidates for BUY_V2.
No new indicators, models, discovery engines, replay, or optimization.
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

from src.research.buy_failure_anatomy_research import (
    _json_safe,
    _normalize_date_key,
)
from src.research.buy_side_frequency_expansion_analysis_research import (
    FORMULA_COMPONENTS,
    FORMULA_RISK_POINTS,
    FORMULA_TEXT,
    MODEL_ID,
    PRODUCTION_GATES,
    SOURCE_EXPORTS as EXPANSION_SOURCE_EXPORTS,
    _build_trap_index,
    _collect_bullish_moves,
    _combo_key,
    _combo_metrics,
    _filter_rows_by_conditions,
    _load_json,
    _performance_metrics,
)
from src.research.buy_v1_production_validation_research import DEFAULT_RISK_POINTS

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_v1_missed_reversal_analysis.json"

SOURCE_EXPORTS = {
    "buy_side_frequency_expansion": RESEARCH_DIR / "buy_side_frequency_expansion_analysis.json",
    **EXPANSION_SOURCE_EXPORTS,
}

BUY_V1_REQUIRED = tuple(FORMULA_COMPONENTS)
ANALYSIS_CONDITIONS = (
    "Liquidity Grab",
    "Failed Breakdown",
    "Near Support",
    "Gap Reversal",
    "Gap Continuation",
    "PDL Sweep",
    "PWL Sweep",
    "Support Reclaim",
    "VWAP Reclaim",
    "HTF Bullish",
)
OUTCOME_TIERS = (40, 60, 80, 100, 200)
FREQUENCY_TARGETS = (15, 20, 30)
TRAP_CAUSAL_EVENTS = frozenset(
    {
        "Liquidity Grab",
        "Failed Breakdown",
        "Gap Reversal",
        "Gap Continuation",
        "PDL Sweep",
        "PWL Sweep",
        "Support Reclaim",
        "Round Number Sweep",
        "Stop Hunt",
    },
)

RECOVERY_STACK_CANDIDATES: list[tuple[str, ...]] = [
    ("Liquidity Grab", "Gap Reversal"),
    ("Liquidity Grab", "PDL Sweep"),
    ("Liquidity Grab", "Support Reclaim"),
    ("Liquidity Grab", "Failed Breakdown", "VWAP Reclaim"),
    ("Liquidity Grab", "Gap Reversal", "Near Support"),
    ("Liquidity Grab", "Failed Breakdown", "Near Support"),
    ("Liquidity Grab", "Gap Continuation", "Near Support"),
    ("Liquidity Grab", "PWL Sweep", "Near Support"),
    ("Liquidity Grab", "PDL Sweep", "Near Support"),
    ("Liquidity Grab", "Failed Breakdown", "Gap Reversal"),
    ("Liquidity Grab", "Near Support"),
    ("Liquidity Grab", "VWAP Reclaim"),
    ("Liquidity Grab", "HTF Bullish"),
    ("Failed Breakdown", "Gap Reversal"),
    ("Failed Breakdown", "Gap Reversal", "Near Support"),
    ("Liquidity Grab", "Failed Breakdown"),
]


class BuyV1MissedReversalAnalysisError(Exception):
    """Raised when BUY_V1 missed reversal synthesis cannot be completed."""


@dataclass
class BuyV1MissedReversalAnalysisReport:
    """BUY_V1 missed reversal synthesis output."""

    report_type: str
    model_id: str
    buy_v1_formula: list[str]
    buy_v1_formula_text: str
    symbol: str
    timeframe: str
    research_window_days: int
    start_date: str
    end_date: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    missed_reversal_summary: dict[str, Any]
    per_missed_reversal: list[dict[str, Any]]
    outcome_measurement: dict[str, Any]
    buy_v1_blocker_analysis: dict[str, Any]
    causal_event_rankings: dict[str, Any]
    recovery_stack_candidates: list[dict[str, Any]]
    essential_optional_bottleneck: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _buy_v1_missing_conditions(conditions: dict[str, bool]) -> list[str]:
    return [name for name in BUY_V1_REQUIRED if not conditions.get(name)]


def _build_drawdown_proxy_index(trap: dict[str, Any]) -> dict[str, float]:
    index: dict[str, float] = {}
    for row in trap.get("trap_event_statistics", []):
        event = str(row.get("event", ""))
        index[event] = float(row.get("average_drawdown_before_expansion", 0.0))
    return index


def _causal_events_for_move(
    *,
    trap_move: dict[str, Any],
    conditions: dict[str, bool],
    drawdown_proxy: dict[str, float],
) -> list[dict[str, Any]]:
    events_by_name: dict[str, dict[str, Any]] = {}
    for item in trap_move.get("events_before_move", []):
        event_name = str(item.get("event", ""))
        if event_name not in TRAP_CAUSAL_EVENTS and event_name not in ANALYSIS_CONDITIONS:
            continue
        bars = int(item.get("bars_before_move", 0))
        existing = events_by_name.get(event_name)
        if existing is None or bars > int(existing.get("bars_before_expansion", 0)):
            events_by_name[event_name] = {
                "event": event_name,
                "bars_before_expansion": bars,
                "minutes_before_expansion": bars * 5,
                "points_before_expansion_proxy": drawdown_proxy.get(event_name),
                "source": "trap_events_before_move",
            }

    for condition in ANALYSIS_CONDITIONS:
        if not conditions.get(condition):
            continue
        if condition in events_by_name:
            continue
        if condition in {"Near Support", "HTF Bullish", "VWAP Reclaim"}:
            proxy_bars = 3 if condition in {"Near Support", "VWAP Reclaim"} else 12
            events_by_name[condition] = {
                "event": condition,
                "bars_before_expansion": proxy_bars,
                "minutes_before_expansion": proxy_bars * 5,
                "points_before_expansion_proxy": None,
                "source": "anatomy_timeline_proxy",
            }
        elif condition == "Support Reclaim":
            events_by_name.setdefault(
                condition,
                {
                    "event": condition,
                    "bars_before_expansion": None,
                    "minutes_before_expansion": None,
                    "points_before_expansion_proxy": drawdown_proxy.get("Failed Breakdown"),
                    "source": "blueprint_inference",
                },
            )

    ordered = sorted(
        events_by_name.values(),
        key=lambda item: -(item.get("bars_before_expansion") or 0),
    )
    return ordered


def _outcome_tiers_met(move_size: float) -> dict[str, Any]:
    met = [tier for tier in OUTCOME_TIERS if move_size >= tier]
    return {
        "move_size_points": round(move_size, 2),
        "tiers_met": met,
        "highest_tier_met": max(met) if met else None,
        "capture_flags": {f"{tier}_plus": move_size >= tier for tier in OUTCOME_TIERS},
    }


def _analyze_missed_reversal(
    row: dict[str, Any],
    *,
    trap_move: dict[str, Any],
    drawdown_proxy: dict[str, float],
) -> dict[str, Any]:
    conditions = row.get("conditions", {})
    analysis_flags = {name: bool(conditions.get(name)) for name in ANALYSIS_CONDITIONS}
    present_stack = [name for name in ANALYSIS_CONDITIONS if analysis_flags[name]]
    missing_buy_v1 = _buy_v1_missing_conditions(analysis_flags)
    first_missing = missing_buy_v1[0] if missing_buy_v1 else None
    last_missing = missing_buy_v1[-1] if missing_buy_v1 else None
    causal_events = _causal_events_for_move(
        trap_move=trap_move,
        conditions=analysis_flags,
        drawdown_proxy=drawdown_proxy,
    )

    return {
        "date": row["date"],
        "move_size_points": row["move_size_points"],
        "duration_minutes": row.get("duration_minutes"),
        "first_event": row.get("first_event"),
        "anatomy_classification": row.get("anatomy_classification"),
        "conditions": analysis_flags,
        "condition_stack_present": present_stack,
        "buy_v1_missing_conditions": missing_buy_v1,
        "first_missing_buy_v1_condition": first_missing,
        "last_missing_buy_v1_condition": last_missing,
        "outcome": _outcome_tiers_met(float(row["move_size_points"])),
        "causal_events": causal_events,
    }


def _outcome_measurement(missed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(missed_rows)
    tier_counts = {tier: 0 for tier in OUTCOME_TIERS}
    for row in missed_rows:
        move_size = float(row["move_size_points"])
        for tier in OUTCOME_TIERS:
            if move_size >= tier:
                tier_counts[tier] += 1

    return {
        "missed_real_reversal_count": total,
        "tier_capture_counts": tier_counts,
        "tier_capture_rates_pct": {
            f"{tier}_plus": round(100.0 * count / max(total, 1), 2)
            for tier, count in tier_counts.items()
        },
        "average_move_size_points": round(mean(float(r["move_size_points"]) for r in missed_rows), 2)
        if missed_rows
        else 0.0,
        "median_move_size_points": round(median(float(r["move_size_points"]) for r in missed_rows), 2)
        if missed_rows
        else 0.0,
    }


def _buy_v1_blocker_analysis(missed_details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(missed_details)
    missing_any = Counter()
    first_missing = Counter()
    last_missing = Counter()
    for row in missed_details:
        for condition in row.get("buy_v1_missing_conditions", []):
            missing_any[condition] += 1
        first = row.get("first_missing_buy_v1_condition")
        last = row.get("last_missing_buy_v1_condition")
        if first:
            first_missing[first] += 1
        if last:
            last_missing[last] += 1

    primary_blocker = missing_any.most_common(1)[0][0] if missing_any else None
    return {
        "total_missed": total,
        "missing_condition_counts": dict(missing_any),
        "missing_condition_rates_pct": {
            condition: round(100.0 * count / max(total, 1), 2)
            for condition, count in missing_any.items()
        },
        "first_missing_condition_counts": dict(first_missing),
        "last_missing_condition_counts": dict(last_missing),
        "primary_prevention_condition": primary_blocker,
        "first_missing_ranking": [
            {"condition": condition, "occurrences": count, "rate_pct": round(100.0 * count / max(total, 1), 2)}
            for condition, count in first_missing.most_common()
        ],
        "interpretation": (
            f"'{primary_blocker}' absent in {missing_any.get(primary_blocker, 0)}/{total} missed reversals "
            f"— most frequent BUY_V1 gate failure."
            if primary_blocker
            else "All BUY_V1 conditions present on missed cohort (timing/export mismatch)."
        ),
    }


def _rank_causal_events(missed_details: list[dict[str, Any]]) -> dict[str, Any]:
    frequency: Counter[str] = Counter()
    bars_by_event: dict[str, list[float]] = defaultdict(list)
    points_by_event: dict[str, list[float]] = defaultdict(list)

    for row in missed_details:
        seen_events: set[str] = set()
        for event in row.get("causal_events", []):
            name = str(event.get("event", ""))
            if name in seen_events:
                continue
            seen_events.add(name)
            frequency[name] += 1
            bars = event.get("bars_before_expansion")
            if bars is not None:
                bars_by_event[name].append(float(bars))
            points = event.get("points_before_expansion_proxy")
            if points is not None:
                points_by_event[name].append(float(points))

    ranked: list[dict[str, Any]] = []
    for event, count in frequency.most_common():
        bars_values = bars_by_event.get(event, [])
        points_values = points_by_event.get(event, [])
        ranked.append(
            {
                "event": event,
                "frequency": count,
                "frequency_pct": round(100.0 * count / max(len(missed_details), 1), 2),
                "average_lead_bars": round(mean(bars_values), 2) if bars_values else None,
                "median_lead_bars": round(median(bars_values), 2) if bars_values else None,
                "average_points_before_expansion_proxy": round(mean(points_values), 2)
                if points_values
                else None,
            },
        )

    ranked_by_lead = sorted(
        [item for item in ranked if item.get("average_lead_bars") is not None],
        key=lambda item: item["average_lead_bars"],
        reverse=True,
    )
    ranked_by_points = sorted(
        [item for item in ranked if item.get("average_points_before_expansion_proxy") is not None],
        key=lambda item: item["average_points_before_expansion_proxy"],
        reverse=True,
    )

    return {
        "missed_cohort_size": len(missed_details),
        "by_frequency": ranked,
        "by_average_lead_bars": ranked_by_lead,
        "by_average_points_before_expansion_proxy": ranked_by_points,
        "note": (
            "Points use trap_event_statistics.average_drawdown_before_expansion per event type; "
            "context conditions (Near Support/HTF/VWAP) use anatomy timeline bar proxy."
        ),
    }


def _evaluate_recovery_stacks(
    all_rows: list[dict[str, Any]],
    missed_details: list[dict[str, Any]],
    *,
    window_days: int,
) -> list[dict[str, Any]]:
    missed_by_date = {_normalize_date_key(str(row["date"])): row for row in missed_details}
    evaluated: list[dict[str, Any]] = []

    for stack in RECOVERY_STACK_CANDIDATES:
        cohort = _filter_rows_by_conditions(all_rows, stack)
        metrics = _combo_metrics(cohort, window_days=window_days, risk_points=FORMULA_RISK_POINTS)
        recovered_dates = [
            _normalize_date_key(str(row["date"]))
            for row in cohort
            if row.get("move_bucket") == "BUY_V1 Missed"
        ]
        recovered_unique = sorted(set(recovered_dates))
        recovered_moves = [missed_by_date[key] for key in recovered_unique if key in missed_by_date]

        def _missed_capture(threshold: int) -> float:
            if not recovered_moves:
                return 0.0
            hits = sum(1 for row in recovered_moves if float(row["move_size_points"]) >= threshold)
            return round(100.0 * hits / len(recovered_moves), 2)

        evaluated.append(
            {
                "stack": list(stack),
                "stack_text": _combo_key(stack),
                "recovered_real_reversals_count": len(recovered_unique),
                "recovered_real_reversals_pct": round(
                    100.0 * len(recovered_unique) / max(len(missed_details), 1),
                    2,
                ),
                "recovered_dates": recovered_unique[:12],
                "signals_per_month": metrics["signals_per_month"],
                "win_rate_pct": metrics["win_rate_pct"],
                "profit_factor": metrics["profit_factor"],
                "expectancy": metrics["expectancy"],
                "capture_40_plus_pct": metrics["capture_40_plus_pct"],
                "capture_60_plus_pct": metrics["capture_60_plus_pct"],
                "capture_100_plus_pct": metrics["capture_100_plus_pct"],
                "real_reversal_rate_pct": metrics["real_reversal_rate_pct"],
                "passes_production_gates": metrics["passes_production_gates"],
                "missed_cohort_capture_40_plus_pct": _missed_capture(40),
                "missed_cohort_capture_60_plus_pct": _missed_capture(60),
                "missed_cohort_capture_100_plus_pct": _missed_capture(100),
                "meets_15_plus_signals_per_month": metrics["signals_per_month"] >= 15,
                "meets_20_plus_signals_per_month": metrics["signals_per_month"] >= 20,
                "meets_30_plus_signals_per_month": metrics["signals_per_month"] >= 30,
            },
        )

    evaluated.sort(
        key=lambda item: (
            item["passes_production_gates"],
            item["recovered_real_reversals_count"],
            item["signals_per_month"],
            item["win_rate_pct"],
        ),
        reverse=True,
    )
    return evaluated


def _essential_optional_bottleneck(
    missed_details: list[dict[str, Any]],
    blocker_analysis: dict[str, Any],
    recovery_stacks: list[dict[str, Any]],
    causal_rankings: dict[str, Any],
) -> dict[str, Any]:
    total = len(missed_details)
    condition_presence = Counter()
    for row in missed_details:
        for name in row.get("condition_stack_present", []):
            condition_presence[name] += 1

    essential = "Liquidity Grab"
    if blocker_analysis.get("primary_prevention_condition"):
        essential = str(blocker_analysis["primary_prevention_condition"])

    optional_candidates = [
        name
        for name, count in condition_presence.most_common()
        if name not in BUY_V1_REQUIRED and count >= total * 0.4
    ]
    optional = optional_candidates[0] if optional_candidates else "Gap Reversal"

    passing_stacks = [stack for stack in recovery_stacks if stack["passes_production_gates"]]
    if passing_stacks:
        bottleneck_stack = min(passing_stacks, key=lambda item: item["signals_per_month"])
        frequency_bottleneck = (
            f"{bottleneck_stack['stack_text']} caps frequency at "
            f"{bottleneck_stack['signals_per_month']}/mo despite {bottleneck_stack['recovered_real_reversals_count']} recoveries."
        )
    else:
        frequency_bottleneck = (
            "Liquidity Grab rarity — present in only "
            f"{condition_presence.get('Liquidity Grab', 0)}/{total} missed reversals; "
            "limits any LG-based recovery stack frequency."
        )

    best_stack = recovery_stacks[0] if recovery_stacks else {}
    return {
        "essential_condition": {
            "condition": essential,
            "rationale": (
                f"Absent in {blocker_analysis.get('missing_condition_counts', {}).get(essential, 0)}/{total} "
                "missed reversals — restoring this condition is required for BUY_V1-class capture."
                if essential in BUY_V1_REQUIRED
                else f"Highest-impact missing gate: {essential}."
            ),
        },
        "optional_condition": {
            "condition": optional,
            "rationale": (
                f"Present in {condition_presence.get(optional, 0)}/{total} missed reversals without being "
                "mandatory for BUY_V1 — adds context but not required for every recovery."
            ),
        },
        "frequency_bottleneck": frequency_bottleneck,
        "best_buy_v2_candidate_stack": {
            "stack_text": best_stack.get("stack_text"),
            "stack": best_stack.get("stack"),
            "recovered_real_reversals_count": best_stack.get("recovered_real_reversals_count"),
            "signals_per_month": best_stack.get("signals_per_month"),
            "win_rate_pct": best_stack.get("win_rate_pct"),
            "profit_factor": best_stack.get("profit_factor"),
            "passes_production_gates": best_stack.get("passes_production_gates"),
        },
        "condition_presence_on_missed_cohort": dict(condition_presence),
    }


def _final_answer(
    recovery_stacks: list[dict[str, Any]],
    missed_count: int,
    expansion_export: dict[str, Any],
) -> dict[str, Any]:
    passing = [stack for stack in recovery_stacks if stack["passes_production_gates"]]
    best_recovery = max(passing, key=lambda item: item["recovered_real_reversals_count"]) if passing else None
    best_frequency = max(passing, key=lambda item: item["signals_per_month"]) if passing else None

    frequency_verdicts: dict[str, str] = {}
    evidence: list[str] = []
    for target in FREQUENCY_TARGETS:
        candidates = [
            stack
            for stack in passing
            if stack["signals_per_month"] >= target
            and stack["recovered_real_reversals_count"] >= missed_count * 0.5
        ]
        if candidates:
            frequency_verdicts[str(target)] = "YES"
            top = max(candidates, key=lambda item: item["recovered_real_reversals_count"])
            evidence.append(
                f"{target}+/mo: '{top['stack_text']}' recovers {top['recovered_real_reversals_count']}/{missed_count} "
                f"missed reversals at WR {top['win_rate_pct']}% / PF {top['profit_factor']}.",
            )
        elif any(stack["signals_per_month"] >= target for stack in passing):
            partial = max(
                (stack for stack in passing if stack["signals_per_month"] >= target),
                key=lambda item: item["recovered_real_reversals_count"],
                default=None,
            )
            frequency_verdicts[str(target)] = "PARTIAL"
            if partial:
                evidence.append(
                    f"{target}+/mo: '{partial['stack_text']}' hits frequency ({partial['signals_per_month']}/mo) "
                    f"but recovers only {partial['recovered_real_reversals_count']}/{missed_count} missed reversals.",
                )
        else:
            frequency_verdicts[str(target)] = "NO"
            evidence.append(f"{target}+/mo: no recovery stack passes WR>65%, PF>2 at this frequency.")

    max_recovered = max((stack["recovered_real_reversals_count"] for stack in passing), default=0)
    recovery_ratio = max_recovered / max(missed_count, 1)

    if best_recovery:
        evidence.append(
            f"Best recovery: '{best_recovery['stack_text']}' captures {best_recovery['recovered_real_reversals_count']}/"
            f"{missed_count} missed real reversals ({best_recovery['recovered_real_reversals_pct']}%).",
        )
    if best_frequency:
        evidence.append(
            f"Highest-frequency passing stack: '{best_frequency['stack_text']}' at "
            f"{best_frequency['signals_per_month']}/mo, WR {best_frequency['win_rate_pct']}%.",
        )

    extraction_note = expansion_export.get("frequency_expansion_candidates", {}).get(
        "final_signal_extraction_note",
        "",
    )
    if extraction_note:
        evidence.append(extraction_note)

    if recovery_ratio >= 0.8 and any(verdict == "YES" for verdict in frequency_verdicts.values()):
        overall = "YES"
    elif recovery_ratio >= 0.3 or any(verdict in {"YES", "PARTIAL"} for verdict in frequency_verdicts.values()):
        overall = "PARTIAL"
    else:
        overall = "NO"

    return {
        "overall_verdict": overall,
        "can_recover_47_missed_reversals": overall,
        "missed_real_reversal_count": missed_count,
        "max_recovered_count": max_recovered,
        "max_recovery_rate_pct": round(100.0 * recovery_ratio, 2),
        "by_frequency_target": frequency_verdicts,
        "can_reach_15_plus_signals_per_month": frequency_verdicts.get("15", "NO"),
        "can_reach_20_plus_signals_per_month": frequency_verdicts.get("20", "NO"),
        "can_reach_30_plus_signals_per_month": frequency_verdicts.get("30", "NO"),
        "production_gates": PRODUCTION_GATES,
        "evidence": evidence,
        "best_recovery_stack": best_recovery,
    }


class BuyV1MissedReversalAnalysisResearch:
    """Synthesize BUY_V1 missed reversal recovery analysis from completed exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        required = {
            "buy_v1_production_validation",
            "buy_failure_anatomy",
            "buy_side_discovery",
            "momentum_anatomy",
            "trap_to_momentum",
            "liquidity_decision_engine",
        }
        for name, path in SOURCE_EXPORTS.items():
            is_required = name in required
            exists = path.exists()
            status = "loaded" if exists else ("missing" if is_required else "optional_missing")
            loaded[name] = {
                "path": str(path),
                "status": status,
                "data": _load_json(path, required=is_required) if exists or is_required else {},
            }
        self.sources = loaded
        return loaded

    def run(self) -> BuyV1MissedReversalAnalysisReport:
        started = time.perf_counter()
        self._load_sources()

        expansion = self.sources.get("buy_side_frequency_expansion", {}).get("data", {})
        buy_v1 = self.sources["buy_v1_production_validation"]["data"]
        discovery = self.sources["buy_side_discovery"]["data"]
        trap = self.sources["trap_to_momentum"]["data"]

        window_days = int(
            expansion.get("research_window_days")
            or discovery.get("research_window", {}).get("research_window_days", buy_v1.get("research_window_days", 120)),
        )
        start_date = str(expansion.get("start_date") or discovery.get("research_window", {}).get("start_date", buy_v1.get("start_date", "")))
        end_date = str(expansion.get("end_date") or discovery.get("research_window", {}).get("end_date", buy_v1.get("end_date", "")))

        buy_v1_occurrences = buy_v1.get("all_occurrences", [])
        buy_v1_dates = {
            _normalize_date_key(str(item.get("move_timestamp", item.get("signal_timestamp", ""))))
            for item in buy_v1_occurrences
        }
        buy_v1_performance = {
            _normalize_date_key(str(item.get("move_timestamp", item.get("signal_timestamp", "")))): item
            for item in buy_v1_occurrences
        }

        all_rows = _collect_bullish_moves(
            self.sources,
            buy_v1_dates=buy_v1_dates,
            buy_v1_performance=buy_v1_performance,
        )
        missed_rows = [row for row in all_rows if row.get("move_bucket") == "BUY_V1 Missed"]
        expected_missed = int(
            expansion.get("bullish_move_classification", {}).get("buy_v1_missed_real_reversal_count", len(missed_rows)),
        )
        if expected_missed and len(missed_rows) != expected_missed:
            logger.warning(
                "Missed reversal count %s differs from expansion export %s",
                len(missed_rows),
                expected_missed,
            )

        trap_index = _build_trap_index(trap)
        drawdown_proxy = _build_drawdown_proxy_index(trap)

        per_missed: list[dict[str, Any]] = []
        for row in sorted(missed_rows, key=lambda item: -float(item["move_size_points"])):
            date_key = _normalize_date_key(str(row["date"]))
            trap_move = trap_index.get(date_key, {"events_before_move": [], "first_event": row.get("first_event")})
            per_missed.append(_analyze_missed_reversal(row, trap_move=trap_move, drawdown_proxy=drawdown_proxy))

        outcome_measurement = _outcome_measurement(missed_rows)
        blocker_analysis = _buy_v1_blocker_analysis(per_missed)
        causal_rankings = _rank_causal_events(per_missed)
        recovery_stacks = _evaluate_recovery_stacks(all_rows, per_missed, window_days=window_days)
        essential_blocker = _essential_optional_bottleneck(
            per_missed,
            blocker_analysis,
            recovery_stacks,
            causal_rankings,
        )
        final_answer = _final_answer(recovery_stacks, len(missed_rows), expansion)

        methodology = {
            "research_only": True,
            "no_new_indicators": True,
            "no_discovery_engines": True,
            "no_new_buy_models": True,
            "no_replay": True,
            "no_optimization": True,
            "buy_v1_formula": FORMULA_TEXT,
            "missed_reversal_definition": "Real Reversal moves not matching buy_v1_production_validation occurrence dates",
            "analysis_conditions": list(ANALYSIS_CONDITIONS),
            "buy_v1_required_conditions": list(BUY_V1_REQUIRED),
            "outcome_tiers_points": list(OUTCOME_TIERS),
            "recovery_stack_universe": [list(stack) for stack in RECOVERY_STACK_CANDIDATES],
            "production_gates": PRODUCTION_GATES,
            "frequency_targets_per_month": list(FREQUENCY_TARGETS),
            "risk_points_proxy": DEFAULT_RISK_POINTS,
            "synthesis_sources": list(SOURCE_EXPORTS.keys()),
            "primary_reference_export": "buy_side_frequency_expansion_analysis.json",
        }

        limitations = [
            "Per-signal trade paths not replayed — WR/PF/expectancy use move_size vs risk proxy except BUY_V1 captured rows.",
            "Points-before-expansion uses trap_event_statistics drawdown proxy when per-move points absent.",
            "Near Support / VWAP / HTF timing uses anatomy T-15/T-60 bar proxy, not native bar scans.",
            "Recovery stacks are cohort-filter counts from existing exports — not forward-validated BUY_V2 engines.",
            "final_signal_extraction rejected all BUY candidates — recovery stacks are research candidates only.",
        ]

        missed_summary = {
            "missed_real_reversal_count": len(missed_rows),
            "expansion_export_missed_count": expected_missed,
            "buy_v1_captured_count": sum(1 for row in all_rows if row.get("buy_v1_captured")),
            "total_real_reversals": sum(1 for row in all_rows if row.get("anatomy_classification") == "Real Reversal"),
            "primary_prevention_condition": blocker_analysis.get("primary_prevention_condition"),
            "outcome_tier_rates_pct": outcome_measurement["tier_capture_rates_pct"],
        }

        conclusions = [
            f"Analyzed {len(missed_rows)} BUY_V1-missed Real Reversal moves across {window_days}-day NIFTY50 window.",
            (
                f"Primary BUY_V1 blocker: {blocker_analysis.get('primary_prevention_condition')} "
                f"({blocker_analysis.get('interpretation', '')})"
            ),
            (
                f"Outcome tiers on missed cohort: "
                + ", ".join(
                    f"{tier}+={outcome_measurement['tier_capture_rates_pct'].get(f'{tier}_plus', 0)}%"
                    for tier in OUTCOME_TIERS
                )
            ),
            (
                f"Best BUY_V2 recovery stack: {essential_blocker['best_buy_v2_candidate_stack'].get('stack_text')} "
                f"— recovers {essential_blocker['best_buy_v2_candidate_stack'].get('recovered_real_reversals_count')}/"
                f"{len(missed_rows)} at {essential_blocker['best_buy_v2_candidate_stack'].get('signals_per_month')}/mo."
            ),
            (
                f"Recovery feasibility: {final_answer['overall_verdict']} "
                f"(15+/20+/30+ per month: "
                f"{final_answer['can_reach_15_plus_signals_per_month']}/"
                f"{final_answer['can_reach_20_plus_signals_per_month']}/"
                f"{final_answer['can_reach_30_plus_signals_per_month']})."
            ),
        ]

        return BuyV1MissedReversalAnalysisReport(
            report_type="BUY_V1 Missed Reversal Analysis",
            model_id=MODEL_ID,
            buy_v1_formula=FORMULA_COMPONENTS,
            buy_v1_formula_text=FORMULA_TEXT,
            symbol=discovery.get("symbol", "NIFTY50"),
            timeframe=discovery.get("research_window", {}).get("primary_timeframe", "5M"),
            research_window_days=window_days,
            start_date=start_date,
            end_date=end_date,
            methodology=methodology,
            source_exports={
                name: {"path": payload["path"], "status": payload["status"]}
                for name, payload in self.sources.items()
            },
            limitations=limitations,
            missed_reversal_summary=missed_summary,
            per_missed_reversal=per_missed,
            outcome_measurement=outcome_measurement,
            buy_v1_blocker_analysis=blocker_analysis,
            causal_event_rankings=causal_rankings,
            recovery_stack_candidates=recovery_stacks,
            essential_optional_bottleneck=essential_blocker,
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: BuyV1MissedReversalAnalysisReport | None = None) -> Path:
        payload = report or self.run()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = _json_safe(asdict(payload))
        self.report_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        logger.info("BUY_V1 missed reversal analysis exported to %s", self.report_path)
        return self.report_path


def generate_buy_v1_missed_reversal_analysis_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export BUY_V1 missed reversal analysis."""
    return BuyV1MissedReversalAnalysisResearch(report_path=report_path).export()
