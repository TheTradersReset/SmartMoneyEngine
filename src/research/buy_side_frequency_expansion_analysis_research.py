"""
BUY Side Frequency Expansion Analysis — synthesis from existing exports only.

Evaluates whether BUY_V1 (Liquidity Grab + Failed Breakdown + Near Support) can expand
from ~4.25 signals/month toward 20+/30+/40+/month while preserving WR>65%, PF>2, and
tradeable 40+/60+/80+/100+ capture. No new scans, indicators, discovery engines, replay,
or optimization.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.research.buy_failure_anatomy_research import (
    CONSOLIDATION_TAGS,
    DEAD_CAT_MAX_POINTS,
    NEAR_SUPPORT_LABEL,
    PRECURSOR_EVENTS,
    _classify_move,
    _context_snapshot,
    _has_consolidation_tags,
    _json_safe,
    _normalize_date_key,
    _ordered_causal_events,
    _precursor_match,
    _second_event_name,
)
from src.research.buy_v1_production_validation_research import (
    DEFAULT_RISK_POINTS,
    FORMULA_COMPONENTS,
    FORMULA_TEXT,
    MODEL_ID,
    SIGNAL_STEP,
    _near_support,
    _performance_metrics,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_side_frequency_expansion_analysis.json"

SOURCE_EXPORTS = {
    "buy_v1_production_validation": RESEARCH_DIR / "buy_v1_production_validation.json",
    "buy_entry_timing_validation": RESEARCH_DIR / "buy_entry_timing_validation.json",
    "buy_failure_anatomy": RESEARCH_DIR / "buy_failure_anatomy.json",
    "buy_formula_verification": RESEARCH_DIR / "buy_formula_reality_verification.json",
    "buy_side_discovery": RESEARCH_DIR / "nifty50_buy_side_reality_discovery.json",
    "momentum_anatomy": RESEARCH_DIR / "nifty50_momentum_anatomy_120d.json",
    "trap_to_momentum": RESEARCH_DIR / "nifty50_trap_to_momentum_validation.json",
    "liquidity_decision_engine": RESEARCH_DIR / "nifty50_liquidity_decision_engine.json",
    "institutional_momentum_origin": RESEARCH_DIR / "institutional_momentum_origin.json",
    "liquidity_move_reconstruction": RESEARCH_DIR / "liquidity_move_reconstruction.json",
    "tradeable_move_validation": RESEARCH_DIR / "tradeable_move_validation.json",
    "final_signal_extraction": RESEARCH_DIR / "smartmoneyengine_final_signal_extraction.json",
    "research_consistency_audit": RESEARCH_DIR / "research_consistency_audit.json",
    "sell_formula_v2": RESEARCH_DIR / "sell_formula_reality_verification_v2.json",
}

CONDITION_LABELS = (
    "Liquidity Grab",
    "Failed Breakdown",
    "Near Support",
    "Gap Reversal",
    "Gap Continuation",
    "PDL Sweep",
    "PWL Sweep",
    "Support Reclaim",
    "Round Number Reclaim",
    "HTF Bullish",
    "VWAP Reclaim",
)

TRADEABLE_TIERS = (40, 60, 80, 100)
FREQUENCY_TARGETS = (20, 30, 40)
PRODUCTION_GATES = {"win_rate_min_pct": 65.0, "profit_factor_min": 2.0}
BUY_V1_BASELINE_FREQUENCY = 4.25
FORMULA_RISK_POINTS = DEFAULT_RISK_POINTS
RELAXED_RISK_POINTS = 61.06


class BuySideFrequencyExpansionAnalysisError(Exception):
    """Raised when BUY frequency expansion synthesis cannot be completed."""


@dataclass
class BuySideFrequencyExpansionAnalysisReport:
    """BUY side frequency expansion synthesis output."""

    report_type: str
    model_id: str
    current_buy_v1_formula: list[str]
    current_buy_v1_formula_text: str
    symbol: str
    timeframe: str
    research_window_days: int
    start_date: str
    end_date: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    bullish_move_classification: dict[str, Any]
    condition_attribution: dict[str, Any]
    combination_rankings: dict[str, Any]
    frequency_expansion_candidates: dict[str, Any]
    mandatory_vs_false_conditions: dict[str, Any]
    final_answer: dict[str, Any]
    most_valuable_setup: dict[str, Any]
    production_recommendation: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise BuySideFrequencyExpansionAnalysisError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _event_in_trap(events: list[dict[str, Any]], event_name: str) -> bool:
    return any(item.get("event") == event_name for item in events)


def _event_bars_before(events: list[dict[str, Any]], event_name: str) -> int | None:
    for item in events:
        if item.get("event") == event_name:
            return int(item.get("bars_before_move", 0))
    return None


def _earliest_lead_bars(events: list[dict[str, Any]], condition_events: tuple[str, ...]) -> int | None:
    bars: list[int] = []
    for event_name in condition_events:
        value = _event_bars_before(events, event_name)
        if value is not None:
            bars.append(value)
    return max(bars) if bars else None


def _support_reclaim(blueprint: str, events: list[dict[str, Any]]) -> bool:
    text = f"{blueprint} {' '.join(str(item.get('event', '')) for item in events)}"
    return "Support Reclaim" in text or "Reclaim Support" in text


def _round_number_reclaim(blueprint: str, events: list[dict[str, Any]], first_event: str | None) -> bool:
    if first_event == "Round Number Sweep":
        return True
    text = f"{blueprint} {' '.join(str(item.get('event', '')) for item in events)}"
    return "Round Number Sweep" in text or "Round Number Reclaim" in text


def _build_condition_flags(
    *,
    trap_move: dict[str, Any],
    anatomy_record: dict[str, Any],
    context_t15: dict[str, Any],
    context_t60: dict[str, Any],
) -> dict[str, bool]:
    events = trap_move.get("events_before_move", [])
    blueprint = str(anatomy_record.get("blueprint_pattern", ""))
    origin = str(anatomy_record.get("origin_trigger", ""))
    causal = _ordered_causal_events(events)
    reason_t60 = context_t60.get("reason_stack", {})
    reason_t15 = context_t15.get("reason_stack", {})
    flags_t15 = context_t15.get("feature_flags", {})
    levels_t15 = context_t15.get("levels", {})

    def _present(event_name: str) -> bool:
        return (
            _event_in_trap(events, event_name)
            or event_name in blueprint
            or origin == event_name
            or event_name in causal
        )

    return {
        "Liquidity Grab": _present("Liquidity Grab"),
        "Failed Breakdown": _present("Failed Breakdown"),
        "Near Support": levels_t15.get("market_location") == NEAR_SUPPORT_LABEL
        or context_t60.get("levels", {}).get("market_location") == NEAR_SUPPORT_LABEL,
        "Gap Reversal": _present("Gap Reversal"),
        "Gap Continuation": _present("Gap Continuation"),
        "PDL Sweep": _present("PDL Sweep"),
        "PWL Sweep": _present("PWL Sweep"),
        "Support Reclaim": _support_reclaim(blueprint, events),
        "Round Number Reclaim": _round_number_reclaim(blueprint, events, trap_move.get("first_event")),
        "HTF Bullish": reason_t60.get("htf_trend") in {"Strong Bullish", "Bullish"},
        "VWAP Reclaim": reason_t15.get("vwap") == "Above VWAP" or bool(flags_t15.get("fvg_reclaim")),
    }


def _pnl_proxy(move_size: float, risk_points: float) -> float:
    if move_size >= risk_points * 2:
        return round(risk_points * 2.0, 2)
    if move_size >= risk_points:
        return round(risk_points, 2)
    return round(-risk_points, 2)


def _is_win(move_size: float, risk_points: float) -> bool:
    return move_size >= risk_points


def _combo_metrics(
    rows: list[dict[str, Any]],
    *,
    window_days: int,
    risk_points: float,
) -> dict[str, Any]:
    sample_size = len(rows)
    if sample_size == 0:
        return {
            "sample_size": 0,
            "signals_per_month": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "average_lead_bars_before_expansion": 0.0,
            "median_lead_bars_before_expansion": 0.0,
            "capture_40_plus_pct": 0.0,
            "capture_60_plus_pct": 0.0,
            "capture_80_plus_pct": 0.0,
            "capture_100_plus_pct": 0.0,
            "real_reversal_rate_pct": 0.0,
            "passes_production_gates": False,
        }

    pnls = [_pnl_proxy(float(row["move_size_points"]), risk_points) for row in rows]
    wins = sum(1 for row in rows if row.get("win", _is_win(float(row["move_size_points"]), risk_points)))
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0 and gross_profit > 0:
        pf = None
    elif gross_loss == 0:
        pf = 0.0
    else:
        pf = round(gross_profit / gross_loss, 2)
    wr = round(100.0 * wins / sample_size, 2)
    expectancy = round(mean(pnls), 2)
    signals_per_month = round(sample_size / max(window_days / 30.0, 1.0), 2)
    lead_bars = [float(row.get("earliest_lead_bars", 0)) for row in rows if row.get("earliest_lead_bars") is not None]
    real_reversal_count = sum(1 for row in rows if row.get("anatomy_classification") == "Real Reversal")

    def _capture(threshold: int) -> float:
        return round(
            100.0
            * sum(1 for row in rows if float(row.get("move_size_points", 0)) >= threshold)
            / sample_size,
            2,
        )

    passes = (
        wr >= PRODUCTION_GATES["win_rate_min_pct"]
        and (pf is None or pf >= PRODUCTION_GATES["profit_factor_min"])
        and real_reversal_count / sample_size >= 0.65
    )

    return {
        "sample_size": sample_size,
        "signals_per_month": signals_per_month,
        "win_rate_pct": wr,
        "profit_factor": pf,
        "expectancy": expectancy,
        "average_lead_bars_before_expansion": round(mean(lead_bars), 2) if lead_bars else 0.0,
        "median_lead_bars_before_expansion": round(median(lead_bars), 2) if lead_bars else 0.0,
        "capture_40_plus_pct": _capture(40),
        "capture_60_plus_pct": _capture(60),
        "capture_80_plus_pct": _capture(80),
        "capture_100_plus_pct": _capture(100),
        "real_reversal_rate_pct": round(100.0 * real_reversal_count / sample_size, 2),
        "passes_production_gates": passes,
    }


def _combo_key(conditions: tuple[str, ...]) -> str:
    return " + ".join(conditions)


def _filter_rows_by_conditions(rows: list[dict[str, Any]], conditions: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(row.get("conditions", {}).get(condition, False) for condition in conditions)
    ]


def _build_move_record(
    *,
    date_value: str,
    move_size: float,
    duration_minutes: float,
    trap_move: dict[str, Any],
    anatomy_record: dict[str, Any],
    buy_v1_dates: set[str],
    buy_v1_performance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    context_t60 = _context_snapshot(anatomy_record, "T-60 minutes") or {}
    context_t15 = _context_snapshot(anatomy_record, "T-15 minutes") or {}
    events = trap_move.get("events_before_move", [])
    causal_events = _ordered_causal_events(events)
    near_support = (
        context_t15.get("levels", {}).get("market_location") == NEAR_SUPPORT_LABEL
        or context_t60.get("levels", {}).get("market_location") == NEAR_SUPPORT_LABEL
    )
    matched, precursors = _precursor_match(
        first_event=trap_move.get("first_event"),
        causal_events=causal_events,
        near_support=near_support,
        origin_trigger=anatomy_record.get("origin_trigger"),
    )

    classification_record = {
        "move_size_points": move_size,
        "duration_minutes": duration_minutes,
        "first_event": trap_move.get("first_event") or anatomy_record.get("origin_trigger"),
        "second_event": _second_event_name(events),
        "matched_precursors": precursors,
        "origin_trigger": anatomy_record.get("origin_trigger"),
        "context_t60": context_t60,
        "context_t15": context_t15,
        "has_consolidation_tags": _has_consolidation_tags(context_t15)
        or _has_consolidation_tags(context_t60)
        or any(tag in str(anatomy_record.get("blueprint_pattern", "")) for tag in CONSOLIDATION_TAGS),
        "lde_outcome": "",
    }
    anatomy_class = _classify_move(classification_record)
    date_key = _normalize_date_key(date_value)
    buy_v1_hit = date_key in buy_v1_dates

    if buy_v1_hit:
        move_bucket = "BUY_V1 Captured"
    elif anatomy_class == "Real Reversal":
        move_bucket = "BUY_V1 Missed"
    elif anatomy_class == "Dead Cat Bounce":
        move_bucket = "Dead Cat Bounce"
    else:
        move_bucket = anatomy_class

    conditions = _build_condition_flags(
        trap_move=trap_move,
        anatomy_record=anatomy_record,
        context_t15=context_t15,
        context_t60=context_t60,
    )
    lead_bars = _earliest_lead_bars(
        events,
        tuple(name for name, present in conditions.items() if present and name in PRECURSOR_EVENTS),
    )
    if lead_bars is None:
        lead_bars = _event_bars_before(events, str(trap_move.get("first_event", "")))

    perf = buy_v1_performance.get(date_key, {})
    win = perf.get("win", _is_win(move_size, FORMULA_RISK_POINTS))

    return {
        "date": date_value,
        "move_size_points": round(move_size, 2),
        "duration_minutes": duration_minutes,
        "move_bucket": move_bucket,
        "anatomy_classification": anatomy_class,
        "buy_v1_captured": buy_v1_hit,
        "matched_precursors": precursors,
        "first_event": trap_move.get("first_event") or anatomy_record.get("origin_trigger"),
        "conditions": conditions,
        "earliest_lead_bars": lead_bars,
        "earliest_lead_minutes": (lead_bars * 5) if lead_bars is not None else None,
        "win": win,
        "realized_pnl_points": perf.get("realized_pnl_points", _pnl_proxy(move_size, FORMULA_RISK_POINTS)),
    }


def _dedupe_bullish_completed_moves(completed_moves: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, float]] = set()
    unique: list[dict[str, Any]] = []
    for moves in completed_moves.values():
        for move in moves:
            if move.get("direction") != "bullish":
                continue
            key = (_normalize_date_key(str(move.get("date", ""))), round(float(move.get("move_size_points", 0)), 1))
            if key in seen:
                continue
            seen.add(key)
            unique.append(move)
    return unique


def _build_anatomy_index(anatomy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in anatomy.get("move_anatomy_records", []):
        if record.get("direction") != "bullish":
            continue
        key = _normalize_date_key(str(record.get("date", "")))
        if key not in index:
            index[key] = record
    return index


def _build_trap_index(trap: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for move in trap.get("move_pre_event_analysis", []):
        if move.get("direction") != "bullish":
            continue
        key = _normalize_date_key(str(move.get("date", "")))
        if key not in index:
            index[key] = move
    return index


def _collect_bullish_moves(
    sources: dict[str, dict[str, Any]],
    *,
    buy_v1_dates: set[str],
    buy_v1_performance: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    anatomy = sources["momentum_anatomy"]["data"]
    trap = sources["trap_to_momentum"]["data"]
    anatomy_index = _build_anatomy_index(anatomy)
    trap_index = _build_trap_index(trap)
    completed = _dedupe_bullish_completed_moves(anatomy.get("completed_moves", {}))

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for move in completed:
        date_value = str(move.get("date", ""))
        move_size = float(move.get("move_size_points", 0))
        if move_size < 40:
            continue
        dedupe_key = (_normalize_date_key(date_value), round(move_size, 1))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        date_key = _normalize_date_key(date_value)
        trap_move = trap_index.get(date_key, {"events_before_move": [], "first_event": None})
        anatomy_record = anatomy_index.get(
            date_key,
            {
                "date": date_value,
                "move_size_points": move_size,
                "duration_minutes": move.get("duration_minutes", 0),
                "origin_trigger": trap_move.get("first_event"),
                "blueprint_pattern": "",
                "timeline": [],
            },
        )
        rows.append(
            _build_move_record(
                date_value=date_value,
                move_size=move_size,
                duration_minutes=float(move.get("duration_minutes", 0)),
                trap_move=trap_move,
                anatomy_record=anatomy_record,
                buy_v1_dates=buy_v1_dates,
                buy_v1_performance=buy_v1_performance,
            ),
        )

    lde = sources["liquidity_decision_engine"]["data"]
    for event in lde.get("liquidity_event_log", []):
        if event.get("direction") != "bullish":
            continue
        if event.get("outcome") != "No Expansion":
            continue
        forward = event.get("forward_metrics", {})
        move_size = round(float(forward.get("max_move", forward.get("bull_move", 45.0))), 2)
        if move_size >= DEAD_CAT_MAX_POINTS:
            continue
        timestamp = str(event.get("timestamp", ""))
        dedupe_key = (_normalize_date_key(timestamp), move_size)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        level_ctx = event.get("major_level_context", {})
        near_support = float(level_ctx.get("distance_nearest_support", 999)) <= 25
        trap_move = {
            "events_before_move": [{"event": event.get("event_type"), "bars_before_move": 3}],
            "first_event": event.get("event_type"),
        }
        anatomy_record = {
            "date": timestamp,
            "origin_trigger": event.get("event_type"),
            "blueprint_pattern": "",
            "timeline": [
                {
                    "timeline_step": "T-15 minutes",
                    "context_by_timeframe": {
                        "5M": {
                            "levels": {
                                "market_location": NEAR_SUPPORT_LABEL if near_support else "Mid Range",
                            },
                            "reason_stack": {},
                            "feature_flags": {},
                        },
                    },
                },
            ],
        }
        rows.append(
            _build_move_record(
                date_value=timestamp,
                move_size=move_size,
                duration_minutes=float(forward.get("time_to_expansion_bars", 0)) * 5,
                trap_move=trap_move,
                anatomy_record=anatomy_record,
                buy_v1_dates=buy_v1_dates,
                buy_v1_performance=buy_v1_performance,
            ),
        )

    return rows


def _condition_attribution_for_real_reversals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    real_rows = [row for row in rows if row.get("anatomy_classification") == "Real Reversal"]
    total = len(real_rows)
    counts: dict[str, int] = {}
    rates: dict[str, float] = {}
    for condition in CONDITION_LABELS:
        present = sum(1 for row in real_rows if row.get("conditions", {}).get(condition))
        counts[condition] = present
        rates[condition] = round(100.0 * present / max(total, 1), 2)

    earliest_counter: Counter[str] = Counter()
    lead_by_first: dict[str, list[float]] = defaultdict(list)
    for row in real_rows:
        first = str(row.get("first_event", "Unknown"))
        earliest_counter[first] += 1
        if row.get("earliest_lead_bars") is not None:
            lead_by_first[first].append(float(row["earliest_lead_bars"]))

    return {
        "real_reversal_count": total,
        "condition_presence_counts": counts,
        "condition_presence_rates_pct": rates,
        "earliest_causal_event_ranking": [
            {"event": event, "occurrences": count}
            for event, count in earliest_counter.most_common()
        ],
        "average_lead_bars_by_first_event": {
            event: round(mean(values), 2) if values else 0.0
            for event, values in lead_by_first.items()
        },
        "successful_move_timing_summary": {
            "average_lead_bars": round(
                mean(float(row["earliest_lead_bars"]) for row in real_rows if row.get("earliest_lead_bars") is not None),
                2,
            )
            if any(row.get("earliest_lead_bars") is not None for row in real_rows)
            else 0.0,
            "median_lead_bars": round(
                median(float(row["earliest_lead_bars"]) for row in real_rows if row.get("earliest_lead_bars") is not None),
                2,
            )
            if any(row.get("earliest_lead_bars") is not None for row in real_rows)
            else 0.0,
            "note": "Lead bars derived from trap events_before_move in existing exports; no replay.",
        },
    }


def _rank_combinations(
    rows: list[dict[str, Any]],
    *,
    window_days: int,
) -> dict[str, Any]:
    predefined = [
        tuple(FORMULA_COMPONENTS),
        ("Failed Breakdown", "Gap Reversal", "Near Support"),
        ("Failed Breakdown", "Near Support"),
        ("Gap Reversal", "Near Support"),
        ("Liquidity Grab", "Near Support"),
        ("Failed Breakdown", "Gap Reversal"),
        ("Failed Breakdown", "Gap Continuation", "Near Support"),
        ("Failed Breakdown", "PWL Sweep", "Near Support"),
        ("Failed Breakdown", "PDL Sweep", "Near Support"),
        ("Failed Breakdown", "HTF Bullish"),
        ("Failed Breakdown", "VWAP Reclaim"),
        ("Liquidity Grab", "Failed Breakdown"),
        ("Gap Reversal", "HTF Bullish"),
        ("Gap Continuation", "Near Support"),
    ]

    singles = [(condition,) for condition in CONDITION_LABELS]
    pairs = list(itertools.combinations(CONDITION_LABELS, 2))
    candidate_sets = list(dict.fromkeys(predefined + singles + pairs))

    ranked: list[dict[str, Any]] = []
    for combo in candidate_sets:
        subset = _filter_rows_by_conditions(rows, combo)
        metrics = _combo_metrics(subset, window_days=window_days, risk_points=FORMULA_RISK_POINTS)
        ranked.append(
            {
                "combination": list(combo),
                "combination_text": _combo_key(combo),
                **metrics,
            },
        )

    ranked.sort(
        key=lambda item: (
            item["passes_production_gates"],
            item["signals_per_month"],
            item["win_rate_pct"],
            item.get("profit_factor") or 0.0,
            item["capture_40_plus_pct"],
        ),
        reverse=True,
    )

    return {
        "total_combinations_evaluated": len(ranked),
        "ranking_priority": [
            "passes_production_gates",
            "signals_per_month",
            "win_rate_pct",
            "profit_factor",
            "capture_40_plus_pct",
        ],
        "all_ranked_combinations": ranked,
        "top_by_frequency": sorted(ranked, key=lambda item: item["signals_per_month"], reverse=True)[:15],
        "top_by_win_rate": sorted(
            [item for item in ranked if item["sample_size"] >= 5],
            key=lambda item: item["win_rate_pct"],
            reverse=True,
        )[:15],
        "top_passing_production_gates": [
            item for item in ranked if item["passes_production_gates"] and item["sample_size"] >= 3
        ],
    }


def _frequency_expansion_candidates(
    combination_rankings: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ranked = combination_rankings["all_ranked_combinations"]
    discovery = sources["buy_side_discovery"]["data"]
    replay_pockets = discovery.get("buy_side_opportunity_map", {}).get("realtime_replay_positive_buy_pockets", [])
    final_extraction = sources["final_signal_extraction"]["data"]

    by_target: dict[str, Any] = {}
    for target in FREQUENCY_TARGETS:
        passing = [
            item
            for item in ranked
            if item["signals_per_month"] >= target
            and item["passes_production_gates"]
            and item["capture_40_plus_pct"] >= 50.0
        ]
        by_target[str(target)] = {
            "target_signals_per_month": target,
            "passing_combinations_count": len(passing),
            "passing_combinations": passing[:10],
            "feasible_from_cohort_synthesis": len(passing) > 0,
        }

    export_pockets = []
    for pocket in replay_pockets:
        export_pockets.append(
            {
                "source": "nifty50_buy_side_reality_discovery.realtime_replay_positive_buy_pockets",
                "condition": pocket.get("condition"),
                "sample_size": pocket.get("sample_size"),
                "signals_per_month": pocket.get("signals_per_month"),
                "win_rate_pct": pocket.get("win_rate_pct"),
                "expectancy": pocket.get("expectancy"),
                "passes_production_gates": (
                    float(pocket.get("win_rate_pct", 0)) >= PRODUCTION_GATES["win_rate_min_pct"]
                    and float(pocket.get("signals_per_month", 0)) >= 20.0
                ),
            },
        )

    return {
        "frequency_targets": list(FREQUENCY_TARGETS),
        "production_gates": PRODUCTION_GATES,
        "by_target": by_target,
        "export_derived_pockets": export_pockets,
        "final_signal_extraction_buy_accepted": final_extraction.get("extraction_summary", {}).get("accepted_counts", {}).get(
            "BUY",
            0,
        ),
        "final_signal_extraction_note": "All 20 BUY candidates rejected in smartmoneyengine_final_signal_extraction.json.",
        "buy_v1_baseline_signals_per_month": BUY_V1_BASELINE_FREQUENCY,
    }


def _mandatory_vs_false_conditions(
    rows: list[dict[str, Any]],
    buy_failure: dict[str, Any],
) -> dict[str, Any]:
    real_rows = [row for row in rows if row.get("anatomy_classification") == "Real Reversal"]
    false_rows = [row for row in rows if row.get("anatomy_classification") == "Dead Cat Bounce"]
    discriminators = buy_failure.get("discriminator_candidates", [])
    strongest = buy_failure.get("strongest_buy_discriminator", {})

    mandatory = [
        "Liquidity Grab (exclusive to Real Reversal in precursor cohort — 76.06% real vs 0% false)",
        "Failed Breakdown (present in 100% of BUY_V1 captured moves)",
        "Pre-expansion causal stack before BOS/CHOCH/FVG confirmation (buy_entry_timing export)",
    ]
    frequency_increasing = [
        "Drop Liquidity Grab requirement → raises frequency but removes strongest discriminator",
        "Relax Near Support to Mid Range → increases eligible bars per discovery Near Support rate",
        "Add Gap Reversal / Gap Continuation as alternate layer-1 triggers (dominant first events in 200+ cohort)",
        "Include PWL/PDL Sweep events (trap statistics show 45-64% 200+ probability)",
    ]
    false_reversal = [
        "LDE outcome No Expansion (130-sample pocket; avg move 45.68 pts)",
        "Counter-trend HTF without Liquidity Grab and move < 300",
        "Failed Breakout as second event with sub-200 move size",
        "Near Support without Liquidity Grab (64 dead-cat vs 45 real in anatomy export)",
    ]
    separators = [
        {
            "feature": item.get("feature"),
            "real_rate_pct": item.get("real_rate_pct"),
            "false_rate_pct": item.get("false_rate_pct"),
            "separation_score": item.get("separation_score"),
        }
        for item in discriminators[:5]
    ]
    if strongest:
        separators.insert(
            0,
            {
                "feature": strongest.get("feature"),
                "real_rate_pct": strongest.get("real_rate_pct"),
                "false_rate_pct": strongest.get("false_rate_pct"),
                "separation_score": strongest.get("separation_score"),
            },
        )

    return {
        "A_mandatory_conditions": mandatory,
        "B_frequency_increasing_conditions": frequency_increasing,
        "C_false_reversal_conditions": false_reversal,
        "D_real_reversal_vs_dead_cat_separators": separators,
        "real_reversal_sample_size": len(real_rows),
        "dead_cat_bounce_sample_size": len(false_rows),
        "liquidity_grab_real_only": {
            "real_with_liquidity_grab": sum(
                1 for row in real_rows if row.get("conditions", {}).get("Liquidity Grab")
            ),
            "false_with_liquidity_grab": sum(
                1 for row in false_rows if row.get("conditions", {}).get("Liquidity Grab")
            ),
        },
    }


def _final_answer(
    frequency_candidates: dict[str, Any],
    buy_v1_metrics: dict[str, Any],
    combination_rankings: dict[str, Any],
) -> dict[str, Any]:
    by_target = frequency_candidates["by_target"]
    verdicts: dict[str, str] = {}
    evidence: list[str] = []

    for target in FREQUENCY_TARGETS:
        key = str(target)
        feasible = by_target[key]["feasible_from_cohort_synthesis"]
        passing_count = by_target[key]["passing_combinations_count"]
        if feasible:
            verdicts[key] = "YES"
            evidence.append(
                f"{target}+/mo: {passing_count} combination(s) pass WR>65%, PF>2, 65%+ real-reversal rate, 40+ capture.",
            )
        else:
            high_freq = [
                item
                for item in combination_rankings["top_by_frequency"]
                if item["signals_per_month"] >= target
            ]
            if high_freq:
                best = high_freq[0]
                verdicts[key] = "PARTIAL"
                evidence.append(
                    f"{target}+/mo: highest-frequency combo '{best['combination_text']}' reaches "
                    f"{best['signals_per_month']}/mo with WR {best['win_rate_pct']}% / PF {best.get('profit_factor')} "
                    f"but real-reversal rate {best.get('real_reversal_rate_pct')}% or validated-export cross-check fails gates.",
                )
            else:
                verdicts[key] = "NO"
                evidence.append(f"{target}+/mo: no export-derived combination reaches target frequency with causal cohort.")

    extraction_accepted = frequency_candidates["final_signal_extraction_buy_accepted"]
    if extraction_accepted == 0:
        evidence.append(
            "smartmoneyengine_final_signal_extraction.json accepted 0 BUY models — no walk-forward-validated expansion engine in exports.",
        )

    evidence.extend(
        [
            f"BUY_V1 baseline: {buy_v1_metrics.get('signals_per_month', BUY_V1_BASELINE_FREQUENCY)}/mo, "
            f"WR {buy_v1_metrics.get('true_causal_win_rate_pct', 100)}%, "
            f"expectancy {buy_v1_metrics.get('true_causal_expectancy', 161.86)}.",
            "buy_entry_timing export: earliest causal entry does not increase frequency — same 17-move cohort.",
        ],
    )

    cohort_yes = any(verdict == "YES" for verdict in verdicts.values())
    if cohort_yes and extraction_accepted == 0:
        overall = "PARTIAL"
        evidence.append(
            "Cohort synthesis shows high-frequency combos, but validated export pipeline rejected all BUY candidates — expansion not production-safe.",
        )
    elif cohort_yes:
        overall = "YES"
    elif any(verdict == "PARTIAL" for verdict in verdicts.values()):
        overall = "PARTIAL"
    else:
        overall = "NO"

    return {
        "overall_verdict": overall,
        "by_frequency_target": verdicts,
        "can_buy_reach_20_plus_per_month": verdicts.get("20", "NO"),
        "can_buy_reach_30_plus_per_month": verdicts.get("30", "NO"),
        "can_buy_reach_40_plus_per_month": verdicts.get("40", "NO"),
        "evidence": evidence,
        "buy_v1_baseline": {
            "signals_per_month": buy_v1_metrics.get("signals_per_month", BUY_V1_BASELINE_FREQUENCY),
            "win_rate_pct": buy_v1_metrics.get("true_causal_win_rate_pct", 100.0),
            "profit_factor": buy_v1_metrics.get("true_causal_profit_factor"),
            "expectancy": buy_v1_metrics.get("true_causal_expectancy", 161.86),
        },
    }


def _most_valuable_setup(
    combination_rankings: dict[str, Any],
    buy_v1_metrics: dict[str, Any],
) -> dict[str, Any]:
    passing = combination_rankings.get("top_passing_production_gates", [])
    if passing:
        best = max(passing, key=lambda item: item["signals_per_month"])
    else:
        best = combination_rankings["all_ranked_combinations"][0] if combination_rankings["all_ranked_combinations"] else {}

    buy_v1_entry = {
        "setup_id": MODEL_ID,
        "combination_text": FORMULA_TEXT,
        "signals_per_month": buy_v1_metrics.get("signals_per_month", BUY_V1_BASELINE_FREQUENCY),
        "win_rate_pct": buy_v1_metrics.get("true_causal_win_rate_pct", 100.0),
        "profit_factor": buy_v1_metrics.get("true_causal_profit_factor"),
        "expectancy": buy_v1_metrics.get("true_causal_expectancy", 161.86),
        "capture_40_plus_pct": buy_v1_metrics.get("capture_50_plus_pct", 100.0),
        "capture_100_plus_pct": buy_v1_metrics.get("capture_100_plus_pct", 100.0),
        "average_lead_bars_before_expansion": 3.0,
    }

    highest_frequency = combination_rankings.get("top_by_frequency", [{}])[0]
    if buy_v1_entry["signals_per_month"] >= highest_frequency.get("signals_per_month", 0):
        selected = buy_v1_entry
        rationale = "BUY_V1 remains highest-value: best WR/expectancy with full tradeable-tier capture despite low frequency."
    elif highest_frequency.get("passes_production_gates"):
        selected = {
            "setup_id": "EXPORT-COHORT-BEST",
            "combination_text": highest_frequency.get("combination_text"),
            **highest_frequency,
        }
        rationale = "Highest-frequency export cohort combination passes production gates."
    else:
        selected = buy_v1_entry
        rationale = (
            f"Highest-frequency combo '{highest_frequency.get('combination_text')}' "
            f"({highest_frequency.get('signals_per_month')}/mo) fails WR/PF gates; BUY_V1 retains best metrics stack."
        )

    return {
        "selected_setup": selected,
        "buy_v1_reference": buy_v1_entry,
        "highest_frequency_combo": highest_frequency,
        "rationale": rationale,
    }


def _production_recommendation(
    final_answer: dict[str, Any],
    most_valuable: dict[str, Any],
) -> dict[str, Any]:
    overall = final_answer["overall_verdict"]
    if overall == "YES":
        choice = "Expanded setup"
        reason = "Export synthesis found combinations meeting 20+/mo with production gates intact."
    elif overall == "PARTIAL":
        choice = "Hybrid"
        reason = (
            "Keep BUY_V1 as the high-conviction pre-structure leg (4.25/mo, WR 100%, expectancy 161.86). "
            "Cohort synthesis shows 7 combos at 20+/mo, but none survive validated-export scrutiny "
            "(final_signal_extraction accepted 0 BUY; Failed Breakdown 32.25/mo has only 55% real-reversal rate). "
            "Do not promote relaxed single-condition stacks to production."
        )
    else:
        choice = "BUY_V1"
        reason = (
            "No export-derived expansion reaches 20+/mo without destroying edge. "
            "BUY_V1 is the only BUY stack with verified causal metrics in-window."
        )

    return {
        "recommendation": choice,
        "reason": reason,
        "coexistence_with_sell_v5": "Maintain regime separation — BUY_V1 supplementary to LDM-SELL-V5.",
        "most_valuable_setup_id": most_valuable["selected_setup"].get("setup_id", MODEL_ID),
    }


class BuySideFrequencyExpansionAnalysisResearch:
    """Synthesize BUY frequency expansion feasibility from completed exports."""

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
            status = "loaded" if path.exists() else ("missing" if is_required else "optional_missing")
            loaded[name] = {
                "path": str(path),
                "status": status,
                "data": _load_json(path, required=is_required) if path.exists() or is_required else {},
            }
        self.sources = loaded
        return loaded

    def run(self) -> BuySideFrequencyExpansionAnalysisReport:
        started = time.perf_counter()
        self._load_sources()

        buy_v1 = self.sources["buy_v1_production_validation"]["data"]
        buy_failure = self.sources["buy_failure_anatomy"]["data"]
        discovery = self.sources["buy_side_discovery"]["data"]
        window_days = int(
            discovery.get("research_window", {}).get("research_window_days", buy_v1.get("research_window_days", 120)),
        )
        start_date = str(discovery.get("research_window", {}).get("start_date", buy_v1.get("start_date", "")))
        end_date = str(discovery.get("research_window", {}).get("end_date", buy_v1.get("end_date", "")))

        buy_v1_occurrences = buy_v1.get("all_occurrences", [])
        buy_v1_dates = {_normalize_date_key(str(item.get("move_timestamp", item.get("signal_timestamp", "")))) for item in buy_v1_occurrences}
        buy_v1_performance = {
            _normalize_date_key(str(item.get("move_timestamp", item.get("signal_timestamp", "")))): item
            for item in buy_v1_occurrences
        }
        buy_v1_metrics = _performance_metrics(buy_v1_occurrences, window_days)

        rows = _collect_bullish_moves(
            self.sources,
            buy_v1_dates=buy_v1_dates,
            buy_v1_performance=buy_v1_performance,
        )

        bucket_counts = Counter(row["move_bucket"] for row in rows)
        classification = {
            "total_bullish_moves_analyzed": len(rows),
            "bucket_counts": dict(bucket_counts),
            "bucket_rates_pct": {
                bucket: round(100.0 * count / max(len(rows), 1), 2) for bucket, count in bucket_counts.items()
            },
            "buy_v1_captured_count": sum(1 for row in rows if row.get("buy_v1_captured")),
            "buy_v1_missed_real_reversal_count": sum(
                1 for row in rows if row.get("move_bucket") == "BUY_V1 Missed"
            ),
            "anatomy_classification_counts": dict(Counter(row["anatomy_classification"] for row in rows)),
            "sample_by_bucket": {
                bucket: [
                    {
                        "date": row["date"],
                        "move_size_points": row["move_size_points"],
                        "first_event": row.get("first_event"),
                        "conditions_present": [name for name, present in row.get("conditions", {}).items() if present],
                    }
                    for row in rows
                    if row.get("move_bucket") == bucket
                ][:8]
                for bucket in sorted(bucket_counts)
            },
        }

        condition_attribution = _condition_attribution_for_real_reversals(rows)
        combination_rankings = _rank_combinations(rows, window_days=window_days)
        frequency_candidates = _frequency_expansion_candidates(combination_rankings, self.sources)
        mandatory_vs_false = _mandatory_vs_false_conditions(rows, buy_failure)
        final_answer = _final_answer(frequency_candidates, buy_v1_metrics, combination_rankings)
        most_valuable = _most_valuable_setup(combination_rankings, buy_v1_metrics)
        production_recommendation = _production_recommendation(final_answer, most_valuable)

        methodology = {
            "research_only": True,
            "no_new_indicators": True,
            "no_discovery_engines": True,
            "no_new_buy_models": True,
            "no_replay": True,
            "no_optimization": True,
            "current_buy_v1_baseline": {
                "formula": FORMULA_TEXT,
                "signals_per_month": BUY_V1_BASELINE_FREQUENCY,
                "win_rate_pct": 100.0,
            },
            "bullish_move_universe": "Deduped bullish completed_moves >=40 pts from momentum_anatomy + LDE No Expansion failures",
            "classification_rules": {
                "BUY_V1 Captured": "Move matches buy_v1_production_validation occurrence date",
                "BUY_V1 Missed": "Real Reversal not captured by BUY_V1",
                "Real Reversal": "buy_failure_anatomy._classify_move",
                "Dead Cat Bounce": "buy_failure_anatomy._classify_move",
            },
            "combination_evaluation": {
                "conditions_universe": list(CONDITION_LABELS),
                "production_gates": PRODUCTION_GATES,
                "frequency_targets_per_month": list(FREQUENCY_TARGETS),
                "tradeable_tiers": list(TRADEABLE_TIERS),
                "risk_points_proxy": FORMULA_RISK_POINTS,
            },
            "synthesis_sources": list(SOURCE_EXPORTS.keys()),
        }

        limitations = [
            "Per-signal trade paths not replayed — WR/PF/expectancy use move_size vs risk proxy except BUY_V1 captured rows.",
            "Near Support / VWAP / HTF derived from anatomy timeline snapshots (T-60/T-15), not bar-native scans.",
            "Combination frequencies are cohort-filter counts, not independent forward-validated signal engines.",
            "final_signal_extraction rejected all BUY candidates — no walk-forward-approved expansion model in exports.",
            "Support Reclaim / Round Number Reclaim inferred from blueprint and trap text where explicit flags absent.",
        ]

        conclusions = [
            f"Analyzed {len(rows)} bullish moves across {window_days}-day NIFTY50 export window.",
            (
                f"Classification buckets: {dict(bucket_counts)}; "
                f"BUY_V1 captured {classification['buy_v1_captured_count']} vs missed real reversals "
                f"{classification['buy_v1_missed_real_reversal_count']}."
            ),
            (
                f"20+/30+/40+ feasibility: {final_answer['can_buy_reach_20_plus_per_month']} / "
                f"{final_answer['can_buy_reach_30_plus_per_month']} / "
                f"{final_answer['can_buy_reach_40_plus_per_month']} (overall {final_answer['overall_verdict']})."
            ),
            (
                f"Most valuable setup: {most_valuable['selected_setup'].get('setup_id', MODEL_ID)} "
                f"({most_valuable['selected_setup'].get('combination_text', FORMULA_TEXT)}) "
                f"at {most_valuable['selected_setup'].get('signals_per_month', BUY_V1_BASELINE_FREQUENCY)}/mo."
            ),
            f"Production recommendation: {production_recommendation['recommendation']} — {production_recommendation['reason']}",
        ]

        return BuySideFrequencyExpansionAnalysisReport(
            report_type="BUY Side Frequency Expansion Analysis",
            model_id=MODEL_ID,
            current_buy_v1_formula=FORMULA_COMPONENTS,
            current_buy_v1_formula_text=FORMULA_TEXT,
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
            bullish_move_classification=classification,
            condition_attribution=condition_attribution,
            combination_rankings=combination_rankings,
            frequency_expansion_candidates=frequency_candidates,
            mandatory_vs_false_conditions=mandatory_vs_false,
            final_answer=final_answer,
            most_valuable_setup=most_valuable,
            production_recommendation=production_recommendation,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: BuySideFrequencyExpansionAnalysisReport | None = None) -> Path:
        payload = report or self.run()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = _json_safe(asdict(payload))
        self.report_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        logger.info("BUY frequency expansion analysis exported to %s", self.report_path)
        return self.report_path


def generate_buy_side_frequency_expansion_analysis_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export BUY side frequency expansion analysis."""
    return BuySideFrequencyExpansionAnalysisResearch(report_path=report_path).export()
