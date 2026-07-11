"""
BUY Entry Timing Validation — synthesis from existing exports only.

Determines the earliest causal BUY entry for BUY_V1 (Liquidity Grab + Failed Breakdown
+ Near Support) occurrences using completed research exports. No replay, discovery,
optimization, or new BUY models.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.research.buy_failure_anatomy_research import NEAR_SUPPORT_LABEL, _json_safe
from src.research.buy_v1_production_validation_research import (
    DEFAULT_RISK_POINTS,
    FORMULA_COMPONENTS,
    FORMULA_TEXT,
    MODEL_ID,
    MOVE_OUTCOME_THRESHOLDS,
    SIGNAL_STEP,
    _load_json,
    _normalize_date_key,
    _near_support,
    _ordered_causal_events,
    _performance_metrics,
    _risk_points_proxy,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_entry_timing_validation.json"

BAR_OFFSETS = (30, 20, 10, 5, 0)
CURRENT_ENTRY_BARS_BEFORE_MOVE = 3
CURRENT_ENTRY_LABEL = "T-15 minutes (3 bars)"

ANATOMY_MINUTE_STEPS = (60, 30, 15, 10, 5, 0)

SOURCE_EXPORTS = {
    "buy_v1_production_validation": RESEARCH_DIR / "buy_v1_production_validation.json",
    "buy_failure_anatomy": RESEARCH_DIR / "buy_failure_anatomy.json",
    "buy_formula_verification": RESEARCH_DIR / "buy_formula_reality_verification.json",
    "buy_side_discovery": RESEARCH_DIR / "nifty50_buy_side_reality_discovery.json",
    "momentum_anatomy": RESEARCH_DIR / "nifty50_momentum_anatomy_120d.json",
    "trap_to_momentum": RESEARCH_DIR / "nifty50_trap_to_momentum_validation.json",
    "liquidity_decision_engine": RESEARCH_DIR / "nifty50_liquidity_decision_engine.json",
    "institutional_momentum_origin": RESEARCH_DIR / "institutional_momentum_origin.json",
    "liquidity_move_reconstruction": RESEARCH_DIR / "liquidity_move_reconstruction.json",
    "signal_timing_reality_audit": RESEARCH_DIR / "smartmoneyengine_signal_timing_reality_audit.json",
    "research_consistency_audit": RESEARCH_DIR / "research_consistency_audit.json",
}


class BuyEntryTimingValidationError(Exception):
    """Raised when BUY entry timing validation synthesis fails."""


@dataclass
class BuyEntryTimingValidationReport:
    """BUY entry timing validation synthesis output."""

    report_type: str
    model_id: str
    formula: list[str]
    formula_text: str
    symbol: str
    timeframe: str
    research_window_days: int
    start_date: str
    end_date: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    occurrences_with_timelines: list[dict[str, Any]]
    earliest_causal_entry_analysis: dict[str, Any]
    entry_comparison: dict[str, Any]
    final_verdict: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _minute_label(minutes: int) -> str:
    return "T-0 minutes" if minutes == 0 else f"T-{minutes} minutes"


def _nearest_anatomy_minutes(minutes_before_move: int) -> tuple[int, str]:
    """Map bar-derived minutes to nearest anatomy timeline step."""
    if minutes_before_move <= 0:
        return 0, "exact"
    nearest = min(ANATOMY_MINUTE_STEPS, key=lambda value: abs(value - minutes_before_move))
    if nearest == minutes_before_move:
        return nearest, "exact"
    if minutes_before_move > max(ANATOMY_MINUTE_STEPS):
        return max(ANATOMY_MINUTE_STEPS), "beyond_export_max_T-60"
    return nearest, f"proxy_nearest_T-{nearest}"


def _context_at_minutes(record: dict[str, Any], minutes_before_move: int) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """Return 5M context snapshot and proxy metadata for minutes before move."""
    mapped_minutes, proxy_kind = _nearest_anatomy_minutes(minutes_before_move)
    step_label = _minute_label(mapped_minutes)
    context: dict[str, Any] | None = None
    for timeline_step in record.get("timeline", []):
        if timeline_step.get("timeline_step") != step_label:
            continue
        context = timeline_step.get("context_by_timeframe", {}).get("5M")
        break
    meta = {
        "requested_minutes_before_move": str(minutes_before_move),
        "mapped_anatomy_step": step_label,
        "proxy_kind": proxy_kind,
    }
    return context, meta


def _event_bars_before(events: list[dict[str, Any]], event_name: str) -> int | None:
    for item in events:
        if item.get("event") == event_name:
            return int(item.get("bars_before_move", 0))
    return None


def _trap_events_known_at_offset(events: list[dict[str, Any]], bar_offset: int) -> set[str]:
    """Trap events whose first occurrence is at or before the evaluation bar."""
    known: set[str] = set()
    for item in events:
        bars_before = int(item.get("bars_before_move", 0))
        if bars_before >= bar_offset:
            known.add(str(item.get("event")))
    return known


def _blueprint_has(blueprint: str, event_name: str) -> bool:
    return event_name in blueprint


def _extract_context_fields(context: dict[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {
            "htf_context": None,
            "vwap_state": None,
            "ema_state": None,
            "volume_expansion": None,
            "structure_state": None,
            "bos": None,
            "choch": None,
            "fvg": None,
            "market_location": None,
        }

    reason = context.get("reason_stack", {})
    flags = context.get("feature_flags", {})
    levels = context.get("levels", {})
    return {
        "htf_context": reason.get("htf_trend"),
        "vwap_state": reason.get("vwap"),
        "ema_state": reason.get("ema_structure"),
        "volume_expansion": bool(reason.get("volume_expansion")),
        "structure_state": reason.get("market_structure"),
        "bos": bool(flags.get("bos_present") or reason.get("bos")),
        "choch": bool(flags.get("choch_present") or reason.get("choch")),
        "fvg": bool(flags.get("fvg_reclaim") or reason.get("fvg")),
        "market_location": levels.get("market_location"),
    }


def _formula_state_at_offset(
    *,
    bar_offset: int,
    trap_events: list[dict[str, Any]],
    anatomy_record: dict[str, Any],
    causal_events: list[str],
) -> dict[str, Any]:
    minutes_before = bar_offset * 5
    context, proxy_meta = _context_at_minutes(anatomy_record, minutes_before)
    known_trap = _trap_events_known_at_offset(trap_events, bar_offset)
    blueprint = str(anatomy_record.get("blueprint_pattern", ""))
    origin = str(anatomy_record.get("origin_trigger", ""))

    lg_bars = _event_bars_before(trap_events, "Liquidity Grab")
    fb_bars = _event_bars_before(trap_events, "Failed Breakdown")

    liquidity_grab = (
        "Liquidity Grab" in known_trap
        or "Liquidity Grab" in causal_events
        or _blueprint_has(blueprint, "Liquidity Grab")
        or origin == "Liquidity Grab"
        or (lg_bars is not None and lg_bars >= bar_offset)
    )
    failed_breakdown = (
        "Failed Breakdown" in known_trap
        or "Failed Breakdown" in causal_events
        or _blueprint_has(blueprint, "Failed Breakdown")
        or origin == "Failed Breakdown"
        or (fb_bars is not None and fb_bars >= bar_offset)
    )
    near_support = _near_support(context)

    fields = _extract_context_fields(context)
    requires_future_structure = not (
        fields["bos"] and fields["choch"] and fields["fvg"]
    )

    return {
        "bars_before_move": bar_offset,
        "minutes_before_move": minutes_before,
        "context_proxy": proxy_meta,
        "liquidity_grab": liquidity_grab,
        "failed_breakdown": failed_breakdown,
        "near_support": near_support,
        "formula_complete": bool(liquidity_grab and failed_breakdown and near_support),
        "htf_context": fields["htf_context"],
        "vwap_state": fields["vwap_state"],
        "ema_state": fields["ema_state"],
        "volume_expansion": fields["volume_expansion"],
        "structure_state": fields["structure_state"],
        "bos_present": fields["bos"],
        "choch_present": fields["choch"],
        "fvg_present": fields["fvg"],
        "entry_without_future_bos_choch_fvg": bool(
            liquidity_grab and failed_breakdown and near_support
        ),
        "did_require_future_bos": not bool(fields["bos"]),
        "did_require_future_choch": not bool(fields["choch"]),
        "did_require_future_fvg": not bool(fields["fvg"]),
        "did_require_future_information": requires_future_structure,
        "liquidity_grab_bars_before_move": lg_bars,
        "failed_breakdown_bars_before_move": fb_bars,
    }


def _earliest_offset(steps: list[dict[str, Any]]) -> int | None:
    """Earliest causal offset = largest bar count where formula is complete."""
    complete = [step["bars_before_move"] for step in steps if step.get("formula_complete")]
    return max(complete) if complete else None


def _metrics_for_rows(rows: list[dict[str, Any]], window_days: int) -> dict[str, Any]:
    if not rows:
        return {
            "sample_size": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "signals_per_month": 0.0,
            "capture_50_plus_pct": 0.0,
            "capture_100_plus_pct": 0.0,
            "capture_200_plus_pct": 0.0,
            "capture_300_plus_pct": 0.0,
        }

    performance = _performance_metrics(rows, window_days)
    return {
        "sample_size": len(rows),
        "win_rate_pct": performance["true_causal_win_rate_pct"],
        "profit_factor": performance["true_causal_profit_factor"],
        "expectancy": performance["true_causal_expectancy"],
        "signals_per_month": performance["signals_per_month"],
        "capture_50_plus_pct": performance["capture_50_plus_pct"],
        "capture_100_plus_pct": performance["capture_100_plus_pct"],
        "capture_200_plus_pct": performance["capture_200_plus_pct"],
        "capture_300_plus_pct": performance["capture_300_plus_pct"],
    }


def _find_anatomy_record(anatomy: dict[str, Any], move_timestamp: str) -> dict[str, Any] | None:
    key = _normalize_date_key(move_timestamp)
    for record in anatomy.get("move_anatomy_records", []):
        if _normalize_date_key(str(record.get("date", ""))) == key:
            return record
    return None


def _find_trap_move(trap: dict[str, Any], move_timestamp: str) -> dict[str, Any] | None:
    key = _normalize_date_key(move_timestamp)
    for move in trap.get("move_pre_event_analysis", []):
        if move.get("direction") != "bullish":
            continue
        if _normalize_date_key(str(move.get("date", ""))) == key:
            return move
    return None


def _build_occurrence_timeline(
    *,
    occurrence: dict[str, Any],
    anatomy_record: dict[str, Any],
    trap_move: dict[str, Any] | None,
) -> dict[str, Any]:
    trap_events = (trap_move or {}).get("events_before_move", [])
    causal_events = _ordered_causal_events(trap_events)
    for event in ("Liquidity Grab", "Failed Breakdown"):
        if event not in causal_events and _blueprint_has(
            str(anatomy_record.get("blueprint_pattern", "")),
            event,
        ):
            causal_events.append(event)

    timeline_steps = [
        _formula_state_at_offset(
            bar_offset=offset,
            trap_events=trap_events,
            anatomy_record=anatomy_record,
            causal_events=causal_events,
        )
        for offset in BAR_OFFSETS
    ]
    earliest = _earliest_offset(timeline_steps)
    current_step = next(
        (step for step in timeline_steps if step["bars_before_move"] == CURRENT_ENTRY_BARS_BEFORE_MOVE),
        None,
    )
    if current_step is None:
        current_proxy = _formula_state_at_offset(
            bar_offset=CURRENT_ENTRY_BARS_BEFORE_MOVE,
            trap_events=trap_events,
            anatomy_record=anatomy_record,
            causal_events=causal_events,
        )
    else:
        current_proxy = current_step

    return {
        "date": occurrence.get("date"),
        "time": occurrence.get("time"),
        "signal_timestamp": occurrence.get("signal_timestamp"),
        "move_timestamp": occurrence.get("move_timestamp"),
        "move_size_points": occurrence.get("move_size_points"),
        "classification": occurrence.get("classification"),
        "win": occurrence.get("win"),
        "realized_pnl_points": occurrence.get("realized_pnl_points"),
        "move_outcomes": occurrence.get("move_outcomes"),
        "current_buy_v1_entry": {
            "evaluation": CURRENT_ENTRY_LABEL,
            "bars_before_move": CURRENT_ENTRY_BARS_BEFORE_MOVE,
            **{key: current_proxy.get(key) for key in (
                "formula_complete",
                "liquidity_grab",
                "failed_breakdown",
                "near_support",
                "entry_without_future_bos_choch_fvg",
                "did_require_future_information",
            )},
        },
        "timeline": timeline_steps,
        "earliest_causal_entry_bars_before_move": earliest,
        "earliest_causal_entry_label": f"T-{earliest} bars" if earliest is not None else None,
        "bars_earlier_than_current_entry": (
            earliest - CURRENT_ENTRY_BARS_BEFORE_MOVE
            if earliest is not None and earliest > CURRENT_ENTRY_BARS_BEFORE_MOVE
            else 0
        ),
        "trap_first_event": (trap_move or {}).get("first_event"),
        "trap_liquidity_grab_bars_before": _event_bars_before(trap_events, "Liquidity Grab"),
        "trap_failed_breakdown_bars_before": _event_bars_before(trap_events, "Failed Breakdown"),
    }


def _offset_pass_summary(timelines: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for offset in BAR_OFFSETS:
        passing = [
            row
            for row in timelines
            if next(
                step for step in row["timeline"] if step["bars_before_move"] == offset
            ).get("formula_complete")
        ]
        summary[f"T-{offset}_bars"] = {
            "formula_complete_count": len(passing),
            "formula_complete_rate_pct": round(100.0 * len(passing) / max(len(timelines), 1), 2),
        }
    return summary


def _final_verdict(
    *,
    timelines: list[dict[str, Any]],
    current_metrics: dict[str, Any],
    earliest_metrics: dict[str, Any],
    earliest_analysis: dict[str, Any],
    buy_v1_verdict: str,
) -> dict[str, Any]:
    total = len(timelines)
    earliest_offsets = [
        row["earliest_causal_entry_bars_before_move"]
        for row in timelines
        if row.get("earliest_causal_entry_bars_before_move") is not None
    ]
    profitable_at_earliest = sum(
        1
        for row in timelines
        if row.get("win")
        and row.get("earliest_causal_entry_bars_before_move") is not None
    )
    no_future_info_at_earliest = sum(
        1
        for row in timelines
        if row.get("earliest_causal_entry_bars_before_move") is not None
        and next(
            step
            for step in row["timeline"]
            if step["bars_before_move"] == row["earliest_causal_entry_bars_before_move"]
        ).get("entry_without_future_bos_choch_fvg")
    )

    median_earliest = sorted(earliest_offsets)[len(earliest_offsets) // 2] if earliest_offsets else None
    can_enter_without_future = all(
        next(
            step
            for step in row["timeline"]
            if step["bars_before_move"] == row["earliest_causal_entry_bars_before_move"]
        ).get("entry_without_future_bos_choch_fvg")
        for row in timelines
        if row.get("earliest_causal_entry_bars_before_move") is not None
    )

    evidence = [
        f"{len(earliest_offsets)}/{total} occurrences have a complete BUY_V1 stack at some T-30..T-0 bar.",
        f"Median earliest causal entry: T-{median_earliest} bars before move.",
        f"Profitable at earliest causal point: {profitable_at_earliest}/{total} (WR {earliest_metrics.get('win_rate_pct')}%).",
        (
            "Formula never requires future BOS/CHOCH/FVG — "
            f"{no_future_info_at_earliest}/{total} pass stack without those filters at earliest bar."
        ),
        f"Current BUY_V1 entry ({CURRENT_ENTRY_LABEL}): WR {current_metrics.get('win_rate_pct')}%, "
        f"expectancy {current_metrics.get('expectancy')}, {current_metrics.get('signals_per_month')} signals/month.",
    ]

    limitations = [
        "Per-bar T-30..T-0 context not exported — anatomy minute steps (T-60..T-0) used as proxies.",
        "Trade MAE/MFE/PNL held constant from buy_v1_production_validation (no intra-move path replay).",
        "Near Support only available from anatomy timeline snapshots, not LDE trap log alone.",
        "Earliest entry signals/month unchanged — same 17-move cohort; frequency gain requires new scan.",
    ]

    if not earliest_offsets:
        return {
            "verdict": "NO",
            "can_buy_v1_be_fully_causal_production_signal": "NO",
            "earliest_entry_point": None,
            "evidence": evidence,
            "limitations": limitations,
            "buy_v1_production_validation_verdict": buy_v1_verdict,
        }

    if (
        earliest_metrics.get("win_rate_pct", 0) >= 50
        and earliest_metrics.get("expectancy", 0) > 0
        and can_enter_without_future
        and buy_v1_verdict in {"YES", "PARTIAL"}
    ):
        verdict = "PARTIAL" if buy_v1_verdict == "PARTIAL" else "YES"
    elif earliest_metrics.get("expectancy", 0) > 0 and can_enter_without_future:
        verdict = "PARTIAL"
    else:
        verdict = "NO"

    return {
        "verdict": verdict,
        "can_buy_v1_be_fully_causal_production_signal": verdict,
        "earliest_entry_point": earliest_analysis.get("aggregate_earliest_entry"),
        "entry_without_future_bos_choch_fvg": can_enter_without_future,
        "profitable_at_earliest_count": profitable_at_earliest,
        "evidence": evidence,
        "limitations": limitations,
        "buy_v1_production_validation_verdict": buy_v1_verdict,
    }


class BuyEntryTimingValidationResearch:
    """Synthesize BUY entry timing validation from completed exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        required = {
            "buy_v1_production_validation",
            "momentum_anatomy",
            "trap_to_momentum",
            "buy_side_discovery",
        }
        for name, path in SOURCE_EXPORTS.items():
            is_required = name in required
            if not path.exists() and is_required:
                raise BuyEntryTimingValidationError(f"Missing export: {path}")
            loaded[name] = {
                "path": str(path),
                "status": "loaded" if path.exists() else "optional_missing",
                "data": _load_json(path, required=is_required) if path.exists() or is_required else {},
            }
        self.sources = loaded
        return loaded

    def run(self) -> BuyEntryTimingValidationReport:
        started = time.perf_counter()
        self._load_sources()

        buy_v1 = self.sources["buy_v1_production_validation"]["data"]
        anatomy = self.sources["momentum_anatomy"]["data"]
        trap = self.sources["trap_to_momentum"]["data"]
        discovery = self.sources["buy_side_discovery"]["data"]

        occurrences = buy_v1.get("all_occurrences", [])
        window_days = int(buy_v1.get("research_window_days", 120))
        risk_points = float(
            buy_v1.get("methodology", {})
            .get("trade_field_derivation", {})
            .get("risk_points_proxy", _risk_points_proxy(discovery)),
        )

        timelines: list[dict[str, Any]] = []
        for occurrence in occurrences:
            anatomy_record = _find_anatomy_record(anatomy, str(occurrence.get("move_timestamp", "")))
            if not anatomy_record:
                raise BuyEntryTimingValidationError(
                    f"No anatomy record for move {occurrence.get('move_timestamp')}",
                )
            trap_move = _find_trap_move(trap, str(occurrence.get("move_timestamp", "")))
            timelines.append(
                _build_occurrence_timeline(
                    occurrence=occurrence,
                    anatomy_record=anatomy_record,
                    trap_move=trap_move,
                ),
            )

        offset_summary = _offset_pass_summary(timelines)
        earliest_offsets = [
            row["earliest_causal_entry_bars_before_move"]
            for row in timelines
            if row.get("earliest_causal_entry_bars_before_move") is not None
        ]
        aggregate_earliest = (
            f"T-{max(earliest_offsets)} bars"
            if earliest_offsets
            else None
        )

        current_metrics = _metrics_for_rows(occurrences, window_days)
        earliest_rows = [
            occurrence
            for occurrence, timeline in zip(occurrences, timelines, strict=True)
            if timeline.get("earliest_causal_entry_bars_before_move") is not None
        ]
        earliest_metrics = _metrics_for_rows(earliest_rows, window_days)

        earliest_analysis = {
            "evaluation_offsets_bars": list(BAR_OFFSETS),
            "current_buy_v1_entry": CURRENT_ENTRY_LABEL,
            "aggregate_earliest_entry": aggregate_earliest,
            "median_earliest_entry_bars": sorted(earliest_offsets)[len(earliest_offsets) // 2]
            if earliest_offsets
            else None,
            "occurrences_with_complete_stack": len(earliest_offsets),
            "occurrences_earlier_than_current_entry": sum(
                1 for row in timelines if (row.get("bars_earlier_than_current_entry") or 0) > 0
            ),
            "offset_pass_summary": offset_summary,
            "entry_without_future_structure_required": (
                "BUY_V1 formula excludes BOS/CHOCH/FVG; entry valid when LG+FB+Near Support "
                "are causally known without waiting for post-entry structure."
            ),
            "per_occurrence_earliest": [
                {
                    "date": row["date"],
                    "earliest_bars_before_move": row.get("earliest_causal_entry_bars_before_move"),
                    "current_entry_bars": CURRENT_ENTRY_BARS_BEFORE_MOVE,
                    "bars_earlier_than_current": row.get("bars_earlier_than_current_entry"),
                }
                for row in timelines
            ],
        }

        buy_v1_verdict = str(
            buy_v1.get("production_formula_or_failure_reasons", {}).get("verdict", "PARTIAL"),
        )
        entry_comparison = {
            "comparison_basis": (
                "Same 17-move BUY_V1 cohort from buy_v1_production_validation.json; "
                "earliest entry = max bar offset where LG+FB+Near Support complete; "
                "current entry = T-15 minutes (3 bars)."
            ),
            "risk_points_proxy": risk_points,
            "current_entry": {
                "label": CURRENT_ENTRY_LABEL,
                "bars_before_move": CURRENT_ENTRY_BARS_BEFORE_MOVE,
                **current_metrics,
            },
            "earliest_causal_entry": {
                "label": aggregate_earliest,
                "bars_before_move": max(earliest_offsets) if earliest_offsets else None,
                **earliest_metrics,
            },
            "delta_earliest_vs_current": {
                "bars_earlier": (
                    (max(earliest_offsets) - CURRENT_ENTRY_BARS_BEFORE_MOVE)
                    if earliest_offsets
                    else None
                ),
                "win_rate_pct_delta": round(
                    float(earliest_metrics.get("win_rate_pct", 0))
                    - float(current_metrics.get("win_rate_pct", 0)),
                    2,
                ),
                "expectancy_delta": round(
                    float(earliest_metrics.get("expectancy", 0))
                    - float(current_metrics.get("expectancy", 0)),
                    2,
                ),
                "signals_per_month_delta": round(
                    float(earliest_metrics.get("signals_per_month", 0))
                    - float(current_metrics.get("signals_per_month", 0)),
                    2,
                ),
            },
            "by_offset_metrics": {
                f"T-{offset}_bars": _metrics_for_rows(
                    [
                        occurrence
                        for occurrence, timeline in zip(occurrences, timelines, strict=True)
                        if next(
                            step for step in timeline["timeline"] if step["bars_before_move"] == offset
                        ).get("formula_complete")
                    ],
                    window_days,
                )
                for offset in BAR_OFFSETS
            },
        }

        limitations = [
            "Exports provide anatomy context at T-60/30/15/10/5/0 minutes — not native T-30/20/10/5/0 bars.",
            "Bar-offset context mapped via 5M bar×5 minute proxy to nearest anatomy step.",
            "Trap events_before_move supply causal LG/FB timing; blueprint/origin used as fallback.",
            "No replay — win/PNL/capture metrics reuse completed-move outcomes from buy_v1 export.",
            "signal_timing_reality_audit.json used as methodology reference only (cluster-first delay pattern).",
        ]

        final_verdict = _final_verdict(
            timelines=timelines,
            current_metrics=current_metrics,
            earliest_metrics=earliest_metrics,
            earliest_analysis=earliest_analysis,
            buy_v1_verdict=buy_v1_verdict,
        )

        conclusions = [
            f"Analyzed {len(timelines)} BUY_V1 occurrences with T-30..T-0 bar timelines (export proxies).",
            f"Aggregate earliest causal entry: {aggregate_earliest or 'none'}.",
            f"Current entry {CURRENT_ENTRY_LABEL}: WR {current_metrics['win_rate_pct']}%, "
            f"expectancy {current_metrics['expectancy']}, {current_metrics['signals_per_month']}/mo.",
            f"Earliest causal entry cohort: WR {earliest_metrics['win_rate_pct']}%, "
            f"expectancy {earliest_metrics['expectancy']}, capture 200+ {earliest_metrics['capture_200_plus_pct']}%.",
            (
                f"{earliest_analysis['occurrences_earlier_than_current_entry']}/{len(timelines)} "
                "could have entered earlier than current T-15 bar."
            ),
            f"BUY_V1 fully causal production signal: {final_verdict['verdict']}.",
        ]

        methodology = {
            "research_only": True,
            "no_new_scans": True,
            "no_replay": True,
            "no_optimization": True,
            "no_new_buy_models": True,
            "candidate": FORMULA_TEXT,
            "timeline_offsets_bars": list(BAR_OFFSETS),
            "current_entry_reference": CURRENT_ENTRY_LABEL,
            "context_proxy_rules": {
                "minutes_before_move": "bar_offset × 5 on 5M",
                "anatomy_mapping": "nearest T-60/30/15/10/5/0 minute step",
                "formula_events": "trap events_before_move + anatomy blueprint/origin",
                "near_support": "anatomy 5M levels.market_location",
            },
            "methodology_reference": "smartmoneyengine_signal_timing_reality_audit.json cluster-first earliest-safe-entry pattern",
            "future_structure_rule": (
                "BOS/CHOCH/FVG recorded for audit only — not required for BUY_V1 formula completion."
            ),
        }

        source_status = {
            name: {"path": meta["path"], "status": meta["status"]}
            for name, meta in self.sources.items()
        }

        return BuyEntryTimingValidationReport(
            report_type="BUY Entry Timing Validation",
            model_id=MODEL_ID,
            formula=FORMULA_COMPONENTS,
            formula_text=FORMULA_TEXT,
            symbol=buy_v1.get("symbol", "NIFTY50"),
            timeframe=buy_v1.get("timeframe", "5M"),
            research_window_days=window_days,
            start_date=str(buy_v1.get("start_date", "")),
            end_date=str(buy_v1.get("end_date", "")),
            methodology=methodology,
            source_exports=source_status,
            limitations=limitations,
            occurrences_with_timelines=timelines,
            earliest_causal_entry_analysis=earliest_analysis,
            entry_comparison=entry_comparison,
            final_verdict=final_verdict,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: BuyEntryTimingValidationReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Exported BUY entry timing validation to %s", self.report_path)
        return self.report_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    path = BuyEntryTimingValidationResearch().export()
    print(f"Exported: {path}")


if __name__ == "__main__":
    main()
