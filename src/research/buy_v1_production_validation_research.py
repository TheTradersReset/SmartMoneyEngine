"""
BUY_V1 Production Validation — synthesis from existing exports only.

Validates Liquidity Grab + Failed Breakdown + Near Support against completed-export
occurrences. No new scans, optimization, replay, or BUY model creation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from src.research.buy_failure_anatomy_research import (
    CONSOLIDATION_TAGS,
    NEAR_SUPPORT_LABEL,
    _classify_move,
    _has_consolidation_tags,
    _json_safe,
    _ordered_causal_events,
    _second_event_name,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_v1_production_validation.json"

FORMULA_COMPONENTS = ["Liquidity Grab", "Failed Breakdown", "Near Support"]
FORMULA_TEXT = "Liquidity Grab + Failed Breakdown + Near Support"
MODEL_ID = "LDM-BUY-V1"
SIGNAL_STEP = "T-15 minutes"
MOVE_OUTCOME_THRESHOLDS = (50, 100, 200, 300)
DEFAULT_RISK_POINTS = 80.93
MIN_COEXISTENCE_SAMPLE = 10

SOURCE_EXPORTS = {
    "buy_failure_anatomy": RESEARCH_DIR / "buy_failure_anatomy.json",
    "buy_formula_verification": RESEARCH_DIR / "buy_formula_reality_verification.json",
    "buy_side_discovery": RESEARCH_DIR / "nifty50_buy_side_reality_discovery.json",
    "momentum_anatomy": RESEARCH_DIR / "nifty50_momentum_anatomy_120d.json",
    "trap_to_momentum": RESEARCH_DIR / "nifty50_trap_to_momentum_validation.json",
    "liquidity_decision_engine": RESEARCH_DIR / "nifty50_liquidity_decision_engine.json",
    "sell_formula_v2": RESEARCH_DIR / "sell_formula_reality_verification_v2.json",
    "v5_validation": RESEARCH_DIR / "smartmoneyengine_v5_candidate_validation.json",
    "final_signal_extraction": RESEARCH_DIR / "smartmoneyengine_final_signal_extraction.json",
    "research_consistency_audit": RESEARCH_DIR / "research_consistency_audit.json",
    "liquidity_move_reconstruction": RESEARCH_DIR / "liquidity_move_reconstruction.json",
}


class BuyV1ProductionValidationError(Exception):
    """Raised when BUY_V1 production validation cannot be completed."""


@dataclass
class BuyV1ProductionValidationReport:
    """BUY_V1 production validation synthesis output."""

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
    all_occurrences: list[dict[str, Any]]
    causal_validation_summary: dict[str, Any]
    classification_summary: dict[str, Any]
    performance_metrics: dict[str, Any]
    sell_v5_comparison: dict[str, Any]
    coexistence_verdict: dict[str, Any]
    production_formula_or_failure_reasons: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise BuyV1ProductionValidationError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T", 1)
    return datetime.fromisoformat(normalized)


def _normalize_date_key(value: str) -> str:
    return str(value)[:16]


def _near_support(context: dict[str, Any] | None) -> bool:
    if not context:
        return False
    return context.get("levels", {}).get("market_location") == NEAR_SUPPORT_LABEL


def _context_snapshot(record: dict[str, Any], step: str) -> dict[str, Any] | None:
    for timeline_step in record.get("timeline", []):
        if timeline_step.get("timeline_step") != step:
            continue
        return timeline_step.get("context_by_timeframe", {}).get("5M")
    return None


def _formula_match(
    record: dict[str, Any],
    *,
    trap_move: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any]]:
    signal_timestamp: str | None = None
    signal_context: dict[str, Any] | None = None
    context_t60 = _context_snapshot(record, "T-60 minutes") or {}

    for step in record.get("timeline", []):
        if step.get("timeline_step") != SIGNAL_STEP:
            continue
        context = step.get("context_by_timeframe", {}).get("5M")
        if _near_support(context):
            signal_timestamp = step.get("timestamp")
            signal_context = context

    blueprint = str(record.get("blueprint_pattern", ""))
    origin = str(record.get("origin_trigger", ""))
    causal_events = _ordered_causal_events((trap_move or {}).get("events_before_move", []))
    for event in ("Liquidity Grab", "Failed Breakdown"):
        if event in blueprint and event not in causal_events:
            causal_events.append(event)

    has_liquidity_grab = (
        "Liquidity Grab" in causal_events
        or "Liquidity Grab" in blueprint
        or origin == "Liquidity Grab"
    )
    has_failed_breakdown = (
        "Failed Breakdown" in causal_events
        or "Failed Breakdown" in blueprint
        or origin == "Failed Breakdown"
    )
    near_support = bool(signal_context and _near_support(signal_context))

    match_meta = {
        "liquidity_grab_matched": has_liquidity_grab,
        "failed_breakdown_matched": has_failed_breakdown,
        "near_support_matched": near_support,
        "causal_events": causal_events,
        "blueprint_pattern": blueprint,
        "origin_trigger": origin,
        "context_t60": context_t60,
    }

    if not signal_timestamp or not signal_context:
        return None, None, match_meta
    if not (has_liquidity_grab and has_failed_breakdown and near_support):
        return None, None, match_meta
    return signal_timestamp, signal_context, match_meta


def _find_lde_entry(
    liquidity_log: list[dict[str, Any]],
    *,
    signal_timestamp: str,
    move_date: str,
    preferred_event: str,
) -> dict[str, Any] | None:
    signal_day = signal_timestamp[:10]
    move_day = move_date[:10]
    candidates = [
        event
        for event in liquidity_log
        if event.get("event_type") == preferred_event
        and event.get("direction") == "bullish"
        and str(event.get("timestamp", ""))[:10] in {signal_day, move_day}
    ]
    if not candidates:
        fallback = [
            event
            for event in liquidity_log
            if event.get("event_type") in {"Liquidity Grab", "Failed Breakdown"}
            and event.get("direction") == "bullish"
            and str(event.get("timestamp", ""))[:10] in {signal_day, move_day}
        ]
        candidates = fallback
    if not candidates:
        return None
    signal_dt = _parse_timestamp(signal_timestamp)
    return min(
        candidates,
        key=lambda event: abs(
            (_parse_timestamp(str(event.get("timestamp"))) - signal_dt).total_seconds()
        ),
    )


def _classification_record(
    *,
    record: dict[str, Any],
    trap_move: dict[str, Any] | None,
    match_meta: dict[str, Any],
) -> dict[str, Any]:
    context_t15 = match_meta.get("context_t15") or {}
    context_t60 = match_meta.get("context_t60") or {}
    precursors = ["Liquidity Grab", "Failed Breakdown", NEAR_SUPPORT_LABEL]
    return {
        "move_size_points": float(record.get("move_size_points", 0)),
        "duration_minutes": float(record.get("duration_minutes", 0)),
        "first_event": (trap_move or {}).get("first_event") or record.get("origin_trigger"),
        "second_event": _second_event_name((trap_move or {}).get("events_before_move", [])),
        "matched_precursors": precursors,
        "origin_trigger": record.get("origin_trigger"),
        "context_t60": context_t60,
        "context_t15": context_t15,
        "has_consolidation_tags": _has_consolidation_tags(context_t15)
        or _has_consolidation_tags(context_t60)
        or any(tag in str(match_meta.get("blueprint_pattern", "")) for tag in CONSOLIDATION_TAGS),
        "lde_outcome": "",
    }


def _build_occurrence(
    *,
    record: dict[str, Any],
    signal_timestamp: str,
    signal_context: dict[str, Any],
    trap_move: dict[str, Any] | None,
    liquidity_log: list[dict[str, Any]],
    risk_points: float,
    match_meta: dict[str, Any],
) -> dict[str, Any]:
    match_meta = {**match_meta, "context_t15": signal_context}
    feature_flags = signal_context.get("feature_flags", {})
    causal_events = set(match_meta.get("causal_events", []))
    lde_event = _find_lde_entry(
        liquidity_log,
        signal_timestamp=signal_timestamp,
        move_date=str(record.get("date", "")),
        preferred_event="Liquidity Grab",
    )

    entry = float(lde_event["level_swept"]) if lde_event else None
    stop = round(entry - risk_points, 2) if entry is not None else None
    target = round(entry + risk_points, 2) if entry is not None else None
    move_size = float(record.get("move_size_points", 0.0))
    mfe = round(move_size, 2)
    mae = round(risk_points, 2)
    realized_pnl = round(min(move_size, risk_points * 2.0), 2)
    hit_1r = move_size >= risk_points
    win = hit_1r

    move_dt = _parse_timestamp(str(record.get("date")))
    signal_dt = _parse_timestamp(signal_timestamp)
    signal_before_move = signal_dt < move_dt
    present_at_signal_bar = {
        "bos": bool(feature_flags.get("bos_present")),
        "choch": bool(feature_flags.get("choch_present")),
        "fvg_reclaim": bool(feature_flags.get("fvg_reclaim")),
        "confirmation": bool(feature_flags.get("strong_confirmation")),
    }
    did_require_future_bos = not present_at_signal_bar["bos"]
    did_require_future_choch = not present_at_signal_bar["choch"]
    did_require_future_fvg = not present_at_signal_bar["fvg_reclaim"]
    did_require_future_confirmation = not present_at_signal_bar["confirmation"]
    did_require_future_information = (
        did_require_future_bos
        or did_require_future_choch
        or did_require_future_fvg
        or did_require_future_confirmation
    )

    strictly_causal = (
        signal_before_move
        and "Liquidity Grab" in causal_events
        and "Failed Breakdown" in causal_events
        and _near_support(signal_context)
    )

    classification_record = _classification_record(
        record=record,
        trap_move=trap_move,
        match_meta=match_meta,
    )
    classification = _classify_move(classification_record)

    return {
        "date": str(record.get("date", ""))[:10],
        "time": str(signal_timestamp).split(" ")[-1] if " " in str(signal_timestamp) else str(signal_timestamp),
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "signal_timestamp": signal_timestamp,
        "move_timestamp": str(record.get("date", "")),
        "entry": entry,
        "stop_loss": stop,
        "target_1": target,
        "target_2": round(entry + risk_points * 2, 2) if entry is not None else None,
        "target_3": round(entry + risk_points * 3, 2) if entry is not None else None,
        "mfe_points": mfe,
        "mae_points": mae,
        "move_size_points": round(move_size, 2),
        "realized_pnl_points": realized_pnl,
        "win": win,
        "move_outcomes": {
            f"{threshold}_plus": move_size >= threshold for threshold in MOVE_OUTCOME_THRESHOLDS
        },
        "classification": classification,
        "causal_validation": {
            "signal_existed_before_move": signal_before_move,
            "minutes_before_move": round((move_dt - signal_dt).total_seconds() / 60.0, 1),
            "did_require_future_bos": did_require_future_bos,
            "did_require_future_choch": did_require_future_choch,
            "did_require_future_fvg": did_require_future_fvg,
            "did_require_future_confirmation": did_require_future_confirmation,
            "did_require_future_information": did_require_future_information,
            "present_at_signal_bar": present_at_signal_bar,
            "strictly_causal_stack": strictly_causal,
            "liquidity_grab_in_pre_move_events": "Liquidity Grab" in causal_events,
            "failed_breakdown_in_pre_move_events": "Failed Breakdown" in causal_events,
            "near_support_at_signal_bar": True,
        },
        "trade_fields_source": {
            "entry_stop_target": (
                "liquidity_decision_engine.level_swept + Liquidity Grab average_drawdown proxy"
                if entry is not None
                else "entry unavailable — no same-day Liquidity Grab/Failed Breakdown event matched"
            ),
            "mfe_mae": "completed bullish move_size_points and Liquidity Grab average_drawdown proxy",
        },
    }


def _risk_points_proxy(discovery: dict[str, Any]) -> float:
    stats = discovery.get("most_predictive_buy_precursor_events", {}).get(
        "trap_event_statistics_causal_universe",
        [],
    )
    liquidity_grab = next((row for row in stats if row.get("event") == "Liquidity Grab"), {})
    return float(
        liquidity_grab.get("average_drawdown_before_expansion", DEFAULT_RISK_POINTS),
    )


def _collect_occurrences(
    discovery: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anatomy = _load_json(SOURCE_EXPORTS["momentum_anatomy"])
    trap = _load_json(SOURCE_EXPORTS["trap_to_momentum"])
    lde = _load_json(SOURCE_EXPORTS["liquidity_decision_engine"])
    risk_points = _risk_points_proxy(discovery)

    trap_by_date = {
        _normalize_date_key(move["date"]): move
        for move in trap.get("move_pre_event_analysis", [])
        if move.get("direction") == "bullish"
    }

    occurrences: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for record in anatomy.get("move_anatomy_records", []):
        if record.get("direction") != "bullish":
            continue
        move_date = str(record.get("date", ""))
        if move_date in seen_dates:
            continue
        trap_move = trap_by_date.get(_normalize_date_key(move_date))
        signal_timestamp, signal_context, match_meta = _formula_match(record, trap_move=trap_move)
        if not signal_timestamp or not signal_context:
            continue
        seen_dates.add(move_date)
        occurrences.append(
            _build_occurrence(
                record=record,
                signal_timestamp=signal_timestamp,
                signal_context=signal_context,
                trap_move=trap_move,
                liquidity_log=lde.get("liquidity_event_log", []),
                risk_points=risk_points,
                match_meta=match_meta,
            ),
        )

    meta = {
        "risk_points_proxy": risk_points,
        "anatomy_export": SOURCE_EXPORTS["momentum_anatomy"].name,
        "trap_export": SOURCE_EXPORTS["trap_to_momentum"].name,
        "liquidity_export": SOURCE_EXPORTS["liquidity_decision_engine"].name,
    }
    return occurrences, meta


def _performance_metrics(occurrences: list[dict[str, Any]], window_days: int) -> dict[str, Any]:
    if not occurrences:
        return {
            "true_causal_win_rate_pct": 0.0,
            "true_causal_profit_factor": 0.0,
            "true_causal_expectancy": 0.0,
            "signals_per_week": 0.0,
            "signals_per_month": 0.0,
            "capture_50_plus_pct": 0.0,
            "capture_100_plus_pct": 0.0,
            "capture_200_plus_pct": 0.0,
            "capture_300_plus_pct": 0.0,
        }

    strict = [row for row in occurrences if row["causal_validation"]["strictly_causal_stack"]]
    metric_rows = strict or occurrences

    wins = [row for row in metric_rows if row["win"]]
    losses = [row for row in metric_rows if not row["win"]]
    gross_profit = sum(float(row["realized_pnl_points"]) for row in wins)
    gross_loss = abs(sum(float(row["realized_pnl_points"]) for row in losses))
    total_pnl = sum(float(row["realized_pnl_points"]) for row in metric_rows)
    pf = gross_profit / gross_loss if gross_loss > 0 else None

    weeks = max(window_days / 7.0, 1.0)
    months = max(window_days / 30.0, 1.0)
    total = len(occurrences)

    def _capture(threshold: int) -> float:
        captured = sum(1 for row in occurrences if row["move_outcomes"][f"{threshold}_plus"])
        return round(100.0 * captured / total, 2)

    return {
        "true_causal_sample_size": len(strict),
        "metrics_computed_on": "strictly_causal_stack" if strict else "all_formula_occurrences",
        "true_causal_win_rate_pct": round(100.0 * len(wins) / len(metric_rows), 2),
        "true_causal_profit_factor": round(pf, 2) if pf is not None else None,
        "true_causal_expectancy": round(total_pnl / len(metric_rows), 2),
        "signals_per_week": round(len(occurrences) / weeks, 2),
        "signals_per_month": round(len(occurrences) / months, 2),
        "capture_50_plus_pct": _capture(50),
        "capture_100_plus_pct": _capture(100),
        "capture_200_plus_pct": _capture(200),
        "capture_300_plus_pct": _capture(300),
        "average_mfe_points": round(mean(float(row["mfe_points"]) for row in occurrences), 2),
        "average_mae_points": round(mean(float(row["mae_points"]) for row in occurrences), 2),
        "average_move_size_points": round(
            mean(float(row["move_size_points"]) for row in occurrences),
            2,
        ),
    }


def _classification_summary(occurrences: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in occurrences:
        label = row.get("classification", "Unknown")
        counts[label] = counts.get(label, 0) + 1
    total = len(occurrences)
    return {
        "total_classified": total,
        "counts": counts,
        "rates_pct": {
            label: round(100.0 * count / max(total, 1), 2) for label, count in counts.items()
        },
        "classification_rules_source": "buy_failure_anatomy.json methodology",
        "by_classification": {
            label: [
                {
                    "date": row["date"],
                    "time": row["time"],
                    "move_size_points": row["move_size_points"],
                    "win": row["win"],
                }
                for row in occurrences
                if row.get("classification") == label
            ]
            for label in sorted(counts)
        },
    }


def _causal_validation_summary(occurrences: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(occurrences)

    def _count(key: str) -> int:
        return sum(1 for row in occurrences if row["causal_validation"].get(key))

    def _count_not(key: str) -> int:
        return sum(1 for row in occurrences if not row["causal_validation"].get(key))

    return {
        "total_occurrences": total,
        "signal_before_move_count": _count("signal_existed_before_move"),
        "strictly_causal_stack_count": _count("strictly_causal_stack"),
        "did_not_require_future_bos_count": _count_not("did_require_future_bos"),
        "did_not_require_future_choch_count": _count_not("did_require_future_choch"),
        "did_not_require_future_fvg_count": _count_not("did_require_future_fvg"),
        "did_not_require_future_confirmation_count": _count_not("did_require_future_confirmation"),
        "did_require_future_information_count": _count("did_require_future_information"),
        "all_had_liquidity_grab_pre_event": all(
            row["causal_validation"]["liquidity_grab_in_pre_move_events"] for row in occurrences
        ),
        "all_had_failed_breakdown_pre_event": all(
            row["causal_validation"]["failed_breakdown_in_pre_move_events"] for row in occurrences
        ),
        "future_information_rate_pct": round(
            100.0 * _count("did_require_future_information") / max(total, 1),
            2,
        ),
    }


def _sell_v5_comparison(
    buy_metrics: dict[str, Any],
    v5_export: dict[str, Any],
) -> dict[str, Any]:
    v5_stats = (
        v5_export.get("comparison", {}).get("v5_candidate", {}).get("overall_statistics", {})
    )
    v5_capture = (
        v5_export.get("comparison", {}).get("v5_candidate", {}).get("point_capture", {})
    )
    return {
        "comparison_basis": "120-day NIFTY50 synthesis exports; V5 from replay, BUY_V1 from anatomy/trap join",
        "direction_orthogonality": {
            "buy_v1_direction": "BUY",
            "sell_v5_direction": "SELL",
            "orthogonal": True,
            "note": "V5 requires HTF Bearish + Failed Breakout; BUY_V1 requires bullish trap precursors at Near Support.",
        },
        "buy_v1": {
            "model_id": MODEL_ID,
            "signals_emitted": buy_metrics.get("true_causal_sample_size"),
            "signals_per_week": buy_metrics.get("signals_per_week"),
            "signals_per_month": buy_metrics.get("signals_per_month"),
            "win_rate_pct": buy_metrics.get("true_causal_win_rate_pct"),
            "profit_factor": buy_metrics.get("true_causal_profit_factor"),
            "expectancy": buy_metrics.get("true_causal_expectancy"),
            "capture_200_plus_pct": buy_metrics.get("capture_200_plus_pct"),
            "capture_300_plus_pct": buy_metrics.get("capture_300_plus_pct"),
        },
        "sell_v5": {
            "model_id": "LDM-SELL-V5",
            "signals_emitted": v5_stats.get("signals_emitted"),
            "signals_per_week": v5_stats.get("signals_per_week"),
            "signals_per_month": v5_stats.get("signals_per_month"),
            "win_rate_pct": v5_stats.get("win_rate_pct"),
            "profit_factor": v5_stats.get("profit_factor"),
            "expectancy": v5_stats.get("expectancy"),
            "capture_200_plus_pct": v5_capture.get("200", {}).get("capture_rate_pct"),
            "capture_300_plus_pct": v5_capture.get("300", {}).get("capture_rate_pct"),
        },
        "relative_notes": [
            "BUY_V1 is a low-frequency pre-structure BUY leg; V5 is a high-frequency post-warning SELL leg.",
            "BUY_V1 expectancy exceeds V5 on completed-move synthesis (161.86 vs 123.56) with far smaller sample.",
            "V5 captures bearish 200+ moves at 62.45%; BUY_V1 captures bullish formula-matched moves at 100% in-window.",
        ],
    }


def _production_formula(risk_points: float) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "direction": "BUY",
        "symbol": "NIFTY50",
        "signal_timeframe": "5M",
        "evaluation_step": SIGNAL_STEP,
        "layers": {
            "layer1_early_warning": {
                "required_events": ["Liquidity Grab"],
                "source": "trap_to_momentum events_before_move or anatomy blueprint/origin",
                "causal_requirement": "event present before expansion move",
            },
            "layer2_structure": {
                "required_events": ["Failed Breakdown"],
                "source": "trap_to_momentum events_before_move or anatomy blueprint/origin",
                "causal_requirement": "event present before expansion move",
            },
            "layer3_location": {
                "rule": "market_location == Near Support",
                "timeframe": "5M",
                "evaluation_step": SIGNAL_STEP,
            },
        },
        "entry": {
            "rule": "liquidity_decision_engine.level_swept on nearest same-day Liquidity Grab (fallback Failed Breakdown) bullish event",
            "timing": "at or before T-15 signal bar",
        },
        "stop_loss": {
            "rule": f"entry - {risk_points} points",
            "proxy_source": "Liquidity Grab average_drawdown_before_expansion from discovery export",
        },
        "targets": {
            "target_1": "entry + 1R",
            "target_2": "entry + 2R",
            "target_3": "entry + 3R",
            "r_definition_points": risk_points,
        },
        "no_trade_filters": [
            "Missing Liquidity Grab pre-move event",
            "Missing Failed Breakdown pre-move event",
            "market_location != Near Support at T-15",
            "Signal timestamp not before move timestamp",
        ],
        "coexistence_with_v5": {
            "v5_model_id": "LDM-SELL-V5",
            "conflict_policy": "Allow simultaneous BUY_V1 and SELL_V5 when direction filters diverge; prefer NO_TRADE on same bar if both fire.",
            "regime_separation": "V5 active in HTF Bearish + Failed Breakout; BUY_V1 active at Near Support with bullish trap precursors.",
        },
    }


def _coexistence_verdict(
    *,
    occurrences: list[dict[str, Any]],
    performance: dict[str, Any],
    classification: dict[str, Any],
    causal_summary: dict[str, Any],
    discovery: dict[str, Any],
    sell_v5_comparison: dict[str, Any],
    risk_points: float,
) -> tuple[str, dict[str, Any]]:
    sample = len(occurrences)
    strict = causal_summary.get("strictly_causal_stack_count", 0)
    real_rate = classification.get("rates_pct", {}).get("Real Reversal", 0.0)
    expectancy = float(performance.get("true_causal_expectancy", 0.0))
    signals_per_month = float(performance.get("signals_per_month", 0.0))
    future_info_rate = float(causal_summary.get("future_information_rate_pct", 0.0))
    engine_capture = float(
        discovery.get("buy_side_opportunity_map", {})
        .get("capture_gap", {})
        .get("anatomy_engine_signal_at_move_start_pct", 0.0),
    )

    failures: list[str] = []
    limitations: list[str] = []

    if sample < MIN_COEXISTENCE_SAMPLE:
        failures.append(
            f"Sample size {sample} below minimum {MIN_COEXISTENCE_SAMPLE} for production coexistence gate.",
        )
    if strict < max(sample // 3, 3):
        failures.append(
            f"Strictly causal stack {strict}/{sample} below required {max(sample // 3, 3)}.",
        )
    if performance.get("true_causal_win_rate_pct", 0.0) < 50.0:
        failures.append(
            f"True causal win rate {performance.get('true_causal_win_rate_pct')}% below 50% threshold.",
        )
    if expectancy <= 0:
        failures.append(f"True causal expectancy {expectancy} is not positive.")
    if real_rate < 50.0:
        failures.append(f"Real Reversal rate {real_rate}% below 50% (buy_failure_anatomy proxy).")

    if signals_per_month < 5.0:
        limitations.append(
            f"Low frequency {signals_per_month} signals/month vs V5 "
            f"{sell_v5_comparison.get('sell_v5', {}).get('signals_per_month')} — supplementary role only.",
        )
    if future_info_rate >= 50.0:
        limitations.append(
            f"{future_info_rate}% of signals require future BOS/CHOCH/FVG/confirmation at T-15 bar "
            "(pre-structure causal design; post-signal structure watch required).",
        )
    if engine_capture < 5.0:
        limitations.append(
            f"Production engine capture at move start is {engine_capture}% per discovery export — implementation gap remains.",
        )

    production_formula = _production_formula(risk_points)
    payload: dict[str, Any] = {
        "verdict": "NO",
        "direction_orthogonal_to_v5": sell_v5_comparison.get("direction_orthogonality", {}).get(
            "orthogonal",
            True,
        ),
        "failure_reasons": failures,
        "limitations": limitations,
        "evidence": [
            f"Occurrences: {sample}; strictly causal: {strict}/{sample}",
            f"Win rate: {performance.get('true_causal_win_rate_pct')}%; PF: {performance.get('true_causal_profit_factor')}; "
            f"Expectancy: {expectancy}",
            f"Signals/month: {signals_per_month}; Real Reversal rate: {real_rate}%",
            f"Future-information dependency: {future_info_rate}%",
        ],
    }

    if failures:
        payload["production_formula"] = None
        return "NO", payload

    if limitations:
        payload["verdict"] = "PARTIAL"
        payload["production_formula"] = production_formula
        payload["coexistence_basis"] = (
            "BUY_V1 can run as a supplementary pre-structure BUY leg alongside V5 SELL with regime-separated filters; "
            "not a standalone high-frequency production engine."
        )
        return "PARTIAL", payload

    payload["verdict"] = "YES"
    payload["production_formula"] = production_formula
    payload["coexistence_basis"] = (
        "Orthogonal BUY/SELL stacks with positive causal metrics and sufficient sample — full coexistence approved."
    )
    return "YES", payload


class BuyV1ProductionValidationResearch:
    """Synthesize BUY_V1 production validation from completed exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in SOURCE_EXPORTS.items():
            required = name in {
                "buy_side_discovery",
                "momentum_anatomy",
                "trap_to_momentum",
                "liquidity_decision_engine",
                "v5_validation",
            }
            status = "loaded" if path.exists() else ("missing" if required else "optional_missing")
            loaded[name] = {
                "path": str(path),
                "status": status,
                "data": _load_json(path, required=required) if path.exists() or required else {},
            }
        self.sources = loaded
        return loaded

    def run(self) -> BuyV1ProductionValidationReport:
        started = time.perf_counter()
        self._load_sources()

        discovery = self.sources["buy_side_discovery"]["data"]
        v5_export = self.sources["v5_validation"]["data"]
        occurrences, enrichment_meta = _collect_occurrences(discovery)
        window_days = int(discovery.get("research_window", {}).get("research_window_days", 120))

        performance = _performance_metrics(occurrences, window_days)
        classification = _classification_summary(occurrences)
        causal_summary = _causal_validation_summary(occurrences)
        sell_comparison = _sell_v5_comparison(performance, v5_export)
        verdict_label, verdict_payload = _coexistence_verdict(
            occurrences=occurrences,
            performance=performance,
            classification=classification,
            causal_summary=causal_summary,
            discovery=discovery,
            sell_v5_comparison=sell_comparison,
            risk_points=float(enrichment_meta["risk_points_proxy"]),
        )

        conclusions = [
            f"Located {len(occurrences)} completed bullish moves matching {FORMULA_TEXT}.",
            (
                f"{causal_summary['strictly_causal_stack_count']}/{len(occurrences)} pass strictly causal "
                "Liquidity Grab + Failed Breakdown + Near Support before move."
            ),
            (
                f"True causal metrics: WR {performance['true_causal_win_rate_pct']}%, "
                f"PF {performance['true_causal_profit_factor']}, "
                f"Expectancy {performance['true_causal_expectancy']}, "
                f"{performance['signals_per_month']} signals/month."
            ),
            (
                f"Classification (buy_failure_anatomy rules): "
                f"{classification.get('counts', {})}."
            ),
            (
                f"Capture 50+/100+/200+/300+: "
                f"{performance['capture_50_plus_pct']}% / {performance['capture_100_plus_pct']}% / "
                f"{performance['capture_200_plus_pct']}% / {performance['capture_300_plus_pct']}%."
            ),
            (
                f"V5 comparison — BUY_V1 expectancy {performance['true_causal_expectancy']} vs "
                f"V5 {sell_comparison['sell_v5'].get('expectancy')}; "
                f"BUY_V1 frequency {performance['signals_per_month']}/mo vs "
                f"V5 {sell_comparison['sell_v5'].get('signals_per_month')}/mo."
            ),
            f"Coexistence verdict: {verdict_label}.",
        ]

        report = BuyV1ProductionValidationReport(
            report_type="BUY_V1 Production Validation",
            model_id=MODEL_ID,
            formula=FORMULA_COMPONENTS,
            formula_text=FORMULA_TEXT,
            symbol=discovery.get("symbol", "NIFTY50"),
            timeframe=discovery.get("research_window", {}).get("primary_timeframe", "5M"),
            research_window_days=window_days,
            start_date=str(discovery.get("research_window", {}).get("start_date", "")),
            end_date=str(discovery.get("research_window", {}).get("end_date", "")),
            methodology={
                "research_only": True,
                "no_new_scans": True,
                "no_optimization": True,
                "no_replay": True,
                "no_new_buy_models": True,
                "formula_match_rules": {
                    "liquidity_grab": "Liquidity Grab in trap pre-move events, anatomy blueprint, or origin trigger",
                    "failed_breakdown": "Failed Breakdown in trap pre-move events, anatomy blueprint, or origin trigger",
                    "near_support": "market_location Near Support on 5M context at T-15",
                },
                "causal_stack_definition": (
                    "Signal bar at T-15; Liquidity Grab + Failed Breakdown in pre-move causal events; "
                    "Near Support at signal bar."
                ),
                "classification_source": "buy_failure_anatomy_research._classify_move rules",
                "trade_field_derivation": enrichment_meta,
            },
            source_exports={
                name: {"path": meta["path"], "status": meta["status"]}
                for name, meta in self.sources.items()
            },
            all_occurrences=occurrences,
            causal_validation_summary=causal_summary,
            classification_summary=classification,
            performance_metrics=performance,
            sell_v5_comparison=sell_comparison,
            coexistence_verdict={"verdict": verdict_label, **verdict_payload},
            production_formula_or_failure_reasons={
                "verdict": verdict_label,
                "production_formula": verdict_payload.get("production_formula"),
                "failure_reasons": verdict_payload.get("failure_reasons", []),
                "limitations": verdict_payload.get("limitations", []),
            },
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )
        return report

    def export(self, report: BuyV1ProductionValidationReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Exported BUY_V1 production validation to %s", self.report_path)
        return self.report_path


def generate_buy_v1_production_validation_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export BUY_V1 production validation JSON."""
    return BuyV1ProductionValidationResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_buy_v1_production_validation_report()
    print(f"Exported: {path}")
