"""
BUY_V3 Candidate Validation — research with actual replay.

Validates Failed Breakdown + Gap Reversal + Liquidity Grab + Near Support + PDL Sweep
(BUY_V3) vs BUY_V1/BUY_V2 on 120-day NIFTY50 5M replay with walk-forward and ablation.
Cross-checks 947 BUY_V2-only false reversals from buy_v2_candidate_validation.json.
Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.buy_failure_anatomy_research import NEAR_SUPPORT_LABEL, REAL_REVERSAL_MIN_POINTS
from src.research.buy_v2_candidate_validation_research import (
    BUY_V1_COMPONENTS,
    BUY_V1_FORMULA_TEXT,
    BUY_V1_MODEL_ID,
    BUY_V2_COMPONENTS,
    BUY_V2_FORMULA_TEXT,
    BUY_V2_MODEL_ID,
    PRODUCTION_GATES,
    TRADING_DAYS_REPLAY,
    TRAIN_TRADING_DAYS,
    VALIDATE_TRADING_DAYS,
    BaseBuyCandidateEngine,
    BuyV1CandidateEngine,
    BuyV2CandidateEngine,
    _bullish_point_capture,
    _classify_failed_buy_signal,
    _filter_signals_by_dates,
    _nearest_bullish_move,
    _passes_production_gates,
    _split_trading_day_sets,
    _walk_forward_metrics,
)
from src.research.filter_research_engine import FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import PRE_EXPANSION_LOOKBACK
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    _CheapMoveCandidate,
)
from src.research.nifty50_trap_to_momentum_validation_research import DEFAULT_SYMBOL
from src.research.smartmoneyengine_v3_implementation_validation_research import (
    ALLOWED_VOLUME_BUCKETS,
    MOVE_DETECTION_TIMEFRAME,
    PIPELINE_TIMEFRAMES,
)
from src.research.smartmoneyengine_v4_candidate_validation_research import (
    _build_statistics,
    _last_n_trading_day_set,
    _signal_before_move,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_v3_candidate_validation.json"
DEFAULT_V2_REPORT_PATH = RESEARCH_DIR / "buy_v2_candidate_validation.json"
DEFAULT_V5_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v5_candidate_validation.json"
DEFAULT_MISSED_REVERSAL_PATH = RESEARCH_DIR / "buy_v1_missed_reversal_analysis.json"

POINT_CAPTURE_THRESHOLDS = (40, 60, 80, 100, 200)
MOVE_DETECTION_THRESHOLD = 40
BAR_MINUTES = 5
FALSE_REVERSAL_CLASSIFICATIONS = frozenset(
    {"False Reversal", "Dead Cat Bounce", "Bull Trap", "No Expansion"},
)
TRADEABILITY_THRESHOLDS = (40, 60, 80, 100)
TRADEABILITY_HORIZONS = ("same_day", "1_trading_day", "2_trading_days", "3_trading_days")

BUY_V3_EVENTS = ("Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep")
BUY_V3_FORMULA_TEXT = "Failed Breakdown + Gap Reversal + Liquidity Grab + Near Support + PDL Sweep"
BUY_V3_MODEL_ID = "LDM-BUY-V3"

BUY_CONFIRMATION_CANDLES = frozenset(
    {
        "Hammer",
        "Bullish Engulfing",
        "Morning Star",
        "Marubozu",
        "None",
    },
)

ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "full_buy_v3": {
        "label": "Full BUY_V3",
        "events": BUY_V3_EVENTS,
        "location": NEAR_SUPPORT_LABEL,
        "removed": None,
    },
    "minus_liquidity_grab": {
        "label": "BUY_V3 minus Liquidity Grab",
        "events": ("Failed Breakdown", "Gap Reversal", "PDL Sweep"),
        "location": NEAR_SUPPORT_LABEL,
        "removed": "Liquidity Grab",
    },
    "minus_near_support": {
        "label": "BUY_V3 minus Near Support",
        "events": BUY_V3_EVENTS,
        "location": None,
        "removed": "Near Support",
    },
    "minus_pdl_sweep": {
        "label": "BUY_V3 minus PDL Sweep",
        "events": ("Failed Breakdown", "Gap Reversal", "Liquidity Grab"),
        "location": NEAR_SUPPORT_LABEL,
        "removed": "PDL Sweep",
    },
    "minus_gap_reversal": {
        "label": "BUY_V3 minus Gap Reversal",
        "events": ("Failed Breakdown", "Liquidity Grab", "PDL Sweep"),
        "location": NEAR_SUPPORT_LABEL,
        "removed": "Gap Reversal",
    },
}


class BuyV3CandidateValidationError(Exception):
    """Raised when BUY_V3 candidate validation fails."""


def _make_buy_engine(
    *,
    model_id: str,
    required_events: tuple[str, ...],
    required_location: str | None,
) -> BaseBuyCandidateEngine:
    """Factory for parameterized BUY candidate engines."""

    class _Engine(BaseBuyCandidateEngine):
        MODEL_ID = model_id
        REQUIRED_EVENTS = required_events
        REQUIRED_LOCATION = required_location

    return _Engine()


class BuyV3CandidateEngine(BaseBuyCandidateEngine):
    """BUY_V3: Failed Breakdown + Gap Reversal + Liquidity Grab + Near Support + PDL Sweep."""

    MODEL_ID = BUY_V3_MODEL_ID
    REQUIRED_EVENTS = BUY_V3_EVENTS
    REQUIRED_LOCATION = NEAR_SUPPORT_LABEL


def _layer1_from_events(
    engine: BaseBuyCandidateEngine,
    *,
    lookback_events: set[str],
    bar_events: set[str],
) -> dict[str, Any]:
    matched = [event for event in engine.REQUIRED_EVENTS if event in lookback_events]
    return {
        "active": len(matched) == len(engine.REQUIRED_EVENTS),
        "events_detected": sorted(lookback_events),
        "events_at_bar": sorted(bar_events),
        "formula_events_matched": matched,
        "formula_events_missing": [event for event in engine.REQUIRED_EVENTS if event not in lookback_events],
        "lookback_bars": PRE_EXPANSION_LOOKBACK,
    }


def _layer2_from_context(engine: BaseBuyCandidateEngine, context: dict[str, str]) -> dict[str, Any]:
    htf = context.get("htf_trend", "Neutral")
    vwap = context.get("vwap")
    ema = context.get("ema_structure", "Mixed")
    location_ok = (
        context.get("location") == engine.REQUIRED_LOCATION
        if engine.REQUIRED_LOCATION
        else True
    )
    if engine.REQUIRE_BULLISH_ALIGNMENT:
        aligned = (
            htf != "Bearish"
            and vwap in {"Above", "Reclaimed", "Rejected"}
            and ema != "Bear Stack"
            and location_ok
        )
    else:
        aligned = location_ok
    return {
        "direction": "BUY" if aligned else "NO_TRADE",
        "htf_trend": htf,
        "vwap_state": vwap,
        "ema_structure": ema,
        "location": context.get("location"),
        "location_required": engine.REQUIRED_LOCATION,
        "location_ok": location_ok,
        "aligned": aligned,
    }


def _layer3_from_context(context: dict[str, str]) -> dict[str, Any]:
    candle = context.get("confirmation_candle", "None")
    volume = context.get("volume", "Normal")
    candle_ok = candle in BUY_CONFIRMATION_CANDLES
    volume_ok = volume in ALLOWED_VOLUME_BUCKETS
    return {
        "confirmation_candle": candle,
        "volume_bucket": volume,
        "confirmed": candle_ok and volume_ok,
        "confirmation_optional": True,
        "candle_required": False,
    }


def _evaluate_buy_bar_fast(
    engine: BaseBuyCandidateEngine,
    *,
    frame: pd.DataFrame,
    bar: int,
    context: dict[str, str],
    lookback_events: set[str],
    bar_events: set[str],
    emitted_bars: set[int],
) -> dict[str, Any]:
    layer1 = _layer1_from_events(engine, lookback_events=lookback_events, bar_events=bar_events)
    layer2 = _layer2_from_context(engine, context)
    layer3 = _layer3_from_context(context)
    layer5 = engine._layer5_no_trade_filters(
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        context=context,
        bar=bar,
        emitted_bars=emitted_bars,
    )
    timestamp = str(frame.iloc[bar].get("Date", ""))
    result: dict[str, Any] = {
        "timestamp": timestamp,
        "bar": bar,
        "verdict": "NO_TRADE",
        "layer1": layer1,
        "layer2": layer2,
        "layer3": layer3,
        "layer5": layer5,
        "context": context,
    }
    if layer5["pass"]:
        execution = engine._layer4_execution(
            frame,
            bar,
            layer1=layer1,
            layer2=layer2,
            layer3=layer3,
            context=context,
        )
        if execution:
            result["verdict"] = "BUY"
            result["layer4"] = execution
    return result


def _precompute_bar_events(
    engine: BaseBuyCandidateEngine,
    *,
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    replay_bars: list[int],
) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    bar_events_cache: dict[int, set[str]] = {}
    lookback_cache: dict[int, set[str]] = {}
    total = len(replay_bars)
    log_every = max(total // 10, 1)
    started = time.perf_counter()
    for index, bar in enumerate(replay_bars):
        if index > 0 and index % log_every == 0:
            logger.info(
                "Event precompute progress: %s/%s bars (%.0f%%) elapsed %.0fs",
                index,
                total,
                index / total * 100,
                time.perf_counter() - started,
            )
        if bar not in bar_events_cache:
            bar_events_cache[bar] = set(engine._detect_events_at_bar(frame, calendar, bar))
        lookback_cache[bar] = _events_in_lookback_cached(
            engine,
            frame=frame,
            calendar=calendar,
            bar=bar,
            bar_events_cache=bar_events_cache,
        )
    logger.info("Event precompute complete for %s bars in %.0fs", total, time.perf_counter() - started)
    return bar_events_cache, lookback_cache


def _events_in_lookback_cached(
    engine: BaseBuyCandidateEngine,
    *,
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    bar: int,
    bar_events_cache: dict[int, set[str]],
) -> set[str]:
    start = max(0, bar - PRE_EXPANSION_LOOKBACK)
    found: set[str] = set()
    for offset in range(start, bar + 1):
        if offset not in bar_events_cache:
            bar_events_cache[offset] = set(engine._detect_events_at_bar(frame, calendar, offset))
        found.update(bar_events_cache[offset])
    return found


@dataclass
class BuyV3CandidateValidationReport:
    """BUY_V3 replay validation output."""

    report_type: str
    engines_compared: list[str]
    buy_v1_formula: list[str]
    buy_v2_formula: list[str]
    buy_v3_formula: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    walk_forward: dict[str, Any]
    methodology: dict[str, Any]
    replay_rules: dict[str, Any]
    comparison: dict[str, Any]
    false_reversal_removal: dict[str, Any]
    ablation_analysis: dict[str, Any]
    signal_timing: dict[str, Any]
    tradeability: dict[str, Any]
    per_signal_details: dict[str, list[dict[str, Any]]]
    failed_signal_classification: dict[str, Any]
    sell_v5_benchmark: dict[str, Any]
    production_safety_check: dict[str, Any]
    final_verdict: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _trading_day_index(dates: list[date], target: date) -> int | None:
    ordered = sorted(set(dates))
    try:
        return ordered.index(target)
    except ValueError:
        return None


def _days_between_trading_days(dates: list[date], start: date, end: date) -> int | None:
    start_idx = _trading_day_index(dates, start)
    end_idx = _trading_day_index(dates, end)
    if start_idx is None or end_idx is None:
        return None
    return end_idx - start_idx


def _signal_timing_analysis(
    signals: list[dict[str, Any]],
    *,
    engine_key: str,
) -> dict[str, Any]:
    before = during = after = no_move = 0
    lead_bars: list[int] = []
    lead_minutes: list[float] = []
    lead_points: list[float] = []

    for signal in signals:
        bars_before = signal.get("bars_before_expansion")
        if bars_before is None:
            no_move += 1
            continue
        if bars_before > 0:
            before += 1
            lead_bars.append(bars_before)
            lead_minutes.append(bars_before * BAR_MINUTES)
            if signal.get("points_before_expansion") is not None:
                lead_points.append(float(signal["points_before_expansion"]))
        elif bars_before == 0:
            during += 1
        else:
            after += 1

    total_with_move = before + during + after
    return {
        "engine": engine_key,
        "before_expansion_count": before,
        "during_expansion_count": during,
        "after_expansion_count": after,
        "no_linked_move_count": no_move,
        "before_expansion_pct": round(100.0 * before / max(total_with_move, 1), 2),
        "during_expansion_pct": round(100.0 * during / max(total_with_move, 1), 2),
        "after_expansion_pct": round(100.0 * after / max(total_with_move, 1), 2),
        "lead_time_bars": {
            "avg": round(mean(lead_bars), 2) if lead_bars else None,
            "median": round(median(lead_bars), 2) if lead_bars else None,
            "max": max(lead_bars) if lead_bars else None,
        },
        "lead_time_minutes": {
            "avg": round(mean(lead_minutes), 2) if lead_minutes else None,
            "median": round(median(lead_minutes), 2) if lead_minutes else None,
            "max": round(max(lead_minutes), 2) if lead_minutes else None,
        },
        "lead_time_points": {
            "avg": round(mean(lead_points), 2) if lead_points else None,
            "median": round(median(lead_points), 2) if lead_points else None,
            "max": round(max(lead_points), 2) if lead_points else None,
        },
    }


def _tradeability_analysis(
    signals: list[dict[str, Any]],
    moves: list[_CheapMoveCandidate],
    frame: pd.DataFrame,
    replay_dates: set[date],
    *,
    engine_key: str,
) -> dict[str, Any]:
    ordered_dates = sorted(replay_dates)
    move_by_start: dict[int, _CheapMoveCandidate] = {move.start_bar: move for move in moves}
    horizons = {1: "same_day", 2: "1_trading_day", 3: "2_trading_days", 4: "3_trading_days"}
    results: dict[str, Any] = {"engine": engine_key, "by_threshold": {}}

    for threshold in TRADEABILITY_THRESHOLDS:
        tier: dict[str, Any] = {"threshold_points": threshold, "horizons": {}}
        bullish = [
            move
            for move in moves
            if move.direction == "bullish"
            and move.magnitude >= threshold
            and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
        ]
        for max_day_offset, horizon_key in horizons.items():
            captured = 0
            for move in bullish:
                pre_start = max(0, move.start_bar - PRE_EXPANSION_LOOKBACK)
                signal_bar = None
                for bar in range(pre_start, move.start_bar + 1):
                    if bar in {signal["bar"] for signal in signals}:
                        signal_bar = bar
                        break
                if signal_bar is None:
                    continue
                signal_day = pd.to_datetime(frame.iloc[signal_bar]["Date"]).date()
                move_day = pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date()
                day_gap = _days_between_trading_days(ordered_dates, signal_day, move_day)
                if day_gap is not None and day_gap <= max_day_offset - 1:
                    captured += 1
            total = len(bullish)
            tier["horizons"][horizon_key] = {
                "captured_moves": captured,
                "total_moves": total,
                "capture_rate_pct": round(100.0 * captured / max(total, 1), 2),
            }
        results["by_threshold"][str(threshold)] = tier

    return results


def _classification_summary(signals: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(signal.get("classification", "Unknown") for signal in signals)
    total = len(signals)
    rates = {label: round(100.0 * count / max(total, 1), 2) for label, count in counts.items()}
    false_count = sum(counts.get(label, 0) for label in FALSE_REVERSAL_CLASSIFICATIONS)
    return {
        "counts": dict(counts),
        "rates_pct": rates,
        "real_reversal_rate_pct": rates.get("Real Reversal", 0.0),
        "false_reversal_rate_pct": round(100.0 * false_count / max(total, 1), 2),
        "dead_cat_bounce_rate_pct": rates.get("Dead Cat Bounce", 0.0),
        "no_expansion_rate_pct": rates.get("No Expansion", 0.0),
        "counter_trend_bounce_rate_pct": rates.get("Counter Trend Bounce", 0.0),
        "bull_trap_rate_pct": rates.get("Bull Trap", 0.0),
        "range_failure_rate_pct": rates.get("Range Failure", 0.0),
    }


def _load_v2_false_reversal_cohort(v2_export: dict[str, Any]) -> list[dict[str, Any]]:
    v1_signals = v2_export.get("per_signal_details", {}).get("buy_v1", [])
    v2_signals = v2_export.get("per_signal_details", {}).get("buy_v2", [])
    cohort: list[dict[str, Any]] = []
    for signal in v2_signals:
        if _signal_before_move(v1_signals, signal["bar"]) is not None:
            continue
        if signal.get("classification") not in FALSE_REVERSAL_CLASSIFICATIONS:
            continue
        cohort.append(signal)
    return cohort


def _false_reversal_removal_analysis(
    *,
    v2_false_cohort: list[dict[str, Any]],
    v1_signals: list[dict[str, Any]],
    v2_signals: list[dict[str, Any]],
    v3_signals: list[dict[str, Any]],
    missed_rows: list[dict[str, Any]],
    moves: list[_CheapMoveCandidate],
    frame: pd.DataFrame,
    replay_dates: set[date],
) -> dict[str, Any]:
    v3_bars = {signal["bar"] for signal in v3_signals}
    v2_bars = {signal["bar"] for signal in v2_signals}

    remaining = 0
    removed = 0
    removal_details: list[dict[str, Any]] = []
    for false_signal in v2_false_cohort:
        bar = false_signal["bar"]
        still_fires = bar in v3_bars
        if still_fires:
            remaining += 1
        else:
            removed += 1
        removal_details.append(
            {
                "timestamp": false_signal.get("timestamp"),
                "bar": bar,
                "classification": false_signal.get("classification"),
                "buy_v3_still_fires": still_fires,
                "removed_by_buy_v3": not still_fires,
            },
        )

    baseline_count = len(v2_false_cohort)
    expected_baseline = 947

    bullish_moves = [
        move
        for move in moves
        if move.direction == "bullish"
        and move.magnitude >= REAL_REVERSAL_MIN_POINTS
        and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
    ]
    move_by_date: dict[str, _CheapMoveCandidate] = {}
    for move in bullish_moves:
        move_day = str(frame.iloc[move.start_bar]["Date"])[:10]
        existing = move_by_date.get(move_day)
        if existing is None or move.magnitude > existing.magnitude:
            move_by_date[move_day] = move

    recovered_v3 = 0
    recovered_v2 = 0
    recovered_v1 = 0
    still_missed = 0
    for row in missed_rows:
        move_day = str(row.get("date", ""))[:10]
        move = move_by_date.get(move_day)
        if move is None:
            continue
        v1_hit = _signal_before_move(v1_signals, move.start_bar) is not None
        v2_hit = _signal_before_move(v2_signals, move.start_bar) is not None
        v3_hit = _signal_before_move(v3_signals, move.start_bar) is not None
        if v1_hit:
            recovered_v1 += 1
        if v2_hit and not v1_hit:
            recovered_v2 += 1
        if v3_hit and not v1_hit:
            recovered_v3 += 1
        if not v1_hit and not v2_hit and not v3_hit:
            still_missed += 1

    v3_only_vs_v2 = [signal for signal in v3_signals if signal["bar"] not in v2_bars]
    v3_new_false = sum(
        1
        for signal in v3_only_vs_v2
        if signal.get("classification") in FALSE_REVERSAL_CLASSIFICATIONS
    )
    v2_only_bad = [
        signal
        for signal in v2_signals
        if signal["bar"] not in v3_bars
        and _signal_before_move(v1_signals, signal["bar"]) is None
        and signal.get("classification") in FALSE_REVERSAL_CLASSIFICATIONS
    ]

    v3_real = sum(1 for signal in v3_signals if signal.get("classification") == "Real Reversal")
    v2_real = sum(1 for signal in v2_signals if signal.get("classification") == "Real Reversal")
    v1_real = sum(1 for signal in v1_signals if signal.get("classification") == "Real Reversal")
    net_quality_change_vs_v2 = v3_real - v2_real
    net_quality_change_vs_v1 = v3_real - v1_real

    removal_pct = round(100.0 * removed / max(baseline_count, 1), 2)
    return {
        "baseline_source": str(DEFAULT_V2_REPORT_PATH.name),
        "baseline_false_reversal_count": baseline_count,
        "expected_baseline_count": expected_baseline,
        "baseline_matches_export": baseline_count == expected_baseline,
        "removed_by_buy_v3": removed,
        "remaining_false_reversals": remaining,
        "removal_rate_pct": removal_pct,
        "recovered_real_reversals_v3": recovered_v3,
        "recovered_real_reversals_v2": recovered_v2,
        "recovered_real_reversals_v1": recovered_v1,
        "still_missed_all_engines": still_missed,
        "missed_reversal_cohort_size": len(missed_rows),
        "new_false_reversals_v3_only": v3_new_false,
        "v2_only_bad_signals_filtered_by_v3": len(v2_only_bad),
        "net_gain_vs_v2": recovered_v3 - remaining + removed - baseline_count + recovered_v2,
        "net_quality_change_real_reversals_vs_v2": net_quality_change_vs_v2,
        "net_quality_change_real_reversals_vs_v1": net_quality_change_vs_v1,
        "summary": (
            f"BUY_V3 removed {removed}/{baseline_count} BUY_V2 false reversals "
            f"({removal_pct}%); {remaining} remain. "
            f"Recovered {recovered_v3} missed reversals vs V1; "
            f"net quality change vs V2: {net_quality_change_vs_v2} real reversals."
        ),
        "per_false_reversal_sample": removal_details[:100],
    }


def _ablation_metrics(
    signals: list[dict[str, Any]],
    moves: list[_CheapMoveCandidate],
    frame: pd.DataFrame,
    replay_dates: set[date],
    *,
    variant_key: str,
    variant_meta: dict[str, Any],
) -> dict[str, Any]:
    stats = _build_statistics(signals, trading_days=TRADING_DAYS_REPLAY)
    capture = _bullish_point_capture(moves, signals, replay_dates, frame, (40, 60, 100))
    classification = _classification_summary(signals)
    return {
        "variant_key": variant_key,
        "label": variant_meta["label"],
        "removed_condition": variant_meta["removed"],
        "signals_emitted_count": len(signals),
        "overall_statistics": stats,
        "point_capture": capture,
        "classification_summary": classification,
        "false_reversal_rate_pct": classification["false_reversal_rate_pct"],
    }


def _ablation_contribution_ranking(ablation_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    full = ablation_results.get("full_buy_v3", {})
    full_wr = float(full.get("overall_statistics", {}).get("win_rate_pct") or 0.0)
    full_pf = float(full.get("overall_statistics", {}).get("profit_factor") or 0.0)
    full_spm = float(full.get("overall_statistics", {}).get("signals_per_month") or 0.0)
    full_false = float(full.get("false_reversal_rate_pct") or 0.0)
    full_capture_40 = float(full.get("point_capture", {}).get("40", {}).get("capture_rate_pct") or 0.0)

    rankings: list[dict[str, Any]] = []
    for key, result in ablation_results.items():
        if key == "full_buy_v3":
            continue
        removed = result.get("removed_condition") or "unknown"
        stats = result.get("overall_statistics", {})
        wr = float(stats.get("win_rate_pct") or 0.0)
        pf = float(stats.get("profit_factor") or 0.0)
        spm = float(stats.get("signals_per_month") or 0.0)
        false_rate = float(result.get("false_reversal_rate_pct") or 0.0)
        capture_40 = float(result.get("point_capture", {}).get("40", {}).get("capture_rate_pct") or 0.0)
        rankings.append(
            {
                "removed_condition": removed,
                "variant_key": key,
                "wr_delta_vs_full_pp": round(wr - full_wr, 2),
                "pf_delta_vs_full": round(pf - full_pf, 2) if pf and full_pf else None,
                "signals_per_month_delta": round(spm - full_spm, 2),
                "false_reversal_rate_delta_pp": round(false_rate - full_false, 2),
                "capture_40_delta_pp": round(capture_40 - full_capture_40, 2),
                "quality_contribution_score": round(full_wr - wr + (full_pf - pf if pf and full_pf else 0), 2),
                "frequency_contribution_score": round(full_spm - spm, 2),
                "false_reduction_contribution_score": round(false_rate - full_false, 2),
            },
        )

    by_quality = sorted(rankings, key=lambda item: item["quality_contribution_score"], reverse=True)
    by_frequency = sorted(rankings, key=lambda item: item["frequency_contribution_score"], reverse=True)
    by_false_reduction = sorted(rankings, key=lambda item: item["false_reduction_contribution_score"], reverse=True)
    return {
        "most_quality_contribution": by_quality[0]["removed_condition"] if by_quality else None,
        "most_frequency_contribution": by_frequency[0]["removed_condition"] if by_frequency else None,
        "most_false_reversal_reduction": by_false_reduction[0]["removed_condition"] if by_false_reduction else None,
        "rankings": rankings,
    }


def _sell_v5_benchmark() -> dict[str, Any]:
    if not DEFAULT_V5_REPORT_PATH.exists():
        return {
            "source": str(DEFAULT_V5_REPORT_PATH.name),
            "status": "missing",
            "note": "SELL_V5 benchmark unavailable — run smartmoneyengine_v5_candidate_validation first.",
        }
    payload = json.loads(DEFAULT_V5_REPORT_PATH.read_text(encoding="utf-8"))
    v5 = payload.get("comparison", {}).get("v5_candidate", {})
    stats = v5.get("overall_statistics", {})
    capture = v5.get("point_capture", {})
    return {
        "source": str(DEFAULT_V5_REPORT_PATH.name),
        "status": "loaded",
        "model_id": "LDM-SELL-V5",
        "signals_emitted": v5.get("signals_emitted_count"),
        "signals_per_month": stats.get("signals_per_month"),
        "win_rate_pct": stats.get("win_rate_pct"),
        "profit_factor": stats.get("profit_factor"),
        "expectancy": stats.get("expectancy"),
        "capture_40_plus_pct": capture.get("40", {}).get("capture_rate_pct"),
        "capture_200_plus_pct": capture.get("200", {}).get("capture_rate_pct"),
    }


def _final_verdict(
    *,
    v3_stats: dict[str, Any],
    v3_capture: dict[str, Any],
    v1_stats: dict[str, Any],
    v2_stats: dict[str, Any],
    false_removal: dict[str, Any],
    walk_forward: dict[str, Any],
    production_safety: dict[str, Any],
    sell_v5: dict[str, Any],
    ablation_ranking: dict[str, Any],
) -> dict[str, Any]:
    def _tier(pass_count: int, total: int = 5) -> str:
        if pass_count >= total:
            return "Production Candidate"
        if pass_count >= 3:
            return "Dry Run Candidate"
        return "Research Candidate"

    v3_gates = production_safety.get("buy_v3", {})
    v3_pass = sum(1 for key, value in v3_gates.items() if key != "all_pass" and value)

    v3_wr = float(v3_stats.get("win_rate_pct") or 0.0)
    v3_pf = float(v3_stats.get("profit_factor") or 0.0)
    v3_spm = float(v3_stats.get("signals_per_month") or 0.0)
    capture_40 = float(v3_capture.get("40", {}).get("capture_rate_pct") or 0.0)
    capture_60 = float(v3_capture.get("60", {}).get("capture_rate_pct") or 0.0)

    gates_met = (
        v3_spm >= PRODUCTION_GATES["signals_per_month_min"]
        and v3_wr >= PRODUCTION_GATES["win_rate_min_pct"]
        and v3_pf >= PRODUCTION_GATES["profit_factor_min"]
    )

    oos = walk_forward.get("validate", {}).get("buy_v3", {}).get("overall_statistics", {})
    train = walk_forward.get("train", {}).get("buy_v3", {}).get("overall_statistics", {})
    wf_stable = (
        float(oos.get("win_rate_pct") or 0.0) >= float(train.get("win_rate_pct") or 0.0) * 0.85
        and float(oos.get("profit_factor") or 0.0) >= float(train.get("profit_factor") or 0.0) * 0.70
    )

    removed = false_removal.get("removed_by_buy_v3", 0)
    baseline = false_removal.get("baseline_false_reversal_count", 947)
    removal_pct = false_removal.get("removal_rate_pct", 0.0)

    if capture_40 >= 5.0 and capture_60 >= 3.0 and v3_spm >= 10:
        tradeable_40_60 = "YES"
    elif capture_40 > 0 and v3_spm >= 5:
        tradeable_40_60 = "PARTIAL"
    else:
        tradeable_40_60 = "NO"

    if gates_met and wf_stable and removal_pct >= 80:
        classification = "Production Candidate"
    elif v3_pass >= 3 or (v3_wr >= 60 and v3_pf >= 1.5):
        classification = "Dry Run Candidate"
    else:
        classification = "Research Candidate"

    production_answer = "YES" if v3_gates.get("all_pass") and wf_stable else (
        "PARTIAL" if v3_pass >= 3 else "NO"
    )

    return {
        "classification": classification,
        "production_candidate": production_answer,
        "replay_validated": True,
        "production_gates_passed": v3_pass,
        "walk_forward_stable": wf_stable,
        "false_reversal_removal": {
            "answer": "YES" if removal_pct >= 90 else ("PARTIAL" if removed > 0 else "NO"),
            "removed": removed,
            "baseline": baseline,
            "remaining": false_removal.get("remaining_false_reversals"),
            "removal_rate_pct": removal_pct,
        },
        "can_deliver_40_60pt_opportunities": {
            "answer": tradeable_40_60,
            "capture_40_pct": capture_40,
            "capture_60_pct": capture_60,
            "signals_per_month": v3_spm,
        },
        "vs_buy_v1": {
            "signals_delta": (v3_stats.get("signals_emitted") or 0) - (v1_stats.get("signals_emitted") or 0),
            "wr_delta_pp": round(v3_wr - float(v1_stats.get("win_rate_pct") or 0.0), 2),
            "pf_delta": round(v3_pf - float(v1_stats.get("profit_factor") or 0.0), 2)
            if v1_stats.get("profit_factor") and v3_stats.get("profit_factor")
            else None,
        },
        "vs_buy_v2": {
            "signals_delta": (v3_stats.get("signals_emitted") or 0) - (v2_stats.get("signals_emitted") or 0),
            "wr_delta_pp": round(v3_wr - float(v2_stats.get("win_rate_pct") or 0.0), 2),
            "pf_delta": round(v3_pf - float(v2_stats.get("profit_factor") or 0.0), 2)
            if v2_stats.get("profit_factor") and v3_stats.get("profit_factor")
            else None,
        },
        "vs_sell_v5": {
            "wr": v3_wr,
            "pf": v3_pf,
            "sell_v5_wr": sell_v5.get("win_rate_pct"),
            "sell_v5_pf": sell_v5.get("profit_factor"),
        },
        "ablation_insights": {
            "most_quality_contribution": ablation_ranking.get("most_quality_contribution"),
            "most_frequency_contribution": ablation_ranking.get("most_frequency_contribution"),
            "most_false_reversal_reduction": ablation_ranking.get("most_false_reversal_reduction"),
        },
        "evidence": [
            false_removal.get("summary", ""),
            (
                f"BUY_V3: {v3_spm}/mo, WR {v3_wr}%, PF {v3_pf}, "
                f"40+ capture {capture_40}%, gates {'PASS' if v3_gates.get('all_pass') else 'FAIL'}."
            ),
        ],
    }


class BuyV3CandidateValidationResearch:
    """Replay BUY_V3 vs BUY_V1/BUY_V2 with ablation on 120-day NIFTY50 window."""

    def __init__(self) -> None:
        self.buy_v1_engine = BuyV1CandidateEngine()
        self.buy_v2_engine = BuyV2CandidateEngine()
        self.buy_v3_engine = BuyV3CandidateEngine()
        self.ablation_engines: dict[str, BaseBuyCandidateEngine] = {
            key: _make_buy_engine(
                model_id=f"LDM-BUY-V3-ABL-{key.upper()}",
                required_events=meta["events"],
                required_location=meta["location"],
            )
            for key, meta in ABLATION_VARIANTS.items()
        }
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=TRADING_DAYS_REPLAY,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    def _build_signal(
        self,
        evaluation: dict[str, Any],
        *,
        engine_version: str,
        moves: list[_CheapMoveCandidate],
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        layer4 = evaluation["layer4"]
        forward = dict(layer4.get("forward_outcome") or {})
        context = evaluation.get("context") or {}
        bar = evaluation["bar"]
        linked_move = _nearest_bullish_move(moves, bar)
        move_start_bar = linked_move.start_bar if linked_move else None
        bars_before_expansion = (move_start_bar - bar) if move_start_bar is not None else None
        points_before_expansion = None
        if move_start_bar is not None and bars_before_expansion is not None and bars_before_expansion >= 0:
            entry = float(forward.get("entry") or frame.iloc[bar]["Close"])
            move_low = float(frame.iloc[bar : move_start_bar + 1]["Low"].astype(float).min())
            points_before_expansion = round(max(entry - move_low, 0.0), 2)

        classification = _classify_failed_buy_signal(
            {
                "mfe_points": forward.get("mfe_points"),
                "mae_points": forward.get("mae_points"),
                "win": forward.get("win"),
            },
            context=context,
        )

        return {
            "timestamp": evaluation["timestamp"],
            "bar": bar,
            "symbol": DEFAULT_SYMBOL,
            "timeframe": MOVE_DETECTION_TIMEFRAME,
            "direction": "BUY",
            "engine_version": engine_version,
            "model_id": layer4.get("model_id"),
            "entry": layer4.get("entry"),
            "stop_loss": layer4.get("stop_loss"),
            "target_1": layer4.get("target_1"),
            "target_2": layer4.get("target_2"),
            "target_3": layer4.get("target_3"),
            "signal_reason_stack": layer4.get("signal_reason_stack"),
            "realized_pnl_points": forward.get("realized_pnl_points"),
            "mfe_points": forward.get("mfe_points"),
            "mae_points": forward.get("mae_points"),
            "hit_1r": forward.get("hit_1r"),
            "hit_2r": forward.get("hit_2r"),
            "hit_3r": forward.get("hit_3r"),
            "win": forward.get("win"),
            "classification": classification,
            "trade_duration_bars": FORWARD_BARS,
            "move_start_bar": move_start_bar,
            "move_start_time": str(frame.iloc[move_start_bar]["Date"]) if move_start_bar is not None else None,
            "bars_before_expansion": bars_before_expansion,
            "points_before_expansion": points_before_expansion,
            "signal_before_expansion": bars_before_expansion is not None and bars_before_expansion >= 0,
            "layers": {
                "layer1": evaluation["layer1"],
                "layer2": evaluation["layer2"],
                "layer3": evaluation["layer3"],
                "layer5": evaluation["layer5"],
            },
        }

    def _replay_combined(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        replay_bars: list[int],
        moves: list[_CheapMoveCandidate],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
        engine_configs: list[tuple[str, BaseBuyCandidateEngine]] = [
            ("buy_v1", self.buy_v1_engine),
            ("buy_v2", self.buy_v2_engine),
            ("buy_v3", self.buy_v3_engine),
        ]
        for key in ABLATION_VARIANTS:
            engine_configs.append((f"ablation_{key}", self.ablation_engines[key]))

        signals: dict[str, list[dict[str, Any]]] = {key: [] for key, _ in engine_configs}
        emitted_bars: dict[str, set[int]] = {key: set() for key, _ in engine_configs}
        rejections: dict[str, dict[str, int]] = {key: {} for key, _ in engine_configs}

        valid_bars = [
            bar
            for bar in replay_bars
            if bar >= PRE_EXPANSION_LOOKBACK and bar < len(frame) - FORWARD_BARS
        ]
        logger.info("Precomputing shared event detection for %s replay bars...", len(valid_bars))
        bar_events_cache, lookback_cache = _precompute_bar_events(
            self.buy_v3_engine,
            frame=frame,
            calendar=calendar,
            replay_bars=valid_bars,
        )

        context_cache: dict[int, dict[str, str]] = {}
        context_log_every = max(len(valid_bars) // 10, 1)
        context_started = time.perf_counter()
        for index, bar in enumerate(valid_bars):
            if index > 0 and index % context_log_every == 0:
                logger.info(
                    "Context precompute progress: %s/%s bars (%.0f%%) elapsed %.0fs",
                    index,
                    len(valid_bars),
                    index / len(valid_bars) * 100,
                    time.perf_counter() - context_started,
                )
            context_cache[bar] = self.buy_v3_engine._context_at_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
            )
        logger.info(
            "Context precompute complete for %s bars in %.0fs",
            len(valid_bars),
            time.perf_counter() - context_started,
        )

        total = len(valid_bars)
        log_every = max(total // 20, 1)
        started = time.perf_counter()

        for index, bar in enumerate(valid_bars):
            if index > 0 and index % log_every == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "BUY_V3 replay progress: %s/%s bars (%.0f%%) elapsed %.0fs | "
                    "v1=%s v2=%s v3=%s",
                    index,
                    total,
                    index / total * 100,
                    elapsed,
                    len(signals["buy_v1"]),
                    len(signals["buy_v2"]),
                    len(signals["buy_v3"]),
                )

            context = context_cache[bar]
            lookback_events = lookback_cache[bar]
            bar_events = bar_events_cache[bar]

            for key, engine in engine_configs:
                version_label = key.upper().replace("ABLATION_", "ABLATION_")
                evaluation = _evaluate_buy_bar_fast(
                    engine,
                    frame=frame,
                    bar=bar,
                    context=context,
                    lookback_events=lookback_events,
                    bar_events=bar_events,
                    emitted_bars=emitted_bars[key],
                )
                if evaluation["verdict"] == "BUY":
                    signals[key].append(
                        self._build_signal(evaluation, engine_version=version_label, moves=moves, frame=frame),
                    )
                    emitted_bars[key].add(bar)
                else:
                    for reason in evaluation["layer5"]["reason_codes"]:
                        rejections[key][reason] = rejections[key].get(reason, 0) + 1

        logger.info(
            "BUY_V3 replay complete: v1=%s v2=%s v3=%s ablations=%s in %.0fs",
            len(signals["buy_v1"]),
            len(signals["buy_v2"]),
            len(signals["buy_v3"]),
            sum(len(signals[f"ablation_{k}"]) for k in ABLATION_VARIANTS),
            time.perf_counter() - started,
        )
        return signals, rejections

    def _load_missed_reversals(self) -> list[dict[str, Any]]:
        if not DEFAULT_MISSED_REVERSAL_PATH.exists():
            logger.warning("Missed reversal export missing: %s", DEFAULT_MISSED_REVERSAL_PATH)
            return []
        payload = json.loads(DEFAULT_MISSED_REVERSAL_PATH.read_text(encoding="utf-8"))
        return list(payload.get("per_missed_reversal") or [])

    def _load_v2_export(self) -> dict[str, Any]:
        if not DEFAULT_V2_REPORT_PATH.exists():
            logger.warning("BUY_V2 export missing: %s", DEFAULT_V2_REPORT_PATH)
            return {}
        return json.loads(DEFAULT_V2_REPORT_PATH.read_text(encoding="utf-8"))

    def run(self, metadata: dict[str, Any]) -> BuyV3CandidateValidationReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = end - timedelta(days=TRADING_DAYS_REPLAY)

        logger.info(
            "BUY_V3 validation starting: %s trading days, %s 5M",
            TRADING_DAYS_REPLAY,
            DEFAULT_SYMBOL,
        )

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=TRADING_DAYS_REPLAY,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        replay_dates = _last_n_trading_day_set(frame, TRADING_DAYS_REPLAY)
        train_dates, validate_dates = _split_trading_day_sets(replay_dates)
        bar_dates = pd.to_datetime(frame["Date"]).dt.date
        replay_bars = [index for index, day in enumerate(bar_dates) if day in replay_dates]

        logger.info("Loading enriched context and intel frames...")
        enriched = self.buy_v1_engine.context_builder.enrich(frame)
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        intel_frames: dict[str, pd.DataFrame] = {"5M": self.buy_v1_engine.intelligence.enrich(frame)}
        for tf in ("15M", "1H"):
            tf_path = filter_engine._ensure_pipeline(tf, start, end)
            tf_frame = pd.read_csv(tf_path).reset_index(drop=True)
            intel_frames[tf] = self.buy_v1_engine.intelligence.enrich(tf_frame)
        intel_frames["1D"] = self.buy_v1_engine.intelligence.enrich(
            self.buy_v1_engine._resample_daily(intel_frames["1H"]),
        )

        logger.info("Detecting bullish moves (threshold=%s)...", MOVE_DETECTION_THRESHOLD)
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, MOVE_DETECTION_THRESHOLD),
        )
        logger.info("Detected %s deduped moves >= %s pts", len(moves), MOVE_DETECTION_THRESHOLD)

        all_signals, all_rejections = self._replay_combined(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            replay_bars=replay_bars,
            moves=moves,
        )

        v1_signals = all_signals["buy_v1"]
        v2_signals = all_signals["buy_v2"]
        v3_signals = all_signals["buy_v3"]

        v1_stats = _build_statistics(v1_signals, trading_days=TRADING_DAYS_REPLAY)
        v2_stats = _build_statistics(v2_signals, trading_days=TRADING_DAYS_REPLAY)
        v3_stats = _build_statistics(v3_signals, trading_days=TRADING_DAYS_REPLAY)
        v1_capture = _bullish_point_capture(moves, v1_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        v2_capture = _bullish_point_capture(moves, v2_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        v3_capture = _bullish_point_capture(moves, v3_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)

        v3_train = _filter_signals_by_dates(v3_signals, frame, train_dates)
        v3_validate = _filter_signals_by_dates(v3_signals, frame, validate_dates)
        v1_train = _filter_signals_by_dates(v1_signals, frame, train_dates)
        v1_validate = _filter_signals_by_dates(v1_signals, frame, validate_dates)
        v2_train = _filter_signals_by_dates(v2_signals, frame, train_dates)
        v2_validate = _filter_signals_by_dates(v2_signals, frame, validate_dates)

        walk_forward = {
            "train_trading_days": len(train_dates),
            "validate_trading_days": len(validate_dates),
            "train_start_date": min(train_dates).isoformat() if train_dates else "",
            "train_end_date": max(train_dates).isoformat() if train_dates else "",
            "validate_start_date": min(validate_dates).isoformat() if validate_dates else "",
            "validate_end_date": max(validate_dates).isoformat() if validate_dates else "",
            "train": {
                "buy_v1": {
                    "overall_statistics": _walk_forward_metrics(v1_train, period_days=len(train_dates)),
                    "signals_emitted_count": len(v1_train),
                },
                "buy_v2": {
                    "overall_statistics": _walk_forward_metrics(v2_train, period_days=len(train_dates)),
                    "signals_emitted_count": len(v2_train),
                },
                "buy_v3": {
                    "overall_statistics": _walk_forward_metrics(v3_train, period_days=len(train_dates)),
                    "signals_emitted_count": len(v3_train),
                },
            },
            "validate": {
                "buy_v1": {
                    "overall_statistics": _walk_forward_metrics(v1_validate, period_days=len(validate_dates)),
                    "signals_emitted_count": len(v1_validate),
                },
                "buy_v2": {
                    "overall_statistics": _walk_forward_metrics(v2_validate, period_days=len(validate_dates)),
                    "signals_emitted_count": len(v2_validate),
                },
                "buy_v3": {
                    "overall_statistics": _walk_forward_metrics(v3_validate, period_days=len(validate_dates)),
                    "signals_emitted_count": len(v3_validate),
                },
            },
        }

        v2_export = self._load_v2_export()
        v2_false_cohort = _load_v2_false_reversal_cohort(v2_export) if v2_export else []
        missed_rows = self._load_missed_reversals()

        false_removal = _false_reversal_removal_analysis(
            v2_false_cohort=v2_false_cohort,
            v1_signals=v1_signals,
            v2_signals=v2_signals,
            v3_signals=v3_signals,
            missed_rows=missed_rows,
            moves=moves,
            frame=frame,
            replay_dates=replay_dates,
        )

        logger.info("Running ablation analysis...")
        ablation_results: dict[str, dict[str, Any]] = {}
        for key, meta in ABLATION_VARIANTS.items():
            ablation_signals = all_signals[f"ablation_{key}"]
            ablation_results[key] = _ablation_metrics(
                ablation_signals,
                moves,
                frame,
                replay_dates,
                variant_key=key,
                variant_meta=meta,
            )
            logger.info(
                "Ablation %s: %s signals, WR %s%%, false rate %s%%",
                key,
                len(ablation_signals),
                ablation_results[key]["overall_statistics"].get("win_rate_pct"),
                ablation_results[key]["false_reversal_rate_pct"],
            )

        ablation_ranking = _ablation_contribution_ranking(ablation_results)

        signal_timing = {
            "buy_v1": _signal_timing_analysis(v1_signals, engine_key="BUY_V1"),
            "buy_v2": _signal_timing_analysis(v2_signals, engine_key="BUY_V2"),
            "buy_v3": _signal_timing_analysis(v3_signals, engine_key="BUY_V3"),
        }

        tradeability = {
            "buy_v1": _tradeability_analysis(v1_signals, moves, frame, replay_dates, engine_key="BUY_V1"),
            "buy_v2": _tradeability_analysis(v2_signals, moves, frame, replay_dates, engine_key="BUY_V2"),
            "buy_v3": _tradeability_analysis(v3_signals, moves, frame, replay_dates, engine_key="BUY_V3"),
        }

        sell_v5 = _sell_v5_benchmark()
        production_safety = {
            "buy_v1": _passes_production_gates(v1_stats, v1_capture),
            "buy_v2": _passes_production_gates(v2_stats, v2_capture),
            "buy_v3": _passes_production_gates(v3_stats, v3_capture),
            "gates_definition": PRODUCTION_GATES,
        }

        final_verdict = _final_verdict(
            v3_stats=v3_stats,
            v3_capture=v3_capture,
            v1_stats=v1_stats,
            v2_stats=v2_stats,
            false_removal=false_removal,
            walk_forward=walk_forward,
            production_safety=production_safety,
            sell_v5=sell_v5,
            ablation_ranking=ablation_ranking,
        )

        replay_start = min(replay_dates).isoformat() if replay_dates else ""
        replay_end = max(replay_dates).isoformat() if replay_dates else ""

        avg_duration = round(
            mean(float(s.get("trade_duration_bars") or FORWARD_BARS) for s in v3_signals),
            2,
        ) if v3_signals else 0.0

        return BuyV3CandidateValidationReport(
            report_type="BUY_V3 Candidate Validation",
            engines_compared=["BUY_V1", "BUY_V2", "BUY_V3", "SELL_V5"],
            buy_v1_formula=list(BUY_V1_COMPONENTS),
            buy_v2_formula=list(BUY_V2_COMPONENTS),
            buy_v3_formula=list(BUY_V3_EVENTS) + [NEAR_SUPPORT_LABEL],
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            trading_days_replayed=TRADING_DAYS_REPLAY,
            replay_start_date=replay_start,
            replay_end_date=replay_end,
            walk_forward=walk_forward,
            methodology={
                "research_only": True,
                "actual_replay": True,
                "synthesis_only": False,
                "buy_v3_engine": "BuyV3CandidateEngine",
                "base_architecture": "BaseBuyCandidateEngine / SmartMoneyEngineV3Engine five-layer stack",
                "buy_v3_formula": BUY_V3_FORMULA_TEXT,
                "event_detection": "Nifty50LiquidityDirectionDecisionMatrixResearch._detect_events_at_bar",
                "formula_lookback_bars": PRE_EXPANSION_LOOKBACK,
                "walk_forward_split": f"train {TRAIN_TRADING_DAYS} / validate {VALIDATE_TRADING_DAYS} trading days",
                "false_reversal_baseline_export": str(DEFAULT_V2_REPORT_PATH),
                "ablation_variants": list(ABLATION_VARIANTS.keys()),
                "move_detection_threshold": MOVE_DETECTION_THRESHOLD,
                "point_capture_thresholds": list(POINT_CAPTURE_THRESHOLDS),
                "no_lookahead": True,
            },
            replay_rules={
                "symbol": DEFAULT_SYMBOL,
                "timeframe": MOVE_DETECTION_TIMEFRAME,
                "trading_days": TRADING_DAYS_REPLAY,
                "candle_by_candle": True,
                "combined_replay": "BUY_V1 + BUY_V2 + BUY_V3 + 5 ablations evaluated once per bar",
                "no_future_bos_choch_fvg": True,
            },
            comparison={
                "buy_v1": {
                    "formula_text": BUY_V1_FORMULA_TEXT,
                    "model_id": BUY_V1_MODEL_ID,
                    "overall_statistics": v1_stats,
                    "point_capture": v1_capture,
                    "layer_rejection_summary": all_rejections["buy_v1"],
                    "signals_emitted_count": len(v1_signals),
                    "classification_summary": _classification_summary(v1_signals),
                    "average_trade_duration_bars": round(
                        mean(float(s.get("trade_duration_bars") or FORWARD_BARS) for s in v1_signals),
                        2,
                    )
                    if v1_signals
                    else 0.0,
                },
                "buy_v2": {
                    "formula_text": BUY_V2_FORMULA_TEXT,
                    "model_id": BUY_V2_MODEL_ID,
                    "overall_statistics": v2_stats,
                    "point_capture": v2_capture,
                    "layer_rejection_summary": all_rejections["buy_v2"],
                    "signals_emitted_count": len(v2_signals),
                    "classification_summary": _classification_summary(v2_signals),
                    "average_trade_duration_bars": round(
                        mean(float(s.get("trade_duration_bars") or FORWARD_BARS) for s in v2_signals),
                        2,
                    )
                    if v2_signals
                    else 0.0,
                },
                "buy_v3": {
                    "formula_text": BUY_V3_FORMULA_TEXT,
                    "model_id": BUY_V3_MODEL_ID,
                    "overall_statistics": v3_stats,
                    "point_capture": v3_capture,
                    "layer_rejection_summary": all_rejections["buy_v3"],
                    "signals_emitted_count": len(v3_signals),
                    "classification_summary": _classification_summary(v3_signals),
                    "average_trade_duration_bars": avg_duration,
                    "average_mfe": v3_stats.get("average_mfe"),
                    "average_mae": v3_stats.get("average_mae"),
                },
                "sell_v5_benchmark": sell_v5,
            },
            false_reversal_removal=false_removal,
            ablation_analysis={
                "variants": ablation_results,
                "contribution_ranking": ablation_ranking,
            },
            signal_timing=signal_timing,
            tradeability=tradeability,
            per_signal_details={
                "buy_v1": v1_signals,
                "buy_v2": v2_signals,
                "buy_v3": v3_signals,
            },
            failed_signal_classification={
                "buy_v1": _classification_summary(v1_signals),
                "buy_v2": _classification_summary(v2_signals),
                "buy_v3": _classification_summary(v3_signals),
            },
            sell_v5_benchmark=sell_v5,
            production_safety_check=production_safety,
            final_verdict=final_verdict,
            conclusions=[
                f"BUY_V1={len(v1_signals)} BUY_V2={len(v2_signals)} BUY_V3={len(v3_signals)} signals over {TRADING_DAYS_REPLAY} days.",
                (
                    f"WR BUY_V3 {v3_stats.get('win_rate_pct')}% | PF {v3_stats.get('profit_factor')} | "
                    f"{v3_stats.get('signals_per_month')}/mo."
                ),
                false_removal.get("summary", ""),
                (
                    f"40+ capture BUY_V3 {v3_capture.get('40', {}).get('capture_rate_pct')}% | "
                    f"60+ {v3_capture.get('60', {}).get('capture_rate_pct')}%."
                ),
                (
                    f"Verdict: {final_verdict['classification']} | "
                    f"40-60pt tradeable: {final_verdict['can_deliver_40_60pt_opportunities']['answer']} | "
                    f"False reversal removal: {final_verdict['false_reversal_removal']['answer']}."
                ),
            ],
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: BuyV3CandidateValidationReport, report_path: Path) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("BUY_V3 candidate validation exported: %s", report_path)
        return report_path


def generate_buy_v3_candidate_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> BuyV3CandidateValidationReport:
    """Run BUY_V3 replay validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise BuyV3CandidateValidationError(f"Filter research report not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    research = BuyV3CandidateValidationResearch()
    report = research.run(metadata)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_buy_v3_candidate_validation_report()
    except BuyV3CandidateValidationError as exc:
        logger.error("BUY_V3 candidate validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected BUY_V3 candidate validation error")
        return 1

    v3 = report.comparison["buy_v3"]["overall_statistics"]
    removal = report.false_reversal_removal
    verdict = report.final_verdict
    print("BUY_V3 Candidate Validation Summary")
    print(f"BUY_V3 signals: {v3['signals_emitted']} | {v3['signals_per_month']}/month")
    print(f"BUY_V3 WR: {v3['win_rate_pct']}% | PF: {v3['profit_factor']} | Expectancy: {v3['expectancy']}")
    print(
        f"False reversals: removed {removal.get('removed_by_buy_v3')}/"
        f"{removal.get('baseline_false_reversal_count')} | remaining {removal.get('remaining_false_reversals')}"
    )
    print(f"Classification: {verdict['classification']}")
    print(f"40-60pt tradeable: {verdict['can_deliver_40_60pt_opportunities']['answer']}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
