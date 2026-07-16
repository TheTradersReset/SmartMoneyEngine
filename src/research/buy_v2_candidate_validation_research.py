"""
BUY_V2 Candidate Validation — research with actual replay.

Validates Failed Breakdown + Gap Reversal (BUY_V2) vs BUY_V1 on 120-day NIFTY50 5M replay.
Walk-forward: train first 80 trading days, validate last 40. Cross-checks 47 missed reversals
from buy_v1_missed_reversal_analysis.json. Research-only; no production modifications.
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
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.buy_failure_anatomy_research import (
    DEAD_CAT_MAX_POINTS,
    NEAR_SUPPORT_LABEL,
    REAL_REVERSAL_MIN_POINTS,
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
    SmartMoneyEngineV3Engine,
)
from src.research.smartmoneyengine_v4_candidate_validation_research import (
    _build_statistics,
    _last_n_trading_day_set,
    _profit_factor,
    _signal_before_move,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_v2_candidate_validation.json"
DEFAULT_MISSED_REVERSAL_PATH = RESEARCH_DIR / "buy_v1_missed_reversal_analysis.json"
DEFAULT_V5_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v5_candidate_validation.json"

TRADING_DAYS_REPLAY = 120
TRAIN_TRADING_DAYS = 80
VALIDATE_TRADING_DAYS = 40
POINT_CAPTURE_THRESHOLDS = (40, 60, 80, 100, 200)
MOVE_DETECTION_THRESHOLD = 40
NO_EXPANSION_MFE = 40.0

BUY_V1_COMPONENTS = ("Liquidity Grab", "Failed Breakdown", NEAR_SUPPORT_LABEL)
BUY_V1_FORMULA_TEXT = "Liquidity Grab + Failed Breakdown + Near Support"
BUY_V1_MODEL_ID = "LDM-BUY-V1"

BUY_V2_COMPONENTS = ("Failed Breakdown", "Gap Reversal")
BUY_V2_FORMULA_TEXT = "Failed Breakdown + Gap Reversal"
BUY_V2_MODEL_ID = "LDM-BUY-V2"

BUY_CONFIRMATION_CANDLES = frozenset(
    {
        "Hammer",
        "Bullish Engulfing",
        "Morning Star",
        "Marubozu",
        "None",
    },
)

PRODUCTION_GATES = {
    "win_rate_min_pct": 65.0,
    "profit_factor_min": 2.0,
    "signals_per_month_min": 20.0,
    "capture_40_plus_min_pct": 1.0,
}


class BuyV2CandidateValidationError(Exception):
    """Raised when BUY_V2 candidate validation fails."""


@dataclass
class BuyV2CandidateValidationReport:
    """BUY_V2 vs BUY_V1 replay validation output."""

    report_type: str
    engines_compared: list[str]
    buy_v1_formula: list[str]
    buy_v2_formula: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    walk_forward: dict[str, Any]
    methodology: dict[str, Any]
    replay_rules: dict[str, Any]
    comparison: dict[str, Any]
    per_signal_details: dict[str, list[dict[str, Any]]]
    missed_reversal_recovery: dict[str, Any]
    failed_signal_classification: dict[str, Any]
    condition_attribution: dict[str, Any]
    sell_v5_benchmark: dict[str, Any]
    production_safety_check: dict[str, Any]
    final_verdicts: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _events_in_lookback(
    engine: SmartMoneyEngineV3Engine,
    *,
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    bar: int,
    lookback: int = PRE_EXPANSION_LOOKBACK,
) -> set[str]:
    start = max(0, bar - lookback)
    found: set[str] = set()
    for offset in range(start, bar + 1):
        for event in engine._detect_events_at_bar(frame, calendar, offset):
            found.add(event)
    return found


def _nearest_bullish_move(
    moves: list[_CheapMoveCandidate],
    signal_bar: int,
    *,
    forward_bars: int = FORWARD_BARS,
) -> _CheapMoveCandidate | None:
    candidates = [
        move
        for move in moves
        if move.direction == "bullish"
        and signal_bar <= move.start_bar <= signal_bar + forward_bars
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.start_bar)


def _classify_failed_buy_signal(
    signal: dict[str, Any],
    *,
    context: dict[str, str],
) -> str:
    mfe = float(signal.get("mfe_points") or 0.0)
    mae = float(signal.get("mae_points") or 0.0)
    win = bool(signal.get("win"))
    htf = context.get("htf_trend", "Neutral")

    if mfe < NO_EXPANSION_MFE:
        return "No Expansion"
    if htf == "Bearish" and mfe < REAL_REVERSAL_MIN_POINTS:
        return "Counter Trend Bounce"
    if mae > mfe and not win:
        return "Bull Trap"
    if mfe >= REAL_REVERSAL_MIN_POINTS and win:
        return "Real Reversal"
    if mfe < DEAD_CAT_MAX_POINTS:
        return "Dead Cat Bounce"
    if mfe < REAL_REVERSAL_MIN_POINTS:
        return "Range Failure"
    return "False Reversal"


def _bullish_point_capture(
    moves: list[_CheapMoveCandidate],
    signals: list[dict[str, Any]],
    replay_dates: set[date],
    frame: pd.DataFrame,
    thresholds: tuple[int, ...],
) -> dict[str, Any]:
    signal_by_bar = {signal["bar"]: signal for signal in signals}
    results: dict[str, Any] = {}
    for threshold in thresholds:
        bullish = [
            move
            for move in moves
            if move.direction == "bullish"
            and move.magnitude >= threshold
            and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
        ]
        captured = 0
        for move in bullish:
            pre_start = max(0, move.start_bar - PRE_EXPANSION_LOOKBACK)
            for bar in range(pre_start, move.start_bar + 1):
                if bar in signal_by_bar:
                    captured += 1
                    break
        total = len(bullish)
        results[str(threshold)] = {
            "total_bullish_moves": total,
            "signals_before_move": captured,
            "missed_moves": total - captured,
            "capture_rate_pct": round(captured / max(total, 1) * 100, 2),
        }
    return results


def _split_trading_day_sets(replay_dates: set[date]) -> tuple[set[date], set[date]]:
    ordered = sorted(replay_dates)
    if len(ordered) < TRAIN_TRADING_DAYS + VALIDATE_TRADING_DAYS:
        split_at = max(len(ordered) - VALIDATE_TRADING_DAYS, 1)
    else:
        split_at = TRAIN_TRADING_DAYS
    return set(ordered[:split_at]), set(ordered[split_at:])


def _filter_signals_by_dates(signals: list[dict[str, Any]], frame: pd.DataFrame, dates: set[date]) -> list[dict[str, Any]]:
    if not dates:
        return []
    return [
        signal
        for signal in signals
        if pd.to_datetime(frame.iloc[signal["bar"]]["Date"]).date() in dates
    ]


def _walk_forward_metrics(
    signals: list[dict[str, Any]],
    *,
    period_days: int,
) -> dict[str, Any]:
    return _build_statistics(signals, trading_days=period_days)


def _passes_production_gates(stats: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    wr = float(stats.get("win_rate_pct") or 0.0)
    pf = stats.get("profit_factor")
    spm = float(stats.get("signals_per_month") or 0.0)
    capture_40 = float(capture.get("40", {}).get("capture_rate_pct") or 0.0)
    checks = {
        "win_rate_above_65_pct": wr > PRODUCTION_GATES["win_rate_min_pct"],
        "profit_factor_above_2": pf is not None and float(pf) > PRODUCTION_GATES["profit_factor_min"],
        "signals_per_month_20_plus": spm >= PRODUCTION_GATES["signals_per_month_min"],
        "capture_40_plus": capture_40 > PRODUCTION_GATES["capture_40_plus_min_pct"],
    }
    checks["all_pass"] = all(checks.values())
    return checks


class BaseBuyCandidateEngine(SmartMoneyEngineV3Engine):
    """Shared BUY replay engine base (five-layer stack, bullish execution)."""

    DIRECTION = "BUY"
    REQUIRED_EVENTS: tuple[str, ...] = ()
    REQUIRED_LOCATION: str | None = None
    REQUIRE_BULLISH_ALIGNMENT = True

    def _layer1_formula_events(
        self,
        *,
        frame: pd.DataFrame,
        calendar: pd.DataFrame,
        bar: int,
    ) -> dict[str, Any]:
        lookback_events = _events_in_lookback(self, frame=frame, calendar=calendar, bar=bar)
        bar_events = set(self._detect_events_at_bar(frame, calendar, bar))
        matched = [event for event in self.REQUIRED_EVENTS if event in lookback_events]
        return {
            "active": len(matched) == len(self.REQUIRED_EVENTS),
            "events_detected": sorted(lookback_events),
            "events_at_bar": sorted(bar_events),
            "formula_events_matched": matched,
            "formula_events_missing": [event for event in self.REQUIRED_EVENTS if event not in lookback_events],
            "lookback_bars": PRE_EXPANSION_LOOKBACK,
        }

    def _layer2_directional_filter(self, context: dict[str, str]) -> dict[str, Any]:
        htf = context.get("htf_trend", "Neutral")
        vwap = context.get("vwap")
        ema = context.get("ema_structure", "Mixed")
        location_ok = (
            context.get("location") == self.REQUIRED_LOCATION
            if self.REQUIRED_LOCATION
            else True
        )
        if self.REQUIRE_BULLISH_ALIGNMENT:
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
            "location_required": self.REQUIRED_LOCATION,
            "location_ok": location_ok,
            "aligned": aligned,
        }

    def _layer3_confirmation(self, context: dict[str, str]) -> dict[str, Any]:
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

    def _layer5_no_trade_filters(
        self,
        *,
        layer1: dict[str, Any],
        layer2: dict[str, Any],
        layer3: dict[str, Any],
        context: dict[str, str],
        bar: int,
        emitted_bars: set[int],
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if not layer1["active"]:
            reasons.append("FORMULA_INCOMPLETE")
            for missing in layer1.get("formula_events_missing", []):
                reasons.append(f"MISSING_{missing.upper().replace(' ', '_')}")
        if layer2.get("htf_trend") == "Bearish":
            reasons.append("HTF_CONFLICT")
        if not layer2.get("location_ok"):
            reasons.append("LOCATION_MISMATCH")
        if not layer2.get("aligned"):
            reasons.append("DIRECTION_NOT_ALIGNED")
        if not layer3.get("confirmed"):
            reasons.append("VOLUME_FAILED")
        if context.get("location") == "Mid Range" and self.REQUIRED_LOCATION:
            reasons.append("LOCATION_MID_RANGE")
        if bar in emitted_bars:
            reasons.append("DUPLICATE_BAR")
        return {"pass": not reasons, "reason_codes": reasons}

    def _layer4_execution(
        self,
        frame: pd.DataFrame,
        bar: int,
        *,
        layer1: dict[str, Any],
        layer2: dict[str, Any],
        layer3: dict[str, Any],
        context: dict[str, str],
    ) -> dict[str, Any] | None:
        outcome = self._trade_outcome(frame, bar, "bullish")
        if not outcome:
            return None
        entry = outcome["entry"]
        risk = outcome["risk_points"]
        target_liq = outcome.get("target")
        return {
            "model_id": self.MODEL_ID,
            "direction": "BUY",
            "entry": entry,
            "stop_loss": outcome["stop_loss"],
            "target_1": round(float(entry) + risk, 2),
            "target_2": round(float(entry) + 2 * risk, 2),
            "target_3": round(float(entry) + 3 * risk, 2),
            "liquidity_target": target_liq,
            "risk_points": risk,
            "signal_reason_stack": {
                "layer1": layer1["formula_events_matched"],
                "layer2": {
                    "htf_trend": layer2["htf_trend"],
                    "vwap": layer2["vwap_state"],
                    "ema_structure": layer2["ema_structure"],
                    "location": layer2.get("location"),
                },
                "layer3": {
                    "confirmation_candle": layer3["confirmation_candle"],
                    "volume": layer3["volume_bucket"],
                },
            },
            "forward_outcome": outcome,
        }

    def evaluate_bar(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        bar: int,
        emitted_bars: set[int],
    ) -> dict[str, Any]:
        context = self._context_at_bar(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            bar=bar,
        )
        layer1 = self._layer1_formula_events(frame=frame, calendar=calendar, bar=bar)
        layer2 = self._layer2_directional_filter(context)
        layer3 = self._layer3_confirmation(context)
        layer5 = self._layer5_no_trade_filters(
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
            execution = self._layer4_execution(
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


class BuyV1CandidateEngine(BaseBuyCandidateEngine):
    """BUY_V1: Liquidity Grab + Failed Breakdown + Near Support."""

    MODEL_ID = BUY_V1_MODEL_ID
    REQUIRED_EVENTS = BUY_V1_COMPONENTS[:2]
    REQUIRED_LOCATION = NEAR_SUPPORT_LABEL


class BuyV2CandidateEngine(BaseBuyCandidateEngine):
    """BUY_V2: Failed Breakdown + Gap Reversal."""

    MODEL_ID = BUY_V2_MODEL_ID
    REQUIRED_EVENTS = BUY_V2_COMPONENTS
    REQUIRED_LOCATION = None


class BuyV2CandidateValidationResearch:
    """Replay BUY_V2 vs BUY_V1 on 120-day NIFTY50 window with walk-forward."""

    def __init__(self) -> None:
        self.buy_v1_engine = BuyV1CandidateEngine()
        self.buy_v2_engine = BuyV2CandidateEngine()
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
            move_low = float(frame.iloc[bar: move_start_bar + 1]["Low"].astype(float).min())
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
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, int],
        dict[str, int],
    ]:
        v1_signals: list[dict[str, Any]] = []
        v2_signals: list[dict[str, Any]] = []
        v1_emitted_bars: set[int] = set()
        v2_emitted_bars: set[int] = set()
        v1_rejections: dict[str, int] = {}
        v2_rejections: dict[str, int] = {}
        total = len(replay_bars)
        log_every = max(total // 20, 1)
        started = time.perf_counter()

        for index, bar in enumerate(replay_bars):
            if index > 0 and index % log_every == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "BUY replay progress: %s/%s bars (%.0f%%) elapsed %.0fs",
                    index,
                    total,
                    index / total * 100,
                    elapsed,
                )

            if bar < PRE_EXPANSION_LOOKBACK or bar >= len(frame) - FORWARD_BARS:
                continue

            v1_eval = self.buy_v1_engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=v1_emitted_bars,
            )
            v2_eval = self.buy_v2_engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=v2_emitted_bars,
            )

            if v1_eval["verdict"] == "BUY":
                v1_signals.append(self._build_signal(v1_eval, engine_version="BUY_V1", moves=moves, frame=frame))
                v1_emitted_bars.add(bar)
            else:
                for reason in v1_eval["layer5"]["reason_codes"]:
                    v1_rejections[reason] = v1_rejections.get(reason, 0) + 1

            if v2_eval["verdict"] == "BUY":
                v2_signals.append(self._build_signal(v2_eval, engine_version="BUY_V2", moves=moves, frame=frame))
                v2_emitted_bars.add(bar)
            else:
                for reason in v2_eval["layer5"]["reason_codes"]:
                    v2_rejections[reason] = v2_rejections.get(reason, 0) + 1

        logger.info(
            "BUY replay complete: BUY_V1=%s BUY_V2=%s signals in %.0fs",
            len(v1_signals),
            len(v2_signals),
            time.perf_counter() - started,
        )
        return v1_signals, v2_signals, v1_rejections, v2_rejections

    def _classification_summary(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(signal.get("classification", "Unknown") for signal in signals)
        total = len(signals)
        rates = {
            label: round(100.0 * count / max(total, 1), 2)
            for label, count in counts.items()
        }
        return {
            "counts": dict(counts),
            "rates_pct": rates,
            "real_reversal_rate_pct": rates.get("Real Reversal", 0.0),
            "false_reversal_rate_pct": rates.get("False Reversal", 0.0),
            "dead_cat_bounce_rate_pct": rates.get("Dead Cat Bounce", 0.0),
            "no_expansion_rate_pct": rates.get("No Expansion", 0.0),
            "counter_trend_bounce_rate_pct": rates.get("Counter Trend Bounce", 0.0),
            "bull_trap_rate_pct": rates.get("Bull Trap", 0.0),
            "range_failure_rate_pct": rates.get("Range Failure", 0.0),
        }

    def _condition_attribution(self, signals: list[dict[str, Any]], *, engine: str) -> dict[str, Any]:
        components = BUY_V2_COMPONENTS if engine == "BUY_V2" else BUY_V1_COMPONENTS
        attribution: dict[str, dict[str, Any]] = {}
        for component in components:
            wins = losses = frequency = false_reversals = 0
            for signal in signals:
                layer1 = signal.get("layers", {}).get("layer1", {})
                matched = layer1.get("formula_events_matched") or layer1.get("events_detected") or []
                if component not in matched and component not in (layer1.get("events_detected") or []):
                    continue
                frequency += 1
                if signal.get("win"):
                    wins += 1
                else:
                    losses += 1
                if signal.get("classification") == "False Reversal":
                    false_reversals += 1
            attribution[component] = {
                "signals_with_condition": frequency,
                "wins": wins,
                "losses": losses,
                "false_reversals": false_reversals,
                "win_rate_pct": round(100.0 * wins / max(frequency, 1), 2),
            }

        most_wins = max(attribution.items(), key=lambda item: item[1]["wins"], default=(None, {}))
        most_losses = max(attribution.items(), key=lambda item: item[1]["losses"], default=(None, {}))
        most_frequency = max(attribution.items(), key=lambda item: item[1]["signals_with_condition"], default=(None, {}))
        most_false = max(attribution.items(), key=lambda item: item[1]["false_reversals"], default=(None, {}))
        return {
            "engine": engine,
            "by_condition": attribution,
            "most_wins_condition": most_wins[0],
            "most_losses_condition": most_losses[0],
            "most_frequency_condition": most_frequency[0],
            "most_false_reversals_condition": most_false[0],
        }

    def _load_missed_reversals(self) -> list[dict[str, Any]]:
        if not DEFAULT_MISSED_REVERSAL_PATH.exists():
            logger.warning("Missed reversal export missing: %s", DEFAULT_MISSED_REVERSAL_PATH)
            return []
        payload = json.loads(DEFAULT_MISSED_REVERSAL_PATH.read_text(encoding="utf-8"))
        return list(payload.get("per_missed_reversal") or [])

    def _missed_reversal_recovery(
        self,
        *,
        missed_rows: list[dict[str, Any]],
        v1_signals: list[dict[str, Any]],
        v2_signals: list[dict[str, Any]],
        moves: list[_CheapMoveCandidate],
        frame: pd.DataFrame,
        replay_dates: set[date],
    ) -> dict[str, Any]:
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

        export_dates = {str(row.get("date", ""))[:10] for row in missed_rows}
        cohort_size = len(missed_rows) if missed_rows else len(
            [
                move
                for move in bullish_moves
                if _signal_before_move(v1_signals, move.start_bar) is None
            ],
        )

        recovered_v2 = 0
        recovered_v1 = 0
        missed_both = 0
        new_false_v2 = 0
        details: list[dict[str, Any]] = []

        if missed_rows:
            for row in missed_rows:
                move_day = str(row.get("date", ""))[:10]
                move = move_by_date.get(move_day)
                if move is None:
                    nearest = [
                        item
                        for item in bullish_moves
                        if str(frame.iloc[item.start_bar]["Date"])[:10] == move_day
                    ]
                    move = nearest[0] if nearest else None
                if move is None:
                    continue
                v1_hit = _signal_before_move(v1_signals, move.start_bar) is not None
                v2_hit = _signal_before_move(v2_signals, move.start_bar)
                if v2_hit and not v1_hit:
                    recovered_v2 += 1
                if v1_hit:
                    recovered_v1 += 1
                if not v1_hit and not v2_hit:
                    missed_both += 1
                details.append(
                    {
                        "date": move_day,
                        "move_size_points": row.get("move_size_points"),
                        "buy_v1_captured": v1_hit,
                        "buy_v2_captured": v2_hit is not None,
                        "buy_v2_entry_time": v2_hit.get("timestamp") if v2_hit else None,
                        "condition_stack_present": row.get("condition_stack_present"),
                        "buy_v1_missing_conditions": row.get("buy_v1_missing_conditions"),
                    },
                )
        else:
            for move in bullish_moves:
                v1_hit = _signal_before_move(v1_signals, move.start_bar) is not None
                v2_hit = _signal_before_move(v2_signals, move.start_bar)
                if not v1_hit and v2_hit:
                    recovered_v2 += 1
                if v1_hit:
                    recovered_v1 += 1
                if not v1_hit and not v2_hit:
                    missed_both += 1

        v2_only_signals = [
            signal
            for signal in v2_signals
            if _signal_before_move(v1_signals, signal["bar"]) is None
        ]
        for signal in v2_only_signals:
            if signal.get("classification") in {"False Reversal", "Dead Cat Bounce", "Bull Trap", "No Expansion"}:
                new_false_v2 += 1

        net_gain = recovered_v2 - new_false_v2
        v2_real = sum(1 for signal in v2_signals if signal.get("classification") == "Real Reversal")
        v1_real = sum(1 for signal in v1_signals if signal.get("classification") == "Real Reversal")
        net_quality_change = v2_real - v1_real

        return {
            "export_source": str(DEFAULT_MISSED_REVERSAL_PATH.name),
            "export_cohort_size": len(missed_rows),
            "export_dates_matched": len(export_dates),
            "cohort_size_used": cohort_size,
            "recovered_by_buy_v2": recovered_v2,
            "recovered_by_buy_v1": recovered_v1,
            "still_missed_both": missed_both,
            "new_false_reversals_buy_v2": new_false_v2,
            "net_gain_signals": net_gain,
            "net_quality_change_real_reversals": net_quality_change,
            "recovery_rate_pct": round(100.0 * recovered_v2 / max(cohort_size, 1), 2),
            "synthesis_vs_replay_note": (
                "Recovery counts are replay-validated signal-before-move matches, "
                "not synthesis-only cohort filters."
            ),
            "per_missed_reversal_details": details[:50],
            "summary": (
                f"BUY_V2 recovered {recovered_v2}/{cohort_size} missed reversals; "
                f"added {new_false_v2} new false reversals; net gain {net_gain}; "
                f"net quality change {net_quality_change} real reversals."
            ),
        }

    def _sell_v5_benchmark(self) -> dict[str, Any]:
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
            "capture_200_plus_pct": capture.get("200", {}).get("capture_rate_pct"),
        }

    def _final_verdicts(
        self,
        *,
        v1_stats: dict[str, Any],
        v2_stats: dict[str, Any],
        v1_capture: dict[str, Any],
        v2_capture: dict[str, Any],
        recovery: dict[str, Any],
        walk_forward: dict[str, Any],
        production_safety: dict[str, Any],
        sell_v5: dict[str, Any],
    ) -> dict[str, Any]:
        def _tier(pass_count: int, total: int = 5) -> str:
            if pass_count >= total:
                return "Production Candidate"
            if pass_count >= 3:
                return "Dry Run Candidate"
            return "Research Candidate"

        v2_gates = production_safety.get("buy_v2", {})
        v2_pass = sum(1 for key, value in v2_gates.items() if key != "all_pass" and value)
        v2_class = _tier(v2_pass)

        v2_wr = float(v2_stats.get("win_rate_pct") or 0.0)
        v2_pf = float(v2_stats.get("profit_factor") or 0.0)
        v5_wr = float(sell_v5.get("win_rate_pct") or 0.0)
        v5_pf = float(sell_v5.get("profit_factor") or 0.0)
        recovered = recovery.get("recovered_by_buy_v2", 0)
        cohort = recovery.get("cohort_size_used", 47)
        replay_valid = recovered > 0 or v2_stats.get("signals_emitted", 0) > 0

        sell_equiv = "NO"
        if v2_wr >= 60 and v2_pf >= 1.5 and v2_stats.get("signals_per_month", 0) >= 10:
            sell_equiv = "PARTIAL"
        if (
            v2_wr >= PRODUCTION_GATES["win_rate_min_pct"]
            and v2_pf >= PRODUCTION_GATES["profit_factor_min"]
            and v2_stats.get("signals_per_month", 0) >= PRODUCTION_GATES["signals_per_month_min"]
            and v2_gates.get("all_pass")
        ):
            sell_equiv = "YES" if v2_wr >= v5_wr * 0.95 and v2_pf >= v5_pf * 0.65 else "PARTIAL"

        oos = walk_forward.get("validate", {}).get("overall_statistics", {})
        train = walk_forward.get("train", {}).get("overall_statistics", {})
        wf_stable = (
            float(oos.get("win_rate_pct") or 0.0) >= float(train.get("win_rate_pct") or 0.0) * 0.85
            and float(oos.get("profit_factor") or 0.0) >= float(train.get("profit_factor") or 0.0) * 0.70
        )

        return {
            "replay_validated_vs_synthesis_only": {
                "answer": "REPLAY-VALID" if replay_valid else "SYNTHESIS-ONLY",
                "evidence": recovery.get("summary", ""),
            },
            "recovered_47_missed_reversals": {
                "answer": "YES" if recovered >= cohort * 0.5 else ("PARTIAL" if recovered > 0 else "NO"),
                "recovered": recovered,
                "cohort": cohort,
                "recovery_rate_pct": recovery.get("recovery_rate_pct"),
            },
            "buy_v2_classification": {
                "verdict": v2_class,
                "production_gates_passed": v2_pass,
                "walk_forward_stable": wf_stable,
            },
            "buy_v2_production_candidate": {
                "answer": "YES" if v2_gates.get("all_pass") and wf_stable else ("PARTIAL" if v2_pass >= 3 else "NO"),
                "gates": v2_gates,
            },
            "buy_v2_sell_v5_equivalent": {
                "answer": sell_equiv,
                "buy_v2_wr": v2_wr,
                "buy_v2_pf": v2_pf,
                "sell_v5_wr": v5_wr,
                "sell_v5_pf": v5_pf,
            },
            "incremental_vs_buy_v1": {
                "additional_signals": (v2_stats.get("signals_emitted") or 0) - (v1_stats.get("signals_emitted") or 0),
                "wr_delta_pp": round(v2_wr - float(v1_stats.get("win_rate_pct") or 0.0), 2),
                "pf_delta": round(v2_pf - float(v1_stats.get("profit_factor") or 0.0), 2)
                if v1_stats.get("profit_factor") and v2_stats.get("profit_factor")
                else None,
                "capture_200_delta": (v2_capture.get("200", {}).get("signals_before_move") or 0)
                - (v1_capture.get("200", {}).get("signals_before_move") or 0),
            },
        }

    def run(self, metadata: dict[str, Any]) -> BuyV2CandidateValidationReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = end - timedelta(days=TRADING_DAYS_REPLAY)

        logger.info(
            "BUY_V2 validation starting: %s trading days, %s 5M",
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

        v1_signals, v2_signals, v1_rej, v2_rej = self._replay_combined(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            replay_bars=replay_bars,
            moves=moves,
        )

        v1_stats = _build_statistics(v1_signals, trading_days=TRADING_DAYS_REPLAY)
        v2_stats = _build_statistics(v2_signals, trading_days=TRADING_DAYS_REPLAY)
        v1_capture = _bullish_point_capture(moves, v1_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        v2_capture = _bullish_point_capture(moves, v2_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)

        v1_train = _filter_signals_by_dates(v1_signals, frame, train_dates)
        v2_train = _filter_signals_by_dates(v2_signals, frame, train_dates)
        v1_validate = _filter_signals_by_dates(v1_signals, frame, validate_dates)
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
            },
        }

        missed_rows = self._load_missed_reversals()
        recovery = self._missed_reversal_recovery(
            missed_rows=missed_rows,
            v1_signals=v1_signals,
            v2_signals=v2_signals,
            moves=moves,
            frame=frame,
            replay_dates=replay_dates,
        )

        sell_v5 = self._sell_v5_benchmark()
        production_safety = {
            "buy_v1": _passes_production_gates(v1_stats, v1_capture),
            "buy_v2": _passes_production_gates(v2_stats, v2_capture),
            "gates_definition": PRODUCTION_GATES,
        }

        final_verdicts = self._final_verdicts(
            v1_stats=v1_stats,
            v2_stats=v2_stats,
            v1_capture=v1_capture,
            v2_capture=v2_capture,
            recovery=recovery,
            walk_forward=walk_forward,
            production_safety=production_safety,
            sell_v5=sell_v5,
        )

        replay_start = min(replay_dates).isoformat() if replay_dates else ""
        replay_end = max(replay_dates).isoformat() if replay_dates else ""

        return BuyV2CandidateValidationReport(
            report_type="BUY_V2 Candidate Validation",
            engines_compared=["BUY_V1", "BUY_V2"],
            buy_v1_formula=list(BUY_V1_COMPONENTS),
            buy_v2_formula=list(BUY_V2_COMPONENTS),
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
                "buy_v1_engine": "BuyV1CandidateEngine",
                "buy_v2_engine": "BuyV2CandidateEngine",
                "base_architecture": "SmartMoneyEngineV3Engine five-layer stack (bullish)",
                "event_detection": "Nifty50LiquidityDirectionDecisionMatrixResearch._detect_events_at_bar",
                "formula_lookback_bars": PRE_EXPANSION_LOOKBACK,
                "walk_forward_split": f"train {TRAIN_TRADING_DAYS} / validate {VALIDATE_TRADING_DAYS} trading days",
                "missed_reversal_export": str(DEFAULT_MISSED_REVERSAL_PATH),
                "move_detection_threshold": MOVE_DETECTION_THRESHOLD,
                "point_capture_thresholds": list(POINT_CAPTURE_THRESHOLDS),
                "no_lookahead": True,
            },
            replay_rules={
                "symbol": DEFAULT_SYMBOL,
                "timeframe": MOVE_DETECTION_TIMEFRAME,
                "trading_days": TRADING_DAYS_REPLAY,
                "candle_by_candle": True,
                "combined_replay": "BUY_V1 + BUY_V2 evaluated once per bar",
                "no_future_bos_choch_fvg": True,
            },
            comparison={
                "buy_v1": {
                    "formula_text": BUY_V1_FORMULA_TEXT,
                    "model_id": BUY_V1_MODEL_ID,
                    "overall_statistics": v1_stats,
                    "point_capture": v1_capture,
                    "layer_rejection_summary": v1_rej,
                    "signals_emitted_count": len(v1_signals),
                    "classification_summary": self._classification_summary(v1_signals),
                },
                "buy_v2": {
                    "formula_text": BUY_V2_FORMULA_TEXT,
                    "model_id": BUY_V2_MODEL_ID,
                    "overall_statistics": v2_stats,
                    "point_capture": v2_capture,
                    "layer_rejection_summary": v2_rej,
                    "signals_emitted_count": len(v2_signals),
                    "classification_summary": self._classification_summary(v2_signals),
                },
            },
            per_signal_details={
                "buy_v1": v1_signals,
                "buy_v2": v2_signals,
            },
            missed_reversal_recovery=recovery,
            failed_signal_classification={
                "buy_v1": self._classification_summary(v1_signals),
                "buy_v2": self._classification_summary(v2_signals),
            },
            condition_attribution={
                "buy_v1": self._condition_attribution(v1_signals, engine="BUY_V1"),
                "buy_v2": self._condition_attribution(v2_signals, engine="BUY_V2"),
            },
            sell_v5_benchmark=sell_v5,
            production_safety_check=production_safety,
            final_verdicts=final_verdicts,
            conclusions=[
                f"BUY_V1={len(v1_signals)} BUY_V2={len(v2_signals)} signals over {TRADING_DAYS_REPLAY} days.",
                (
                    f"WR BUY_V1 {v1_stats.get('win_rate_pct')}% vs BUY_V2 {v2_stats.get('win_rate_pct')}% | "
                    f"PF {v1_stats.get('profit_factor')} vs {v2_stats.get('profit_factor')}."
                ),
                (
                    f"200+ capture BUY_V1 {v1_capture.get('200', {}).get('capture_rate_pct')}% vs "
                    f"BUY_V2 {v2_capture.get('200', {}).get('capture_rate_pct')}%."
                ),
                recovery.get("summary", ""),
                (
                    f"Replay verdict: {final_verdicts['replay_validated_vs_synthesis_only']['answer']} | "
                    f"BUY_V2 class: {final_verdicts['buy_v2_classification']['verdict']} | "
                    f"SELL_V5 equivalent: {final_verdicts['buy_v2_sell_v5_equivalent']['answer']}."
                ),
            ],
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: BuyV2CandidateValidationReport, report_path: Path) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("BUY_V2 candidate validation exported: %s", report_path)
        return report_path


def generate_buy_v2_candidate_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> BuyV2CandidateValidationReport:
    """Run BUY_V2 vs BUY_V1 replay validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise BuyV2CandidateValidationError(f"Filter research report not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    research = BuyV2CandidateValidationResearch()
    report = research.run(metadata)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_buy_v2_candidate_validation_report()
    except BuyV2CandidateValidationError as exc:
        logger.error("BUY_V2 candidate validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected BUY_V2 candidate validation error")
        return 1

    v1 = report.comparison["buy_v1"]["overall_statistics"]
    v2 = report.comparison["buy_v2"]["overall_statistics"]
    recovery = report.missed_reversal_recovery
    verdicts = report.final_verdicts
    print("BUY_V2 Candidate Validation Summary")
    print(f"BUY_V1 signals: {v1['signals_emitted']} | BUY_V2 signals: {v2['signals_emitted']}")
    print(f"BUY_V1 WR: {v1['win_rate_pct']}% | BUY_V2 WR: {v2['win_rate_pct']}%")
    print(f"BUY_V1 PF: {v1['profit_factor']} | BUY_V2 PF: {v2['profit_factor']}")
    print(f"Recovered missed reversals: {recovery.get('recovered_by_buy_v2')}/{recovery.get('cohort_size_used')}")
    print(f"Replay verdict: {verdicts['replay_validated_vs_synthesis_only']['answer']}")
    print(f"BUY_V2 class: {verdicts['buy_v2_classification']['verdict']}")
    print(f"SELL_V5 equivalent: {verdicts['buy_v2_sell_v5_equivalent']['answer']}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
