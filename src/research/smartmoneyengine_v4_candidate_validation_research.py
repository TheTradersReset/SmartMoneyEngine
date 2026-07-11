"""
SmartMoneyEngine V4 Candidate Validation — research only.

Compares frozen V3 stack vs V4 Candidate on 120-day NIFTY50 5M replay.
V4 changes: EMA22 + EMA200 context (no EMA50 stack), optional confirmation candle.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
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

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v4_candidate_validation.json"
DEFAULT_GAP_ANALYSIS_PATH = RESEARCH_DIR / "smartmoneyengine_engine_gap_analysis.json"
DEFAULT_V31_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v31_validation.json"

TRADING_DAYS_REPLAY = 120
POINT_CAPTURE_THRESHOLDS = (50, 100, 200, 300, 500)
MAJOR_MOVE_THRESHOLDS = (100, 200, 300)

# Documented V4 EMA rule (computed from Close via ewm span=22 on enriched frame).
V4_EMA22_RULE = "Close < EMA22 AND EMA22 < EMA200"
V4_EMA_BULL_CONTEXT = "Bull Context"  # Close > EMA22 AND EMA22 > EMA200
V4_EMA_BEAR_CONTEXT = "Bear Context"  # Close < EMA22 AND EMA22 < EMA200


class SmartMoneyEngineV4CandidateValidationError(Exception):
    """Raised when V4 candidate validation fails."""


@dataclass
class SmartMoneyEngineV4CandidateValidationReport:
    """V3 vs V4 Candidate comparison validation output."""

    report_type: str
    engine_versions_compared: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    v4_change_summary: dict[str, Any]
    methodology: dict[str, Any]
    replay_rules: dict[str, Any]
    comparison: dict[str, Any]
    missed_move_recovery: dict[str, Any]
    entry_timing_delta: dict[str, Any]
    major_move_capture: dict[str, Any]
    final_questions: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _profit_factor(pnls: list[float]) -> float | None:
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    if losses == 0:
        return None if wins == 0 else round(wins, 2)
    return round(wins / losses, 2)


def _last_n_trading_day_set(frame: pd.DataFrame, n: int) -> set[date]:
    dates = pd.to_datetime(frame["Date"]).dt.date
    unique = sorted(set(dates))
    return set(unique[-n:])


def _attach_ema22(enriched: pd.DataFrame) -> pd.DataFrame:
    """Add EMA22 column to enriched frame (not in default FilterContextBuilder periods)."""
    working = enriched.copy()
    if "_ema_22" not in working.columns:
        close = working["Close"].astype(float)
        working["_ema_22"] = close.ewm(span=22, adjust=False).mean()
    return working


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _v4_ema_evaluation(enriched: pd.DataFrame, bar: int, close: float) -> dict[str, Any]:
    ema22 = _safe_float(enriched.iloc[bar].get("_ema_22"))
    ema200 = _safe_float(enriched.iloc[bar].get("_ema_200"))
    if ema22 is None or ema200 is None:
        return {
            "v4_ema_bearish": False,
            "v4_ema_structure": "Unknown",
            "ema22": ema22,
            "ema200": ema200,
            "close_vs_ema22": "Unknown",
            "rule": V4_EMA22_RULE,
        }
    if close < ema22 and ema22 < ema200:
        structure = V4_EMA_BEAR_CONTEXT
        bearish = True
    elif close > ema22 and ema22 > ema200:
        structure = V4_EMA_BULL_CONTEXT
        bearish = False
    else:
        structure = "Mixed"
        bearish = False
    return {
        "v4_ema_bearish": bearish,
        "v4_ema_structure": structure,
        "ema22": round(ema22, 2),
        "ema200": round(ema200, 2),
        "close_vs_ema22": "Below" if close < ema22 else "Above",
        "rule": V4_EMA22_RULE,
    }


class V4CandidateEngine(SmartMoneyEngineV3Engine):
    """V4 Candidate: EMA22+EMA200 context, optional confirmation candle."""

    MODEL_ID = "LDM-SELL-V4"

    def _context_at_bar(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        bar: int,
    ) -> dict[str, str]:
        context = super()._context_at_bar(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            bar=bar,
        )
        close = float(frame.iloc[bar]["Close"])
        v4_ema = _v4_ema_evaluation(enriched, bar, close)
        context["v4_ema_bearish"] = str(v4_ema["v4_ema_bearish"])
        context["v4_ema_structure"] = v4_ema["v4_ema_structure"]
        context["v4_ema22"] = str(v4_ema.get("ema22", ""))
        context["v4_ema200"] = str(v4_ema.get("ema200", ""))
        context["v4_close_vs_ema22"] = v4_ema["close_vs_ema22"]
        return context

    def _layer2_directional_filter(self, context: dict[str, str]) -> dict[str, Any]:
        ema_bearish = context.get("v4_ema_bearish") == "True"
        aligned = (
            context.get("htf_trend") == "Bearish"
            and context.get("vwap") == "Below"
            and ema_bearish
        )
        return {
            "direction": "SELL" if aligned else "NO_TRADE",
            "htf_trend": context.get("htf_trend"),
            "vwap_state": context.get("vwap"),
            "ema_structure": context.get("v4_ema_structure"),
            "v4_ema_rule": V4_EMA22_RULE,
            "v4_ema_bearish": ema_bearish,
            "aligned": aligned,
        }

    def _layer3_confirmation(self, context: dict[str, str]) -> dict[str, Any]:
        candle = context.get("confirmation_candle", "None")
        volume = context.get("volume", "Normal")
        volume_ok = volume in ALLOWED_VOLUME_BUCKETS
        return {
            "confirmation_candle": candle,
            "volume_bucket": volume,
            "confirmed": volume_ok,
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
            reasons.append("NO_EARLY_WARNING")
        if not layer1["failed_breakout_present"]:
            reasons.append("NO_FAILED_BREAKOUT")
        if layer2.get("htf_trend") == "Bullish":
            reasons.append("HTF_CONFLICT")
        if layer2.get("vwap_state") != "Below":
            reasons.append("VWAP_MISMATCH")
        if layer2.get("ema_structure") == V4_EMA_BULL_CONTEXT:
            reasons.append("EMA_MISMATCH")
        if not layer2.get("aligned"):
            reasons.append("DIRECTION_NOT_ALIGNED")
        if not layer3.get("confirmed"):
            reasons.append("VOLUME_FAILED")
        if context.get("location") == "Mid Range":
            reasons.append("LOCATION_MID_RANGE")
        if bar in emitted_bars:
            reasons.append("DUPLICATE_BAR")
        return {"pass": not reasons, "reason_codes": reasons}


def _point_capture(
    moves: list[_CheapMoveCandidate],
    signals: list[dict[str, Any]],
    replay_dates: set[date],
    frame: pd.DataFrame,
    thresholds: tuple[int, ...],
) -> dict[str, Any]:
    signal_by_bar = {signal["bar"]: signal for signal in signals}
    results: dict[str, Any] = {}
    for threshold in thresholds:
        bearish = [
            move
            for move in moves
            if move.direction == "bearish"
            and move.magnitude >= threshold
            and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
        ]
        captured = 0
        for move in bearish:
            pre_start = max(0, move.start_bar - PRE_EXPANSION_LOOKBACK)
            for bar in range(pre_start, move.start_bar + 1):
                if bar in signal_by_bar:
                    captured += 1
                    break
        total = len(bearish)
        results[str(threshold)] = {
            "total_bearish_moves": total,
            "signals_before_move": captured,
            "missed_moves": total - captured,
            "capture_rate_pct": round(captured / max(total, 1) * 100, 2),
        }
    return results


def _build_statistics(
    signals: list[dict[str, Any]],
    *,
    trading_days: int,
) -> dict[str, Any]:
    pnls = [float(s.get("realized_pnl_points") or 0.0) for s in signals]
    trading_weeks = max(trading_days / 5.0, 1.0)
    trading_months = max(trading_days / 22.0, 1.0)
    return {
        "signals_emitted": len(signals),
        "signals_per_week": round(len(signals) / trading_weeks, 2),
        "signals_per_month": round(len(signals) / trading_months, 2),
        "win_rate_pct": round(sum(1 for s in signals if s.get("win")) / max(len(signals), 1) * 100, 2),
        "profit_factor": _profit_factor(pnls),
        "expectancy": round(mean(pnls), 2) if pnls else 0.0,
        "hit_1r_rate_pct": round(sum(1 for s in signals if s.get("hit_1r")) / max(len(signals), 1) * 100, 2),
        "hit_2r_rate_pct": round(sum(1 for s in signals if s.get("hit_2r")) / max(len(signals), 1) * 100, 2),
        "hit_3r_rate_pct": round(sum(1 for s in signals if s.get("hit_3r")) / max(len(signals), 1) * 100, 2),
        "average_mfe": round(mean(float(s.get("mfe_points") or 0) for s in signals), 2) if signals else 0.0,
        "average_mae": round(mean(float(s.get("mae_points") or 0) for s in signals), 2) if signals else 0.0,
    }


def _signal_before_move(
    signals: list[dict[str, Any]],
    move_bar: int,
) -> dict[str, Any] | None:
    pre_start = max(0, move_bar - PRE_EXPANSION_LOOKBACK)
    candidates = [s for s in signals if pre_start <= s["bar"] <= move_bar]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s["bar"])


class SmartMoneyEngineV4CandidateValidationResearch:
    """Compare V3 vs V4 Candidate on 120-day NIFTY50 replay."""

    def __init__(self) -> None:
        self.v3_engine = SmartMoneyEngineV3Engine()
        self.v4_engine = V4CandidateEngine()
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=120,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    def _build_signal(
        self,
        evaluation: dict[str, Any],
        *,
        engine_version: str,
    ) -> dict[str, Any]:
        layer4 = evaluation["layer4"]
        forward = dict(layer4.get("forward_outcome") or {})
        return {
            "timestamp": evaluation["timestamp"],
            "bar": evaluation["bar"],
            "symbol": DEFAULT_SYMBOL,
            "timeframe": MOVE_DETECTION_TIMEFRAME,
            "direction": "SELL",
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
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, int]]:
        v3_signals: list[dict[str, Any]] = []
        v4_signals: list[dict[str, Any]] = []
        v3_emitted_bars: set[int] = set()
        v4_emitted_bars: set[int] = set()
        v3_rejections: dict[str, int] = {}
        v4_rejections: dict[str, int] = {}
        total = len(replay_bars)
        log_every = max(total // 20, 1)

        for index, bar in enumerate(replay_bars):
            if index > 0 and index % log_every == 0:
                logger.info("Replay progress: %s/%s bars (%.0f%%)", index, total, index / total * 100)

            if bar < PRE_EXPANSION_LOOKBACK or bar >= len(frame) - FORWARD_BARS:
                continue

            v3_eval = self.v3_engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=v3_emitted_bars,
            )
            v4_eval = self.v4_engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=v4_emitted_bars,
            )

            if v3_eval["verdict"] == "SELL":
                v3_signals.append(self._build_signal(v3_eval, engine_version="V3"))
                v3_emitted_bars.add(bar)
            else:
                for reason in v3_eval["layer5"]["reason_codes"]:
                    v3_rejections[reason] = v3_rejections.get(reason, 0) + 1

            if v4_eval["verdict"] == "SELL":
                v4_signals.append(self._build_signal(v4_eval, engine_version="V4 Candidate"))
                v4_emitted_bars.add(bar)
            else:
                for reason in v4_eval["layer5"]["reason_codes"]:
                    v4_rejections[reason] = v4_rejections.get(reason, 0) + 1

        return v3_signals, v4_signals, v3_rejections, v4_rejections

    def _missed_move_recovery(
        self,
        moves: list[_CheapMoveCandidate],
        v3_signals: list[dict[str, Any]],
        v4_signals: list[dict[str, Any]],
        replay_dates: set[date],
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        by_threshold: dict[str, Any] = {}
        recovered_details: list[dict[str, Any]] = []

        for threshold in MAJOR_MOVE_THRESHOLDS:
            bearish = [
                move
                for move in moves
                if move.direction == "bearish"
                and move.magnitude >= threshold
                and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
            ]
            v3_missed_v4_captured = 0
            both_captured = 0
            v3_only = 0
            neither = 0

            for move in bearish:
                v3_entry = _signal_before_move(v3_signals, move.start_bar)
                v4_entry = _signal_before_move(v4_signals, move.start_bar)
                v3_ok = v3_entry is not None
                v4_ok = v4_entry is not None

                if v3_ok and v4_ok:
                    both_captured += 1
                elif v4_ok and not v3_ok:
                    v3_missed_v4_captured += 1
                    recovered_details.append(
                        {
                            "threshold_points": threshold,
                            "move_start_bar": move.start_bar,
                            "move_start_time": str(frame.iloc[move.start_bar]["Date"]),
                            "move_magnitude_points": round(move.magnitude, 2),
                            "v4_entry_bar": v4_entry["bar"],
                            "v4_entry_time": v4_entry["timestamp"],
                            "v4_entry_price": v4_entry.get("entry"),
                            "v4_mfe_points": v4_entry.get("mfe_points"),
                        }
                    )
                elif v3_ok and not v4_ok:
                    v3_only += 1
                else:
                    neither += 1

            total = len(bearish)
            v3_missed = total - sum(
                1 for move in bearish if _signal_before_move(v3_signals, move.start_bar) is not None
            )
            by_threshold[str(threshold)] = {
                "total_bearish_moves": total,
                "v3_missed_moves": v3_missed,
                "v3_missed_v4_captured": v3_missed_v4_captured,
                "recovery_rate_pct": round(v3_missed_v4_captured / max(v3_missed, 1) * 100, 2),
                "both_captured": both_captured,
                "v3_only_captured": v3_only,
                "neither_captured": neither,
            }

        return {
            "by_threshold": by_threshold,
            "recovered_move_details": recovered_details[:100],
            "total_v3_missed_v4_recovered_200_plus": by_threshold.get("200", {}).get("v3_missed_v4_captured", 0),
            "summary": (
                f"V4 recovered {by_threshold.get('200', {}).get('v3_missed_v4_captured', 0)} "
                f"previously V3-missed 200+ bearish moves."
            ),
        }

    def _entry_timing_delta(
        self,
        moves: list[_CheapMoveCandidate],
        v3_signals: list[dict[str, Any]],
        v4_signals: list[dict[str, Any]],
        replay_dates: set[date],
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        bars_earlier: list[int] = []
        points_earlier: list[float] = []
        paired_comparisons: list[dict[str, Any]] = []

        for threshold in MAJOR_MOVE_THRESHOLDS:
            bearish = [
                move
                for move in moves
                if move.direction == "bearish"
                and move.magnitude >= threshold
                and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
            ]
            for move in bearish:
                v3_entry = _signal_before_move(v3_signals, move.start_bar)
                v4_entry = _signal_before_move(v4_signals, move.start_bar)
                if not v3_entry or not v4_entry:
                    continue
                delta_bars = int(v3_entry["bar"]) - int(v4_entry["bar"])
                v3_mfe = float(v3_entry.get("mfe_points") or 0.0)
                v4_mfe = float(v4_entry.get("mfe_points") or 0.0)
                delta_points = round(max(0.0, v4_mfe - v3_mfe), 2)
                bars_earlier.append(delta_bars)
                points_earlier.append(delta_points)
                if len(paired_comparisons) < 50:
                    paired_comparisons.append(
                        {
                            "threshold_points": threshold,
                            "move_start_bar": move.start_bar,
                            "move_magnitude_points": round(move.magnitude, 2),
                            "v3_entry_bar": v3_entry["bar"],
                            "v4_entry_bar": v4_entry["bar"],
                            "bars_earlier_v4_vs_v3": delta_bars,
                            "points_earlier_v4_vs_v3": delta_points,
                            "v4_enters_earlier": delta_bars > 0,
                        }
                    )

        v4_only_earlier: list[int] = []
        for detail in self._missed_move_recovery(
            moves, v3_signals, v4_signals, replay_dates, frame
        )["recovered_move_details"]:
            move_bar = detail["move_start_bar"]
            v4_bar = detail["v4_entry_bar"]
            v4_only_earlier.append(move_bar - v4_bar)

        return {
            "paired_moves_both_captured": len(bars_earlier),
            "average_bars_earlier_v4_vs_v3": round(mean(bars_earlier), 2) if bars_earlier else 0.0,
            "median_bars_earlier_v4_vs_v3": round(median(bars_earlier), 2) if bars_earlier else 0.0,
            "average_points_earlier_v4_vs_v3": round(mean(points_earlier), 2) if points_earlier else 0.0,
            "median_points_earlier_v4_vs_v3": round(median(points_earlier), 2) if points_earlier else 0.0,
            "v4_enters_earlier_count": sum(1 for b in bars_earlier if b > 0),
            "v4_enters_later_count": sum(1 for b in bars_earlier if b < 0),
            "same_bar_count": sum(1 for b in bars_earlier if b == 0),
            "v4_only_recovery_avg_bars_before_move": round(mean(v4_only_earlier), 2) if v4_only_earlier else 0.0,
            "paired_comparisons_sample": paired_comparisons,
        }

    def _final_questions(
        self,
        v3_stats: dict[str, Any],
        v4_stats: dict[str, Any],
        v3_capture: dict[str, Any],
        v4_capture: dict[str, Any],
        recovery: dict[str, Any],
        timing: dict[str, Any],
    ) -> dict[str, Any]:
        capture_improved = any(
            v4_capture.get(str(threshold), {}).get("capture_rate_pct", 0)
            > v3_capture.get(str(threshold), {}).get("capture_rate_pct", 0)
            for threshold in (100, 200, 300, 500)
        )
        pf_ok = (v4_stats.get("profit_factor") or 0) >= 2.0
        wr_ok = (v4_stats.get("win_rate_pct") or 0) >= 65.0
        superior = (
            capture_improved
            and pf_ok
            and wr_ok
            and (v4_stats.get("expectancy") or 0) >= (v3_stats.get("expectancy") or 0) * 0.85
        )

        def _answer(yes: bool, partial: bool, evidence: str) -> dict[str, Any]:
            verdict = "YES" if yes else ("PARTIAL" if partial else "NO")
            return {"answer": verdict, "evidence": evidence}

        recovered_200 = recovery.get("by_threshold", {}).get("200", {}).get("v3_missed_v4_captured", 0)

        return {
            "1_does_v4_improve_capture": _answer(
                capture_improved,
                capture_improved
                and not all(
                    v4_capture.get(str(t), {}).get("capture_rate_pct", 0)
                    >= v3_capture.get(str(t), {}).get("capture_rate_pct", 0)
                    for t in POINT_CAPTURE_THRESHOLDS
                ),
                f"200+ capture V3 {v3_capture.get('200', {}).get('capture_rate_pct')}% vs "
                f"V4 {v4_capture.get('200', {}).get('capture_rate_pct')}%; "
                f"recovered {recovered_200} V3-missed 200+ moves.",
            ),
            "2_does_pf_remain_above_2": _answer(
                pf_ok,
                (v4_stats.get("profit_factor") or 0) >= 1.8,
                f"V4 PF {v4_stats.get('profit_factor')} vs V3 {v3_stats.get('profit_factor')}.",
            ),
            "3_does_wr_remain_above_65": _answer(
                wr_ok,
                (v4_stats.get("win_rate_pct") or 0) >= 60.0,
                f"V4 WR {v4_stats.get('win_rate_pct')}% vs V3 {v3_stats.get('win_rate_pct')}%.",
            ),
            "4_is_v4_superior_to_v3": _answer(
                superior,
                capture_improved or pf_ok,
                f"Expectancy V4 {v4_stats.get('expectancy')} vs V3 {v3_stats.get('expectancy')}; "
                f"avg bars earlier {timing.get('average_bars_earlier_v4_vs_v3')}; "
                f"signals/month V4 {v4_stats.get('signals_per_month')} vs V3 {v3_stats.get('signals_per_month')}.",
            ),
        }

    def run(self, metadata: dict[str, Any]) -> SmartMoneyEngineV4CandidateValidationReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = end - timedelta(days=120)

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=120,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        replay_dates = _last_n_trading_day_set(frame, TRADING_DAYS_REPLAY)
        bar_dates = pd.to_datetime(frame["Date"]).dt.date
        replay_bars = [index for index, day in enumerate(bar_dates) if day in replay_dates]

        enriched = _attach_ema22(self.v3_engine.context_builder.enrich(frame))
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        intel_frames: dict[str, pd.DataFrame] = {"5M": self.v3_engine.intelligence.enrich(frame)}
        for tf in ("15M", "1H"):
            tf_path = filter_engine._ensure_pipeline(tf, start, end)
            tf_frame = pd.read_csv(tf_path).reset_index(drop=True)
            intel_frames[tf] = self.v3_engine.intelligence.enrich(tf_frame)
        intel_frames["1D"] = self.v3_engine.intelligence.enrich(
            self.v3_engine._resample_daily(intel_frames["1H"]),
        )

        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, 50),
        )

        v3_signals, v4_signals, v3_rejections, v4_rejections = self._replay_combined(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            replay_bars=replay_bars,
        )

        v3_stats = _build_statistics(v3_signals, trading_days=TRADING_DAYS_REPLAY)
        v4_stats = _build_statistics(v4_signals, trading_days=TRADING_DAYS_REPLAY)
        v3_capture = _point_capture(moves, v3_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        v4_capture = _point_capture(moves, v4_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        recovery = self._missed_move_recovery(moves, v3_signals, v4_signals, replay_dates, frame)
        timing = self._entry_timing_delta(moves, v3_signals, v4_signals, replay_dates, frame)

        replay_start = min(replay_dates).isoformat() if replay_dates else ""
        replay_end = max(replay_dates).isoformat() if replay_dates else ""

        final_questions = self._final_questions(
            v3_stats, v4_stats, v3_capture, v4_capture, recovery, timing
        )

        reference_context: dict[str, Any] = {}
        if DEFAULT_V31_REPORT_PATH.exists():
            reference_context["v31_baseline"] = {
                "source": str(DEFAULT_V31_REPORT_PATH.name),
                "v3_120d_signals": json.loads(DEFAULT_V31_REPORT_PATH.read_text(encoding="utf-8"))
                .get("comparison", {})
                .get("v3", {})
                .get("signals_emitted_count"),
            }
        if DEFAULT_GAP_ANALYSIS_PATH.exists():
            gap = json.loads(DEFAULT_GAP_ANALYSIS_PATH.read_text(encoding="utf-8"))
            reference_context["gap_analysis"] = {
                "source": str(DEFAULT_GAP_ANALYSIS_PATH.name),
                "v3_120d_missed_200_plus": gap.get("capture_baseline", {})
                .get("v3_120d", {})
                .get("200_plus", {})
                .get("missed_moves"),
            }

        return SmartMoneyEngineV4CandidateValidationReport(
            report_type="SmartMoneyEngine V4 Candidate Validation",
            engine_versions_compared=["SmartMoneyEngine V3", "SmartMoneyEngine V4 Candidate"],
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            trading_days_replayed=TRADING_DAYS_REPLAY,
            replay_start_date=replay_start,
            replay_end_date=replay_end,
            v4_change_summary={
                "unchanged": [
                    "Failed Breakout",
                    "HTF Bearish",
                    "VWAP Below",
                    "Location Filters",
                    "Volume bucket gate",
                ],
                "modified": {
                    "ema_structure": {
                        "v3": "EMA20 < EMA50 < EMA200 Bear Stack",
                        "v4": V4_EMA22_RULE,
                        "ema50_removed": True,
                        "ema200_context_only": True,
                    },
                    "confirmation": {
                        "v3": "Confirmation candle required (CONFIRMATION_FAILED hard gate)",
                        "v4": "Confirmation candle optional; volume bucket still required",
                    },
                },
            },
            methodology={
                "research_only": True,
                "single_pass_replay": True,
                "v3_engine": "SmartMoneyEngineV3Engine (frozen)",
                "v4_engine": "V4CandidateEngine",
                "ema22_computation": "close.ewm(span=22, adjust=False).mean() on enriched frame",
                "ema22_bearish_rule": V4_EMA22_RULE,
                "ema_bull_reject_rule": f"Close > EMA22 AND EMA22 > EMA200 ({V4_EMA_BULL_CONTEXT})",
                "ema_mixed_behavior": "Mixed/Unknown EMA context fails directional alignment (not bearish)",
                "confirmation_optional_rule": "Layer3 passes without confirmation candle; VOLUME_FAILED still blocks",
                "missed_move_recovery_method": (
                    "Bearish move >= threshold in replay window; V3 missed = no signal within "
                    f"PRE_EXPANSION_LOOKBACK ({PRE_EXPANSION_LOOKBACK}) bars before move start; "
                    "V4 captured = signal present in same window when V3 absent"
                ),
                "entry_timing_method": (
                    "For moves captured by both engines, delta = V3 entry bar - V4 entry bar "
                    "(positive = V4 earlier); points delta uses MFE difference"
                ),
                "reference_exports": reference_context,
            },
            replay_rules={
                "symbol": DEFAULT_SYMBOL,
                "timeframe": MOVE_DETECTION_TIMEFRAME,
                "trading_days": TRADING_DAYS_REPLAY,
                "candle_by_candle": True,
                "no_look_ahead": True,
                "research_only": True,
            },
            comparison={
                "v3": {
                    "overall_statistics": v3_stats,
                    "point_capture": v3_capture,
                    "layer_rejection_summary": v3_rejections,
                    "signals_emitted_count": len(v3_signals),
                },
                "v4_candidate": {
                    "overall_statistics": v4_stats,
                    "point_capture": v4_capture,
                    "layer_rejection_summary": v4_rejections,
                    "signals_emitted_count": len(v4_signals),
                },
                "delta_v4_minus_v3": {
                    "signals_emitted": len(v4_signals) - len(v3_signals),
                    "win_rate_pp": round(
                        (v4_stats.get("win_rate_pct") or 0) - (v3_stats.get("win_rate_pct") or 0),
                        2,
                    ),
                    "profit_factor": round(
                        (v4_stats.get("profit_factor") or 0) - (v3_stats.get("profit_factor") or 0),
                        2,
                    )
                    if v4_stats.get("profit_factor") and v3_stats.get("profit_factor")
                    else None,
                    "expectancy": round(
                        (v4_stats.get("expectancy") or 0) - (v3_stats.get("expectancy") or 0),
                        2,
                    ),
                    "signals_per_month": round(
                        (v4_stats.get("signals_per_month") or 0) - (v3_stats.get("signals_per_month") or 0),
                        2,
                    ),
                    "200_plus_capture_pp": round(
                        v4_capture.get("200", {}).get("capture_rate_pct", 0)
                        - v3_capture.get("200", {}).get("capture_rate_pct", 0),
                        2,
                    ),
                },
            },
            missed_move_recovery=recovery,
            entry_timing_delta=timing,
            major_move_capture={
                "v3": {k: v3_capture[k] for k in ("50", "100", "200", "300", "500")},
                "v4_candidate": {k: v4_capture[k] for k in ("50", "100", "200", "300", "500")},
            },
            final_questions=final_questions,
            conclusions=[
                f"V3 emitted {len(v3_signals)} signals vs V4 {len(v4_signals)} over {TRADING_DAYS_REPLAY} trading days.",
                f"Win rate V3 {v3_stats.get('win_rate_pct')}% vs V4 {v4_stats.get('win_rate_pct')}%.",
                f"PF V3 {v3_stats.get('profit_factor')} vs V4 {v4_stats.get('profit_factor')}.",
                f"200+ capture V3 {v3_capture.get('200', {}).get('capture_rate_pct')}% vs "
                f"V4 {v4_capture.get('200', {}).get('capture_rate_pct')}%.",
                f"V4 recovered {recovery.get('total_v3_missed_v4_recovered_200_plus', 0)} V3-missed 200+ moves.",
                f"Avg entry timing delta (V4 vs V3): {timing.get('average_bars_earlier_v4_vs_v3')} bars / "
                f"{timing.get('average_points_earlier_v4_vs_v3')} points.",
                f"Final: V4 superior to V3 — {final_questions['4_is_v4_superior_to_v3']['answer']}.",
            ],
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: SmartMoneyEngineV4CandidateValidationReport, report_path: Path) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("V4 candidate validation exported: %s", report_path)
        return report_path


def generate_smartmoneyengine_v4_candidate_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> SmartMoneyEngineV4CandidateValidationReport:
    """Run V3 vs V4 Candidate validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise SmartMoneyEngineV4CandidateValidationError(
            f"Filter research report not found: {metadata_path}",
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    research = SmartMoneyEngineV4CandidateValidationResearch()
    report = research.run(metadata)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_smartmoneyengine_v4_candidate_validation_report()
    except SmartMoneyEngineV4CandidateValidationError as exc:
        logger.error("V4 candidate validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected V4 candidate validation error")
        return 1

    v3 = report.comparison["v3"]["overall_statistics"]
    v4 = report.comparison["v4_candidate"]["overall_statistics"]
    print("SmartMoneyEngine V4 Candidate Validation Summary")
    print(f"V3 signals: {v3['signals_emitted']} | V4 signals: {v4['signals_emitted']}")
    print(f"V3 WR: {v3['win_rate_pct']}% | V4 WR: {v4['win_rate_pct']}%")
    print(f"V3 PF: {v3['profit_factor']} | V4 PF: {v4['profit_factor']}")
    for key in report.final_questions:
        print(f"{key}: {report.final_questions[key]['answer']}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
