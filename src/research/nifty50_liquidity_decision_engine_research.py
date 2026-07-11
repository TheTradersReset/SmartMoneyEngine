"""
NIFTY50 Liquidity Decision Engine research for SmartMoneyEngine.

Discovers conditions that convert liquidity events into BUY, SELL, or NO TRADE
and identifies earliest reliable warnings before momentum expansion.
Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine
from src.research.filter_research_engine import FilterContextBuilder, FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import (
    LEVEL_CLUSTER_POINTS,
    LEVEL_TOUCH_ATR_RATIO,
    PRE_EXPANSION_LOOKBACK,
    VOLUME_LOOKBACK,
    InstitutionalExpansionTriggerDiscoveryResearch,
)
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    TIMEFRAME_MINUTES,
    _CheapMoveCandidate,
)
from src.research.nifty50_trap_to_momentum_validation_research import (
    DEFAULT_SYMBOL,
    MOVE_THRESHOLDS,
    RESEARCH_WINDOW_DAYS,
    TRAP_EVENTS,
    Nifty50TrapToMomentumValidationResearch,
)
from src.research.smartmoneyengine_reality_check_validation_research import (
    SmartMoneyEngineRealityCheckValidationResearch,
)
from src.research.tiered_signal_framework_research import TieredSignalFrameworkResearch
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "nifty50_liquidity_decision_engine.json"

MOVE_DETECTION_TIMEFRAME = "5M"
CONTEXT_TIMEFRAMES = ("15M", "1H", "1D")
PIPELINE_TIMEFRAMES = ("5M", "15M", "1H")
LIQUIDITY_EVENTS = TRAP_EVENTS
EARLY_WARNING_EVENTS = (
    "Liquidity Event",
    "Gap Event",
    "Failed Break",
    "CHOCH",
    "BOS",
    "FVG",
    "Order Block",
    "Volume Spike",
)
MIN_MATRIX_SAMPLES = 50
MAX_EVENT_EXPORT = 2000
MAX_COMBO_FEATURES = 4
STRUCTURE_LOOKAHEAD = 30
MOMENTUM_THRESHOLDS = (50, 100, 200, 300, 500)


class Nifty50LiquidityDecisionEngineError(Exception):
    """Raised when liquidity decision engine research fails."""


@dataclass
class Nifty50LiquidityDecisionEngineReport:
    """Full liquidity decision engine research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    primary_timeframe: str
    context_timeframes: list[str]
    liquidity_events_detected: int
    liquidity_event_log: list[dict[str, Any]]
    outcome_summary: dict[str, Any]
    decision_matrix: dict[str, Any]
    earliest_warning_analysis: dict[str, Any]
    engine_comparison_200_plus: list[dict[str, Any]]
    final_questions: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Nifty50LiquidityDecisionEngineResearch(Nifty50TrapToMomentumValidationResearch):
    """Discover BUY/SELL/NO TRADE conditions from NIFTY50 liquidity events."""

    def __init__(self) -> None:
        super().__init__()
        self.context_builder = FilterContextBuilder()
        self.reality_engine = SmartMoneyEngineRealityCheckValidationResearch(
            symbols=(DEFAULT_SYMBOL,),
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        self.trade_engine = TradeConstructionValidationResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    @staticmethod
    def _resample_daily(frame_1h: pd.DataFrame) -> pd.DataFrame:
        working = frame_1h.copy()
        working["Date"] = pd.to_datetime(working["Date"])
        working = working.set_index("Date")
        daily = working.resample("1D").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"},
        )
        daily = daily.dropna(subset=["Open", "Close"]).reset_index()
        daily["Date"] = daily["Date"].astype(str)
        return daily

    @staticmethod
    def _distance_bucket(distance: float | None) -> str:
        if distance is None:
            return "50+"
        if distance <= 10:
            return "0-10"
        if distance <= 25:
            return "10-25"
        if distance <= 50:
            return "25-50"
        return "50+"

    @staticmethod
    def _test_bucket(tests: int) -> str:
        if tests <= 1:
            return "1 test"
        if tests == 2:
            return "2 tests"
        if tests == 3:
            return "3 tests"
        return "4+ tests"

    @staticmethod
    def _probability(values: list[bool]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values) * 100, 2)

    def _level_swept(
        self,
        event: str,
        row: pd.Series,
        cal_row: pd.Series,
        levels: dict[str, Any],
        close: float,
    ) -> tuple[str | None, str]:
        if event in {"Liquidity Grab", "Stop Hunt"}:
            if self.discovery._is_active(row.get("Sell_Liquidity_Sweep")):
                return self.discovery._to_float(row.get("Sell_Side_Liquidity")), "bearish"
            if self.discovery._is_active(row.get("Buy_Liquidity_Sweep")):
                return self.discovery._to_float(row.get("Buy_Side_Liquidity")), "bullish"
        mapping = {
            "PDH Sweep": ("_pdh", "bearish"),
            "PDL Sweep": ("_pdl", "bullish"),
            "PWH Sweep": ("_pwh", "bearish"),
            "PWL Sweep": ("_pwl", "bullish"),
            "Equal High Sweep": ("Equal_High", "bearish"),
            "Equal Low Sweep": ("Equal_Low", "bullish"),
            "Failed Breakout": ("major_resistance", "bearish"),
            "Failed Breakdown": ("major_support", "bullish"),
        }
        if event == "Round Number Sweep":
            level = self._round_number_level(close)
            return level, "neutral"
        if event in {"Gap Reversal", "Gap Continuation"}:
            return None, "bullish" if float(row["Close"]) >= float(row["Open"]) else "bearish"
        if event in mapping:
            key, direction = mapping[event]
            if key.startswith("_"):
                return self.discovery._to_float(cal_row.get(key)), direction
            return levels.get(key), direction
        return None, "neutral"

    def _major_level_context(
        self,
        frame: pd.DataFrame,
        calendar: pd.DataFrame,
        bar: int,
    ) -> dict[str, Any]:
        levels = self.discovery._market_levels(frame, bar)
        cal_row = calendar.iloc[bar]
        close = float(frame.iloc[bar]["Close"])
        row = frame.iloc[bar]

        def dist(value: float | None) -> float | None:
            return round(abs(close - value), 2) if value is not None else None

        demand = self.discovery._to_float(row.get("Bullish_OB_Low"))
        supply = self.discovery._to_float(row.get("Bearish_OB_High"))
        round_level = self._round_number_level(close)

        context = {
            "nearest_support": levels.get("major_support"),
            "nearest_resistance": levels.get("major_resistance"),
            "nearest_demand_zone": demand,
            "nearest_supply_zone": supply,
            "pdh": self.discovery._to_float(cal_row.get("_pdh")),
            "pdl": self.discovery._to_float(cal_row.get("_pdl")),
            "pwh": self.discovery._to_float(cal_row.get("_pwh")),
            "pwl": self.discovery._to_float(cal_row.get("_pwl")),
            "monthly_high": self.discovery._to_float(cal_row.get("_pmh")),
            "monthly_low": self.discovery._to_float(cal_row.get("_pml")),
            "round_number": round_level,
        }
        distances = {f"distance_{key}": dist(value) for key, value in context.items()}
        buckets = {
            f"bucket_{key}": self._distance_bucket(distances[f"distance_{key}"])
            for key in context
        }
        return {**context, **distances, **buckets}

    def _level_pressure_before(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel: pd.DataFrame,
        bar: int,
        direction: str,
    ) -> dict[str, Any]:
        start = max(0, bar - PRE_EXPANSION_LOOKBACK)
        measurements = self.discovery._combined_pre_expansion_measurements(
            frame,
            enriched,
            calendar,
            intel,
            start,
            max(start, bar - 1),
            bar,
            direction,
        )
        sr = measurements["support_resistance"]
        liquidity = measurements["liquidity"]
        absorption = measurements["absorption"]
        tests = int(sr.get("number_of_tests", 0))
        return {
            "number_of_tests": tests,
            "test_bucket": self._test_bucket(tests),
            "number_of_retests": max(tests - 1, 0),
            "failed_breakouts": int(sr.get("failed_breakout_count", 0)),
            "failed_breakdowns": int(sr.get("failed_breakdown_count", 0)),
            "time_spent_near_level_bars": int(sr.get("bars_near_level", 0)),
            "average_rejection_size": float(absorption.get("average_wick_size_points", 0.0)),
            "maximum_rejection_size": float(absorption.get("maximum_wick_size_points", 0.0)),
            "liquidity_taken_before_break": int(liquidity.get("liquidity_grab_count", 0)),
        }

    def _confirmation_analysis(
        self,
        frame: pd.DataFrame,
        bar: int,
        direction: str,
    ) -> dict[str, Any]:
        end = min(len(frame) - 1, bar + 3)
        window = frame.iloc[bar : end + 1]
        candles: list[dict[str, Any]] = []
        patterns: list[str] = []

        for offset in range(len(window)):
            index = bar + offset
            trigger = self.discovery._measure_expansion_trigger(frame, index, direction)
            parts = self.discovery._candle_parts(frame.iloc[index])
            label = "event_candle" if offset == 0 else f"follow_candle_{offset}"
            candles.append(
                {
                    "label": label,
                    "body_size": round(parts["body"], 2),
                    "upper_wick": round(parts["upper_wick"], 2),
                    "lower_wick": round(parts["lower_wick"], 2),
                    "body_pct": round(parts["body_pct"] * 100, 2),
                    "wick_pct": round(parts["wick_pct"] * 100, 2),
                    "close_location_pct": round(parts["close_location_pct"] * 100, 2),
                    "volume_expansion": trigger.get("volume_expansion_ratio", 1.0),
                    "atr_expansion": trigger.get("atr_expansion_ratio", 1.0),
                },
            )
            if trigger.get("hammer"):
                patterns.append("Hammer")
            if trigger.get("shooting_star"):
                patterns.append("Shooting Star")
            if trigger.get("engulfing") and direction == "bullish":
                patterns.append("Bullish Engulfing")
            if trigger.get("engulfing") and direction == "bearish":
                patterns.append("Bearish Engulfing")
            if trigger.get("bullish_harami"):
                patterns.append("Bullish Harami")
            if trigger.get("bearish_harami"):
                patterns.append("Bearish Harami")
            if trigger.get("morning_star"):
                patterns.append("Morning Star")
            if trigger.get("evening_star"):
                patterns.append("Evening Star")
            if trigger.get("marubozu") and parts["bullish"]:
                patterns.append("Bullish Marubozu")
            if trigger.get("marubozu") and parts["bearish"]:
                patterns.append("Bearish Marubozu")

        if bar >= 1:
            prev = self.discovery._candle_parts(frame.iloc[bar - 1])
            curr = self.discovery._candle_parts(frame.iloc[bar])
            if curr["high"] <= prev["high"] and curr["low"] >= prev["low"]:
                patterns.append("Inside Bar")
            if curr["high"] >= prev["high"] and curr["low"] <= prev["low"]:
                patterns.append("Outside Bar")

        ranges = [
            float(frame.iloc[max(0, index - 6) : index + 1]["High"].astype(float).max())
            - float(frame.iloc[max(0, index - 6) : index + 1]["Low"].astype(float).min())
            for index in range(bar, end + 1)
        ]
        if ranges and ranges[-1] <= min(ranges):
            patterns.append("NR7")

        return {
            "candles": candles,
            "patterns": sorted(set(patterns)),
            "primary_pattern": patterns[0] if patterns else "None",
        }

    def _structure_after(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel: pd.DataFrame,
        bar: int,
        direction: str,
    ) -> dict[str, Any]:
        end = min(len(frame) - 1, bar + STRUCTURE_LOOKAHEAD)
        structure = self.discovery._measure_structure(
            frame,
            enriched,
            intel,
            bar,
            end,
            direction,
        )
        filters = self.context_builder.filter_state(enriched, end)
        row = frame.iloc[end]
        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)
        volume = self.discovery._to_float(row.get("Volume")) or 0.0
        vol_start = max(0, end - VOLUME_LOOKBACK)
        avg_volume = mean(
            self.discovery._to_float(frame.iloc[offset].get("Volume")) or 0.0
            for offset in range(vol_start, end)
        ) if end > vol_start else volume
        return {
            "choch": structure.get("choch_count", 0) > 0,
            "bos": structure.get("bos_count", 0) > 0,
            "fvg_creation": structure.get("fvg_count", 0) > 0,
            "fvg_reclaim": structure.get("fvg_count", 0) > 0,
            "order_block_creation": structure.get("ob_count", 0) > 0,
            "order_block_reclaim": structure.get("ob_count", 0) > 0,
            "displacement_strength": displacement.value,
            "vwap_reclaim": filters.vwap_position in {"Above VWAP", "Below VWAP"},
            "ema_alignment": filters.ema_alignment,
            "rsi_bucket": filters.rsi_band,
            "volume_spike": volume >= avg_volume * 1.5 if avg_volume > 0 else False,
        }

    @staticmethod
    def _forward_directional_moves(
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        bar: int,
        forward_bars: int,
    ) -> tuple[float, float]:
        end = min(len(highs) - 1, bar + forward_bars)
        base = float(closes.iloc[bar])
        bull = float(highs.iloc[bar : end + 1].max()) - base
        bear = base - float(lows.iloc[bar : end + 1].min())
        return max(bull, 0.0), max(bear, 0.0)

    def _classify_outcome(self, bull: float, bear: float, event_direction: str, event: str) -> str:
        if max(bull, bear) < 50:
            return "No Expansion"
        reversal_events = {
            "Failed Breakdown",
            "Failed Breakout",
            "Equal Low Sweep",
            "Equal High Sweep",
            "PDL Sweep",
            "PDH Sweep",
            "Gap Reversal",
        }
        if bull > bear:
            return "Bullish Reversal" if event in reversal_events or event_direction == "bullish" else "Bullish Continuation"
        return "Bearish Reversal" if event in reversal_events or event_direction == "bearish" else "Bearish Continuation"

    @staticmethod
    def _assign_decision(outcome: str) -> str:
        if outcome in {"Bullish Reversal", "Bullish Continuation"}:
            return "BUY"
        if outcome in {"Bearish Reversal", "Bearish Continuation"}:
            return "SELL"
        return "NO TRADE"

    def _forward_trade_metrics(
        self,
        frame: pd.DataFrame,
        bar: int,
        direction: str,
    ) -> dict[str, Any]:
        side = "BUY" if direction == "bullish" else "SELL"
        entry = round(float(frame.iloc[bar]["Close"]), 2)
        stop, risk = self.trade_engine._structural_stop(frame, bar, entry, direction)
        if risk <= 0:
            return {}
        end = min(len(frame) - 1, bar + FORWARD_BARS)
        highs = frame.iloc[bar + 1 : end + 1]["High"].astype(float)
        lows = frame.iloc[bar + 1 : end + 1]["Low"].astype(float)
        if highs.empty:
            return {}
        if direction == "bullish":
            mfe = float(highs.max()) - entry
            mae = entry - float(lows.min())
            hit_1r = mfe >= risk
            hit_2r = mfe >= 2 * risk
            hit_3r = mfe >= 3 * risk
            realized = float(frame.iloc[end]["Close"]) - entry
        else:
            mfe = entry - float(lows.min())
            mae = float(highs.max()) - entry
            hit_1r = mfe >= risk
            hit_2r = mfe >= 2 * risk
            hit_3r = mfe >= 3 * risk
            realized = entry - float(frame.iloc[end]["Close"])
        return {
            "mfe_points": round(mfe, 2),
            "mae_points": round(mae, 2),
            "max_drawdown_points": round(mae, 2),
            "hit_1r": bool(hit_1r),
            "hit_2r": bool(hit_2r),
            "hit_3r": bool(hit_3r),
            "realized_pnl_points": round(realized, 2),
            "risk_points": round(risk, 2),
        }

    def _real_vs_fake(
        self,
        frame: pd.DataFrame,
        bar: int,
        levels: dict[str, Any],
    ) -> dict[str, Any]:
        row = frame.iloc[bar]
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        open_price = float(row["Open"])
        volume = float(row.get("Volume", 0.0))
        atr = self.discovery._atr(frame, bar)
        displacement = round(abs(close - open_price) / max(atr, 0.01), 2)
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        classification = "None"
        follow_through = False
        retracement = 0.0
        failure_bars = -1

        if resistance is not None and high > resistance:
            follow_through = close > resistance
            retracement = round(max(resistance - close, 0.0), 2)
            classification = "Real Breakout" if follow_through else "Fake Breakout"
        elif support is not None and low < support:
            follow_through = close < support
            retracement = round(max(close - support, 0.0), 2)
            classification = "Real Breakdown" if follow_through else "Fake Breakdown"

        if classification.startswith("Fake"):
            end = min(len(frame) - 1, bar + 20)
            for index in range(bar + 1, end + 1):
                future = frame.iloc[index]
                if classification == "Fake Breakout" and resistance is not None:
                    if float(future["Close"]) < resistance:
                        failure_bars = index - bar
                        break
                if classification == "Fake Breakdown" and support is not None:
                    if float(future["Close"]) > support:
                        failure_bars = index - bar
                        break

        return {
            "classification": classification,
            "volume": volume,
            "displacement": displacement,
            "follow_through": follow_through,
            "retracement": retracement,
            "time_until_failure_bars": failure_bars,
        }

    def _matrix_key(self, record: dict[str, Any]) -> str:
        parts = [
            record["event_type"],
            record["level_pressure"]["test_bucket"],
            record["confirmation"]["primary_pattern"],
            record["major_level_context"].get("bucket_nearest_support", "50+"),
        ]
        if record["level_pressure"]["failed_breakdowns"] > 0:
            parts.append("Failed Breakdown")
        if record["level_pressure"]["failed_breakouts"] > 0:
            parts.append("Failed Breakout")
        if record["structure_after"].get("volume_spike"):
            parts.append("Volume Expansion")
        return " + ".join(parts)

    def _build_decision_matrix(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for record in records:
            buckets[self._matrix_key(record)].append(record["decision"])

        rows: list[dict[str, Any]] = []
        for combo, decisions in buckets.items():
            if len(decisions) < MIN_MATRIX_SAMPLES:
                continue
            buy = sum(1 for item in decisions if item == "BUY")
            sell = sum(1 for item in decisions if item == "SELL")
            no_trade = sum(1 for item in decisions if item == "NO TRADE")
            total = len(decisions)
            rows.append(
                {
                    "combination": combo,
                    "sample_size": total,
                    "buy_probability_pct": round(buy / total * 100, 2),
                    "sell_probability_pct": round(sell / total * 100, 2),
                    "no_trade_probability_pct": round(no_trade / total * 100, 2),
                },
            )

        return {
            "minimum_sample_size": MIN_MATRIX_SAMPLES,
            "top_50_buy_combinations": sorted(
                rows,
                key=lambda row: (row["buy_probability_pct"], row["sample_size"]),
                reverse=True,
            )[:50],
            "top_50_sell_combinations": sorted(
                rows,
                key=lambda row: (row["sell_probability_pct"], row["sample_size"]),
                reverse=True,
            )[:50],
            "top_50_no_trade_combinations": sorted(
                rows,
                key=lambda row: (row["no_trade_probability_pct"], row["sample_size"]),
                reverse=True,
            )[:50],
        }

    def _aggregate_outcomes(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record["outcome"]].append(record)

        summary: dict[str, Any] = {}
        for outcome, items in grouped.items():
            magnitudes = [float(item["forward_metrics"]["max_move"]) for item in items]
            drawdowns = [float(item["forward_metrics"]["max_drawdown"]) for item in items]
            times = [float(item["forward_metrics"]["time_to_expansion_bars"]) for item in items if item["forward_metrics"]["time_to_expansion_bars"] >= 0]
            mfes = [float(item["trade_metrics"].get("mfe_points", 0.0)) for item in items if item.get("trade_metrics")]
            maes = [float(item["trade_metrics"].get("mae_points", 0.0)) for item in items if item.get("trade_metrics")]
            summary[outcome] = {
                "sample_size": len(items),
                "probability_50_plus_pct": self._probability([item["forward_metrics"]["max_move"] >= 50 for item in items]),
                "probability_100_plus_pct": self._probability([item["forward_metrics"]["max_move"] >= 100 for item in items]),
                "probability_200_plus_pct": self._probability([item["forward_metrics"]["max_move"] >= 200 for item in items]),
                "probability_300_plus_pct": self._probability([item["forward_metrics"]["max_move"] >= 300 for item in items]),
                "probability_500_plus_pct": self._probability([item["forward_metrics"]["max_move"] >= 500 for item in items]),
                "average_move_size": round(mean(magnitudes), 2) if magnitudes else 0.0,
                "maximum_move_size": round(max(magnitudes), 2) if magnitudes else 0.0,
                "average_drawdown": round(mean(drawdowns), 2) if drawdowns else 0.0,
                "maximum_drawdown": round(max(drawdowns), 2) if drawdowns else 0.0,
                "average_time_to_expansion_bars": round(mean(times), 2) if times else 0.0,
                "average_mfe": round(mean(mfes), 2) if mfes else 0.0,
                "average_mae": round(mean(maes), 2) if maes else 0.0,
                "hit_1r_rate_pct": self._probability([item["trade_metrics"].get("hit_1r") for item in items if item.get("trade_metrics")]),
                "hit_2r_rate_pct": self._probability([item["trade_metrics"].get("hit_2r") for item in items if item.get("trade_metrics")]),
                "hit_3r_rate_pct": self._probability([item["trade_metrics"].get("hit_3r") for item in items if item.get("trade_metrics")]),
            }
        return summary

    def _earliest_warning_analysis(
        self,
        moves: list[_CheapMoveCandidate],
        frame: pd.DataFrame,
        calendar: pd.DataFrame,
    ) -> dict[str, Any]:
        warning_buckets: dict[str, list[int]] = defaultdict(list)
        for move in moves:
            if move.magnitude < 200:
                continue
            pre_start = max(PRE_EXPANSION_LOOKBACK, move.start_bar - PRE_EXPANSION_LOOKBACK)
            first_seen: dict[str, int] = {}
            for bar in range(pre_start, move.start_bar + 1):
                events = self._detect_events_at_bar(frame, calendar, bar)
                liquidity = [event for event in events if event in LIQUIDITY_EVENTS]
                if liquidity and "Liquidity Event" not in first_seen:
                    first_seen["Liquidity Event"] = bar
                if any(event.startswith("Gap") for event in events) and "Gap Event" not in first_seen:
                    first_seen["Gap Event"] = bar
                if any(event.startswith("Failed") for event in events) and "Failed Break" not in first_seen:
                    first_seen["Failed Break"] = bar
                row = frame.iloc[bar]
                if self.discovery._is_active(row.get("Bullish_CHOCH")) or self.discovery._is_active(row.get("Bearish_CHOCH")):
                    first_seen.setdefault("CHOCH", bar)
                if self.discovery._is_active(row.get("Bullish_BOS")) or self.discovery._is_active(row.get("Bearish_BOS")):
                    first_seen.setdefault("BOS", bar)
                if self.discovery._is_active(row.get("Bullish_FVG_Top")) or self.discovery._is_active(row.get("Bearish_FVG_Top")):
                    first_seen.setdefault("FVG", bar)
                if self.discovery._is_active(row.get("Bullish_OB_High")) or self.discovery._is_active(row.get("Bearish_OB_High")):
                    first_seen.setdefault("Order Block", bar)
                volume = self.discovery._to_float(row.get("Volume")) or 0.0
                vol_start = max(0, bar - VOLUME_LOOKBACK)
                avg_volume = mean(
                    self.discovery._to_float(frame.iloc[offset].get("Volume")) or 0.0
                    for offset in range(vol_start, bar)
                ) if bar > vol_start else volume
                if volume >= avg_volume * 1.5 and avg_volume > 0:
                    first_seen.setdefault("Volume Spike", bar)

            for label, event_bar in first_seen.items():
                warning_buckets[label].append(move.start_bar - event_bar)

        ranked = []
        for label in EARLY_WARNING_EVENTS:
            values = warning_buckets.get(label, [])
            if not values:
                continue
            ranked.append(
                {
                    "warning_type": label,
                    "sample_size": len(values),
                    "average_bars_before_move": round(mean(values), 2),
                    "median_bars_before_move": round(median(values), 2),
                    "minimum_bars_before_move": min(values),
                },
            )
        ranked.sort(key=lambda row: row["average_bars_before_move"])

        major_500 = [move for move in moves if move.magnitude >= 500]
        warning_500: dict[str, list[int]] = defaultdict(list)
        for move in major_500:
            pre_start = max(PRE_EXPANSION_LOOKBACK, move.start_bar - PRE_EXPANSION_LOOKBACK)
            for bar in range(pre_start, move.start_bar + 1):
                events = self._detect_events_at_bar(frame, calendar, bar)
                if any(event in LIQUIDITY_EVENTS for event in events):
                    warning_500["Liquidity Event"].append(move.start_bar - bar)
                    break

        return {
            "ranked_warnings_200_plus": ranked,
            "earliest_reliable_warning_200_plus": ranked[0] if ranked else None,
            "earliest_reliable_warning_500_plus": {
                "warning_type": "Liquidity Event",
                "average_bars_before_move": round(mean(warning_500["Liquidity Event"]), 2)
                if warning_500["Liquidity Event"]
                else 0.0,
                "sample_size": len(warning_500["Liquidity Event"]),
            },
        }

    def _compare_engine(self, move: _CheapMoveCandidate, frame: pd.DataFrame, **context: Any) -> dict[str, Any]:
        default_miss = context.pop("default_miss", ["No aligned signal"])
        move_side = "BUY" if move.direction == "bullish" else "SELL"
        best: dict[str, Any] | None = None
        for offset in (60, 30, 15, 10, 5, 0):
            eval_bar = (
                move.start_bar
                if offset == 0
                else max(PRE_EXPANSION_LOOKBACK, move.start_bar - max(1, int(round(offset / 5))))
            )
            state = self.reality_engine._evaluate_at_bar(
                bar=eval_bar,
                symbol=DEFAULT_SYMBOL,
                timeframe_label=MOVE_DETECTION_TIMEFRAME,
                frame=frame,
                **context,
            )
            signal = state.get("signal_direction", "NO_TRADE")
            if signal == move_side and (best is None or offset > best["minutes_before"]):
                outcome = state.get("outcome", {})
                best = {
                    "minutes_before": offset,
                    "signal_direction": signal,
                    "points_captured": float(outcome.get("realized_pnl_points") or 0.0),
                    "missing_conditions": state.get("missing_conditions", []),
                }
        return {
            "move_date": str(frame.iloc[move.start_bar].get("Date", "")),
            "move_size_points": round(move.magnitude, 2),
            "direction": move.direction,
            "engine_detected": best is not None,
            "minutes_early": best["minutes_before"] if best else None,
            "points_captured": best["points_captured"] if best else 0.0,
            "points_missed": round(max(move.magnitude - (best["points_captured"] if best else 0.0), 0.0), 2),
            "missed_reasons": best["missing_conditions"] if best else default_miss,
        }

    def _build_final_questions(
        self,
        records: list[dict[str, Any]],
        outcome_summary: dict[str, Any],
        matrix: dict[str, Any],
        warning: dict[str, Any],
        engine_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        event_counter = Counter(record["event_type"] for record in records)
        level_counter = Counter()
        ignored = Counter()
        fake = [record for record in records if "Fake" in record["real_vs_fake"]["classification"]]
        real = [record for record in records if record["real_vs_fake"]["classification"].startswith("Real")]

        for record in records:
            ctx = record["major_level_context"]
            for key in ("pdh", "pdl", "pwh", "pwl", "nearest_support", "nearest_resistance", "round_number"):
                if ctx.get(key) is not None and ctx.get(f"distance_{key}") is not None:
                    if ctx[f"distance_{key}"] <= 25:
                        level_counter[key] += 1
                    if ctx[f"distance_{key}"] > 50:
                        ignored[key] += 1

        missed = Counter()
        for row in engine_rows:
            if row.get("engine_detected"):
                continue
            for reason in row.get("missed_reasons", []):
                missed[reason] += 1

        return {
            "1_what_causes_nifty50_momentum": (
                f"Liquidity events with confirmation and structure follow-through; "
                f"top event: {event_counter.most_common(1)[0][0] if event_counter else 'N/A'}."
            ),
            "2_most_important_liquidity_events": [event for event, _ in event_counter.most_common(5)],
            "3_most_important_levels": [level for level, _ in level_counter.most_common(5)],
            "4_levels_usually_ignored": [level for level, _ in ignored.most_common(5)],
            "5_conditions_create_buy": matrix.get("top_50_buy_combinations", [])[:5],
            "6_conditions_create_sell": matrix.get("top_50_sell_combinations", [])[:5],
            "7_conditions_create_no_trade": matrix.get("top_50_no_trade_combinations", [])[:5],
            "8_conditions_create_fake_moves": [
                {"classification": record["real_vs_fake"]["classification"], "event": record["event_type"]}
                for record in fake[:10]
            ],
            "9_conditions_create_real_moves": [
                {"classification": record["real_vs_fake"]["classification"], "event": record["event_type"]}
                for record in real[:10]
            ],
            "10_earliest_warning_200_plus": warning.get("earliest_reliable_warning_200_plus"),
            "11_earliest_warning_500_plus": warning.get("earliest_reliable_warning_500_plus"),
            "12_biggest_improvement_opportunity": missed.most_common(1)[0][0] if missed else "Capture pre-tier liquidity context",
            "supporting_metrics": {
                "total_liquidity_events": len(records),
                "engine_detection_rate_200_plus_pct": round(
                    sum(1 for row in engine_rows if row.get("engine_detected")) / max(len(engine_rows), 1) * 100,
                    2,
                ),
                "top_outcome": max(outcome_summary, key=lambda key: outcome_summary[key]["sample_size"])
                if outcome_summary
                else None,
            },
        }

    def run(self, metadata: dict[str, Any]) -> Nifty50LiquidityDecisionEngineReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=RESEARCH_WINDOW_DAYS)
        )

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        enriched = self.context_builder.enrich(frame)
        liquidity_map = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)
        calendar = liquidity_map._attach_calendar_levels(frame)
        intel = self.discovery.intelligence_engine.enrich(frame)
        prechecks = self.reality_engine._build_prechecks(frame)

        frozen = self.reality_engine._load_frozen_exports()
        v2_card = frozen["v2_production_card"]
        archetypes = frozen["production_candidate_archetypes"] or frozen["top_50_archetypes"]
        buy_keys = self.reality_engine._keys_from_labels(v2_card["buy_rules"]["filter_stack"])
        sell_keys = self.reality_engine._keys_from_labels(v2_card["sell_rules"]["filter_stack"])
        no_trade_rules = list(v2_card.get("no_trade_rules", []))
        tier2_by_bar = {
            signal.bos_bar: signal
            for signal in TieredSignalFrameworkResearch(
                symbol=DEFAULT_SYMBOL,
                research_days=RESEARCH_WINDOW_DAYS,
                timeframes=(MOVE_DETECTION_TIMEFRAME,),
            )._detect_tier2(frame, MOVE_DETECTION_TIMEFRAME)
        }

        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        closes = frame["Close"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, MOVE_THRESHOLDS[0]),
        )

        records: list[dict[str, Any]] = []
        scan_end = len(frame) - FORWARD_BARS
        for bar in range(PRE_EXPANSION_LOOKBACK, scan_end):
            if bar % 1000 == 0:
                logger.info("Liquidity decision scan bar=%s/%s events=%s", bar, scan_end, len(records))
            events = [event for event in self._detect_events_at_bar(frame, calendar, bar) if event in LIQUIDITY_EVENTS]
            if not events:
                continue

            row = frame.iloc[bar]
            cal_row = calendar.iloc[bar]
            close = float(row["Close"])
            levels = self.discovery._market_levels(frame, bar)
            bull, bear = self._forward_directional_moves(highs, lows, closes, bar, FORWARD_BARS)
            linked_move = self._find_next_move(moves, bar, FORWARD_BARS)

            for event in events:
                level_swept, event_direction = self._level_swept(event, row, cal_row, levels, close)
                distance = None
                if level_swept is not None:
                    distance = round(abs(close - level_swept), 2)

                direction = "bullish" if event_direction == "bullish" else "bearish"
                outcome = self._classify_outcome(bull, bear, event_direction, event)
                decision = self._assign_decision(outcome)
                trade_metrics = self._forward_trade_metrics(frame, bar, direction)

                record = {
                    "timestamp": str(row.get("Date", "")),
                    "bar": bar,
                    "event_type": event,
                    "direction": event_direction,
                    "level_swept": level_swept,
                    "distance_from_level": distance,
                    "bars_since_level_creation": PRE_EXPANSION_LOOKBACK,
                    "major_level_context": self._major_level_context(frame, calendar, bar),
                    "level_pressure": self._level_pressure_before(
                        frame,
                        enriched,
                        calendar,
                        intel,
                        bar,
                        direction,
                    ),
                    "confirmation": self._confirmation_analysis(frame, bar, direction),
                    "structure_after": self._structure_after(frame, enriched, intel, bar, direction),
                    "outcome": outcome,
                    "decision": decision,
                    "forward_metrics": {
                        "bull_move": round(bull, 2),
                        "bear_move": round(bear, 2),
                        "max_move": round(max(bull, bear), 2),
                        "max_drawdown": trade_metrics.get("mae_points", 0.0),
                        "time_to_expansion_bars": linked_move.start_bar - bar if linked_move else -1,
                    },
                    "real_vs_fake": self._real_vs_fake(frame, bar, levels),
                    "trade_metrics": trade_metrics,
                }
                records.append(record)

        outcome_summary = self._aggregate_outcomes(records)
        matrix = self._build_decision_matrix(records)
        warning = self._earliest_warning_analysis(moves, frame, calendar)

        engine_context = {
            "enriched": enriched,
            "calendar": calendar,
            "intel": intel,
            "prechecks": prechecks,
            "tier2_by_bar": tier2_by_bar,
            "buy_keys": buy_keys,
            "sell_keys": sell_keys,
            "no_trade_rules": no_trade_rules,
            "archetypes": archetypes,
            "trade_engine": self.trade_engine,
        }
        engine_rows = [
            self._compare_engine(
                move,
                frame,
                default_miss=["No aligned signal"],
                **engine_context,
            )
            for move in moves
            if move.magnitude >= 200
        ]

        final_questions = self._build_final_questions(
            records,
            outcome_summary,
            matrix,
            warning,
            engine_rows,
        )
        conclusions = [
            "NIFTY50 Liquidity Decision Engine research complete (120-day 5M scan).",
            f"Liquidity events analyzed: {len(records)}.",
            f"Top BUY combination: {matrix['top_50_buy_combinations'][0]['combination'] if matrix['top_50_buy_combinations'] else 'N/A'}.",
            f"Earliest 200+ warning: {warning['earliest_reliable_warning_200_plus']['warning_type'] if warning.get('earliest_reliable_warning_200_plus') else 'N/A'}.",
            f"Biggest engine gap: {final_questions['12_biggest_improvement_opportunity']}.",
        ]

        return Nifty50LiquidityDecisionEngineReport(
            symbol=DEFAULT_SYMBOL,
            research_window_days=RESEARCH_WINDOW_DAYS,
            start_date=metadata.get("start_date", start.isoformat()),
            end_date=metadata.get("end_date", end.isoformat()),
            primary_timeframe=MOVE_DETECTION_TIMEFRAME,
            context_timeframes=list(CONTEXT_TIMEFRAMES),
            liquidity_events_detected=len(records),
            liquidity_event_log=records[:MAX_EVENT_EXPORT],
            outcome_summary=outcome_summary,
            decision_matrix=matrix,
            earliest_warning_analysis=warning,
            engine_comparison_200_plus=engine_rows,
            final_questions=final_questions,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_nifty50_liquidity_decision_engine_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Nifty50LiquidityDecisionEngineReport:
    """Run liquidity decision engine research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Nifty50LiquidityDecisionEngineError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata = {
        **metadata,
        "research_window_days": RESEARCH_WINDOW_DAYS,
        "start_date": (
            date.fromisoformat(metadata["end_date"]) - timedelta(days=RESEARCH_WINDOW_DAYS)
        ).isoformat(),
    }

    engine = Nifty50LiquidityDecisionEngineResearch()
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info("NIFTY50 liquidity decision engine report exported: %s", destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_nifty50_liquidity_decision_engine_report()
    except Nifty50LiquidityDecisionEngineError as exc:
        logger.error("Liquidity decision engine error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected liquidity decision engine error")
        return 1

    print("NIFTY50 Liquidity Decision Engine Research Summary")
    print(f"Liquidity events: {report.liquidity_events_detected}")
    top_buy = report.decision_matrix.get("top_50_buy_combinations", [])
    print(f"Top BUY combo: {top_buy[0]['combination'] if top_buy else 'N/A'}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
