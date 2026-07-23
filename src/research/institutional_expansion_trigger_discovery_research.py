"""
Institutional Expansion Trigger Discovery research.

Starts from completed directional moves (100+/200+/300+/500+ points), reconstructs
the 100 bars before expansion, and ranks momentum blueprints by reliability.
Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from itertools import combinations
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine
from src.context.market_intelligence_engine import MarketIntelligenceEngine, RsiState
from src.research.filter_research_engine import (
    FilterContextBuilder,
    FilterResearchEngine,
    RESEARCH_DAYS,
    _json_safe,
)
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    _CheapMoveCandidate,
)
from src.research.major_level_strength_research import LevelStrengthFeatures, MajorLevelStrengthResearch
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_expansion_trigger_discovery.json"

MOVE_THRESHOLDS = (100, 200, 300, 500)
PRE_EXPANSION_LOOKBACK = 100
TOP_BLUEPRINT_COUNT = 20
MIN_BLUEPRINT_SAMPLES = 100
MIN_SIGNIFICANT_EDGE_PCT = 3.0
SIGNIFICANCE_ALPHA = 0.05
MAX_BLUEPRINT_COMBO_SIZE = 3
MAX_MOVES_PER_TIMEFRAME = 400
MAX_EXPORT_RECORDS = 150
BLUEPRINT_ARROW = " -> "
LEVEL_TOUCH_ATR_RATIO = 0.5
CONSOLIDATION_ATR_RATIO = 1.5
VOLUME_LOOKBACK = 20
LOCATION_LOOKBACK = 200
LEVEL_CLUSTER_POINTS = 5.0

DEFAULT_SYMBOLS = ("NIFTY50", "BANKNIFTY", "FINNIFTY")
TIMEFRAMES = ("5M", "15M", "1H")
TIMEFRAME_MINUTES = {"5M": 5, "15M": 15, "1H": 60}


class ExpansionTriggerDiscoveryError(Exception):
    """Raised when expansion trigger discovery research fails."""


@dataclass(frozen=True)
class ExpansionMoveRecord:
    """One completed expansion move with pre-expansion measurements."""

    symbol: str
    timeframe: str
    direction: str
    origin_bar: int
    expansion_bar: int
    origin_timestamp: str
    expansion_timestamp: str
    move_magnitude_points: float
    hit_100_plus: bool
    hit_200_plus: bool
    hit_300_plus: bool
    hit_500_plus: bool
    measurements: dict[str, Any]
    blueprint_tags: tuple[str, ...]
    blueprint_pattern: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MomentumBlueprintRank:
    """Ranked momentum blueprint with outcome statistics."""

    blueprint: str
    direction: str
    occurrences: int
    hit_1r_rate_pct: float
    hit_2r_rate_pct: float
    hit_3r_rate_pct: float
    hit_4r_rate_pct: float
    hit_5r_rate_pct: float
    average_move_points: float
    average_drawdown_points: float
    average_time_to_expansion_bars: float
    reliability_score: float
    statistically_significant: bool
    significance_p_value: float
    edge_vs_baseline_3r_pct: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExpansionTriggerDiscoveryReport:
    """Full expansion trigger discovery output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    move_thresholds_points: list[int]
    pre_expansion_lookback_bars: int
    total_moves_analyzed: int
    moves_by_threshold: dict[str, int]
    moves_by_direction: dict[str, int]
    baseline_metrics: dict[str, dict[str, float]]
    measurement_categories: list[str]
    top_20_bullish_momentum_blueprints: list[dict[str, Any]]
    top_20_bearish_momentum_blueprints: list[dict[str, Any]]
    rejected_blueprints_below_sample_threshold: int
    expansion_records: list[dict[str, Any]]
    expansion_records_total: int
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalExpansionTriggerDiscoveryResearch:
    """Discover pre-expansion conditions that precede institutional momentum."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = TIMEFRAMES,
    ) -> None:
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.research_days = research_days
        self.timeframes = timeframes
        self.move_engine = LiquidityMoveReconstructionResearch(research_days=research_days)
        self.level_engine = MajorLevelStrengthResearch(research_days=research_days)
        self.trade_engine = TradeConstructionValidationResearch(research_days=research_days)
        self.narrative_engine = LiquidityNarrativeEngine()
        self.intelligence_engine = MarketIntelligenceEngine()
        self.context_builder = FilterContextBuilder()
        self.liquidity_map_engine = InstitutionalLiquidityMapEngine()
        self._market_levels_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def clear_market_levels_cache(self) -> None:
        self._market_levels_cache.clear()

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    @classmethod
    def _two_proportion_p_value(
        cls,
        success_a: int,
        total_a: int,
        success_b: int,
        total_b: int,
    ) -> float:
        if total_a <= 0 or total_b <= 0:
            return 1.0
        p_a = success_a / total_a
        p_b = success_b / total_b
        pooled = (success_a + success_b) / (total_a + total_b)
        if pooled <= 0.0 or pooled >= 1.0:
            return 1.0
        standard_error = math.sqrt(pooled * (1.0 - pooled) * ((1.0 / total_a) + (1.0 / total_b)))
        if standard_error == 0.0:
            return 1.0
        z_score = abs(p_a - p_b) / standard_error
        return round(2.0 * (1.0 - cls._normal_cdf(z_score)), 6)

    @staticmethod
    def _atr(frame: pd.DataFrame, index: int, period: int = 14) -> float:
        start = max(0, index - period)
        window = frame.iloc[start : index + 1]
        if len(window) < 2:
            return 1.0
        tr_values: list[float] = []
        for offset in range(1, len(window)):
            high = float(window.iloc[offset]["High"])
            low = float(window.iloc[offset]["Low"])
            prev_close = float(window.iloc[offset - 1]["Close"])
            tr_values.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        return max(mean(tr_values), 1.0) if tr_values else 1.0

    @staticmethod
    def _candle_parts(row: pd.Series) -> dict[str, float]:
        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        body = abs(close - open_price)
        candle_range = max(high - low, 0.01)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        return {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "body": body,
            "range": candle_range,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_pct": body / candle_range,
            "wick_pct": (upper_wick + lower_wick) / candle_range,
            "close_location_pct": (close - low) / candle_range,
            "bullish": close > open_price,
            "bearish": close < open_price,
        }

    def _filter_engine(self, symbol: str) -> FilterResearchEngine:
        return FilterResearchEngine(
            symbol=symbol,
            research_days=self.research_days,
            timeframes=self.timeframes,
        )

    def _detect_moves(self, frame: pd.DataFrame) -> list[_CheapMoveCandidate]:
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        candidates = self.move_engine._detect_moves_cheap(highs, lows, MOVE_THRESHOLDS[0])
        return self.move_engine._dedupe_cheap_moves(candidates)

    def _market_levels(self, frame: pd.DataFrame, index: int) -> dict[str, Any]:
        cache_key = (id(frame), index)
        cached = self._market_levels_cache.get(cache_key)
        if cached is not None:
            return cached
        start = max(0, index - LOCATION_LOOKBACK)
        window = frame.iloc[start : index + 1]
        close = self._to_float(frame.iloc[index]["Close"]) or 0.0

        def collect(column: str) -> list[float]:
            values: list[float] = []
            if column not in window.columns:
                return values
            for value in window[column]:
                parsed = self._to_float(value)
                if parsed is not None and self._is_active(value):
                    values.append(parsed)
            return values

        supports = (
            collect("Swing_Low")
            + collect("Equal_Low")
            + collect("Bullish_OB_Low")
            + collect("Sell_Side_Liquidity")
        )
        resistances = (
            collect("Swing_High")
            + collect("Equal_High")
            + collect("Bearish_OB_High")
            + collect("Buy_Side_Liquidity")
        )
        major_support = max([level for level in supports if level <= close], default=None)
        major_resistance = min([level for level in resistances if level >= close], default=None)
        if major_support is None and supports:
            major_support = min(supports)
        if major_resistance is None and resistances:
            major_resistance = max(resistances)

        distances: list[float] = []
        if major_support is not None:
            distances.append(abs(close - major_support))
        if major_resistance is not None:
            distances.append(abs(close - major_resistance))
        nearest_distance = min(distances) if distances else None

        result = {
            "major_support": major_support,
            "major_resistance": major_resistance,
            "distance_from_nearest_level": round(nearest_distance, 2) if nearest_distance is not None else None,
        }
        self._market_levels_cache[cache_key] = result
        return result

    @staticmethod
    def _level_overlap(level: float | None, price: float, tolerance: float = LEVEL_CLUSTER_POINTS) -> bool:
        return level is not None and abs(level - price) <= tolerance

    def _calendar_interactions(
        self,
        enriched: pd.DataFrame,
        start_bar: int,
        end_bar: int,
    ) -> dict[str, int]:
        pdh = pdl = pwh = pwl = pmh = pml = 0
        for index in range(start_bar, end_bar + 1):
            row = enriched.iloc[index]
            close = float(row["Close"])
            if self._level_overlap(self._to_float(row.get("_pdh")), close):
                pdh += 1
            if self._level_overlap(self._to_float(row.get("_pdl")), close):
                pdl += 1
            if self._level_overlap(self._to_float(row.get("_pwh")), close):
                pwh += 1
            if self._level_overlap(self._to_float(row.get("_pwl")), close):
                pwl += 1
            if self._level_overlap(self._to_float(row.get("_pmh")), close):
                pmh += 1
            if self._level_overlap(self._to_float(row.get("_pml")), close):
                pml += 1
        return {
            "pdh_interactions": pdh,
            "pdl_interactions": pdl,
            "pwh_interactions": pwh,
            "pwl_interactions": pwl,
            "monthly_high_interactions": pmh,
            "monthly_low_interactions": pml,
        }

    def _measure_support_resistance(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        direction: str,
        levels: dict[str, Any],
        atr: float,
    ) -> dict[str, Any]:
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        touch_threshold = atr * LEVEL_TOUCH_ATR_RATIO
        tests = bars_near = 0
        failed_breakouts = failed_breakdowns = 0
        false_breakout_depths: list[float] = []
        false_breakdown_depths: list[float] = []
        breakout_attempt_sizes: list[float] = []

        for index in range(start_bar, end_bar + 1):
            close = float(frame.iloc[index]["Close"])
            high = float(frame.iloc[index]["High"])
            low = float(frame.iloc[index]["Low"])

            for level in (support, resistance):
                if level is None:
                    continue
                if abs(close - level) <= touch_threshold:
                    tests += 1
                    bars_near += 1

            if resistance is not None:
                if high > resistance:
                    breakout_attempt_sizes.append(round(high - resistance, 2))
                    if close <= resistance:
                        failed_breakouts += 1
                        false_breakout_depths.append(round(high - resistance, 2))
            if support is not None:
                if low < support:
                    breakout_attempt_sizes.append(round(support - low, 2))
                    if close >= support:
                        failed_breakdowns += 1
                        false_breakdown_depths.append(round(support - low, 2))

        calendar = self._calendar_interactions(enriched, start_bar, end_bar)
        close = float(frame.iloc[end_bar]["Close"])
        level_features = LevelStrengthFeatures(
            number_of_touches=tests,
            days_level_survived=0,
            bars_near_level=bars_near,
            bounce_count=sum(
                1
                for index in range(start_bar, end_bar + 1)
                if support is not None
                and abs(float(frame.iloc[index]["Close"]) - support) <= touch_threshold
            ),
            rejection_count=sum(
                1
                for index in range(start_bar, end_bar + 1)
                if resistance is not None
                and abs(resistance - float(frame.iloc[index]["Close"])) <= touch_threshold
            ),
            liquidity_grabs=0,
            equal_highs_lows_nearby=0,
            previous_day_overlap=calendar["pdh_interactions"] + calendar["pdl_interactions"] > 0,
            weekly_overlap=calendar["pwh_interactions"] + calendar["pwl_interactions"] > 0,
            monthly_overlap=calendar["monthly_high_interactions"] + calendar["monthly_low_interactions"] > 0,
            demand_supply_zone_overlap=False,
            round_number_overlap=self.level_engine._round_number_overlap(close),
            gap_interactions=0,
            average_volume_expansion=1.0,
            source_column="Swing_Low" if direction == "bullish" else "Swing_High",
        )
        level_score = self.level_engine._compute_strength_score(level_features)
        level_category = self.level_engine._classify_strength(level_score)

        return {
            "number_of_tests": tests,
            "time_spent_near_level_bars": bars_near,
            "distance_from_level_points": levels.get("distance_from_nearest_level"),
            "failed_breakout_count": failed_breakouts,
            "failed_breakdown_count": failed_breakdowns,
            "false_breakout_depth_avg": round(mean(false_breakout_depths), 2) if false_breakout_depths else 0.0,
            "false_breakdown_depth_avg": round(mean(false_breakdown_depths), 2) if false_breakdown_depths else 0.0,
            "average_breakout_attempt_size": round(mean(breakout_attempt_sizes), 2) if breakout_attempt_sizes else 0.0,
            "level_strength_score": level_score,
            "level_strength_category": level_category,
            "round_number_proximity": level_features.round_number_overlap,
            **calendar,
        }

    def _measure_absorption(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        levels: dict[str, Any],
        atr: float,
    ) -> dict[str, Any]:
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        touch_threshold = atr * LEVEL_TOUCH_ATR_RATIO
        rejection_wicks = 0
        wick_sizes: list[float] = []
        wick_body_ratios: list[float] = []
        strong_body_count = 0
        close_locations: list[float] = []
        absorption_support = absorption_resistance = 0

        for index in range(start_bar, end_bar + 1):
            row = frame.iloc[index]
            parts = self._candle_parts(row)
            total_wick = parts["upper_wick"] + parts["lower_wick"]
            wick_sizes.append(total_wick)
            if parts["body"] > 0:
                wick_body_ratios.append(total_wick / parts["body"])
            if total_wick >= atr * 0.35:
                rejection_wicks += 1
            if parts["body_pct"] >= 0.65:
                strong_body_count += 1
            close_locations.append(parts["close_location_pct"] * 100)

            close = float(row["Close"])
            if support is not None and abs(close - support) <= touch_threshold and parts["lower_wick"] > parts["body"]:
                absorption_support += 1
            if resistance is not None and abs(resistance - close) <= touch_threshold and parts["upper_wick"] > parts["body"]:
                absorption_resistance += 1

        return {
            "rejection_wick_count": rejection_wicks,
            "average_wick_size_points": round(mean(wick_sizes), 2) if wick_sizes else 0.0,
            "maximum_wick_size_points": round(max(wick_sizes), 2) if wick_sizes else 0.0,
            "wick_to_body_ratio_avg": round(mean(wick_body_ratios), 2) if wick_body_ratios else 0.0,
            "strong_body_candle_count": strong_body_count,
            "close_location_pct_avg": round(mean(close_locations), 2) if close_locations else 50.0,
            "absorption_candles_at_support": absorption_support,
            "absorption_candles_at_resistance": absorption_resistance,
        }

    def _measure_liquidity(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
    ) -> dict[str, Any]:
        buy_sweeps = sell_sweeps = both_sweeps = 0
        grab_depths: list[float] = []
        grab_durations: list[int] = []
        stop_hunts = false_moves = 0

        active_grab: str | None = None
        grab_start = 0
        for index in range(start_bar, end_bar + 1):
            row = frame.iloc[index]
            buy = self._is_active(row.get("Buy_Liquidity_Sweep"))
            sell = self._is_active(row.get("Sell_Liquidity_Sweep"))
            if buy and sell:
                both_sweeps += 1
                stop_hunts += 1
            elif buy:
                buy_sweeps += 1
                stop_hunts += 1
            elif sell:
                sell_sweeps += 1
                stop_hunts += 1

            if buy or sell:
                depth = max(
                    float(row["High"]) - float(row["Close"]),
                    float(row["Close"]) - float(row["Low"]),
                )
                grab_depths.append(round(depth, 2))
                if active_grab is None:
                    active_grab = "both" if buy and sell else "buy" if buy else "sell"
                    grab_start = index
                grab_durations.append(index - grab_start + 1)
            else:
                active_grab = None

            if index >= start_bar + 1:
                window = frame.iloc[max(start_bar, index - 20) : index]
                prior_high = float(window["High"].astype(float).max())
                prior_low = float(window["Low"].astype(float).min())
                high = float(row["High"])
                low = float(row["Low"])
                close = float(row["Close"])
                if high > prior_high and close < prior_high:
                    false_moves += 1
                if low < prior_low and close > prior_low:
                    false_moves += 1

        return {
            "liquidity_grab_count": buy_sweeps + sell_sweeps + both_sweeps,
            "liquidity_grab_depth_avg": round(mean(grab_depths), 2) if grab_depths else 0.0,
            "liquidity_grab_duration_avg_bars": round(mean(grab_durations), 2) if grab_durations else 0.0,
            "buy_side_sweeps": buy_sweeps,
            "sell_side_sweeps": sell_sweeps,
            "both_side_sweeps": both_sweeps,
            "stop_hunt_count": stop_hunts,
            "false_move_count": false_moves,
        }

    def _measure_compression(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
    ) -> dict[str, Any]:
        atr_start = self._atr(frame, start_bar)
        atr_end = self._atr(frame, end_bar)
        atr_contraction = round(atr_end / atr_start, 3) if atr_start > 0 else 1.0

        window = frame.iloc[start_bar : end_bar + 1]
        range_start = float(window.iloc[: max(1, len(window) // 4)]["High"].astype(float).max()) - float(
            window.iloc[: max(1, len(window) // 4)]["Low"].astype(float).min(),
        )
        range_end = float(window.iloc[-max(1, len(window) // 4) :]["High"].astype(float).max()) - float(
            window.iloc[-max(1, len(window) // 4) :]["Low"].astype(float).min(),
        )
        range_contraction = round(range_end / range_start, 3) if range_start > 0 else 1.0

        volume_start = mean(self._to_float(row.get("Volume")) or 0.0 for _, row in window.iloc[:10].iterrows())
        volume_end = mean(self._to_float(row.get("Volume")) or 0.0 for _, row in window.iloc[-10:].iterrows())
        volume_contraction = round(volume_end / volume_start, 3) if volume_start > 0 else 1.0

        inside_bars = nr4 = nr7 = 0
        ranges: list[float] = []
        for index in range(start_bar, end_bar + 1):
            high = float(frame.iloc[index]["High"])
            low = float(frame.iloc[index]["Low"])
            bar_range = high - low
            ranges.append(bar_range)
            if index >= start_bar + 1:
                prev = frame.iloc[index - 1]
                if high <= float(prev["High"]) and low >= float(prev["Low"]):
                    inside_bars += 1
            if len(ranges) >= 4 and bar_range <= min(ranges[-4:]):
                nr4 += 1
            if len(ranges) >= 7 and bar_range <= min(ranges[-7:]):
                nr7 += 1

        consolidation_bars = 0
        atr = self._atr(frame, end_bar)
        for index in range(start_bar, end_bar + 1):
            local = frame.iloc[max(start_bar, index - 5) : index + 1]
            local_width = float(local["High"].astype(float).max()) - float(local["Low"].astype(float).min())
            if local_width <= atr * CONSOLIDATION_ATR_RATIO:
                consolidation_bars += 1

        compression_score = round(
            (1.0 - min(atr_contraction, 1.0)) * 40
            + (1.0 - min(range_contraction, 1.0)) * 30
            + (inside_bars / max(end_bar - start_bar + 1, 1)) * 100 * 0.2
            + (consolidation_bars / max(end_bar - start_bar + 1, 1)) * 100 * 0.1,
            2,
        )

        return {
            "atr_contraction_ratio": atr_contraction,
            "range_contraction_ratio": range_contraction,
            "volume_contraction_ratio": volume_contraction,
            "inside_bar_count": inside_bars,
            "nr4_count": nr4,
            "nr7_count": nr7,
            "volatility_compression_score": compression_score,
            "consolidation_duration_bars": consolidation_bars,
        }

    def _measure_expansion_trigger(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        row = frame.iloc[trigger_bar]
        parts = self._candle_parts(row)
        atr = self._atr(frame, trigger_bar)
        volume = self._to_float(row.get("Volume")) or 0.0
        vol_start = max(0, trigger_bar - VOLUME_LOOKBACK)
        avg_volume = mean(
            self._to_float(frame.iloc[offset].get("Volume")) or 0.0
            for offset in range(vol_start, trigger_bar)
        ) if trigger_bar > vol_start else volume
        volume_expansion = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0
        atr_expansion = round(parts["range"] / atr, 2) if atr > 0 else 1.0

        gap_up = gap_down = False
        if trigger_bar >= 1:
            gap = float(row["Open"]) - float(frame.iloc[trigger_bar - 1]["Close"])
            gap_up = gap > 0.5
            gap_down = gap < -0.5

        engulfing = marubozu = hammer = shooting_star = False
        morning_star = evening_star = bull_harami = bear_harami = False
        if trigger_bar >= 1:
            prev = self._candle_parts(frame.iloc[trigger_bar - 1])
            if prev["bearish"] and parts["bullish"] and parts["close"] > prev["open"] and parts["open"] < prev["close"]:
                engulfing = direction == "bullish"
            if prev["bullish"] and parts["bearish"] and parts["close"] < prev["open"] and parts["open"] > prev["close"]:
                engulfing = direction == "bearish"
            if prev["bearish"] and parts["bullish"] and parts["open"] >= prev["close"] and parts["close"] <= prev["open"]:
                bull_harami = True
            if prev["bullish"] and parts["bearish"] and parts["open"] <= prev["close"] and parts["close"] >= prev["open"]:
                bear_harami = True
        if trigger_bar >= 2:
            first = self._candle_parts(frame.iloc[trigger_bar - 2])
            middle = self._candle_parts(frame.iloc[trigger_bar - 1])
            midpoint = (first["open"] + first["close"]) / 2
            if first["bearish"] and middle["body"] <= first["body"] * 0.35 and parts["bullish"] and parts["close"] > midpoint:
                morning_star = True
            if first["bullish"] and middle["body"] <= first["body"] * 0.35 and parts["bearish"] and parts["close"] < midpoint:
                evening_star = True

        if parts["body_pct"] >= 0.85:
            marubozu = True
        if parts["body"] > 0:
            if parts["lower_wick"] >= 2 * parts["body"]:
                hammer = direction == "bullish"
            if parts["upper_wick"] >= 2 * parts["body"]:
                shooting_star = direction == "bearish"

        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)

        return {
            "body_pct": round(parts["body_pct"] * 100, 2),
            "wick_pct": round(parts["wick_pct"] * 100, 2),
            "volume_expansion_ratio": volume_expansion,
            "atr_expansion_ratio": atr_expansion,
            "gap_up": gap_up,
            "gap_down": gap_down,
            "engulfing": engulfing,
            "marubozu": marubozu,
            "hammer": hammer,
            "shooting_star": shooting_star,
            "morning_star": morning_star,
            "evening_star": evening_star,
            "bullish_harami": bull_harami,
            "bearish_harami": bear_harami,
            "displacement_strength": displacement.value,
        }

    def _measure_structure(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        choch_count = bos_count = fvg_count = ob_count = 0
        for index in range(start_bar, end_bar + 1):
            row = frame.iloc[index]
            if self._is_active(row.get("Bullish_CHOCH")) or self._is_active(row.get("Bearish_CHOCH")):
                choch_count += 1
            if self._is_active(row.get("Bullish_BOS")) or self._is_active(row.get("Bearish_BOS")):
                bos_count += 1
            if (
                self._is_active(row.get("Bullish_FVG_Top"))
                or self._is_active(row.get("Bearish_FVG_Top"))
            ):
                fvg_count += 1
            if self._is_active(row.get("Bullish_OB_High")) or self._is_active(row.get("Bearish_OB_High")):
                ob_count += 1

        close = float(frame.iloc[end_bar]["Close"])
        levels = self._market_levels(frame, end_bar)
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        if support is not None and resistance is not None:
            premium_discount = "Premium" if close > (support + resistance) / 2 else "Discount"
        else:
            premium_discount = "Equilibrium"

        ema_alignment = enriched.iloc[end_bar].get("ema_alignment", "Mixed") if "ema_alignment" in enriched.columns else "Mixed"
        intelligence = self.intelligence_engine.evaluate_bar(intel_frame, end_bar)
        htf_aligned = (
            (direction == "bullish" and intelligence.trend_state == "Bullish")
            or (direction == "bearish" and intelligence.trend_state == "Bearish")
        )

        return {
            "choch_count": choch_count,
            "bos_count": bos_count,
            "fvg_count": fvg_count,
            "ob_count": ob_count,
            "premium_discount": premium_discount,
            "trend_alignment": str(ema_alignment),
            "htf_alignment": htf_aligned,
            "trend_state": intelligence.trend_state,
        }

    def _measure_momentum_outcome(
        self,
        frame: pd.DataFrame,
        origin_bar: int,
        expansion_bar: int,
        direction: str,
        move_magnitude: float,
        timeframe_label: str,
    ) -> dict[str, Any]:
        entry_price = round(float(frame.iloc[origin_bar]["Close"]), 2)
        stop, risk = self.trade_engine._structural_stop(frame, origin_bar, entry_price, direction)
        end = min(len(frame) - 1, origin_bar + FORWARD_BARS)

        mfe = mae = 0.0
        hit_1r = hit_2r = hit_3r = hit_4r = hit_5r = False
        time_to_1r = time_to_2r = time_to_3r = None
        max_drawdown = 0.0
        expansion_time_bars = max(expansion_bar - origin_bar, 0)

        for index in range(origin_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])
            if direction == "bullish":
                favorable = bar_high - entry_price
                adverse = entry_price - bar_low
                mfe = max(mfe, favorable)
                mae = max(mae, adverse)
                if bar_low <= stop:
                    max_drawdown = max(max_drawdown, entry_price - bar_low)
                    break
                if risk > 0:
                    if favorable >= risk and time_to_1r is None:
                        hit_1r = True
                        time_to_1r = index - origin_bar
                    if favorable >= risk * 2 and time_to_2r is None:
                        hit_2r = True
                        time_to_2r = index - origin_bar
                    if favorable >= risk * 3 and time_to_3r is None:
                        hit_3r = True
                        time_to_3r = index - origin_bar
                    if favorable >= risk * 4:
                        hit_4r = True
                    if favorable >= risk * 5:
                        hit_5r = True
            else:
                favorable = entry_price - bar_low
                adverse = bar_high - entry_price
                mfe = max(mfe, favorable)
                mae = max(mae, adverse)
                if bar_high >= stop:
                    max_drawdown = max(max_drawdown, bar_high - entry_price)
                    break
                if risk > 0:
                    if favorable >= risk and time_to_1r is None:
                        hit_1r = True
                        time_to_1r = index - origin_bar
                    if favorable >= risk * 2 and time_to_2r is None:
                        hit_2r = True
                        time_to_2r = index - origin_bar
                    if favorable >= risk * 3 and time_to_3r is None:
                        hit_3r = True
                        time_to_3r = index - origin_bar
                    if favorable >= risk * 4:
                        hit_4r = True
                    if favorable >= risk * 5:
                        hit_5r = True

        return {
            "move_size_points": round(move_magnitude, 2),
            "mfe_points": round(mfe, 2),
            "mae_points": round(mae, 2),
            "max_drawdown_points": round(max(max_drawdown, mae), 2),
            "hit_1r": hit_1r,
            "hit_2r": hit_2r,
            "hit_3r": hit_3r,
            "hit_4r": hit_4r,
            "hit_5r": hit_5r,
            "time_to_1r_bars": time_to_1r,
            "time_to_2r_bars": time_to_2r,
            "time_to_3r_bars": time_to_3r,
            "time_to_expansion_bars": expansion_time_bars,
            "time_to_expansion_minutes": round(
                expansion_time_bars * TIMEFRAME_MINUTES.get(timeframe_label, 5),
                1,
            ),
            "risk_points": risk,
        }

    def _build_blueprint_tags(self, measurements: dict[str, Any], direction: str) -> tuple[str, ...]:
        sr = measurements["support_resistance"]
        absorption = measurements["absorption"]
        liquidity = measurements["liquidity"]
        compression = measurements["compression"]
        trigger = measurements["expansion_trigger_candle"]
        structure = measurements["structure"]
        tags: list[str] = []

        if sr["failed_breakdown_count"] >= 2 and direction == "bullish":
            tags.append("Failed Breakdown x2+")
        elif sr["failed_breakdown_count"] >= 1 and direction == "bullish":
            tags.append("Failed Breakdown")
        if sr["failed_breakout_count"] >= 2 and direction == "bearish":
            tags.append("Failed Breakout x2+")
        elif sr["failed_breakout_count"] >= 1 and direction == "bearish":
            tags.append("Failed Breakout")
        if sr["number_of_tests"] >= 5:
            tags.append("Level Tests x5+")
        elif sr["number_of_tests"] >= 2:
            tags.append("Level Tests")
        if sr["round_number_proximity"]:
            tags.append("Round Number")
        if sr["pdh_interactions"] + sr["pdl_interactions"] >= 1:
            tags.append("PDH/PDL")
        if sr["pwh_interactions"] + sr["pwl_interactions"] >= 1:
            tags.append("PWH/PWL")
        if sr["monthly_high_interactions"] + sr["monthly_low_interactions"] >= 1:
            tags.append("Monthly Level")
        tags.append(f"Level:{sr['level_strength_category']}")

        if absorption["absorption_candles_at_support"] >= 2 and direction == "bullish":
            tags.append("Support Absorption")
        if absorption["absorption_candles_at_resistance"] >= 2 and direction == "bearish":
            tags.append("Resistance Absorption")
        if absorption["rejection_wick_count"] >= 5:
            tags.append("Heavy Rejection Wicks")

        if liquidity["both_side_sweeps"] >= 1:
            tags.append("Both-Side Sweep")
        elif liquidity["sell_side_sweeps"] >= 2 and direction == "bullish":
            tags.append("Sell-Side Sweeps")
        elif liquidity["buy_side_sweeps"] >= 2 and direction == "bearish":
            tags.append("Buy-Side Sweeps")
        elif liquidity["liquidity_grab_count"] >= 1:
            tags.append("Liquidity Grab")
        if liquidity["false_move_count"] >= 3:
            tags.append("False Moves x3+")

        if compression["volatility_compression_score"] >= 50:
            tags.append("High Compression")
        elif compression["consolidation_duration_bars"] >= 30:
            tags.append("Consolidation")
        if compression["nr7_count"] >= 2:
            tags.append("NR7 Cluster")

        if trigger["engulfing"]:
            tags.append("Engulfing Trigger")
        if trigger["marubozu"]:
            tags.append("Marubozu Trigger")
        if trigger["hammer"] and direction == "bullish":
            tags.append("Hammer Trigger")
        if trigger["shooting_star"] and direction == "bearish":
            tags.append("Shooting Star Trigger")
        if trigger["morning_star"] and direction == "bullish":
            tags.append("Morning Star Trigger")
        if trigger["evening_star"] and direction == "bearish":
            tags.append("Evening Star Trigger")
        if trigger["volume_expansion_ratio"] >= 1.5:
            tags.append("Volume Expansion Trigger")
        tags.append(f"Displacement:{trigger['displacement_strength']}")

        if structure["choch_count"] >= 1:
            tags.append("CHOCH")
        if structure["bos_count"] >= 1:
            tags.append("BOS")
        if structure["fvg_count"] >= 1:
            tags.append("FVG")
        if structure["ob_count"] >= 1:
            tags.append("Order Block")
        tags.append(f"Zone:{structure['premium_discount']}")
        if structure["htf_alignment"]:
            tags.append("HTF Aligned")

        return tuple(tags)

    def tags_at_bar(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar_enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        bar: int,
        direction: str,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        """Compute blueprint tags and measurements at any historical bar."""
        start_bar = max(0, bar - PRE_EXPANSION_LOOKBACK)
        pre_end = bar
        measurements = self._combined_pre_expansion_measurements(
            frame,
            enriched,
            calendar_enriched,
            intel_frame,
            start_bar,
            pre_end,
            bar,
            direction,
        )
        return self._build_blueprint_tags(measurements, direction), measurements

    def _combined_pre_expansion_measurements(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar_enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        trigger_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        """Single-pass pre-expansion measurement for forward scanning performance."""
        levels = self._market_levels(frame, trigger_bar)
        atr = self._atr(frame, trigger_bar)
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        touch_threshold = atr * LEVEL_TOUCH_ATR_RATIO

        tests = bars_near = bounce_count = rejection_count = 0
        failed_breakouts = failed_breakdowns = 0
        false_breakout_depths: list[float] = []
        false_breakdown_depths: list[float] = []
        breakout_attempt_sizes: list[float] = []
        rejection_wicks = absorption_support = absorption_resistance = 0
        wick_sizes: list[float] = []
        wick_body_ratios: list[float] = []
        strong_body_count = 0
        close_locations: list[float] = []
        buy_sweeps = sell_sweeps = both_sweeps = stop_hunts = false_moves = 0
        grab_depths: list[float] = []
        inside_bars = nr4 = nr7 = 0
        consolidation_bars = 0
        choch_count = bos_count = fvg_count = ob_count = 0
        pdh = pdl = pwh = pwl = pmh = pml = 0
        bar_ranges: list[float] = []

        for index in range(start_bar, end_bar + 1):
            row = frame.iloc[index]
            cal_row = calendar_enriched.iloc[index]
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            bar_range = high - low
            bar_ranges.append(bar_range)

            parts = self._candle_parts(row)
            total_wick = parts["upper_wick"] + parts["lower_wick"]
            wick_sizes.append(total_wick)
            if parts["body"] > 0:
                wick_body_ratios.append(total_wick / parts["body"])
            if total_wick >= atr * 0.35:
                rejection_wicks += 1
            if parts["body_pct"] >= 0.65:
                strong_body_count += 1
            close_locations.append(parts["close_location_pct"] * 100)

            for level in (support, resistance):
                if level is not None and abs(close - level) <= touch_threshold:
                    tests += 1
                    bars_near += 1
            if support is not None and abs(close - support) <= touch_threshold:
                bounce_count += 1
                if parts["lower_wick"] > parts["body"]:
                    absorption_support += 1
            if resistance is not None and abs(resistance - close) <= touch_threshold:
                rejection_count += 1
                if parts["upper_wick"] > parts["body"]:
                    absorption_resistance += 1

            if resistance is not None:
                if high > resistance:
                    breakout_attempt_sizes.append(round(high - resistance, 2))
                    if close <= resistance:
                        failed_breakouts += 1
                        false_breakout_depths.append(round(high - resistance, 2))
            if support is not None:
                if low < support:
                    breakout_attempt_sizes.append(round(support - low, 2))
                    if close >= support:
                        failed_breakdowns += 1
                        false_breakdown_depths.append(round(support - low, 2))

            buy = self._is_active(row.get("Buy_Liquidity_Sweep"))
            sell = self._is_active(row.get("Sell_Liquidity_Sweep"))
            if buy and sell:
                both_sweeps += 1
                stop_hunts += 1
            elif buy:
                buy_sweeps += 1
                stop_hunts += 1
            elif sell:
                sell_sweeps += 1
                stop_hunts += 1
            if buy or sell:
                grab_depths.append(round(max(high - close, close - low), 2))

            if index >= start_bar + 1:
                window = frame.iloc[max(start_bar, index - 20) : index]
                prior_high = float(window["High"].astype(float).max())
                prior_low = float(window["Low"].astype(float).min())
                if high > prior_high and close < prior_high:
                    false_moves += 1
                if low < prior_low and close > prior_low:
                    false_moves += 1
                prev = frame.iloc[index - 1]
                if high <= float(prev["High"]) and low >= float(prev["Low"]):
                    inside_bars += 1

            if len(bar_ranges) >= 4 and bar_range <= min(bar_ranges[-4:]):
                nr4 += 1
            if len(bar_ranges) >= 7 and bar_range <= min(bar_ranges[-7:]):
                nr7 += 1

            local = frame.iloc[max(start_bar, index - 5) : index + 1]
            local_width = float(local["High"].astype(float).max()) - float(local["Low"].astype(float).min())
            if local_width <= atr * CONSOLIDATION_ATR_RATIO:
                consolidation_bars += 1

            if self._is_active(row.get("Bullish_CHOCH")) or self._is_active(row.get("Bearish_CHOCH")):
                choch_count += 1
            if self._is_active(row.get("Bullish_BOS")) or self._is_active(row.get("Bearish_BOS")):
                bos_count += 1
            if self._is_active(row.get("Bullish_FVG_Top")) or self._is_active(row.get("Bearish_FVG_Top")):
                fvg_count += 1
            if self._is_active(row.get("Bullish_OB_High")) or self._is_active(row.get("Bearish_OB_High")):
                ob_count += 1

            if self._level_overlap(self._to_float(cal_row.get("_pdh")), close):
                pdh += 1
            if self._level_overlap(self._to_float(cal_row.get("_pdl")), close):
                pdl += 1
            if self._level_overlap(self._to_float(cal_row.get("_pwh")), close):
                pwh += 1
            if self._level_overlap(self._to_float(cal_row.get("_pwl")), close):
                pwl += 1
            if self._level_overlap(self._to_float(cal_row.get("_pmh")), close):
                pmh += 1
            if self._level_overlap(self._to_float(cal_row.get("_pml")), close):
                pml += 1

        atr_start = self._atr(frame, start_bar)
        atr_contraction = round(atr / atr_start, 3) if atr_start > 0 else 1.0
        window = frame.iloc[start_bar : end_bar + 1]
        range_start = float(window.iloc[: max(1, len(window) // 4)]["High"].astype(float).max()) - float(
            window.iloc[: max(1, len(window) // 4)]["Low"].astype(float).min(),
        )
        range_end = float(window.iloc[-max(1, len(window) // 4) :]["High"].astype(float).max()) - float(
            window.iloc[-max(1, len(window) // 4) :]["Low"].astype(float).min(),
        )
        range_contraction = round(range_end / range_start, 3) if range_start > 0 else 1.0
        volume_start = mean(self._to_float(row.get("Volume")) or 0.0 for _, row in window.iloc[:10].iterrows())
        volume_end = mean(self._to_float(row.get("Volume")) or 0.0 for _, row in window.iloc[-10:].iterrows())
        volume_contraction = round(volume_end / volume_start, 3) if volume_start > 0 else 1.0
        compression_score = round(
            (1.0 - min(atr_contraction, 1.0)) * 40
            + (1.0 - min(range_contraction, 1.0)) * 30
            + (inside_bars / max(end_bar - start_bar + 1, 1)) * 100 * 0.2
            + (consolidation_bars / max(end_bar - start_bar + 1, 1)) * 100 * 0.1,
            2,
        )

        close = float(frame.iloc[trigger_bar]["Close"])
        level_features = LevelStrengthFeatures(
            number_of_touches=tests,
            days_level_survived=0,
            bars_near_level=bars_near,
            bounce_count=bounce_count,
            rejection_count=rejection_count,
            liquidity_grabs=buy_sweeps + sell_sweeps + both_sweeps,
            equal_highs_lows_nearby=0,
            previous_day_overlap=pdh + pdl > 0,
            weekly_overlap=pwh + pwl > 0,
            monthly_overlap=pmh + pml > 0,
            demand_supply_zone_overlap=False,
            round_number_overlap=self.level_engine._round_number_overlap(close),
            gap_interactions=0,
            average_volume_expansion=1.0,
            source_column="Swing_Low" if direction == "bullish" else "Swing_High",
        )
        level_score = self.level_engine._compute_strength_score(level_features)
        level_category = self.level_engine._classify_strength(level_score)

        if support is not None and resistance is not None:
            premium_discount = "Premium" if close > (support + resistance) / 2 else "Discount"
        else:
            premium_discount = "Equilibrium"
        ema_alignment = enriched.iloc[trigger_bar].get("ema_alignment", "Mixed") if "ema_alignment" in enriched.columns else "Mixed"
        trend_state = self._fast_trend_state(intel_frame, trigger_bar)
        htf_aligned = (
            (direction == "bullish" and trend_state == "Bullish")
            or (direction == "bearish" and trend_state == "Bearish")
        )

        return {
            "support_resistance": {
                "number_of_tests": tests,
                "time_spent_near_level_bars": bars_near,
                "distance_from_level_points": levels.get("distance_from_nearest_level"),
                "failed_breakout_count": failed_breakouts,
                "failed_breakdown_count": failed_breakdowns,
                "false_breakout_depth_avg": round(mean(false_breakout_depths), 2) if false_breakout_depths else 0.0,
                "false_breakdown_depth_avg": round(mean(false_breakdown_depths), 2) if false_breakdown_depths else 0.0,
                "average_breakout_attempt_size": round(mean(breakout_attempt_sizes), 2) if breakout_attempt_sizes else 0.0,
                "level_strength_score": level_score,
                "level_strength_category": level_category,
                "round_number_proximity": level_features.round_number_overlap,
                "pdh_interactions": pdh,
                "pdl_interactions": pdl,
                "pwh_interactions": pwh,
                "pwl_interactions": pwl,
                "monthly_high_interactions": pmh,
                "monthly_low_interactions": pml,
            },
            "absorption": {
                "rejection_wick_count": rejection_wicks,
                "average_wick_size_points": round(mean(wick_sizes), 2) if wick_sizes else 0.0,
                "maximum_wick_size_points": round(max(wick_sizes), 2) if wick_sizes else 0.0,
                "wick_to_body_ratio_avg": round(mean(wick_body_ratios), 2) if wick_body_ratios else 0.0,
                "strong_body_candle_count": strong_body_count,
                "close_location_pct_avg": round(mean(close_locations), 2) if close_locations else 50.0,
                "absorption_candles_at_support": absorption_support,
                "absorption_candles_at_resistance": absorption_resistance,
            },
            "liquidity": {
                "liquidity_grab_count": buy_sweeps + sell_sweeps + both_sweeps,
                "liquidity_grab_depth_avg": round(mean(grab_depths), 2) if grab_depths else 0.0,
                "liquidity_grab_duration_avg_bars": 0.0,
                "buy_side_sweeps": buy_sweeps,
                "sell_side_sweeps": sell_sweeps,
                "both_side_sweeps": both_sweeps,
                "stop_hunt_count": stop_hunts,
                "false_move_count": false_moves,
            },
            "compression": {
                "atr_contraction_ratio": atr_contraction,
                "range_contraction_ratio": range_contraction,
                "volume_contraction_ratio": volume_contraction,
                "inside_bar_count": inside_bars,
                "nr4_count": nr4,
                "nr7_count": nr7,
                "volatility_compression_score": compression_score,
                "consolidation_duration_bars": consolidation_bars,
            },
            "expansion_trigger_candle": self._measure_expansion_trigger(frame, trigger_bar, direction),
            "structure": {
                "choch_count": choch_count,
                "bos_count": bos_count,
                "fvg_count": fvg_count,
                "ob_count": ob_count,
                "premium_discount": premium_discount,
                "trend_alignment": str(ema_alignment),
                "htf_alignment": htf_aligned,
                "trend_state": trend_state,
            },
        }

    @staticmethod
    def _fast_trend_state(intel_frame: pd.DataFrame, index: int) -> str:
        row = intel_frame.iloc[index]
        if "Trend" in intel_frame.columns:
            trend = str(row.get("Trend", "Neutral"))
            if trend in {"Bullish", "Bearish", "Neutral"}:
                return trend
        try:
            ema20 = float(row["EMA20"])
            ema50 = float(row["EMA50"])
            ema200 = float(row["EMA200"])
            close = float(row["Close"])
        except (KeyError, TypeError, ValueError):
            return "Neutral"
        if close > ema20 > ema50 > ema200:
            return "Bullish"
        if close < ema20 < ema50 < ema200:
            return "Bearish"
        return "Neutral"

    def _analyze_move(
        self,
        symbol: str,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar_enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        candidate: _CheapMoveCandidate,
        timeframe_label: str,
    ) -> ExpansionMoveRecord:
        origin_bar = candidate.start_bar
        expansion_bar = candidate.expansion_bar
        start_bar = max(0, origin_bar - PRE_EXPANSION_LOOKBACK)
        pre_end = origin_bar
        levels = self._market_levels(frame, origin_bar)
        atr = self._atr(frame, origin_bar)

        measurements = {
            "support_resistance": self._measure_support_resistance(
                frame,
                calendar_enriched,
                start_bar,
                pre_end,
                candidate.direction,
                levels,
                atr,
            ),
            "absorption": self._measure_absorption(frame, start_bar, pre_end, levels, atr),
            "liquidity": self._measure_liquidity(frame, start_bar, pre_end),
            "compression": self._measure_compression(frame, start_bar, pre_end),
            "expansion_trigger_candle": self._measure_expansion_trigger(
                frame,
                origin_bar,
                candidate.direction,
            ),
            "structure": self._measure_structure(
                frame,
                enriched,
                intel_frame,
                start_bar,
                pre_end,
                candidate.direction,
            ),
            "momentum_outcome": self._measure_momentum_outcome(
                frame,
                origin_bar,
                expansion_bar,
                candidate.direction,
                candidate.magnitude,
                timeframe_label,
            ),
        }
        tags = self._build_blueprint_tags(measurements, candidate.direction)
        magnitude = candidate.magnitude

        return ExpansionMoveRecord(
            symbol=symbol,
            timeframe=timeframe_label,
            direction=candidate.direction,
            origin_bar=origin_bar,
            expansion_bar=expansion_bar,
            origin_timestamp=str(frame.iloc[origin_bar]["Date"]),
            expansion_timestamp=str(frame.iloc[expansion_bar]["Date"]),
            move_magnitude_points=round(magnitude, 2),
            hit_100_plus=magnitude >= 100,
            hit_200_plus=magnitude >= 200,
            hit_300_plus=magnitude >= 300,
            hit_500_plus=magnitude >= 500,
            measurements=measurements,
            blueprint_tags=tags,
            blueprint_pattern=BLUEPRINT_ARROW.join(tags) if tags else "No Context",
        )

    def _collect_records(self, metadata: dict[str, Any]) -> list[ExpansionMoveRecord]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        records: list[ExpansionMoveRecord] = []
        seen: set[tuple[str, str, int, str]] = set()

        for symbol in self.symbols:
            filter_engine = self._filter_engine(symbol)
            liquidity_map = InstitutionalLiquidityMapEngine(symbol=symbol)
            for timeframe_label in self.timeframes:
                path = filter_engine._pipeline_path(timeframe_label)
                if not path.exists():
                    try:
                        path = filter_engine._ensure_pipeline(timeframe_label, start, end)
                    except Exception as exc:
                        logger.warning("Skipping %s/%s pipeline: %s", symbol, timeframe_label, exc)
                        continue

                frame = pd.read_csv(path).reset_index(drop=True)
                if len(frame) < FORWARD_BARS + PRE_EXPANSION_LOOKBACK:
                    continue

                enriched = self.context_builder.enrich(frame)
                calendar_enriched = liquidity_map._attach_calendar_levels(frame)
                intel_frame = self.intelligence_engine.enrich(frame)
                candidates = self._detect_moves(frame)
                if len(candidates) > MAX_MOVES_PER_TIMEFRAME:
                    candidates = sorted(candidates, key=lambda item: item.magnitude, reverse=True)[
                        :MAX_MOVES_PER_TIMEFRAME
                    ]
                logger.info("Expansion trigger: %s/%s moves=%s", symbol, timeframe_label, len(candidates))

                for candidate in candidates:
                    if candidate.start_bar < PRE_EXPANSION_LOOKBACK:
                        continue
                    key = (symbol, timeframe_label, candidate.expansion_bar, candidate.direction)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        self._analyze_move(
                            symbol,
                            frame,
                            enriched,
                            calendar_enriched,
                            intel_frame,
                            candidate,
                            timeframe_label,
                        ),
                    )
        return records

    @staticmethod
    def _baseline_metrics(records: list[ExpansionMoveRecord], direction: str) -> dict[str, float]:
        bucket = [record for record in records if record.direction == direction]
        total = len(bucket) or 1
        outcomes = [record.measurements["momentum_outcome"] for record in bucket]
        return {
            "occurrences": float(len(bucket)),
            "hit_1r_rate_pct": round(sum(1 for item in outcomes if item["hit_1r"]) / total * 100, 2),
            "hit_2r_rate_pct": round(sum(1 for item in outcomes if item["hit_2r"]) / total * 100, 2),
            "hit_3r_rate_pct": round(sum(1 for item in outcomes if item["hit_3r"]) / total * 100, 2),
            "hit_4r_rate_pct": round(sum(1 for item in outcomes if item["hit_4r"]) / total * 100, 2),
            "hit_5r_rate_pct": round(sum(1 for item in outcomes if item["hit_5r"]) / total * 100, 2),
            "average_move_points": round(mean(record.move_magnitude_points for record in bucket), 2) if bucket else 0.0,
            "average_drawdown_points": round(mean(item["max_drawdown_points"] for item in outcomes), 2) if outcomes else 0.0,
            "average_time_to_expansion_bars": round(mean(item["time_to_expansion_bars"] for item in outcomes), 2) if outcomes else 0.0,
        }

    @staticmethod
    def _reliability_score(
        hit_1r: float,
        hit_2r: float,
        hit_3r: float,
        hit_5r: float,
        average_move: float,
        average_drawdown: float,
    ) -> float:
        drawdown_penalty = min(average_drawdown / 100.0, 1.0)
        move_bonus = min(average_move / 500.0, 1.0)
        return round(
            hit_1r * 0.05
            + hit_2r * 0.10
            + hit_3r * 0.25
            + hit_5r * 0.25
            + move_bonus * 100 * 0.20
            + (1.0 - drawdown_penalty) * 100 * 0.15,
            4,
        )

    @staticmethod
    def _blueprint_keys(tags: tuple[str, ...]) -> set[str]:
        keys: set[str] = set(tags)
        sorted_tags = sorted(tags)
        for size in range(2, min(MAX_BLUEPRINT_COMBO_SIZE, len(sorted_tags)) + 1):
            for combo in combinations(sorted_tags, size):
                keys.add(BLUEPRINT_ARROW.join(combo))
        return keys

    def _rank_blueprints(
        self,
        records: list[ExpansionMoveRecord],
        direction: str,
        baseline: dict[str, float],
    ) -> tuple[list[MomentumBlueprintRank], int]:
        bucket = [record for record in records if record.direction == direction]
        grouped: dict[str, list[ExpansionMoveRecord]] = defaultdict(list)
        for record in bucket:
            for key in self._blueprint_keys(record.blueprint_tags):
                grouped[key].append(record)

        rejected = 0
        ranked: list[MomentumBlueprintRank] = []
        baseline_3r_success = int(round(baseline["hit_3r_rate_pct"] / 100 * baseline["occurrences"]))
        baseline_total = int(baseline["occurrences"])

        for pattern, items in grouped.items():
            if len(items) < MIN_BLUEPRINT_SAMPLES:
                rejected += 1
                continue

            outcomes = [item.measurements["momentum_outcome"] for item in items]
            hit_1r = sum(1 for item in outcomes if item["hit_1r"])
            hit_2r = sum(1 for item in outcomes if item["hit_2r"])
            hit_3r = sum(1 for item in outcomes if item["hit_3r"])
            hit_4r = sum(1 for item in outcomes if item["hit_4r"])
            hit_5r = sum(1 for item in outcomes if item["hit_5r"])
            total = len(items)
            hit_1r_rate = round(hit_1r / total * 100, 2)
            hit_2r_rate = round(hit_2r / total * 100, 2)
            hit_3r_rate = round(hit_3r / total * 100, 2)
            hit_4r_rate = round(hit_4r / total * 100, 2)
            hit_5r_rate = round(hit_5r / total * 100, 2)
            avg_move = round(mean(item.move_magnitude_points for item in items), 2)
            avg_drawdown = round(mean(item["max_drawdown_points"] for item in outcomes), 2)
            avg_time = round(mean(item["time_to_expansion_bars"] for item in outcomes), 2)
            edge = round(hit_3r_rate - baseline["hit_3r_rate_pct"], 2)
            p_value = self._two_proportion_p_value(hit_3r, total, baseline_3r_success, baseline_total)
            move_edge = round(avg_move - baseline["average_move_points"], 2)
            significant = (edge >= MIN_SIGNIFICANT_EDGE_PCT or move_edge >= 50.0) and p_value < SIGNIFICANCE_ALPHA
            if not significant:
                rejected += 1
                continue

            reliability = self._reliability_score(
                hit_1r_rate,
                hit_2r_rate,
                hit_3r_rate,
                hit_5r_rate,
                avg_move,
                avg_drawdown,
            )
            ranked.append(
                MomentumBlueprintRank(
                    blueprint=pattern,
                    direction=direction,
                    occurrences=total,
                    hit_1r_rate_pct=hit_1r_rate,
                    hit_2r_rate_pct=hit_2r_rate,
                    hit_3r_rate_pct=hit_3r_rate,
                    hit_4r_rate_pct=hit_4r_rate,
                    hit_5r_rate_pct=hit_5r_rate,
                    average_move_points=avg_move,
                    average_drawdown_points=avg_drawdown,
                    average_time_to_expansion_bars=avg_time,
                    reliability_score=reliability,
                    statistically_significant=significant,
                    significance_p_value=p_value,
                    edge_vs_baseline_3r_pct=edge,
                ),
            )

        ranked.sort(
            key=lambda item: (item.reliability_score, item.hit_3r_rate_pct, item.occurrences),
            reverse=True,
        )
        for index, item in enumerate(ranked[:TOP_BLUEPRINT_COUNT], start=1):
            item.rank = index
        return ranked[:TOP_BLUEPRINT_COUNT], rejected

    def run(self, metadata: dict[str, Any]) -> ExpansionTriggerDiscoveryReport:
        started = time.perf_counter()
        records = self._collect_records(metadata)

        baseline_bull = self._baseline_metrics(records, "bullish")
        baseline_bear = self._baseline_metrics(records, "bearish")
        bullish_blueprints, rejected_bull = self._rank_blueprints(records, "bullish", baseline_bull)
        bearish_blueprints, rejected_bear = self._rank_blueprints(records, "bearish", baseline_bear)

        moves_by_threshold = Counter(
            f"{threshold}_plus"
            for record in records
            for threshold in MOVE_THRESHOLDS
            if record.move_magnitude_points >= threshold
        )
        moves_by_direction = Counter(record.direction for record in records)

        top_bull = bullish_blueprints[0] if bullish_blueprints else None
        top_bear = bearish_blueprints[0] if bearish_blueprints else None
        conclusions = [
            f"Analyzed {len(records)} completed expansions (100+ points) with {PRE_EXPANSION_LOOKBACK}-bar pre-context.",
            f"Bullish baseline 3R={baseline_bull['hit_3r_rate_pct']}% | Bearish baseline 3R={baseline_bear['hit_3r_rate_pct']}%.",
            (
                f"Top bullish blueprint: {top_bull.blueprint[:80]} "
                f"(3R={top_bull.hit_3r_rate_pct}%, reliability={top_bull.reliability_score}, n={top_bull.occurrences})"
                if top_bull
                else "No bullish blueprints met sample and significance thresholds."
            ),
            (
                f"Top bearish blueprint: {top_bear.blueprint[:80]} "
                f"(3R={top_bear.hit_3r_rate_pct}%, reliability={top_bear.reliability_score}, n={top_bear.occurrences})"
                if top_bear
                else "No bearish blueprints met sample and significance thresholds."
            ),
            f"Rejected blueprint patterns: {rejected_bull + rejected_bear} (n<{MIN_BLUEPRINT_SAMPLES} or not significant).",
        ]

        return ExpansionTriggerDiscoveryReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            move_thresholds_points=list(MOVE_THRESHOLDS),
            pre_expansion_lookback_bars=PRE_EXPANSION_LOOKBACK,
            total_moves_analyzed=len(records),
            moves_by_threshold=dict(moves_by_threshold),
            moves_by_direction=dict(moves_by_direction),
            baseline_metrics={"bullish": baseline_bull, "bearish": baseline_bear},
            measurement_categories=[
                "support_resistance",
                "absorption",
                "liquidity",
                "compression",
                "expansion_trigger_candle",
                "structure",
                "momentum_outcome",
            ],
            top_20_bullish_momentum_blueprints=[item.as_dict() for item in bullish_blueprints],
            top_20_bearish_momentum_blueprints=[item.as_dict() for item in bearish_blueprints],
            rejected_blueprints_below_sample_threshold=rejected_bull + rejected_bear,
            expansion_records=[record.as_dict() for record in records[:MAX_EXPORT_RECORDS]],
            expansion_records_total=len(records),
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_expansion_trigger_discovery_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> ExpansionTriggerDiscoveryReport:
    """Run expansion trigger discovery research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise ExpansionTriggerDiscoveryError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalExpansionTriggerDiscoveryResearch(
        symbols=symbols,
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", TIMEFRAMES)),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Expansion trigger discovery completed: moves=%s bullish=%s bearish=%s",
        report.total_moves_analyzed,
        len(report.top_20_bullish_momentum_blueprints),
        len(report.top_20_bearish_momentum_blueprints),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_expansion_trigger_discovery_report()
        print("Institutional Expansion Trigger Discovery Summary")
        print(f"Moves analyzed: {report.total_moves_analyzed}")
        print(f"Bullish blueprints: {len(report.top_20_bullish_momentum_blueprints)}")
        print(f"Bearish blueprints: {len(report.top_20_bearish_momentum_blueprints)}")
        if report.top_20_bullish_momentum_blueprints:
            top = report.top_20_bullish_momentum_blueprints[0]
            print(f"Top bullish: {top['blueprint'][:90]} (1R={top['hit_1r_rate_pct']}%)")
        if report.top_20_bearish_momentum_blueprints:
            top = report.top_20_bearish_momentum_blueprints[0]
            print(f"Top bearish: {top['blueprint'][:90]} (1R={top['hit_1r_rate_pct']}%)")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except ExpansionTriggerDiscoveryError as exc:
        logger.error("Expansion trigger discovery error: %s", exc)
        print(f"Expansion trigger discovery error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected expansion trigger discovery error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
