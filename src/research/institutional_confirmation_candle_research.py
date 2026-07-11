"""
Institutional Confirmation Candle research for SmartMoneyEngine.

Identifies exact candle characteristics on the trigger bar immediately before
real directional moves. Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine
from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import (
    RESEARCH_DAYS,
    FilterContextBuilder,
    FilterResearchEngine,
    _json_safe,
)
from src.research.historical_expansion_validator import SUPPORTED_SYMBOLS
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    TIMEFRAME_MINUTES,
    _CheapMoveCandidate,
)
from src.research.major_level_strength_research import (
    LevelStrengthFeatures,
    MajorLevelStrengthResearch,
)
from src.research.rsi_divergence_research_engine import DivergenceType, RsiDivergenceDetector
from src.research.support_resistance_pressure_research import SupportResistancePressureResearch
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_confirmation_candle.json"
DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "outputs" / "pipeline"

MOVE_THRESHOLDS = (50, 100, 150, 200, 300)
PRE_CANDLE_LOOKBACK = 20
VOLUME_LOOKBACK = 20
ATR_LOOKBACK = 14
STRONG_BODY_RATIO = 0.65
MEDIUM_BODY_RATIO = 0.45
VOLUME_EXPANSION_THRESHOLD = 1.5
ATR_EXPANSION_THRESHOLD = 1.2
CLOSE_TOP_THRESHOLD = 0.80
CLOSE_BOTTOM_THRESHOLD = 0.20
WICK_SWEEP_RATIO = 2.0
MAX_MOVES_PER_TIMEFRAME = 300
COHORT_TOP_FRACTION = 0.20
COHORT_BOTTOM_FRACTION = 0.20
MIN_FEATURE_LIFT = 8.0
MIN_PATTERN_SAMPLES = 5
TOP_PATTERN_COUNT = 25
SIGNATURE_ARROW = " + "

TIMEFRAMES = ("5M", "15M", "1H")


class InstitutionalConfirmationCandleError(Exception):
    """Raised when institutional confirmation candle research fails."""


class MoveScenario(str, Enum):
    MOVE_100_PLUS = "move_100_plus"
    MOVE_200_PLUS = "move_200_plus"
    MOVE_300_PLUS = "move_300_plus"
    BUY = "buy"
    SELL = "sell"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


@dataclass(frozen=True)
class ConfirmationCandleRecord:
    """Trigger-candle analysis for one directional move."""

    symbol: str
    timeframe: str
    direction: str
    signal_side: str
    move_magnitude_points: float
    move_threshold_tier: int
    origin_bar: int
    expansion_bar: int
    trigger_bar: int
    trigger_timestamp: str
    candle_characteristics: dict[str, Any]
    feature_signature: str
    feature_tags: tuple[str, ...]
    scenarios: tuple[str, ...]
    confirmation_score: float
    expansion_outcome: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CharacteristicFrequency:
    characteristic: str
    threshold_label: str
    frequency_pct: float
    sample_count: int
    total_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbabilityMatrixEntry:
    signature: str
    sample_count: int
    probability_100_plus_pct: float
    probability_200_plus_pct: float
    probability_300_plus_pct: float
    average_move_magnitude: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BestConfirmationCandle:
    scenario: str
    signature: str
    sample_count: int
    frequency_pct: float
    average_move_magnitude: float
    average_confirmation_score: float
    probability_100_plus_pct: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CharacteristicPredictivePower:
    characteristic: str
    signal_side: str
    top_cohort_frequency_pct: float
    bottom_cohort_frequency_pct: float
    lift_pct: float
    predictive_power_score: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalConfirmationCandleReport:
    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    move_thresholds_points: list[int]
    total_moves_analyzed: int
    moves_by_symbol: dict[str, int]
    moves_by_threshold: dict[str, int]
    characteristic_frequency_by_threshold: dict[str, list[dict[str, Any]]]
    probability_matrix: list[dict[str, Any]]
    institutional_confirmation_candle_score: dict[str, Any]
    best_confirmation_candles: dict[str, list[dict[str, Any]]]
    characteristic_predictive_power: dict[str, list[dict[str, Any]]]
    aggregate_confirmation_profiles: dict[str, dict[str, Any]]
    confirmation_records: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalConfirmationCandleResearch:
    """Analyze trigger-candle characteristics before directional moves."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = TIMEFRAMES,
    ) -> None:
        self.symbols = symbols or tuple(self.discover_symbols())
        self.research_days = research_days
        self.timeframes = timeframes
        self.move_engine = LiquidityMoveReconstructionResearch(research_days=research_days)
        self.level_strength_engine = MajorLevelStrengthResearch(research_days=research_days)
        self.pressure_engine = SupportResistancePressureResearch(research_days=research_days)
        self.liquidity_map_engine = InstitutionalLiquidityMapEngine()
        self.intelligence_engine = MarketIntelligenceEngine()
        self.context_builder = FilterContextBuilder()
        self.rsi_detector = RsiDivergenceDetector()

    @staticmethod
    def discover_symbols() -> list[str]:
        symbols: list[str] = []
        for symbol in SUPPORTED_SYMBOLS:
            if symbol in ("NIFTY50", "BANKNIFTY"):
                symbols.append(symbol)
        if DEFAULT_PIPELINE_DIR.exists():
            for path in DEFAULT_PIPELINE_DIR.glob("*_5m_pipeline.csv"):
                slug = path.stem.replace("_pipeline", "")
                parts = slug.rsplit("_", 1)
                if len(parts) == 2 and parts[0] not in symbols:
                    symbols.append(parts[0])
        return symbols

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
    def _rsi_band(rsi: float) -> str:
        if rsi < 30:
            return "<30"
        if rsi < 40:
            return "30-40"
        if rsi < 50:
            return "40-50"
        if rsi < 60:
            return "50-60"
        if rsi < 70:
            return "60-70"
        return ">70"

    @staticmethod
    def _minutes_per_bar(timeframe_label: str) -> int:
        return TIMEFRAME_MINUTES.get(timeframe_label.upper(), 5)

    @staticmethod
    def _candle_parts(row: pd.Series) -> dict[str, Any]:
        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        body = abs(close - open_price)
        candle_range = max(high - low, 0.01)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        wick_total = max(upper_wick + lower_wick, 0.01)
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
            "wick_pct": wick_total / candle_range,
            "body_to_wick_ratio": body / wick_total,
            "close_location_pct": (close - low) / candle_range,
            "bullish": close > open_price,
            "bearish": close < open_price,
        }

    def _filter_engine(self, symbol: str) -> FilterResearchEngine:
        return FilterResearchEngine(symbol=symbol, research_days=self.research_days, timeframes=self.timeframes)

    def _detect_moves(self, frame: pd.DataFrame) -> list[_CheapMoveCandidate]:
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        candidates = self.move_engine._detect_moves_cheap(highs, lows, MOVE_THRESHOLDS[0])
        return self.move_engine._dedupe_cheap_moves(candidates)

    def _market_levels(self, frame: pd.DataFrame, index: int) -> dict[str, Any]:
        start = max(0, index - 200)
        window = frame.iloc[start : index + 1]
        close = float(frame.iloc[index]["Close"])

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
        return {
            "major_support": major_support,
            "major_resistance": major_resistance,
        }

    def _level_strength_at_bar(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        index: int,
        direction: str,
    ) -> dict[str, Any]:
        levels = self._market_levels(frame, index)
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        close = float(frame.iloc[index]["Close"])
        atr = self.pressure_engine._atr(frame, index)

        tests = bars_near = 0
        was_near = False
        start = max(0, index - PRE_CANDLE_LOOKBACK)
        target = support if direction == "bullish" else resistance
        for bar in range(start, index + 1):
            if target is None:
                break
            row = frame.iloc[bar]
            high = float(row["High"])
            low = float(row["Low"])
            c = float(row["Close"])
            touch_band = atr * 0.5
            near = abs(c - target) <= touch_band or (low - touch_band <= target <= high + touch_band)
            if near:
                bars_near += 1
                if not was_near:
                    tests += 1
            was_near = near

        features = LevelStrengthFeatures(
            number_of_touches=tests,
            days_level_survived=0,
            bars_near_level=bars_near,
            bounce_count=0,
            rejection_count=0,
            liquidity_grabs=0,
            equal_highs_lows_nearby=0,
            previous_day_overlap=False,
            weekly_overlap=False,
            monthly_overlap=False,
            demand_supply_zone_overlap=False,
            round_number_overlap=self.level_strength_engine._round_number_overlap(close),
            gap_interactions=0,
            average_volume_expansion=1.0,
            source_column="Swing_Low" if direction == "bullish" else "Swing_High",
        )
        pd_ov, wk_ov, mo_ov = self.level_strength_engine._calendar_overlaps(enriched, index, close)
        features = LevelStrengthFeatures(
            **{
                **features.as_dict(),
                "previous_day_overlap": pd_ov,
                "weekly_overlap": wk_ov,
                "monthly_overlap": mo_ov,
                "demand_supply_zone_overlap": self.level_strength_engine._demand_supply_overlap(
                    frame,
                    index,
                    close,
                ),
            },
        )
        score = self.level_strength_engine._compute_strength_score(features)
        category = self.level_strength_engine._classify_strength(score)
        return {
            "level_strength_score": score,
            "level_strength_category": category,
            "nearest_support": support,
            "nearest_resistance": resistance,
            "distance_from_support": round(abs(close - support), 2) if support is not None else None,
            "distance_from_resistance": round(abs(resistance - close), 2) if resistance is not None else None,
        }

    def _pre_candle_liquidity(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
    ) -> dict[str, Any]:
        start = max(0, trigger_bar - PRE_CANDLE_LOOKBACK)
        end = max(start, trigger_bar - 1)
        buy_grab = sell_grab = False
        failed_breakouts = failed_breakdowns = 0
        atr = self.pressure_engine._atr(frame, trigger_bar)

        for index in range(start, trigger_bar + 1):
            row = frame.iloc[index]
            if self._is_active(row.get("Buy_Liquidity_Sweep")):
                buy_grab = True
            if self._is_active(row.get("Sell_Liquidity_Sweep")):
                sell_grab = True

            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            swing_high = float(frame.iloc[max(0, index - 20) : index + 1]["High"].astype(float).max())
            swing_low = float(frame.iloc[max(0, index - 20) : index + 1]["Low"].astype(float).min())
            if index <= end:
                if high > swing_high and close <= swing_high:
                    failed_breakouts += 1
                if low < swing_low and close >= swing_low:
                    failed_breakdowns += 1

        if buy_grab and sell_grab:
            grab_type = "Both-side sweep"
        elif buy_grab:
            grab_type = "Buy-side sweep"
        elif sell_grab:
            grab_type = "Sell-side sweep"
        else:
            grab_type = "No sweep"

        return {
            "liquidity_grab_before_candle": buy_grab or sell_grab,
            "liquidity_grab_type": grab_type,
            "false_breakout_before_candle": failed_breakouts >= 1,
            "false_breakdown_before_candle": failed_breakdowns >= 1,
            "false_breakout_count": failed_breakouts,
            "false_breakdown_count": failed_breakdowns,
            "stop_hunt_size_points": round(atr * 0.15, 2),
        }

    def _structure_at_trigger(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        row = frame.iloc[trigger_bar]
        choch = False
        bos = False
        fvg = False

        if direction == "bullish":
            choch = self._is_active(row.get("Bullish_CHOCH"))
            bos = self._is_active(row.get("Bullish_BOS"))
            fvg = self._is_active(row.get("Bullish_FVG_Top")) and self._is_active(row.get("Bullish_FVG_Bottom"))
        else:
            choch = self._is_active(row.get("Bearish_CHOCH"))
            bos = self._is_active(row.get("Bearish_BOS"))
            fvg = self._is_active(row.get("Bearish_FVG_Top")) and self._is_active(row.get("Bearish_FVG_Bottom"))

        return {
            "choch_present": choch,
            "bos_present": bos,
            "fvg_present": fvg,
        }

    def _analyze_trigger_candle(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        row = frame.iloc[trigger_bar]
        parts = self._candle_parts(row)
        atr = self.pressure_engine._atr(frame, trigger_bar)

        atr_start = max(0, trigger_bar - ATR_LOOKBACK * 2)
        prior_atrs = [
            self.pressure_engine._atr(frame, index)
            for index in range(atr_start, trigger_bar)
        ]
        avg_prior_atr = mean(prior_atrs) if prior_atrs else atr
        atr_expansion = round(atr / avg_prior_atr, 2) if avg_prior_atr > 0 else 1.0

        volume = self._to_float(row.get("Volume")) or 0.0
        vol_start = max(0, trigger_bar - VOLUME_LOOKBACK)
        avg_volume = mean(
            self._to_float(frame.iloc[index].get("Volume")) or 0.0
            for index in range(vol_start, trigger_bar)
        ) if trigger_bar > vol_start else volume
        volume_expansion = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0

        level = self._level_strength_at_bar(frame, enriched, trigger_bar, direction)
        liquidity = self._pre_candle_liquidity(frame, trigger_bar)
        structure = self._structure_at_trigger(frame, trigger_bar, direction)

        prev_close = float(frame.iloc[trigger_bar - 1]["Close"]) if trigger_bar > 0 else parts["open"]
        gap = parts["open"] - prev_close
        internal = self.liquidity_map_engine._internal_liquidity(frame, trigger_bar, parts["close"])
        intel = self.intelligence_engine.evaluate_bar(intel_frame, trigger_bar)

        rsi_series = enriched["_rsi"] if "_rsi" in enriched.columns else enriched.get("RSI")
        if rsi_series is None:
            rsi_val = 50.0
        else:
            rsi_val = float(rsi_series.iloc[trigger_bar]) if pd.notna(rsi_series.iloc[trigger_bar]) else 50.0
        divergences = self.rsi_detector.detect(enriched, trigger_bar, enriched["_rsi"])
        div_labels = [item.value for item in divergences]

        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)

        return {
            "body_size_points": round(parts["body"], 2),
            "upper_wick_size_points": round(parts["upper_wick"], 2),
            "lower_wick_size_points": round(parts["lower_wick"], 2),
            "body_to_wick_ratio": round(parts["body_to_wick_ratio"], 2),
            "close_location_pct": round(parts["close_location_pct"] * 100, 2),
            "volume_expansion_ratio": volume_expansion,
            "atr_expansion_ratio": atr_expansion,
            "distance_from_support": level["distance_from_support"],
            "distance_from_resistance": level["distance_from_resistance"],
            "level_strength_score": level["level_strength_score"],
            "level_strength_category": level["level_strength_category"],
            "liquidity_grab_before_candle": liquidity["liquidity_grab_before_candle"],
            "liquidity_grab_type": liquidity["liquidity_grab_type"],
            "false_breakout_before_candle": liquidity["false_breakout_before_candle"],
            "false_breakdown_before_candle": liquidity["false_breakdown_before_candle"],
            "choch_present": structure["choch_present"],
            "bos_present": structure["bos_present"],
            "fvg_present": structure["fvg_present"],
            "gap_up": gap >= 0.5,
            "gap_down": gap <= -0.5,
            "gap_size_points": round(abs(gap), 2),
            "premium_discount": internal.price_zone,
            "rsi_level": round(rsi_val, 2),
            "rsi_band": self._rsi_band(rsi_val),
            "bullish_divergence": DivergenceType.BULLISH.value in div_labels
            or DivergenceType.HIDDEN_BULLISH.value in div_labels,
            "bearish_divergence": DivergenceType.BEARISH.value in div_labels
            or DivergenceType.HIDDEN_BEARISH.value in div_labels,
            "displacement_strength": displacement.value,
            "strong_body": parts["body_pct"] >= STRONG_BODY_RATIO,
            "body_pct": round(parts["body_pct"] * 100, 2),
        }

    def _body_label(self, body_pct: float) -> str:
        if body_pct >= STRONG_BODY_RATIO * 100:
            return "Strong Body"
        if body_pct >= MEDIUM_BODY_RATIO * 100:
            return "Medium Body"
        return "Weak Body"

    def _close_label(self, close_location_pct: float, direction: str) -> str:
        if direction == "bullish":
            if close_location_pct >= CLOSE_TOP_THRESHOLD * 100:
                return "Close Top 20%"
            if close_location_pct <= CLOSE_BOTTOM_THRESHOLD * 100:
                return "Close Bottom 20%"
        else:
            if close_location_pct <= CLOSE_BOTTOM_THRESHOLD * 100:
                return "Close Bottom 20%"
            if close_location_pct >= CLOSE_TOP_THRESHOLD * 100:
                return "Close Top 20%"
        return "Close Mid"

    def _wick_label(self, parts: dict[str, Any], direction: str) -> str:
        body = parts.get("body_size_points", 0.0) or 0.01
        lower = parts.get("lower_wick_size_points", 0.0)
        upper = parts.get("upper_wick_size_points", 0.0)
        if direction == "bullish" and lower >= WICK_SWEEP_RATIO * body:
            return "Lower Wick Sweep"
        if direction == "bearish" and upper >= WICK_SWEEP_RATIO * body:
            return "Upper Wick Sweep"
        if upper >= WICK_SWEEP_RATIO * body or lower >= WICK_SWEEP_RATIO * body:
            return "Wick Sweep"
        return "Balanced Wick"

    def _build_feature_tags(self, candle: dict[str, Any], direction: str) -> tuple[str, ...]:
        tags: list[str] = []
        tags.append(self._body_label(candle["body_pct"]))
        tags.append(self._close_label(candle["close_location_pct"], direction))
        tags.append(self._wick_label(candle, direction))
        if candle["volume_expansion_ratio"] >= VOLUME_EXPANSION_THRESHOLD:
            tags.append("Volume Expansion")
        if candle["atr_expansion_ratio"] >= ATR_EXPANSION_THRESHOLD:
            tags.append("ATR Expansion")
        tags.append(f"Level:{candle['level_strength_category']}")
        if candle["liquidity_grab_before_candle"]:
            tags.append(f"Sweep:{candle['liquidity_grab_type']}")
        if candle["false_breakout_before_candle"]:
            tags.append("False Breakout")
        if candle["false_breakdown_before_candle"]:
            tags.append("False Breakdown")
        if candle["choch_present"]:
            tags.append(f"CHOCH:{direction}")
        if candle["bos_present"]:
            tags.append(f"BOS:{direction}")
        if candle["fvg_present"]:
            tags.append("FVG Present")
        if candle["gap_up"]:
            tags.append("Gap Up")
        if candle["gap_down"]:
            tags.append("Gap Down")
        if candle["premium_discount"]:
            tags.append(f"Zone:{candle['premium_discount']}")
        tags.append(f"RSI:{candle['rsi_band']}")
        if candle["bullish_divergence"] and direction == "bullish":
            tags.append("Bullish Divergence")
        if candle["bearish_divergence"] and direction == "bearish":
            tags.append("Bearish Divergence")
        if candle["displacement_strength"] in {"Strong", "Medium"}:
            tags.append(f"Displacement:{candle['displacement_strength']}")
        return tuple(tags)

    def _build_signature(self, tags: tuple[str, ...]) -> str:
        return SIGNATURE_ARROW.join(tags) if tags else "No Context"

    def _assign_scenarios(
        self,
        magnitude: float,
        direction: str,
        candle: dict[str, Any],
    ) -> tuple[str, ...]:
        scenarios: list[str] = []
        if direction == "bullish":
            scenarios.append(MoveScenario.BUY.value)
        else:
            scenarios.append(MoveScenario.SELL.value)
        if magnitude >= 100:
            scenarios.append(MoveScenario.MOVE_100_PLUS.value)
        if magnitude >= 200:
            scenarios.append(MoveScenario.MOVE_200_PLUS.value)
        if magnitude >= 300:
            scenarios.append(MoveScenario.MOVE_300_PLUS.value)

        support = candle.get("distance_from_support")
        resistance = candle.get("distance_from_resistance")
        atr_proxy = max(candle.get("body_size_points", 1.0) * 3, 20.0)

        if direction == "bullish" and support is not None and support <= atr_proxy:
            scenarios.append(MoveScenario.SUPPORT_BOUNCE.value)
        if direction == "bearish" and resistance is not None and resistance <= atr_proxy:
            scenarios.append(MoveScenario.RESISTANCE_REJECTION.value)
        if candle.get("false_breakout_before_candle"):
            scenarios.append(MoveScenario.BREAKOUT.value)
        if candle.get("false_breakdown_before_candle"):
            scenarios.append(MoveScenario.BREAKDOWN.value)
        return tuple(scenarios)

    def _expansion_outcome(
        self,
        frame: pd.DataFrame,
        origin_bar: int,
        expansion_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        end = min(len(frame) - 1, expansion_bar + FORWARD_BARS)
        origin_close = float(frame.iloc[origin_bar]["Close"])
        atr = self.pressure_engine._atr(frame, origin_bar)
        risk = max(atr * 1.5, 20.0)

        if direction == "bullish":
            sl = float(frame.iloc[origin_bar : expansion_bar + 1]["Low"].astype(float).min())
            risk = max(origin_close - sl, risk)
            segment = frame.iloc[expansion_bar + 1 : end + 1]
            mfe = float(segment["High"].astype(float).max()) - origin_close if len(segment) else 0.0
            mae = origin_close - float(segment["Low"].astype(float).min()) if len(segment) else 0.0
        else:
            sl = float(frame.iloc[origin_bar : expansion_bar + 1]["High"].astype(float).max())
            risk = max(sl - origin_close, risk)
            segment = frame.iloc[expansion_bar + 1 : end + 1]
            mfe = origin_close - float(segment["Low"].astype(float).min()) if len(segment) else 0.0
            mae = float(segment["High"].astype(float).max()) - origin_close if len(segment) else 0.0

        mfe = max(mfe, 0.0)
        mae = max(mae, 0.0)

        def hit_r(multiple: int) -> bool:
            return mfe >= risk * multiple

        return {
            "maximum_move_points": round(mfe, 2),
            "maximum_favorable_excursion": round(mfe, 2),
            "maximum_adverse_excursion": round(mae, 2),
            "risk_points": round(risk, 2),
            "hit_1r": hit_r(1),
            "hit_2r": hit_r(2),
            "hit_3r": hit_r(3),
            "hit_4r": hit_r(4),
            "hit_5r": hit_r(5),
        }

    def _analyze_move(
        self,
        symbol: str,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        candidate: _CheapMoveCandidate,
        timeframe_label: str,
    ) -> ConfirmationCandleRecord:
        origin_bar = candidate.start_bar
        expansion_bar = candidate.expansion_bar
        trigger_bar = max(0, expansion_bar - 1)
        direction = candidate.direction
        signal_side = "BUY" if direction == "bullish" else "SELL"

        candle = self._analyze_trigger_candle(frame, enriched, intel_frame, trigger_bar, direction)
        tags = self._build_feature_tags(candle, direction)
        signature = self._build_signature(tags)
        scenarios = self._assign_scenarios(candidate.magnitude, direction, candle)
        expansion = self._expansion_outcome(frame, origin_bar, expansion_bar, direction)

        tier = MOVE_THRESHOLDS[0]
        for threshold in MOVE_THRESHOLDS:
            if candidate.magnitude >= threshold:
                tier = threshold

        return ConfirmationCandleRecord(
            symbol=symbol,
            timeframe=timeframe_label,
            direction=direction,
            signal_side=signal_side,
            move_magnitude_points=candidate.magnitude,
            move_threshold_tier=tier,
            origin_bar=origin_bar,
            expansion_bar=expansion_bar,
            trigger_bar=trigger_bar,
            trigger_timestamp=str(frame.iloc[trigger_bar]["Date"]),
            candle_characteristics=candle,
            feature_signature=signature,
            feature_tags=tags,
            scenarios=scenarios,
            confirmation_score=0.0,
            expansion_outcome=expansion,
        )

    def _collect_records(self, metadata: dict[str, Any]) -> list[ConfirmationCandleRecord]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        records: list[ConfirmationCandleRecord] = []
        seen: set[tuple[str, str, int, str]] = set()

        for symbol in self.symbols:
            filter_engine = self._filter_engine(symbol)
            for timeframe_label in self.timeframes:
                path = filter_engine._pipeline_path(timeframe_label)
                if not path.exists():
                    try:
                        path = filter_engine._ensure_pipeline(timeframe_label, start, end)
                    except Exception as exc:
                        logger.warning("Skipping %s/%s pipeline: %s", symbol, timeframe_label, exc)
                        continue

                frame = pd.read_csv(path).reset_index(drop=True)
                if len(frame) < FORWARD_BARS + PRE_CANDLE_LOOKBACK:
                    continue

                enriched = self.context_builder.enrich(frame)
                enriched = self.liquidity_map_engine._attach_calendar_levels(enriched)
                intel_frame = self.intelligence_engine.enrich(frame)
                candidates = self._detect_moves(frame)
                if len(candidates) > MAX_MOVES_PER_TIMEFRAME:
                    candidates = sorted(candidates, key=lambda item: item.magnitude, reverse=True)[
                        :MAX_MOVES_PER_TIMEFRAME
                    ]
                logger.info(
                    "Confirmation candle: %s/%s moves=%s",
                    symbol,
                    timeframe_label,
                    len(candidates),
                )

                for candidate in candidates:
                    if candidate.start_bar < PRE_CANDLE_LOOKBACK:
                        continue
                    key = (symbol, timeframe_label, candidate.expansion_bar, candidate.direction)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        self._analyze_move(symbol, frame, enriched, intel_frame, candidate, timeframe_label),
                    )

        return records

    def _characteristic_frequency(
        self,
        records: list[ConfirmationCandleRecord],
        threshold: int,
    ) -> list[CharacteristicFrequency]:
        bucket = [record for record in records if record.move_magnitude_points >= threshold]
        total = len(bucket)
        if total == 0:
            return []

        tag_counts: Counter[str] = Counter()
        for record in bucket:
            tag_counts.update(record.feature_tags)

        results: list[CharacteristicFrequency] = []
        for tag, count in tag_counts.most_common(TOP_PATTERN_COUNT):
            results.append(
                CharacteristicFrequency(
                    characteristic=tag,
                    threshold_label=f"{threshold}+",
                    frequency_pct=round((count / total) * 100, 2),
                    sample_count=count,
                    total_count=total,
                ),
            )
        return results

    def _probability_matrix(
        self,
        records: list[ConfirmationCandleRecord],
    ) -> list[ProbabilityMatrixEntry]:
        signature_groups: dict[str, list[ConfirmationCandleRecord]] = defaultdict(list)
        for record in records:
            signature_groups[record.feature_signature].append(record)

        entries: list[ProbabilityMatrixEntry] = []
        for signature, group in signature_groups.items():
            if len(group) < MIN_PATTERN_SAMPLES:
                continue
            total = len(group)
            prob_100 = sum(1 for item in group if item.move_magnitude_points >= 100) / total * 100
            prob_200 = sum(1 for item in group if item.move_magnitude_points >= 200) / total * 100
            prob_300 = sum(1 for item in group if item.move_magnitude_points >= 300) / total * 100
            avg_mag = mean(item.move_magnitude_points for item in group)
            entries.append(
                ProbabilityMatrixEntry(
                    signature=signature,
                    sample_count=total,
                    probability_100_plus_pct=round(prob_100, 2),
                    probability_200_plus_pct=round(prob_200, 2),
                    probability_300_plus_pct=round(prob_300, 2),
                    average_move_magnitude=round(avg_mag, 2),
                ),
            )

        entries.sort(
            key=lambda item: (item.probability_100_plus_pct, item.sample_count, item.average_move_magnitude),
            reverse=True,
        )
        for index, entry in enumerate(entries[:TOP_PATTERN_COUNT], start=1):
            entry.rank = index
        return entries[:TOP_PATTERN_COUNT]

    def _predictive_power(
        self,
        records: list[ConfirmationCandleRecord],
    ) -> tuple[dict[str, list[CharacteristicPredictivePower]], dict[str, float]]:
        results: dict[str, list[CharacteristicPredictivePower]] = {}
        methodology_weights: dict[str, float] = {"BUY": 0.0, "SELL": 0.0}

        for signal_side in ("BUY", "SELL"):
            bucket = sorted(
                [record for record in records if record.signal_side == signal_side],
                key=lambda item: item.move_magnitude_points,
                reverse=True,
            )
            if len(bucket) < 10:
                results[signal_side] = []
                continue

            top_n = max(1, int(len(bucket) * COHORT_TOP_FRACTION))
            bottom_n = max(1, int(len(bucket) * COHORT_BOTTOM_FRACTION))
            top = bucket[:top_n]
            bottom = bucket[-bottom_n:]

            all_tags = set()
            for record in bucket:
                all_tags.update(record.feature_tags)

            rows: list[CharacteristicPredictivePower] = []
            lifts: list[float] = []
            for tag in sorted(all_tags):
                top_freq = sum(1 for record in top if tag in record.feature_tags) / len(top) * 100
                bottom_freq = sum(1 for record in bottom if tag in record.feature_tags) / len(bottom) * 100
                lift = round(top_freq - bottom_freq, 2)
                power = round(lift * (top_freq / 100) if top_freq else 0.0, 4)
                rows.append(
                    CharacteristicPredictivePower(
                        characteristic=tag,
                        signal_side=signal_side,
                        top_cohort_frequency_pct=round(top_freq, 2),
                        bottom_cohort_frequency_pct=round(bottom_freq, 2),
                        lift_pct=lift,
                        predictive_power_score=power,
                    ),
                )
                if top_freq >= 25 and lift >= MIN_FEATURE_LIFT:
                    lifts.append(lift)

            rows.sort(key=lambda item: (item.predictive_power_score, item.lift_pct), reverse=True)
            for index, row in enumerate(rows, start=1):
                row.rank = index
            results[signal_side] = rows
            if lifts:
                methodology_weights[signal_side] = round(min(sum(lifts[:8]) / 8 * 10, 100), 2)

        return results, methodology_weights

    def _apply_scores(
        self,
        records: list[ConfirmationCandleRecord],
        power: dict[str, list[CharacteristicPredictivePower]],
    ) -> list[ConfirmationCandleRecord]:
        weight_maps: dict[str, dict[str, float]] = {}
        for side, rows in power.items():
            weight_maps[side] = {
                row.characteristic: max(row.lift_pct, 0.0)
                for row in rows
                if row.lift_pct >= MIN_FEATURE_LIFT and row.top_cohort_frequency_pct >= 25
            }

        updated: list[ConfirmationCandleRecord] = []
        for record in records:
            weights = weight_maps.get(record.signal_side, {})
            if not weights:
                updated.append(record)
                continue
            hits = [weights[tag] for tag in record.feature_tags if tag in weights]
            score = round(min(sum(hits) / max(sum(weights.values()), 1) * 100, 100), 2)
            updated.append(replace(record, confirmation_score=score))
        return updated

    def _best_confirmation_candles(
        self,
        records: list[ConfirmationCandleRecord],
        scenario: str,
    ) -> list[BestConfirmationCandle]:
        if scenario == MoveScenario.BUY.value:
            bucket = [record for record in records if record.signal_side == "BUY"]
        elif scenario == MoveScenario.SELL.value:
            bucket = [record for record in records if record.signal_side == "SELL"]
        else:
            bucket = [record for record in records if scenario in record.scenarios]
        if not bucket:
            return []

        groups: dict[str, list[ConfirmationCandleRecord]] = defaultdict(list)
        for record in bucket:
            groups[record.feature_signature].append(record)

        ranked: list[BestConfirmationCandle] = []
        total = len(bucket)
        for signature, group in groups.items():
            if len(group) < 2:
                continue
            prob_100 = sum(1 for item in group if item.move_magnitude_points >= 100) / len(group) * 100
            ranked.append(
                BestConfirmationCandle(
                    scenario=scenario,
                    signature=signature,
                    sample_count=len(group),
                    frequency_pct=round((len(group) / total) * 100, 2),
                    average_move_magnitude=round(mean(item.move_magnitude_points for item in group), 2),
                    average_confirmation_score=round(mean(item.confirmation_score for item in group), 2),
                    probability_100_plus_pct=round(prob_100, 2),
                ),
            )

        ranked.sort(
            key=lambda item: (item.sample_count, item.probability_100_plus_pct, item.average_move_magnitude),
            reverse=True,
        )
        for index, item in enumerate(ranked[:10], start=1):
            item.rank = index
        return ranked[:10]

    def _aggregate_profiles(
        self,
        records: list[ConfirmationCandleRecord],
    ) -> dict[str, dict[str, Any]]:
        profiles: dict[str, dict[str, Any]] = {}
        for side in ("BUY", "SELL"):
            bucket = [record for record in records if record.signal_side == side]
            if not bucket:
                continue
            numeric_keys = (
                "body_size_points",
                "upper_wick_size_points",
                "lower_wick_size_points",
                "body_to_wick_ratio",
                "close_location_pct",
                "volume_expansion_ratio",
                "atr_expansion_ratio",
                "level_strength_score",
            )
            profile: dict[str, Any] = {"sample_count": len(bucket)}
            for key in numeric_keys:
                values = [record.candle_characteristics[key] for record in bucket if key in record.candle_characteristics]
                profile[f"average_{key}"] = round(mean(values), 2) if values else None
            profile["strong_body_pct"] = round(
                sum(1 for record in bucket if record.candle_characteristics.get("strong_body")) / len(bucket) * 100,
                2,
            )
            profile["average_confirmation_score"] = round(mean(record.confirmation_score for record in bucket), 2)
            profiles[side] = profile
        return profiles

    def run(self, metadata: dict[str, Any]) -> InstitutionalConfirmationCandleReport:
        started = time.perf_counter()
        records = self._collect_records(metadata)
        power, score_methodology = self._predictive_power(records)
        records = self._apply_scores(records, power)

        freq_by_threshold: dict[str, list[dict[str, Any]]] = {}
        for threshold in (100, 200, 300):
            freq_by_threshold[f"{threshold}_plus"] = [
                item.as_dict() for item in self._characteristic_frequency(records, threshold)
            ]

        matrix = [item.as_dict() for item in self._probability_matrix(records)]

        best_candles = {
            MoveScenario.BUY.value: [item.as_dict() for item in self._best_confirmation_candles(records, MoveScenario.BUY.value)],
            MoveScenario.SELL.value: [item.as_dict() for item in self._best_confirmation_candles(records, MoveScenario.SELL.value)],
            MoveScenario.SUPPORT_BOUNCE.value: [
                item.as_dict() for item in self._best_confirmation_candles(records, MoveScenario.SUPPORT_BOUNCE.value)
            ],
            MoveScenario.RESISTANCE_REJECTION.value: [
                item.as_dict() for item in self._best_confirmation_candles(records, MoveScenario.RESISTANCE_REJECTION.value)
            ],
            MoveScenario.BREAKOUT.value: [
                item.as_dict() for item in self._best_confirmation_candles(records, MoveScenario.BREAKOUT.value)
            ],
            MoveScenario.BREAKDOWN.value: [
                item.as_dict() for item in self._best_confirmation_candles(records, MoveScenario.BREAKDOWN.value)
            ],
        }

        top_matrix = matrix[0] if matrix else None
        score_doc = {
            "methodology": "Top 20% vs bottom 20% move magnitude cohort comparison on candle feature tags",
            "score_range": "0-100",
            "buy_baseline_score": score_methodology.get("BUY", 0.0),
            "sell_baseline_score": score_methodology.get("SELL", 0.0),
            "top_predictive_characteristics": {
                side: [row.as_dict() for row in rows[:8]] for side, rows in power.items()
            },
            "example_probability_entry": top_matrix,
        }

        moves_by_symbol = Counter(record.symbol for record in records)
        moves_by_threshold = Counter(str(record.move_threshold_tier) for record in records)

        conclusions = [
            f"Analyzed {len(records)} trigger candles across {self.symbols}.",
            f"Most common before 100+ moves: {freq_by_threshold.get('100_plus', [{}])[0].get('characteristic', 'N/A')}.",
            f"Top probability signature: {matrix[0]['signature'][:100] if matrix else 'N/A'} "
            f"({matrix[0]['probability_100_plus_pct'] if matrix else 0}% for 100+).",
            f"Best BUY candle: {best_candles['buy'][0]['signature'][:100] if best_candles['buy'] else 'N/A'}.",
            f"Best SELL candle: {best_candles['sell'][0]['signature'][:100] if best_candles['sell'] else 'N/A'}.",
        ]

        return InstitutionalConfirmationCandleReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            move_thresholds_points=list(MOVE_THRESHOLDS),
            total_moves_analyzed=len(records),
            moves_by_symbol=dict(moves_by_symbol),
            moves_by_threshold=dict(moves_by_threshold),
            characteristic_frequency_by_threshold=freq_by_threshold,
            probability_matrix=matrix,
            institutional_confirmation_candle_score=score_doc,
            best_confirmation_candles=best_candles,
            characteristic_predictive_power={
                side: [row.as_dict() for row in rows] for side, rows in power.items()
            },
            aggregate_confirmation_profiles=self._aggregate_profiles(records),
            confirmation_records=[record.as_dict() for record in records],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_confirmation_candle_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> InstitutionalConfirmationCandleReport:
    """Run institutional confirmation candle research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalConfirmationCandleError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalConfirmationCandleResearch(
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
        "Institutional confirmation candle completed: moves=%s",
        report.total_moves_analyzed,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_confirmation_candle_report()
        print("Institutional Confirmation Candle Research Summary")
        print(f"Moves analyzed: {report.total_moves_analyzed}")
        print(f"Symbols: {report.symbols_analyzed}")
        if report.probability_matrix:
            top = report.probability_matrix[0]
            print(
                f"Top probability: {top['probability_100_plus_pct']}% for 100+ move "
                f"({top['sample_count']} samples)",
            )
        if report.best_confirmation_candles.get("buy"):
            print(f"Best BUY candle rank 1: {report.best_confirmation_candles['buy'][0]['signature'][:120]}...")
        if report.best_confirmation_candles.get("sell"):
            print(f"Best SELL candle rank 1: {report.best_confirmation_candles['sell'][0]['signature'][:120]}...")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalConfirmationCandleError as exc:
        logger.error("Institutional confirmation candle error: %s", exc)
        print(f"Institutional confirmation candle error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional confirmation candle error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
