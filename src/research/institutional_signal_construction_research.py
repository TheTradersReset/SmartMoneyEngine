"""
Institutional Signal Construction research for SmartMoneyEngine.

Determines the exact sequence of events before the highest-quality BUY and
SELL moves by starting from major directional moves (not Tier-2 logic).
Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.context.liquidity_narrative_engine import DisplacementStrength, LiquidityNarrativeEngine
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
    MIN_MOVE_SEPARATION_BARS,
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
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_signal_construction.json"
DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "outputs" / "pipeline"

MOVE_THRESHOLDS = (50, 100, 150, 200, 300)
PRE_MOVE_LOOKBACK = 50
TOP_NARRATIVE_COUNT = 50
MAX_MOVES_PER_TIMEFRAME = 300
COHORT_TOP_FRACTION = 0.20
COHORT_BOTTOM_FRACTION = 0.20
MIN_FEATURE_LIFT = 8.0
NARRATIVE_ARROW = " -> "

TIMEFRAMES = ("5M", "15M", "1H")


class InstitutionalSignalConstructionError(Exception):
    """Raised when institutional signal construction research fails."""


class MoveScenario(str, Enum):
    MOVE_100_PLUS = "move_100_plus"
    MOVE_200_PLUS = "move_200_plus"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


@dataclass(frozen=True)
class PreMoveContext:
    """Full pre-move market state for one directional move."""

    symbol: str
    timeframe: str
    direction: str
    signal_side: str
    move_magnitude_points: float
    move_threshold_tier: int
    origin_bar: int
    expansion_bar: int
    origin_timestamp: str
    expansion_timestamp: str
    major_level: dict[str, Any]
    liquidity_behavior: dict[str, Any]
    candle_structure: dict[str, Any]
    structure: dict[str, Any]
    fvg_ob: dict[str, Any]
    rsi: dict[str, Any]
    market_context: dict[str, Any]
    expansion_outcome: dict[str, Any]
    narrative: str
    feature_tags: tuple[str, ...]
    scenarios: tuple[str, ...]
    buy_score: float = 0.0
    sell_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NarrativeRank:
    narrative: str
    signal_side: str
    count: int
    frequency_pct: float
    average_move_magnitude: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeaturePredictivePower:
    feature: str
    signal_side: str
    top_cohort_frequency_pct: float
    bottom_cohort_frequency_pct: float
    lift_pct: float
    predictive_power_score: float
    in_top_not_bottom: bool
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalSignalConstructionReport:
    """Aggregate institutional signal construction output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    move_thresholds_points: list[int]
    total_moves_analyzed: int
    moves_by_symbol: dict[str, int]
    moves_by_threshold: dict[str, int]
    top_50_buy_narratives: list[dict[str, Any]]
    top_50_sell_narratives: list[dict[str, Any]]
    common_structures: dict[str, list[dict[str, Any]]]
    production_candidate_features: dict[str, list[dict[str, Any]]]
    feature_predictive_power: dict[str, list[dict[str, Any]]]
    level_strength_matrix: dict[str, dict[str, float | None]]
    recommended_production_signal_architecture: dict[str, Any]
    move_records: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalSignalConstructionResearch:
    """Reconstruct pre-move state from major directional moves."""

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
        self.narrative_engine = LiquidityNarrativeEngine()
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
                symbol = parts[0].upper() if parts else slug.upper()
                if symbol not in symbols and not symbol.endswith("50") and "NIFTY" not in symbol:
                    symbols.append(symbol)
        return symbols or list(SUPPORTED_SYMBOLS)

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
    def _rsi_band(value: float) -> str:
        if value < 30:
            return "<30"
        if value < 40:
            return "30-40"
        if value < 50:
            return "40-50"
        if value < 60:
            return "50-60"
        if value < 70:
            return "60-70"
        return ">70"

    @staticmethod
    def _minutes_per_bar(timeframe_label: str) -> int:
        return TIMEFRAME_MINUTES.get(timeframe_label, 5)

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

    def _find_event_bar(
        self,
        frame: pd.DataFrame,
        start: int,
        end: int,
        columns: tuple[str, ...],
    ) -> int | None:
        for index in range(end, start - 1, -1):
            row = frame.iloc[index]
            if any(self._is_active(row.get(column)) for column in columns):
                return index
        return None

    def _liquidity_type(self, frame: pd.DataFrame, start: int, end: int) -> str:
        buy = any(
            self._is_active(frame.iloc[i].get("Buy_Liquidity_Sweep")) for i in range(start, end + 1)
        )
        sell = any(
            self._is_active(frame.iloc[i].get("Sell_Liquidity_Sweep")) for i in range(start, end + 1)
        )
        if buy and sell:
            return "Both-side sweep"
        if buy:
            return "Buy-side sweep"
        if sell:
            return "Sell-side sweep"
        return "No sweep"

    def _candle_analysis(self, window: pd.DataFrame) -> dict[str, Any]:
        wicks: list[float] = []
        bodies: list[float] = []
        engulfing = hammer = shooting_star = marubozu = inside_bar = outside_bar = 0

        for index in range(len(window)):
            parts = self._candle_parts(window.iloc[index])
            wicks.append(parts["upper_wick"] + parts["lower_wick"])
            bodies.append(parts["body"])
            if parts["body_pct"] >= 0.85:
                marubozu += 1
            if parts["body"] > 0:
                if parts["lower_wick"] >= 2 * parts["body"]:
                    hammer += 1
                if parts["upper_wick"] >= 2 * parts["body"]:
                    shooting_star += 1
            if index >= 1:
                prev = self._candle_parts(window.iloc[index - 1])
                curr = parts
                if curr["high"] <= prev["high"] and curr["low"] >= prev["low"]:
                    inside_bar += 1
                if curr["high"] >= prev["high"] and curr["low"] <= prev["low"]:
                    outside_bar += 1
                if prev["bearish"] and curr["bullish"] and curr["close"] > prev["open"]:
                    engulfing += 1
                if prev["bullish"] and curr["bearish"] and curr["close"] < prev["open"]:
                    engulfing += 1

        confirm_row = window.iloc[-1]
        confirm = self._candle_parts(confirm_row)
        direction = "bullish" if confirm["bullish"] else "bearish"
        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(confirm_row, direction)

        return {
            "largest_wick_points": round(max(wicks), 2) if wicks else 0.0,
            "largest_body_points": round(max(bodies), 2) if bodies else 0.0,
            "engulfing_count": engulfing,
            "hammer_count": hammer,
            "shooting_star_count": shooting_star,
            "marubozu_count": marubozu,
            "inside_bar_count": inside_bar,
            "outside_bar_count": outside_bar,
            "confirmation_candle": {
                "body_pct": round(confirm["body_pct"] * 100, 2),
                "displacement_strength": displacement.value,
                "strong_body": confirm["body_pct"] >= 0.65,
            },
        }

    def _structure_analysis(
        self,
        frame: pd.DataFrame,
        start: int,
        end: int,
        direction: str,
        expansion_bar: int,
        timeframe_label: str,
    ) -> dict[str, Any]:
        choch_count = sum(
            1
            for i in range(start, end + 1)
            if self._is_active(frame.iloc[i].get("Bullish_CHOCH"))
            or self._is_active(frame.iloc[i].get("Bearish_CHOCH"))
        )
        bos_count = sum(
            1
            for i in range(start, end + 1)
            if self._is_active(frame.iloc[i].get("Bullish_BOS"))
            or self._is_active(frame.iloc[i].get("Bearish_BOS"))
        )
        choch_bar = self._find_event_bar(
            frame,
            start,
            end,
            ("Bullish_CHOCH", "Bearish_CHOCH"),
        )
        bos_bar = self._find_event_bar(frame, start, end, ("Bullish_BOS", "Bearish_BOS"))
        choch_to_bos = bos_bar - choch_bar if choch_bar is not None and bos_bar is not None else None
        bos_to_expansion = expansion_bar - bos_bar if bos_bar is not None else None
        minutes = self._minutes_per_bar(timeframe_label)

        return {
            "direction": direction,
            "choch_count": choch_count,
            "bos_count": bos_count,
            "choch_to_bos_bars": choch_to_bos,
            "bos_to_expansion_bars": bos_to_expansion,
            "choch_to_bos_minutes": round(choch_to_bos * minutes, 1) if choch_to_bos is not None else None,
            "bos_to_expansion_minutes": round(bos_to_expansion * minutes, 1) if bos_to_expansion is not None else None,
        }

    def _fvg_ob_analysis(self, frame: pd.DataFrame, start: int, end: int, direction: str) -> dict[str, Any]:
        fvg_created = fvg_reclaimed = False
        ob_present = False
        retest_count = 0
        freshness = "Stale"

        for index in range(start, end + 1):
            row = frame.iloc[index]
            bull_fvg = self._is_active(row.get("Bullish_FVG_Top")) and self._is_active(
                row.get("Bullish_FVG_Bottom"),
            )
            bear_fvg = self._is_active(row.get("Bearish_FVG_Top")) and self._is_active(
                row.get("Bearish_FVG_Bottom"),
            )
            if bull_fvg or bear_fvg:
                fvg_created = True
                if index >= end - 5:
                    freshness = "Fresh"
            if self._is_active(row.get("Bullish_OB_High")) or self._is_active(row.get("Bearish_OB_High")):
                ob_present = True
            close = float(row["Close"])
            if direction == "bullish" and bull_fvg:
                bottom = self._to_float(row.get("Bullish_FVG_Bottom"))
                top = self._to_float(row.get("Bullish_FVG_Top"))
                if bottom is not None and top is not None and bottom <= close <= top:
                    retest_count += 1
                    fvg_reclaimed = True
            if direction == "bearish" and bear_fvg:
                bottom = self._to_float(row.get("Bearish_FVG_Bottom"))
                top = self._to_float(row.get("Bearish_FVG_Top"))
                if bottom is not None and top is not None and bottom <= close <= top:
                    retest_count += 1
                    fvg_reclaimed = True

        return {
            "fvg_created": fvg_created,
            "fvg_reclaimed": fvg_reclaimed,
            "order_block_present": ob_present,
            "retest_count": retest_count,
            "freshness": freshness,
        }

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
        max_move = mfe

        def hit_r(multiple: int) -> bool:
            return mfe >= risk * multiple

        return {
            "maximum_move_points": round(max_move, 2),
            "maximum_favorable_excursion": round(mfe, 2),
            "maximum_adverse_excursion": round(mae, 2),
            "risk_points": round(risk, 2),
            "hit_1r": hit_r(1),
            "hit_2r": hit_r(2),
            "hit_3r": hit_r(3),
            "hit_4r": hit_r(4),
            "hit_5r": hit_r(5),
        }

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

    def _major_level_context(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        origin_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        levels = self._market_levels(frame, origin_bar)
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        close = float(frame.iloc[origin_bar]["Close"])
        atr = self.pressure_engine._atr(frame, origin_bar)

        tests = bars_near = 0
        was_near = False
        start = max(0, origin_bar - PRE_MOVE_LOOKBACK)
        for index in range(start, origin_bar + 1):
            touch_band = atr * 0.5
            row = frame.iloc[index]
            high = float(row["High"])
            low = float(row["Low"])
            c = float(row["Close"])
            target = support if direction == "bullish" else resistance
            if target is None:
                continue
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
        pd_ov, wk_ov, mo_ov = self.level_strength_engine._calendar_overlaps(enriched, origin_bar, close)
        features = LevelStrengthFeatures(
            **{
                **features.as_dict(),
                "previous_day_overlap": pd_ov,
                "weekly_overlap": wk_ov,
                "monthly_overlap": mo_ov,
                "demand_supply_zone_overlap": self.level_strength_engine._demand_supply_overlap(
                    frame,
                    origin_bar,
                    close,
                ),
            },
        )
        score = self.level_strength_engine._compute_strength_score(features)
        category = self.level_strength_engine._classify_strength(score)

        break_prob = None
        matrix_path = PROJECT_ROOT / "outputs" / "research" / "major_level_strength.json"
        if matrix_path.exists():
            matrix = json.loads(matrix_path.read_text(encoding="utf-8")).get("level_strength_matrix", {})
            row = matrix.get(category, {})
            break_prob = row.get("breakdown_probability_pct") if direction == "bearish" else row.get(
                "bounce_probability_pct",
            )

        return {
            "nearest_support": support,
            "nearest_resistance": resistance,
            "level_strength_score": score,
            "level_strength_category": category,
            "number_of_tests": tests,
            "bars_near_level": bars_near,
            "break_probability_pct": break_prob,
        }

    def _liquidity_behavior(
        self,
        frame: pd.DataFrame,
        start: int,
        end: int,
        direction: str,
    ) -> dict[str, Any]:
        failed_breakouts = failed_breakdowns = 0
        stop_hunts: list[float] = []
        atr = self.pressure_engine._atr(frame, end)

        for index in range(start, end + 1):
            row = frame.iloc[index]
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            swing_high = float(frame.iloc[max(0, index - 20) : index + 1]["High"].astype(float).max())
            swing_low = float(frame.iloc[max(0, index - 20) : index + 1]["Low"].astype(float).min())
            if high > swing_high and close <= swing_high:
                failed_breakouts += 1
                stop_hunts.append(high - swing_high)
            if low < swing_low and close >= swing_low:
                failed_breakdowns += 1
                stop_hunts.append(swing_low - low)

        return {
            "sweep_type": self._liquidity_type(frame, start, end),
            "failed_breakouts": failed_breakouts,
            "failed_breakdowns": failed_breakdowns,
            "stop_hunt_size_points": round(mean(stop_hunts), 2) if stop_hunts else 0.0,
            "max_stop_hunt_size_points": round(max(stop_hunts), 2) if stop_hunts else 0.0,
        }

    def _market_context(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        origin_bar: int,
    ) -> dict[str, Any]:
        row = enriched.iloc[origin_bar]
        intel = self.intelligence_engine.evaluate_bar(intel_frame, origin_bar)
        prev_close = float(frame.iloc[origin_bar - 1]["Close"]) if origin_bar > 0 else float(row["Open"])
        gap = float(row["Open"]) - prev_close

        internal = self.liquidity_map_engine._internal_liquidity(frame, origin_bar, float(row["Close"]))
        zone = internal.price_zone

        return {
            "gap_up": gap >= 0.5,
            "gap_down": gap <= -0.5,
            "gap_size_points": round(abs(gap), 2),
            "previous_day_high": self._to_float(row.get("_pdh")),
            "previous_day_low": self._to_float(row.get("_pdl")),
            "previous_week_high": self._to_float(row.get("_pwh")),
            "previous_week_low": self._to_float(row.get("_pwl")),
            "premium_discount": zone,
            "market_location": intel.market_location,
        }

    def _rsi_context(self, enriched: pd.DataFrame, origin_bar: int) -> dict[str, Any]:
        rsi_series = enriched["_rsi"] if "_rsi" in enriched.columns else enriched.get("RSI")
        if rsi_series is None:
            rsi_val = 50.0
        else:
            rsi_val = float(rsi_series.iloc[origin_bar]) if pd.notna(rsi_series.iloc[origin_bar]) else 50.0
        divergences = self.rsi_detector.detect(enriched, origin_bar, enriched["_rsi"])
        div_labels = [item.value for item in divergences]
        return {
            "rsi_level": round(rsi_val, 2),
            "rsi_band": self._rsi_band(rsi_val),
            "bullish_divergence": DivergenceType.BULLISH.value in div_labels
            or DivergenceType.HIDDEN_BULLISH.value in div_labels,
            "bearish_divergence": DivergenceType.BEARISH.value in div_labels
            or DivergenceType.HIDDEN_BEARISH.value in div_labels,
        }

    def _assign_scenarios(
        self,
        magnitude: float,
        direction: str,
        major_level: dict[str, Any],
        liquidity: dict[str, Any],
    ) -> tuple[str, ...]:
        scenarios: list[str] = []
        if magnitude >= 100:
            scenarios.append(MoveScenario.MOVE_100_PLUS.value)
        if magnitude >= 200:
            scenarios.append(MoveScenario.MOVE_200_PLUS.value)

        score = major_level.get("level_strength_score", 0)
        near_support = major_level.get("nearest_support") is not None
        near_resistance = major_level.get("nearest_resistance") is not None

        if direction == "bullish" and near_support:
            scenarios.append(MoveScenario.SUPPORT_BOUNCE.value)
        if direction == "bearish" and near_resistance:
            scenarios.append(MoveScenario.RESISTANCE_REJECTION.value)
        if direction == "bullish" and liquidity.get("failed_breakouts", 0) >= 1:
            scenarios.append(MoveScenario.BREAKOUT.value)
        if direction == "bearish" and liquidity.get("failed_breakdowns", 0) >= 1:
            scenarios.append(MoveScenario.BREAKDOWN.value)
        if score >= 50 and direction == "bullish":
            scenarios.append("strong_level_bullish")
        return tuple(scenarios)

    def _build_feature_tags(self, ctx: dict[str, Any], direction: str) -> tuple[str, ...]:
        tags: list[str] = []
        ml = ctx["major_level"]
        liq = ctx["liquidity"]
        candle = ctx["candle"]
        struct = ctx["structure"]
        fvg = ctx["fvg_ob"]
        rsi = ctx["rsi"]
        mkt = ctx["market_context"]

        tags.append(f"Level:{ml['level_strength_category']}")
        tags.append(f"Sweep:{liq['sweep_type']}")
        if candle["confirmation_candle"]["strong_body"]:
            tags.append("Strong Confirmation Candle")
        if candle["confirmation_candle"]["displacement_strength"] in {"Strong", "Medium"}:
            tags.append(f"Displacement:{candle['confirmation_candle']['displacement_strength']}")
        if struct["choch_count"] >= 1:
            tags.append(f"CHOCH:{direction}")
        if struct["bos_count"] >= 1:
            tags.append(f"BOS:{direction}")
        if fvg["fvg_reclaimed"]:
            tags.append("FVG Reclaim")
        if fvg["order_block_present"]:
            tags.append("Order Block")
        tags.append(f"RSI:{rsi['rsi_band']}")
        if rsi["bullish_divergence"] and direction == "bullish":
            tags.append("Bullish Divergence")
        if rsi["bearish_divergence"] and direction == "bearish":
            tags.append("Bearish Divergence")
        if mkt["premium_discount"]:
            tags.append(f"Zone:{mkt['premium_discount']}")
        if liq["failed_breakouts"] >= 2:
            tags.append("Failed Breakouts")
        if liq["failed_breakdowns"] >= 2:
            tags.append("Failed Breakdowns")
        if candle["engulfing_count"] >= 1:
            tags.append("Engulfing")
        if candle["hammer_count"] >= 1 and direction == "bullish":
            tags.append("Hammer")
        if candle["shooting_star_count"] >= 1 and direction == "bearish":
            tags.append("Shooting Star")
        return tuple(tags)

    def _build_narrative(self, tags: tuple[str, ...]) -> str:
        return NARRATIVE_ARROW.join(tags) if tags else "No Context"

    def _analyze_move(
        self,
        symbol: str,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        candidate: _CheapMoveCandidate,
        timeframe_label: str,
    ) -> PreMoveContext:
        origin_bar = candidate.start_bar
        expansion_bar = candidate.expansion_bar
        start = max(0, origin_bar - PRE_MOVE_LOOKBACK)
        window = frame.iloc[start : origin_bar + 1]
        direction = candidate.direction
        signal_side = "BUY" if direction == "bullish" else "SELL"

        major_level = self._major_level_context(frame, enriched, origin_bar, direction)
        liquidity = self._liquidity_behavior(frame, start, origin_bar, direction)
        candle = self._candle_analysis(window)
        structure = self._structure_analysis(frame, start, origin_bar, direction, expansion_bar, timeframe_label)
        fvg_ob = self._fvg_ob_analysis(frame, start, origin_bar, direction)
        rsi = self._rsi_context(enriched, origin_bar)
        market_context = self._market_context(frame, enriched, intel_frame, origin_bar)
        expansion = self._expansion_outcome(frame, origin_bar, expansion_bar, direction)

        ctx = {
            "major_level": major_level,
            "liquidity": liquidity,
            "candle": candle,
            "structure": structure,
            "fvg_ob": fvg_ob,
            "rsi": rsi,
            "market_context": market_context,
        }
        tags = self._build_feature_tags(ctx, direction)
        scenarios = self._assign_scenarios(candidate.magnitude, direction, major_level, liquidity)

        tier = MOVE_THRESHOLDS[0]
        for threshold in MOVE_THRESHOLDS:
            if candidate.magnitude >= threshold:
                tier = threshold

        return PreMoveContext(
            symbol=symbol,
            timeframe=timeframe_label,
            direction=direction,
            signal_side=signal_side,
            move_magnitude_points=candidate.magnitude,
            move_threshold_tier=tier,
            origin_bar=origin_bar,
            expansion_bar=expansion_bar,
            origin_timestamp=str(frame.iloc[origin_bar]["Date"]),
            expansion_timestamp=str(frame.iloc[expansion_bar]["Date"]),
            major_level=major_level,
            liquidity_behavior=liquidity,
            candle_structure=candle,
            structure=structure,
            fvg_ob=fvg_ob,
            rsi=rsi,
            market_context=market_context,
            expansion_outcome=expansion,
            narrative=self._build_narrative(tags),
            feature_tags=tags,
            scenarios=scenarios,
        )

    def _collect_moves(self, metadata: dict[str, Any]) -> list[PreMoveContext]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        moves: list[PreMoveContext] = []
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
                if len(frame) < FORWARD_BARS + PRE_MOVE_LOOKBACK:
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
                    "Signal construction: %s/%s moves=%s",
                    symbol,
                    timeframe_label,
                    len(candidates),
                )

                for candidate in candidates:
                    if candidate.start_bar < PRE_MOVE_LOOKBACK:
                        continue
                    key = (symbol, timeframe_label, candidate.expansion_bar, candidate.direction)
                    if key in seen:
                        continue
                    seen.add(key)
                    moves.append(
                        self._analyze_move(symbol, frame, enriched, intel_frame, candidate, timeframe_label),
                    )

        return moves

    @staticmethod
    def _rank_narratives(moves: list[PreMoveContext], signal_side: str) -> list[NarrativeRank]:
        bucket = [move for move in moves if move.signal_side == signal_side]
        grouped: dict[str, list[PreMoveContext]] = defaultdict(list)
        for move in bucket:
            grouped[move.narrative].append(move)
        total = len(bucket)
        ranked: list[NarrativeRank] = []
        for narrative, items in grouped.items():
            ranked.append(
                NarrativeRank(
                    narrative=narrative,
                    signal_side=signal_side,
                    count=len(items),
                    frequency_pct=round((len(items) / total) * 100, 2) if total else 0.0,
                    average_move_magnitude=round(mean(item.move_magnitude_points for item in items), 2),
                ),
            )
        ranked.sort(key=lambda item: (item.count, item.average_move_magnitude), reverse=True)
        for index, item in enumerate(ranked[:TOP_NARRATIVE_COUNT], start=1):
            item.rank = index
        return ranked[:TOP_NARRATIVE_COUNT]

    @staticmethod
    def _common_structures(moves: list[PreMoveContext], scenario: str) -> list[dict[str, Any]]:
        bucket = [move for move in moves if scenario in move.scenarios]
        grouped: Counter[str] = Counter(move.narrative for move in bucket)
        total = len(bucket)
        results = []
        for narrative, count in grouped.most_common(20):
            avg_mag = mean(
                move.move_magnitude_points for move in bucket if move.narrative == narrative
            )
            results.append(
                {
                    "narrative": narrative,
                    "count": count,
                    "frequency_pct": round((count / total) * 100, 2) if total else 0.0,
                    "average_move_magnitude": round(avg_mag, 2),
                },
            )
        return results

    def _production_candidate_analysis(
        self,
        moves: list[PreMoveContext],
    ) -> tuple[dict[str, list[FeaturePredictivePower]], dict[str, list[dict[str, Any]]], dict[str, float]]:
        results: dict[str, list[FeaturePredictivePower]] = {}
        candidates: dict[str, list[dict[str, Any]]] = {}
        side_scores: dict[str, float] = {"BUY": 0.0, "SELL": 0.0}

        for signal_side in ("BUY", "SELL"):
            bucket = sorted(
                [move for move in moves if move.signal_side == signal_side],
                key=lambda item: item.move_magnitude_points,
                reverse=True,
            )
            if len(bucket) < 10:
                results[signal_side] = []
                candidates[signal_side] = []
                continue

            top_n = max(1, int(len(bucket) * COHORT_TOP_FRACTION))
            bottom_n = max(1, int(len(bucket) * COHORT_BOTTOM_FRACTION))
            top = bucket[:top_n]
            bottom = bucket[-bottom_n:]

            all_tags = set()
            for move in bucket:
                all_tags.update(move.feature_tags)

            feature_rows: list[FeaturePredictivePower] = []
            production_feats: list[dict[str, Any]] = []

            for tag in sorted(all_tags):
                top_freq = sum(1 for move in top if tag in move.feature_tags) / len(top) * 100
                bottom_freq = sum(1 for move in bottom if tag in move.feature_tags) / len(bottom) * 100
                lift = round(top_freq - bottom_freq, 2)
                power = round(lift * (top_freq / 100) if top_freq else 0.0, 4)
                in_top_not_bottom = top_freq >= 25 and lift >= MIN_FEATURE_LIFT
                feature_rows.append(
                    FeaturePredictivePower(
                        feature=tag,
                        signal_side=signal_side,
                        top_cohort_frequency_pct=round(top_freq, 2),
                        bottom_cohort_frequency_pct=round(bottom_freq, 2),
                        lift_pct=lift,
                        predictive_power_score=power,
                        in_top_not_bottom=in_top_not_bottom,
                    ),
                )
                if in_top_not_bottom:
                    production_feats.append(
                        {
                            "feature": tag,
                            "lift_pct": lift,
                            "top_cohort_frequency_pct": round(top_freq, 2),
                        },
                    )

            feature_rows.sort(key=lambda item: (item.predictive_power_score, item.lift_pct), reverse=True)
            for index, row in enumerate(feature_rows, start=1):
                row.rank = index

            results[signal_side] = feature_rows
            candidates[signal_side] = sorted(production_feats, key=lambda item: item["lift_pct"], reverse=True)

            if production_feats:
                side_scores[signal_side] = round(
                    min(sum(item["lift_pct"] for item in production_feats[:8]) / 8 * 10, 100),
                    2,
                )

        return results, candidates, side_scores

    def _apply_scores(
        self,
        moves: list[PreMoveContext],
        feature_power: dict[str, list[FeaturePredictivePower]],
    ) -> list[PreMoveContext]:
        weight_maps: dict[str, dict[str, float]] = {}
        for side, rows in feature_power.items():
            weight_maps[side] = {
                row.feature: max(row.lift_pct, 0.0) for row in rows if row.in_top_not_bottom
            }

        updated: list[PreMoveContext] = []
        for move in moves:
            weights = weight_maps.get(move.signal_side, {})
            if not weights:
                updated.append(move)
                continue
            hits = [weights[tag] for tag in move.feature_tags if tag in weights]
            score = round(min(sum(hits) / max(sum(weights.values()), 1) * 100, 100), 2)
            if move.signal_side == "BUY":
                updated.append(replace(move, buy_score=score))
            else:
                updated.append(replace(move, sell_score=score))
        return updated

    def _recommended_architecture(
        self,
        candidates: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        buy_chain = [item["feature"] for item in candidates.get("BUY", [])[:8]]
        sell_chain = [item["feature"] for item in candidates.get("SELL", [])[:8]]

        if not buy_chain:
            buy_chain = [
                "Level:Strong",
                "Sweep:Sell-side sweep",
                "Strong Confirmation Candle",
                "Displacement:Strong",
                "CHOCH:bullish",
                "BOS:bullish",
                "FVG Reclaim",
            ]
        if not sell_chain:
            sell_chain = [
                "Level:Strong",
                "Failed Breakouts",
                "Sweep:Buy-side sweep",
                "Displacement:Strong",
                "CHOCH:bearish",
                "BOS:bearish",
                "FVG Reclaim",
            ]

        return {
            "buy_signal_architecture": buy_chain,
            "sell_signal_architecture": sell_chain,
            "buy_formula": " + ".join(buy_chain) + " = BUY Signal",
            "sell_formula": " + ".join(sell_chain) + " = SELL Signal",
        }

    def run(self, metadata: dict[str, Any]) -> InstitutionalSignalConstructionReport:
        started = time.perf_counter()
        moves = self._collect_moves(metadata)
        feature_power, production_candidates, _ = self._production_candidate_analysis(moves)
        moves = self._apply_scores(moves, feature_power)

        buy_narratives = [item.as_dict() for item in self._rank_narratives(moves, "BUY")]
        sell_narratives = [item.as_dict() for item in self._rank_narratives(moves, "SELL")]

        common = {
            MoveScenario.MOVE_100_PLUS.value: self._common_structures(moves, MoveScenario.MOVE_100_PLUS.value),
            MoveScenario.MOVE_200_PLUS.value: self._common_structures(moves, MoveScenario.MOVE_200_PLUS.value),
            MoveScenario.SUPPORT_BOUNCE.value: self._common_structures(moves, MoveScenario.SUPPORT_BOUNCE.value),
            MoveScenario.RESISTANCE_REJECTION.value: self._common_structures(
                moves,
                MoveScenario.RESISTANCE_REJECTION.value,
            ),
            MoveScenario.BREAKOUT.value: self._common_structures(moves, MoveScenario.BREAKOUT.value),
            MoveScenario.BREAKDOWN.value: self._common_structures(moves, MoveScenario.BREAKDOWN.value),
        }

        matrix_path = PROJECT_ROOT / "outputs" / "research" / "major_level_strength.json"
        level_matrix: dict[str, Any] = {}
        if matrix_path.exists():
            level_matrix = json.loads(matrix_path.read_text(encoding="utf-8")).get("level_strength_matrix", {})

        architecture = self._recommended_architecture(production_candidates)

        moves_by_symbol = Counter(move.symbol for move in moves)
        moves_by_threshold = Counter(str(move.move_threshold_tier) for move in moves)

        conclusions = [
            f"Analyzed {len(moves)} directional moves across {self.symbols}.",
            f"Top BUY narrative: {buy_narratives[0]['narrative'] if buy_narratives else 'N/A'}.",
            f"Top SELL narrative: {sell_narratives[0]['narrative'] if sell_narratives else 'N/A'}.",
            f"Recommended BUY architecture: {architecture['buy_formula']}.",
            f"Recommended SELL architecture: {architecture['sell_formula']}.",
        ]

        return InstitutionalSignalConstructionReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            move_thresholds_points=list(MOVE_THRESHOLDS),
            total_moves_analyzed=len(moves),
            moves_by_symbol=dict(moves_by_symbol),
            moves_by_threshold=dict(moves_by_threshold),
            top_50_buy_narratives=buy_narratives,
            top_50_sell_narratives=sell_narratives,
            common_structures=common,
            production_candidate_features={
                side: items for side, items in production_candidates.items()
            },
            feature_predictive_power={
                side: [row.as_dict() for row in rows] for side, rows in feature_power.items()
            },
            level_strength_matrix=level_matrix,
            recommended_production_signal_architecture=architecture,
            move_records=[move.as_dict() for move in moves],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_signal_construction_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> InstitutionalSignalConstructionReport:
    """Run institutional signal construction research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalSignalConstructionError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalSignalConstructionResearch(
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
        "Institutional signal construction completed: moves=%s",
        report.total_moves_analyzed,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_signal_construction_report()
        print("Institutional Signal Construction Research Summary")
        print(f"Moves analyzed: {report.total_moves_analyzed}")
        print(f"Symbols: {report.symbols_analyzed}")
        print(f"Moves by symbol: {report.moves_by_symbol}")
        if report.top_50_buy_narratives:
            print(f"Top BUY narrative: {report.top_50_buy_narratives[0]['narrative'][:120]}...")
        if report.top_50_sell_narratives:
            print(f"Top SELL narrative: {report.top_50_sell_narratives[0]['narrative'][:120]}...")
        arch = report.recommended_production_signal_architecture
        print(f"BUY formula: {arch.get('buy_formula', '')}")
        print(f"SELL formula: {arch.get('sell_formula', '')}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalSignalConstructionError as exc:
        logger.error("Institutional signal construction error: %s", exc)
        print(f"Institutional signal construction error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional signal construction failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
