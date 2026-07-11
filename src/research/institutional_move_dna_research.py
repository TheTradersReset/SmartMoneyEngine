"""
Institutional Move DNA research for SmartMoneyEngine.

Starts from real directional expansions (100+/200+/300+/500+ points), finds
the origin bar, analyzes the previous 50 bars, and ranks traits and DNA
patterns by predictive power. Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import DisplacementStrength, LiquidityNarrativeEngine
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
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_move_dna.json"

MOVE_THRESHOLDS = (100, 200, 300, 500)
PRE_MOVE_LOOKBACK = 50
TOP_DNA_COUNT = 20
MIN_PATTERN_SAMPLES = 5
MIN_TRAIT_SAMPLES = 10
MAX_MOVES_PER_TIMEFRAME = 300
FORWARD_SCAN_STEP = 10
MAX_EXPORT_RECORDS = 200
DNA_ARROW = " -> "

DEFAULT_SYMBOLS = ("NIFTY50", "BANKNIFTY", "FINNIFTY")
TIMEFRAMES = ("5M", "15M", "1H")
LEVEL_TOUCH_ATR_RATIO = 0.5
CONSOLIDATION_ATR_RATIO = 1.5
VOLUME_LOOKBACK = 20
LOCATION_LOOKBACK = 200


class InstitutionalMoveDnaError(Exception):
    """Raised when institutional move DNA research fails."""


@dataclass(frozen=True)
class MoveDnaRecord:
    """DNA profile for one directional expansion move."""

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
    trait_tags: tuple[str, ...]
    dna_pattern: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraitPredictivePower:
    """Predictive ranking for one measurable trait."""

    trait: str
    count_100_plus: int
    count_200_plus: int
    count_300_plus: int
    count_500_plus: int
    frequency_100_plus_pct: float
    frequency_200_plus_pct: float
    frequency_300_plus_pct: float
    frequency_500_plus_pct: float
    predictive_power_score: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DnaPatternRank:
    """Ranked DNA pattern with move-size probabilities."""

    pattern: str
    direction: str
    sample_count: int
    probability_100_plus_pct: float
    probability_200_plus_pct: float
    probability_300_plus_pct: float
    probability_500_plus_pct: float
    average_move_magnitude: float
    predictive_power_score: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalMoveDnaReport:
    """Full institutional move DNA research output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    move_thresholds_points: list[int]
    pre_move_lookback_bars: int
    total_moves_analyzed: int
    moves_by_symbol: dict[str, int]
    moves_by_threshold: dict[str, int]
    trait_frequency_by_threshold: dict[str, list[dict[str, Any]]]
    trait_predictive_power: list[dict[str, Any]]
    institutional_move_dna: dict[str, Any]
    top_20_bullish_dna_patterns: list[dict[str, Any]]
    top_20_bearish_dna_patterns: list[dict[str, Any]]
    move_dna_records: list[dict[str, Any]]
    move_dna_records_total: int
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalMoveDnaResearch:
    """Extract and rank institutional move DNA from real expansions."""

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
        self.level_strength_engine = MajorLevelStrengthResearch(research_days=research_days)
        self.narrative_engine = LiquidityNarrativeEngine()
        self.intelligence_engine = MarketIntelligenceEngine()
        self.context_builder = FilterContextBuilder()

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

        return {
            "nearest_support": major_support,
            "nearest_resistance": major_resistance,
            "distance_from_nearest_level": round(nearest_distance, 2) if nearest_distance is not None else None,
        }

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
            "body": body,
            "range": candle_range,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_pct": body / candle_range,
            "wick_pct": (upper_wick + lower_wick) / candle_range,
            "bullish": close > open_price,
            "bearish": close < open_price,
        }

    def _measure_window(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        origin_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        del intel_frame
        start_bar = max(0, origin_bar - PRE_MOVE_LOOKBACK)
        window = frame.iloc[start_bar : origin_bar + 1]
        levels = self._market_levels(frame, origin_bar)
        atr = self._atr(frame, origin_bar)
        touch_threshold = atr * LEVEL_TOUCH_ATR_RATIO

        buy_grabs = sell_grabs = 0
        failed_breakouts = failed_breakdowns = 0
        support_tests = resistance_tests = 0
        choch_count = bos_count = 0
        fvg_created = fvg_reclaimed = False
        ob_reactions = 0
        gap_ups = gap_downs = 0
        round_number_interactions = 0
        volume_ratios: list[float] = []
        wick_points: list[float] = []

        hammer_count = shooting_star_count = 0
        engulfing_count = marubozu_count = 0
        inside_bar_count = outside_bar_count = 0

        close = float(frame.iloc[origin_bar]["Close"])
        support = levels.get("nearest_support")
        resistance = levels.get("nearest_resistance")

        if self.level_strength_engine._round_number_overlap(close):
            round_number_interactions += 1

        for index in range(start_bar, origin_bar + 1):
            row = frame.iloc[index]
            if self._is_active(row.get("Buy_Liquidity_Sweep")):
                buy_grabs += 1
            if self._is_active(row.get("Sell_Liquidity_Sweep")):
                sell_grabs += 1

            high = float(row["High"])
            low = float(row["Low"])
            bar_close = float(row["Close"])

            if resistance is not None and high > resistance and bar_close <= resistance:
                failed_breakouts += 1
            if support is not None and low < support and bar_close >= support:
                failed_breakdowns += 1

            if support is not None and abs(bar_close - support) <= touch_threshold:
                support_tests += 1
            if resistance is not None and abs(resistance - bar_close) <= touch_threshold:
                resistance_tests += 1

            if self._is_active(row.get("Bullish_CHOCH")) or self._is_active(row.get("Bearish_CHOCH")):
                choch_count += 1
            if self._is_active(row.get("Bullish_BOS")) or self._is_active(row.get("Bearish_BOS")):
                bos_count += 1

            bull_fvg = self._is_active(row.get("Bullish_FVG_Top")) and self._is_active(
                row.get("Bullish_FVG_Bottom"),
            )
            bear_fvg = self._is_active(row.get("Bearish_FVG_Top")) and self._is_active(
                row.get("Bearish_FVG_Bottom"),
            )
            if bull_fvg or bear_fvg:
                fvg_created = True
            if direction == "bullish" and bull_fvg:
                bottom = self._to_float(row.get("Bullish_FVG_Bottom"))
                top = self._to_float(row.get("Bullish_FVG_Top"))
                if bottom is not None and top is not None and bottom <= bar_close <= top:
                    fvg_reclaimed = True
            if direction == "bearish" and bear_fvg:
                bottom = self._to_float(row.get("Bearish_FVG_Bottom"))
                top = self._to_float(row.get("Bearish_FVG_Top"))
                if bottom is not None and top is not None and bottom <= bar_close <= top:
                    fvg_reclaimed = True

            if self._is_active(row.get("Bullish_OB_High")) or self._is_active(row.get("Bearish_OB_High")):
                ob_reactions += 1

            if index >= start_bar + 1:
                prev_close = float(frame.iloc[index - 1]["Close"])
                open_price = float(row["Open"])
                gap = open_price - prev_close
                if gap > 0.5:
                    gap_ups += 1
                elif gap < -0.5:
                    gap_downs += 1

            parts = self._candle_parts(row)
            wick_points.append(parts["upper_wick"] + parts["lower_wick"])
            if parts["body"] > 0:
                if parts["lower_wick"] >= 2 * parts["body"]:
                    hammer_count += 1
                if parts["upper_wick"] >= 2 * parts["body"]:
                    shooting_star_count += 1
            if parts["body_pct"] >= 0.85:
                marubozu_count += 1

            volume = self._to_float(row.get("Volume")) or 0.0
            vol_start = max(start_bar, index - VOLUME_LOOKBACK)
            avg_volume = mean(
                self._to_float(frame.iloc[offset].get("Volume")) or 0.0
                for offset in range(vol_start, index)
            ) if index > vol_start else volume
            if avg_volume > 0:
                volume_ratios.append(volume / avg_volume)

        origin_row = frame.iloc[origin_bar]
        origin_parts = self._candle_parts(origin_row)
        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(origin_row, direction)

        rsi_series = enriched["_rsi"] if "_rsi" in enriched.columns else enriched.get("RSI")
        if rsi_series is None:
            rsi_value = 50.0
        else:
            rsi_value = float(rsi_series.iloc[origin_bar]) if pd.notna(rsi_series.iloc[origin_bar]) else 50.0
        rsi_state = self.intelligence_engine._rsi_state(rsi_value)
        rsi_band = (
            "Oversold"
            if rsi_state in {RsiState.OVERSOLD, RsiState.WEAK}
            else "Overbought"
            if rsi_state in {RsiState.OVERBOUGHT, RsiState.STRONG}
            else "Neutral"
        )

        bullish_div = bearish_div = False

        support_level = levels.get("nearest_support")
        resistance_level = levels.get("nearest_resistance")
        if support_level is not None and resistance_level is not None:
            midpoint = (support_level + resistance_level) / 2.0
            premium_discount = "Premium Zone" if close > midpoint else "Discount Zone"
        else:
            premium_discount = "Equilibrium"

        features = LevelStrengthFeatures(
            number_of_touches=support_tests + resistance_tests,
            days_level_survived=0,
            bars_near_level=support_tests + resistance_tests,
            bounce_count=support_tests,
            rejection_count=resistance_tests,
            liquidity_grabs=buy_grabs + sell_grabs,
            equal_highs_lows_nearby=0,
            previous_day_overlap=False,
            weekly_overlap=False,
            monthly_overlap=False,
            demand_supply_zone_overlap=False,
            round_number_overlap=round_number_interactions > 0,
            gap_interactions=gap_ups + gap_downs,
            average_volume_expansion=round(mean(volume_ratios), 2) if volume_ratios else 1.0,
            source_column="Swing_Low" if direction == "bullish" else "Swing_High",
        )
        level_score = self.level_strength_engine._compute_strength_score(features)
        level_category = self.level_strength_engine._classify_strength(level_score)

        consolidation_bars = 0
        for index in range(start_bar, origin_bar + 1):
            local = frame.iloc[max(start_bar, index - 5) : index + 1]
            local_width = float(local["High"].astype(float).max()) - float(local["Low"].astype(float).min())
            if local_width <= atr * CONSOLIDATION_ATR_RATIO:
                consolidation_bars += 1

        return {
            "liquidity_grabs": buy_grabs + sell_grabs,
            "buy_side_grabs": buy_grabs,
            "sell_side_grabs": sell_grabs,
            "false_breakouts": failed_breakouts,
            "false_breakdowns": failed_breakdowns,
            "support_tests": support_tests,
            "resistance_tests": resistance_tests,
            "round_number_interactions": round_number_interactions,
            "choch_count": choch_count,
            "bos_count": bos_count,
            "fvg_created": fvg_created,
            "fvg_reclaimed": fvg_reclaimed,
            "order_block_reactions": ob_reactions,
            "hammer_count": hammer_count,
            "shooting_star_count": shooting_star_count,
            "engulfing_count": engulfing_count,
            "marubozu_count": marubozu_count,
            "inside_bar_count": inside_bar_count,
            "outside_bar_count": outside_bar_count,
            "average_wick_points": round(mean(wick_points), 2) if wick_points else 0.0,
            "largest_wick_points": round(max(wick_points), 2) if wick_points else 0.0,
            "gap_up_count": gap_ups,
            "gap_down_count": gap_downs,
            "volume_expansion_ratio": round(mean(volume_ratios), 2) if volume_ratios else 1.0,
            "rsi_value": round(rsi_value, 2),
            "rsi_band": rsi_band,
            "bullish_divergence": bullish_div,
            "bearish_divergence": bearish_div,
            "premium_discount_zone": premium_discount,
            "level_strength_score": level_score,
            "level_strength_category": level_category,
            "distance_from_nearest_support_resistance": levels.get("distance_from_nearest_level"),
            "consolidation_bars": consolidation_bars,
            "origin_displacement_strength": displacement.value,
            "origin_body_pct": round(origin_parts["body_pct"] * 100, 2),
            "origin_wick_pct": round(origin_parts["wick_pct"] * 100, 2),
        }

    def _build_trait_tags(self, measurements: dict[str, Any], direction: str) -> tuple[str, ...]:
        tags: list[str] = []

        if measurements["liquidity_grabs"] >= 2:
            tags.append("Liquidity Grab x2+")
        elif measurements["liquidity_grabs"] >= 1:
            tags.append("Liquidity Grab x1")
        if measurements["false_breakouts"] >= 2:
            tags.append("False Breakouts x2+")
        if measurements["false_breakdowns"] >= 2:
            tags.append("False Breakdowns x2+")
        if measurements["support_tests"] >= 3:
            tags.append("Support Tests x3+")
        elif measurements["support_tests"] >= 1:
            tags.append("Support Tests")
        if measurements["resistance_tests"] >= 3:
            tags.append("Resistance Tests x3+")
        elif measurements["resistance_tests"] >= 1:
            tags.append("Resistance Tests")
        if measurements["round_number_interactions"] >= 1:
            tags.append("Round Number")
        if measurements["choch_count"] >= 2:
            tags.append("CHOCH x2+")
        elif measurements["choch_count"] >= 1:
            tags.append("CHOCH")
        if measurements["bos_count"] >= 2:
            tags.append("BOS x2+")
        elif measurements["bos_count"] >= 1:
            tags.append("BOS")
        if measurements["fvg_created"]:
            tags.append("FVG Created")
        if measurements["fvg_reclaimed"]:
            tags.append("FVG Reclaim")
        if measurements["order_block_reactions"] >= 1:
            tags.append("Order Block Reaction")
        if measurements["hammer_count"] >= 1 and direction == "bullish":
            tags.append("Hammer")
        if measurements["shooting_star_count"] >= 1 and direction == "bearish":
            tags.append("Shooting Star")
        if measurements["marubozu_count"] >= 2:
            tags.append("Marubozu x2+")
        if measurements["gap_up_count"] >= 1:
            tags.append("Gap Up")
        if measurements["gap_down_count"] >= 1:
            tags.append("Gap Down")
        if measurements["volume_expansion_ratio"] >= 1.5:
            tags.append("Volume Expansion")
        if measurements["average_wick_points"] >= 5:
            tags.append("Large Wicks")
        tags.append(f"RSI:{measurements['rsi_band']}")
        if measurements["bullish_divergence"] and direction == "bullish":
            tags.append("Bullish Divergence")
        if measurements["bearish_divergence"] and direction == "bearish":
            tags.append("Bearish Divergence")
        if measurements["premium_discount_zone"]:
            tags.append(f"Zone:{measurements['premium_discount_zone']}")
        tags.append(f"Level:{measurements['level_strength_category']}")
        if measurements["consolidation_bars"] >= 20:
            tags.append("Consolidation")
        if measurements["origin_displacement_strength"] in {"Strong", "Medium"}:
            tags.append(f"Displacement:{measurements['origin_displacement_strength']}")
        return tuple(tags)

    @staticmethod
    def _build_dna_pattern(tags: tuple[str, ...]) -> str:
        return DNA_ARROW.join(tags) if tags else "No Context"

    def _analyze_move(
        self,
        symbol: str,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        candidate: _CheapMoveCandidate,
        timeframe_label: str,
    ) -> MoveDnaRecord:
        origin_bar = candidate.start_bar
        measurements = self._measure_window(frame, enriched, intel_frame, origin_bar, candidate.direction)
        tags = self._build_trait_tags(measurements, candidate.direction)
        magnitude = candidate.magnitude

        return MoveDnaRecord(
            symbol=symbol,
            timeframe=timeframe_label,
            direction=candidate.direction,
            origin_bar=origin_bar,
            expansion_bar=candidate.expansion_bar,
            origin_timestamp=str(frame.iloc[origin_bar]["Date"]),
            expansion_timestamp=str(frame.iloc[candidate.expansion_bar]["Date"]),
            move_magnitude_points=round(magnitude, 2),
            hit_100_plus=magnitude >= 100,
            hit_200_plus=magnitude >= 200,
            hit_300_plus=magnitude >= 300,
            hit_500_plus=magnitude >= 500,
            measurements=measurements,
            trait_tags=tags,
            dna_pattern=self._build_dna_pattern(tags),
        )

    def _forward_move(self, frame: pd.DataFrame, origin_bar: int, direction: str) -> float:
        end = min(len(frame) - 1, origin_bar + FORWARD_BARS)
        if direction == "bullish":
            origin = float(frame.iloc[origin_bar]["Low"])
            return round(float(frame.iloc[origin_bar : end + 1]["High"].astype(float).max()) - origin, 2)
        origin = float(frame.iloc[origin_bar]["High"])
        return round(origin - float(frame.iloc[origin_bar : end + 1]["Low"].astype(float).min()), 2)

    def _collect_records(self, metadata: dict[str, Any]) -> list[MoveDnaRecord]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        records: list[MoveDnaRecord] = []
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
                intel_frame = self.intelligence_engine.enrich(frame)
                candidates = self._detect_moves(frame)
                if len(candidates) > MAX_MOVES_PER_TIMEFRAME:
                    candidates = sorted(candidates, key=lambda item: item.magnitude, reverse=True)[
                        :MAX_MOVES_PER_TIMEFRAME
                    ]
                logger.info("Move DNA: %s/%s moves=%s", symbol, timeframe_label, len(candidates))

                for candidate in candidates:
                    if candidate.start_bar < PRE_MOVE_LOOKBACK:
                        continue
                    key = (symbol, timeframe_label, candidate.expansion_bar, candidate.direction)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        self._analyze_move(symbol, frame, enriched, intel_frame, candidate, timeframe_label),
                    )

        return records

    @staticmethod
    def _trait_frequency_by_threshold(
        records: list[MoveDnaRecord],
    ) -> dict[str, list[dict[str, Any]]]:
        tiers = {
            "100_plus": [record for record in records if record.hit_100_plus],
            "200_plus": [record for record in records if record.hit_200_plus],
            "300_plus": [record for record in records if record.hit_300_plus],
            "500_plus": [record for record in records if record.hit_500_plus],
        }
        output: dict[str, list[dict[str, Any]]] = {}

        for tier_name, bucket in tiers.items():
            if not bucket:
                output[tier_name] = []
                continue
            trait_counter: Counter[str] = Counter()
            for record in bucket:
                trait_counter.update(record.trait_tags)
            total = len(bucket)
            rows = [
                {
                    "trait": trait,
                    "count": count,
                    "frequency_pct": round(count / total * 100, 2),
                }
                for trait, count in trait_counter.most_common(50)
            ]
            output[tier_name] = rows
        return output

    @staticmethod
    def _rank_trait_predictive_power(records: list[MoveDnaRecord]) -> list[TraitPredictivePower]:
        all_traits: set[str] = set()
        for record in records:
            all_traits.update(record.trait_tags)

        tiers = {
            100: [record for record in records if record.hit_100_plus],
            200: [record for record in records if record.hit_200_plus],
            300: [record for record in records if record.hit_300_plus],
            500: [record for record in records if record.hit_500_plus],
        }
        total_100 = len(tiers[100]) or 1

        ranked: list[TraitPredictivePower] = []
        for trait in sorted(all_traits):
            counts = {
                threshold: sum(1 for record in bucket if trait in record.trait_tags)
                for threshold, bucket in tiers.items()
            }
            if counts[100] < MIN_TRAIT_SAMPLES:
                continue
            freq_100 = round(counts[100] / total_100 * 100, 2)
            freq_200 = round(counts[200] / len(tiers[200]) * 100, 2) if tiers[200] else 0.0
            freq_300 = round(counts[300] / len(tiers[300]) * 100, 2) if tiers[300] else 0.0
            freq_500 = round(counts[500] / len(tiers[500]) * 100, 2) if tiers[500] else 0.0
            power = round(
                (freq_500 * 3 + freq_300 * 2 + freq_200) / max(freq_100, 1),
                4,
            )
            ranked.append(
                TraitPredictivePower(
                    trait=trait,
                    count_100_plus=counts[100],
                    count_200_plus=counts[200],
                    count_300_plus=counts[300],
                    count_500_plus=counts[500],
                    frequency_100_plus_pct=freq_100,
                    frequency_200_plus_pct=freq_200,
                    frequency_300_plus_pct=freq_300,
                    frequency_500_plus_pct=freq_500,
                    predictive_power_score=power,
                ),
            )

        ranked.sort(key=lambda item: (item.predictive_power_score, item.frequency_500_plus_pct), reverse=True)
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked

    @staticmethod
    def _cohort_probabilities(items: list[MoveDnaRecord]) -> dict[str, float]:
        total = len(items)
        return {
            "probability_100_plus_pct": 100.0,
            "probability_200_plus_pct": round(
                sum(1 for item in items if item.hit_200_plus) / total * 100,
                2,
            ),
            "probability_300_plus_pct": round(
                sum(1 for item in items if item.hit_300_plus) / total * 100,
                2,
            ),
            "probability_500_plus_pct": round(
                sum(1 for item in items if item.hit_500_plus) / total * 100,
                2,
            ),
        }

    def _rank_dna_patterns(
        self,
        records: list[MoveDnaRecord],
        direction: str,
    ) -> list[DnaPatternRank]:
        bucket = [record for record in records if record.direction == direction]
        grouped: dict[str, list[MoveDnaRecord]] = defaultdict(list)
        for record in bucket:
            grouped[record.dna_pattern].append(record)

        ranked: list[DnaPatternRank] = []
        for pattern, items in grouped.items():
            if len(items) < MIN_PATTERN_SAMPLES:
                continue

            forward_probs = self._cohort_probabilities(items)

            avg_mag = round(mean(item.move_magnitude_points for item in items), 2)
            power = round(
                forward_probs["probability_500_plus_pct"] * 0.4
                + forward_probs["probability_300_plus_pct"] * 0.3
                + forward_probs["probability_200_plus_pct"] * 0.2
                + forward_probs["probability_100_plus_pct"] * 0.1,
                4,
            )
            ranked.append(
                DnaPatternRank(
                    pattern=pattern,
                    direction=direction,
                    sample_count=len(items),
                    average_move_magnitude=avg_mag,
                    predictive_power_score=power,
                    **forward_probs,
                ),
            )

        ranked.sort(
            key=lambda item: (item.predictive_power_score, item.sample_count, item.average_move_magnitude),
            reverse=True,
        )
        for index, item in enumerate(ranked[:TOP_DNA_COUNT], start=1):
            item.rank = index
        return ranked[:TOP_DNA_COUNT]

    def run(self, metadata: dict[str, Any]) -> InstitutionalMoveDnaReport:
        started = time.perf_counter()
        records = self._collect_records(metadata)

        trait_frequency = self._trait_frequency_by_threshold(records)
        trait_power = self._rank_trait_predictive_power(records)
        bullish_patterns = self._rank_dna_patterns(records, "bullish")
        bearish_patterns = self._rank_dna_patterns(records, "bearish")

        moves_by_symbol = Counter(record.symbol for record in records)
        moves_by_threshold = Counter(
            str(threshold)
            for record in records
            for threshold in MOVE_THRESHOLDS
            if record.move_magnitude_points >= threshold
        )

        top_trait = trait_power[0] if trait_power else None
        top_bull = bullish_patterns[0] if bullish_patterns else None
        top_bear = bearish_patterns[0] if bearish_patterns else None

        conclusions = [
            f"Analyzed {len(records)} real directional expansions across {self.symbols}.",
            f"Unique traits measured: {len({tag for record in records for tag in record.trait_tags})}.",
            (
                f"Top predictive trait: {top_trait.trait} (score {top_trait.predictive_power_score})"
                if top_trait
                else "No traits met minimum sample threshold."
            ),
            (
                f"Top bullish DNA: {top_bull.pattern[:80]} "
                f"({top_bull.probability_200_plus_pct}% reach 200+)"
                if top_bull
                else "No bullish DNA patterns ranked."
            ),
            (
                f"Top bearish DNA: {top_bear.pattern[:80]} "
                f"({top_bear.probability_200_plus_pct}% reach 200+)"
                if top_bear
                else "No bearish DNA patterns ranked."
            ),
        ]

        return InstitutionalMoveDnaReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            move_thresholds_points=list(MOVE_THRESHOLDS),
            pre_move_lookback_bars=PRE_MOVE_LOOKBACK,
            total_moves_analyzed=len(records),
            moves_by_symbol=dict(moves_by_symbol),
            moves_by_threshold=dict(moves_by_threshold),
            trait_frequency_by_threshold=trait_frequency,
            trait_predictive_power=[item.as_dict() for item in trait_power],
            institutional_move_dna={
                "description": "Institutional Move DNA derived from 50-bar pre-origin analysis",
                "probability_note": (
                    "100+ probability is 100% by detection cohort; "
                    "200/300/500 are upgrade rates within matched moves."
                ),
                "top_traits": [item.as_dict() for item in trait_power[:20]],
                "bullish_pattern_count": len(bullish_patterns),
                "bearish_pattern_count": len(bearish_patterns),
            },
            top_20_bullish_dna_patterns=[item.as_dict() for item in bullish_patterns],
            top_20_bearish_dna_patterns=[item.as_dict() for item in bearish_patterns],
            move_dna_records=[record.as_dict() for record in records[:MAX_EXPORT_RECORDS]],
            move_dna_records_total=len(records),
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_move_dna_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> InstitutionalMoveDnaReport:
    """Run institutional move DNA research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalMoveDnaError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalMoveDnaResearch(
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
        "Institutional move DNA completed: moves=%s traits=%s",
        report.total_moves_analyzed,
        len(report.trait_predictive_power),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_move_dna_report()
        print("Institutional Move DNA Research Summary")
        print(f"Moves analyzed: {report.total_moves_analyzed}")
        print(f"Traits ranked: {len(report.trait_predictive_power)}")
        if report.top_20_bullish_dna_patterns:
            top = report.top_20_bullish_dna_patterns[0]
            print(
                f"Top bullish DNA: {top['pattern'][:100]} "
                f"(100+={top['probability_100_plus_pct']}%, 200+={top['probability_200_plus_pct']}%)",
            )
        if report.top_20_bearish_dna_patterns:
            top = report.top_20_bearish_dna_patterns[0]
            print(
                f"Top bearish DNA: {top['pattern'][:100]} "
                f"(100+={top['probability_100_plus_pct']}%, 200+={top['probability_200_plus_pct']}%)",
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalMoveDnaError as exc:
        logger.error("Institutional move DNA error: %s", exc)
        print(f"Institutional move DNA error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional move DNA error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
