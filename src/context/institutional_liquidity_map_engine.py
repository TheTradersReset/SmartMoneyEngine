"""
Institutional Liquidity Map Engine for SmartMoneyEngine.

Maps external and internal liquidity, sweep events, objectives, and narrative
for every pipeline candle before signal generation. Research-only context layer;
does not create trades, entries, or production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import (
    DisplacementStrength,
    LiquidityNarrativeEngine,
)
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "context" / "liquidity_map_report.json"

INTERNAL_RANGE_LOOKBACK = 20
DEALING_RANGE_LOOKBACK = 20
VOLUME_LOOKBACK = 20
DISPLACEMENT_AFTER_BARS = 3
SAMPLE_SUMMARY_COUNT = 12
TOP_SWEEP_EXAMPLE_COUNT = 15

REQUIRED_COLUMNS = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Swing_High",
    "Swing_Low",
    "Bullish_BOS",
    "Bearish_BOS",
    "Bullish_CHOCH",
    "Bearish_CHOCH",
    "Equal_High",
    "Equal_Low",
    "Buy_Side_Liquidity",
    "Sell_Side_Liquidity",
    "Buy_Liquidity_Sweep",
    "Sell_Liquidity_Sweep",
    "Liquidity_Strength",
)

SWEEP_SCORE_WEIGHTS = {
    "sweep_size": 25,
    "close_back_into_range": 25,
    "displacement": 20,
    "volume_expansion": 15,
    "structure_shift": 15,
}


class InstitutionalLiquidityMapError(Exception):
    """Raised when institutional liquidity map evaluation fails."""


class SweepClassification(str, Enum):
    """Liquidity sweep strength classification."""

    WEAK = "Weak Sweep"
    MEDIUM = "Medium Sweep"
    STRONG = "Strong Sweep"
    INSTITUTIONAL = "Institutional Sweep"
    NONE = "No Sweep"


class LiquidityTargetType(str, Enum):
    """Whether price is targeting internal or external liquidity."""

    INTERNAL = "Internal Liquidity"
    EXTERNAL = "External Liquidity"
    NONE = "No Clear Target"


class ObjectiveDirection(str, Enum):
    """Direction of liquidity objective."""

    BULLISH = "Bullish Objective"
    BEARISH = "Bearish Objective"
    NEUTRAL = "Neutral"


@dataclass(frozen=True)
class ExternalLiquidityMap:
    """External liquidity reference levels for one candle."""

    previous_day_high: float | None
    previous_day_low: float | None
    previous_week_high: float | None
    previous_week_low: float | None
    previous_month_high: float | None
    previous_month_low: float | None
    equal_high: float | None
    equal_low: float | None
    nearest_external_level: str | None
    distance_to_nearest_external: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InternalLiquidityMap:
    """Internal liquidity and dealing range context."""

    internal_swing_high: float | None
    internal_swing_low: float | None
    range_high: float | None
    range_low: float | None
    equilibrium: float | None
    price_zone: str
    premium_zone_top: float | None
    premium_zone_bottom: float | None
    discount_zone_top: float | None
    discount_zone_bottom: float | None
    active_buy_side_pool: float | None
    active_sell_side_pool: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiquidityEventMap:
    """Sweep event details for one candle."""

    event_type: str
    sweep_side: str | None
    sweep_price: float | None
    swept_level: float | None
    sweep_size_points: float | None
    wick_pct: float | None
    body_pct: float | None
    close_location_pct: float | None
    close_back_into_range: bool
    displacement_after_sweep: str
    classification: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiquidityObjectiveMap:
    """Liquidity raid objective for one candle."""

    target_type: str
    direction: str
    target_level: float | None
    target_label: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SweepQualityComponents:
    """Sub-scores for sweep quality."""

    sweep_size: float
    close_back_into_range: float
    displacement: float
    volume_expansion: float
    structure_shift: float

    @property
    def total(self) -> float:
        return (
            self.sweep_size * SWEEP_SCORE_WEIGHTS["sweep_size"]
            + self.close_back_into_range * SWEEP_SCORE_WEIGHTS["close_back_into_range"]
            + self.displacement * SWEEP_SCORE_WEIGHTS["displacement"]
            + self.volume_expansion * SWEEP_SCORE_WEIGHTS["volume_expansion"]
            + self.structure_shift * SWEEP_SCORE_WEIGHTS["structure_shift"]
        ) / 100.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CandleLiquidityMap:
    """Full institutional liquidity map for one candle."""

    index: int
    timestamp: str
    close: float
    external_liquidity: dict[str, Any]
    internal_liquidity: dict[str, Any]
    liquidity_event: dict[str, Any]
    liquidity_objective: dict[str, Any]
    sweep_quality_score: float
    sweep_quality_components: dict[str, float]
    market_narrative: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalLiquidityMapReport:
    """Aggregate institutional liquidity map report."""

    symbol: str
    timeframe: str
    source_csv: str
    total_candles: int
    average_sweep_quality_score: float
    sweep_event_distribution: dict[str, int]
    sweep_classification_distribution: dict[str, int]
    objective_target_distribution: dict[str, int]
    objective_direction_distribution: dict[str, int]
    price_zone_distribution: dict[str, int]
    external_proximity_distribution: dict[str, int]
    top_sweep_examples: list[dict[str, Any]]
    sample_candle_maps: list[dict[str, Any]]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalLiquidityMapEngine:
    """
    Evaluate institutional liquidity map for every pipeline candle.

    Uses existing SMC pipeline columns only. Does not generate trades or entries.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5M",
        internal_range_lookback: int = INTERNAL_RANGE_LOOKBACK,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.internal_range_lookback = internal_range_lookback
        self.narrative_helper = LiquidityNarrativeEngine(
            symbol=symbol,
            timeframe=timeframe,
        )

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        return LiquidityNarrativeEngine._to_float(value)

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise InstitutionalLiquidityMapError(
                f"Pipeline frame missing liquidity map columns: {missing}",
            )

    @staticmethod
    def _ensure_timestamps(frame: pd.DataFrame) -> pd.Series:
        timestamps = pd.to_datetime(frame["Date"], errors="coerce")
        if timestamps.dt.tz is None:
            return timestamps.dt.tz_localize("Asia/Kolkata")
        return timestamps.dt.tz_convert("Asia/Kolkata")

    def _attach_calendar_levels(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Precompute PDH/PDL, PWH/PWL, and prior month levels."""
        working = frame.copy()
        timestamps = self._ensure_timestamps(working)
        working["_ts"] = timestamps
        working["_date"] = timestamps.dt.date
        working["_week"] = timestamps.dt.to_period("W-SUN")
        working["_month"] = timestamps.dt.to_period("M")

        daily = (
            working.groupby("_date", sort=True)
            .agg(_day_high=("High", "max"), _day_low=("Low", "min"))
            .reset_index()
        )
        daily["_pdh"] = daily["_day_high"].shift(1)
        daily["_pdl"] = daily["_day_low"].shift(1)
        working = working.merge(
            daily[["_date", "_pdh", "_pdl"]],
            on="_date",
            how="left",
        )

        weekly = (
            working.groupby("_week", sort=True)
            .agg(_week_high=("High", "max"), _week_low=("Low", "min"))
            .reset_index()
        )
        weekly["_pwh"] = weekly["_week_high"].shift(1)
        weekly["_pwl"] = weekly["_week_low"].shift(1)
        working = working.merge(
            weekly[["_week", "_pwh", "_pwl"]],
            on="_week",
            how="left",
        )

        monthly = (
            working.groupby("_month", sort=True)
            .agg(_month_high=("High", "max"), _month_low=("Low", "min"))
            .reset_index()
        )
        monthly["_pmh"] = monthly["_month_high"].shift(1)
        monthly["_pml"] = monthly["_month_low"].shift(1)
        working = working.merge(
            monthly[["_month", "_pmh", "_pml"]],
            on="_month",
            how="left",
        )

        return working

    @staticmethod
    def _latest_active_value(window: pd.DataFrame, column: str) -> float | None:
        for offset in range(len(window) - 1, -1, -1):
            value = LiquidityNarrativeEngine._to_float(window.iloc[offset].get(column))
            if value is not None:
                return value
        return None

    def _external_liquidity(
        self,
        enriched: pd.DataFrame,
        index: int,
        close: float,
    ) -> ExternalLiquidityMap:
        row = enriched.iloc[index]
        pdh = self._to_float(row.get("_pdh"))
        pdl = self._to_float(row.get("_pdl"))
        pwh = self._to_float(row.get("_pwh"))
        pwl = self._to_float(row.get("_pwl"))
        pmh = self._to_float(row.get("_pmh"))
        pml = self._to_float(row.get("_pml"))
        equal_high = self._to_float(row.get("Equal_High"))
        equal_low = self._to_float(row.get("Equal_Low"))
        if equal_high is None:
            equal_high = self._to_float(row.get("Buy_Side_Liquidity"))
        if equal_low is None:
            equal_low = self._to_float(row.get("Sell_Side_Liquidity"))

        candidates: list[tuple[str, float]] = []
        for label, level in (
            ("PDH", pdh),
            ("PDL", pdl),
            ("PWH", pwh),
            ("PWL", pwl),
            ("Monthly High", pmh),
            ("Monthly Low", pml),
            ("Equal High", equal_high),
            ("Equal Low", equal_low),
        ):
            if level is not None:
                candidates.append((label, level))

        nearest_label: str | None = None
        nearest_distance: float | None = None
        if candidates:
            nearest_label, nearest_level = min(
                candidates,
                key=lambda item: abs(close - item[1]),
            )
            nearest_distance = round(abs(close - nearest_level), 2)

        return ExternalLiquidityMap(
            previous_day_high=round(pdh, 2) if pdh is not None else None,
            previous_day_low=round(pdl, 2) if pdl is not None else None,
            previous_week_high=round(pwh, 2) if pwh is not None else None,
            previous_week_low=round(pwl, 2) if pwl is not None else None,
            previous_month_high=round(pmh, 2) if pmh is not None else None,
            previous_month_low=round(pml, 2) if pml is not None else None,
            equal_high=round(equal_high, 2) if equal_high is not None else None,
            equal_low=round(equal_low, 2) if equal_low is not None else None,
            nearest_external_level=nearest_label,
            distance_to_nearest_external=nearest_distance,
        )

    def _internal_liquidity(
        self,
        frame: pd.DataFrame,
        index: int,
        close: float,
    ) -> InternalLiquidityMap:
        start = max(0, index - self.internal_range_lookback)
        window = frame.iloc[start : index + 1]
        dealing_start = max(0, index - DEALING_RANGE_LOOKBACK)
        dealing_window = frame.iloc[dealing_start : index + 1]

        swing_high = self._latest_active_value(dealing_window, "Swing_High")
        swing_low = self._latest_active_value(dealing_window, "Swing_Low")
        if swing_high is None:
            swing_high = float(dealing_window["High"].astype(float).max())
        if swing_low is None:
            swing_low = float(dealing_window["Low"].astype(float).min())

        range_high = round(float(window["High"].astype(float).max()), 2)
        range_low = round(float(window["Low"].astype(float).min()), 2)
        equilibrium = round((swing_high + swing_low) / 2.0, 2)
        range_size = max(swing_high - swing_low, 0.01)

        if close > equilibrium + range_size * 0.05:
            price_zone = "Premium Zone"
        elif close < equilibrium - range_size * 0.05:
            price_zone = "Discount Zone"
        else:
            price_zone = "Equilibrium"

        row = frame.iloc[index]
        return InternalLiquidityMap(
            internal_swing_high=round(swing_high, 2),
            internal_swing_low=round(swing_low, 2),
            range_high=range_high,
            range_low=range_low,
            equilibrium=equilibrium,
            price_zone=price_zone,
            premium_zone_top=round(swing_high, 2),
            premium_zone_bottom=equilibrium,
            discount_zone_top=equilibrium,
            discount_zone_bottom=round(swing_low, 2),
            active_buy_side_pool=self._to_float(row.get("Buy_Side_Liquidity")),
            active_sell_side_pool=self._to_float(row.get("Sell_Side_Liquidity")),
        )

    @staticmethod
    def _candle_geometry(row: pd.Series) -> dict[str, float | bool]:
        open_price = InstitutionalLiquidityMapEngine._to_float(row.get("Open")) or 0.0
        high = InstitutionalLiquidityMapEngine._to_float(row.get("High")) or 0.0
        low = InstitutionalLiquidityMapEngine._to_float(row.get("Low")) or 0.0
        close = InstitutionalLiquidityMapEngine._to_float(row.get("Close")) or 0.0
        candle_range = max(high - low, 0.0001)
        body = abs(close - open_price)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        return {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "range": candle_range,
            "body_pct": round(body / candle_range * 100, 2),
            "upper_wick_pct": round(upper_wick / candle_range * 100, 2),
            "lower_wick_pct": round(lower_wick / candle_range * 100, 2),
            "close_location_pct": round((close - low) / candle_range * 100, 2),
            "bullish_close": close > open_price,
            "bearish_close": close < open_price,
        }

    def _displacement_after(
        self,
        frame: pd.DataFrame,
        index: int,
        direction: str,
    ) -> DisplacementStrength:
        end = min(len(frame) - 1, index + DISPLACEMENT_AFTER_BARS)
        strongest = DisplacementStrength.NONE
        rank = {
            DisplacementStrength.NONE: 0,
            DisplacementStrength.WEAK: 1,
            DisplacementStrength.MEDIUM: 2,
            DisplacementStrength.STRONG: 3,
        }
        for bar in range(index, end + 1):
            strength = LiquidityNarrativeEngine._displacement_strength_for_bar(
                frame.iloc[bar],
                direction,
            )
            if rank[strength] > rank[strongest]:
                strongest = strength
        return strongest

    def _classify_sweep(
        self,
        geometry: dict[str, float | bool],
        close_back: bool,
        displacement: DisplacementStrength,
        sweep_side: str,
    ) -> SweepClassification:
        if sweep_side == "Buy Side Sweep":
            wick_pct = float(geometry["upper_wick_pct"])
            rejection = bool(geometry["bearish_close"]) and float(geometry["close_location_pct"]) <= 55
        else:
            wick_pct = float(geometry["lower_wick_pct"])
            rejection = bool(geometry["bullish_close"]) and float(geometry["close_location_pct"]) >= 45

        if not close_back:
            return SweepClassification.WEAK

        if (
            wick_pct >= 50
            and rejection
            and displacement == DisplacementStrength.STRONG
            and float(geometry["body_pct"]) <= 35
        ):
            return SweepClassification.INSTITUTIONAL
        if wick_pct >= 40 and rejection and displacement in {
            DisplacementStrength.STRONG,
            DisplacementStrength.MEDIUM,
        }:
            return SweepClassification.STRONG
        if wick_pct >= 25 and close_back:
            return SweepClassification.MEDIUM
        return SweepClassification.WEAK

    def _liquidity_event(
        self,
        frame: pd.DataFrame,
        index: int,
        row: pd.Series,
    ) -> LiquidityEventMap:
        buy_sweep = self._is_active(row.get("Buy_Liquidity_Sweep"))
        sell_sweep = self._is_active(row.get("Sell_Liquidity_Sweep"))
        geometry = self._candle_geometry(row)

        if buy_sweep and sell_sweep:
            event_type = "Both Side Sweeps"
            sweep_side = "Both"
            sweep_price = float(geometry["high"])
            swept_level = self._to_float(row.get("Buy_Side_Liquidity"))
            direction = "bearish"
            wick_pct = max(float(geometry["upper_wick_pct"]), float(geometry["lower_wick_pct"]))
            close_back = (
                swept_level is not None
                and float(geometry["close"]) < swept_level
                and float(geometry["close"]) > (self._to_float(row.get("Sell_Side_Liquidity")) or 0)
            )
        elif buy_sweep:
            event_type = "Buy Side Sweep"
            sweep_side = "Buy Side Sweep"
            sweep_price = self._to_float(row.get("Buy_Liquidity_Sweep")) or float(geometry["high"])
            swept_level = self._to_float(row.get("Buy_Side_Liquidity")) or sweep_price
            direction = "bearish"
            wick_pct = float(geometry["upper_wick_pct"])
            close_back = swept_level is not None and float(geometry["close"]) < swept_level
        elif sell_sweep:
            event_type = "Sell Side Sweep"
            sweep_side = "Sell Side Sweep"
            sweep_price = self._to_float(row.get("Sell_Liquidity_Sweep")) or float(geometry["low"])
            swept_level = self._to_float(row.get("Sell_Side_Liquidity")) or sweep_price
            direction = "bullish"
            wick_pct = float(geometry["lower_wick_pct"])
            close_back = swept_level is not None and float(geometry["close"]) > swept_level
        else:
            return LiquidityEventMap(
                event_type="No Sweep",
                sweep_side=None,
                sweep_price=None,
                swept_level=None,
                sweep_size_points=None,
                wick_pct=None,
                body_pct=float(geometry["body_pct"]),
                close_location_pct=float(geometry["close_location_pct"]),
                close_back_into_range=False,
                displacement_after_sweep=DisplacementStrength.NONE.value,
                classification=SweepClassification.NONE.value,
            )

        sweep_size = (
            round(float(geometry["high"]) - swept_level, 2)
            if buy_sweep and swept_level is not None
            else round(swept_level - float(geometry["low"]), 2)
            if sell_sweep and swept_level is not None
            else None
        )
        displacement = self._displacement_after(frame, index, direction)
        classification = self._classify_sweep(geometry, close_back, displacement, sweep_side or "")

        return LiquidityEventMap(
            event_type=event_type,
            sweep_side=sweep_side,
            sweep_price=round(sweep_price, 2) if sweep_price is not None else None,
            swept_level=round(swept_level, 2) if swept_level is not None else None,
            sweep_size_points=sweep_size,
            wick_pct=wick_pct,
            body_pct=float(geometry["body_pct"]),
            close_location_pct=float(geometry["close_location_pct"]),
            close_back_into_range=close_back,
            displacement_after_sweep=displacement.value,
            classification=classification.value,
        )

    @staticmethod
    def _is_external_level(label: str | None) -> bool:
        if label is None:
            return False
        return label in {
            "PDH",
            "PDL",
            "PWH",
            "PWL",
            "Monthly High",
            "Monthly Low",
        }

    def _liquidity_objective(
        self,
        external: ExternalLiquidityMap,
        internal: InternalLiquidityMap,
        event: LiquidityEventMap,
        close: float,
    ) -> LiquidityObjectiveMap:
        if event.event_type == "No Sweep":
            if internal.price_zone == "Discount Zone":
                return LiquidityObjectiveMap(
                    target_type=LiquidityTargetType.INTERNAL.value,
                    direction=ObjectiveDirection.BULLISH.value,
                    target_level=internal.premium_zone_top,
                    target_label="Internal Premium / Range High",
                )
            if internal.price_zone == "Premium Zone":
                return LiquidityObjectiveMap(
                    target_type=LiquidityTargetType.INTERNAL.value,
                    direction=ObjectiveDirection.BEARISH.value,
                    target_level=internal.discount_zone_bottom,
                    target_label="Internal Discount / Range Low",
                )
            return LiquidityObjectiveMap(
                target_type=LiquidityTargetType.NONE.value,
                direction=ObjectiveDirection.NEUTRAL.value,
                target_level=None,
                target_label=None,
            )

        if event.event_type in {"Buy Side Sweep", "Both Side Sweeps"}:
            direction = ObjectiveDirection.BEARISH.value
            if external.previous_day_low is not None and close > external.previous_day_low:
                return LiquidityObjectiveMap(
                    target_type=LiquidityTargetType.EXTERNAL.value,
                    direction=direction,
                    target_level=external.previous_day_low,
                    target_label="Previous Day Low",
                )
            if internal.active_sell_side_pool is not None:
                return LiquidityObjectiveMap(
                    target_type=LiquidityTargetType.INTERNAL.value,
                    direction=direction,
                    target_level=internal.active_sell_side_pool,
                    target_label="Sell-Side Internal Liquidity",
                )
            return LiquidityObjectiveMap(
                target_type=LiquidityTargetType.INTERNAL.value,
                direction=direction,
                target_level=internal.range_low,
                target_label="Internal Range Low",
            )

        direction = ObjectiveDirection.BULLISH.value
        if external.previous_day_high is not None and close < external.previous_day_high:
            return LiquidityObjectiveMap(
                target_type=LiquidityTargetType.EXTERNAL.value,
                direction=direction,
                target_level=external.previous_day_high,
                target_label="Previous Day High",
            )
        if internal.active_buy_side_pool is not None:
            return LiquidityObjectiveMap(
                target_type=LiquidityTargetType.INTERNAL.value,
                direction=direction,
                target_level=internal.active_buy_side_pool,
                target_label="Buy-Side Internal Liquidity",
            )
        return LiquidityObjectiveMap(
            target_type=LiquidityTargetType.INTERNAL.value,
            direction=direction,
            target_level=internal.range_high,
            target_label="Internal Range High",
        )

    def _volume_expansion_score(self, frame: pd.DataFrame, index: int) -> float:
        start = max(0, index - VOLUME_LOOKBACK + 1)
        volumes = frame.iloc[start : index + 1]["Volume"].astype(float)
        if volumes.empty:
            return 0.0
        current = float(volumes.iloc[-1])
        baseline = float(volumes.mean())
        if baseline <= 0:
            return 0.0
        ratio = current / baseline
        if ratio >= 1.8:
            return 100.0
        if ratio >= 1.5:
            return 80.0
        if ratio >= 1.2:
            return 55.0
        if ratio >= 1.0:
            return 35.0
        return 15.0

    def _structure_shift_score(self, frame: pd.DataFrame, index: int) -> float:
        start = max(0, index - DISPLACEMENT_AFTER_BARS)
        window = frame.iloc[start : index + 1]
        has_choch = any(
            self._is_active(row.get("Bullish_CHOCH")) or self._is_active(row.get("Bearish_CHOCH"))
            for _, row in window.iterrows()
        )
        has_bos = any(
            self._is_active(row.get("Bullish_BOS")) or self._is_active(row.get("Bearish_BOS"))
            for _, row in window.iterrows()
        )
        if has_choch and has_bos:
            return 100.0
        if has_choch:
            return 85.0
        if has_bos:
            return 70.0
        return 20.0

    def _sweep_quality_score(
        self,
        frame: pd.DataFrame,
        index: int,
        event: LiquidityEventMap,
    ) -> tuple[float, SweepQualityComponents]:
        if event.event_type == "No Sweep":
            components = SweepQualityComponents(0, 0, 0, 0, 0)
            return 0.0, components

        size_score = 20.0
        if event.sweep_size_points is not None:
            if event.sweep_size_points >= 15:
                size_score = 100.0
            elif event.sweep_size_points >= 8:
                size_score = 75.0
            elif event.sweep_size_points >= 3:
                size_score = 50.0

        close_back_score = 100.0 if event.close_back_into_range else 15.0
        displacement_score = {
            DisplacementStrength.STRONG.value: 100.0,
            DisplacementStrength.MEDIUM.value: 70.0,
            DisplacementStrength.WEAK.value: 40.0,
            DisplacementStrength.NONE.value: 10.0,
        }[event.displacement_after_sweep]
        volume_score = self._volume_expansion_score(frame, index)
        structure_score = self._structure_shift_score(frame, index)

        components = SweepQualityComponents(
            sweep_size=size_score,
            close_back_into_range=close_back_score,
            displacement=displacement_score,
            volume_expansion=volume_score,
            structure_shift=structure_score,
        )
        return round(components.total, 2), components

    @staticmethod
    def _build_market_narrative(
        external: ExternalLiquidityMap,
        internal: InternalLiquidityMap,
        event: LiquidityEventMap,
        objective: LiquidityObjectiveMap,
    ) -> str:
        sentences: list[str] = []

        if event.event_type == "Buy Side Sweep":
            if external.previous_day_high is not None:
                sentences.append("Previous Day High swept.")
            else:
                sentences.append("Buy-side liquidity swept above internal highs.")
            if event.classification in {
                SweepClassification.STRONG.value,
                SweepClassification.INSTITUTIONAL.value,
            }:
                sentences.append(f"{event.classification.replace(' Sweep', '')} bearish rejection.")
            if event.displacement_after_sweep != DisplacementStrength.NONE.value:
                sentences.append(
                    f"Bearish displacement {event.displacement_after_sweep.lower()} confirmed.",
                )
            if objective.target_label:
                sentences.append(
                    f"Price likely targeting {objective.target_label.lower()} "
                    f"({objective.direction.lower()}).",
                )
        elif event.event_type == "Sell Side Sweep":
            if external.previous_day_low is not None:
                sentences.append("Previous Day Low swept.")
            else:
                sentences.append("Sell-side liquidity swept below internal lows.")
            if event.classification in {
                SweepClassification.STRONG.value,
                SweepClassification.INSTITUTIONAL.value,
            }:
                sentences.append(f"{event.classification.replace(' Sweep', '')} bullish rejection.")
            if event.displacement_after_sweep != DisplacementStrength.NONE.value:
                sentences.append(
                    f"Bullish displacement {event.displacement_after_sweep.lower()} confirmed.",
                )
            if objective.target_label:
                sentences.append(
                    f"Price likely targeting {objective.target_label.lower()} "
                    f"({objective.direction.lower()}).",
                )
        elif event.event_type == "Both Side Sweeps":
            sentences.append("Both buy-side and sell-side liquidity swept in the lookback window.")
        else:
            sentences.append(
                f"Price trading in {internal.price_zone.lower()} with no fresh sweep on this candle.",
            )
            if objective.target_label:
                sentences.append(
                    f"Bias toward {objective.target_label.lower()} ({objective.direction.lower()}).",
                )

        if not sentences:
            return "No institutional liquidity narrative available for this candle."
        return " ".join(sentences)

    def evaluate_bar(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        index: int,
    ) -> CandleLiquidityMap:
        """Build institutional liquidity map for one candle."""
        row = frame.iloc[index]
        close = self._to_float(row.get("Close")) or 0.0
        external = self._external_liquidity(enriched, index, close)
        internal = self._internal_liquidity(frame, index, close)
        event = self._liquidity_event(frame, index, row)
        objective = self._liquidity_objective(external, internal, event, close)
        score, components = self._sweep_quality_score(frame, index, event)
        narrative = self._build_market_narrative(external, internal, event, objective)

        return CandleLiquidityMap(
            index=index,
            timestamp=str(row.get("Date")),
            close=round(close, 2),
            external_liquidity=external.as_dict(),
            internal_liquidity=internal.as_dict(),
            liquidity_event=event.as_dict(),
            liquidity_objective=objective.as_dict(),
            sweep_quality_score=score,
            sweep_quality_components=components.as_dict(),
            market_narrative=narrative,
        )

    def evaluate(self, frame: pd.DataFrame) -> list[CandleLiquidityMap]:
        """Evaluate institutional liquidity map for every candle."""
        self._validate_frame(frame)
        working = frame.reset_index(drop=True)
        enriched = self._attach_calendar_levels(working)
        return [
            self.evaluate_bar(working, enriched, index)
            for index in range(len(working))
        ]

    @staticmethod
    def _external_proximity_label(distance: float | None) -> str:
        if distance is None:
            return "Unknown"
        if distance <= 20:
            return "Near External (<=20 pts)"
        if distance <= 50:
            return "Moderate External (20-50 pts)"
        return "Far From External (>50 pts)"

    def build_report(
        self,
        evaluations: list[CandleLiquidityMap],
        source_csv: Path | str,
        execution_time_seconds: float,
    ) -> InstitutionalLiquidityMapReport:
        """Build aggregate report from per-candle maps."""
        sweep_events = Counter(item.liquidity_event["event_type"] for item in evaluations)
        sweep_classes = Counter(item.liquidity_event["classification"] for item in evaluations)
        target_types = Counter(item.liquidity_objective["target_type"] for item in evaluations)
        directions = Counter(item.liquidity_objective["direction"] for item in evaluations)
        price_zones = Counter(item.internal_liquidity["price_zone"] for item in evaluations)
        external_proximity = Counter(
            self._external_proximity_label(item.external_liquidity["distance_to_nearest_external"])
            for item in evaluations
        )

        sweep_examples = sorted(
            [item for item in evaluations if item.liquidity_event["event_type"] != "No Sweep"],
            key=lambda item: item.sweep_quality_score,
            reverse=True,
        )[:TOP_SWEEP_EXAMPLE_COUNT]

        if evaluations:
            step = max(1, len(evaluations) // SAMPLE_SUMMARY_COUNT)
            sample_indices = list(range(0, len(evaluations), step))[:SAMPLE_SUMMARY_COUNT]
        else:
            sample_indices = []

        sweep_scores = [item.sweep_quality_score for item in evaluations if item.sweep_quality_score > 0]
        average_score = round(mean(sweep_scores), 2) if sweep_scores else 0.0

        return InstitutionalLiquidityMapReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=str(source_csv),
            total_candles=len(evaluations),
            average_sweep_quality_score=average_score,
            sweep_event_distribution=dict(sorted(sweep_events.items())),
            sweep_classification_distribution=dict(sorted(sweep_classes.items())),
            objective_target_distribution=dict(sorted(target_types.items())),
            objective_direction_distribution=dict(sorted(directions.items())),
            price_zone_distribution=dict(sorted(price_zones.items())),
            external_proximity_distribution=dict(sorted(external_proximity.items())),
            top_sweep_examples=[item.as_dict() for item in sweep_examples],
            sample_candle_maps=[evaluations[index].as_dict() for index in sample_indices],
            execution_time_seconds=round(execution_time_seconds, 3),
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    return value


def generate_liquidity_map_report(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5M",
) -> InstitutionalLiquidityMapReport:
    """Evaluate institutional liquidity map and export JSON report."""
    source = Path(pipeline_csv or DEFAULT_PIPELINE_CSV)
    if not source.exists():
        raise InstitutionalLiquidityMapError(f"Pipeline CSV not found: {source}")

    started = time.perf_counter()
    frame = pd.read_csv(source)
    engine = InstitutionalLiquidityMapEngine(symbol=symbol, timeframe=timeframe)
    evaluations = engine.evaluate(frame)
    report = engine.build_report(
        evaluations,
        source_csv=source,
        execution_time_seconds=time.perf_counter() - started,
    )

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Institutional liquidity map completed: candles=%s avg_sweep_score=%s",
        report.total_candles,
        report.average_sweep_quality_score,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_liquidity_map_report()
        print("Institutional Liquidity Map Summary")
        print(f"Candles: {report.total_candles}")
        print(f"Avg sweep quality score: {report.average_sweep_quality_score}")
        print("Sweep events:")
        for label, count in report.sweep_event_distribution.items():
            print(f"  {label}: {count}")
        if report.top_sweep_examples:
            sample = report.top_sweep_examples[0]
            print(f"Top sweep narrative: {sample['market_narrative']}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalLiquidityMapError as exc:
        logger.error("Institutional liquidity map error: %s", exc)
        print(f"Institutional liquidity map error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional liquidity map failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
