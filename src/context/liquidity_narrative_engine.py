"""
Liquidity Narrative Engine V1 for SmartMoneyEngine.

Determines market intent from institutional liquidity, structure, displacement,
and FVG context before any signal generation. Research-only context layer;
does not create trades, entries, or indicator optimization.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "context" / "liquidity_narrative_report.json"

NARRATIVE_LOOKBACK_BARS = 20
SAMPLE_SUMMARY_COUNT = 12
TOP_EXAMPLE_COUNT = 10

NARRATIVE_WEIGHTS: dict[str, int] = {
    "liquidity_event": 25,
    "structure_shift": 25,
    "displacement": 20,
    "fvg_context": 15,
    "intent_clarity": 15,
}

REQUIRED_COLUMNS = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Trend",
    "Trend_Strength",
    "Bullish_BOS",
    "Bearish_BOS",
    "Bullish_CHOCH",
    "Bearish_CHOCH",
    "Bullish_FVG_Top",
    "Bullish_FVG_Bottom",
    "Bearish_FVG_Top",
    "Bearish_FVG_Bottom",
    "Buy_Side_Liquidity",
    "Sell_Side_Liquidity",
    "Buy_Liquidity_Sweep",
    "Sell_Liquidity_Sweep",
    "Liquidity_Strength",
)


class LiquidityNarrativeError(Exception):
    """Raised when liquidity narrative evaluation fails."""


class DisplacementStrength(str, Enum):
    WEAK = "Weak"
    MEDIUM = "Medium"
    STRONG = "Strong"
    NONE = "None"


class FvgContext(str, Enum):
    CREATED = "FVG Created"
    RECLAIMED = "FVG Reclaimed"
    FAILED = "FVG Failed"
    NONE = "None"


class MarketIntent(str, Enum):
    ACCUMULATION = "Accumulation"
    DISTRIBUTION = "Distribution"
    EXPANSION = "Expansion"
    REVERSAL = "Reversal"
    CONTINUATION = "Continuation"
    RANGE = "Range"


@dataclass(frozen=True)
class LiquidityEventSnapshot:
    """Recent liquidity events within the narrative lookback."""

    buy_side_liquidity_taken: bool
    sell_side_liquidity_taken: bool
    latest_buy_sweep_price: float | None
    latest_sell_sweep_price: float | None
    active_buy_side_liquidity: float | None
    active_sell_side_liquidity: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructureShiftSnapshot:
    """Recent structure shift events."""

    bullish_choch: bool
    bearish_choch: bool
    bullish_bos: bool
    bearish_bos: bool
    latest_bullish_choch_price: float | None
    latest_bearish_choch_price: float | None
    latest_bullish_bos_price: float | None
    latest_bearish_bos_price: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NarrativeComponents:
    """Weighted sub-scores contributing to narrative strength."""

    liquidity_event: float
    structure_shift: float
    displacement: float
    fvg_context: float
    intent_clarity: float

    @property
    def total(self) -> float:
        return (
            self.liquidity_event * NARRATIVE_WEIGHTS["liquidity_event"]
            + self.structure_shift * NARRATIVE_WEIGHTS["structure_shift"]
            + self.displacement * NARRATIVE_WEIGHTS["displacement"]
            + self.fvg_context * NARRATIVE_WEIGHTS["fvg_context"]
            + self.intent_clarity * NARRATIVE_WEIGHTS["intent_clarity"]
        ) / 100.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CandleNarrative:
    """Liquidity narrative evaluation for one candle."""

    index: int
    timestamp: str
    close: float
    liquidity_events: dict[str, Any]
    structure_shift: dict[str, Any]
    displacement_strength: str
    fvg_context: str
    fvg_bias: str | None
    market_intent: str
    narrative_strength_score: float
    components: dict[str, float]
    narrative: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiquidityNarrativeReport:
    """Aggregate liquidity narrative report."""

    symbol: str
    timeframe: str
    source_csv: str
    total_candles: int
    average_narrative_strength: float
    score_distribution: dict[str, int]
    intent_distribution: dict[str, int]
    displacement_distribution: dict[str, int]
    fvg_context_distribution: dict[str, int]
    liquidity_event_distribution: dict[str, int]
    top_narrative_examples: list[dict[str, Any]]
    sample_summaries: list[dict[str, Any]]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiquidityNarrativeEngine:
    """
    Evaluate institutional liquidity narrative for every pipeline candle.

    Uses existing SMC pipeline columns only. Does not generate trades or
    optimize indicators.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5M",
        lookback_bars: int = NARRATIVE_LOOKBACK_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_trend(value: Any) -> str:
        return str(value).strip().upper()

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise LiquidityNarrativeError(f"Pipeline frame missing narrative columns: {missing}")

    def _window(self, frame: pd.DataFrame, index: int) -> pd.DataFrame:
        start = max(0, index - self.lookback_bars)
        return frame.iloc[start : index + 1]

    def _latest_active_value(
        self,
        window: pd.DataFrame,
        column: str,
    ) -> float | None:
        for offset in range(len(window) - 1, -1, -1):
            value = self._to_float(window.iloc[offset].get(column))
            if value is not None:
                return value
        return None

    def _column_active_in_window(self, window: pd.DataFrame, column: str) -> bool:
        if column not in window.columns:
            return False
        return any(self._is_active(value) for value in window[column])

    def _liquidity_events(
        self,
        frame: pd.DataFrame,
        index: int,
        window: pd.DataFrame,
    ) -> LiquidityEventSnapshot:
        row = frame.iloc[index]
        return LiquidityEventSnapshot(
            buy_side_liquidity_taken=self._column_active_in_window(
                window,
                "Buy_Liquidity_Sweep",
            ),
            sell_side_liquidity_taken=self._column_active_in_window(
                window,
                "Sell_Liquidity_Sweep",
            ),
            latest_buy_sweep_price=self._latest_active_value(window, "Buy_Liquidity_Sweep"),
            latest_sell_sweep_price=self._latest_active_value(window, "Sell_Liquidity_Sweep"),
            active_buy_side_liquidity=self._to_float(row.get("Buy_Side_Liquidity")),
            active_sell_side_liquidity=self._to_float(row.get("Sell_Side_Liquidity")),
        )

    def _structure_shift(self, window: pd.DataFrame) -> StructureShiftSnapshot:
        return StructureShiftSnapshot(
            bullish_choch=self._column_active_in_window(window, "Bullish_CHOCH"),
            bearish_choch=self._column_active_in_window(window, "Bearish_CHOCH"),
            bullish_bos=self._column_active_in_window(window, "Bullish_BOS"),
            bearish_bos=self._column_active_in_window(window, "Bearish_BOS"),
            latest_bullish_choch_price=self._latest_active_value(window, "Bullish_CHOCH"),
            latest_bearish_choch_price=self._latest_active_value(window, "Bearish_CHOCH"),
            latest_bullish_bos_price=self._latest_active_value(window, "Bullish_BOS"),
            latest_bearish_bos_price=self._latest_active_value(window, "Bearish_BOS"),
        )

    def _latest_structural_bar(
        self,
        window: pd.DataFrame,
    ) -> tuple[int | None, str | None]:
        for offset in range(len(window) - 1, -1, -1):
            row = window.iloc[offset]
            if self._is_active(row.get("Bullish_BOS")):
                return offset, "bullish"
            if self._is_active(row.get("Bearish_BOS")):
                return offset, "bearish"
            if self._is_active(row.get("Bullish_CHOCH")):
                return offset, "bullish"
            if self._is_active(row.get("Bearish_CHOCH")):
                return offset, "bearish"
        return None, None

    @staticmethod
    def _displacement_strength_for_bar(
        row: pd.Series,
        direction: str | None,
    ) -> DisplacementStrength:
        open_price = LiquidityNarrativeEngine._to_float(row.get("Open"))
        high = LiquidityNarrativeEngine._to_float(row.get("High"))
        low = LiquidityNarrativeEngine._to_float(row.get("Low"))
        close = LiquidityNarrativeEngine._to_float(row.get("Close"))
        if None in (open_price, high, low, close) or high <= low:
            return DisplacementStrength.NONE

        body = abs(close - open_price)
        candle_range = high - low
        body_ratio = body / candle_range

        if direction == "bullish" and close <= open_price:
            return DisplacementStrength.WEAK
        if direction == "bearish" and close >= open_price:
            return DisplacementStrength.WEAK

        if body_ratio >= 0.75:
            return DisplacementStrength.STRONG
        if body_ratio >= 0.55:
            return DisplacementStrength.MEDIUM
        return DisplacementStrength.WEAK

    def _displacement_strength(
        self,
        window: pd.DataFrame,
    ) -> DisplacementStrength:
        bar_offset, direction = self._latest_structural_bar(window)
        if bar_offset is None:
            return DisplacementStrength.NONE
        return self._displacement_strength_for_bar(window.iloc[bar_offset], direction)

    def _latest_fvg(
        self,
        window: pd.DataFrame,
    ) -> tuple[str, float, float, int] | None:
        for offset in range(len(window) - 1, -1, -1):
            row = window.iloc[offset]
            bullish_top = self._to_float(row.get("Bullish_FVG_Top"))
            bullish_bottom = self._to_float(row.get("Bullish_FVG_Bottom"))
            if bullish_top is not None and bullish_bottom is not None:
                return ("bullish", bullish_bottom, bullish_top, offset)

            bearish_top = self._to_float(row.get("Bearish_FVG_Top"))
            bearish_bottom = self._to_float(row.get("Bearish_FVG_Bottom"))
            if bearish_top is not None and bearish_bottom is not None:
                return ("bearish", bearish_bottom, bearish_top, offset)
        return None

    def _fvg_created_on_bar(self, row: pd.Series, previous: pd.Series | None) -> bool:
        current_bull = self._is_active(row.get("Bullish_FVG_Top")) and self._is_active(
            row.get("Bullish_FVG_Bottom"),
        )
        current_bear = self._is_active(row.get("Bearish_FVG_Top")) and self._is_active(
            row.get("Bearish_FVG_Bottom"),
        )
        if previous is None:
            return current_bull or current_bear

        prev_bull = self._is_active(previous.get("Bullish_FVG_Top")) and self._is_active(
            previous.get("Bullish_FVG_Bottom"),
        )
        prev_bear = self._is_active(previous.get("Bearish_FVG_Top")) and self._is_active(
            previous.get("Bearish_FVG_Bottom"),
        )
        return (current_bull and not prev_bull) or (current_bear and not prev_bear)

    def _fvg_context_state(
        self,
        frame: pd.DataFrame,
        index: int,
        window: pd.DataFrame,
    ) -> tuple[FvgContext, str | None]:
        row = frame.iloc[index]
        previous = frame.iloc[index - 1] if index > 0 else None

        if self._fvg_created_on_bar(row, previous):
            if self._is_active(row.get("Bullish_FVG_Top")):
                return FvgContext.CREATED, "bullish"
            if self._is_active(row.get("Bearish_FVG_Top")):
                return FvgContext.CREATED, "bearish"

        latest = self._latest_fvg(window)
        if latest is None:
            return FvgContext.NONE, None

        bias, bottom, top, _ = latest
        close = self._to_float(row.get("Close")) or 0.0
        low = self._to_float(row.get("Low")) or close
        high = self._to_float(row.get("High")) or close

        if bias == "bullish":
            if close > top:
                return FvgContext.RECLAIMED, bias
            if low <= top and close < bottom:
                return FvgContext.FAILED, bias
        else:
            if close < bottom:
                return FvgContext.RECLAIMED, bias
            if high >= bottom and close > top:
                return FvgContext.FAILED, bias

        return FvgContext.NONE, bias

    def _market_intent(
        self,
        row: pd.Series,
        liquidity: LiquidityEventSnapshot,
        structure: StructureShiftSnapshot,
        displacement: DisplacementStrength,
        fvg_context: FvgContext,
        fvg_bias: str | None,
    ) -> MarketIntent:
        trend = self._normalize_trend(row.get("Trend"))
        trend_strength = self._to_float(row.get("Trend_Strength")) or 0.0

        if structure.bearish_choch and liquidity.buy_side_liquidity_taken:
            return MarketIntent.DISTRIBUTION
        if structure.bullish_choch and liquidity.sell_side_liquidity_taken:
            return MarketIntent.ACCUMULATION

        if structure.bearish_choch or structure.bullish_choch:
            return MarketIntent.REVERSAL

        if (
            structure.bearish_bos
            and liquidity.buy_side_liquidity_taken
            and displacement in {DisplacementStrength.MEDIUM, DisplacementStrength.STRONG}
        ):
            return MarketIntent.DISTRIBUTION

        if (
            structure.bullish_bos
            and liquidity.sell_side_liquidity_taken
            and displacement in {DisplacementStrength.MEDIUM, DisplacementStrength.STRONG}
        ):
            return MarketIntent.ACCUMULATION

        if displacement == DisplacementStrength.STRONG and (
            structure.bullish_bos or structure.bearish_bos
        ):
            return MarketIntent.EXPANSION

        if structure.bullish_bos and trend == "BULLISH":
            return MarketIntent.CONTINUATION
        if structure.bearish_bos and trend == "BEARISH":
            return MarketIntent.CONTINUATION

        if fvg_context == FvgContext.RECLAIMED and fvg_bias == "bullish":
            return MarketIntent.ACCUMULATION
        if fvg_context == FvgContext.RECLAIMED and fvg_bias == "bearish":
            return MarketIntent.DISTRIBUTION

        if (
            not structure.bullish_bos
            and not structure.bearish_bos
            and not structure.bullish_choch
            and not structure.bearish_choch
            and not liquidity.buy_side_liquidity_taken
            and not liquidity.sell_side_liquidity_taken
            and trend_strength <= 1
        ):
            return MarketIntent.RANGE

        return MarketIntent.CONTINUATION

    @staticmethod
    def _liquidity_event_label(liquidity: LiquidityEventSnapshot) -> str:
        if liquidity.buy_side_liquidity_taken and liquidity.sell_side_liquidity_taken:
            return "Both Sides Swept"
        if liquidity.buy_side_liquidity_taken:
            return "Buy Side Liquidity Taken"
        if liquidity.sell_side_liquidity_taken:
            return "Sell Side Liquidity Taken"
        return "No Recent Sweep"

    def _component_scores(
        self,
        liquidity: LiquidityEventSnapshot,
        structure: StructureShiftSnapshot,
        displacement: DisplacementStrength,
        fvg_context: FvgContext,
        market_intent: MarketIntent,
        row: pd.Series,
    ) -> NarrativeComponents:
        if liquidity.buy_side_liquidity_taken and liquidity.sell_side_liquidity_taken:
            liquidity_score = 55.0
        elif liquidity.buy_side_liquidity_taken or liquidity.sell_side_liquidity_taken:
            liquidity_score = 100.0
        else:
            liquidity_score = 25.0

        liquidity_strength = self._to_float(row.get("Liquidity_Strength")) or 0.0
        liquidity_score = min(100.0, liquidity_score + min(liquidity_strength, 3) * 5.0)

        if (structure.bullish_choch or structure.bearish_choch) and (
            structure.bullish_bos or structure.bearish_bos
        ):
            structure_score = 100.0
        elif structure.bullish_choch or structure.bearish_choch:
            structure_score = 85.0
        elif structure.bullish_bos or structure.bearish_bos:
            structure_score = 70.0
        else:
            structure_score = 20.0

        displacement_score = {
            DisplacementStrength.STRONG: 100.0,
            DisplacementStrength.MEDIUM: 70.0,
            DisplacementStrength.WEAK: 40.0,
            DisplacementStrength.NONE: 15.0,
        }[displacement]

        fvg_score = {
            FvgContext.RECLAIMED: 100.0,
            FvgContext.CREATED: 85.0,
            FvgContext.FAILED: 45.0,
            FvgContext.NONE: 20.0,
        }[fvg_context]

        intent_score = {
            MarketIntent.DISTRIBUTION: 95.0,
            MarketIntent.ACCUMULATION: 95.0,
            MarketIntent.REVERSAL: 90.0,
            MarketIntent.EXPANSION: 85.0,
            MarketIntent.CONTINUATION: 75.0,
            MarketIntent.RANGE: 35.0,
        }[market_intent]

        return NarrativeComponents(
            liquidity_event=round(liquidity_score, 2),
            structure_shift=round(structure_score, 2),
            displacement=round(displacement_score, 2),
            fvg_context=round(fvg_score, 2),
            intent_clarity=round(intent_score, 2),
        )

    def _expected_path(
        self,
        market_intent: MarketIntent,
        liquidity: LiquidityEventSnapshot,
    ) -> str:
        if market_intent in {MarketIntent.DISTRIBUTION, MarketIntent.REVERSAL}:
            if liquidity.active_sell_side_liquidity is not None:
                return "Expected path toward sell-side liquidity."
            return "Expected path lower toward resting sell-side liquidity."
        if market_intent in {MarketIntent.ACCUMULATION, MarketIntent.EXPANSION}:
            if liquidity.active_buy_side_liquidity is not None:
                return "Expected path toward buy-side liquidity."
            return "Expected path higher toward resting buy-side liquidity."
        if market_intent == MarketIntent.CONTINUATION:
            trend_target = (
                "buy-side liquidity"
                if liquidity.active_buy_side_liquidity is not None
                else "the next structural liquidity pool"
            )
            return f"Expected continuation toward {trend_target}."
        return "Expected two-sided rotation within the current range."

    def _build_narrative(
        self,
        liquidity: LiquidityEventSnapshot,
        structure: StructureShiftSnapshot,
        displacement: DisplacementStrength,
        fvg_context: FvgContext,
        fvg_bias: str | None,
        market_intent: MarketIntent,
    ) -> str:
        sentences: list[str] = []

        if liquidity.buy_side_liquidity_taken:
            sentences.append(
                "Buy-side liquidity above prior highs was swept.",
            )
        if liquidity.sell_side_liquidity_taken:
            sentences.append(
                "Sell-side liquidity below prior lows was swept.",
            )
        if not liquidity.buy_side_liquidity_taken and not liquidity.sell_side_liquidity_taken:
            sentences.append("No recent liquidity sweep in the lookback window.")

        if displacement == DisplacementStrength.STRONG:
            sentences.append("Strong institutional displacement followed.")
        elif displacement == DisplacementStrength.MEDIUM:
            sentences.append("Moderate displacement followed.")
        elif displacement == DisplacementStrength.WEAK:
            sentences.append("Weak displacement followed.")
        else:
            sentences.append("No meaningful displacement detected.")

        if structure.bullish_choch:
            sentences.append("Bullish CHOCH confirmed.")
        if structure.bearish_choch:
            sentences.append("Bearish CHOCH confirmed.")
        if structure.bullish_bos:
            sentences.append("Bullish BOS confirmed.")
        if structure.bearish_bos:
            sentences.append("Bearish BOS confirmed.")
        if not (
            structure.bullish_choch
            or structure.bearish_choch
            or structure.bullish_bos
            or structure.bearish_bos
        ):
            sentences.append("No confirmed structure shift in the lookback window.")

        if fvg_context == FvgContext.CREATED:
            bias_label = "bullish" if fvg_bias == "bullish" else "bearish"
            sentences.append(f"A fresh {bias_label} FVG was created.")
        elif fvg_context == FvgContext.RECLAIMED:
            bias_label = "bullish" if fvg_bias == "bullish" else "bearish"
            sentences.append(f"{bias_label.title()} FVG was reclaimed.")
        elif fvg_context == FvgContext.FAILED:
            sentences.append("FVG reclaim attempt failed.")

        intent_messages = {
            MarketIntent.ACCUMULATION: "Market appears to be accumulating inventory.",
            MarketIntent.DISTRIBUTION: "Market appears to be distributing inventory.",
            MarketIntent.EXPANSION: "Market appears to be expanding aggressively.",
            MarketIntent.REVERSAL: "Market appears to be reversing prior structure.",
            MarketIntent.CONTINUATION: "Market appears to be continuing the prevailing move.",
            MarketIntent.RANGE: "Market appears to be ranging without clear intent.",
        }
        sentences.append(intent_messages[market_intent])
        sentences.append(self._expected_path(market_intent, liquidity))

        return " ".join(sentences)

    def evaluate_bar(self, frame: pd.DataFrame, index: int) -> CandleNarrative:
        """Evaluate liquidity narrative for one candle."""
        row = frame.iloc[index]
        window = self._window(frame, index)
        close = self._to_float(row.get("Close")) or 0.0

        liquidity = self._liquidity_events(frame, index, window)
        structure = self._structure_shift(window)
        displacement = self._displacement_strength(window)
        fvg_context, fvg_bias = self._fvg_context_state(frame, index, window)
        market_intent = self._market_intent(
            row,
            liquidity,
            structure,
            displacement,
            fvg_context,
            fvg_bias,
        )
        components = self._component_scores(
            liquidity,
            structure,
            displacement,
            fvg_context,
            market_intent,
            row,
        )
        score = round(components.total, 2)
        narrative = self._build_narrative(
            liquidity,
            structure,
            displacement,
            fvg_context,
            fvg_bias,
            market_intent,
        )

        return CandleNarrative(
            index=index,
            timestamp=str(row.get("Date")),
            close=round(close, 2),
            liquidity_events={
                **liquidity.as_dict(),
                "event_label": self._liquidity_event_label(liquidity),
            },
            structure_shift=structure.as_dict(),
            displacement_strength=displacement.value,
            fvg_context=fvg_context.value,
            fvg_bias=fvg_bias,
            market_intent=market_intent.value,
            narrative_strength_score=score,
            components=components.as_dict(),
            narrative=narrative,
        )

    def evaluate(self, frame: pd.DataFrame) -> list[CandleNarrative]:
        """Evaluate liquidity narrative for every candle."""
        self._validate_frame(frame)
        working = frame.reset_index(drop=True)
        return [self.evaluate_bar(working, index) for index in range(len(working))]

    @staticmethod
    def _score_bucket(score: float) -> str:
        if score < 20:
            return "0-19"
        if score < 40:
            return "20-39"
        if score < 60:
            return "40-59"
        if score < 80:
            return "60-79"
        return "80-100"

    def build_report(
        self,
        evaluations: list[CandleNarrative],
        source_csv: Path | str,
        execution_time_seconds: float,
    ) -> LiquidityNarrativeReport:
        """Build aggregate report from per-candle evaluations."""
        score_distribution = Counter(
            self._score_bucket(item.narrative_strength_score) for item in evaluations
        )
        intent_distribution = Counter(item.market_intent for item in evaluations)
        displacement_distribution = Counter(item.displacement_strength for item in evaluations)
        fvg_context_distribution = Counter(item.fvg_context for item in evaluations)
        liquidity_event_distribution = Counter(
            item.liquidity_events["event_label"] for item in evaluations
        )

        ranked = sorted(
            evaluations,
            key=lambda item: item.narrative_strength_score,
            reverse=True,
        )
        top_examples = [item.as_dict() for item in ranked[:TOP_EXAMPLE_COUNT]]

        if evaluations:
            step = max(1, len(evaluations) // SAMPLE_SUMMARY_COUNT)
            sample_indices = list(range(0, len(evaluations), step))[:SAMPLE_SUMMARY_COUNT]
        else:
            sample_indices = []

        sample_summaries = [
            {
                "timestamp": evaluations[index].timestamp,
                "narrative_strength_score": evaluations[index].narrative_strength_score,
                "market_intent": evaluations[index].market_intent,
                "narrative": evaluations[index].narrative,
            }
            for index in sample_indices
        ]

        average_score = (
            round(
                sum(item.narrative_strength_score for item in evaluations) / len(evaluations),
                2,
            )
            if evaluations
            else 0.0
        )

        return LiquidityNarrativeReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=str(source_csv),
            total_candles=len(evaluations),
            average_narrative_strength=average_score,
            score_distribution=dict(sorted(score_distribution.items())),
            intent_distribution=dict(sorted(intent_distribution.items())),
            displacement_distribution=dict(sorted(displacement_distribution.items())),
            fvg_context_distribution=dict(sorted(fvg_context_distribution.items())),
            liquidity_event_distribution=dict(sorted(liquidity_event_distribution.items())),
            top_narrative_examples=top_examples,
            sample_summaries=sample_summaries,
            execution_time_seconds=round(execution_time_seconds, 3),
        )


def generate_liquidity_narrative_report(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5M",
) -> LiquidityNarrativeReport:
    """Evaluate liquidity narrative and export JSON report."""
    source = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    if not source.exists():
        raise LiquidityNarrativeError(f"Pipeline CSV not found: {source}")

    frame = pd.read_csv(source)
    engine = LiquidityNarrativeEngine(symbol=symbol, timeframe=timeframe)
    started = time.perf_counter()
    evaluations = engine.evaluate(frame)
    report = engine.build_report(
        evaluations,
        source_csv=source,
        execution_time_seconds=time.perf_counter() - started,
    )

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Liquidity narrative completed: candles=%s avg_score=%s",
        report.total_candles,
        report.average_narrative_strength,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_liquidity_narrative_report()
        print("Liquidity Narrative Summary")
        print(f"Symbol: {report.symbol} | Timeframe: {report.timeframe}")
        print(f"Candles: {report.total_candles}")
        print(f"Average Narrative Strength: {report.average_narrative_strength}")
        print("Intent Distribution:")
        for intent, count in report.intent_distribution.items():
            print(f"  {intent}: {count}")
        print("Displacement Distribution:")
        for label, count in report.displacement_distribution.items():
            print(f"  {label}: {count}")
        if report.top_narrative_examples:
            best = report.top_narrative_examples[0]
            print(
                f"Top Example: score={best['narrative_strength_score']} "
                f"intent={best['market_intent']} @ {best['timestamp']}"
            )
            print(f"Narrative: {best['narrative']}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except LiquidityNarrativeError as exc:
        logger.error("Liquidity narrative error: %s", exc)
        print(f"Liquidity narrative error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected liquidity narrative failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
