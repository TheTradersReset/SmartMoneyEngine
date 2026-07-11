"""
Institutional Market Narrative Engine V2 for SmartMoneyEngine.

Converts Tier-2 institutional event sequences into market intent narratives
using existing SMC pipeline structure only. Research-only context layer;
does not modify production signals or generate entries.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine, MarketIntent
from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.research.institutional_edge_extraction_research import (
    EdgeFeatureRecord,
    InstitutionalEdgeExtractionResearch,
)
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.tiered_signal_framework_research import TierSignal, TieredSignalFrameworkResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "context" / "institutional_market_narrative_v2.json"

DEALING_RANGE_LOOKBACK = 60
SAMPLE_NARRATIVE_COUNT = 12


class InstitutionalMarketNarrativeV2Error(Exception):
    """Raised when institutional market narrative V2 evaluation fails."""


class LiquidityObjective(str, Enum):
    TARGETING_BUY_SIDE = "Targeting Buy Side Liquidity"
    TARGETING_SELL_SIDE = "Targeting Sell Side Liquidity"
    MOVING_TO_PREMIUM = "Moving To Premium"
    MOVING_TO_DISCOUNT = "Moving To Discount"
    INTERNAL_LIQUIDITY_RAID = "Internal Liquidity Raid"
    EXTERNAL_LIQUIDITY_RAID = "External Liquidity Raid"


class StructuralQuality(str, Enum):
    WEAK = "Weak"
    MEDIUM = "Medium"
    STRONG = "Strong"
    INSTITUTIONAL = "Institutional"


class ExpansionQuality(str, Enum):
    WEAK = "Weak"
    MEDIUM = "Medium"
    STRONG = "Strong"
    INSTITUTIONAL = "Institutional"


@dataclass(frozen=True)
class DealingRangeAnalysis:
    """Dealing range context for one Tier-2 signal."""

    swing_high: float
    swing_low: float
    equilibrium: float
    current_price: float
    price_zone: str
    premium_zone_top: float
    premium_zone_bottom: float
    discount_zone_top: float
    discount_zone_bottom: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Tier2MarketNarrative:
    """Institutional market narrative for one Tier-2 signal."""

    timeframe: str
    direction: str
    bos_timestamp: str
    market_phase: str
    liquidity_objective: str
    structural_quality: str
    dealing_range: dict[str, Any]
    institutional_quality_score: int
    narrative_confidence: int
    narrative: str
    expected_liquidity_path: str
    expected_expansion_direction: str
    expected_expansion_quality: str
    sequence_events: list[str]
    feature_summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalMarketNarrativeV2Report:
    """Aggregate institutional market narrative V2 output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    total_signals: int
    market_phase_distribution: dict[str, int]
    liquidity_objective_distribution: dict[str, int]
    structural_quality_distribution: dict[str, int]
    average_narrative_confidence: float
    narratives: list[dict[str, Any]]
    sample_narratives: list[dict[str, Any]]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalMarketNarrativeEngineV2:
    """
    Evaluate market intent narratives for Tier-2 institutional signals.

    Uses existing SMC pipeline columns and validated Tier-2 sequence logic only.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.tier_engine = TieredSignalFrameworkResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.edge_engine = InstitutionalEdgeExtractionResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.quality_engine = InstitutionalQualityValidationResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)
        self.filter_engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )

    @staticmethod
    def _latest_active_value(window: pd.DataFrame, column: str) -> float | None:
        for offset in range(len(window) - 1, -1, -1):
            value = LiquidityNarrativeEngine._to_float(window.iloc[offset].get(column))
            if value is not None:
                return value
        return None

    def _dealing_range(self, frame: pd.DataFrame, bos_bar: int) -> DealingRangeAnalysis:
        window = frame.iloc[max(0, bos_bar - DEALING_RANGE_LOOKBACK) : bos_bar + 1]
        swing_high = self._latest_active_value(window, "Swing_High")
        swing_low = self._latest_active_value(window, "Swing_Low")

        if swing_high is None:
            swing_high = float(window["High"].astype(float).max())
        if swing_low is None:
            swing_low = float(window["Low"].astype(float).min())

        equilibrium = round((swing_high + swing_low) / 2.0, 2)
        current_price = round(float(frame.iloc[bos_bar]["Close"]), 2)
        range_size = max(swing_high - swing_low, 0.01)

        if current_price > equilibrium + range_size * 0.05:
            price_zone = "Premium Zone"
        elif current_price < equilibrium - range_size * 0.05:
            price_zone = "Discount Zone"
        else:
            price_zone = "Equilibrium"

        premium_bottom = equilibrium
        discount_top = equilibrium

        return DealingRangeAnalysis(
            swing_high=round(swing_high, 2),
            swing_low=round(swing_low, 2),
            equilibrium=equilibrium,
            current_price=current_price,
            price_zone=price_zone,
            premium_zone_top=round(swing_high, 2),
            premium_zone_bottom=premium_bottom,
            discount_zone_top=discount_top,
            discount_zone_bottom=round(swing_low, 2),
        )

    def _liquidity_objective(
        self,
        direction: str,
        frame: pd.DataFrame,
        bos_bar: int,
        dealing_range: DealingRangeAnalysis,
    ) -> LiquidityObjective:
        window = self.narrative_engine._window(frame, bos_bar)
        liquidity = self.narrative_engine._liquidity_events(frame, bos_bar, window)
        close = dealing_range.current_price

        buy_pool = liquidity.active_buy_side_liquidity
        sell_pool = liquidity.active_sell_side_liquidity

        if liquidity.buy_side_liquidity_taken or liquidity.sell_side_liquidity_taken:
            sweep_high = liquidity.latest_buy_sweep_price
            sweep_low = liquidity.latest_sell_sweep_price
            if sweep_high is not None and sweep_high >= dealing_range.swing_high:
                return LiquidityObjective.EXTERNAL_LIQUIDITY_RAID
            if sweep_low is not None and sweep_low <= dealing_range.swing_low:
                return LiquidityObjective.EXTERNAL_LIQUIDITY_RAID
            if self._latest_active_value(window, "Equal_High") is not None:
                return LiquidityObjective.INTERNAL_LIQUIDITY_RAID
            if self._latest_active_value(window, "Equal_Low") is not None:
                return LiquidityObjective.INTERNAL_LIQUIDITY_RAID
            if liquidity.buy_side_liquidity_taken or liquidity.sell_side_liquidity_taken:
                return LiquidityObjective.INTERNAL_LIQUIDITY_RAID

        if direction == "bullish":
            if buy_pool is not None and close <= buy_pool:
                return LiquidityObjective.TARGETING_BUY_SIDE
            if dealing_range.price_zone == "Discount Zone":
                return LiquidityObjective.MOVING_TO_PREMIUM
            if sell_pool is not None:
                return LiquidityObjective.TARGETING_BUY_SIDE
            return LiquidityObjective.MOVING_TO_PREMIUM

        if sell_pool is not None and close >= sell_pool:
            return LiquidityObjective.TARGETING_SELL_SIDE
        if dealing_range.price_zone == "Premium Zone":
            return LiquidityObjective.MOVING_TO_DISCOUNT
        if buy_pool is not None:
            return LiquidityObjective.TARGETING_SELL_SIDE
        return LiquidityObjective.MOVING_TO_DISCOUNT

    def _structural_quality_score(self, record: EdgeFeatureRecord) -> int:
        score = 0

        displacement_scores = {
            "Strong": 25,
            "Medium": 15,
            "Weak": 5,
            "None": 0,
        }
        score += displacement_scores.get(record.displacement_strength, 0)

        if 6 <= record.fvg_freshness_bars <= 15:
            score += 15
        elif record.fvg_freshness_bars <= 5:
            score += 10
        else:
            score += 5

        if record.fvg_retests == 1:
            score += 15
        elif record.fvg_retests == 0:
            score += 8
        else:
            score += 3

        if 90 <= record.choch_to_bos_minutes < 240:
            score += 15
        elif 30 <= record.choch_to_bos_minutes < 90:
            score += 10
        else:
            score += 5

        if record.distance_from_swing_points < 20:
            score += 10
        elif record.distance_from_swing_points < 50:
            score += 7
        else:
            score += 3

        if record.distance_from_liquidity_pool_points < 50:
            score += 10
        elif record.distance_from_liquidity_pool_points < 100:
            score += 7
        else:
            score += 4

        if record.fvg_size_points >= 30:
            score += 10
        elif record.fvg_size_points >= 15:
            score += 7
        else:
            score += 4

        return min(score, 100)

    @staticmethod
    def _structural_quality_label(score: int) -> StructuralQuality:
        if score >= 80:
            return StructuralQuality.INSTITUTIONAL
        if score >= 60:
            return StructuralQuality.STRONG
        if score >= 40:
            return StructuralQuality.MEDIUM
        return StructuralQuality.WEAK

    def _market_phase(
        self,
        frame: pd.DataFrame,
        bos_bar: int,
        direction: str,
    ) -> MarketIntent:
        window = self.narrative_engine._window(frame, bos_bar)
        row = frame.iloc[bos_bar]
        liquidity = self.narrative_engine._liquidity_events(frame, bos_bar, window)
        structure = self.narrative_engine._structure_shift(window)
        displacement = self.narrative_engine._displacement_strength(window)
        fvg_context, fvg_bias = self.narrative_engine._fvg_context_state(frame, bos_bar, window)

        intent = self.narrative_engine._market_intent(
            row,
            liquidity,
            structure,
            displacement,
            fvg_context,
            fvg_bias,
        )

        if direction == "bullish" and intent == MarketIntent.DISTRIBUTION:
            return MarketIntent.ACCUMULATION
        if direction == "bearish" and intent == MarketIntent.ACCUMULATION:
            return MarketIntent.DISTRIBUTION
        return intent

    def _sequence_events(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
        record: EdgeFeatureRecord,
    ) -> list[str]:
        events: list[str] = []
        window = self.narrative_engine._window(frame, signal.bos_bar)
        liquidity = self.narrative_engine._liquidity_events(frame, signal.bos_bar, window)

        if liquidity.sell_side_liquidity_taken:
            events.append("Sell-side liquidity was taken.")
        if liquidity.buy_side_liquidity_taken:
            events.append("Buy-side liquidity was taken.")

        displacement_label = record.displacement_strength.lower()
        if displacement_label in {"strong", "medium"}:
            events.append(f"{displacement_label.title()} {signal.direction} displacement appeared.")
        else:
            events.append(f"Weak {signal.direction} displacement appeared.")

        if signal.direction == "bullish":
            events.append("Bullish CHOCH confirmed.")
            events.append("Bullish BOS confirmed.")
            events.append("Bullish FVG reclaimed.")
        else:
            events.append("Bearish CHOCH confirmed.")
            events.append("Bearish BOS confirmed.")
            events.append("Bearish FVG reclaimed.")

        return events

    def _build_narrative_text(
        self,
        events: list[str],
        market_phase: MarketIntent,
        liquidity_objective: LiquidityObjective,
    ) -> str:
        body = " ".join(events)
        phase_messages = {
            MarketIntent.ACCUMULATION: (
                "Market appears to be accumulating inventory before seeking higher liquidity."
            ),
            MarketIntent.DISTRIBUTION: (
                "Market appears to be distributing inventory before moving toward sell-side liquidity."
            ),
            MarketIntent.EXPANSION: (
                "Market appears to be expanding aggressively in the direction of institutional intent."
            ),
            MarketIntent.REVERSAL: (
                "Market appears to be reversing prior structure toward a new liquidity objective."
            ),
            MarketIntent.CONTINUATION: (
                "Market appears to be continuing the prevailing institutional trend."
            ),
            MarketIntent.RANGE: (
                "Market appears to be rotating inside the current dealing range."
            ),
        }
        objective_messages = {
            LiquidityObjective.TARGETING_BUY_SIDE: (
                "Liquidity objective: targeting buy-side liquidity above."
            ),
            LiquidityObjective.TARGETING_SELL_SIDE: (
                "Liquidity objective: targeting sell-side liquidity below."
            ),
            LiquidityObjective.MOVING_TO_PREMIUM: (
                "Liquidity objective: moving toward premium within the dealing range."
            ),
            LiquidityObjective.MOVING_TO_DISCOUNT: (
                "Liquidity objective: moving toward discount within the dealing range."
            ),
            LiquidityObjective.INTERNAL_LIQUIDITY_RAID: (
                "Liquidity objective: internal liquidity raid within the current range."
            ),
            LiquidityObjective.EXTERNAL_LIQUIDITY_RAID: (
                "Liquidity objective: external liquidity raid beyond the dealing range."
            ),
        }
        return f"{body} {phase_messages[market_phase]} {objective_messages[liquidity_objective]}"

    def _expected_liquidity_path(
        self,
        liquidity_objective: LiquidityObjective,
        dealing_range: DealingRangeAnalysis,
        direction: str,
    ) -> str:
        mapping = {
            LiquidityObjective.TARGETING_BUY_SIDE: (
                f"Expected path higher toward buy-side liquidity near {dealing_range.swing_high}."
            ),
            LiquidityObjective.TARGETING_SELL_SIDE: (
                f"Expected path lower toward sell-side liquidity near {dealing_range.swing_low}."
            ),
            LiquidityObjective.MOVING_TO_PREMIUM: (
                f"Expected rotation from discount toward premium above equilibrium "
                f"{dealing_range.equilibrium}."
            ),
            LiquidityObjective.MOVING_TO_DISCOUNT: (
                f"Expected rotation from premium toward discount below equilibrium "
                f"{dealing_range.equilibrium}."
            ),
            LiquidityObjective.INTERNAL_LIQUIDITY_RAID: (
                "Expected raid of internal range liquidity before directional expansion."
            ),
            LiquidityObjective.EXTERNAL_LIQUIDITY_RAID: (
                "Expected continuation toward external liquidity beyond the dealing range."
            ),
        }
        return mapping[liquidity_objective]

    @staticmethod
    def _expansion_quality_label(structural: StructuralQuality) -> ExpansionQuality:
        return ExpansionQuality(structural.value)

    def _narrative_confidence(
        self,
        structural_score: int,
        institutional_quality_score: int,
        market_phase: MarketIntent,
    ) -> int:
        phase_bonus = {
            MarketIntent.DISTRIBUTION: 5,
            MarketIntent.ACCUMULATION: 5,
            MarketIntent.EXPANSION: 8,
            MarketIntent.REVERSAL: 6,
            MarketIntent.CONTINUATION: 4,
            MarketIntent.RANGE: 0,
        }[market_phase]
        blended = (structural_score * 0.55) + (institutional_quality_score * 0.45) + phase_bonus
        return min(100, max(0, round(blended)))

    def evaluate_signal(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
        record: EdgeFeatureRecord,
    ) -> Tier2MarketNarrative:
        """Build institutional market narrative for one Tier-2 signal."""
        dealing_range = self._dealing_range(frame, signal.bos_bar)
        market_phase = self._market_phase(frame, signal.bos_bar, signal.direction)
        liquidity_objective = self._liquidity_objective(
            signal.direction,
            frame,
            signal.bos_bar,
            dealing_range,
        )
        structural_score = self._structural_quality_score(record)
        structural_quality = self._structural_quality_label(structural_score)
        institutional_quality_score, _, _ = self.quality_engine.compute_quality_score(record)
        events = self._sequence_events(frame, signal, record)
        narrative = self._build_narrative_text(events, market_phase, liquidity_objective)
        confidence = self._narrative_confidence(
            structural_score,
            institutional_quality_score,
            market_phase,
        )
        expansion_quality = self._expansion_quality_label(structural_quality)

        return Tier2MarketNarrative(
            timeframe=signal.timeframe,
            direction=signal.direction,
            bos_timestamp=signal.bos_timestamp,
            market_phase=market_phase.value,
            liquidity_objective=liquidity_objective.value,
            structural_quality=structural_quality.value,
            dealing_range=dealing_range.as_dict(),
            institutional_quality_score=institutional_quality_score,
            narrative_confidence=confidence,
            narrative=narrative,
            expected_liquidity_path=self._expected_liquidity_path(
                liquidity_objective,
                dealing_range,
                signal.direction,
            ),
            expected_expansion_direction=signal.direction,
            expected_expansion_quality=expansion_quality.value,
            sequence_events=events,
            feature_summary={
                "displacement_strength": record.displacement_strength,
                "fvg_size_points": record.fvg_size_points,
                "fvg_freshness_bars": record.fvg_freshness_bars,
                "fvg_retests": record.fvg_retests,
                "distance_from_liquidity_pool_points": record.distance_from_liquidity_pool_points,
                "distance_from_swing_points": record.distance_from_swing_points,
                "choch_to_bos_minutes": record.choch_to_bos_minutes,
                "structural_quality_score": structural_score,
            },
        )

    def _collect_signal_pairs_direct(
        self,
        metadata: dict[str, Any],
        start: date,
        end: date,
    ) -> list[tuple[TierSignal, EdgeFeatureRecord, pd.DataFrame]]:
        pairs: list[tuple[TierSignal, EdgeFeatureRecord, pd.DataFrame]] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            for signal in self.tier_engine._detect_tier2(frame, timeframe_label):
                detail = self.edge_engine.distribution_engine._simulate_detailed(frame, signal)
                if detail is None:
                    continue
                record = self.edge_engine._extract_features(
                    frame,
                    signal,
                    detail.realized_pnl_points,
                    detail.win,
                    detail.risk_points,
                )
                if record:
                    pairs.append((signal, record, frame))
        pairs.sort(key=lambda item: pd.Timestamp(item[0].bos_timestamp))
        return pairs

    def run(self, metadata: dict[str, Any]) -> InstitutionalMarketNarrativeV2Report:
        """Evaluate narratives for all Tier-2 signals in the research window."""
        started = time.perf_counter()

        end = (
            date.fromisoformat(metadata["end_date"])
            if metadata.get("end_date")
            else date.today()
        )
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        pairs = self._collect_signal_pairs_direct(metadata, start, end)
        narratives = [
            self.evaluate_signal(frame, signal, record).as_dict()
            for signal, record, frame in pairs
        ]

        phase_dist = Counter(item["market_phase"] for item in narratives)
        objective_dist = Counter(item["liquidity_objective"] for item in narratives)
        quality_dist = Counter(item["structural_quality"] for item in narratives)
        avg_confidence = round(
            mean(item["narrative_confidence"] for item in narratives),
            2,
        ) if narratives else 0.0

        return InstitutionalMarketNarrativeV2Report(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            total_signals=len(narratives),
            market_phase_distribution=dict(phase_dist),
            liquidity_objective_distribution=dict(objective_dist),
            structural_quality_distribution=dict(quality_dist),
            average_narrative_confidence=avg_confidence,
            narratives=narratives,
            sample_narratives=sorted(
                narratives,
                key=lambda item: item["narrative_confidence"],
                reverse=True,
            )[:SAMPLE_NARRATIVE_COUNT],
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_market_narrative_v2_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> InstitutionalMarketNarrativeV2Report:
    """Run institutional market narrative V2 and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalMarketNarrativeV2Error(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalMarketNarrativeEngineV2(
        symbol=metadata.get("symbol", "NIFTY50"),
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", ("5M", "15M", "1H"))),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Institutional market narrative V2 completed: signals=%s avg_confidence=%s",
        report.total_signals,
        report.average_narrative_confidence,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_market_narrative_v2_report()
        print("Institutional Market Narrative Engine V2 Summary")
        print(f"Signals: {report.total_signals}")
        print(f"Avg confidence: {report.average_narrative_confidence}")
        print(f"Market phases: {report.market_phase_distribution}")
        print(f"Structural quality: {report.structural_quality_distribution}")
        if report.sample_narratives:
            sample = report.sample_narratives[0]
            print(f"Top narrative confidence: {sample['narrative_confidence']}")
            print(f"Phase: {sample['market_phase']}")
            print(f"Objective: {sample['liquidity_objective']}")
            print(f"Narrative: {sample['narrative'][:200]}...")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalMarketNarrativeV2Error as exc:
        logger.error("Institutional market narrative V2 error: %s", exc)
        print(f"Institutional market narrative V2 error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional market narrative V2 failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
