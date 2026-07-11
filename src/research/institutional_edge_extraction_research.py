"""
Institutional edge extraction research for SmartMoneyEngine.

Analyzes Raw Tier-2 signals to extract structural characteristics that
differentiate top winners from bottom losers. Research-only; no production
logic, indicators, or entries.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine
from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.tier2_trade_distribution_research import (
    TIER2_DEFINITION,
    Tier2TradeDistributionResearch,
)
from src.research.tiered_signal_framework_research import (
    TIMEFRAME_MINUTES,
    TierSignal,
    TieredSignalFrameworkResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_edge_extraction.json"

COHORT_FRACTION = 0.20
FVG_LOOKBACK = 60
EXPANSION_THRESHOLD = 50
TOP_TRAIT_COUNT = 10

FVG_SIZE_BUCKETS = (
    ("Small (<15 pts)", 0, 15),
    ("Medium (15-30 pts)", 15, 30),
    ("Large (30-60 pts)", 30, 60),
    ("XLarge (60+ pts)", 60, float("inf")),
)

FRESHNESS_BUCKETS = (
    ("Fresh (<=5 bars)", 0, 6),
    ("Recent (6-15 bars)", 6, 16),
    ("Mature (16-30 bars)", 16, 31),
    ("Stale (31+ bars)", 31, float("inf")),
)

DISTANCE_BUCKETS = (
    ("Very Close (<20 pts)", 0, 20),
    ("Close (20-50 pts)", 20, 50),
    ("Moderate (50-100 pts)", 50, 100),
    ("Far (100+ pts)", 100, float("inf")),
)

TIMING_BUCKETS = (
    ("Fast (<30 min)", 0, 30),
    ("Moderate (30-90 min)", 30, 90),
    ("Slow (90-240 min)", 90, 240),
    ("Extended (240+ min)", 240, float("inf")),
)

EXPANSION_SIZE_BUCKETS = (
    ("Small (<100 pts)", 0, 100),
    ("Medium (100-200 pts)", 100, 200),
    ("Large (200-400 pts)", 200, 400),
    ("XLarge (400+ pts)", 400, float("inf")),
)

EXPANSION_SPEED_BUCKETS = (
    ("Slow (<0.5 pts/min)", 0, 0.5),
    ("Moderate (0.5-1.5 pts/min)", 0.5, 1.5),
    ("Fast (1.5-3 pts/min)", 1.5, 3.0),
    ("Very Fast (3+ pts/min)", 3.0, float("inf")),
)


class InstitutionalEdgeExtractionError(Exception):
    """Raised when institutional edge extraction fails."""


@dataclass(frozen=True)
class EdgeFeatureRecord:
    """Structural features for one Tier-2 signal."""

    timeframe: str
    direction: str
    bos_timestamp: str
    realized_pnl_points: float
    risk_points: float
    win: bool
    displacement_strength: str
    fvg_size_points: float
    fvg_freshness_bars: int
    fvg_retests: int
    distance_from_liquidity_pool_points: float
    distance_from_swing_points: float
    choch_to_bos_minutes: float
    bos_to_fvg_reclaim_minutes: float
    expansion_speed_points_per_minute: float
    expansion_size_points: float
    trait_tags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraitComparison:
    """Frequency comparison for one trait between cohorts."""

    trait: str
    top_winners_pct: float
    bottom_losers_pct: float
    delta_pct: float
    top_winners_count: int
    bottom_losers_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalEdgeExtractionReport:
    """Full institutional edge extraction output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    total_signals: int
    cohort_size: int
    top_winners_summary: dict[str, Any]
    bottom_losers_summary: dict[str, Any]
    feature_comparison: dict[str, dict[str, Any]]
    top_10_winning_traits: list[dict[str, Any]]
    top_10_losing_traits: list[dict[str, Any]]
    recommended_quality_scoring_model: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float
    sample_top_winners: list[dict[str, Any]] = field(default_factory=list)
    sample_bottom_losers: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalEdgeExtractionResearch:
    """Extract institutional edge traits from Raw Tier-2 signals."""

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
        self.distribution_engine = Tier2TradeDistributionResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)

    @staticmethod
    def _is_active(value: Any) -> bool:
        from src.signals.decision_engine import DecisionEngine

        return DecisionEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        return LiquidityNarrativeEngine._to_float(value)

    @staticmethod
    def _bucket_label(value: float, buckets: tuple[tuple[str, float, float], ...]) -> str:
        for label, lower, upper in buckets:
            if lower <= value < upper:
                return label
        return buckets[-1][0]

    @staticmethod
    def _bars_to_minutes(bars: int, timeframe_label: str) -> float:
        return round(bars * TIMEFRAME_MINUTES.get(timeframe_label, 5), 1)

    def _fvg_columns(self, direction: str) -> tuple[str, str]:
        if direction == "bullish":
            return "Bullish_FVG_Top", "Bullish_FVG_Bottom"
        return "Bearish_FVG_Top", "Bearish_FVG_Bottom"

    def _fvg_bounds_at_bar(
        self,
        frame: pd.DataFrame,
        index: int,
        direction: str,
    ) -> tuple[float, float] | None:
        top_col, bottom_col = self._fvg_columns(direction)
        top = self._to_float(frame.iloc[index].get(top_col))
        bottom = self._to_float(frame.iloc[index].get(bottom_col))
        if top is None or bottom is None:
            return None
        return bottom, top

    def _find_fvg_creation_bar(
        self,
        frame: pd.DataFrame,
        bos_bar: int,
        direction: str,
    ) -> int | None:
        top_col, bottom_col = self._fvg_columns(direction)
        for index in range(bos_bar, max(-1, bos_bar - FVG_LOOKBACK) - 1, -1):
            row = frame.iloc[index]
            previous = frame.iloc[index - 1] if index > 0 else None
            if self.narrative_engine._fvg_created_on_bar(row, previous):
                if direction == "bullish" and self._is_active(row.get(top_col)):
                    return index
                if direction == "bearish" and self._is_active(row.get(top_col)):
                    return index
        for index in range(bos_bar, max(-1, bos_bar - FVG_LOOKBACK) - 1, -1):
            if self._fvg_bounds_at_bar(frame, index, direction):
                return index
        return None

    def _count_fvg_retests(
        self,
        frame: pd.DataFrame,
        creation_bar: int,
        bos_bar: int,
        direction: str,
        bottom: float,
        top: float,
    ) -> int:
        retests = 0
        outside = True
        for index in range(creation_bar + 1, bos_bar + 1):
            row = frame.iloc[index]
            low = self._to_float(row.get("Low")) or 0.0
            high = self._to_float(row.get("High")) or 0.0
            close = self._to_float(row.get("Close")) or 0.0

            if direction == "bullish":
                inside = low <= top and close >= bottom
            else:
                inside = high >= bottom and close <= top

            if inside and outside:
                retests += 1
            outside = not inside
        return max(retests - 1, 0)

    def _distance_from_liquidity_pool(
        self,
        row: pd.Series,
        direction: str,
    ) -> float:
        close = self._to_float(row.get("Close")) or 0.0
        buy_pool = self._to_float(row.get("Buy_Side_Liquidity"))
        sell_pool = self._to_float(row.get("Sell_Side_Liquidity"))

        if direction == "bullish" and sell_pool is not None:
            return round(abs(close - sell_pool), 2)
        if direction == "bearish" and buy_pool is not None:
            return round(abs(close - buy_pool), 2)

        candidates = [value for value in (buy_pool, sell_pool) if value is not None]
        if not candidates:
            return 0.0
        return round(min(abs(close - value) for value in candidates), 2)

    def _distance_from_swing(
        self,
        row: pd.Series,
        direction: str,
    ) -> float:
        close = self._to_float(row.get("Close")) or 0.0
        swing_low = self._to_float(row.get("Swing_Low"))
        swing_high = self._to_float(row.get("Swing_High"))

        if direction == "bullish" and swing_low is not None:
            return round(abs(close - swing_low), 2)
        if direction == "bearish" and swing_high is not None:
            return round(abs(close - swing_high), 2)

        candidates = [value for value in (swing_low, swing_high) if value is not None]
        if not candidates:
            return 0.0
        return round(min(abs(close - value) for value in candidates), 2)

    def _bos_to_fvg_reclaim_minutes(
        self,
        frame: pd.DataFrame,
        bos_bar: int,
        direction: str,
        timeframe_label: str,
    ) -> float:
        for offset in range(0, 6):
            index = bos_bar + offset
            if index >= len(frame):
                break
            if self.tier_engine._fvg_reclaimed_at_bar(frame, index, direction):
                return self._bars_to_minutes(offset, timeframe_label)
        return self._bars_to_minutes(0, timeframe_label)

    def _expansion_metrics(
        self,
        frame: pd.DataFrame,
        bos_bar: int,
        direction: str,
        timeframe_label: str,
    ) -> tuple[float, float]:
        detail = self.distribution_engine._simulate_detailed(
            frame,
            TierSignal(
                tier="tier_2",
                timeframe=timeframe_label,
                direction=direction,
                bos_bar=bos_bar,
                bos_timestamp=str(frame.iloc[bos_bar]["Date"]),
            ),
        )
        if detail is None:
            return 0.0, 0.0

        expansion_size = detail.mfe_points
        minutes = detail.minutes_to_target or detail.minutes_to_stop
        if minutes and minutes > 0:
            speed = round(expansion_size / minutes, 2)
        else:
            speed = 0.0
        return expansion_size, speed

    def _build_trait_tags(self, record: EdgeFeatureRecord) -> tuple[str, ...]:
        tags: list[str] = []
        tags.append(f"Displacement {record.displacement_strength}")
        tags.append(f"FVG Size {self._bucket_label(record.fvg_size_points, FVG_SIZE_BUCKETS)}")
        tags.append(
            f"FVG Freshness {self._bucket_label(float(record.fvg_freshness_bars), FRESHNESS_BUCKETS)}"
        )
        tags.append(
            "FVG Retests 0"
            if record.fvg_retests == 0
            else ("FVG Retests 1" if record.fvg_retests == 1 else "FVG Retests 2+")
        )
        tags.append(
            "Liquidity Distance "
            + self._bucket_label(record.distance_from_liquidity_pool_points, DISTANCE_BUCKETS)
        )
        tags.append(
            "Swing Distance "
            + self._bucket_label(record.distance_from_swing_points, DISTANCE_BUCKETS)
        )
        tags.append(
            "CHOCH->BOS "
            + self._bucket_label(record.choch_to_bos_minutes, TIMING_BUCKETS)
        )
        tags.append(
            "BOS->FVG Reclaim "
            + self._bucket_label(record.bos_to_fvg_reclaim_minutes, TIMING_BUCKETS)
        )
        tags.append(
            "Expansion Speed "
            + self._bucket_label(record.expansion_speed_points_per_minute, EXPANSION_SPEED_BUCKETS)
        )
        tags.append(
            "Expansion Size "
            + self._bucket_label(record.expansion_size_points, EXPANSION_SIZE_BUCKETS)
        )
        return tuple(tags)

    def _extract_features(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
        realized_pnl: float,
        win: bool,
        risk_points: float,
    ) -> EdgeFeatureRecord | None:
        bos_bar = signal.bos_bar
        if bos_bar >= len(frame):
            return None

        direction = signal.direction
        row = frame.iloc[bos_bar]
        choch_bar = signal.choch_bar
        displacement_bar = signal.displacement_bar or bos_bar

        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(
            frame.iloc[displacement_bar],
            direction,
        )

        creation_bar = self._find_fvg_creation_bar(frame, bos_bar, direction)
        bounds = self._fvg_bounds_at_bar(frame, bos_bar, direction)
        if bounds is None and creation_bar is not None:
            bounds = self._fvg_bounds_at_bar(frame, creation_bar, direction)
        if bounds is None:
            return None

        bottom, top = bounds
        fvg_size = round(top - bottom, 2)
        freshness_bars = bos_bar - creation_bar if creation_bar is not None else 0
        retests = (
            self._count_fvg_retests(frame, creation_bar, bos_bar, direction, bottom, top)
            if creation_bar is not None
            else 0
        )

        choch_to_bos = 0.0
        if choch_bar is not None:
            choch_to_bos = self._bars_to_minutes(bos_bar - choch_bar, signal.timeframe)

        bos_to_reclaim = self._bos_to_fvg_reclaim_minutes(
            frame,
            bos_bar,
            direction,
            signal.timeframe,
        )
        expansion_size, expansion_speed = self._expansion_metrics(
            frame,
            bos_bar,
            direction,
            signal.timeframe,
        )

        record_without_tags = EdgeFeatureRecord(
            timeframe=signal.timeframe,
            direction=direction,
            bos_timestamp=signal.bos_timestamp,
            realized_pnl_points=realized_pnl,
            risk_points=risk_points,
            win=win,
            displacement_strength=displacement.value,
            fvg_size_points=fvg_size,
            fvg_freshness_bars=freshness_bars,
            fvg_retests=retests,
            distance_from_liquidity_pool_points=self._distance_from_liquidity_pool(row, direction),
            distance_from_swing_points=self._distance_from_swing(row, direction),
            choch_to_bos_minutes=choch_to_bos,
            bos_to_fvg_reclaim_minutes=bos_to_reclaim,
            expansion_speed_points_per_minute=expansion_speed,
            expansion_size_points=expansion_size,
            trait_tags=(),
        )
        return EdgeFeatureRecord(
            **{
                **asdict(record_without_tags),
                "trait_tags": self._build_trait_tags(record_without_tags),
            }
        )

    @staticmethod
    def _numeric_summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(mean(values), 2),
            "median": round(median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    def _cohort_summary(self, records: list[EdgeFeatureRecord]) -> dict[str, Any]:
        return {
            "count": len(records),
            "average_pnl": round(mean(r.realized_pnl_points for r in records), 2) if records else 0.0,
            "win_rate_pct": round(sum(1 for r in records if r.win) / len(records) * 100, 2)
            if records
            else 0.0,
            "displacement_strength": dict(
                Counter(r.displacement_strength for r in records)
            ),
            "fvg_size": self._numeric_summary([r.fvg_size_points for r in records]),
            "fvg_freshness_bars": self._numeric_summary(
                [float(r.fvg_freshness_bars) for r in records]
            ),
            "fvg_retests": self._numeric_summary([float(r.fvg_retests) for r in records]),
            "distance_from_liquidity_pool": self._numeric_summary(
                [r.distance_from_liquidity_pool_points for r in records]
            ),
            "distance_from_swing": self._numeric_summary(
                [r.distance_from_swing_points for r in records]
            ),
            "choch_to_bos_minutes": self._numeric_summary(
                [r.choch_to_bos_minutes for r in records]
            ),
            "bos_to_fvg_reclaim_minutes": self._numeric_summary(
                [r.bos_to_fvg_reclaim_minutes for r in records]
            ),
            "expansion_speed": self._numeric_summary(
                [r.expansion_speed_points_per_minute for r in records]
            ),
            "expansion_size": self._numeric_summary([r.expansion_size_points for r in records]),
        }

    def _trait_comparisons(
        self,
        top: list[EdgeFeatureRecord],
        bottom: list[EdgeFeatureRecord],
    ) -> list[TraitComparison]:
        top_tags: list[str] = [tag for record in top for tag in record.trait_tags]
        bottom_tags: list[str] = [tag for record in bottom for tag in record.trait_tags]
        all_traits = sorted(set(top_tags) | set(bottom_tags))

        comparisons: list[TraitComparison] = []
        for trait in all_traits:
            top_count = top_tags.count(trait)
            bottom_count = bottom_tags.count(trait)
            top_pct = round(top_count / len(top) * 100, 2) if top else 0.0
            bottom_pct = round(bottom_count / len(bottom) * 100, 2) if bottom else 0.0
            comparisons.append(
                TraitComparison(
                    trait=trait,
                    top_winners_pct=top_pct,
                    bottom_losers_pct=bottom_pct,
                    delta_pct=round(top_pct - bottom_pct, 2),
                    top_winners_count=top_count,
                    bottom_losers_count=bottom_count,
                )
            )
        return comparisons

    def _quality_scoring_model(
        self,
        winning_traits: list[TraitComparison],
    ) -> dict[str, Any]:
        pre_trade_prefixes = (
            "Displacement ",
            "FVG Size ",
            "FVG Freshness ",
            "FVG Retests ",
            "Liquidity Distance ",
            "Swing Distance ",
            "CHOCH->BOS ",
            "BOS->FVG Reclaim ",
        )
        eligible = [
            trait
            for trait in winning_traits
            if trait.trait.startswith(pre_trade_prefixes)
        ]

        total_weight = 0
        components: list[dict[str, Any]] = []
        for trait in eligible[:8]:
            weight = max(5, min(20, int(abs(trait.delta_pct))))
            total_weight += weight
            components.append(
                {
                    "trait": trait.trait,
                    "weight_points": weight,
                    "condition": f"Award {weight} points when {trait.trait} is present",
                    "winner_frequency_pct": trait.top_winners_pct,
                    "loser_frequency_pct": trait.bottom_losers_pct,
                    "edge_delta_pct": trait.delta_pct,
                }
            )

        scale = 100 / total_weight if total_weight else 1.0
        for component in components:
            component["normalized_weight"] = round(component["weight_points"] * scale, 1)

        return {
            "model_name": "Tier-2 Institutional Quality Score",
            "max_score": 100,
            "usage": "Research-only pre-filter using pre-trade structural traits at BOS confirmation.",
            "components": components,
            "minimum_recommended_score": 55,
            "scoring_notes": [
                "Weights derived from top-20% vs bottom-20% trait frequency deltas.",
                "Pre-trade traits only; expansion metrics excluded from scoring (post-confirmation).",
                "Strong displacement, moderate CHOCH->BOS timing, and single FVG retest favor winners.",
            ],
        }

    def _feature_comparison_dict(
        self,
        top: list[EdgeFeatureRecord],
        bottom: list[EdgeFeatureRecord],
    ) -> dict[str, dict[str, Any]]:
        top_summary = self._cohort_summary(top)
        bottom_summary = self._cohort_summary(bottom)
        keys = [
            "fvg_size",
            "fvg_freshness_bars",
            "fvg_retests",
            "distance_from_liquidity_pool",
            "distance_from_swing",
            "choch_to_bos_minutes",
            "bos_to_fvg_reclaim_minutes",
            "expansion_speed",
            "expansion_size",
        ]
        comparison: dict[str, dict[str, Any]] = {}
        for key in keys:
            top_mean = top_summary[key]["mean"]
            bottom_mean = bottom_summary[key]["mean"]
            comparison[key] = {
                "top_winners_mean": top_mean,
                "bottom_losers_mean": bottom_mean,
                "delta_mean": round(top_mean - bottom_mean, 2),
                "top_winners": top_summary[key],
                "bottom_losers": bottom_summary[key],
            }
        comparison["displacement_strength"] = {
            "top_winners": top_summary["displacement_strength"],
            "bottom_losers": bottom_summary["displacement_strength"],
        }
        return comparison

    def _collect_records(self, metadata: dict[str, Any]) -> list[EdgeFeatureRecord]:
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

        records: list[EdgeFeatureRecord] = []
        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)

            for signal in self.tier_engine._detect_tier2(frame, timeframe_label):
                detail = self.distribution_engine._simulate_detailed(frame, signal)
                if detail is None:
                    continue
                feature = self._extract_features(
                    frame,
                    signal,
                    detail.realized_pnl_points,
                    detail.win,
                    detail.risk_points,
                )
                if feature:
                    records.append(feature)

        records.sort(key=lambda item: pd.Timestamp(item.bos_timestamp))
        return records

    def run(self, metadata: dict[str, Any]) -> InstitutionalEdgeExtractionReport:
        """Run institutional edge extraction research."""
        started = time.perf_counter()

        records = self._collect_records(metadata)
        if not records:
            raise InstitutionalEdgeExtractionError("No Tier-2 feature records extracted.")

        cohort_size = max(1, int(len(records) * COHORT_FRACTION))
        ranked = sorted(records, key=lambda item: item.realized_pnl_points, reverse=True)
        top_winners = ranked[:cohort_size]
        bottom_losers = ranked[-cohort_size:]

        comparisons = self._trait_comparisons(top_winners, bottom_losers)
        winning_sorted = sorted(comparisons, key=lambda item: item.delta_pct, reverse=True)
        losing_sorted = sorted(comparisons, key=lambda item: item.delta_pct)

        top_10_winning = [item.as_dict() for item in winning_sorted[:TOP_TRAIT_COUNT]]
        top_10_losing = [item.as_dict() for item in losing_sorted[:TOP_TRAIT_COUNT]]
        quality_model = self._quality_scoring_model(winning_sorted)

        conclusions = [
            f"Analyzed {len(records)} Raw Tier-2 signals; cohort size {cohort_size} per tail.",
            (
                f"Top 20% avg PnL {round(mean(r.realized_pnl_points for r in top_winners), 2)} vs "
                f"bottom 20% {round(mean(r.realized_pnl_points for r in bottom_losers), 2)}."
            ),
        ]
        if top_10_winning:
            conclusions.append(
                f"Strongest winning edge: {top_10_winning[0]['trait']} "
                f"(+{top_10_winning[0]['delta_pct']} pp vs losers)."
            )
        if top_10_losing:
            conclusions.append(
                f"Strongest losing marker: {top_10_losing[0]['trait']} "
                f"({top_10_losing[0]['delta_pct']} pp vs winners)."
            )

        return InstitutionalEdgeExtractionReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            total_signals=len(records),
            cohort_size=cohort_size,
            top_winners_summary=self._cohort_summary(top_winners),
            bottom_losers_summary=self._cohort_summary(bottom_losers),
            feature_comparison=self._feature_comparison_dict(top_winners, bottom_losers),
            top_10_winning_traits=top_10_winning,
            top_10_losing_traits=top_10_losing,
            recommended_quality_scoring_model=quality_model,
            conclusions=conclusions,
            sample_top_winners=[record.as_dict() for record in top_winners[:5]],
            sample_bottom_losers=[record.as_dict() for record in bottom_losers[:5]],
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_edge_extraction_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> InstitutionalEdgeExtractionReport:
    """Run institutional edge extraction and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalEdgeExtractionError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalEdgeExtractionResearch(
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
        "Institutional edge extraction completed: signals=%s",
        report.total_signals,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_edge_extraction_report()
        print("Institutional Edge Extraction Summary")
        print(f"Signals: {report.total_signals} | Cohort: {report.cohort_size}")
        print("Top 10 Winning Traits:")
        for index, trait in enumerate(report.top_10_winning_traits, start=1):
            print(f"  {index}. {trait['trait']} (+{trait['delta_pct']} pp)")
        print("Top 10 Losing Traits:")
        for index, trait in enumerate(report.top_10_losing_traits, start=1):
            print(f"  {index}. {trait['trait']} ({trait['delta_pct']} pp)")
        print(f"Quality model: {report.recommended_quality_scoring_model['model_name']}")
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalEdgeExtractionError as exc:
        logger.error("Institutional edge extraction error: %s", exc)
        print(f"Institutional edge extraction error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional edge extraction failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
