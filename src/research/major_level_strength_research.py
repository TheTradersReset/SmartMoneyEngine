"""
Major Level Strength research for SmartMoneyEngine.

Identifies which support and resistance levels are truly important and how
strength correlates with bounce, rejection, breakout, and breakdown outcomes.
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
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.research.support_resistance_pressure_research import (
    LEVEL_CLUSTER_POINTS,
    MajorLevel,
    SupportResistancePressureResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "major_level_strength.json"

ROUND_NUMBER_STEP = 100
ROUND_NUMBER_TOLERANCE = 25.0
OVERLAP_TOLERANCE = 15.0
VOLUME_EXPANSION_THRESHOLD = 1.3
GAP_MIN_POINTS = 0.5

STRENGTH_WEIGHTS: dict[str, float] = {
    "touches": 18.0,
    "days_survived": 8.0,
    "bars_near_level": 10.0,
    "reaction_history": 12.0,
    "liquidity_grabs": 8.0,
    "equal_levels_nearby": 10.0,
    "calendar_overlap": 15.0,
    "demand_supply_overlap": 10.0,
    "round_number_overlap": 12.0,
    "gap_interaction": 4.0,
    "volume_expansion": 5.0,
    "source_quality": 8.0,
}

SOURCE_SCORES: dict[str, float] = {
    "Swing_High": 1.0,
    "Swing_Low": 1.0,
    "Equal_High": 0.85,
    "Equal_Low": 0.85,
    "Buy_Side_Liquidity": 0.7,
    "Sell_Side_Liquidity": 0.7,
}


class MajorLevelStrengthError(Exception):
    """Raised when major level strength research fails."""


class StrengthCategory(str, Enum):
    """Level strength classification."""

    WEAK = "Weak"
    MODERATE = "Moderate"
    STRONG = "Strong"
    INSTITUTIONAL = "Institutional"


@dataclass(frozen=True)
class LevelStrengthFeatures:
    """Raw inputs used to compute level strength."""

    number_of_touches: int
    days_level_survived: int
    bars_near_level: int
    bounce_count: int
    rejection_count: int
    liquidity_grabs: int
    equal_highs_lows_nearby: int
    previous_day_overlap: bool
    weekly_overlap: bool
    monthly_overlap: bool
    demand_supply_zone_overlap: bool
    round_number_overlap: bool
    gap_interactions: int
    average_volume_expansion: float
    source_column: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LevelStrengthEvent:
    """One level reaction with strength score and outcome."""

    timeframe: str
    level_price: float
    level_side: str
    level_source: str
    formation_bar: int
    event_bar: int
    formation_timestamp: str
    event_timestamp: str
    outcome: str
    level_strength_score: float
    strength_category: str
    features: dict[str, Any]
    distance_traveled_after_reaction: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MajorLevelStrengthReport:
    """Aggregate major level strength research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    total_levels_analyzed: int
    total_reaction_events: int
    strength_category_distribution: dict[str, int]
    level_strength_matrix: dict[str, dict[str, float | None]]
    strength_score_components: dict[str, float]
    level_events: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MajorLevelStrengthResearch:
    """Score major S/R levels and correlate strength with reaction probabilities."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.filter_engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.pressure_engine = SupportResistancePressureResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.liquidity_map_engine = InstitutionalLiquidityMapEngine(symbol=symbol)

    @staticmethod
    def _classify_strength(score: float) -> str:
        if score >= 70:
            return StrengthCategory.INSTITUTIONAL.value
        if score >= 50:
            return StrengthCategory.STRONG.value
        if score >= 30:
            return StrengthCategory.MODERATE.value
        return StrengthCategory.WEAK.value

    @staticmethod
    def _round_number_overlap(level_price: float) -> bool:
        nearest_hundred = round(level_price / ROUND_NUMBER_STEP) * ROUND_NUMBER_STEP
        return abs(level_price - nearest_hundred) <= ROUND_NUMBER_TOLERANCE

    @staticmethod
    def _level_overlaps(value: float | None, level_price: float, tolerance: float = OVERLAP_TOLERANCE) -> bool:
        if value is None:
            return False
        return abs(float(value) - level_price) <= tolerance

    @staticmethod
    def _demand_supply_overlap(frame: pd.DataFrame, index: int, level_price: float) -> bool:
        row = frame.iloc[index]
        zones: list[tuple[float, float]] = []
        for low_col, high_col in (
            ("Bullish_OB_Low", "Bullish_OB_High"),
            ("Bearish_OB_Low", "Bearish_OB_High"),
        ):
            low = MajorLevelStrengthResearch._to_float(row.get(low_col))
            high = MajorLevelStrengthResearch._to_float(row.get(high_col))
            if low is not None and high is not None:
                zones.append((min(low, high), max(low, high)))
        for start, end in zones:
            if start - OVERLAP_TOLERANCE <= level_price <= end + OVERLAP_TOLERANCE:
                return True
        return False

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _count_equal_levels_nearby(
        self,
        frame: pd.DataFrame,
        formation_bar: int,
        level_price: float,
        level_side: str,
    ) -> int:
        start = max(0, formation_bar - 100)
        window = frame.iloc[start : formation_bar + 1]
        columns = ("Equal_High", "Buy_Side_Liquidity") if level_side == "resistance" else (
            "Equal_Low",
            "Sell_Side_Liquidity",
        )
        count = 0
        seen: set[float] = set()
        for column in columns:
            if column not in window.columns:
                continue
            for value in window[column]:
                parsed = self._to_float(value)
                if parsed is None:
                    continue
                if abs(parsed - level_price) <= LEVEL_CLUSTER_POINTS and parsed not in seen:
                    seen.add(parsed)
                    count += 1
        return count

    def _calendar_overlaps(
        self,
        enriched: pd.DataFrame,
        index: int,
        level_price: float,
    ) -> tuple[bool, bool, bool]:
        row = enriched.iloc[index]
        pd_overlap = (
            self._level_overlaps(self._to_float(row.get("_pdh")), level_price)
            or self._level_overlaps(self._to_float(row.get("_pdl")), level_price)
        )
        weekly_overlap = (
            self._level_overlaps(self._to_float(row.get("_pwh")), level_price)
            or self._level_overlaps(self._to_float(row.get("_pwl")), level_price)
        )
        monthly_overlap = (
            self._level_overlaps(self._to_float(row.get("_pmh")), level_price)
            or self._level_overlaps(self._to_float(row.get("_pml")), level_price)
        )
        return pd_overlap, weekly_overlap, monthly_overlap

    def _days_survived(
        self,
        frame: pd.DataFrame,
        formation_bar: int,
        event_bar: int,
    ) -> int:
        start_ts = pd.Timestamp(frame.iloc[formation_bar]["Date"])
        end_ts = pd.Timestamp(frame.iloc[event_bar]["Date"])
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert("Asia/Kolkata")
            end_ts = end_ts.tz_convert("Asia/Kolkata")
        return max((end_ts.date() - start_ts.date()).days, 0)

    def _volume_expansion_near_level(
        self,
        frame: pd.DataFrame,
        formation_bar: int,
        event_bar: int,
        level_price: float,
    ) -> float:
        expansions: list[float] = []
        for index in range(formation_bar + 1, event_bar + 1):
            atr = self.pressure_engine._atr(frame, index)
            touch_band = atr * 0.5
            row = frame.iloc[index]
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            near = abs(close - level_price) <= touch_band or (
                low - touch_band <= level_price <= high + touch_band
            )
            if not near:
                continue
            volume = self._to_float(row.get("Volume")) or 0.0
            vol_start = max(0, index - 20)
            avg_volume = mean(
                self._to_float(frame.iloc[offset].get("Volume")) or 0.0
                for offset in range(vol_start, index)
            ) if index > vol_start else volume
            if avg_volume > 0:
                expansions.append(volume / avg_volume)
        return round(mean(expansions), 2) if expansions else 1.0

    def _gap_interactions(
        self,
        frame: pd.DataFrame,
        formation_bar: int,
        event_bar: int,
        level_price: float,
    ) -> int:
        count = 0
        for index in range(max(formation_bar + 1, 1), event_bar + 1):
            atr = self.pressure_engine._atr(frame, index)
            touch_band = atr * 0.5
            row = frame.iloc[index]
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            near = abs(close - level_price) <= touch_band or (
                low - touch_band <= level_price <= high + touch_band
            )
            if not near:
                continue
            prev_close = float(frame.iloc[index - 1]["Close"])
            gap = float(row["Open"]) - prev_close
            if abs(gap) >= GAP_MIN_POINTS:
                count += 1
        return count

    def _build_features(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        level: MajorLevel,
        pressure_record: dict[str, Any],
        bounce_count: int,
        rejection_count: int,
    ) -> LevelStrengthFeatures:
        event_bar = int(pressure_record["event_bar"])
        pd_overlap, weekly_overlap, monthly_overlap = self._calendar_overlaps(
            enriched,
            level.formation_bar,
            level.level_price,
        )
        return LevelStrengthFeatures(
            number_of_touches=int(pressure_record["number_of_tests"]),
            days_level_survived=self._days_survived(frame, level.formation_bar, event_bar),
            bars_near_level=int(pressure_record["bars_near_level"]),
            bounce_count=bounce_count,
            rejection_count=rejection_count,
            liquidity_grabs=int(pressure_record["liquidity_grabs"]),
            equal_highs_lows_nearby=self._count_equal_levels_nearby(
                frame,
                level.formation_bar,
                level.level_price,
                level.level_side,
            ),
            previous_day_overlap=pd_overlap,
            weekly_overlap=weekly_overlap,
            monthly_overlap=monthly_overlap,
            demand_supply_zone_overlap=self._demand_supply_overlap(
                frame,
                level.formation_bar,
                level.level_price,
            ),
            round_number_overlap=self._round_number_overlap(level.level_price),
            gap_interactions=self._gap_interactions(
                frame,
                level.formation_bar,
                event_bar,
                level.level_price,
            ),
            average_volume_expansion=self._volume_expansion_near_level(
                frame,
                level.formation_bar,
                event_bar,
                level.level_price,
            ),
            source_column=level.source_column,
        )

    def _compute_strength_score(self, features: LevelStrengthFeatures) -> float:
        touch_score = min(features.number_of_touches / 3.0, 1.0) * STRENGTH_WEIGHTS["touches"]
        days_score = min(features.days_level_survived / 5.0, 1.0) * STRENGTH_WEIGHTS["days_survived"]
        near_score = min(features.bars_near_level / 10.0, 1.0) * STRENGTH_WEIGHTS["bars_near_level"]
        reaction_score = (
            min((features.bounce_count + features.rejection_count) / 2.0, 1.0)
            * STRENGTH_WEIGHTS["reaction_history"]
        )
        liquidity_score = min(features.liquidity_grabs / 2.0, 1.0) * STRENGTH_WEIGHTS["liquidity_grabs"]
        equal_score = min(features.equal_highs_lows_nearby / 1.0, 1.0) * STRENGTH_WEIGHTS["equal_levels_nearby"]

        calendar_points = 0.0
        if features.previous_day_overlap:
            calendar_points += 5.0
        if features.weekly_overlap:
            calendar_points += 5.0
        if features.monthly_overlap:
            calendar_points += 5.0
        calendar_score = min(calendar_points, STRENGTH_WEIGHTS["calendar_overlap"])

        demand_score = (
            STRENGTH_WEIGHTS["demand_supply_overlap"] if features.demand_supply_zone_overlap else 0.0
        )
        round_score = STRENGTH_WEIGHTS["round_number_overlap"] if features.round_number_overlap else 0.0
        gap_score = min(features.gap_interactions / 1.0, 1.0) * STRENGTH_WEIGHTS["gap_interaction"]
        volume_score = (
            min(max(features.average_volume_expansion - 1.0, 0.0) / 0.5, 1.0)
            * STRENGTH_WEIGHTS["volume_expansion"]
        )
        source_score = SOURCE_SCORES.get(features.source_column, 0.5) * STRENGTH_WEIGHTS["source_quality"]

        total = (
            touch_score
            + days_score
            + near_score
            + reaction_score
            + liquidity_score
            + equal_score
            + calendar_score
            + demand_score
            + round_score
            + gap_score
            + volume_score
            + source_score
        )
        return round(min(total, 100.0), 2)

    def _build_matrix(self, events: list[LevelStrengthEvent]) -> dict[str, dict[str, float | None]]:
        categories = [item.value for item in StrengthCategory]
        matrix: dict[str, dict[str, float | None]] = {
            category: {
                "bounce_probability_pct": None,
                "rejection_probability_pct": None,
                "breakout_probability_pct": None,
                "breakdown_probability_pct": None,
                "sample_support_interactions": 0,
                "sample_resistance_interactions": 0,
            }
            for category in categories
        }

        grouped: dict[str, list[LevelStrengthEvent]] = defaultdict(list)
        for event in events:
            grouped[event.strength_category].append(event)

        for category in categories:
            bucket = grouped.get(category, [])
            support_bounce = sum(1 for item in bucket if item.outcome == "support_bounce")
            support_break = sum(1 for item in bucket if item.outcome == "support_break")
            resistance_rejection = sum(1 for item in bucket if item.outcome == "resistance_rejection")
            resistance_break = sum(1 for item in bucket if item.outcome == "resistance_break")

            support_total = support_bounce + support_break
            resistance_total = resistance_rejection + resistance_break

            matrix[category]["sample_support_interactions"] = support_total
            matrix[category]["sample_resistance_interactions"] = resistance_total
            matrix[category]["bounce_probability_pct"] = (
                round((support_bounce / support_total) * 100, 2) if support_total else None
            )
            matrix[category]["rejection_probability_pct"] = (
                round((resistance_rejection / resistance_total) * 100, 2) if resistance_total else None
            )
            matrix[category]["breakout_probability_pct"] = (
                round((resistance_break / resistance_total) * 100, 2) if resistance_total else None
            )
            matrix[category]["breakdown_probability_pct"] = (
                round((support_break / support_total) * 100, 2) if support_total else None
            )

        return matrix

    def _collect_events(self, metadata: dict[str, Any]) -> list[LevelStrengthEvent]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        events: list[LevelStrengthEvent] = []
        seen: set[tuple[int, float, str]] = set()

        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            enriched = self.liquidity_map_engine._attach_calendar_levels(frame)
            levels = self.pressure_engine._extract_levels(frame)
            logger.info(
                "Major level strength: %s levels=%s",
                timeframe_label,
                len(levels),
            )

            for level in levels:
                pressure_records = self.pressure_engine._evaluate_level(frame, level, timeframe_label)
                if not pressure_records:
                    continue

                bounce_count = sum(
                    1 for record in pressure_records if record.outcome == "support_bounce"
                )
                rejection_count = sum(
                    1 for record in pressure_records if record.outcome == "resistance_rejection"
                )

                for record in pressure_records:
                    event_key = (record.event_bar, record.level_price, record.outcome)
                    if event_key in seen:
                        continue
                    seen.add(event_key)

                    features = self._build_features(
                        frame,
                        enriched,
                        level,
                        record.as_dict(),
                        bounce_count,
                        rejection_count,
                    )
                    score = self._compute_strength_score(features)
                    events.append(
                        LevelStrengthEvent(
                            timeframe=timeframe_label,
                            level_price=level.level_price,
                            level_side=level.level_side,
                            level_source=level.source_column,
                            formation_bar=level.formation_bar,
                            event_bar=record.event_bar,
                            formation_timestamp=level.formation_timestamp,
                            event_timestamp=record.event_timestamp,
                            outcome=record.outcome,
                            level_strength_score=score,
                            strength_category=self._classify_strength(score),
                            features=features.as_dict(),
                            distance_traveled_after_reaction=record.distance_traveled_after_event,
                        ),
                    )

        return events

    def run(self, metadata: dict[str, Any]) -> MajorLevelStrengthReport:
        started = time.perf_counter()
        events = self._collect_events(metadata)

        category_counts = Counter(event.strength_category for event in events)
        matrix = self._build_matrix(events)

        conclusions = [
            f"Scored {len(events)} level reaction events across {list(self.timeframes)}.",
            "Level strength matrix:",
        ]
        for category in StrengthCategory:
            row = matrix[category.value]
            conclusions.append(
                f"  {category.value}: bounce={row['bounce_probability_pct']}% "
                f"rejection={row['rejection_probability_pct']}% "
                f"breakout={row['breakout_probability_pct']}% "
                f"breakdown={row['breakdown_probability_pct']}%",
            )

        best_bounce = max(
            StrengthCategory,
            key=lambda cat: matrix[cat.value]["bounce_probability_pct"] or -1,
        )
        conclusions.append(f"Best bounce probability: {best_bounce.value}.")
        best_rejection = max(
            StrengthCategory,
            key=lambda cat: matrix[cat.value]["rejection_probability_pct"] or -1,
        )
        conclusions.append(f"Best rejection probability: {best_rejection.value}.")

        return MajorLevelStrengthReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            total_levels_analyzed=len({(e.level_price, e.level_side, e.formation_bar) for e in events}),
            total_reaction_events=len(events),
            strength_category_distribution=dict(category_counts),
            level_strength_matrix=matrix,
            strength_score_components=STRENGTH_WEIGHTS,
            level_events=[event.as_dict() for event in events],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_major_level_strength_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> MajorLevelStrengthReport:
    """Run major level strength research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise MajorLevelStrengthError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = MajorLevelStrengthResearch(
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
        "Major level strength research completed: events=%s",
        report.total_reaction_events,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_major_level_strength_report()
        print("Major Level Strength Research Summary")
        print(f"Reaction events: {report.total_reaction_events}")
        print("Strength distribution:", report.strength_category_distribution)
        print("Level Strength Matrix:")
        for category, row in report.level_strength_matrix.items():
            print(
                f"  {category}: bounce={row['bounce_probability_pct']}% "
                f"rejection={row['rejection_probability_pct']}% "
                f"breakout={row['breakout_probability_pct']}% "
                f"breakdown={row['breakdown_probability_pct']}%",
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except MajorLevelStrengthError as exc:
        logger.error("Major level strength research error: %s", exc)
        print(f"Major level strength research error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected major level strength research failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
