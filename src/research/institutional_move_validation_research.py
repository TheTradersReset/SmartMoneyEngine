"""
Institutional move validation research for SmartMoneyEngine.

Validates whether the full institutional sequence (Liquidity Sweep -> Displacement
-> CHOCH -> BOS -> FVG Reclaim) precedes large moves better than isolated
components. Research-only; no trades, signals, or setup changes.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import (
    DisplacementStrength,
    FvgContext,
    LiquidityNarrativeEngine,
)
from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_move_validation.json"

MOVE_THRESHOLDS = (50, 100, 150)
FORWARD_BARS = 80
SEQUENCE_LOOKBACK = 40
MIN_EVENT_SEPARATION = 15
RANDOM_SAMPLE_SIZE = 500
RANDOM_SEED = 42

TIMEFRAME_MINUTES = {"5M": 5, "15M": 15, "1H": 60}

EVENT_MAP = {
    "bullish": {
        "sweep": "Sell_Liquidity_Sweep",
        "choch": "Bullish_CHOCH",
        "bos": "Bullish_BOS",
        "fvg_bias": "bullish",
    },
    "bearish": {
        "sweep": "Buy_Liquidity_Sweep",
        "choch": "Bearish_CHOCH",
        "bos": "Bearish_BOS",
        "fvg_bias": "bearish",
    },
}


class InstitutionalMoveValidationError(Exception):
    """Raised when institutional move validation fails."""


@dataclass(frozen=True)
class SequenceInstance:
    """One detected institutional sequence anchored at BOS confirmation."""

    timeframe: str
    direction: str
    sweep_bar: int
    displacement_bar: int
    choch_bar: int
    bos_bar: int
    sweep_timestamp: str
    bos_timestamp: str
    displacement_strength: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForwardMoveOutcome:
    """Forward price move measured after an anchor bar."""

    timeframe: str
    direction: str
    anchor_bar: int
    anchor_timestamp: str
    anchor_type: str
    forward_move_points: float
    expansion_bar: int | None
    bos_to_expansion_bars: int | None
    bos_to_expansion_minutes: float | None
    moved_over_50: bool
    moved_over_100: bool
    moved_over_150: bool
    directional_win: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CohortMetrics:
    """Performance metrics for one event cohort."""

    label: str
    occurrences: int
    moves_over_50: int
    moves_over_100: int
    moves_over_150: int
    pct_moves_over_50: float
    pct_moves_over_100: float
    pct_moves_over_150: float
    average_move_size: float
    win_rate_pct: float
    win_rate_by_direction: dict[str, float]
    average_bos_to_expansion_minutes: float | None
    sample_size_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalMoveValidationReport:
    """Aggregate institutional move validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    institutional_sequence: dict[str, Any]
    comparison: dict[str, dict[str, Any]]
    sequence_outperforms_components: dict[str, bool]
    win_rate_by_direction: dict[str, dict[str, float]]
    conclusions: list[str]
    execution_time_seconds: float
    detected_sequences: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalMoveValidationResearch:
    """Validate institutional sequence predictive power vs isolated components."""

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
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _minutes_per_bar(timeframe_label: str) -> int:
        return TIMEFRAME_MINUTES.get(timeframe_label, 5)

    @staticmethod
    def _bars_to_minutes(bars: int | None, timeframe_label: str) -> float | None:
        if bars is None:
            return None
        return round(bars * InstitutionalMoveValidationResearch._minutes_per_bar(timeframe_label), 1)

    def _find_last_event_bar(
        self,
        frame: pd.DataFrame,
        column: str,
        before_index: int,
        lookback: int = SEQUENCE_LOOKBACK,
    ) -> int | None:
        start = max(0, before_index - lookback)
        for index in range(before_index, start - 1, -1):
            if self._is_active(frame.iloc[index].get(column)):
                return index
        return None

    def _displacement_at_bar(
        self,
        frame: pd.DataFrame,
        index: int,
        direction: str,
    ) -> DisplacementStrength:
        return LiquidityNarrativeEngine._displacement_strength_for_bar(
            frame.iloc[index],
            direction,
        )

    def _has_displacement_between(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        direction: str,
    ) -> tuple[bool, int, str]:
        for index in range(start_bar, end_bar + 1):
            strength = self._displacement_at_bar(frame, index, direction)
            if strength in {DisplacementStrength.MEDIUM, DisplacementStrength.STRONG}:
                return True, index, strength.value
        return False, end_bar, DisplacementStrength.NONE.value

    def _fvg_reclaimed_at_bar(
        self,
        frame: pd.DataFrame,
        index: int,
        direction: str,
    ) -> bool:
        window = self.narrative_engine._window(frame, index)
        fvg_context, fvg_bias = self.narrative_engine._fvg_context_state(frame, index, window)
        expected_bias = EVENT_MAP[direction]["fvg_bias"]
        return fvg_context == FvgContext.RECLAIMED and fvg_bias == expected_bias

    def _detect_institutional_sequences(
        self,
        frame: pd.DataFrame,
        timeframe_label: str,
    ) -> list[SequenceInstance]:
        sequences: list[SequenceInstance] = []
        seen_bos: set[int] = set()

        for direction, mapping in EVENT_MAP.items():
            bos_column = mapping["bos"]
            for bos_bar in range(len(frame)):
                if not self._is_active(frame.iloc[bos_bar].get(bos_column)):
                    continue
                if bos_bar in seen_bos:
                    continue

                sweep_bar = self._find_last_event_bar(
                    frame,
                    mapping["sweep"],
                    bos_bar - 1,
                )
                choch_bar = self._find_last_event_bar(
                    frame,
                    mapping["choch"],
                    bos_bar - 1,
                )
                if sweep_bar is None or choch_bar is None:
                    continue
                if not (sweep_bar < choch_bar < bos_bar):
                    continue

                has_displacement, displacement_bar, strength = self._has_displacement_between(
                    frame,
                    sweep_bar,
                    bos_bar,
                    direction,
                )
                if not has_displacement:
                    continue
                if not self._fvg_reclaimed_at_bar(frame, bos_bar, direction):
                    continue

                sequences.append(
                    SequenceInstance(
                        timeframe=timeframe_label,
                        direction=direction,
                        sweep_bar=sweep_bar,
                        displacement_bar=displacement_bar,
                        choch_bar=choch_bar,
                        bos_bar=bos_bar,
                        sweep_timestamp=str(frame.iloc[sweep_bar]["Date"]),
                        bos_timestamp=str(frame.iloc[bos_bar]["Date"]),
                        displacement_strength=strength,
                    )
                )
                seen_bos.add(bos_bar)

        return self._dedupe_sequences(sequences)

    @staticmethod
    def _dedupe_sequences(sequences: list[SequenceInstance]) -> list[SequenceInstance]:
        ranked = sorted(sequences, key=lambda item: item.bos_bar)
        kept: list[SequenceInstance] = []
        last_bar = -MIN_EVENT_SEPARATION
        for sequence in ranked:
            if sequence.bos_bar - last_bar < MIN_EVENT_SEPARATION:
                continue
            kept.append(sequence)
            last_bar = sequence.bos_bar
        return kept

    def _forward_move_from_bar(
        self,
        frame: pd.DataFrame,
        anchor_bar: int,
        direction: str,
        timeframe_label: str,
        anchor_type: str,
    ) -> ForwardMoveOutcome | None:
        if anchor_bar >= len(frame) - 1:
            return None

        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        closes = frame["Close"].astype(float)
        end = min(len(frame) - 1, anchor_bar + FORWARD_BARS)

        origin = float(closes.iloc[anchor_bar])
        expansion_bar: int | None = None

        if direction == "bullish":
            forward_move = float(highs.iloc[anchor_bar + 1 : end + 1].max()) - origin
            for index in range(anchor_bar + 1, end + 1):
                move = float(highs.iloc[anchor_bar + 1 : index + 1].max()) - origin
                if move >= MOVE_THRESHOLDS[0] and expansion_bar is None:
                    expansion_bar = index
        else:
            forward_move = origin - float(lows.iloc[anchor_bar + 1 : end + 1].min())
            for index in range(anchor_bar + 1, end + 1):
                move = origin - float(lows.iloc[anchor_bar + 1 : index + 1].min())
                if move >= MOVE_THRESHOLDS[0] and expansion_bar is None:
                    expansion_bar = index

        forward_move = round(max(forward_move, 0.0), 2)
        bos_to_expansion = (
            expansion_bar - anchor_bar
            if expansion_bar is not None and anchor_type in {"institutional_sequence", "bos"}
            else None
        )

        return ForwardMoveOutcome(
            timeframe=timeframe_label,
            direction=direction,
            anchor_bar=anchor_bar,
            anchor_timestamp=str(frame.iloc[anchor_bar]["Date"]),
            anchor_type=anchor_type,
            forward_move_points=forward_move,
            expansion_bar=expansion_bar,
            bos_to_expansion_bars=bos_to_expansion,
            bos_to_expansion_minutes=self._bars_to_minutes(bos_to_expansion, timeframe_label),
            moved_over_50=forward_move > 50,
            moved_over_100=forward_move > 100,
            moved_over_150=forward_move > 150,
            directional_win=forward_move > 0,
        )

    def _collect_component_events(
        self,
        frame: pd.DataFrame,
        timeframe_label: str,
    ) -> tuple[list[ForwardMoveOutcome], list[ForwardMoveOutcome], list[ForwardMoveOutcome]]:
        bos_outcomes: list[ForwardMoveOutcome] = []
        choch_outcomes: list[ForwardMoveOutcome] = []
        fvg_outcomes: list[ForwardMoveOutcome] = []

        for index in range(len(frame)):
            for direction, mapping in EVENT_MAP.items():
                if self._is_active(frame.iloc[index].get(mapping["bos"])):
                    outcome = self._forward_move_from_bar(
                        frame,
                        index,
                        direction,
                        timeframe_label,
                        "bos",
                    )
                    if outcome:
                        bos_outcomes.append(outcome)

                if self._is_active(frame.iloc[index].get(mapping["choch"])):
                    outcome = self._forward_move_from_bar(
                        frame,
                        index,
                        direction,
                        timeframe_label,
                        "choch",
                    )
                    if outcome:
                        choch_outcomes.append(outcome)

                if self._fvg_reclaimed_at_bar(frame, index, direction):
                    outcome = self._forward_move_from_bar(
                        frame,
                        index,
                        direction,
                        timeframe_label,
                        "fvg_reclaim",
                    )
                    if outcome:
                        fvg_outcomes.append(outcome)

        return bos_outcomes, choch_outcomes, fvg_outcomes

    @staticmethod
    def _sample_controls(
        outcomes: list[ForwardMoveOutcome],
        sample_size: int = RANDOM_SAMPLE_SIZE,
    ) -> list[ForwardMoveOutcome]:
        if len(outcomes) <= sample_size:
            return outcomes
        rng = random.Random(RANDOM_SEED)
        return rng.sample(outcomes, sample_size)

    def _cohort_metrics(
        self,
        label: str,
        outcomes: list[ForwardMoveOutcome],
        sample_note: str = "",
    ) -> CohortMetrics:
        if not outcomes:
            return CohortMetrics(
                label=label,
                occurrences=0,
                moves_over_50=0,
                moves_over_100=0,
                moves_over_150=0,
                pct_moves_over_50=0.0,
                pct_moves_over_100=0.0,
                pct_moves_over_150=0.0,
                average_move_size=0.0,
                win_rate_pct=0.0,
                win_rate_by_direction={},
                average_bos_to_expansion_minutes=None,
                sample_size_note=sample_note,
            )

        bullish = [item for item in outcomes if item.direction == "bullish"]
        bearish = [item for item in outcomes if item.direction == "bearish"]
        expansion_times = [
            item.bos_to_expansion_minutes
            for item in outcomes
            if item.bos_to_expansion_minutes is not None
        ]

        def direction_win_rate(bucket: list[ForwardMoveOutcome]) -> float:
            if not bucket:
                return 0.0
            wins = sum(1 for item in bucket if item.directional_win)
            return round((wins / len(bucket)) * 100, 2)

        total = len(outcomes)
        return CohortMetrics(
            label=label,
            occurrences=total,
            moves_over_50=sum(1 for item in outcomes if item.moved_over_50),
            moves_over_100=sum(1 for item in outcomes if item.moved_over_100),
            moves_over_150=sum(1 for item in outcomes if item.moved_over_150),
            pct_moves_over_50=round(
                sum(1 for item in outcomes if item.moved_over_50) / total * 100,
                2,
            ),
            pct_moves_over_100=round(
                sum(1 for item in outcomes if item.moved_over_100) / total * 100,
                2,
            ),
            pct_moves_over_150=round(
                sum(1 for item in outcomes if item.moved_over_150) / total * 100,
                2,
            ),
            average_move_size=round(mean(item.forward_move_points for item in outcomes), 2),
            win_rate_pct=round(
                sum(1 for item in outcomes if item.directional_win) / total * 100,
                2,
            ),
            win_rate_by_direction={
                "bullish": direction_win_rate(bullish),
                "bearish": direction_win_rate(bearish),
            },
            average_bos_to_expansion_minutes=round(mean(expansion_times), 1)
            if expansion_times
            else None,
            sample_size_note=sample_note,
        )

    def _outperforms(
        self,
        sequence: CohortMetrics,
        baseline: CohortMetrics,
    ) -> bool:
        if sequence.occurrences == 0 or baseline.occurrences == 0:
            return False
        return (
            sequence.pct_moves_over_50 >= baseline.pct_moves_over_50
            and sequence.pct_moves_over_100 >= baseline.pct_moves_over_100
            and sequence.average_move_size >= baseline.average_move_size
            and sequence.win_rate_pct >= baseline.win_rate_pct
        )

    def _conclusions(
        self,
        sequence: CohortMetrics,
        comparison: dict[str, CohortMetrics],
        outperforms: dict[str, bool],
    ) -> list[str]:
        notes: list[str] = []
        notes.append(
            f"Institutional sequences detected: {sequence.occurrences} "
            f"(avg move {sequence.average_move_size} pts, "
            f">{MOVE_THRESHOLDS[0]} pts: {sequence.pct_moves_over_50}%)."
        )
        for label, improves in outperforms.items():
            baseline = comparison[label]
            notes.append(
                f"vs {label}: sequence avg move {sequence.average_move_size} vs "
                f"{baseline.average_move_size}; "
                f">{MOVE_THRESHOLDS[0]} pts {sequence.pct_moves_over_50}% vs "
                f"{baseline.pct_moves_over_50}%; "
                f"outperforms={improves}."
            )
        if sequence.average_bos_to_expansion_minutes is not None:
            notes.append(
                f"Average BOS-to-expansion time: "
                f"{sequence.average_bos_to_expansion_minutes} minutes."
            )
        complete_wins = all(outperforms.values()) if outperforms else False
        notes.append(
            f"Complete institutional sequence outperforms all components: {complete_wins}."
        )
        return notes

    def run(self, metadata: dict[str, Any]) -> InstitutionalMoveValidationReport:
        """Run institutional move validation research."""
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

        all_sequences: list[SequenceInstance] = []
        sequence_outcomes: list[ForwardMoveOutcome] = []
        all_bos: list[ForwardMoveOutcome] = []
        all_choch: list[ForwardMoveOutcome] = []
        all_fvg: list[ForwardMoveOutcome] = []

        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)

            sequences = self._detect_institutional_sequences(frame, timeframe_label)
            all_sequences.extend(sequences)

            for sequence in sequences:
                outcome = self._forward_move_from_bar(
                    frame,
                    sequence.bos_bar,
                    sequence.direction,
                    timeframe_label,
                    "institutional_sequence",
                )
                if outcome:
                    sequence_outcomes.append(outcome)

            bos, choch, fvg = self._collect_component_events(frame, timeframe_label)
            all_bos.extend(bos)
            all_choch.extend(choch)
            all_fvg.extend(fvg)

        random_bos = self._sample_controls(all_bos)
        random_choch = self._sample_controls(all_choch)
        random_fvg = self._sample_controls(all_fvg)

        sequence_metrics = self._cohort_metrics(
            "Institutional Sequence",
            sequence_outcomes,
        )
        comparison = {
            "all_bos_events": self._cohort_metrics("All BOS Events", all_bos),
            "random_bos_sample": self._cohort_metrics(
                "Random BOS Sample",
                random_bos,
                sample_note=f"Sampled {len(random_bos)} of {len(all_bos)}",
            ),
            "all_choch_events": self._cohort_metrics("All CHOCH Events", all_choch),
            "random_choch_sample": self._cohort_metrics(
                "Random CHOCH Sample",
                random_choch,
                sample_note=f"Sampled {len(random_choch)} of {len(all_choch)}",
            ),
            "all_fvg_reclaim_events": self._cohort_metrics(
                "All FVG Reclaim Events",
                all_fvg,
            ),
            "random_fvg_reclaim_sample": self._cohort_metrics(
                "Random FVG Reclaim Sample",
                random_fvg,
                sample_note=f"Sampled {len(random_fvg)} of {len(all_fvg)}",
            ),
        }

        outperforms = {
            "random_bos_sample": self._outperforms(
                sequence_metrics,
                comparison["random_bos_sample"],
            ),
            "random_choch_sample": self._outperforms(
                sequence_metrics,
                comparison["random_choch_sample"],
            ),
            "random_fvg_reclaim_sample": self._outperforms(
                sequence_metrics,
                comparison["random_fvg_reclaim_sample"],
            ),
        }

        return InstitutionalMoveValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            institutional_sequence=sequence_metrics.as_dict(),
            comparison={key: value.as_dict() for key, value in comparison.items()},
            sequence_outperforms_components=outperforms,
            win_rate_by_direction=sequence_metrics.win_rate_by_direction,
            detected_sequences=[item.as_dict() for item in all_sequences[:50]],
            conclusions=self._conclusions(sequence_metrics, comparison, outperforms),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_move_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> InstitutionalMoveValidationReport:
    """Run institutional move validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalMoveValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalMoveValidationResearch(
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
        "Institutional move validation completed: sequences=%s",
        report.institutional_sequence["occurrences"],
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_move_validation_report()
        seq = report.institutional_sequence
        print("Institutional Move Validation Summary")
        print(f"Sequences detected: {seq['occurrences']}")
        print(f"Avg move after BOS: {seq['average_move_size']} pts")
        print(f">50 pts: {seq['pct_moves_over_50']}% | >100: {seq['pct_moves_over_100']}%")
        print(f">150 pts: {seq['pct_moves_over_150']}%")
        print(f"Win rate: {seq['win_rate_pct']}%")
        print(f"BOS to expansion: {seq['average_bos_to_expansion_minutes']} min")
        for label, improves in report.sequence_outperforms_components.items():
            print(f"Outperforms {label}: {improves}")
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalMoveValidationError as exc:
        logger.error("Institutional move validation error: %s", exc)
        print(f"Institutional move validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional move validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
