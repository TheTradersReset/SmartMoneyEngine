"""
NIFTY50 Trap-to-Momentum Validation research for SmartMoneyEngine.

Measures trap and liquidity events before NIFTY50 momentum moves and validates
their predictive power versus structure events (CHOCH/BOS/FVG/OB).
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
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.filter_research_engine import FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import (
    LEVEL_CLUSTER_POINTS,
    PRE_EXPANSION_LOOKBACK,
    InstitutionalExpansionTriggerDiscoveryResearch,
)
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    TIMEFRAME_MINUTES,
    _CheapMoveCandidate,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "nifty50_trap_to_momentum_validation.json"

DEFAULT_SYMBOL = "NIFTY50"
RESEARCH_WINDOW_DAYS = 120
MOVE_DETECTION_TIMEFRAME = "5M"
MOVE_THRESHOLDS = (100, 200, 300, 500)
ROUND_NUMBER_STEP = 100.0
MAX_MOVE_EXPORT = 400
MAX_COMBO_SIZE = 3
MIN_COMBO_SAMPLES = 5

TRAP_EVENTS = (
    "Gap Reversal",
    "Gap Continuation",
    "Failed Breakout",
    "Failed Breakdown",
    "Liquidity Grab",
    "Stop Hunt",
    "Equal High Sweep",
    "Equal Low Sweep",
    "PDH Sweep",
    "PDL Sweep",
    "PWH Sweep",
    "PWL Sweep",
    "Round Number Sweep",
)
STRUCTURE_EVENTS = ("CHOCH", "BOS", "FVG", "Order Block")
ALL_EVENTS = TRAP_EVENTS + STRUCTURE_EVENTS


class Nifty50TrapToMomentumValidationError(Exception):
    """Raised when trap-to-momentum validation fails."""


@dataclass
class Nifty50TrapToMomentumValidationReport:
    """Full trap-to-momentum validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframe: str
    move_thresholds_points: list[int]
    trap_event_statistics: list[dict[str, Any]]
    structure_event_statistics: list[dict[str, Any]]
    move_pre_event_analysis: list[dict[str, Any]]
    moves_by_threshold: dict[str, list[dict[str, Any]]]
    final_answers: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Nifty50TrapToMomentumValidationResearch:
    """Validate trap/liquidity events as precursors to NIFTY50 momentum moves."""

    def __init__(self) -> None:
        self.discovery = InstitutionalExpansionTriggerDiscoveryResearch(
            symbols=(DEFAULT_SYMBOL,),
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    @staticmethod
    def _round_number_level(price: float) -> float:
        return round(price / ROUND_NUMBER_STEP) * ROUND_NUMBER_STEP

    def _detect_events_at_bar(
        self,
        frame: pd.DataFrame,
        calendar: pd.DataFrame,
        bar: int,
    ) -> tuple[str, ...]:
        row = frame.iloc[bar]
        cal_row = calendar.iloc[bar]
        close = float(row["Close"])
        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        levels = self.discovery._market_levels(frame, bar)
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        events: list[str] = []

        if bar >= 1:
            prev_close = float(frame.iloc[bar - 1]["Close"])
            gap = open_price - prev_close
            bullish_bar = close > open_price
            bearish_bar = close < open_price
            if gap > 0.5:
                events.append("Gap Continuation" if bullish_bar else "Gap Reversal")
            elif gap < -0.5:
                events.append("Gap Continuation" if bearish_bar else "Gap Reversal")

        if resistance is not None and high > resistance and close <= resistance:
            events.append("Failed Breakout")
        if support is not None and low < support and close >= support:
            events.append("Failed Breakdown")

        buy_sweep = self.discovery._is_active(row.get("Buy_Liquidity_Sweep"))
        sell_sweep = self.discovery._is_active(row.get("Sell_Liquidity_Sweep"))
        if buy_sweep or sell_sweep:
            events.extend(["Liquidity Grab", "Stop Hunt"])

        equal_high = self.discovery._to_float(row.get("Equal_High"))
        if equal_high is not None and high > equal_high and close < equal_high:
            events.append("Equal High Sweep")
        equal_low = self.discovery._to_float(row.get("Equal_Low"))
        if equal_low is not None and low < equal_low and close > equal_low:
            events.append("Equal Low Sweep")

        pdh = self.discovery._to_float(cal_row.get("_pdh"))
        if pdh is not None and high > pdh and close < pdh:
            events.append("PDH Sweep")
        pdl = self.discovery._to_float(cal_row.get("_pdl"))
        if pdl is not None and low < pdl and close > pdl:
            events.append("PDL Sweep")
        pwh = self.discovery._to_float(cal_row.get("_pwh"))
        if pwh is not None and high > pwh and close < pwh:
            events.append("PWH Sweep")
        pwl = self.discovery._to_float(cal_row.get("_pwl"))
        if pwl is not None and low < pwl and close > pwl:
            events.append("PWL Sweep")

        round_level = self._round_number_level(close)
        if high > round_level + LEVEL_CLUSTER_POINTS and close < round_level:
            events.append("Round Number Sweep")
        elif low < round_level - LEVEL_CLUSTER_POINTS and close > round_level:
            events.append("Round Number Sweep")

        if self.discovery._is_active(row.get("Bullish_CHOCH")) or self.discovery._is_active(
            row.get("Bearish_CHOCH"),
        ):
            events.append("CHOCH")
        if self.discovery._is_active(row.get("Bullish_BOS")) or self.discovery._is_active(
            row.get("Bearish_BOS"),
        ):
            events.append("BOS")
        if self.discovery._is_active(row.get("Bullish_FVG_Top")) or self.discovery._is_active(
            row.get("Bearish_FVG_Top"),
        ):
            events.append("FVG")
        if self.discovery._is_active(row.get("Bullish_OB_High")) or self.discovery._is_active(
            row.get("Bearish_OB_High"),
        ):
            events.append("Order Block")

        return tuple(dict.fromkeys(events))

    @staticmethod
    def _forward_max_move(
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        bar: int,
        forward_bars: int,
    ) -> float:
        end = min(len(highs) - 1, bar + forward_bars)
        if end <= bar:
            return 0.0
        base = float(closes.iloc[bar])
        segment_high = float(highs.iloc[bar : end + 1].max())
        segment_low = float(lows.iloc[bar : end + 1].min())
        return max(segment_high - base, base - segment_low)

    @staticmethod
    def _drawdown_before_expansion(
        frame: pd.DataFrame,
        event_bar: int,
        move_start_bar: int,
        direction: str,
    ) -> float:
        if move_start_bar <= event_bar:
            return 0.0
        window = frame.iloc[event_bar : move_start_bar + 1]
        entry = float(frame.iloc[event_bar]["Close"])
        if direction == "bullish":
            adverse = entry - float(window["Low"].astype(float).min())
        else:
            adverse = float(window["High"].astype(float).max()) - entry
        return round(max(adverse, 0.0), 2)

    @staticmethod
    def _find_next_move(
        moves: list[_CheapMoveCandidate],
        bar: int,
        forward_bars: int,
    ) -> _CheapMoveCandidate | None:
        candidates = [
            move
            for move in moves
            if bar <= move.start_bar <= bar + forward_bars
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.start_bar)

    @staticmethod
    def _probability(values: list[bool]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values) * 100, 2)

    def _aggregate_event_stats(
        self,
        buckets: dict[str, list[dict[str, Any]]],
        events: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in events:
            samples = buckets.get(event, [])
            if not samples:
                rows.append(
                    {
                        "event": event,
                        "occurrences": 0,
                        "probability_100_plus_pct": 0.0,
                        "probability_200_plus_pct": 0.0,
                        "probability_300_plus_pct": 0.0,
                        "probability_500_plus_pct": 0.0,
                        "average_move_size": 0.0,
                        "average_drawdown_before_expansion": 0.0,
                        "average_time_to_expansion_bars": 0.0,
                    },
                )
                continue

            move_sizes = [float(item["linked_move_size"]) for item in samples if item["linked_move_size"] > 0]
            drawdowns = [float(item["drawdown"]) for item in samples if item["linked_move_size"] > 0]
            times = [float(item["bars_to_expansion"]) for item in samples if item["bars_to_expansion"] >= 0]

            rows.append(
                {
                    "event": event,
                    "occurrences": len(samples),
                    "probability_100_plus_pct": self._probability(
                        [item["forward_magnitude"] >= 100 for item in samples],
                    ),
                    "probability_200_plus_pct": self._probability(
                        [item["forward_magnitude"] >= 200 for item in samples],
                    ),
                    "probability_300_plus_pct": self._probability(
                        [item["forward_magnitude"] >= 300 for item in samples],
                    ),
                    "probability_500_plus_pct": self._probability(
                        [item["forward_magnitude"] >= 500 for item in samples],
                    ),
                    "average_move_size": round(mean(move_sizes), 2) if move_sizes else 0.0,
                    "average_drawdown_before_expansion": round(mean(drawdowns), 2) if drawdowns else 0.0,
                    "average_time_to_expansion_bars": round(mean(times), 2) if times else 0.0,
                },
            )
        return rows

    def _analyze_move_pre_events(
        self,
        move: _CheapMoveCandidate,
        frame: pd.DataFrame,
        calendar: pd.DataFrame,
    ) -> dict[str, Any]:
        pre_start = max(PRE_EXPANSION_LOOKBACK, move.start_bar - PRE_EXPANSION_LOOKBACK)
        first_by_event: dict[str, int] = {}

        for bar in range(pre_start, move.start_bar + 1):
            for event in self._detect_events_at_bar(frame, calendar, bar):
                first_by_event.setdefault(event, bar)

        ordered = sorted(first_by_event.items(), key=lambda item: item[1])
        first_event = ordered[0][0] if ordered else None
        first_bar = ordered[0][1] if ordered else None
        bars_before = move.start_bar - first_bar if first_bar is not None else None

        return {
            "date": str(frame.iloc[move.start_bar].get("Date", "")),
            "direction": move.direction,
            "move_size_points": round(move.magnitude, 2),
            "threshold_tiers": [threshold for threshold in MOVE_THRESHOLDS if move.magnitude >= threshold],
            "events_before_move": [
                {
                    "event": event,
                    "first_bar": bar,
                    "bars_before_move": move.start_bar - bar,
                }
                for event, bar in ordered
            ],
            "first_event": first_event,
            "first_event_bars_before_move": bars_before,
        }

    @staticmethod
    def _predictive_score(row: dict[str, Any]) -> float:
        return (
            float(row.get("probability_500_plus_pct", 0.0)) * 4
            + float(row.get("probability_300_plus_pct", 0.0)) * 3
            + float(row.get("probability_200_plus_pct", 0.0)) * 2
            + float(row.get("probability_100_plus_pct", 0.0))
        )

    def _earliest_warning_combination(
        self,
        move_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        combo_buckets: dict[str, list[int]] = defaultdict(list)
        for record in move_records:
            trap_events = {
                item["event"]
                for item in record.get("events_before_move", [])
                if item["event"] in TRAP_EVENTS
            }
            if len(trap_events) < 2:
                continue
            sorted_events = sorted(trap_events)
            for size in range(2, min(MAX_COMBO_SIZE, len(sorted_events)) + 1):
                for combo in combinations(sorted_events, size):
                    earliest = min(
                        item["bars_before_move"]
                        for item in record.get("events_before_move", [])
                        if item["event"] in combo
                    )
                    combo_buckets[" + ".join(combo)].append(earliest)

        ranked = [
            {
                "combination": combo,
                "sample_size": len(values),
                "average_bars_before_move": round(mean(values), 2),
                "minimum_bars_before_move": min(values),
            }
            for combo, values in combo_buckets.items()
            if len(values) >= MIN_COMBO_SAMPLES
        ]
        ranked.sort(key=lambda row: (row["average_bars_before_move"], -row["sample_size"]))
        return ranked[0] if ranked else {"combination": "None", "sample_size": 0, "average_bars_before_move": 0.0}

    def run(self, metadata: dict[str, Any]) -> Nifty50TrapToMomentumValidationReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=RESEARCH_WINDOW_DAYS)
        )

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        if len(frame) <= PRE_EXPANSION_LOOKBACK + FORWARD_BARS:
            raise Nifty50TrapToMomentumValidationError("Insufficient NIFTY50 pipeline data for validation.")

        liquidity_map = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)
        calendar = liquidity_map._attach_calendar_levels(frame)
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        closes = frame["Close"].astype(float)

        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, MOVE_THRESHOLDS[0]),
        )
        moves = sorted(moves, key=lambda item: -item.magnitude)

        event_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        scan_end = len(frame) - FORWARD_BARS
        for bar in range(PRE_EXPANSION_LOOKBACK, scan_end):
            events = self._detect_events_at_bar(frame, calendar, bar)
            if not events:
                continue
            forward_magnitude = self._forward_max_move(highs, lows, closes, bar, FORWARD_BARS)
            linked_move = self._find_next_move(moves, bar, FORWARD_BARS)
            linked_size = linked_move.magnitude if linked_move else 0.0
            bars_to_expansion = linked_move.start_bar - bar if linked_move else -1
            drawdown = (
                self._drawdown_before_expansion(frame, bar, linked_move.start_bar, linked_move.direction)
                if linked_move
                else 0.0
            )
            sample = {
                "bar": bar,
                "forward_magnitude": forward_magnitude,
                "linked_move_size": linked_size,
                "bars_to_expansion": bars_to_expansion,
                "drawdown": drawdown,
            }
            for event in events:
                event_buckets[event].append(sample)

        trap_stats = self._aggregate_event_stats(event_buckets, TRAP_EVENTS)
        structure_stats = self._aggregate_event_stats(event_buckets, STRUCTURE_EVENTS)

        move_records: list[dict[str, Any]] = []
        moves_by_threshold: dict[str, list[dict[str, Any]]] = {
            str(threshold): [] for threshold in MOVE_THRESHOLDS
        }
        for move in moves[:MAX_MOVE_EXPORT]:
            record = self._analyze_move_pre_events(move, frame, calendar)
            move_records.append(record)
            entry = {
                "date": record["date"],
                "direction": record["direction"],
                "move_size_points": record["move_size_points"],
                "duration_minutes": (move.expansion_bar - move.start_bar)
                * TIMEFRAME_MINUTES[MOVE_DETECTION_TIMEFRAME],
                "first_event": record["first_event"],
                "first_event_bars_before_move": record["first_event_bars_before_move"],
                "events_before_move": record["events_before_move"],
            }
            for threshold in record["threshold_tiers"]:
                moves_by_threshold[str(threshold)].append(entry)

        first_event_counter = Counter(
            record["first_event"] for record in move_records if record.get("first_event")
        )
        all_event_stats = trap_stats + structure_stats
        most_predictive = max(all_event_stats, key=self._predictive_score) if all_event_stats else {}
        avg_bars = [
            record["first_event_bars_before_move"]
            for record in move_records
            if record.get("first_event_bars_before_move") is not None
        ]
        earliest_combo = self._earliest_warning_combination(move_records)

        final_answers = {
            "which_event_appears_first": [
                {"event": event, "occurrences": count}
                for event, count in first_event_counter.most_common(10)
            ],
            "most_predictive_event": most_predictive.get("event"),
            "most_predictive_event_scores": {
                "probability_100_plus_pct": most_predictive.get("probability_100_plus_pct", 0.0),
                "probability_200_plus_pct": most_predictive.get("probability_200_plus_pct", 0.0),
                "probability_300_plus_pct": most_predictive.get("probability_300_plus_pct", 0.0),
                "probability_500_plus_pct": most_predictive.get("probability_500_plus_pct", 0.0),
                "predictive_score": round(self._predictive_score(most_predictive), 2) if most_predictive else 0.0,
            },
            "average_bars_before_move": round(mean(avg_bars), 2) if avg_bars else 0.0,
            "earliest_warning_combination": earliest_combo,
            "structure_vs_trap_summary": {
                "best_trap_event": max(trap_stats, key=self._predictive_score)["event"] if trap_stats else None,
                "best_structure_event": max(structure_stats, key=self._predictive_score)["event"]
                if structure_stats
                else None,
            },
        }

        conclusions = [
            "NIFTY50 Trap-to-Momentum Validation complete (120-day 5M scan).",
            f"First event before moves most often: {first_event_counter.most_common(1)[0][0] if first_event_counter else 'N/A'}.",
            f"Most predictive event: {final_answers['most_predictive_event']}.",
            f"Earliest warning combination: {earliest_combo.get('combination', 'None')}.",
        ]

        return Nifty50TrapToMomentumValidationReport(
            symbol=DEFAULT_SYMBOL,
            research_window_days=RESEARCH_WINDOW_DAYS,
            start_date=metadata.get("start_date", start.isoformat()),
            end_date=metadata.get("end_date", end.isoformat()),
            timeframe=MOVE_DETECTION_TIMEFRAME,
            move_thresholds_points=list(MOVE_THRESHOLDS),
            trap_event_statistics=trap_stats,
            structure_event_statistics=structure_stats,
            move_pre_event_analysis=move_records,
            moves_by_threshold=moves_by_threshold,
            final_answers=final_answers,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_nifty50_trap_to_momentum_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Nifty50TrapToMomentumValidationReport:
    """Run trap-to-momentum validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Nifty50TrapToMomentumValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata = {
        **metadata,
        "research_window_days": RESEARCH_WINDOW_DAYS,
        "start_date": (
            date.fromisoformat(metadata["end_date"]) - timedelta(days=RESEARCH_WINDOW_DAYS)
        ).isoformat(),
    }

    engine = Nifty50TrapToMomentumValidationResearch()
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info("NIFTY50 trap-to-momentum validation exported: %s", destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_nifty50_trap_to_momentum_validation_report()
    except Nifty50TrapToMomentumValidationError as exc:
        logger.error("Trap-to-momentum validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected trap-to-momentum validation error")
        return 1

    answers = report.final_answers
    print("NIFTY50 Trap-to-Momentum Validation Summary")
    print(f"Most predictive event: {answers.get('most_predictive_event')}")
    print(f"Average bars before move: {answers.get('average_bars_before_move')}")
    print(f"Earliest combo: {answers.get('earliest_warning_combination', {}).get('combination')}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
