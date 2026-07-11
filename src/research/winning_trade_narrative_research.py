"""
Winning trade narrative research for the validated production stack.

Analyzes narrative context appearing before profitable production-stack trades.
Research-only; no setup, signal, or production logic changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import (
    FvgContext,
    LiquidityEventSnapshot,
    LiquidityNarrativeEngine,
    StructureShiftSnapshot,
)
from src.context.market_intelligence_engine import MarketIntelligenceEngine, RsiState
from src.research.filter_research_engine import (
    RESEARCH_DAYS,
    FilterResearchEngine,
    FilteredTradeRecord,
    _json_safe,
)
from src.research.production_stack_analyzer import PRODUCTION_FILTERS, PRODUCTION_SETUP

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "winning_trade_narratives.json"

MIN_INTELLIGENCE_SCORE = 65
MIN_RANK_SEQUENCE_TRADES = 2
LARGE_WIN_PERCENTILE = 75
TOP_SEQUENCE_COUNT = 20


class WinningTradeNarrativeError(Exception):
    """Raised when winning trade narrative research fails."""


@dataclass(frozen=True)
class NarrativeTradeRecord:
    """Production-stack trade with pre-entry narrative context."""

    setup_type: str
    direction: str
    direction_label: str
    timeframe: str
    session: str
    trigger_timestamp: str
    outcome: str
    realized_pnl_points: float
    realized_rr: float
    liquidity_event: str
    structure_sequence: str
    structure_bucket: str
    fvg_state: str
    market_location: str
    rsi_state: str
    intelligence_score: float
    narrative_strength_score: float
    core_narrative_sequence: str
    full_narrative_sequence: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceMetrics:
    """Performance metrics for one narrative sequence."""

    sequence: str
    trades: int
    wins: int
    losses: int
    frequency: int
    winning_trade_frequency_pct: float
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    average_intelligence_score: float
    average_narrative_strength: float
    total_pnl_points: float
    segment_key: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WinningTradeNarrativeReport:
    """Aggregate winning trade narrative research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    production_stack: dict[str, Any]
    total_stack_trades: int
    total_winning_trades: int
    total_losing_trades: int
    large_win_pnl_threshold: float
    by_liquidity_event: dict[str, dict[str, Any]]
    by_structure_sequence: dict[str, dict[str, Any]]
    by_fvg_state: dict[str, dict[str, Any]]
    by_market_location: dict[str, dict[str, Any]]
    by_rsi_state: dict[str, dict[str, Any]]
    narrative_sequences: list[dict[str, Any]]
    top_20_narrative_sequences: list[dict[str, Any]]
    most_common_before_large_wins: dict[str, Any]
    winning_trade_summary: dict[str, Any]
    sample_winning_trades: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class WinningTradeNarrativeResearch:
    """Research narrative patterns before production-stack winning trades."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
        min_intelligence_score: float = MIN_INTELLIGENCE_SCORE,
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.min_intelligence_score = min_intelligence_score
        self.filter_engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.intelligence_engine = MarketIntelligenceEngine(symbol=symbol)
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _direction_label(direction: str) -> str:
        return "BUY" if direction == "bullish" else "SELL"

    @staticmethod
    def _is_production_stack(trade: FilteredTradeRecord) -> bool:
        if trade.setup_type != PRODUCTION_SETUP:
            return False
        if not trade.entry_hit:
            return False
        for dimension, value in PRODUCTION_FILTERS.items():
            if getattr(trade.filters, dimension) != value:
                return False
        return True

    @staticmethod
    def _liquidity_event_label(liquidity: LiquidityEventSnapshot) -> str:
        if liquidity.buy_side_liquidity_taken and liquidity.sell_side_liquidity_taken:
            return "Both"
        if liquidity.buy_side_liquidity_taken:
            return "Buy Side Sweep"
        if liquidity.sell_side_liquidity_taken:
            return "Sell Side Sweep"
        return "None"

    @staticmethod
    def _structure_labels(
        structure: StructureShiftSnapshot,
    ) -> tuple[str, str]:
        parts: list[str] = []
        if structure.bearish_choch:
            parts.append("Bearish CHOCH")
        elif structure.bullish_choch:
            parts.append("Bullish CHOCH")
        if structure.bearish_bos:
            parts.append("Bearish BOS")
        elif structure.bullish_bos:
            parts.append("Bullish BOS")

        detailed = " + ".join(parts) if parts else "None"
        has_choch = structure.bullish_choch or structure.bearish_choch
        has_bos = structure.bullish_bos or structure.bearish_bos
        if has_choch and has_bos:
            bucket = "CHOCH + BOS"
        elif has_choch:
            bucket = "CHOCH"
        elif has_bos:
            bucket = "BOS"
        else:
            bucket = "None"
        return detailed, bucket

    @staticmethod
    def _fvg_state_label(fvg_context: FvgContext, fvg_bias: str | None) -> str:
        if fvg_context == FvgContext.NONE:
            return "None"
        if fvg_context == FvgContext.FAILED:
            return "Failed"
        bias_label = "Bullish" if fvg_bias == "bullish" else "Bearish"
        if fvg_context == FvgContext.CREATED:
            return f"{bias_label} FVG Created"
        return f"{bias_label} FVG Reclaim"

    @staticmethod
    def _market_location_label(location: str) -> str:
        mapping = {
            "Near Support": "Support",
            "Near Resistance": "Resistance",
            "Mid Range": "Mid Range",
        }
        return mapping.get(location, location)

    @staticmethod
    def _rsi_bucket(rsi_state: RsiState) -> str:
        if rsi_state in {RsiState.WEAK, RsiState.OVERSOLD}:
            return "Weak"
        if rsi_state in {RsiState.STRONG, RsiState.OVERBOUGHT}:
            return "Strong"
        return "Neutral"

    @staticmethod
    def _core_sequence(
        liquidity_event: str,
        structure_sequence: str,
        fvg_state: str,
    ) -> str:
        parts: list[str] = []
        if liquidity_event != "None":
            parts.append(liquidity_event)
        if structure_sequence != "None":
            parts.extend(structure_sequence.split(" + "))
        if fvg_state != "None":
            parts.append(fvg_state)
        return " + ".join(parts) if parts else "No Narrative Pattern"

    @staticmethod
    def _full_sequence(
        liquidity_event: str,
        structure_sequence: str,
        fvg_state: str,
        market_location: str,
        rsi_state: str,
    ) -> str:
        return (
            f"{liquidity_event} | {structure_sequence} | {fvg_state} | "
            f"{market_location} | RSI {rsi_state}"
        )

    def _is_near_support(
        self,
        intel_enriched: pd.DataFrame,
        trigger_bar: int,
    ) -> bool:
        atr_series = intel_enriched["_atr"]
        levels = self.intelligence_engine._market_levels(intel_enriched, trigger_bar)
        atr = (
            float(atr_series.iloc[trigger_bar])
            if pd.notna(atr_series.iloc[trigger_bar])
            else 1.0
        )
        close = levels["close"]
        support = levels["major_support"]
        resistance = levels["major_resistance"]

        def atr_ratio(distance: float | None) -> float | None:
            if distance is None or atr <= 0:
                return None
            return abs(distance) / atr

        support_ratio = atr_ratio(close - support if support is not None else None)
        resistance_ratio = atr_ratio(resistance - close if resistance is not None else None)
        near_support = support_ratio is not None and support_ratio <= 0.5
        near_resistance = resistance_ratio is not None and resistance_ratio <= 0.5
        if near_support and not near_resistance:
            return True
        if near_support and near_resistance:
            return (support_ratio or 0) <= (resistance_ratio or 0)
        return False

    def _sequence_metrics(
        self,
        trades: list[NarrativeTradeRecord],
        sequence: str,
        total_winners: int,
        segment_key: dict[str, str] | None = None,
    ) -> SequenceMetrics:
        pnls = [trade.realized_pnl_points for trade in trades]
        rrs = [trade.realized_rr for trade in trades]
        wins = sum(1 for trade in trades if trade.outcome == "Win")
        losses = sum(1 for trade in trades if trade.outcome == "Loss")
        winner_count = wins
        return SequenceMetrics(
            sequence=sequence,
            trades=len(trades),
            wins=wins,
            losses=losses,
            frequency=winner_count,
            winning_trade_frequency_pct=round(
                (winner_count / total_winners) * 100,
                2,
            )
            if total_winners
            else 0.0,
            win_rate_pct=round((wins / len(trades)) * 100, 2) if trades else 0.0,
            profit_factor=self._profit_factor(pnls),
            expectancy=round(sum(pnls) / len(trades), 2) if trades else 0.0,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            average_intelligence_score=round(
                mean(trade.intelligence_score for trade in trades),
                2,
            )
            if trades
            else 0.0,
            average_narrative_strength=round(
                mean(trade.narrative_strength_score for trade in trades),
                2,
            )
            if trades
            else 0.0,
            total_pnl_points=round(sum(pnls), 2),
            segment_key=segment_key or {},
        )

    def _collect_trades(self, metadata: dict[str, Any]) -> list[NarrativeTradeRecord]:
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

        records: list[NarrativeTradeRecord] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            intel_enriched = self.intelligence_engine.enrich(frame)
            pipeline = frame.reset_index(drop=True)

            for trade in self.filter_engine._collect_trades(frame, timeframe_label):
                if not self._is_production_stack(trade):
                    continue

                intelligence = self.intelligence_engine.evaluate_bar(
                    intel_enriched,
                    trade.trigger_bar,
                )
                if intelligence.intelligence_score < self.min_intelligence_score:
                    continue
                if self._is_near_support(intel_enriched, trade.trigger_bar):
                    continue

                narrative = self.narrative_engine.evaluate_bar(pipeline, trade.trigger_bar)
                window = self.narrative_engine._window(pipeline, trade.trigger_bar)
                liquidity = self.narrative_engine._liquidity_events(
                    pipeline,
                    trade.trigger_bar,
                    window,
                )
                structure = self.narrative_engine._structure_shift(window)
                fvg_context, fvg_bias = self.narrative_engine._fvg_context_state(
                    pipeline,
                    trade.trigger_bar,
                    window,
                )

                liquidity_event = self._liquidity_event_label(liquidity)
                structure_sequence, structure_bucket = self._structure_labels(structure)
                fvg_state = self._fvg_state_label(fvg_context, fvg_bias)
                market_location = self._market_location_label(intelligence.market_location)
                rsi_state = self._rsi_bucket(
                    self.intelligence_engine._rsi_state(
                        float(intel_enriched.iloc[trade.trigger_bar]["RSI"])
                        if pd.notna(intel_enriched.iloc[trade.trigger_bar]["RSI"])
                        else 50.0,
                    ),
                )
                core_sequence = self._core_sequence(
                    liquidity_event,
                    structure_sequence,
                    fvg_state,
                )
                full_sequence = self._full_sequence(
                    liquidity_event,
                    structure_sequence,
                    fvg_state,
                    market_location,
                    rsi_state,
                )

                records.append(
                    NarrativeTradeRecord(
                        setup_type=trade.setup_type,
                        direction=trade.direction,
                        direction_label=self._direction_label(trade.direction),
                        timeframe=trade.timeframe,
                        session=trade.filters.session,
                        trigger_timestamp=trade.trigger_timestamp,
                        outcome=trade.outcome,
                        realized_pnl_points=trade.realized_pnl_points,
                        realized_rr=trade.realized_rr,
                        liquidity_event=liquidity_event,
                        structure_sequence=structure_sequence,
                        structure_bucket=structure_bucket,
                        fvg_state=fvg_state,
                        market_location=market_location,
                        rsi_state=rsi_state,
                        intelligence_score=intelligence.intelligence_score,
                        narrative_strength_score=narrative.narrative_strength_score,
                        core_narrative_sequence=core_sequence,
                        full_narrative_sequence=full_sequence,
                    )
                )
        return records

    def _rank_sequences(
        self,
        trades: list[NarrativeTradeRecord],
        total_winners: int,
        accessor: Any,
    ) -> list[SequenceMetrics]:
        grouped: dict[str, list[NarrativeTradeRecord]] = defaultdict(list)
        for trade in trades:
            grouped[accessor(trade)].append(trade)
        return [
            self._sequence_metrics(bucket, label, total_winners)
            for label, bucket in grouped.items()
        ]

    def _large_win_threshold(self, winners: list[NarrativeTradeRecord]) -> float:
        if not winners:
            return 0.0
        pnls = sorted(trade.realized_pnl_points for trade in winners)
        index = max(0, int(len(pnls) * LARGE_WIN_PERCENTILE / 100) - 1)
        return pnls[min(index, len(pnls) - 1)]

    def _most_common_before_large_wins(
        self,
        winners: list[NarrativeTradeRecord],
        threshold: float,
    ) -> dict[str, Any]:
        large_wins = [
            trade for trade in winners if trade.realized_pnl_points >= threshold
        ]
        if not large_wins:
            return {
                "large_win_threshold_pnl": threshold,
                "large_win_count": 0,
                "most_common_sequence": None,
                "sequence_counts": {},
            }

        counter = Counter(trade.core_narrative_sequence for trade in large_wins)
        most_common = counter.most_common(1)[0]
        return {
            "large_win_threshold_pnl": threshold,
            "large_win_count": len(large_wins),
            "most_common_sequence": most_common[0],
            "most_common_count": most_common[1],
            "most_common_pct": round((most_common[1] / len(large_wins)) * 100, 2),
            "sequence_counts": dict(counter.most_common(10)),
        }

    def _conclusions(
        self,
        top_sequences: list[SequenceMetrics],
        large_win_analysis: dict[str, Any],
        winners: list[NarrativeTradeRecord],
    ) -> list[str]:
        notes: list[str] = []
        if top_sequences:
            best = top_sequences[0]
            notes.append(
                f"Top narrative sequence: {best.sequence} "
                f"(expectancy {best.expectancy}, win rate {best.win_rate_pct}%, "
                f"n={best.trades})."
            )
        if large_win_analysis.get("most_common_sequence"):
            notes.append(
                f"Most common narrative before large wins (>= "
                f"{large_win_analysis['large_win_threshold_pnl']} pts): "
                f"{large_win_analysis['most_common_sequence']} "
                f"({large_win_analysis['most_common_count']} of "
                f"{large_win_analysis['large_win_count']} large wins)."
            )
        if winners:
            avg_mi = round(mean(trade.intelligence_score for trade in winners), 2)
            avg_narrative = round(
                mean(trade.narrative_strength_score for trade in winners),
                2,
            )
            notes.append(
                f"Winning trades average intelligence score {avg_mi} "
                f"and narrative strength {avg_narrative}."
            )
        liquidity_counter = Counter(trade.liquidity_event for trade in winners)
        if liquidity_counter:
            top_liq = liquidity_counter.most_common(1)[0]
            notes.append(
                f"Most common liquidity event among winners: {top_liq[0]} "
                f"({top_liq[1]} trades)."
            )
        return notes

    def run(self, metadata: dict[str, Any]) -> WinningTradeNarrativeReport:
        """Run winning trade narrative research."""
        started = time.perf_counter()
        trades = self._collect_trades(metadata)
        winners = [trade for trade in trades if trade.outcome == "Win"]
        losers = [trade for trade in trades if trade.outcome == "Loss"]
        total_winners = len(winners)

        large_threshold = self._large_win_threshold(winners)
        large_win_analysis = self._most_common_before_large_wins(winners, large_threshold)

        by_liquidity = self._dimension_metrics(
            trades,
            lambda item: item.liquidity_event,
            total_winners,
        )
        by_structure = self._dimension_metrics(trades, lambda item: item.structure_bucket, total_winners)
        by_fvg = self._dimension_metrics(trades, lambda item: item.fvg_state, total_winners)
        by_location = self._dimension_metrics(trades, lambda item: item.market_location, total_winners)
        by_rsi = self._dimension_metrics(trades, lambda item: item.rsi_state, total_winners)

        all_sequences = self._rank_sequences(
            trades,
            total_winners,
            lambda item: item.core_narrative_sequence,
        )
        ranked = sorted(
            all_sequences,
            key=lambda item: (item.expectancy, item.total_pnl_points, item.trades),
            reverse=True,
        )
        eligible = [item for item in ranked if item.trades >= MIN_RANK_SEQUENCE_TRADES]
        top_20 = eligible[:TOP_SEQUENCE_COUNT]

        ranked_winners = sorted(
            winners,
            key=lambda item: item.realized_pnl_points,
            reverse=True,
        )
        sample_winners = [trade.as_dict() for trade in ranked_winners[:15]]

        return WinningTradeNarrativeReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            production_stack={
                "setup": PRODUCTION_SETUP,
                "filters": {
                    **PRODUCTION_FILTERS,
                    "min_intelligence_score": self.min_intelligence_score,
                    "avoid_near_support": True,
                },
            },
            total_stack_trades=len(trades),
            total_winning_trades=total_winners,
            total_losing_trades=len(losers),
            large_win_pnl_threshold=large_threshold,
            by_liquidity_event=by_liquidity,
            by_structure_sequence=by_structure,
            by_fvg_state=by_fvg,
            by_market_location=by_location,
            by_rsi_state=by_rsi,
            narrative_sequences=[item.as_dict() for item in ranked],
            top_20_narrative_sequences=[item.as_dict() for item in top_20],
            most_common_before_large_wins=large_win_analysis,
            winning_trade_summary={
                "average_pnl": round(mean(t.realized_pnl_points for t in winners), 2)
                if winners
                else 0.0,
                "average_intelligence_score": round(
                    mean(t.intelligence_score for t in winners),
                    2,
                )
                if winners
                else 0.0,
                "average_narrative_strength": round(
                    mean(t.narrative_strength_score for t in winners),
                    2,
                )
                if winners
                else 0.0,
                "core_sequence_distribution": dict(
                    Counter(t.core_narrative_sequence for t in winners).most_common(15),
                ),
            },
            sample_winning_trades=sample_winners,
            conclusions=self._conclusions(top_20, large_win_analysis, winners),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )

    def _dimension_metrics(
        self,
        trades: list[NarrativeTradeRecord],
        accessor: Any,
        total_winners: int,
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[NarrativeTradeRecord]] = defaultdict(list)
        for trade in trades:
            grouped[accessor(trade)].append(trade)
        return {
            label: self._sequence_metrics(bucket, label, total_winners).as_dict()
            for label, bucket in sorted(grouped.items())
        }


def generate_winning_trade_narrative_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> WinningTradeNarrativeReport:
    """Run winning trade narrative research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise WinningTradeNarrativeError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = WinningTradeNarrativeResearch(
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
        "Winning trade narrative research completed: winners=%s sequences=%s",
        report.total_winning_trades,
        len(report.narrative_sequences),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_winning_trade_narrative_report()
        print("Winning Trade Narrative Research Summary")
        print(f"Stack Trades: {report.total_stack_trades}")
        print(f"Winning Trades: {report.total_winning_trades}")
        print(f"Large Win Threshold: {report.large_win_pnl_threshold} pts")
        if report.top_20_narrative_sequences:
            best = report.top_20_narrative_sequences[0]
            print(f"Top Sequence: {best['sequence']}")
            print(
                f"  Expectancy={best['expectancy']} WR={best['win_rate_pct']}% "
                f"n={best['trades']}"
            )
        if report.most_common_before_large_wins.get("most_common_sequence"):
            print(
                f"Most Common Before Large Wins: "
                f"{report.most_common_before_large_wins['most_common_sequence']}"
            )
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except WinningTradeNarrativeError as exc:
        logger.error("Winning trade narrative research error: %s", exc)
        print(f"Winning trade narrative research error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected winning trade narrative research failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
