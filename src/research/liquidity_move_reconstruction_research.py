"""
Liquidity move reconstruction research for SmartMoneyEngine.

Detects large directional price moves in pipeline data and reconstructs the
institutional sequence that preceded each move. Research-only; no trades,
signals, or setup changes.
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

from src.context.liquidity_narrative_engine import FvgContext, LiquidityNarrativeEngine
from src.context.market_intelligence_engine import MarketIntelligenceEngine, RsiState
from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "liquidity_move_reconstruction.json"

MOVE_THRESHOLDS = (50, 100, 150)
FORWARD_BARS = 80
PRE_MOVE_LOOKBACK = 20
MIN_MOVE_SEPARATION_BARS = 15
TOP_SEQUENCE_COUNT = 20
NARRATIVE_ARROW = " -> "

TIMEFRAME_MINUTES = {"5M": 5, "15M": 15, "1H": 60}

EVENT_COLUMNS: dict[str, tuple[str, ...]] = {
    "buy_sweep": ("Buy_Liquidity_Sweep",),
    "sell_sweep": ("Sell_Liquidity_Sweep",),
    "bullish_choch": ("Bullish_CHOCH",),
    "bearish_choch": ("Bearish_CHOCH",),
    "bullish_bos": ("Bullish_BOS",),
    "bearish_bos": ("Bearish_BOS",),
}


class LiquidityMoveReconstructionError(Exception):
    """Raised when liquidity move reconstruction fails."""


@dataclass(frozen=True)
class _CheapMoveCandidate:
    """Detected move before narrative enrichment."""

    start_bar: int
    expansion_bar: int
    direction: str
    magnitude: float


@dataclass(frozen=True)
class _BarContext:
    """Precomputed institutional context for one bar."""

    liquidity_event: str
    structure_sequence: str
    fvg_behavior: str
    market_location: str
    rsi_context: str
    intelligence_context: str
    intelligence_score: float
    timing: MoveEventTiming
    pre_move_sequence: str


@dataclass(frozen=True)
class MoveEventTiming:
    """Bar offsets and minutes between structural events."""

    liquidity_sweep_bar: int | None
    choch_bar: int | None
    bos_bar: int | None
    expansion_bar: int
    sweep_to_choch_bars: int | None
    choch_to_bos_bars: int | None
    bos_to_expansion_bars: int | None
    sweep_to_choch_minutes: float | None
    choch_to_bos_minutes: float | None
    bos_to_expansion_minutes: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructedMove:
    """One detected move with pre-move institutional context."""

    timeframe: str
    direction: str
    threshold_points: int
    move_magnitude_points: float
    start_timestamp: str
    expansion_timestamp: str
    start_bar: int
    expansion_bar: int
    liquidity_event: str
    structure_sequence: str
    fvg_behavior: str
    market_location: str
    rsi_context: str
    intelligence_context: str
    intelligence_score: float
    pre_move_sequence: str
    timing: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceRankMetrics:
    """Frequency metrics for a move-starting sequence."""

    sequence: str
    count: int
    frequency_pct: float
    average_move_magnitude: float
    average_sweep_to_choch_minutes: float | None
    average_choch_to_bos_minutes: float | None
    average_bos_to_expansion_minutes: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiquidityMoveReconstructionReport:
    """Aggregate liquidity move reconstruction output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    move_thresholds: list[int]
    total_moves_detected: dict[str, int]
    moves: list[dict[str, Any]]
    average_timing: dict[str, dict[str, Any]]
    ranked_sequences_by_threshold: dict[str, list[dict[str, Any]]]
    top_move_starting_sequences: dict[str, list[dict[str, Any]]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiquidityMoveReconstructionResearch:
    """Detect and reconstruct institutional context before large price moves."""

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
        self.intelligence_engine = MarketIntelligenceEngine(symbol=symbol)
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
        return round(bars * LiquidityMoveReconstructionResearch._minutes_per_bar(timeframe_label), 1)

    @staticmethod
    def _intelligence_bucket(score: float) -> str:
        if score < 70:
            return "MI 65-69"
        if score < 80:
            return "MI 70-79"
        return "MI 80-100"

    @staticmethod
    def _rsi_context(rsi_state: RsiState) -> str:
        if rsi_state in {RsiState.WEAK, RsiState.OVERSOLD}:
            return "RSI Weak"
        if rsi_state in {RsiState.STRONG, RsiState.OVERBOUGHT}:
            return "RSI Strong"
        return "RSI Neutral"

    @staticmethod
    def _liquidity_event(buy_sweep: bool, sell_sweep: bool) -> str:
        if buy_sweep and sell_sweep:
            return "Both Sides"
        if buy_sweep:
            return "Buy Side Sweep"
        if sell_sweep:
            return "Sell Side Sweep"
        return "No Liquidity Sweep"

    @staticmethod
    def _structure_sequence(
        bullish_choch: bool,
        bearish_choch: bool,
        bullish_bos: bool,
        bearish_bos: bool,
    ) -> str:
        parts: list[str] = []
        if bearish_choch:
            parts.append("Bearish CHOCH")
        elif bullish_choch:
            parts.append("Bullish CHOCH")
        if bearish_bos:
            parts.append("Bearish BOS")
        elif bullish_bos:
            parts.append("Bullish BOS")
        return " + ".join(parts) if parts else "No Structure"

    @staticmethod
    def _fvg_behavior(fvg_context: FvgContext, fvg_bias: str | None) -> str:
        if fvg_context == FvgContext.NONE:
            return "No FVG Context"
        if fvg_context == FvgContext.FAILED:
            return "Failed"
        if fvg_context == FvgContext.CREATED:
            return "Created"
        return "Reclaimed"

    def _find_last_event_bar(
        self,
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        before_index: int,
        lookback: int = PRE_MOVE_LOOKBACK,
    ) -> int | None:
        start = max(0, before_index - lookback)
        for index in range(before_index, start - 1, -1):
            row = frame.iloc[index]
            if any(self._is_active(row.get(column)) for column in columns):
                return index
        return None

    def _find_sweep_bar(
        self,
        frame: pd.DataFrame,
        before_index: int,
    ) -> tuple[int | None, str | None]:
        buy_bar = self._find_last_event_bar(frame, EVENT_COLUMNS["buy_sweep"], before_index)
        sell_bar = self._find_last_event_bar(frame, EVENT_COLUMNS["sell_sweep"], before_index)
        if buy_bar is None and sell_bar is None:
            return None, None
        if buy_bar is not None and sell_bar is not None:
            if buy_bar == sell_bar:
                return buy_bar, "both"
            latest = max(buy_bar, sell_bar)
            return latest, "both"
        if buy_bar is not None:
            return buy_bar, "buy"
        return sell_bar, "sell"

    def _find_choch_bar(self, frame: pd.DataFrame, before_index: int) -> int | None:
        bear = self._find_last_event_bar(frame, EVENT_COLUMNS["bearish_choch"], before_index)
        bull = self._find_last_event_bar(frame, EVENT_COLUMNS["bullish_choch"], before_index)
        if bear is None:
            return bull
        if bull is None:
            return bear
        return max(bear, bull)

    def _find_bos_bar(self, frame: pd.DataFrame, before_index: int) -> int | None:
        bear = self._find_last_event_bar(frame, EVENT_COLUMNS["bearish_bos"], before_index)
        bull = self._find_last_event_bar(frame, EVENT_COLUMNS["bullish_bos"], before_index)
        if bear is None:
            return bull
        if bull is None:
            return bear
        return max(bear, bull)

    def _build_timing(
        self,
        sweep_bar: int | None,
        choch_bar: int | None,
        bos_bar: int | None,
        expansion_bar: int,
        timeframe_label: str,
    ) -> MoveEventTiming:
        sweep_to_choch = (
            choch_bar - sweep_bar
            if sweep_bar is not None
            and choch_bar is not None
            and sweep_bar <= choch_bar
            else None
        )
        choch_to_bos = (
            bos_bar - choch_bar
            if choch_bar is not None
            and bos_bar is not None
            and choch_bar <= bos_bar
            else None
        )
        bos_to_expansion = (
            expansion_bar - bos_bar
            if bos_bar is not None and bos_bar <= expansion_bar
            else None
        )

        return MoveEventTiming(
            liquidity_sweep_bar=sweep_bar,
            choch_bar=choch_bar,
            bos_bar=bos_bar,
            expansion_bar=expansion_bar,
            sweep_to_choch_bars=sweep_to_choch,
            choch_to_bos_bars=choch_to_bos,
            bos_to_expansion_bars=bos_to_expansion,
            sweep_to_choch_minutes=self._bars_to_minutes(sweep_to_choch, timeframe_label),
            choch_to_bos_minutes=self._bars_to_minutes(choch_to_bos, timeframe_label),
            bos_to_expansion_minutes=self._bars_to_minutes(bos_to_expansion, timeframe_label),
        )

    def _pre_move_context(
        self,
        frame: pd.DataFrame,
        intel_enriched: pd.DataFrame,
        expansion_bar: int,
        timeframe_label: str,
    ) -> tuple[str, str, str, str, str, str, float, MoveEventTiming]:
        window = self.narrative_engine._window(frame, expansion_bar)
        liquidity = self.narrative_engine._liquidity_events(frame, expansion_bar, window)
        structure = self.narrative_engine._structure_shift(window)
        fvg_context, fvg_bias = self.narrative_engine._fvg_context_state(
            frame,
            expansion_bar,
            window,
        )
        intelligence = self.intelligence_engine.evaluate_bar(intel_enriched, expansion_bar)
        rsi_value = (
            float(intel_enriched.iloc[expansion_bar]["RSI"])
            if pd.notna(intel_enriched.iloc[expansion_bar]["RSI"])
            else 50.0
        )

        liquidity_event = self._liquidity_event(
            liquidity.buy_side_liquidity_taken,
            liquidity.sell_side_liquidity_taken,
        )
        structure_sequence = self._structure_sequence(
            structure.bullish_choch,
            structure.bearish_choch,
            structure.bullish_bos,
            structure.bearish_bos,
        )
        fvg_behavior = self._fvg_behavior(fvg_context, fvg_bias)
        market_location = intelligence.market_location
        rsi_context = self._rsi_context(self.intelligence_engine._rsi_state(rsi_value))
        intelligence_context = self._intelligence_bucket(intelligence.intelligence_score)

        sweep_bar, _ = self._find_sweep_bar(frame, expansion_bar)
        choch_bar = self._find_choch_bar(frame, expansion_bar)
        bos_bar = self._find_bos_bar(frame, expansion_bar)
        timing = self._build_timing(
            sweep_bar,
            choch_bar,
            bos_bar,
            expansion_bar,
            timeframe_label,
        )

        parts = [
            liquidity_event,
            structure_sequence,
            fvg_behavior,
            market_location,
            rsi_context,
            intelligence_context,
        ]
        pre_move_sequence = NARRATIVE_ARROW.join(parts)

        return (
            liquidity_event,
            structure_sequence,
            fvg_behavior,
            market_location,
            rsi_context,
            intelligence_context,
            intelligence.intelligence_score,
            timing,
            pre_move_sequence,
        )

    def _precompute_bar_contexts(
        self,
        frame: pd.DataFrame,
        intel_enriched: pd.DataFrame,
        timeframe_label: str,
    ) -> list[_BarContext]:
        contexts: list[_BarContext] = []
        for index in range(len(frame)):
            (
                liquidity_event,
                structure_sequence,
                fvg_behavior,
                market_location,
                rsi_context,
                intelligence_context,
                intelligence_score,
                timing,
                pre_move_sequence,
            ) = self._pre_move_context(frame, intel_enriched, index, timeframe_label)
            contexts.append(
                _BarContext(
                    liquidity_event=liquidity_event,
                    structure_sequence=structure_sequence,
                    fvg_behavior=fvg_behavior,
                    market_location=market_location,
                    rsi_context=rsi_context,
                    intelligence_context=intelligence_context,
                    intelligence_score=intelligence_score,
                    timing=timing,
                    pre_move_sequence=pre_move_sequence,
                )
            )
        return contexts

    def _detect_moves_cheap(
        self,
        highs: pd.Series,
        lows: pd.Series,
        threshold: int,
    ) -> list[_CheapMoveCandidate]:
        length = len(highs)
        candidates: list[_CheapMoveCandidate] = []

        for expansion_bar in range(FORWARD_BARS, length):
            best_bull = 0.0
            best_bear = 0.0
            best_bull_start: int | None = None
            best_bear_start: int | None = None

            for start_bar in range(expansion_bar - FORWARD_BARS, expansion_bar + 1):
                origin_high = float(highs.iloc[start_bar])
                origin_low = float(lows.iloc[start_bar])
                bull_move = float(highs.iloc[start_bar : expansion_bar + 1].max()) - origin_low
                bear_move = origin_high - float(lows.iloc[start_bar : expansion_bar + 1].min())
                if bull_move >= threshold and bull_move >= best_bull:
                    best_bull = bull_move
                    best_bull_start = start_bar
                if bear_move >= threshold and bear_move >= best_bear:
                    best_bear = bear_move
                    best_bear_start = start_bar

            if best_bull < threshold and best_bear < threshold:
                continue

            if best_bull >= best_bear and best_bull_start is not None:
                candidates.append(
                    _CheapMoveCandidate(
                        start_bar=best_bull_start,
                        expansion_bar=expansion_bar,
                        direction="bullish",
                        magnitude=round(best_bull, 2),
                    )
                )
            elif best_bear_start is not None:
                candidates.append(
                    _CheapMoveCandidate(
                        start_bar=best_bear_start,
                        expansion_bar=expansion_bar,
                        direction="bearish",
                        magnitude=round(best_bear, 2),
                    )
                )
        return candidates

    def _detect_moves_for_threshold(
        self,
        frame: pd.DataFrame,
        bar_contexts: list[_BarContext],
        timeframe_label: str,
        threshold: int,
    ) -> list[ReconstructedMove]:
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        cheap_moves = self._detect_moves_cheap(highs, lows, threshold)
        deduped = self._dedupe_cheap_moves(cheap_moves)

        moves: list[ReconstructedMove] = []
        for candidate in deduped:
            context = bar_contexts[candidate.expansion_bar]
            moves.append(
                ReconstructedMove(
                    timeframe=timeframe_label,
                    direction=candidate.direction,
                    threshold_points=threshold,
                    move_magnitude_points=candidate.magnitude,
                    start_timestamp=str(frame.iloc[candidate.start_bar]["Date"]),
                    expansion_timestamp=str(frame.iloc[candidate.expansion_bar]["Date"]),
                    start_bar=candidate.start_bar,
                    expansion_bar=candidate.expansion_bar,
                    liquidity_event=context.liquidity_event,
                    structure_sequence=context.structure_sequence,
                    fvg_behavior=context.fvg_behavior,
                    market_location=context.market_location,
                    rsi_context=context.rsi_context,
                    intelligence_context=context.intelligence_context,
                    intelligence_score=round(context.intelligence_score, 2),
                    pre_move_sequence=context.pre_move_sequence,
                    timing=context.timing.as_dict(),
                )
            )
        return moves

    @staticmethod
    def _dedupe_cheap_moves(
        candidates: list[_CheapMoveCandidate],
    ) -> list[_CheapMoveCandidate]:
        if not candidates:
            return []
        ranked = sorted(
            candidates,
            key=lambda item: (item.expansion_bar, -item.magnitude),
        )
        kept: list[_CheapMoveCandidate] = []
        last_bar = -MIN_MOVE_SEPARATION_BARS
        for candidate in ranked:
            if candidate.expansion_bar - last_bar < MIN_MOVE_SEPARATION_BARS:
                continue
            kept.append(candidate)
            last_bar = candidate.expansion_bar
        return kept

    @staticmethod
    def _dedupe_moves(moves: list[ReconstructedMove]) -> list[ReconstructedMove]:
        if not moves:
            return []
        ranked = sorted(
            moves,
            key=lambda item: (item.expansion_bar, -item.move_magnitude_points),
        )
        kept: list[ReconstructedMove] = []
        last_bar = -MIN_MOVE_SEPARATION_BARS
        for move in ranked:
            if move.expansion_bar - last_bar < MIN_MOVE_SEPARATION_BARS:
                continue
            kept.append(move)
            last_bar = move.expansion_bar
        return kept

    def _rank_sequences(
        self,
        moves: list[ReconstructedMove],
    ) -> list[SequenceRankMetrics]:
        grouped: dict[str, list[ReconstructedMove]] = defaultdict(list)
        for move in moves:
            grouped[move.pre_move_sequence].append(move)
        total = len(moves)
        metrics: list[SequenceRankMetrics] = []
        for sequence, bucket in grouped.items():
            sweep_times = [
                move.timing["sweep_to_choch_minutes"]
                for move in bucket
                if move.timing.get("sweep_to_choch_minutes") is not None
            ]
            choch_times = [
                move.timing["choch_to_bos_minutes"]
                for move in bucket
                if move.timing.get("choch_to_bos_minutes") is not None
            ]
            bos_times = [
                move.timing["bos_to_expansion_minutes"]
                for move in bucket
                if move.timing.get("bos_to_expansion_minutes") is not None
            ]
            metrics.append(
                SequenceRankMetrics(
                    sequence=sequence,
                    count=len(bucket),
                    frequency_pct=round((len(bucket) / total) * 100, 2) if total else 0.0,
                    average_move_magnitude=round(
                        mean(move.move_magnitude_points for move in bucket),
                        2,
                    ),
                    average_sweep_to_choch_minutes=round(mean(sweep_times), 1)
                    if sweep_times
                    else None,
                    average_choch_to_bos_minutes=round(mean(choch_times), 1)
                    if choch_times
                    else None,
                    average_bos_to_expansion_minutes=round(mean(bos_times), 1)
                    if bos_times
                    else None,
                )
            )
        return sorted(metrics, key=lambda item: (item.count, item.average_move_magnitude), reverse=True)

    def _average_timing(self, moves: list[ReconstructedMove]) -> dict[str, Any]:
        def avg_field(field: str) -> float | None:
            values = [
                move.timing[field]
                for move in moves
                if move.timing.get(field) is not None
            ]
            return round(mean(values), 1) if values else None

        return {
            "liquidity_sweep_to_choch_minutes": avg_field("sweep_to_choch_minutes"),
            "choch_to_bos_minutes": avg_field("choch_to_bos_minutes"),
            "bos_to_expansion_minutes": avg_field("bos_to_expansion_minutes"),
            "sample_sizes": {
                "sweep_to_choch": sum(
                    1 for move in moves if move.timing.get("sweep_to_choch_minutes") is not None
                ),
                "choch_to_bos": sum(
                    1 for move in moves if move.timing.get("choch_to_bos_minutes") is not None
                ),
                "bos_to_expansion": sum(
                    1 for move in moves if move.timing.get("bos_to_expansion_minutes") is not None
                ),
            },
        }

    def _collect_moves(self, metadata: dict[str, Any]) -> list[ReconstructedMove]:
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

        all_moves: list[ReconstructedMove] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            intel_enriched = self.intelligence_engine.enrich(frame)
            highs = frame["High"].astype(float)
            lows = frame["Low"].astype(float)

            cheap_by_threshold: dict[int, list[_CheapMoveCandidate]] = {}
            expansion_bars: set[int] = set()
            for threshold in MOVE_THRESHOLDS:
                cheap_moves = self._dedupe_cheap_moves(
                    self._detect_moves_cheap(highs, lows, threshold),
                )
                cheap_by_threshold[threshold] = cheap_moves
                expansion_bars.update(move.expansion_bar for move in cheap_moves)

            bar_contexts: dict[int, _BarContext] = {}
            for bar_index in sorted(expansion_bars):
                (
                    liquidity_event,
                    structure_sequence,
                    fvg_behavior,
                    market_location,
                    rsi_context,
                    intelligence_context,
                    intelligence_score,
                    timing,
                    pre_move_sequence,
                ) = self._pre_move_context(frame, intel_enriched, bar_index, timeframe_label)
                bar_contexts[bar_index] = _BarContext(
                    liquidity_event=liquidity_event,
                    structure_sequence=structure_sequence,
                    fvg_behavior=fvg_behavior,
                    market_location=market_location,
                    rsi_context=rsi_context,
                    intelligence_context=intelligence_context,
                    intelligence_score=intelligence_score,
                    timing=timing,
                    pre_move_sequence=pre_move_sequence,
                )

            for threshold in MOVE_THRESHOLDS:
                for candidate in cheap_by_threshold[threshold]:
                    context = bar_contexts[candidate.expansion_bar]
                    all_moves.append(
                        ReconstructedMove(
                            timeframe=timeframe_label,
                            direction=candidate.direction,
                            threshold_points=threshold,
                            move_magnitude_points=candidate.magnitude,
                            start_timestamp=str(frame.iloc[candidate.start_bar]["Date"]),
                            expansion_timestamp=str(frame.iloc[candidate.expansion_bar]["Date"]),
                            start_bar=candidate.start_bar,
                            expansion_bar=candidate.expansion_bar,
                            liquidity_event=context.liquidity_event,
                            structure_sequence=context.structure_sequence,
                            fvg_behavior=context.fvg_behavior,
                            market_location=context.market_location,
                            rsi_context=context.rsi_context,
                            intelligence_context=context.intelligence_context,
                            intelligence_score=round(context.intelligence_score, 2),
                            pre_move_sequence=context.pre_move_sequence,
                            timing=context.timing.as_dict(),
                        )
                    )
        return all_moves

    def _conclusions(
        self,
        moves_by_threshold: dict[str, list[ReconstructedMove]],
        ranked_by_threshold: dict[str, list[SequenceRankMetrics]],
        average_timing: dict[str, dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        for threshold in MOVE_THRESHOLDS:
            key = str(threshold)
            moves = moves_by_threshold.get(key, [])
            ranked = ranked_by_threshold.get(key, [])
            notes.append(f"Moves > {threshold} pts detected: {len(moves)}.")
            if ranked:
                top = ranked[0]
                notes.append(
                    f"Most common pre-move sequence (> {threshold} pts): "
                    f"{top.sequence} ({top.count} moves, "
                    f"{top.frequency_pct}%)."
                )
            timing = average_timing.get(key, {})
            if timing.get("liquidity_sweep_to_choch_minutes") is not None:
                notes.append(
                    f"Avg timing > {threshold} pts: sweep->CHOCH "
                    f"{timing['liquidity_sweep_to_choch_minutes']}m, "
                    f"CHOCH->BOS {timing['choch_to_bos_minutes']}m, "
                    f"BOS->expansion {timing['bos_to_expansion_minutes']}m."
                )
        return notes

    def run(self, metadata: dict[str, Any]) -> LiquidityMoveReconstructionReport:
        """Run liquidity move reconstruction research."""
        started = time.perf_counter()
        all_moves = self._collect_moves(metadata)

        moves_by_threshold: dict[str, list[ReconstructedMove]] = {
            str(threshold): [move for move in all_moves if move.threshold_points == threshold]
            for threshold in MOVE_THRESHOLDS
        }
        ranked_by_threshold: dict[str, list[SequenceRankMetrics]] = {
            key: self._rank_sequences(moves) for key, moves in moves_by_threshold.items()
        }
        average_timing = {
            key: self._average_timing(moves) for key, moves in moves_by_threshold.items()
        }
        top_sequences = {
            key: [item.as_dict() for item in ranked[:TOP_SEQUENCE_COUNT]]
            for key, ranked in ranked_by_threshold.items()
        }

        return LiquidityMoveReconstructionReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            move_thresholds=list(MOVE_THRESHOLDS),
            total_moves_detected={
                key: len(moves) for key, moves in moves_by_threshold.items()
            },
            moves=[move.as_dict() for move in all_moves],
            average_timing=average_timing,
            ranked_sequences_by_threshold={
                key: [item.as_dict() for item in ranked]
                for key, ranked in ranked_by_threshold.items()
            },
            top_move_starting_sequences=top_sequences,
            conclusions=self._conclusions(moves_by_threshold, ranked_by_threshold, average_timing),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_liquidity_move_reconstruction_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> LiquidityMoveReconstructionReport:
    """Run move reconstruction research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise LiquidityMoveReconstructionError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = LiquidityMoveReconstructionResearch(
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
        "Liquidity move reconstruction completed: moves=%s",
        sum(report.total_moves_detected.values()),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_liquidity_move_reconstruction_report()
        print("Liquidity Move Reconstruction Summary")
        print(f"Timeframes: {report.timeframes_analyzed}")
        for threshold, count in report.total_moves_detected.items():
            print(f"Moves > {threshold} pts: {count}")
            timing = report.average_timing.get(threshold, {})
            if timing.get("liquidity_sweep_to_choch_minutes") is not None:
                print(
                    f"  Avg sweep->CHOCH: {timing['liquidity_sweep_to_choch_minutes']}m | "
                    f"CHOCH->BOS: {timing['choch_to_bos_minutes']}m | "
                    f"BOS->expansion: {timing['bos_to_expansion_minutes']}m"
                )
            top = report.top_move_starting_sequences.get(threshold, [])
            if top:
                print(f"  Top sequence: {top[0]['sequence']} ({top[0]['count']} moves)")
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except LiquidityMoveReconstructionError as exc:
        logger.error("Liquidity move reconstruction error: %s", exc)
        print(f"Liquidity move reconstruction error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected liquidity move reconstruction failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
