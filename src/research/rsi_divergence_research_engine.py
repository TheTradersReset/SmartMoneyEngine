"""
RSI divergence research for validated profitable setups.

Detects RSI divergence at existing setup triggers and compares performance
with versus without divergence. Research-only; no new signals or trade logic.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.research.filter_research_engine import (
    RESEARCH_DAYS,
    FilterContextBuilder,
    FilterResearchEngine,
    FilteredTradeRecord,
    _json_safe,
)
from src.research.production_stack_analyzer import PRODUCTION_FILTERS, PRODUCTION_SETUP
from src.signals.setup_classifier import SetupType

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "rsi_divergence_research.json"

PROFITABLE_SETUPS = frozenset(
    {
        SetupType.LIQUIDITY_GRAB_FVG_RECLAIM.value,
        SetupType.CONTINUATION_BOS.value,
    }
)

RSI_PERIOD = 14
SWING_LOOKBACK = 50
PIVOT_STRENGTH = 2
MIN_RANK_TRADES = 5


class RsiDivergenceResearchError(Exception):
    """Raised when RSI divergence research fails."""


class DivergenceType(str, Enum):
    """RSI divergence classifications."""

    BULLISH = "Bullish Divergence"
    BEARISH = "Bearish Divergence"
    HIDDEN_BULLISH = "Hidden Bullish Divergence"
    HIDDEN_BEARISH = "Hidden Bearish Divergence"
    NONE = "None"


@dataclass(frozen=True)
class DivergenceTradeRecord:
    """One setup trade enriched with RSI divergence context."""

    setup_type: str
    direction: str
    direction_label: str
    timeframe: str
    session: str
    trigger_bar: int
    trigger_timestamp: str
    entry_hit: bool
    outcome: str
    realized_pnl_points: float
    realized_rr: float
    divergence_types: tuple[str, ...]
    primary_divergence: str
    has_divergence: bool
    production_stack: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DivergenceMetrics:
    """Performance metrics for one research segment."""

    label: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    max_drawdown: float
    segment_key: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DivergenceComparison:
    """With versus without divergence comparison."""

    with_divergence: dict[str, Any]
    without_divergence: dict[str, Any]
    expectancy_delta: float
    profit_factor_delta: float | None
    win_rate_delta: float
    divergence_improves_expectancy: bool
    divergence_improves_profit_factor: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RsiDivergenceResearchReport:
    """Aggregate RSI divergence research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    setups_analyzed: list[str]
    production_stack: dict[str, Any]
    total_trades: int
    setup_comparisons: dict[str, dict[str, Any]]
    production_stack_comparison: dict[str, Any]
    by_divergence_type: dict[str, dict[str, dict[str, Any]]]
    by_timeframe: dict[str, dict[str, dict[str, Any]]]
    by_direction: dict[str, dict[str, dict[str, Any]]]
    by_session: dict[str, dict[str, dict[str, Any]]]
    best_divergence_combinations: list[dict[str, Any]]
    worst_divergence_combinations: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RsiDivergenceDetector:
    """Detect RSI divergence using recent price and RSI swing pivots."""

    def __init__(
        self,
        lookback: int = SWING_LOOKBACK,
        pivot_strength: int = PIVOT_STRENGTH,
    ) -> None:
        self.lookback = lookback
        self.pivot_strength = pivot_strength

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    def _swing_lows(self, lows: pd.Series, end_index: int) -> list[tuple[int, float]]:
        pivots: list[tuple[int, float]] = []
        start = max(self.pivot_strength, end_index - self.lookback)
        for index in range(start, end_index - self.pivot_strength + 1):
            value = float(lows.iloc[index])
            if all(value <= float(lows.iloc[index - offset]) for offset in range(1, self.pivot_strength + 1)):
                if all(value <= float(lows.iloc[index + offset]) for offset in range(1, self.pivot_strength + 1)):
                    pivots.append((index, value))
        return pivots

    def _swing_highs(self, highs: pd.Series, end_index: int) -> list[tuple[int, float]]:
        pivots: list[tuple[int, float]] = []
        start = max(self.pivot_strength, end_index - self.lookback)
        for index in range(start, end_index - self.pivot_strength + 1):
            value = float(highs.iloc[index])
            if all(value >= float(highs.iloc[index - offset]) for offset in range(1, self.pivot_strength + 1)):
                if all(value >= float(highs.iloc[index + offset]) for offset in range(1, self.pivot_strength + 1)):
                    pivots.append((index, value))
        return pivots

    def detect(self, frame: pd.DataFrame, index: int, rsi: pd.Series) -> list[DivergenceType]:
        """Return divergence types detected at one bar."""
        if index < self.pivot_strength * 2:
            return []

        types: list[DivergenceType] = []
        lows = frame["Low"].astype(float)
        highs = frame["High"].astype(float)

        swing_lows = self._swing_lows(lows, index)
        if len(swing_lows) >= 2:
            (_, price_prev), (_, price_last) = swing_lows[-2], swing_lows[-1]
            rsi_prev = float(rsi.iloc[swing_lows[-2][0]])
            rsi_last = float(rsi.iloc[swing_lows[-1][0]])
            if price_last < price_prev and rsi_last > rsi_prev:
                types.append(DivergenceType.BULLISH)
            elif price_last > price_prev and rsi_last < rsi_prev:
                types.append(DivergenceType.HIDDEN_BULLISH)

        swing_highs = self._swing_highs(highs, index)
        if len(swing_highs) >= 2:
            (_, price_prev), (_, price_last) = swing_highs[-2], swing_highs[-1]
            rsi_prev = float(rsi.iloc[swing_highs[-2][0]])
            rsi_last = float(rsi.iloc[swing_highs[-1][0]])
            if price_last > price_prev and rsi_last < rsi_prev:
                types.append(DivergenceType.BEARISH)
            elif price_last < price_prev and rsi_last > rsi_prev:
                types.append(DivergenceType.HIDDEN_BEARISH)

        return types

    @staticmethod
    def primary_divergence(
        divergence_types: list[DivergenceType],
        direction: str,
    ) -> DivergenceType:
        if not divergence_types:
            return DivergenceType.NONE
        if direction == "bullish":
            for candidate in (DivergenceType.BULLISH, DivergenceType.HIDDEN_BULLISH):
                if candidate in divergence_types:
                    return candidate
        if direction == "bearish":
            for candidate in (DivergenceType.BEARISH, DivergenceType.HIDDEN_BEARISH):
                if candidate in divergence_types:
                    return candidate
        return divergence_types[0]


class RsiDivergenceResearchEngine:
    """Research RSI divergence impact on profitable setup performance."""

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
        self.context_builder = FilterContextBuilder()
        self.detector = RsiDivergenceDetector()

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _max_drawdown(pnls: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return round(max_dd, 2)

    @staticmethod
    def _direction_label(direction: str) -> str:
        return "BUY" if direction == "bullish" else "SELL"

    @staticmethod
    def _is_production_stack(trade: FilteredTradeRecord) -> bool:
        if trade.setup_type != PRODUCTION_SETUP:
            return False
        for dimension, value in PRODUCTION_FILTERS.items():
            if getattr(trade.filters, dimension) != value:
                return False
        return True

    def _metrics(self, trades: list[DivergenceTradeRecord], label: str) -> DivergenceMetrics:
        entries = [trade for trade in trades if trade.entry_hit]
        pnls = [trade.realized_pnl_points for trade in entries]
        rrs = [trade.realized_rr for trade in entries]
        wins = sum(1 for trade in entries if trade.outcome == "Win")
        losses = sum(1 for trade in entries if trade.outcome == "Loss")
        pf = self._profit_factor(pnls)
        return DivergenceMetrics(
            label=label,
            trades=len(entries),
            wins=wins,
            losses=losses,
            win_rate_pct=round((wins / len(entries)) * 100, 2) if entries else 0.0,
            profit_factor=pf,
            expectancy=round(sum(pnls) / len(entries), 2) if entries else 0.0,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            max_drawdown=self._max_drawdown(pnls),
        )

    def _comparison(self, trades: list[DivergenceTradeRecord], label: str) -> DivergenceComparison:
        with_div = [trade for trade in trades if trade.has_divergence]
        without_div = [trade for trade in trades if not trade.has_divergence]
        with_metrics = self._metrics(with_div, f"{label} | With Divergence")
        without_metrics = self._metrics(without_div, f"{label} | Without Divergence")
        pf_delta = None
        if with_metrics.profit_factor is not None and without_metrics.profit_factor is not None:
            pf_delta = round(with_metrics.profit_factor - without_metrics.profit_factor, 2)
        return DivergenceComparison(
            with_divergence=with_metrics.as_dict(),
            without_divergence=without_metrics.as_dict(),
            expectancy_delta=round(with_metrics.expectancy - without_metrics.expectancy, 2),
            profit_factor_delta=pf_delta,
            win_rate_delta=round(with_metrics.win_rate_pct - without_metrics.win_rate_pct, 2),
            divergence_improves_expectancy=with_metrics.expectancy > without_metrics.expectancy,
            divergence_improves_profit_factor=(with_metrics.profit_factor or 0)
            > (without_metrics.profit_factor or 0),
        )

    def _collect_trades(self, metadata: dict[str, Any]) -> list[DivergenceTradeRecord]:
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

        records: list[DivergenceTradeRecord] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            enriched = self.context_builder.enrich(frame)
            rsi = self.detector._compute_rsi(enriched["Close"].astype(float))

            for trade in self.filter_engine._collect_trades(frame, timeframe_label):
                if trade.setup_type not in PROFITABLE_SETUPS or not trade.entry_hit:
                    continue
                divergence_types = self.detector.detect(enriched, trade.trigger_bar, rsi)
                primary = self.detector.primary_divergence(divergence_types, trade.direction)
                records.append(
                    DivergenceTradeRecord(
                        setup_type=trade.setup_type,
                        direction=trade.direction,
                        direction_label=self._direction_label(trade.direction),
                        timeframe=trade.timeframe,
                        session=trade.filters.session,
                        trigger_bar=trade.trigger_bar,
                        trigger_timestamp=trade.trigger_timestamp,
                        entry_hit=trade.entry_hit,
                        outcome=trade.outcome,
                        realized_pnl_points=trade.realized_pnl_points,
                        realized_rr=trade.realized_rr,
                        divergence_types=tuple(item.value for item in divergence_types),
                        primary_divergence=primary.value,
                        has_divergence=bool(divergence_types),
                        production_stack=self._is_production_stack(trade),
                    )
                )
        return records

    def _group_metrics(
        self,
        trades: list[DivergenceTradeRecord],
        dimension: str,
        accessor: Any,
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[DivergenceTradeRecord]] = defaultdict(list)
        for trade in trades:
            grouped[accessor(trade)].append(trade)
        return {
            label: self._metrics(bucket, label).as_dict()
            for label, bucket in sorted(grouped.items())
        }

    def _combination_pool(
        self,
        trades: list[DivergenceTradeRecord],
    ) -> list[DivergenceMetrics]:
        grouped: dict[tuple[str, str, str, str, str], list[DivergenceTradeRecord]] = defaultdict(list)
        for trade in trades:
            if not trade.has_divergence:
                continue
            for divergence_type in trade.divergence_types:
                combo_key = (
                    trade.setup_type,
                    divergence_type,
                    trade.timeframe,
                    trade.direction_label,
                    trade.session,
                )
                grouped[combo_key].append(trade)

        results: list[DivergenceMetrics] = []
        for combo_key, bucket in grouped.items():
            if len(bucket) < MIN_RANK_TRADES:
                continue
            setup_type, divergence_type, timeframe, direction, session = combo_key
            label = (
                f"{setup_type}: {divergence_type} | {timeframe} | {direction} | {session}"
            )
            metrics = self._metrics(bucket, label)
            metrics.segment_key = {
                "setup_type": setup_type,
                "divergence_type": divergence_type,
                "timeframe": timeframe,
                "direction": direction,
                "session": session,
            }
            results.append(metrics)
        return results

    def _conclusions(self, setup_comparisons: dict[str, DivergenceComparison]) -> list[str]:
        notes: list[str] = []
        for setup_type, comparison in setup_comparisons.items():
            with_exp = comparison.with_divergence["expectancy"]
            without_exp = comparison.without_divergence["expectancy"]
            if comparison.divergence_improves_expectancy:
                notes.append(
                    f"{setup_type}: divergence improves expectancy "
                    f"({with_exp} vs {without_exp}, delta {comparison.expectancy_delta})."
                )
            else:
                notes.append(
                    f"{setup_type}: divergence does not improve expectancy "
                    f"({with_exp} vs {without_exp}, delta {comparison.expectancy_delta})."
                )
        return notes

    def run(self, metadata: dict[str, Any]) -> RsiDivergenceResearchReport:
        """Run RSI divergence research."""
        started = time.perf_counter()
        trades = self._collect_trades(metadata)

        by_setup: dict[str, list[DivergenceTradeRecord]] = defaultdict(list)
        for trade in trades:
            by_setup[trade.setup_type].append(trade)

        setup_comparisons = {
            setup_type: self._comparison(by_setup.get(setup_type, []), setup_type).as_dict()
            for setup_type in PROFITABLE_SETUPS
        }

        production_trades = [trade for trade in trades if trade.production_stack]
        production_comparison = self._comparison(
            production_trades,
            PRODUCTION_SETUP,
        ).as_dict()

        by_divergence_type = {
            setup_type: self._group_metrics(
                [trade for trade in trades if trade.setup_type == setup_type],
                "divergence_type",
                lambda item: item.primary_divergence if item.has_divergence else DivergenceType.NONE.value,
            )
            for setup_type in PROFITABLE_SETUPS
        }

        by_timeframe = {
            setup_type: self._group_metrics(
                [trade for trade in trades if trade.setup_type == setup_type and trade.has_divergence],
                "timeframe",
                lambda item: item.timeframe,
            )
            for setup_type in PROFITABLE_SETUPS
        }
        by_direction = {
            setup_type: self._group_metrics(
                [trade for trade in trades if trade.setup_type == setup_type and trade.has_divergence],
                "direction",
                lambda item: item.direction_label,
            )
            for setup_type in PROFITABLE_SETUPS
        }
        by_session = {
            setup_type: self._group_metrics(
                [trade for trade in trades if trade.setup_type == setup_type and trade.has_divergence],
                "session",
                lambda item: item.session,
            )
            for setup_type in PROFITABLE_SETUPS
        }

        combo_pool = self._combination_pool(trades)
        ranked = sorted(combo_pool, key=lambda item: item.expectancy, reverse=True)
        best = [item.as_dict() for item in ranked[:10]]
        worst = [item.as_dict() for item in ranked[-10:][::-1]]

        comparisons_obj = {
            setup_type: DivergenceComparison(**payload)
            for setup_type, payload in setup_comparisons.items()
        }

        return RsiDivergenceResearchReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            setups_analyzed=sorted(PROFITABLE_SETUPS),
            production_stack={
                "setup": PRODUCTION_SETUP,
                "filters": PRODUCTION_FILTERS,
                "total_trades": len(production_trades),
            },
            total_trades=len(trades),
            setup_comparisons=setup_comparisons,
            production_stack_comparison=production_comparison,
            by_divergence_type=by_divergence_type,
            by_timeframe=by_timeframe,
            by_direction=by_direction,
            by_session=by_session,
            best_divergence_combinations=best,
            worst_divergence_combinations=worst,
            conclusions=self._conclusions(comparisons_obj),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_rsi_divergence_research_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> RsiDivergenceResearchReport:
    """Run RSI divergence research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise RsiDivergenceResearchError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = RsiDivergenceResearchEngine(
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
        "RSI divergence research completed: trades=%s production_stack=%s",
        report.total_trades,
        report.production_stack["total_trades"],
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_rsi_divergence_research_report()
        print("RSI Divergence Research Summary")
        print(f"Total Trades: {report.total_trades}")
        print(f"Production Stack Trades: {report.production_stack['total_trades']}")
        for setup_type in report.setups_analyzed:
            comparison = report.setup_comparisons[setup_type]
            print(f"{setup_type}:")
            print(f"  With divergence expectancy: {comparison['with_divergence']['expectancy']}")
            print(f"  Without divergence expectancy: {comparison['without_divergence']['expectancy']}")
            print(f"  Delta: {comparison['expectancy_delta']}")
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except RsiDivergenceResearchError as exc:
        logger.error("RSI divergence research error: %s", exc)
        print(f"RSI divergence research error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected RSI divergence research failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
