"""
Institutional narrative ranking research for the validated production stack.

Ranks complete pre-trade narrative sequences across winning and losing trades.
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
    LiquidityNarrativeEngine,
)
from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import (
    RESEARCH_DAYS,
    FilterResearchEngine,
    FilteredTradeRecord,
    _json_safe,
)
from src.research.production_stack_analyzer import PRODUCTION_FILTERS, PRODUCTION_SETUP
from src.research.rsi_divergence_research_engine import (
    DivergenceType,
    RsiDivergenceDetector,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_narrative_ranking.json"

MIN_INTELLIGENCE_SCORE = 65
MIN_RANK_TRADES = 2
TOP_N = 20
LARGE_MOVE_THRESHOLDS = (50, 100, 150)
NARRATIVE_ARROW = " -> "


class InstitutionalNarrativeRankingError(Exception):
    """Raised when institutional narrative ranking research fails."""


@dataclass(frozen=True)
class InstitutionalTradeRecord:
    """Production-stack trade with full institutional narrative context."""

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
    choch_state: str
    bos_state: str
    fvg_context: str
    intelligence_score: float
    intelligence_bucket: str
    rsi_band: str
    rsi_divergence: str
    market_location: str
    institutional_narrative: str
    narrative_key: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NarrativeRankMetrics:
    """Performance metrics for one institutional narrative."""

    narrative: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    max_drawdown: float
    average_intelligence_score: float
    total_pnl_points: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalNarrativeRankingReport:
    """Aggregate institutional narrative ranking output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    production_stack: dict[str, Any]
    total_trades: int
    total_winning_trades: int
    total_losing_trades: int
    unique_narratives: int
    ranked_narratives: list[dict[str, Any]]
    top_20_narratives: list[dict[str, Any]]
    worst_20_narratives: list[dict[str, Any]]
    large_move_analysis: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalNarrativeRankingResearch:
    """Rank institutional narrative sequences for production-stack trades."""

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
        self.divergence_detector = RsiDivergenceDetector()

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
        if not trade.entry_hit:
            return False
        for dimension, value in PRODUCTION_FILTERS.items():
            if getattr(trade.filters, dimension) != value:
                return False
        return True

    @staticmethod
    def _liquidity_event(
        buy_sweep: bool,
        sell_sweep: bool,
    ) -> str:
        if buy_sweep and sell_sweep:
            return "Both Side Sweeps"
        if buy_sweep:
            return "Buy Side Sweep"
        if sell_sweep:
            return "Sell Side Sweep"
        return "No Liquidity Sweep"

    @staticmethod
    def _choch_label(bullish: bool, bearish: bool) -> str:
        if bearish:
            return "Bearish CHOCH"
        if bullish:
            return "Bullish CHOCH"
        return "No CHOCH"

    @staticmethod
    def _bos_label(bullish: bool, bearish: bool) -> str:
        if bearish:
            return "Bearish BOS"
        if bullish:
            return "Bullish BOS"
        return "No BOS"

    @staticmethod
    def _fvg_label(fvg_context: FvgContext, fvg_bias: str | None) -> str:
        if fvg_context == FvgContext.NONE:
            return "No FVG Context"
        if fvg_context == FvgContext.FAILED:
            return "FVG Failed"
        bias = "Bullish" if fvg_bias == "bullish" else "Bearish"
        if fvg_context == FvgContext.CREATED:
            return f"{bias} FVG Created"
        return f"{bias} FVG Reclaim"

    @staticmethod
    def _intelligence_bucket(score: float) -> str:
        if score < 70:
            return "MI 65-69"
        if score < 80:
            return "MI 70-79"
        return "MI 80-100"

    @staticmethod
    def _divergence_label(primary: DivergenceType) -> str:
        if primary == DivergenceType.NONE:
            return "No RSI Divergence"
        return f"RSI {primary.value}"

    @staticmethod
    def _market_location_label(location: str) -> str:
        if location == "Near Support":
            return "Near Support"
        if location == "Near Resistance":
            return "Near Resistance"
        return "Mid Range"

    @staticmethod
    def _rsi_band_label(rsi_band: str) -> str:
        return f"RSI {rsi_band}"

    @staticmethod
    def _build_narrative(
        liquidity_event: str,
        choch_state: str,
        bos_state: str,
        fvg_context: str,
        intelligence_bucket: str,
        rsi_band: str,
        rsi_divergence: str,
        market_location: str,
    ) -> tuple[str, str]:
        parts = [
            liquidity_event,
            choch_state,
            bos_state,
            fvg_context,
            intelligence_bucket,
            rsi_band,
            rsi_divergence,
            market_location,
        ]
        narrative = NARRATIVE_ARROW.join(parts)
        key = narrative
        return narrative, key

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

    def _metrics(
        self,
        trades: list[InstitutionalTradeRecord],
        narrative: str,
    ) -> NarrativeRankMetrics:
        pnls = [trade.realized_pnl_points for trade in trades]
        rrs = [trade.realized_rr for trade in trades]
        wins = sum(1 for trade in trades if trade.outcome == "Win")
        losses = sum(1 for trade in trades if trade.outcome == "Loss")
        return NarrativeRankMetrics(
            narrative=narrative,
            trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate_pct=round((wins / len(trades)) * 100, 2) if trades else 0.0,
            profit_factor=self._profit_factor(pnls),
            expectancy=round(sum(pnls) / len(trades), 2) if trades else 0.0,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            max_drawdown=self._max_drawdown(pnls),
            average_intelligence_score=round(
                mean(trade.intelligence_score for trade in trades),
                2,
            )
            if trades
            else 0.0,
            total_pnl_points=round(sum(pnls), 2),
        )

    def _collect_trades(self, metadata: dict[str, Any]) -> list[InstitutionalTradeRecord]:
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

        records: list[InstitutionalTradeRecord] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            intel_enriched = self.intelligence_engine.enrich(frame)
            pipeline = frame.reset_index(drop=True)
            rsi = self.divergence_detector._compute_rsi(pipeline["Close"].astype(float))

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
                divergence_types = self.divergence_detector.detect(
                    pipeline,
                    trade.trigger_bar,
                    rsi,
                )
                primary_div = self.divergence_detector.primary_divergence(
                    divergence_types,
                    trade.direction,
                )

                liquidity_event = self._liquidity_event(
                    liquidity.buy_side_liquidity_taken,
                    liquidity.sell_side_liquidity_taken,
                )
                choch_state = self._choch_label(
                    structure.bullish_choch,
                    structure.bearish_choch,
                )
                bos_state = self._bos_label(
                    structure.bullish_bos,
                    structure.bearish_bos,
                )
                fvg_label = self._fvg_label(fvg_context, fvg_bias)
                intelligence_bucket = self._intelligence_bucket(
                    intelligence.intelligence_score,
                )
                rsi_band = self._rsi_band_label(trade.filters.rsi_band)
                rsi_divergence = self._divergence_label(primary_div)
                market_location = self._market_location_label(intelligence.market_location)
                narrative, narrative_key = self._build_narrative(
                    liquidity_event,
                    choch_state,
                    bos_state,
                    fvg_label,
                    intelligence_bucket,
                    rsi_band,
                    rsi_divergence,
                    market_location,
                )

                records.append(
                    InstitutionalTradeRecord(
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
                        choch_state=choch_state,
                        bos_state=bos_state,
                        fvg_context=fvg_label,
                        intelligence_score=intelligence.intelligence_score,
                        intelligence_bucket=intelligence_bucket,
                        rsi_band=rsi_band,
                        rsi_divergence=rsi_divergence,
                        market_location=market_location,
                        institutional_narrative=narrative,
                        narrative_key=narrative_key,
                    )
                )
        return records

    def _rank_narratives(
        self,
        trades: list[InstitutionalTradeRecord],
    ) -> list[NarrativeRankMetrics]:
        grouped: dict[str, list[InstitutionalTradeRecord]] = defaultdict(list)
        for trade in trades:
            grouped[trade.narrative_key].append(trade)
        return [
            self._metrics(bucket, narrative)
            for narrative, bucket in grouped.items()
        ]

    def _large_move_analysis(
        self,
        trades: list[InstitutionalTradeRecord],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for threshold in LARGE_MOVE_THRESHOLDS:
            large_moves = [
                trade
                for trade in trades
                if trade.outcome == "Win" and trade.realized_pnl_points > threshold
            ]
            if not large_moves:
                results[str(threshold)] = {
                    "threshold_points": threshold,
                    "trade_count": 0,
                    "most_common_narrative": None,
                    "narrative_counts": {},
                }
                continue
            counter = Counter(trade.narrative_key for trade in large_moves)
            most_common = counter.most_common(1)[0]
            results[str(threshold)] = {
                "threshold_points": threshold,
                "trade_count": len(large_moves),
                "most_common_narrative": most_common[0],
                "most_common_count": most_common[1],
                "most_common_pct": round((most_common[1] / len(large_moves)) * 100, 2),
                "narrative_counts": dict(counter.most_common(10)),
            }
        return results

    def _conclusions(
        self,
        top: list[NarrativeRankMetrics],
        worst: list[NarrativeRankMetrics],
        large_moves: dict[str, Any],
        total_trades: int,
        unique_narratives: int,
    ) -> list[str]:
        notes: list[str] = []
        notes.append(
            f"Analyzed {total_trades} production-stack trades across "
            f"{unique_narratives} unique institutional narratives."
        )
        if top:
            best = top[0]
            notes.append(
                f"Top narrative: {best.narrative} "
                f"(expectancy {best.expectancy}, WR {best.win_rate_pct}%, n={best.trades})."
            )
        if worst:
            weak = worst[0]
            notes.append(
                f"Worst narrative: {weak.narrative} "
                f"(expectancy {weak.expectancy}, WR {weak.win_rate_pct}%, n={weak.trades})."
            )
        for threshold in LARGE_MOVE_THRESHOLDS:
            payload = large_moves.get(str(threshold), {})
            narrative = payload.get("most_common_narrative")
            if narrative:
                notes.append(
                    f"Most common narrative before moves > {threshold} pts: "
                    f"{narrative} ({payload['most_common_count']} of "
                    f"{payload['trade_count']} trades)."
                )
        return notes

    def run(self, metadata: dict[str, Any]) -> InstitutionalNarrativeRankingReport:
        """Run institutional narrative ranking research."""
        started = time.perf_counter()
        trades = self._collect_trades(metadata)
        winners = [trade for trade in trades if trade.outcome == "Win"]
        losers = [trade for trade in trades if trade.outcome == "Loss"]

        ranked = self._rank_narratives(trades)
        ranked_sorted = sorted(
            ranked,
            key=lambda item: (item.expectancy, item.total_pnl_points, item.trades),
            reverse=True,
        )
        eligible = [item for item in ranked_sorted if item.trades >= MIN_RANK_TRADES]
        top_20 = eligible[:TOP_N]
        worst_pool = sorted(
            eligible,
            key=lambda item: (item.expectancy, item.total_pnl_points),
        )
        worst_20 = worst_pool[:TOP_N]

        large_move_analysis = self._large_move_analysis(trades)

        return InstitutionalNarrativeRankingReport(
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
            total_trades=len(trades),
            total_winning_trades=len(winners),
            total_losing_trades=len(losers),
            unique_narratives=len(ranked),
            ranked_narratives=[item.as_dict() for item in ranked_sorted],
            top_20_narratives=[item.as_dict() for item in top_20],
            worst_20_narratives=[item.as_dict() for item in worst_20],
            large_move_analysis=large_move_analysis,
            conclusions=self._conclusions(
                top_20,
                worst_20,
                large_move_analysis,
                len(trades),
                len(ranked),
            ),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_narrative_ranking_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> InstitutionalNarrativeRankingReport:
    """Run institutional narrative ranking and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalNarrativeRankingError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalNarrativeRankingResearch(
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
        "Institutional narrative ranking completed: trades=%s narratives=%s",
        report.total_trades,
        report.unique_narratives,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_narrative_ranking_report()
        print("Institutional Narrative Ranking Summary")
        print(f"Trades: {report.total_trades} (W={report.total_winning_trades} L={report.total_losing_trades})")
        print(f"Unique Narratives: {report.unique_narratives}")
        if report.top_20_narratives:
            best = report.top_20_narratives[0]
            print(f"Top Narrative Expectancy: {best['expectancy']} (n={best['trades']})")
        for threshold, payload in report.large_move_analysis.items():
            if payload.get("most_common_narrative"):
                print(
                    f"Moves > {threshold} pts: {payload['most_common_narrative']} "
                    f"({payload['most_common_count']}/{payload['trade_count']})"
                )
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalNarrativeRankingError as exc:
        logger.error("Institutional narrative ranking error: %s", exc)
        print(f"Institutional narrative ranking error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional narrative ranking failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
