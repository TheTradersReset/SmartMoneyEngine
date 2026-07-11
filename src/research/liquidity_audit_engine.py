"""
Liquidity audit diagnostics for SmartMoneyEngine.

Investigates EQH/EQL clusters, pool lifecycle, sweep frequency, and
near-miss conditions without modifying ``LiquidityDetector``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.market_data import MarketData
from src.pipeline.market_pipeline import MarketPipelineRunner
from src.signals.decision_engine import DecisionEngine
from src.smc.bos import BreakOfStructure
from src.smc.choch import ChangeOfCharacter
from src.smc.liquidity import LiquidityDetector, LiquidityPoolRecord, LiquiditySide

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "liquidity_audit_report.json"

LIQUIDITY_COLUMNS = (
    "Swing_High",
    "Swing_Low",
    "High",
    "Low",
    "Close",
    "Equal_High",
    "Equal_Low",
    "Buy_Side_Liquidity",
    "Sell_Side_Liquidity",
    "Buy_Liquidity_Sweep",
    "Sell_Liquidity_Sweep",
    "Liquidity_Strength",
    "Bullish_BOS",
    "Bearish_BOS",
    "Bullish_CHOCH",
    "Bearish_CHOCH",
)


class LiquidityAuditError(Exception):
    """Raised when liquidity audit analysis fails."""


class MissedSweepReason(str, Enum):
    """Why a pool interaction failed strict sweep detection."""

    WICK_ONLY_NO_CLOSE_BACK = "wick_only_no_close_back"
    CLOSE_THROUGH_NO_WICK = "close_through_no_wick"
    TOUCH_NO_PIERCE = "touch_no_pierce"
    NO_INTERACTION = "no_interaction"


@dataclass(frozen=True)
class ClusterSummary:
    """Summary for one EQH/EQL cluster derived from pipeline output."""

    side: str
    level: float
    touch_count: int
    strength: int
    confirmed_position: int
    confirmed_timestamp: str | None
    pool_swept: bool
    sweep_timestamp: str | None
    lifetime_bars: int
    sweep_delay_bars: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MissedSweepCandidate:
    """Pool that interacted with price but failed strict sweep rules."""

    side: str
    level: float
    strength: int
    confirmed_position: int
    interaction_position: int
    interaction_timestamp: str | None
    reason: str
    high: float | None
    low: float | None
    close: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiquidityAuditReport:
    """Aggregate liquidity audit report."""

    symbol: str
    timeframe: str
    source_csv: str
    start_date: str | None
    end_date: str | None
    total_candles: int
    tolerance_ratio: float
    pool_counts: dict[str, int]
    cluster_counts: dict[str, int]
    eqh_eql_touch_counts: dict[str, int]
    lifetime_metrics: dict[str, float | int | None]
    sweep_metrics: dict[str, float | int]
    frequency_comparison: dict[str, int | float]
    restrictive_conditions: list[dict[str, Any]] = field(default_factory=list)
    missed_sweep_summary: dict[str, int] = field(default_factory=dict)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    missed_sweep_samples: list[dict[str, Any]] = field(default_factory=list)
    execution_time_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiquidityAuditEngine:
    """
    Diagnose liquidity pool formation and sweep detection behavior.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Timeframe label such as ``5``.
    tolerance_ratio : float, optional
        Tolerance passed to ``LiquidityDetector`` (for reference only).
    max_missed_samples : int, optional
        Maximum missed-sweep examples to include in the report.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        tolerance_ratio: float = LiquidityDetector.DEFAULT_TOLERANCE_RATIO,
        max_missed_samples: int = 50,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.tolerance_ratio = tolerance_ratio
        self.max_missed_samples = max_missed_samples
        self.detector = LiquidityDetector(tolerance_ratio=tolerance_ratio)

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _timestamp_at(frame: pd.DataFrame, position: int) -> str | None:
        if "Date" not in frame.columns or position < 0 or position >= len(frame):
            return None
        return str(frame["Date"].iloc[position])

    @staticmethod
    def _position_lookup(frame: pd.DataFrame) -> dict[Any, int]:
        return {index: position for position, index in enumerate(frame.index)}

    def _required_columns(self, frame: pd.DataFrame) -> None:
        missing = [column for column in LIQUIDITY_COLUMNS if column not in frame.columns]
        if missing:
            raise LiquidityAuditError(f"Pipeline frame missing liquidity audit columns: {missing}")

    def _run_detector(self, frame: pd.DataFrame) -> tuple[MarketData, tuple[LiquidityPoolRecord, ...]]:
        """Re-run LiquidityDetector on swing/OHLC columns for pool records."""
        input_columns = ["Swing_High", "Swing_Low", "High", "Low", "Close"]
        market = MarketData(frame[input_columns].copy())
        self.detector.detect(market)
        return market, self.detector.liquidity_pools

    @staticmethod
    def _count_active_events(frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
        total = 0
        for column in columns:
            if column in frame.columns:
                total += int(frame[column].notna().sum())
        return total

    def _summarize_eqh_eql_touches(self, frame: pd.DataFrame) -> dict[str, int]:
        eqh_touches = int(frame["Equal_High"].notna().sum()) if "Equal_High" in frame.columns else 0
        eql_touches = int(frame["Equal_Low"].notna().sum()) if "Equal_Low" in frame.columns else 0
        return {
            "equal_high_touches": eqh_touches,
            "equal_low_touches": eql_touches,
            "total_eqh_eql_touches": eqh_touches + eql_touches,
        }

    def _build_cluster_summaries(
        self,
        frame: pd.DataFrame,
        pools: tuple[LiquidityPoolRecord, ...],
    ) -> list[ClusterSummary]:
        """Build per-cluster lifecycle summaries from detector pool records."""
        position_by_index = self._position_lookup(frame)
        summaries: list[ClusterSummary] = []

        for pool in pools:
            confirmed_position = pool.confirmed_position
            sweep_position: int | None = None
            if pool.swept and pool.sweep_index is not None:
                sweep_position = position_by_index.get(pool.sweep_index)

            end_position = sweep_position if sweep_position is not None else len(frame) - 1
            lifetime_bars = max(end_position - confirmed_position, 0)
            sweep_delay = (
                sweep_position - confirmed_position
                if sweep_position is not None and pool.swept
                else None
            )

            touch_count = self._estimate_touch_count(frame, pool)
            summaries.append(
                ClusterSummary(
                    side=pool.side.value,
                    level=round(pool.level, 4),
                    touch_count=touch_count,
                    strength=pool.strength,
                    confirmed_position=confirmed_position,
                    confirmed_timestamp=self._timestamp_at(frame, confirmed_position),
                    pool_swept=pool.swept,
                    sweep_timestamp=(
                        self._timestamp_at(frame, sweep_position)
                        if sweep_position is not None and pool.swept
                        else None
                    ),
                    lifetime_bars=lifetime_bars,
                    sweep_delay_bars=sweep_delay,
                )
            )

        return summaries

    def _estimate_touch_count(self, frame: pd.DataFrame, pool: LiquidityPoolRecord) -> int:
        column = "Equal_High" if pool.side == LiquiditySide.BUY else "Equal_Low"
        if column not in frame.columns:
            return 2
        matches = frame[column].apply(
            lambda value: pd.notna(value) and abs(float(value) - pool.level) < 1e-6
        )
        return max(int(matches.sum()), 2)

    def _classify_pool_interaction(
        self,
        pool: LiquidityPoolRecord,
        high: float,
        low: float,
        close: float,
    ) -> MissedSweepReason | None:
        """
        Classify non-strict pool interactions for missed-sweep diagnosis.

        Returns ``None`` when the candle satisfies strict detector sweep rules.
        """
        if pool.side == LiquiditySide.BUY:
            strict = high > pool.level and close < pool.level
            if strict:
                return None
            if high > pool.level and close >= pool.level:
                return MissedSweepReason.WICK_ONLY_NO_CLOSE_BACK
            if high <= pool.level and close < pool.level:
                return MissedSweepReason.CLOSE_THROUGH_NO_WICK
            if high >= pool.level * (1.0 - self.tolerance_ratio):
                return MissedSweepReason.TOUCH_NO_PIERCE
            return MissedSweepReason.NO_INTERACTION

        strict = low < pool.level and close > pool.level
        if strict:
            return None
        if low < pool.level and close <= pool.level:
            return MissedSweepReason.WICK_ONLY_NO_CLOSE_BACK
        if low >= pool.level and close > pool.level:
            return MissedSweepReason.CLOSE_THROUGH_NO_WICK
        if low <= pool.level * (1.0 + self.tolerance_ratio):
            return MissedSweepReason.TOUCH_NO_PIERCE
        return MissedSweepReason.NO_INTERACTION

    def _find_missed_sweeps(
        self,
        frame: pd.DataFrame,
        pools: tuple[LiquidityPoolRecord, ...],
    ) -> tuple[dict[str, int], list[MissedSweepCandidate]]:
        """Identify pools that pierced liquidity but failed strict sweep rules."""
        position_by_index = self._position_lookup(frame)
        summary = {reason.value: 0 for reason in MissedSweepReason}
        samples: list[MissedSweepCandidate] = []

        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        closes = frame["Close"].astype(float)

        for pool in pools:
            if pool.swept:
                continue

            first_interaction: MissedSweepCandidate | None = None
            for row_index in frame.index[pool.confirmed_position :]:
                position = position_by_index[row_index]
                high = float(highs.loc[row_index])
                low = float(lows.loc[row_index])
                close = float(closes.loc[row_index])
                reason = self._classify_pool_interaction(pool, high, low, close)
                if reason is None or reason == MissedSweepReason.NO_INTERACTION:
                    continue

                summary[reason.value] += 1
                if first_interaction is None and reason in {
                    MissedSweepReason.WICK_ONLY_NO_CLOSE_BACK,
                    MissedSweepReason.CLOSE_THROUGH_NO_WICK,
                }:
                    first_interaction = MissedSweepCandidate(
                        side=pool.side.value,
                        level=round(pool.level, 4),
                        strength=pool.strength,
                        confirmed_position=pool.confirmed_position,
                        interaction_position=position,
                        interaction_timestamp=self._timestamp_at(frame, position),
                        reason=reason.value,
                        high=round(high, 4),
                        low=round(low, 4),
                        close=round(close, 4),
                    )

            if first_interaction is not None and len(samples) < self.max_missed_samples:
                samples.append(first_interaction)

        return summary, samples

    @staticmethod
    def _average(values: list[int]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 2)

    def _build_restrictive_conditions(
        self,
        frame: pd.DataFrame,
        pools: tuple[LiquidityPoolRecord, ...],
        missed_summary: dict[str, int],
        sweep_count: int,
    ) -> list[dict[str, Any]]:
        """Summarize likely restrictive sweep detection rules."""
        unswept = [pool for pool in pools if not pool.swept]
        wick_only = missed_summary.get(MissedSweepReason.WICK_ONLY_NO_CLOSE_BACK.value, 0)
        close_only = missed_summary.get(MissedSweepReason.CLOSE_THROUGH_NO_WICK.value, 0)

        conditions: list[dict[str, Any]] = [
            {
                "condition": "equal_cluster_minimum_touches",
                "description": "EQH/EQL clusters require at least 2 confirmed swing touches.",
                "impact": "Single swing highs/lows never become liquidity pools.",
            },
            {
                "condition": "tolerance_ratio",
                "description": (
                    f"Swing prices must be within {self.tolerance_ratio * 100:.2f}% "
                    "midpoint tolerance to form a cluster."
                ),
                "impact": "Distant equal highs/lows remain separate swings, reducing pool count.",
            },
            {
                "condition": "strict_sweep_wick_and_close_back",
                "description": (
                    "Buy-side sweep requires High > pool AND Close < pool; "
                    "sell-side sweep requires Low < pool AND Close > pool."
                ),
                "impact": (
                    f"Detected {sweep_count} strict sweeps vs "
                    f"{wick_only} wick-only pierces and {close_only} close-through events "
                    "that failed the close-back requirement."
                ),
            },
            {
                "condition": "first_sweep_only",
                "description": "Each pool records only the first qualifying sweep candle.",
                "impact": "Later re-tests of the same pool level are ignored after first sweep.",
            },
            {
                "condition": "unswept_pool_ratio",
                "description": "Pools that never satisfy strict sweep rules expire at dataset end.",
                "impact": (
                    f"{len(unswept)} of {len(pools)} pools ({round(len(unswept) / max(len(pools), 1) * 100, 1)}%) "
                    "expired without a strict sweep."
                ),
            },
        ]

        single_touch_swings = int(frame["Swing_High"].notna().sum() + frame["Swing_Low"].notna().sum())
        pool_count = len(pools)
        if single_touch_swings > pool_count * 2:
            conditions.append(
                {
                    "condition": "swing_to_pool_conversion",
                    "description": "Many confirmed swings never form multi-touch EQH/EQL clusters.",
                    "impact": (
                        f"{single_touch_swings} swing observations produced only {pool_count} liquidity pools."
                    ),
                }
            )

        return conditions

    def analyze(self, frame: pd.DataFrame, source_csv: str = "") -> LiquidityAuditReport:
        """Run liquidity audit diagnostics on a pipeline dataframe."""
        started = time.perf_counter()
        if frame.empty:
            raise LiquidityAuditError("Pipeline frame is empty.")

        self._required_columns(frame)
        market, pools = self._run_detector(frame)
        clusters = self._build_cluster_summaries(frame, pools)
        missed_summary, missed_samples = self._find_missed_sweeps(frame, pools)

        swept_pools = sum(1 for pool in pools if pool.swept)
        expired_pools = len(pools) - swept_pools
        buy_sweeps = int(market.get_column("Buy_Liquidity_Sweep").notna().sum())
        sell_sweeps = int(market.get_column("Sell_Liquidity_Sweep").notna().sum())
        sweep_count = buy_sweeps + sell_sweeps

        lifetimes = [cluster.lifetime_bars for cluster in clusters]
        sweep_delays = [
            cluster.sweep_delay_bars
            for cluster in clusters
            if cluster.sweep_delay_bars is not None
        ]

        bos_count = self._count_active_events(
            frame,
            (BreakOfStructure.BULLISH_BOS_COLUMN, BreakOfStructure.BEARISH_BOS_COLUMN),
        )
        choch_count = self._count_active_events(
            frame,
            (ChangeOfCharacter.BULLISH_CHOCH_COLUMN, ChangeOfCharacter.BEARISH_CHOCH_COLUMN),
        )

        eqh_clusters = sum(1 for cluster in clusters if cluster.side == LiquiditySide.BUY.value)
        eql_clusters = sum(1 for cluster in clusters if cluster.side == LiquiditySide.SELL.value)
        touch_counts = self._summarize_eqh_eql_touches(frame)

        total_candles = len(frame)
        frequency_comparison = {
            "bos_events": bos_count,
            "choch_events": choch_count,
            "liquidity_sweeps": sweep_count,
            "buy_liquidity_sweeps": buy_sweeps,
            "sell_liquidity_sweeps": sell_sweeps,
            "bos_to_sweep_ratio": round(bos_count / max(sweep_count, 1), 2),
            "choch_to_sweep_ratio": round(choch_count / max(sweep_count, 1), 2),
            "sweep_rate_per_1000_candles": round(sweep_count / max(total_candles, 1) * 1000, 2),
            "bos_rate_per_1000_candles": round(bos_count / max(total_candles, 1) * 1000, 2),
            "choch_rate_per_1000_candles": round(choch_count / max(total_candles, 1) * 1000, 2),
        }

        restrictive_conditions = self._build_restrictive_conditions(
            frame,
            pools,
            missed_summary,
            sweep_count,
        )

        return LiquidityAuditReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=source_csv,
            start_date=str(frame["Date"].iloc[0]) if "Date" in frame.columns else None,
            end_date=str(frame["Date"].iloc[-1]) if "Date" in frame.columns else None,
            total_candles=total_candles,
            tolerance_ratio=self.tolerance_ratio,
            pool_counts={
                "total_liquidity_pools": len(pools),
                "buy_side_pools": sum(1 for pool in pools if pool.side == LiquiditySide.BUY),
                "sell_side_pools": sum(1 for pool in pools if pool.side == LiquiditySide.SELL),
                "active_pools_at_end": expired_pools,
                "swept_pools": swept_pools,
                "expired_unswept_pools": expired_pools,
            },
            cluster_counts={
                "total_eqh_eql_clusters": len(clusters),
                "equal_high_clusters": eqh_clusters,
                "equal_low_clusters": eql_clusters,
                **touch_counts,
            },
            eqh_eql_touch_counts=touch_counts,
            lifetime_metrics={
                "average_pool_lifetime_bars": self._average(lifetimes),
                "average_swept_pool_lifetime_bars": self._average(
                    [cluster.lifetime_bars for cluster in clusters if cluster.pool_swept]
                ),
                "average_expired_pool_lifetime_bars": self._average(
                    [cluster.lifetime_bars for cluster in clusters if not cluster.pool_swept]
                ),
                "average_sweep_delay_bars": self._average([value for value in sweep_delays if value is not None]),
                "median_sweep_delay_bars": (
                    int(sorted(sweep_delays)[len(sweep_delays) // 2]) if sweep_delays else None
                ),
                "max_pool_lifetime_bars": max(lifetimes) if lifetimes else None,
            },
            sweep_metrics={
                "total_sweeps": sweep_count,
                "sweep_frequency_pct": round(sweep_count / max(total_candles, 1) * 100, 4),
                "pools_swept_pct": round(swept_pools / max(len(pools), 1) * 100, 2),
                "pools_expired_pct": round(expired_pools / max(len(pools), 1) * 100, 2),
                "missed_wick_only_events": missed_summary.get(
                    MissedSweepReason.WICK_ONLY_NO_CLOSE_BACK.value,
                    0,
                ),
                "missed_close_through_events": missed_summary.get(
                    MissedSweepReason.CLOSE_THROUGH_NO_WICK.value,
                    0,
                ),
            },
            frequency_comparison=frequency_comparison,
            restrictive_conditions=restrictive_conditions,
            missed_sweep_summary=missed_summary,
            clusters=[cluster.as_dict() for cluster in clusters],
            missed_sweep_samples=[sample.as_dict() for sample in missed_samples],
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )

    def run_from_csv(self, pipeline_csv: Path | str) -> LiquidityAuditReport:
        """Load pipeline CSV and run liquidity audit."""
        csv_path = Path(pipeline_csv)
        engine = DecisionEngine(symbol=self.symbol, timeframe=self.timeframe)
        frame = engine.load_pipeline_csv(csv_path)
        return self.analyze(frame, source_csv=str(csv_path))

    def run_from_pipeline(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        output_csv: Path | str | None = None,
        lookback_days: int = 365,
    ) -> LiquidityAuditReport:
        """Run market pipeline then audit liquidity behavior."""
        end = end_date if end_date is not None else date.today()
        start = start_date if start_date is not None else end - timedelta(days=lookback_days)
        destination = Path(output_csv) if output_csv is not None else DEFAULT_PIPELINE_CSV

        runner = MarketPipelineRunner(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start,
            end_date=end,
            output_csv=destination,
        )
        report = runner.run()
        if not report.success or report.output_csv is None:
            raise LiquidityAuditError(report.failure_message or "Market pipeline failed.")
        return self.run_from_csv(report.output_csv)


def generate_liquidity_audit_report(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
    run_pipeline: bool = False,
    lookback_days: int = 365,
) -> LiquidityAuditReport:
    """Run liquidity audit and export JSON report."""
    auditor = LiquidityAuditEngine(symbol=symbol, timeframe=timeframe)

    if run_pipeline:
        audit_report = auditor.run_from_pipeline(lookback_days=lookback_days)
    else:
        csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
        audit_report = auditor.run_from_csv(csv_path)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(audit_report.as_dict(), handle, indent=2)

    logger.info(
        "Liquidity audit completed: pools=%s sweeps=%s",
        audit_report.pool_counts.get("total_liquidity_pools"),
        audit_report.sweep_metrics.get("total_sweeps"),
    )
    return audit_report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_liquidity_audit_report()
        print("Liquidity Audit Summary")
        print(f"Symbol: {report.symbol} | Timeframe: {report.timeframe}")
        print(f"Candles: {report.total_candles}")
        print(f"Pools: {report.pool_counts['total_liquidity_pools']} "
              f"(swept={report.pool_counts['swept_pools']}, "
              f"expired={report.pool_counts['expired_unswept_pools']})")
        print(f"Sweeps: {report.sweep_metrics['total_sweeps']} "
              f"({report.sweep_metrics['sweep_frequency_pct']}% of candles)")
        print("Frequency Comparison:")
        for key, value in report.frequency_comparison.items():
            print(f"  - {key}: {value}")
        print("Top Restrictive Conditions:")
        for condition in report.restrictive_conditions[:3]:
            print(f"  - {condition['condition']}: {condition['impact']}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except LiquidityAuditError as exc:
        logger.error("Liquidity audit error: %s", exc)
        print(f"Liquidity audit error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected liquidity audit failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
