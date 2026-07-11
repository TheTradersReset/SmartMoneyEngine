"""
Filter research for profitable institutional setups.

Evaluates contextual filters (EMA, VWAP, RSI, session, ATR, volume) on
existing Liquidity Grab + FVG Reclaim and Continuation BOS setups only.
Does not introduce new signals or setup types.
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.pipeline.market_pipeline import MarketPipelineRunner
from src.research.setup_research_analyzer import TIMEFRAME_MAP, _json_safe
from src.signals.setup_classifier import (
    SetupBacktestSimulator,
    SetupClassifier,
    SetupType,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "outputs" / "pipeline"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"

RESEARCH_DAYS = 365
MIN_COMBO_TRADES = 15
MIN_SINGLE_FILTER_TRADES = 10

PROFITABLE_SETUPS = frozenset(
    {
        SetupType.LIQUIDITY_GRAB_FVG_RECLAIM.value,
        SetupType.CONTINUATION_BOS.value,
    }
)

FILTER_DIMENSIONS: tuple[str, ...] = (
    "ema_alignment",
    "vwap_position",
    "rsi_band",
    "session",
    "atr_percentile",
    "volume_spike",
)

EMA_PERIODS: tuple[int, ...] = (20, 50, 200)
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_LOOKBACK = 100
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLIER = 1.5


class FilterResearchError(Exception):
    """Raised when filter research fails."""


@dataclass(frozen=True)
class FilterState:
    """Contextual filter values at setup trigger."""

    ema_alignment: str
    vwap_position: str
    rsi_band: str
    session: str
    atr_percentile: str
    volume_spike: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FilteredTradeRecord:
    """One profitable-setup trade enriched with filter context."""

    setup_type: str
    direction: str
    timeframe: str
    trigger_bar: int
    trigger_timestamp: str
    entry_hit: bool
    outcome: str
    realized_pnl_points: float
    realized_rr: float
    filters: FilterState

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filters"] = self.filters.as_dict()
        return payload


@dataclass
class FilterMetrics:
    """Performance metrics for a filter bucket or combination."""

    label: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    baseline_expectancy: float | None = None
    expectancy_improvement: float | None = None
    filters: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FilterResearchReport:
    """Aggregate filter research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    setups_analyzed: list[str]
    total_trades: int
    baseline: dict[str, dict[str, Any]]
    single_filter_analysis: dict[str, dict[str, list[dict[str, Any]]]]
    top_20_combinations: list[dict[str, Any]]
    recommendations: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FilterContextBuilder:
    """Compute research-only indicator context on pipeline OHLCV."""

    @staticmethod
    def _ensure_timestamp(frame: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(frame["Date"], errors="coerce")

    @staticmethod
    def _session_label(timestamp: pd.Timestamp) -> str:
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Kolkata")
        else:
            ts = ts.tz_convert("Asia/Kolkata")
        minutes = ts.hour * 60 + ts.minute
        if 9 * 60 + 15 <= minutes < 10 * 60 + 30:
            return "Opening"
        if 10 * 60 + 30 <= minutes < 14 * 60 + 30:
            return "Midday"
        if 14 * 60 + 30 <= minutes <= 15 * 60 + 30:
            return "Closing"
        return "Outside"

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(frame: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        high = frame["High"].astype(float)
        low = frame["Low"].astype(float)
        close = frame["Close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def _compute_vwap(frame: pd.DataFrame, timestamps: pd.Series) -> pd.Series:
        typical = (
            frame["High"].astype(float)
            + frame["Low"].astype(float)
            + frame["Close"].astype(float)
        ) / 3.0
        volume = frame["Volume"].astype(float).fillna(0.0)
        session_day = timestamps.dt.tz_convert("Asia/Kolkata").dt.date
        cumulative_tpv = (typical * volume).groupby(session_day).cumsum()
        cumulative_volume = volume.groupby(session_day).cumsum().replace(0, pd.NA)
        return cumulative_tpv / cumulative_volume

    @staticmethod
    def _rsi_band(value: float) -> str:
        if pd.isna(value):
            return "Unknown"
        if value < 40:
            return "Below 40"
        if value < 50:
            return "40-50"
        if value < 60:
            return "50-60"
        if value < 70:
            return "60-70"
        return "Above 70"

    @staticmethod
    def _atr_percentile_bucket(atr_series: pd.Series, index: int) -> str:
        start = max(0, index - ATR_LOOKBACK + 1)
        window = atr_series.iloc[start : index + 1].dropna()
        if window.empty:
            return "Unknown"
        current = atr_series.iloc[index]
        if pd.isna(current):
            return "Unknown"
        percentile = (window <= current).sum() / len(window) * 100
        if percentile <= 33:
            return "Low (0-33)"
        if percentile <= 66:
            return "Mid (34-66)"
        return "High (67-100)"

    @staticmethod
    def _ema_alignment_label(ema20: float, ema50: float, ema200: float) -> str:
        if any(pd.isna(value) for value in (ema20, ema50, ema200)):
            return "Unknown"
        if ema20 > ema50 > ema200:
            return "EMA20 > EMA50 > EMA200"
        if ema20 < ema50 < ema200:
            return "EMA20 < EMA50 < EMA200"
        return "Mixed"

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach research indicator columns used for filter evaluation."""
        working = frame.reset_index(drop=True).copy()
        close = working["Close"].astype(float)
        working["_timestamp"] = self._ensure_timestamp(working)

        for period in EMA_PERIODS:
            working[f"_ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        working["_rsi"] = self._compute_rsi(close)
        working["_atr"] = self._compute_atr(working)
        working["_vwap"] = self._compute_vwap(working, working["_timestamp"])

        volume = working["Volume"].astype(float)
        volume_mean = volume.rolling(window=VOLUME_LOOKBACK, min_periods=1).mean()
        working["_volume_spike"] = volume >= (volume_mean * VOLUME_SPIKE_MULTIPLIER)
        return working

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

    def filter_state(self, enriched: pd.DataFrame, index: int) -> FilterState:
        """Build filter labels for one trigger bar."""
        row = enriched.iloc[index]
        timestamp = row["_timestamp"]
        session = (
            self._session_label(timestamp)
            if pd.notna(timestamp)
            else "Unknown"
        )
        close = self._safe_float(row["Close"]) or 0.0
        vwap = self._safe_float(row["_vwap"])
        if vwap is None:
            vwap_position = "Unknown"
        else:
            vwap_position = "Above VWAP" if close >= vwap else "Below VWAP"

        ema20 = self._safe_float(row["_ema_20"])
        ema50 = self._safe_float(row["_ema_50"])
        ema200 = self._safe_float(row["_ema_200"])
        rsi = self._safe_float(row["_rsi"])

        return FilterState(
            ema_alignment=self._ema_alignment_label(
                ema20 if ema20 is not None else float("nan"),
                ema50 if ema50 is not None else float("nan"),
                ema200 if ema200 is not None else float("nan"),
            ),
            vwap_position=vwap_position,
            rsi_band=self._rsi_band(rsi if rsi is not None else float("nan")),
            session=session,
            atr_percentile=self._atr_percentile_bucket(enriched["_atr"], index),
            volume_spike="Yes" if bool(row["_volume_spike"]) else "No",
        )


class FilterResearchEngine:
    """
    Research contextual filters on profitable setup classifications.

    Parameters
    ----------
    symbol : str, optional
        Symbol to analyze.
    research_days : int, optional
        Calendar days of history.
    timeframes : tuple[str, ...], optional
        Timeframe labels to include.
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
        self.classifier = SetupClassifier()
        self.simulator = SetupBacktestSimulator()
        self.context_builder = FilterContextBuilder()

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _entry_trades(records: list[FilteredTradeRecord]) -> list[FilteredTradeRecord]:
        return [record for record in records if record.entry_hit]

    def _metrics_for_trades(
        self,
        label: str,
        trades: list[FilteredTradeRecord],
        baseline_expectancy: float | None = None,
        filters: dict[str, str] | None = None,
    ) -> FilterMetrics:
        entries = self._entry_trades(trades)
        wins = sum(1 for trade in entries if trade.outcome == "Win")
        losses = sum(1 for trade in entries if trade.outcome == "Loss")
        pnls = [trade.realized_pnl_points for trade in entries]
        rrs = [trade.realized_rr for trade in entries]
        expectancy = round(sum(pnls) / len(entries), 2) if entries else 0.0
        improvement = (
            round(expectancy - baseline_expectancy, 2)
            if baseline_expectancy is not None
            else None
        )
        return FilterMetrics(
            label=label,
            trades=len(entries),
            wins=wins,
            losses=losses,
            win_rate_pct=round((wins / len(entries)) * 100, 2) if entries else 0.0,
            profit_factor=self._profit_factor(pnls),
            expectancy=expectancy,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            baseline_expectancy=baseline_expectancy,
            expectancy_improvement=improvement,
            filters=filters or {},
        )

    def _pipeline_path(self, timeframe_label: str) -> Path:
        slug = f"{self.symbol}_{timeframe_label.lower()}"
        return DEFAULT_PIPELINE_DIR / f"{slug}_pipeline.csv"

    def _ensure_pipeline(self, timeframe_label: str, start: date, end: date) -> Path:
        path = self._pipeline_path(timeframe_label)
        if path.exists():
            return path
        storage_tf = TIMEFRAME_MAP[timeframe_label]
        runner = MarketPipelineRunner(
            symbol=self.symbol,
            timeframe=storage_tf,
            start_date=start,
            end_date=end,
            output_csv=path,
        )
        report = runner.run()
        if not report.success or not path.exists():
            raise FilterResearchError(
                f"Failed to build pipeline for {self.symbol}/{timeframe_label}: "
                f"{report.failure_message}"
            )
        return path

    def _collect_trades(
        self,
        frame: pd.DataFrame,
        timeframe_label: str,
    ) -> list[FilteredTradeRecord]:
        enriched = self.context_builder.enrich(frame)
        records: list[FilteredTradeRecord] = []

        for setup in self.classifier.classify(enriched):
            if setup.setup_type not in PROFITABLE_SETUPS:
                continue
            backtest = self.simulator.simulate(enriched, setup)
            filters = self.context_builder.filter_state(enriched, setup.trigger_bar)
            records.append(
                FilteredTradeRecord(
                    setup_type=setup.setup_type,
                    direction=setup.direction,
                    timeframe=timeframe_label,
                    trigger_bar=setup.trigger_bar,
                    trigger_timestamp=setup.trigger_timestamp,
                    entry_hit=backtest.entry_hit,
                    outcome=backtest.outcome,
                    realized_pnl_points=backtest.realized_pnl_points,
                    realized_rr=backtest.realized_rr,
                    filters=filters,
                )
            )
        return records

    def _single_filter_analysis(
        self,
        trades: list[FilteredTradeRecord],
        baseline_expectancy: float,
    ) -> dict[str, list[dict[str, Any]]]:
        analysis: dict[str, list[dict[str, Any]]] = {}
        entries = self._entry_trades(trades)

        for dimension in FILTER_DIMENSIONS:
            grouped: dict[str, list[FilteredTradeRecord]] = defaultdict(list)
            for trade in entries:
                bucket = getattr(trade.filters, dimension)
                grouped[bucket].append(trade)

            dimension_metrics: list[dict[str, Any]] = []
            for bucket, bucket_trades in sorted(grouped.items()):
                if len(bucket_trades) < MIN_SINGLE_FILTER_TRADES:
                    continue
                metrics = self._metrics_for_trades(
                    label=bucket,
                    trades=bucket_trades,
                    baseline_expectancy=baseline_expectancy,
                    filters={dimension: bucket},
                )
                dimension_metrics.append(metrics.as_dict())

            dimension_metrics.sort(
                key=lambda item: item.get("expectancy_improvement") or float("-inf"),
                reverse=True,
            )
            analysis[dimension] = dimension_metrics
        return analysis

    def _combination_analysis(
        self,
        trades: list[FilteredTradeRecord],
        baseline_expectancy: float,
        setup_type: str | None = None,
    ) -> list[FilterMetrics]:
        entries = self._entry_trades(trades)
        combinations: list[FilterMetrics] = []

        for size in range(2, len(FILTER_DIMENSIONS) + 1):
            for dimension_subset in itertools.combinations(FILTER_DIMENSIONS, size):
                grouped: dict[tuple[tuple[str, str], ...], list[FilteredTradeRecord]] = (
                    defaultdict(list)
                )
                for trade in entries:
                    key = tuple(
                        (dimension, getattr(trade.filters, dimension))
                        for dimension in dimension_subset
                    )
                    grouped[key].append(trade)

                for key, bucket_trades in grouped.items():
                    if len(bucket_trades) < MIN_COMBO_TRADES:
                        continue
                    filters = dict(key)
                    label = " | ".join(f"{name}={value}" for name, value in key)
                    if setup_type:
                        label = f"{setup_type}: {label}"
                    combinations.append(
                        self._metrics_for_trades(
                            label=label,
                            trades=bucket_trades,
                            baseline_expectancy=baseline_expectancy,
                            filters=filters,
                        )
                    )
        return combinations

    def _recommendations(
        self,
        baseline: dict[str, dict[str, Any]],
        top_combinations: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        combined_baseline = baseline.get("combined", {}).get("expectancy", 0.0)
        notes.append(
            f"Combined baseline expectancy across profitable setups: {combined_baseline} points/trade."
        )

        improving = [
            combo
            for combo in top_combinations
            if (combo.get("expectancy_improvement") or 0) > 0
        ]
        if improving:
            best = improving[0]
            notes.append(
                f"Best filter combination: {best['label']} "
                f"(expectancy {best['expectancy']}, improvement {best['expectancy_improvement']})."
            )
        else:
            notes.append("No multi-filter combination beat baseline with minimum sample size.")

        for setup_type in PROFITABLE_SETUPS:
            setup_baseline = baseline.get(setup_type, {}).get("expectancy")
            setup_top = next(
                (combo for combo in top_combinations if setup_type in combo.get("label", "")),
                None,
            )
            if setup_baseline is not None and setup_top:
                notes.append(
                    f"{setup_type}: baseline {setup_baseline}, "
                    f"best combo expectancy {setup_top['expectancy']}."
                )
        return notes

    def run(
        self,
        end_date: date | None = None,
        pipeline_paths: dict[str, Path] | None = None,
    ) -> FilterResearchReport:
        """Run filter research across configured timeframes."""
        started = time.perf_counter()
        end = end_date if end_date is not None else date.today()
        start = end - timedelta(days=self.research_days)

        all_trades: list[FilteredTradeRecord] = []
        for timeframe_label in self.timeframes:
            if pipeline_paths and timeframe_label in pipeline_paths:
                path = pipeline_paths[timeframe_label]
            else:
                path = self._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            logger.info(
                "Filter research on %s (%s rows).",
                path.name,
                len(frame),
            )
            all_trades.extend(self._collect_trades(frame, timeframe_label))

        by_setup: dict[str, list[FilteredTradeRecord]] = defaultdict(list)
        for trade in all_trades:
            by_setup[trade.setup_type].append(trade)

        baseline: dict[str, dict[str, Any]] = {}
        for setup_type in PROFITABLE_SETUPS:
            setup_trades = by_setup.get(setup_type, [])
            metrics = self._metrics_for_trades(setup_type, setup_trades)
            baseline[setup_type] = metrics.as_dict()

        combined_metrics = self._metrics_for_trades("combined", all_trades)
        baseline["combined"] = combined_metrics.as_dict()
        combined_expectancy = combined_metrics.expectancy

        single_filter_analysis: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for setup_type in PROFITABLE_SETUPS:
            setup_baseline = baseline[setup_type]["expectancy"]
            single_filter_analysis[setup_type] = self._single_filter_analysis(
                by_setup.get(setup_type, []),
                setup_baseline,
            )
        single_filter_analysis["combined"] = self._single_filter_analysis(
            all_trades,
            combined_expectancy,
        )

        all_combos: list[FilterMetrics] = []
        all_combos.extend(
            self._combination_analysis(all_trades, combined_expectancy, setup_type=None)
        )
        for setup_type in PROFITABLE_SETUPS:
            setup_baseline = baseline[setup_type]["expectancy"]
            all_combos.extend(
                self._combination_analysis(
                    by_setup.get(setup_type, []),
                    setup_baseline,
                    setup_type=setup_type,
                )
            )

        ranked = sorted(
            all_combos,
            key=lambda item: (item.expectancy, item.trades),
            reverse=True,
        )
        top_20 = [combo.as_dict() for combo in ranked[:20]]
        for index, combo in enumerate(top_20, start=1):
            combo["rank"] = index

        return FilterResearchReport(
            symbol=self.symbol,
            research_window_days=self.research_days,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            timeframes_analyzed=list(self.timeframes),
            setups_analyzed=sorted(PROFITABLE_SETUPS),
            total_trades=len(self._entry_trades(all_trades)),
            baseline=baseline,
            single_filter_analysis=single_filter_analysis,
            top_20_combinations=top_20,
            recommendations=self._recommendations(baseline, top_20),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_filter_research_report(
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    research_days: int = RESEARCH_DAYS,
    timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    end_date: date | None = None,
) -> FilterResearchReport:
    """Run filter research and export JSON report."""
    engine = FilterResearchEngine(
        symbol=symbol,
        research_days=research_days,
        timeframes=timeframes,
    )
    report = engine.run(end_date=end_date)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Filter research completed: trades=%s top_combos=%s",
        report.total_trades,
        len(report.top_20_combinations),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_filter_research_report()
        print("Filter Research Summary")
        print(f"Symbol: {report.symbol} | Window: {report.research_window_days} days")
        print(f"Setups: {', '.join(report.setups_analyzed)}")
        print(f"Total Trades: {report.total_trades}")
        print("Baseline Expectancy:")
        for setup_type in report.setups_analyzed:
            expectancy = report.baseline[setup_type]["expectancy"]
            print(f"  {setup_type}: {expectancy}")
        print(f"  Combined: {report.baseline['combined']['expectancy']}")
        print("Top 5 Filter Combinations:")
        for combo in report.top_20_combinations[:5]:
            print(
                f"  #{combo['rank']} {combo['label']} | "
                f"trades={combo['trades']} expectancy={combo['expectancy']} "
                f"improvement={combo['expectancy_improvement']}"
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except FilterResearchError as exc:
        logger.error("Filter research error: %s", exc)
        print(f"Filter research error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected filter research failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
