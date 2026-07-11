"""
SmartMoneyEngine batch backtester.

Runs the complete signal pipeline across symbols and timeframes on large
historical datasets, with automatic data download and master reporting.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine, BacktestEngineError
from src.brokers.fyers.client import FyersClient, FyersClientError
from src.brokers.fyers.historical import HistoricalDownloadError, HistoricalDownloader
from src.data.loader.data_loader import DataLoaderError, HistoricalDataLoader
from src.pipeline.market_pipeline import MarketPipelineError, MarketPipelineRunner
from src.signals.decision_engine import DecisionEngineError, evaluate_pipeline
from src.signals.multi_timeframe_engine import (
    MultiTimeframeEngineError,
    generate_multi_timeframe_report,
)
from src.signals.signal_quality_engine import (
    SignalQualityEngineError,
    generate_signal_quality_report,
)
from src.signals.trade_plan_engine_v2 import (
    TradePlanEngineV2Error,
    generate_trade_plans_v2,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "backtesting" / "batch"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "backtesting" / "batch_report.json"
DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "outputs" / "pipeline"

SUPPORTED_SYMBOLS: tuple[str, ...] = ("NIFTY50", "BANKNIFTY", "FINNIFTY")
TIMEFRAME_MAP: dict[str, str] = {
    "5M": "5",
    "15M": "15",
    "1H": "60",
}
FYERS_SYMBOL_MAP: dict[str, str] = {
    "NIFTY50": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX",
}
MTF_BASE_TIMEFRAME = "5"
GRADE_BUCKETS: tuple[str, ...] = ("A+", "A", "B", "C")
CLOSED_OUTCOMES = frozenset({"Win", "Loss", "Breakeven"})
MIN_ANALYSIS_DAYS = 90


class BatchBacktesterError(Exception):
    """Raised when batch backtesting fails."""


@dataclass
class BatchRunMetrics:
    """Backtest metrics for one symbol/timeframe run."""

    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    analysis_days: int
    success: bool
    total_trades: int
    win_rate_pct: float
    average_rr: float
    profit_factor: float | None
    expectancy: float
    maximum_drawdown_points: float
    net_points: float
    grade_statistics: dict[str, dict[str, Any]]
    pipeline_rows: int
    buy_signals: int
    sell_signals: int
    valid_trade_plans: int
    average_quality_score: float
    error_message: str | None = None
    execution_time_seconds: float = 0.0
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable metrics dictionary."""
        payload = asdict(self)
        if payload.get("profit_factor") == float("inf"):
            payload["profit_factor"] = "inf"
        return payload


@dataclass
class BatchReport:
    """Master batch backtesting report."""

    analysis_days: int
    start_date: str
    end_date: str
    symbols: list[str]
    timeframes: list[str]
    total_runs: int
    successful_runs: int
    failed_runs: int
    best_symbol: str | None
    best_timeframe: str | None
    runs: list[dict[str, Any]]
    rankings: dict[str, list[dict[str, Any]]]
    grade_statistics_aggregate: dict[str, dict[str, Any]]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return asdict(self)


class BatchBacktester:
    """
    Orchestrate end-to-end batch backtesting across symbols and timeframes.

    Parameters
    ----------
    symbols : tuple[str, ...] | None, optional
        Symbols to evaluate.
    timeframes : tuple[str, ...] | None, optional
        Timeframe labels such as ``5M``.
    analysis_days : int, optional
        Minimum lookback window in calendar days.
    auto_download : bool, optional
        Download missing FYERS history before running.
    """

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        timeframes: tuple[str, ...] | None = None,
        analysis_days: int = MIN_ANALYSIS_DAYS,
        auto_download: bool = True,
    ) -> None:
        self.symbols = symbols if symbols is not None else SUPPORTED_SYMBOLS
        self.timeframes = timeframes if timeframes is not None else tuple(TIMEFRAME_MAP)
        if analysis_days < MIN_ANALYSIS_DAYS:
            raise BatchBacktesterError(
                f"analysis_days must be at least {MIN_ANALYSIS_DAYS}, got {analysis_days}."
            )
        self.analysis_days = analysis_days
        self.auto_download = auto_download
        self._loader = HistoricalDataLoader()

    @staticmethod
    def _slug(symbol: str, timeframe_label: str) -> str:
        return f"{symbol}_{timeframe_label.lower()}"

    def artifact_paths(self, symbol: str, timeframe_label: str) -> dict[str, Path]:
        """Resolve output artifact paths for one run."""
        slug = self._slug(symbol, timeframe_label)
        run_dir = DEFAULT_OUTPUT_DIR / slug
        pipeline_dir = DEFAULT_PIPELINE_DIR
        return {
            "run_dir": run_dir,
            "pipeline_csv": pipeline_dir / f"{slug}_pipeline.csv",
            "pipeline_report": pipeline_dir / f"{slug}_pipeline_report.json",
            "decision_report": run_dir / "decision_report.json",
            "mtf_report": run_dir / "multi_timeframe_report.json",
            "trade_plan_v2": run_dir / "trade_plan_report_v2.json",
            "signal_quality": run_dir / "signal_quality_report.json",
            "backtest_report": run_dir / "backtest_report.json",
            "trade_results": run_dir / "trade_results.csv",
        }

    def _date_range(self, end_date: date | None = None) -> tuple[date, date]:
        end = end_date if end_date is not None else date.today()
        start = end - timedelta(days=self.analysis_days)
        return start, end

    def _storage_timeframe(self, timeframe_label: str) -> str:
        if timeframe_label not in TIMEFRAME_MAP:
            raise BatchBacktesterError(f"Unsupported timeframe label: {timeframe_label}")
        return TIMEFRAME_MAP[timeframe_label]

    def _data_available(self, symbol: str, timeframe: str, start: date, end: date) -> bool:
        try:
            frame = self._loader.load(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start,
                end_date=end,
            )
            return not frame.empty
        except DataLoaderError:
            return False

    def _download_history(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> None:
        fyers_symbol = FYERS_SYMBOL_MAP.get(symbol)
        if fyers_symbol is None:
            raise BatchBacktesterError(f"No FYERS mapping for symbol: {symbol}")

        try:
            client = FyersClient.from_token_file()
            downloader = HistoricalDownloader(client=client)
            downloader.download(
                symbol=fyers_symbol,
                resolution=timeframe,
                from_date=start,
                to_date=end,
                save=True,
            )
        except (FyersClientError, HistoricalDownloadError) as exc:
            raise BatchBacktesterError(
                f"Failed to download {symbol}/{timeframe} history: {exc}"
            ) from exc

    def ensure_data(
        self,
        symbol: str,
        timeframe_label: str,
        start: date,
        end: date,
    ) -> None:
        """Ensure historical data exists, optionally downloading it."""
        storage_tf = self._storage_timeframe(timeframe_label)
        required = [storage_tf]
        if storage_tf != MTF_BASE_TIMEFRAME:
            required.append(MTF_BASE_TIMEFRAME)

        for tf in required:
            if self._data_available(symbol, tf, start, end):
                continue
            if not self.auto_download:
                raise BatchBacktesterError(
                    f"Missing historical data for {symbol}/{tf} "
                    f"between {start} and {end}."
                )
            logger.info("Downloading missing history for %s/%s", symbol, tf)
            self._download_history(symbol, tf, start, end)

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
        return round(max_drawdown, 2)

    def _compute_grade_statistics(
        self,
        quality_payload: dict[str, Any],
        backtest_payload: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        quality_by_index = {
            int(item["signal_index"]): item for item in quality_payload.get("signals", [])
        }
        results_by_index = {
            int(item["signal_index"]): item
            for item in backtest_payload.get("trade_results", [])
        }

        stats: dict[str, dict[str, Any]] = {
            grade: {
                "signals": 0,
                "trades": 0,
                "wins": 0,
                "win_rate_pct": 0.0,
                "net_points": 0.0,
                "expectancy": 0.0,
            }
            for grade in GRADE_BUCKETS
        }

        for signal_index, quality in quality_by_index.items():
            grade = str(quality.get("grade", "C"))
            if grade not in stats:
                continue
            stats[grade]["signals"] += 1

            result = results_by_index.get(signal_index)
            if result is None or not result.get("entry_hit"):
                continue
            if str(result.get("outcome")) not in CLOSED_OUTCOMES:
                continue

            pnl = float(result.get("realized_pnl_points", 0.0))
            stats[grade]["trades"] += 1
            stats[grade]["net_points"] = round(stats[grade]["net_points"] + pnl, 2)
            if str(result.get("outcome")) == "Win":
                stats[grade]["wins"] += 1

        for grade, bucket in stats.items():
            trades = bucket["trades"]
            if trades:
                bucket["win_rate_pct"] = round(bucket["wins"] / trades * 100, 2)
                bucket["expectancy"] = round(bucket["net_points"] / trades, 2)
            bucket["net_points"] = round(bucket["net_points"], 2)
        return stats

    def _compute_metrics_from_backtest(
        self,
        backtest_payload: dict[str, Any],
    ) -> dict[str, Any]:
        results = backtest_payload.get("trade_results", [])
        executed = [
            item
            for item in results
            if item.get("entry_hit") and str(item.get("outcome")) in CLOSED_OUTCOMES
        ]
        if not executed:
            return {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "average_rr": 0.0,
                "profit_factor": None,
                "expectancy": 0.0,
                "maximum_drawdown_points": 0.0,
                "net_points": 0.0,
            }

        pnls = [float(item.get("realized_pnl_points", 0.0)) for item in executed]
        wins = sum(1 for item in executed if str(item.get("outcome")) == "Win")
        total = len(executed)
        return {
            "total_trades": total,
            "win_rate_pct": round(wins / total * 100, 2),
            "average_rr": round(
                sum(float(item.get("realized_rr", 0.0)) for item in executed) / total,
                2,
            ),
            "profit_factor": self._profit_factor(pnls),
            "expectancy": round(sum(pnls) / total, 2),
            "maximum_drawdown_points": self._maximum_drawdown(pnls),
            "net_points": round(sum(pnls), 2),
        }

    def run_single(
        self,
        symbol: str,
        timeframe_label: str,
        end_date: date | None = None,
    ) -> BatchRunMetrics:
        """Execute the full engine stack for one symbol/timeframe."""
        started = time.perf_counter()
        start, end = self._date_range(end_date)
        storage_tf = self._storage_timeframe(timeframe_label)
        paths = self.artifact_paths(symbol, timeframe_label)

        try:
            self.ensure_data(symbol, timeframe_label, start, end)

            paths["run_dir"].mkdir(parents=True, exist_ok=True)
            paths["pipeline_csv"].parent.mkdir(parents=True, exist_ok=True)

            pipeline_report = MarketPipelineRunner(
                symbol=symbol,
                timeframe=storage_tf,
                start_date=start,
                end_date=end,
                output_csv=paths["pipeline_csv"],
                report_json=paths["pipeline_report"],
            ).run()
            if not pipeline_report.success:
                raise MarketPipelineError(pipeline_report.failure_message or "Pipeline failed")

            decision_report = evaluate_pipeline(
                pipeline_csv=paths["pipeline_csv"],
                report_path=paths["decision_report"],
                symbol=symbol,
                timeframe=storage_tf,
            )[1]

            generate_multi_timeframe_report(
                symbol=symbol,
                lookback_days=self.analysis_days,
                pipeline_csv=paths["pipeline_csv"]
                if storage_tf == MTF_BASE_TIMEFRAME
                else None,
                report_path=paths["mtf_report"],
            )

            trade_plan_report = generate_trade_plans_v2(
                pipeline_csv=paths["pipeline_csv"],
                mtf_report_path=paths["mtf_report"],
                report_path=paths["trade_plan_v2"],
                symbol=symbol,
                timeframe=storage_tf,
            )[1]

            quality_report = generate_signal_quality_report(
                pipeline_csv=paths["pipeline_csv"],
                mtf_report_path=paths["mtf_report"],
                trade_plan_path=paths["trade_plan_v2"],
                report_path=paths["signal_quality"],
                symbol=symbol,
                timeframe=storage_tf,
            )

            backtest_report = BacktestEngine(symbol=symbol, timeframe=storage_tf).run(
                trade_plan_report=paths["trade_plan_v2"],
            )
            paths["backtest_report"].parent.mkdir(parents=True, exist_ok=True)
            with paths["backtest_report"].open("w", encoding="utf-8") as handle:
                serializable = backtest_report.as_dict()
                if serializable.get("profit_factor") == float("inf"):
                    serializable["profit_factor"] = "inf"
                json.dump(serializable, handle, indent=2)
            pd.DataFrame(backtest_report.trade_results).to_csv(paths["trade_results"], index=False)

            with paths["signal_quality"].open("r", encoding="utf-8") as handle:
                quality_payload = json.load(handle)
            grade_stats = self._compute_grade_statistics(
                quality_payload,
                serializable,
            )
            metrics = self._compute_metrics_from_backtest(serializable)

            elapsed = time.perf_counter() - started
            return BatchRunMetrics(
                symbol=symbol,
                timeframe=timeframe_label,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                analysis_days=self.analysis_days,
                success=True,
                total_trades=metrics["total_trades"],
                win_rate_pct=metrics["win_rate_pct"],
                average_rr=metrics["average_rr"],
                profit_factor=metrics["profit_factor"],
                expectancy=metrics["expectancy"],
                maximum_drawdown_points=metrics["maximum_drawdown_points"],
                net_points=metrics["net_points"],
                grade_statistics=grade_stats,
                pipeline_rows=pipeline_report.rows,
                buy_signals=decision_report.buy_count,
                sell_signals=decision_report.sell_count,
                valid_trade_plans=trade_plan_report.valid_trade_plans,
                average_quality_score=quality_report.average_score,
                execution_time_seconds=elapsed,
                artifact_paths={key: str(value) for key, value in paths.items()},
            )
        except (
            BatchBacktesterError,
            MarketPipelineError,
            DecisionEngineError,
            MultiTimeframeEngineError,
            TradePlanEngineV2Error,
            SignalQualityEngineError,
            BacktestEngineError,
            DataLoaderError,
        ) as exc:
            elapsed = time.perf_counter() - started
            logger.error(
                "Batch run failed for %s/%s: %s",
                symbol,
                timeframe_label,
                exc,
            )
            return BatchRunMetrics(
                symbol=symbol,
                timeframe=timeframe_label,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                analysis_days=self.analysis_days,
                success=False,
                total_trades=0,
                win_rate_pct=0.0,
                average_rr=0.0,
                profit_factor=None,
                expectancy=0.0,
                maximum_drawdown_points=0.0,
                net_points=0.0,
                grade_statistics={grade: {"signals": 0, "trades": 0} for grade in GRADE_BUCKETS},
                pipeline_rows=0,
                buy_signals=0,
                sell_signals=0,
                valid_trade_plans=0,
                average_quality_score=0.0,
                error_message=str(exc),
                execution_time_seconds=elapsed,
                artifact_paths={key: str(value) for key, value in paths.items()},
            )

    @staticmethod
    def _aggregate_grade_statistics(
        runs: list[BatchRunMetrics],
    ) -> dict[str, dict[str, Any]]:
        aggregate: dict[str, dict[str, Any]] = {
            grade: {
                "signals": 0,
                "trades": 0,
                "wins": 0,
                "win_rate_pct": 0.0,
                "net_points": 0.0,
                "expectancy": 0.0,
            }
            for grade in GRADE_BUCKETS
        }
        for run in runs:
            if not run.success:
                continue
            for grade in GRADE_BUCKETS:
                bucket = run.grade_statistics.get(grade, {})
                aggregate[grade]["signals"] += int(bucket.get("signals", 0))
                aggregate[grade]["trades"] += int(bucket.get("trades", 0))
                aggregate[grade]["wins"] += int(bucket.get("wins", 0))
                aggregate[grade]["net_points"] = round(
                    aggregate[grade]["net_points"] + float(bucket.get("net_points", 0.0)),
                    2,
                )

        for grade, bucket in aggregate.items():
            trades = bucket["trades"]
            if trades:
                bucket["win_rate_pct"] = round(bucket["wins"] / trades * 100, 2)
                bucket["expectancy"] = round(bucket["net_points"] / trades, 2)
        return aggregate

    @staticmethod
    def _rank_runs(runs: list[BatchRunMetrics]) -> dict[str, list[dict[str, Any]]]:
        successful = [run for run in runs if run.success and run.total_trades > 0]
        if not successful:
            return {"by_expectancy": [], "by_win_rate": [], "by_profit_factor": []}

        def pf_value(value: float | None) -> float:
            if value is None:
                return 0.0
            if value == float("inf"):
                return 9999.0
            return float(value)

        by_expectancy = sorted(successful, key=lambda run: run.expectancy, reverse=True)
        by_win_rate = sorted(successful, key=lambda run: run.win_rate_pct, reverse=True)
        by_profit_factor = sorted(
            successful,
            key=lambda run: pf_value(run.profit_factor),
            reverse=True,
        )

        def _compact(run: BatchRunMetrics) -> dict[str, Any]:
            return {
                "symbol": run.symbol,
                "timeframe": run.timeframe,
                "expectancy": run.expectancy,
                "win_rate_pct": run.win_rate_pct,
                "profit_factor": "inf"
                if run.profit_factor == float("inf")
                else run.profit_factor,
                "total_trades": run.total_trades,
            }

        return {
            "by_expectancy": [_compact(run) for run in by_expectancy],
            "by_win_rate": [_compact(run) for run in by_win_rate],
            "by_profit_factor": [_compact(run) for run in by_profit_factor],
        }

    @staticmethod
    def _best_symbol_and_timeframe(
        runs: list[BatchRunMetrics],
    ) -> tuple[str | None, str | None]:
        successful = [run for run in runs if run.success]
        if not successful:
            return None, None

        by_expectancy = sorted(successful, key=lambda run: run.expectancy, reverse=True)
        best_run = by_expectancy[0]

        symbol_scores: dict[str, float] = {}
        timeframe_scores: dict[str, float] = {}
        for run in successful:
            symbol_scores[run.symbol] = symbol_scores.get(run.symbol, 0.0) + run.expectancy
            timeframe_scores[run.timeframe] = (
                timeframe_scores.get(run.timeframe, 0.0) + run.expectancy
            )

        best_symbol = max(symbol_scores, key=symbol_scores.get)
        best_timeframe = max(timeframe_scores, key=timeframe_scores.get)
        if best_run.total_trades == 0 and all(run.total_trades == 0 for run in successful):
            return best_symbol, best_timeframe
        return best_symbol, best_timeframe

    def run_all(self, end_date: date | None = None) -> BatchReport:
        """Run the complete batch across configured symbols and timeframes."""
        started = time.perf_counter()
        start, end = self._date_range(end_date)
        runs: list[BatchRunMetrics] = []

        for symbol in self.symbols:
            for timeframe_label in self.timeframes:
                logger.info("Starting batch run: %s / %s", symbol, timeframe_label)
                runs.append(self.run_single(symbol, timeframe_label, end_date=end))

        successful = [run for run in runs if run.success]
        rankings = self._rank_runs(runs)
        best_symbol, best_timeframe = self._best_symbol_and_timeframe(runs)
        aggregate_grades = self._aggregate_grade_statistics(runs)
        elapsed = time.perf_counter() - started

        return BatchReport(
            analysis_days=self.analysis_days,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            symbols=list(self.symbols),
            timeframes=list(self.timeframes),
            total_runs=len(runs),
            successful_runs=len(successful),
            failed_runs=len(runs) - len(successful),
            best_symbol=best_symbol,
            best_timeframe=best_timeframe,
            runs=[run.as_dict() for run in runs],
            rankings=rankings,
            grade_statistics_aggregate=aggregate_grades,
            execution_time_seconds=elapsed,
        )


def generate_batch_report(
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    analysis_days: int = MIN_ANALYSIS_DAYS,
    auto_download: bool = True,
    report_path: Path | str | None = None,
) -> BatchReport:
    """Run batch backtesting and export the master JSON report."""
    backtester = BatchBacktester(
        symbols=symbols,
        timeframes=timeframes,
        analysis_days=analysis_days,
        auto_download=auto_download,
    )
    report = backtester.run_all()

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Batch backtest completed: runs=%s success=%s best=%s/%s",
        report.total_runs,
        report.successful_runs,
        report.best_symbol,
        report.best_timeframe,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_batch_report()
        print("Batch Backtester Summary")
        print(f"Analysis Window: {report.analysis_days} days")
        print(f"Date Range: {report.start_date} -> {report.end_date}")
        print(f"Runs: {report.successful_runs}/{report.total_runs} successful")
        print(f"Best Symbol: {report.best_symbol}")
        print(f"Best Timeframe: {report.best_timeframe}")
        print("Per Run Metrics:")
        for run in report.runs:
            if not run["success"]:
                print(
                    f"  - {run['symbol']} {run['timeframe']}: FAILED "
                    f"({run.get('error_message')})"
                )
                continue
            pf = run["profit_factor"]
            pf_display = "inf" if pf == "inf" else (pf if pf is not None else "N/A")
            print(
                f"  - {run['symbol']} {run['timeframe']}: trades={run['total_trades']} "
                f"win_rate={run['win_rate_pct']}% pf={pf_display} "
                f"expectancy={run['expectancy']} drawdown={run['maximum_drawdown_points']}"
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0 if report.failed_runs == 0 else 1
    except BatchBacktesterError as exc:
        logger.error("Batch backtester error: %s", exc)
        print(f"Batch backtester error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected batch backtester failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
