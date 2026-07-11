"""
Real-market SmartMoneyEngine integration pipeline.

Loads FYERS historical data, validates it, runs the full SMC detector
stack, and exports enriched results with a structured pipeline report.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader.data_loader import HistoricalDataLoader
from src.data.validation.dataset_validator import DatasetValidator, ValidationResult
from src.models.market_data import MarketData
from src.smc.base_smc import BaseSMC
from src.smc.bos import BreakOfStructure
from src.smc.choch import ChangeOfCharacter
from src.smc.fvg import FairValueGap
from src.smc.liquidity import LiquidityDetector
from src.smc.market_structure import MarketStructure
from src.smc.order_block import OrderBlockDetector
from src.smc.swing_detector import SwingDetector
from src.smc.trend_engine import TrendEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_JSON = DEFAULT_OUTPUT_DIR / "pipeline_report.json"

BASE_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MAX_NEW_NAN_RATIO = 0.95


class MarketPipelineError(Exception):
    """Raised when the market pipeline cannot complete."""


@dataclass(frozen=True)
class StageDefinition:
    """Pipeline stage metadata."""

    name: str
    factory: Callable[[], BaseSMC]
    required_columns: tuple[str, ...]
    output_columns: tuple[str, ...]


@dataclass
class PipelineStageResult:
    """Outcome of a single pipeline stage."""

    name: str
    success: bool
    duration_seconds: float
    rows: int
    columns_before: tuple[str, ...]
    columns_after: tuple[str, ...]
    columns_added: tuple[str, ...]
    memory_bytes: int
    removed_columns: tuple[str, ...] = ()
    duplicate_columns: tuple[str, ...] = ()
    base_nan_count: int = 0
    total_nan_count: int = 0
    error_message: str | None = None


@dataclass
class MarketPipelineReport:
    """Aggregate report for a market pipeline run."""

    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    rows: int
    columns: list[str]
    execution_time_seconds: float
    memory_bytes: int
    validation: dict[str, Any]
    signals: dict[str, int]
    bos_count: int
    choch_count: int
    fvg_count: int
    bullish_ob_count: int
    bearish_ob_count: int
    liquidity_count: int
    stages: list[dict[str, Any]] = field(default_factory=list)
    output_csv: str | None = None
    report_json: str | None = None
    success: bool = False
    failure_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return asdict(self)


def _build_pipeline_stages() -> tuple[StageDefinition, ...]:
    """Define the canonical SmartMoneyEngine detector pipeline."""
    return (
        StageDefinition(
            name="SwingDetector",
            factory=SwingDetector,
            required_columns=SwingDetector.REQUIRED_COLUMNS,
            output_columns=(
                SwingDetector.SWING_HIGH_COLUMN,
                SwingDetector.SWING_LOW_COLUMN,
            ),
        ),
        StageDefinition(
            name="MarketStructure",
            factory=MarketStructure,
            required_columns=MarketStructure.REQUIRED_COLUMNS,
            output_columns=(
                MarketStructure.HH_COLUMN,
                MarketStructure.HL_COLUMN,
                MarketStructure.LH_COLUMN,
                MarketStructure.LL_COLUMN,
            ),
        ),
        StageDefinition(
            name="TrendEngine",
            factory=TrendEngine,
            required_columns=TrendEngine.REQUIRED_COLUMNS,
            output_columns=(
                TrendEngine.TREND_COLUMN,
                TrendEngine.TREND_STRENGTH_COLUMN,
            ),
        ),
        StageDefinition(
            name="BreakOfStructure",
            factory=BreakOfStructure,
            required_columns=BreakOfStructure.REQUIRED_COLUMNS,
            output_columns=(
                BreakOfStructure.BULLISH_BOS_COLUMN,
                BreakOfStructure.BEARISH_BOS_COLUMN,
            ),
        ),
        StageDefinition(
            name="ChangeOfCharacter",
            factory=ChangeOfCharacter,
            required_columns=ChangeOfCharacter.REQUIRED_COLUMNS,
            output_columns=(
                ChangeOfCharacter.BULLISH_CHOCH_COLUMN,
                ChangeOfCharacter.BEARISH_CHOCH_COLUMN,
            ),
        ),
        StageDefinition(
            name="FairValueGap",
            factory=FairValueGap,
            required_columns=FairValueGap.REQUIRED_COLUMNS,
            output_columns=(
                FairValueGap.BULLISH_FVG_TOP_COLUMN,
                FairValueGap.BULLISH_FVG_BOTTOM_COLUMN,
                FairValueGap.BEARISH_FVG_TOP_COLUMN,
                FairValueGap.BEARISH_FVG_BOTTOM_COLUMN,
            ),
        ),
        StageDefinition(
            name="OrderBlockDetector",
            factory=OrderBlockDetector,
            required_columns=OrderBlockDetector.REQUIRED_COLUMNS,
            output_columns=(
                OrderBlockDetector.BULLISH_OB_HIGH_COLUMN,
                OrderBlockDetector.BULLISH_OB_LOW_COLUMN,
                OrderBlockDetector.BEARISH_OB_HIGH_COLUMN,
                OrderBlockDetector.BEARISH_OB_LOW_COLUMN,
                OrderBlockDetector.BULLISH_OB_MITIGATED_COLUMN,
                OrderBlockDetector.BEARISH_OB_MITIGATED_COLUMN,
            ),
        ),
        StageDefinition(
            name="LiquidityDetector",
            factory=LiquidityDetector,
            required_columns=LiquidityDetector.REQUIRED_COLUMNS,
            output_columns=(
                LiquidityDetector.EQUAL_HIGH_COLUMN,
                LiquidityDetector.EQUAL_LOW_COLUMN,
                LiquidityDetector.BUY_SIDE_LIQUIDITY_COLUMN,
                LiquidityDetector.SELL_SIDE_LIQUIDITY_COLUMN,
                LiquidityDetector.BUY_LIQUIDITY_SWEEP_COLUMN,
                LiquidityDetector.SELL_LIQUIDITY_SWEEP_COLUMN,
                LiquidityDetector.LIQUIDITY_STRENGTH_COLUMN,
            ),
        ),
    )


def _duplicate_column_names(columns: Sequence[str]) -> tuple[str, ...]:
    """Return duplicate column names."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
            continue
        seen.add(column)
    return tuple(duplicates)


def _missing_columns(columns: Sequence[str], required: Sequence[str]) -> tuple[str, ...]:
    """Return required columns missing from the frame."""
    column_set = set(columns)
    return tuple(column for column in required if column not in column_set)


def _memory_bytes(dataframe: pd.DataFrame) -> int:
    """Return deep memory usage for a DataFrame."""
    return int(dataframe.memory_usage(deep=True).sum())


def _count_nan(dataframe: pd.DataFrame, columns: Sequence[str]) -> int:
    """Count NaN values across selected columns."""
    total = 0
    for column in columns:
        if column in dataframe.columns:
            total += int(dataframe[column].isna().sum())
    return total


def prepare_market_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize FYERS loader output into SmartMoneyEngine OHLCV schema.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw loader output with lowercase OHLCV columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Date``, ``Open``, ``High``, ``Low``, ``Close``,
        and ``Volume`` columns.
    """
    working = frame.copy().reset_index(drop=True)

    rename_map = {
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    working = working.rename(columns={key: value for key, value in rename_map.items() if key in working.columns})

    if "Date" in working.columns and pd.api.types.is_datetime64_any_dtype(working["Date"]):
        working["Date"] = working["Date"].dt.tz_convert("Asia/Kolkata").astype(str)

    missing = _missing_columns(working.columns, BASE_OHLCV_COLUMNS)
    if missing:
        raise MarketPipelineError(
            f"Prepared market dataframe is missing required columns: {missing}"
        )

    return working


def _count_signal(frame: pd.DataFrame, columns: Sequence[str]) -> int:
    """Count non-null signal occurrences across columns."""
    total = 0
    for column in columns:
        if column in frame.columns:
            total += int(frame[column].notna().sum())
    return total


def _extract_signal_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Extract aggregate SMC signal counts from the final frame."""
    bos_count = _count_signal(
        frame,
        (BreakOfStructure.BULLISH_BOS_COLUMN, BreakOfStructure.BEARISH_BOS_COLUMN),
    )
    choch_count = _count_signal(
        frame,
        (ChangeOfCharacter.BULLISH_CHOCH_COLUMN, ChangeOfCharacter.BEARISH_CHOCH_COLUMN),
    )
    fvg_count = _count_signal(
        frame,
        (FairValueGap.BULLISH_FVG_TOP_COLUMN, FairValueGap.BEARISH_FVG_TOP_COLUMN),
    )
    bullish_ob_count = _count_signal(frame, (OrderBlockDetector.BULLISH_OB_HIGH_COLUMN,))
    bearish_ob_count = _count_signal(frame, (OrderBlockDetector.BEARISH_OB_HIGH_COLUMN,))
    liquidity_count = _count_signal(
        frame,
        (
            LiquidityDetector.BUY_SIDE_LIQUIDITY_COLUMN,
            LiquidityDetector.SELL_SIDE_LIQUIDITY_COLUMN,
        ),
    )

    return {
        "bos_count": bos_count,
        "choch_count": choch_count,
        "fvg_count": fvg_count,
        "bullish_ob_count": bullish_ob_count,
        "bearish_ob_count": bearish_ob_count,
        "liquidity_count": liquidity_count,
        "total_signals": bos_count + choch_count + fvg_count + bullish_ob_count + bearish_ob_count + liquidity_count,
    }


class MarketPipelineRunner:
    """
    Execute the SmartMoneyEngine pipeline on real historical market data.

    Parameters
    ----------
    symbol : str
        Symbol to load, for example ``NIFTY50``.
    timeframe : str
        Candle timeframe such as ``5``.
    start_date : date | str
        Inclusive start date.
    end_date : date | str
        Inclusive end date.
    output_csv : Path | None, optional
        Output CSV path.
    report_json : Path | None, optional
        Output JSON report path.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        output_csv: Path | None = None,
        report_json: Path | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.end_date = end_date if end_date is not None else date.today()
        self.start_date = (
            start_date
            if start_date is not None
            else (self._parse_date(self.end_date) - timedelta(days=30))
        )
        self.output_csv = output_csv if output_csv is not None else DEFAULT_OUTPUT_CSV
        self.report_json = report_json if report_json is not None else DEFAULT_REPORT_JSON
        self.stages = _build_pipeline_stages()

    @staticmethod
    def _parse_date(value: date | str) -> date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value.strip())
        raise MarketPipelineError(f"Unsupported date value: {value!r}")

    def run(self) -> MarketPipelineReport:
        """Execute the complete market pipeline."""
        pipeline_started = time.perf_counter()
        stage_results: list[PipelineStageResult] = []
        failure_message: str | None = None
        validation_result: ValidationResult | None = None
        market: MarketData | None = None
        final_frame: pd.DataFrame | None = None

        try:
            loader = HistoricalDataLoader()
            raw_frame = loader.load(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
            )

            validator = DatasetValidator(timeframe=self.timeframe)
            validation_result = validator.validate(raw_frame)
            if not validation_result.is_valid:
                raise MarketPipelineError(
                    "Dataset validation failed: " + "; ".join(validation_result.errors)
                )

            prepared = prepare_market_dataframe(raw_frame)
            market = MarketData(prepared)

            if _duplicate_column_names(market.columns):
                raise MarketPipelineError(
                    "Duplicate column names detected before pipeline execution."
                )

            for stage in self.stages:
                result = self._run_stage(market, stage)
                stage_results.append(result)
                if not result.success:
                    failure_message = result.error_message
                    break

            if failure_message is None and market is not None:
                final_frame = market.data
                self.output_csv.parent.mkdir(parents=True, exist_ok=True)
                final_frame.to_csv(self.output_csv, index=False)

        except Exception as exc:
            failure_message = str(exc)

        elapsed = time.perf_counter() - pipeline_started
        rows = 0 if market is None else market.rows
        columns = [] if market is None else list(market.columns)
        memory_bytes = 0 if market is None else _memory_bytes(market.data)

        signals: dict[str, int] = {}
        if final_frame is not None:
            signals = _extract_signal_counts(final_frame)

        report = MarketPipelineReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self._parse_date(self.start_date).isoformat(),
            end_date=self._parse_date(self.end_date).isoformat(),
            rows=rows,
            columns=columns,
            execution_time_seconds=elapsed,
            memory_bytes=memory_bytes,
            validation=validation_result.summary() if validation_result is not None else {},
            signals=signals,
            bos_count=signals.get("bos_count", 0),
            choch_count=signals.get("choch_count", 0),
            fvg_count=signals.get("fvg_count", 0),
            bullish_ob_count=signals.get("bullish_ob_count", 0),
            bearish_ob_count=signals.get("bearish_ob_count", 0),
            liquidity_count=signals.get("liquidity_count", 0),
            stages=[asdict(stage) for stage in stage_results],
            output_csv=str(self.output_csv) if failure_message is None else None,
            success=failure_message is None and bool(stage_results) and all(stage.success for stage in stage_results),
            failure_message=failure_message,
        )

        if report.success:
            self.report_json.parent.mkdir(parents=True, exist_ok=True)
            with self.report_json.open("w", encoding="utf-8") as handle:
                json.dump(report.as_dict(), handle, indent=2)
            report.report_json = str(self.report_json)

        return report

    def _run_stage(self, market: MarketData, stage: StageDefinition) -> PipelineStageResult:
        """Execute and validate a single detector stage."""
        stage_started = time.perf_counter()
        columns_before = tuple(market.columns)
        base_nan_before = _count_nan(market.data, BASE_OHLCV_COLUMNS)

        missing_inputs = _missing_columns(columns_before, stage.required_columns)
        if missing_inputs:
            return PipelineStageResult(
                name=stage.name,
                success=False,
                duration_seconds=time.perf_counter() - stage_started,
                rows=market.rows,
                columns_before=columns_before,
                columns_after=columns_before,
                columns_added=(),
                memory_bytes=_memory_bytes(market.data),
                error_message=f"{stage.name} missing required input columns: {', '.join(missing_inputs)}",
            )

        try:
            detector = stage.factory()
            detector.detect(market)
        except Exception as exc:
            return PipelineStageResult(
                name=stage.name,
                success=False,
                duration_seconds=time.perf_counter() - stage_started,
                rows=market.rows,
                columns_before=columns_before,
                columns_after=tuple(market.columns),
                columns_added=(),
                memory_bytes=_memory_bytes(market.data),
                error_message=f"{stage.name} failed: {exc}",
            )

        columns_after = tuple(market.columns)
        columns_added = tuple(column for column in stage.output_columns if column in columns_after)
        removed_columns = tuple(column for column in columns_before if column not in columns_after)
        duplicate_columns = _duplicate_column_names(columns_after)
        base_nan_after = _count_nan(market.data, BASE_OHLCV_COLUMNS)
        total_nan_after = int(market.data.isna().sum().sum())
        duration = time.perf_counter() - stage_started
        memory_bytes = _memory_bytes(market.data)

        logger.info(
            "Stage %s completed in %.3fs | rows=%s | added=%s | memory=%.2f MB",
            stage.name,
            duration,
            market.rows,
            ", ".join(columns_added) if columns_added else "none",
            memory_bytes / (1024 * 1024),
        )

        if removed_columns:
            return PipelineStageResult(
                name=stage.name,
                success=False,
                duration_seconds=duration,
                rows=market.rows,
                columns_before=columns_before,
                columns_after=columns_after,
                columns_added=columns_added,
                memory_bytes=memory_bytes,
                removed_columns=removed_columns,
                duplicate_columns=duplicate_columns,
                base_nan_count=base_nan_after,
                total_nan_count=total_nan_after,
                error_message=f"{stage.name} removed prior columns: {', '.join(removed_columns)}",
            )

        if duplicate_columns:
            return PipelineStageResult(
                name=stage.name,
                success=False,
                duration_seconds=duration,
                rows=market.rows,
                columns_before=columns_before,
                columns_after=columns_after,
                columns_added=columns_added,
                memory_bytes=memory_bytes,
                removed_columns=removed_columns,
                duplicate_columns=duplicate_columns,
                base_nan_count=base_nan_after,
                total_nan_count=total_nan_after,
                error_message=f"Duplicate column names detected: {', '.join(duplicate_columns)}",
            )

        missing_outputs = _missing_columns(columns_after, stage.output_columns)
        if missing_outputs:
            return PipelineStageResult(
                name=stage.name,
                success=False,
                duration_seconds=duration,
                rows=market.rows,
                columns_before=columns_before,
                columns_after=columns_after,
                columns_added=columns_added,
                memory_bytes=memory_bytes,
                removed_columns=removed_columns,
                duplicate_columns=duplicate_columns,
                base_nan_count=base_nan_after,
                total_nan_count=total_nan_after,
                error_message=f"{stage.name} did not create expected columns: {', '.join(missing_outputs)}",
            )

        if base_nan_after > base_nan_before:
            return PipelineStageResult(
                name=stage.name,
                success=False,
                duration_seconds=duration,
                rows=market.rows,
                columns_before=columns_before,
                columns_after=columns_after,
                columns_added=columns_added,
                memory_bytes=memory_bytes,
                removed_columns=removed_columns,
                duplicate_columns=duplicate_columns,
                base_nan_count=base_nan_after,
                total_nan_count=total_nan_after,
                error_message=f"{stage.name} introduced NaN values in base OHLCV columns.",
            )

        new_columns = [column for column in columns_added if column in market.data.columns]
        if new_columns:
            new_nan_ratio = market.data[new_columns].isna().mean().mean()
            if new_nan_ratio > MAX_NEW_NAN_RATIO:
                logger.warning(
                    "Stage %s produced sparse outputs (NaN ratio %.2f).",
                    stage.name,
                    new_nan_ratio,
                )

        return PipelineStageResult(
            name=stage.name,
            success=True,
            duration_seconds=duration,
            rows=market.rows,
            columns_before=columns_before,
            columns_after=columns_after,
            columns_added=columns_added,
            memory_bytes=memory_bytes,
            removed_columns=removed_columns,
            duplicate_columns=duplicate_columns,
            base_nan_count=base_nan_after,
            total_nan_count=total_nan_after,
        )


def run_market_pipeline(
    symbol: str = "NIFTY50",
    timeframe: str = "5",
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> MarketPipelineReport:
    """Run the market pipeline with default output locations."""
    runner = MarketPipelineRunner(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    return runner.run()


def print_pipeline_summary(report: MarketPipelineReport) -> None:
    """Print a human-readable pipeline summary."""
    print("Market Pipeline Summary")
    print(f"Symbol: {report.symbol}")
    print(f"Timeframe: {report.timeframe}")
    print(f"Date Range: {report.start_date} to {report.end_date}")
    print(f"Rows: {report.rows}")
    print(f"Columns: {len(report.columns)}")
    print(f"Execution Time: {report.execution_time_seconds:.3f}s")
    print(f"Memory: {report.memory_bytes / (1024 * 1024):.2f} MB")
    print(f"Success: {report.success}")
    print(f"BOS Count: {report.bos_count}")
    print(f"CHOCH Count: {report.choch_count}")
    print(f"FVG Count: {report.fvg_count}")
    print(f"Bullish OB Count: {report.bullish_ob_count}")
    print(f"Bearish OB Count: {report.bearish_ob_count}")
    print(f"Liquidity Count: {report.liquidity_count}")
    if report.output_csv:
        print(f"Output CSV: {report.output_csv}")
    if report.report_json:
        print(f"Report JSON: {report.report_json}")
    if report.failure_message:
        print(f"Failure: {report.failure_message}")


def main() -> int:
    """CLI entry point."""
    try:
        report = run_market_pipeline()
        print_pipeline_summary(report)
        return 0 if report.success else 1
    except Exception as exc:
        logger.exception("Market pipeline failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
