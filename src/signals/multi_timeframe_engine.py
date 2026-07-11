"""
SmartMoneyEngine multi-timeframe analysis engine.

Resamples real FYERS historical data into multiple timeframes, runs the SMC
pipeline and decision layer on each series, and produces an alignment report.
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

from src.data.loader.data_loader import HistoricalDataLoader
from src.models.market_data import MarketData
from src.pipeline.market_pipeline import prepare_market_dataframe
from src.signals.decision_engine import DecisionEngine, InstitutionalBias, MarketBias
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
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "signals" / "multi_timeframe_report.json"

TIMEFRAME_SPECS: tuple[tuple[str, str], ...] = (
    ("1D", "1D"),
    ("4H", "4h"),
    ("1H", "1h"),
    ("15M", "15min"),
    ("5M", "5min"),
)

RECENT_EVENT_WINDOW = 5
STRUCTURE_LOOKBACK = 20
MIN_BARS_REQUIRED = 20


class OverallBias(str, Enum):
    """Aggregate multi-timeframe bias."""

    STRONG_BULLISH = "Strong Bullish"
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"
    STRONG_BEARISH = "Strong Bearish"


class StructureState(str, Enum):
    """Dominant market structure state."""

    HH_HL = "HH-HL"
    LH_LL = "LH-LL"
    RANGE = "Range"


class MultiTimeframeEngineError(Exception):
    """Raised when multi-timeframe analysis fails."""


@dataclass(frozen=True)
class TimeframeAnalysis:
    """Analysis summary for one timeframe."""

    timeframe: str
    trend: str
    structure: str
    bos_status: str
    choch_status: str
    liquidity_status: str
    institutional_bias: str
    bars_analyzed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiTimeframeReport:
    """Aggregate multi-timeframe report."""

    symbol: str
    source_timeframe: str
    start_date: str
    end_date: str
    execution_time_seconds: float
    alignment_score: int
    overall_bias: str
    bullish_timeframes: int
    bearish_timeframes: int
    neutral_timeframes: int
    timeframes: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiTimeframeEngine:
    """
    Analyze SmartMoneyEngine outputs across multiple timeframes.

    Parameters
    ----------
    symbol : str, optional
        Symbol to analyze.
    lookback_days : int, optional
        Calendar days of 5-minute history to load for resampling.
    """

    def __init__(self, symbol: str = "NIFTY50", lookback_days: int = 30) -> None:
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.decision_engine = DecisionEngine(symbol=symbol, timeframe="5")

    @staticmethod
    def _is_active(value: Any) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return not pd.isna(value)
        return bool(str(value).strip())

    @staticmethod
    def _normalize_loader_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure lowercase OHLCV columns and timezone-aware timestamps."""
        working = frame.copy()
        if "timestamp" not in working.columns and "Date" in working.columns:
            working = working.rename(columns={"Date": "timestamp"})
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
        if working["timestamp"].dt.tz is None:
            working["timestamp"] = working["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            working["timestamp"] = working["timestamp"].dt.tz_convert("Asia/Kolkata")

        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        working = working.rename(columns={key: value for key, value in rename.items() if key in working.columns})
        return working.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV data to a higher timeframe."""
        indexed = frame.set_index("timestamp")
        resampled = (
            indexed.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        return resampled

    def _run_smc_pipeline(self, prepared: pd.DataFrame) -> pd.DataFrame:
        """Run the SMC detector stack on a prepared OHLCV dataframe."""
        market = MarketData(prepared.copy())
        SwingDetector().detect(market)
        MarketStructure().detect(market)
        TrendEngine().detect(market)
        BreakOfStructure().detect(market)
        ChangeOfCharacter().detect(market)
        FairValueGap().detect(market)
        OrderBlockDetector().detect(market)
        LiquidityDetector().detect(market)
        return self.decision_engine.evaluate(market.data)

    def _load_base_5m_frame(
        self,
        start_date: date,
        end_date: date,
        pipeline_csv: Path | None = None,
    ) -> pd.DataFrame:
        """Load base 5-minute OHLCV data."""
        if pipeline_csv is not None and pipeline_csv.exists():
            pipeline = pd.read_csv(pipeline_csv)
            if {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(pipeline.columns):
                base = pipeline[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
                base = base.rename(columns={"Date": "timestamp"})
                base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")
                return self._normalize_loader_frame(base)

        loader = HistoricalDataLoader()
        loaded = loader.load(
            symbol=self.symbol,
            timeframe="5",
            start_date=start_date,
            end_date=end_date,
        )
        return self._normalize_loader_frame(loaded)

    def _structure_state(self, frame: pd.DataFrame, trend: str) -> str:
        """Classify dominant structure labels over a recent window."""
        if trend == MarketBias.NEUTRAL.value:
            return StructureState.RANGE.value

        window = frame.tail(STRUCTURE_LOOKBACK)
        hh = int(window["HH"].notna().sum()) if "HH" in window.columns else 0
        hl = int(window["HL"].notna().sum()) if "HL" in window.columns else 0
        lh = int(window["LH"].notna().sum()) if "LH" in window.columns else 0
        ll = int(window["LL"].notna().sum()) if "LL" in window.columns else 0

        bullish_structure = hh + hl
        bearish_structure = lh + ll

        if bullish_structure > bearish_structure * 1.25:
            return StructureState.HH_HL.value
        if bearish_structure > bullish_structure * 1.25:
            return StructureState.LH_LL.value
        return StructureState.RANGE.value

    @staticmethod
    def _recent_event_status(
        frame: pd.DataFrame,
        bullish_column: str,
        bearish_column: str,
        bullish_label: str,
        bearish_label: str,
    ) -> str:
        """Summarize recent bullish/bearish event activity."""
        window = frame.tail(RECENT_EVENT_WINDOW)
        bullish_active = any(
            MultiTimeframeEngine._is_active(value) for value in window.get(bullish_column, [])
        )
        bearish_active = any(
            MultiTimeframeEngine._is_active(value) for value in window.get(bearish_column, [])
        )
        if bullish_active and bearish_active:
            return "Mixed"
        if bullish_active:
            return bullish_label
        if bearish_active:
            return bearish_label
        return "None"

    @staticmethod
    def _liquidity_status(frame: pd.DataFrame) -> str:
        """Summarize liquidity conditions on the latest bars."""
        window = frame.tail(RECENT_EVENT_WINDOW)
        buy_sweep = any(
            MultiTimeframeEngine._is_active(value)
            for value in window.get("Buy_Liquidity_Sweep", [])
        )
        sell_sweep = any(
            MultiTimeframeEngine._is_active(value)
            for value in window.get("Sell_Liquidity_Sweep", [])
        )
        buy_pool = any(
            MultiTimeframeEngine._is_active(value)
            for value in window.get("Buy_Side_Liquidity", [])
        )
        sell_pool = any(
            MultiTimeframeEngine._is_active(value)
            for value in window.get("Sell_Side_Liquidity", [])
        )

        if buy_sweep and sell_sweep:
            return "Mixed liquidity sweeps"
        if buy_sweep:
            return "Buy-side liquidity swept"
        if sell_sweep:
            return "Sell-side liquidity swept"
        if buy_pool or sell_pool:
            return "Liquidity pools active"
        return "Neutral"

    def _analyze_timeframe(self, label: str, frame: pd.DataFrame) -> TimeframeAnalysis:
        """Analyze one timeframe dataframe."""
        if len(frame) < MIN_BARS_REQUIRED:
            raise MultiTimeframeEngineError(
                f"Insufficient bars for {label}: {len(frame)} (< {MIN_BARS_REQUIRED})."
            )

        prepared = prepare_market_dataframe(frame.rename(columns={"timestamp": "timestamp"}))
        enriched = self._run_smc_pipeline(prepared)
        latest = enriched.iloc[-1]

        trend = str(latest.get("Market_Bias", MarketBias.NEUTRAL.value))
        structure = self._structure_state(enriched, trend)
        bos_status = self._recent_event_status(
            enriched,
            "Bullish_BOS",
            "Bearish_BOS",
            "Bullish BOS active",
            "Bearish BOS active",
        )
        choch_status = self._recent_event_status(
            enriched,
            "Bullish_CHOCH",
            "Bearish_CHOCH",
            "Bullish CHOCH active",
            "Bearish CHOCH active",
        )

        return TimeframeAnalysis(
            timeframe=label,
            trend=trend,
            structure=structure,
            bos_status=bos_status,
            choch_status=choch_status,
            liquidity_status=self._liquidity_status(enriched),
            institutional_bias=str(
                latest.get("Institutional_Bias", InstitutionalBias.NEUTRAL.value)
            ),
            bars_analyzed=len(enriched),
        )

    @staticmethod
    def _compute_alignment(analyses: list[TimeframeAnalysis]) -> tuple[int, str, int, int, int]:
        """Compute alignment score and overall bias."""
        bullish = sum(1 for item in analyses if item.trend == MarketBias.BULLISH.value)
        bearish = sum(1 for item in analyses if item.trend == MarketBias.BEARISH.value)
        neutral = len(analyses) - bullish - bearish

        strong_bull = sum(
            1
            for item in analyses
            if item.institutional_bias == InstitutionalBias.STRONG_BULLISH.value
        )
        strong_bear = sum(
            1
            for item in analyses
            if item.institutional_bias == InstitutionalBias.STRONG_BEARISH.value
        )

        if bullish > bearish:
            alignment = bullish * 20
            if bullish == 5:
                overall = OverallBias.STRONG_BULLISH.value
            elif bullish == 4 and strong_bull >= 1:
                overall = OverallBias.STRONG_BULLISH.value
            elif bullish >= 4:
                overall = OverallBias.BULLISH.value
            elif bullish == 3:
                overall = OverallBias.BULLISH.value
            else:
                overall = OverallBias.NEUTRAL.value
        elif bearish > bullish:
            alignment = bearish * 20
            if bearish == 5:
                overall = OverallBias.STRONG_BEARISH.value
            elif bearish == 4 and strong_bear >= 1:
                overall = OverallBias.STRONG_BEARISH.value
            elif bearish >= 4:
                overall = OverallBias.BEARISH.value
            elif bearish == 3:
                overall = OverallBias.BEARISH.value
            else:
                overall = OverallBias.NEUTRAL.value
        else:
            alignment = max(bullish, bearish) * 20
            overall = OverallBias.NEUTRAL.value

        return alignment, overall, bullish, bearish, neutral

    def analyze(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        pipeline_csv: Path | None = None,
    ) -> MultiTimeframeReport:
        """Run multi-timeframe analysis and return the aggregate report."""
        started = time.perf_counter()
        end = end_date if end_date is not None else date.today()
        start = start if start_date is not None else end - timedelta(days=self.lookback_days)

        base_5m = self._load_base_5m_frame(start, end, pipeline_csv=pipeline_csv)
        analyses: list[TimeframeAnalysis] = []

        for label, rule in TIMEFRAME_SPECS:
            if label == "5M":
                frame = base_5m.copy()
            else:
                frame = self._resample_ohlcv(base_5m, rule)

            logger.info("Analyzing timeframe %s with %s bars.", label, len(frame))
            analyses.append(self._analyze_timeframe(label, frame))

        alignment, overall, bullish, bearish, neutral = self._compute_alignment(analyses)
        elapsed = time.perf_counter() - started

        return MultiTimeframeReport(
            symbol=self.symbol,
            source_timeframe="5M",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            execution_time_seconds=elapsed,
            alignment_score=alignment,
            overall_bias=overall,
            bullish_timeframes=bullish,
            bearish_timeframes=bearish,
            neutral_timeframes=neutral,
            timeframes=[item.as_dict() for item in analyses],
        )


def generate_multi_timeframe_report(
    symbol: str = "NIFTY50",
    lookback_days: int = 30,
    pipeline_csv: Path | str | None = DEFAULT_PIPELINE_CSV,
    report_path: Path | str | None = DEFAULT_REPORT_PATH,
) -> MultiTimeframeReport:
    """Run multi-timeframe analysis and export the JSON report."""
    engine = MultiTimeframeEngine(symbol=symbol, lookback_days=lookback_days)
    report = engine.analyze(
        pipeline_csv=Path(pipeline_csv) if pipeline_csv is not None else None,
    )

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Multi-timeframe analysis completed: overall=%s alignment=%s",
        report.overall_bias,
        report.alignment_score,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_multi_timeframe_report()
        print("Multi-Timeframe Analysis Summary")
        print(f"Overall Bias: {report.overall_bias}")
        print(f"Alignment Score: {report.alignment_score}")
        print(f"Bullish Timeframes: {report.bullish_timeframes}")
        print(f"Bearish Timeframes: {report.bearish_timeframes}")
        print(f"Neutral Timeframes: {report.neutral_timeframes}")
        print("Per Timeframe Bias:")
        for item in report.timeframes:
            print(
                f"  - {item['timeframe']}: trend={item['trend']} | "
                f"structure={item['structure']} | institutional={item['institutional_bias']}"
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except MultiTimeframeEngineError as exc:
        logger.error("Multi-timeframe engine error: %s", exc)
        print(f"Multi-timeframe engine error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected multi-timeframe engine failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
