"""
Signal funnel diagnostics for SmartMoneyEngine.

Tracks per-candle progression through SMC filter stages to identify
where actionable signals are lost. Does not modify strategy logic.
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
from src.pipeline.market_pipeline import MarketPipelineRunner, prepare_market_dataframe
from src.signals.decision_engine import DecisionEngine, TradeDecision
from src.smc.market_structure import MarketStructure
from src.smc.swing_detector import SwingDetector
from src.smc.trend_engine import TrendEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "signal_funnel_report.json"

STAGE_ORDER: tuple[str, ...] = (
    "trend_qualified",
    "bos_qualified",
    "choch_qualified",
    "fvg_qualified",
    "liquidity_qualified",
    "htf_aligned",
    "decision_buy_sell",
)

STAGE_LABELS: dict[str, str] = {
    "trend_qualified": "Trend Qualified",
    "bos_qualified": "BOS Qualified",
    "choch_qualified": "CHOCH Qualified",
    "fvg_qualified": "FVG Qualified",
    "liquidity_qualified": "Liquidity Qualified",
    "htf_aligned": "HTF Aligned",
    "decision_buy_sell": "Decision Engine BUY/SELL",
}


class SignalFunnelError(Exception):
    """Raised when signal funnel analysis fails."""


class TrendPath(str, Enum):
    """Directional path used for stage qualification."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class StageMetrics:
    """Funnel metrics for one filter stage."""

    stage: str
    label: str
    pass_count: int
    rejection_count: int
    pass_pct: float
    rejection_pct: float
    drop_from_previous_pct: float
    independent_pass_count: int
    independent_pass_pct: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SignalFunnelReport:
    """Aggregate signal funnel diagnostic report."""

    symbol: str
    timeframe: str
    source_csv: str
    start_date: str | None
    end_date: str | None
    total_candles: int
    final_signals: int
    buy_signals: int
    sell_signals: int
    cumulative_final: int
    overall_conversion_pct: float
    stages: list[dict[str, Any]] = field(default_factory=list)
    top_bottlenecks: list[dict[str, Any]] = field(default_factory=list)
    most_restrictive_filters: list[dict[str, Any]] = field(default_factory=list)
    signal_loss_pct_by_stage: dict[str, float] = field(default_factory=dict)
    execution_time_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignalFunnelAnalyzer:
    """
    Diagnose signal attrition across SMC filter stages.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Storage timeframe label such as ``5``.
    lookback_days : int, optional
        Calendar days of history when running the pipeline automatically.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        lookback_days: int = 365,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.decision_engine = DecisionEngine(symbol=symbol, timeframe=timeframe)

    @staticmethod
    def _is_active(value: Any) -> bool:
        """Mirror DecisionEngine active-signal detection."""
        return DecisionEngine._is_active(value)

    @staticmethod
    def _normalize_trend(value: Any) -> str:
        trend = str(value).strip().upper()
        if trend in {"BULLISH", "BEARISH", "SIDEWAYS"}:
            return trend
        return "SIDEWAYS"

    @staticmethod
    def _trend_path(trend: str) -> TrendPath:
        if trend == "BULLISH":
            return TrendPath.BULLISH
        if trend == "BEARISH":
            return TrendPath.BEARISH
        return TrendPath.NEUTRAL

    @staticmethod
    def _htf_trend_label(value: Any) -> str:
        trend = str(value).strip().upper()
        if trend == "BULLISH":
            return "Bullish"
        if trend == "BEARISH":
            return "Bearish"
        return "Neutral"

    @staticmethod
    def _normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Convert pipeline OHLCV into a resample-ready timestamp frame."""
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise SignalFunnelError(f"Pipeline frame missing OHLCV columns: {sorted(missing)}")

        working = frame.loc[:, ["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        working["timestamp"] = pd.to_datetime(working["Date"], errors="coerce")
        if working["timestamp"].dt.tz is None:
            working["timestamp"] = working["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            working["timestamp"] = working["timestamp"].dt.tz_convert("Asia/Kolkata")

        return working.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        indexed = frame.set_index("timestamp")
        resampled = (
            indexed.resample(rule)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna(subset=["Open", "High", "Low", "Close"])
            .reset_index()
        )
        resampled["Date"] = resampled["timestamp"].astype(str)
        return resampled

    def _trend_on_resampled(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Run minimal SMC stack to derive HTF trend labels."""
        if len(frame) < 20:
            empty = frame[["timestamp"]].copy()
            empty["Trend"] = "SIDEWAYS"
            return empty

        input_frame = frame[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
        input_frame = input_frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        prepared = prepare_market_dataframe(input_frame)
        market = MarketData(prepared.copy())
        SwingDetector().detect(market)
        MarketStructure().detect(market)
        TrendEngine().detect(market)

        result = market.data[["Date", "Trend"]].copy()
        result = result.rename(columns={"Date": "timestamp"})
        result["timestamp"] = self._ensure_ist(result["timestamp"])
        return result[["timestamp", "Trend"]].sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _ensure_ist(series: pd.Series) -> pd.Series:
        """Normalize timestamps to Asia/Kolkata for stable as-of merges."""
        timestamps = pd.to_datetime(series, errors="coerce")
        if timestamps.dt.tz is None:
            return timestamps.dt.tz_localize("Asia/Kolkata")
        return timestamps.dt.tz_convert("Asia/Kolkata")

    def _build_htf_lookup(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach per-candle 1D and 4H trend context via as-of merge."""
        ohlcv = self._normalize_ohlcv_frame(frame)
        base = ohlcv[["timestamp"]].copy()
        base["timestamp"] = self._ensure_ist(base["timestamp"])

        for rule, column in (("4h", "HTF_4H_Trend"), ("1D", "HTF_1D_Trend")):
            resampled = self._resample_ohlcv(ohlcv, rule)
            trend_frame = self._trend_on_resampled(resampled)
            trend_frame["timestamp"] = self._ensure_ist(trend_frame["timestamp"])
            merged = pd.merge_asof(
                base.sort_values("timestamp"),
                trend_frame.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            base[column] = merged["Trend"].fillna("SIDEWAYS")

        return base

    def _stage_flags(
        self,
        row: pd.Series,
        path: TrendPath,
        htf_1d: str,
        htf_4h: str,
        decision: str,
    ) -> dict[str, bool]:
        """Evaluate all funnel stages for one candle."""
        trend_ok = path != TrendPath.NEUTRAL

        if path == TrendPath.BULLISH:
            bos_ok = self._is_active(row.get("Bullish_BOS"))
            choch_ok = self._is_active(row.get("Bullish_CHOCH"))
            fvg_ok = self._is_active(row.get("Bullish_FVG_Top"))
            liquidity_ok = self._is_active(row.get("Sell_Liquidity_Sweep"))
            htf_values = [self._htf_trend_label(htf_1d), self._htf_trend_label(htf_4h)]
            htf_aligned = sum(1 for value in htf_values if value == "Bullish") >= 1
            htf_aligned = htf_aligned and not any(value == "Bearish" for value in htf_values)
        elif path == TrendPath.BEARISH:
            bos_ok = self._is_active(row.get("Bearish_BOS"))
            choch_ok = self._is_active(row.get("Bearish_CHOCH"))
            fvg_ok = self._is_active(row.get("Bearish_FVG_Top"))
            liquidity_ok = self._is_active(row.get("Buy_Liquidity_Sweep"))
            htf_values = [self._htf_trend_label(htf_1d), self._htf_trend_label(htf_4h)]
            htf_aligned = sum(1 for value in htf_values if value == "Bearish") >= 1
            htf_aligned = htf_aligned and not any(value == "Bullish" for value in htf_values)
        else:
            bos_ok = choch_ok = fvg_ok = liquidity_ok = htf_aligned = False

        decision_ok = decision in {TradeDecision.BUY.value, TradeDecision.SELL.value}

        cumulative = {
            "trend_qualified": trend_ok,
            "bos_qualified": trend_ok and bos_ok,
            "choch_qualified": trend_ok and bos_ok and choch_ok,
            "fvg_qualified": trend_ok and bos_ok and choch_ok and fvg_ok,
            "liquidity_qualified": trend_ok and bos_ok and choch_ok and fvg_ok and liquidity_ok,
            "htf_aligned": trend_ok and bos_ok and choch_ok and fvg_ok and liquidity_ok and htf_aligned,
            "decision_buy_sell": (
                trend_ok
                and bos_ok
                and choch_ok
                and fvg_ok
                and liquidity_ok
                and htf_aligned
                and decision_ok
            ),
        }
        independent = {
            "trend_qualified": trend_ok,
            "bos_qualified": bos_ok,
            "choch_qualified": choch_ok,
            "fvg_qualified": fvg_ok,
            "liquidity_qualified": liquidity_ok,
            "htf_aligned": htf_aligned,
            "decision_buy_sell": decision_ok,
        }
        return {"cumulative": cumulative, "independent": independent}

    @staticmethod
    def _pct(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return round((numerator / denominator) * 100.0, 2)

    def _build_stage_metrics(
        self,
        cumulative_counts: dict[str, int],
        independent_counts: dict[str, int],
        total_candles: int,
    ) -> list[StageMetrics]:
        """Convert raw pass counts into funnel stage metrics."""
        metrics: list[StageMetrics] = []
        previous_pass = total_candles

        for stage in STAGE_ORDER:
            pass_count = cumulative_counts[stage]
            rejection_count = previous_pass - pass_count
            drop_pct = self._pct(rejection_count, previous_pass) if previous_pass else 0.0
            metrics.append(
                StageMetrics(
                    stage=stage,
                    label=STAGE_LABELS[stage],
                    pass_count=pass_count,
                    rejection_count=rejection_count,
                    pass_pct=self._pct(pass_count, total_candles),
                    rejection_pct=self._pct(rejection_count, total_candles),
                    drop_from_previous_pct=drop_pct,
                    independent_pass_count=independent_counts[stage],
                    independent_pass_pct=self._pct(independent_counts[stage], total_candles),
                )
            )
            previous_pass = pass_count

        return metrics

    @staticmethod
    def _identify_bottlenecks(stage_metrics: list[StageMetrics]) -> list[dict[str, Any]]:
        """Return stages with the largest sequential drop-offs."""
        ranked = sorted(
            stage_metrics,
            key=lambda item: (item.drop_from_previous_pct, item.rejection_count),
            reverse=True,
        )
        return [
            {
                "stage": item.stage,
                "label": item.label,
                "drop_from_previous_pct": item.drop_from_previous_pct,
                "rejection_count": item.rejection_count,
                "remaining_pass_count": item.pass_count,
            }
            for item in ranked[:3]
        ]

    @staticmethod
    def _identify_restrictive_filters(stage_metrics: list[StageMetrics]) -> list[dict[str, Any]]:
        """Return stages with the lowest independent pass rates."""
        ranked = sorted(stage_metrics, key=lambda item: item.independent_pass_count)
        return [
            {
                "stage": item.stage,
                "label": item.label,
                "independent_pass_count": item.independent_pass_count,
                "independent_pass_pct": item.independent_pass_pct,
            }
            for item in ranked[:3]
        ]

    def analyze(self, frame: pd.DataFrame, source_csv: str = "") -> SignalFunnelReport:
        """Run funnel diagnostics on an evaluated or raw pipeline frame."""
        started = time.perf_counter()
        if frame.empty:
            raise SignalFunnelError("Pipeline frame is empty.")

        evaluated = (
            frame
            if "Decision" in frame.columns
            else self.decision_engine.evaluate(frame.copy())
        )
        self.decision_engine._validate_frame(evaluated)

        htf_lookup = self._build_htf_lookup(evaluated)
        working = evaluated.copy()
        working["timestamp"] = self._ensure_ist(working["Date"])
        htf_lookup["timestamp"] = self._ensure_ist(htf_lookup["timestamp"])

        merged = pd.merge_asof(
            working.sort_values("timestamp").reset_index(drop=True),
            htf_lookup.sort_values("timestamp").reset_index(drop=True),
            on="timestamp",
            direction="backward",
        )
        merged["HTF_1D_Trend"] = merged["HTF_1D_Trend"].fillna("SIDEWAYS")
        merged["HTF_4H_Trend"] = merged["HTF_4H_Trend"].fillna("SIDEWAYS")

        cumulative_counts = {stage: 0 for stage in STAGE_ORDER}
        independent_counts = {stage: 0 for stage in STAGE_ORDER}

        for _, row in merged.iterrows():
            path = self._trend_path(self._normalize_trend(row.get("Trend")))
            flags = self._stage_flags(
                row=row,
                path=path,
                htf_1d=str(row.get("HTF_1D_Trend", "SIDEWAYS")),
                htf_4h=str(row.get("HTF_4H_Trend", "SIDEWAYS")),
                decision=str(row.get("Decision", TradeDecision.WAIT.value)),
            )
            for stage in STAGE_ORDER:
                if flags["cumulative"][stage]:
                    cumulative_counts[stage] += 1
                if flags["independent"][stage]:
                    independent_counts[stage] += 1

        total_candles = len(evaluated)
        stage_metrics = self._build_stage_metrics(
            cumulative_counts,
            independent_counts,
            total_candles,
        )
        buy_signals = int((evaluated["Decision"] == TradeDecision.BUY.value).sum())
        sell_signals = int((evaluated["Decision"] == TradeDecision.SELL.value).sum())
        final_signals = buy_signals + sell_signals
        cumulative_final = cumulative_counts["decision_buy_sell"]

        signal_loss_pct = {
            item.stage: round(100.0 - item.pass_pct, 2)
            for item in stage_metrics
        }

        start_date = str(evaluated["Date"].iloc[0]) if total_candles else None
        end_date = str(evaluated["Date"].iloc[-1]) if total_candles else None

        return SignalFunnelReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=source_csv,
            start_date=start_date,
            end_date=end_date,
            total_candles=total_candles,
            final_signals=final_signals,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            cumulative_final=cumulative_final,
            overall_conversion_pct=self._pct(final_signals, total_candles),
            stages=[item.as_dict() for item in stage_metrics],
            top_bottlenecks=self._identify_bottlenecks(stage_metrics),
            most_restrictive_filters=self._identify_restrictive_filters(stage_metrics),
            signal_loss_pct_by_stage=signal_loss_pct,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )

    def run_from_csv(self, pipeline_csv: Path | str) -> SignalFunnelReport:
        """Load a pipeline CSV and analyze it."""
        csv_path = Path(pipeline_csv)
        frame = self.decision_engine.load_pipeline_csv(csv_path)
        return self.analyze(frame, source_csv=str(csv_path))

    def run_from_pipeline(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        output_csv: Path | str | None = None,
    ) -> SignalFunnelReport:
        """Execute the market pipeline, then analyze the exported CSV."""
        end = end_date if end_date is not None else date.today()
        start = start_date if start_date is not None else end - timedelta(days=self.lookback_days)
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
            raise SignalFunnelError(report.failure_message or "Market pipeline failed.")

        return self.run_from_csv(report.output_csv)


def generate_signal_funnel_report(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
    lookback_days: int = 365,
    run_pipeline: bool = False,
) -> SignalFunnelReport:
    """Run signal funnel analysis and export JSON report."""
    analyzer = SignalFunnelAnalyzer(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
    )

    if run_pipeline:
        funnel_report = analyzer.run_from_pipeline()
    else:
        csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
        funnel_report = analyzer.run_from_csv(csv_path)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(funnel_report.as_dict(), handle, indent=2)

    logger.info(
        "Signal funnel analysis completed: candles=%s final_signals=%s",
        funnel_report.total_candles,
        funnel_report.final_signals,
    )
    return funnel_report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_signal_funnel_report()
        print("Signal Funnel Analysis Summary")
        print(f"Symbol: {report.symbol} | Timeframe: {report.timeframe}")
        print(f"Candles: {report.total_candles}")
        print(f"Final Signals: {report.final_signals} (BUY={report.buy_signals}, SELL={report.sell_signals})")
        print(f"Overall Conversion: {report.overall_conversion_pct}%")
        print("Stage Funnel:")
        for stage in report.stages:
            print(
                f"  - {stage['label']}: pass={stage['pass_count']} "
                f"reject={stage['rejection_count']} ({stage['rejection_pct']}%) "
                f"drop={stage['drop_from_previous_pct']}%"
            )
        print("Top Bottlenecks:")
        for bottleneck in report.top_bottlenecks:
            print(
                f"  - {bottleneck['label']}: drop={bottleneck['drop_from_previous_pct']}% "
                f"reject={bottleneck['rejection_count']}"
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except SignalFunnelError as exc:
        logger.error("Signal funnel analysis error: %s", exc)
        print(f"Signal funnel analysis error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected signal funnel analysis failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
