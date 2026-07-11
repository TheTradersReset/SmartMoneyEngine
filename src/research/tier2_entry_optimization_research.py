"""
Tier-2 entry optimization research for SmartMoneyEngine.

Evaluates alternative entry timings for Tier-2 institutional signals using
existing SMC structure only. Research-only; no production logic or entries.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine
from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.tiered_signal_framework_research import (
    FORWARD_BARS,
    MIN_RISK_POINTS,
    RISK_LOOKBACK,
    SL_BUFFER_POINTS,
    TierSignal,
    TieredSignalFrameworkResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tier2_entry_optimization.json"

ENTRY_SCAN_BARS = 40
MOVE_MISS_THRESHOLD = 50
MIN_METHOD_TRADES = 20
ENTRY_TOLERANCE = 0.05

ENTRY_METHODS = {
    "A_bos_close": "BOS Close Entry",
    "B_first_fvg_retest": "First FVG Retest",
    "C_fvg_50_percent": "50% FVG Entry",
    "D_order_block_retest": "Order Block Retest",
    "E_liquidity_retest": "Liquidity Re-test Entry",
}


class Tier2EntryOptimizationError(Exception):
    """Raised when Tier-2 entry optimization research fails."""


@dataclass(frozen=True)
class EntryTrigger:
    """Resolved research entry for one method and signal."""

    triggered: bool
    entry_bar: int | None
    entry_price: float | None
    missed_move: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntryMethodOutcome:
    """Simulated outcome for one entry method on one Tier-2 signal."""

    method_key: str
    method_label: str
    timeframe: str
    direction: str
    bos_timestamp: str
    entry_triggered: bool
    entry_bar: int | None
    entry_price: float | None
    risk_points: float
    mfe_points: float
    mae_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    missed_move: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntryMethodMetrics:
    """Aggregate metrics for one entry method."""

    method_key: str
    method_label: str
    trades: int
    entries_triggered: int
    entries_missed_move: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    maximum_drawdown_points: float
    average_mae: float
    average_mfe: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2EntryOptimizationReport:
    """Full Tier-2 entry optimization research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    total_signals: int
    entry_methods: dict[str, str]
    method_metrics: dict[str, dict[str, Any]]
    best_entry_model: str
    best_rr_model: str
    best_accuracy_model: str
    recommendations: dict[str, str]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tier2EntryOptimizationResearch:
    """Compare Tier-2 entry timing methods across historical signals."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.tier_engine = TieredSignalFrameworkResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        return LiquidityNarrativeEngine._to_float(value)

    @staticmethod
    def _is_active(value: Any) -> bool:
        from src.signals.decision_engine import DecisionEngine

        return DecisionEngine._is_active(value)

    def _fvg_bounds(
        self,
        frame: pd.DataFrame,
        bos_bar: int,
        direction: str,
    ) -> tuple[float, float] | None:
        if direction == "bullish":
            top = self._to_float(frame.iloc[bos_bar].get("Bullish_FVG_Top"))
            bottom = self._to_float(frame.iloc[bos_bar].get("Bullish_FVG_Bottom"))
        else:
            top = self._to_float(frame.iloc[bos_bar].get("Bearish_FVG_Top"))
            bottom = self._to_float(frame.iloc[bos_bar].get("Bearish_FVG_Bottom"))

        if top is None or bottom is None:
            window = frame.iloc[max(0, bos_bar - 40) : bos_bar + 1]
            for offset in range(len(window) - 1, -1, -1):
                row = window.iloc[offset]
                if direction == "bullish":
                    top = self._to_float(row.get("Bullish_FVG_Top"))
                    bottom = self._to_float(row.get("Bullish_FVG_Bottom"))
                else:
                    top = self._to_float(row.get("Bearish_FVG_Top"))
                    bottom = self._to_float(row.get("Bearish_FVG_Bottom"))
                if top is not None and bottom is not None:
                    break

        if top is None or bottom is None:
            return None
        return bottom, top

    def _ob_bounds(
        self,
        frame: pd.DataFrame,
        index: int,
        direction: str,
    ) -> tuple[float, float] | None:
        row = frame.iloc[index]
        if direction == "bullish":
            if self._is_active(row.get("Bullish_OB_Mitigated")):
                return None
            high = self._to_float(row.get("Bullish_OB_High"))
            low = self._to_float(row.get("Bullish_OB_Low"))
        else:
            if self._is_active(row.get("Bearish_OB_Mitigated")):
                return None
            high = self._to_float(row.get("Bearish_OB_High"))
            low = self._to_float(row.get("Bearish_OB_Low"))

        if high is None or low is None:
            return None
        return low, high

    def _liquidity_level(
        self,
        frame: pd.DataFrame,
        bos_bar: int,
        direction: str,
    ) -> float | None:
        window = frame.iloc[max(0, bos_bar - 40) : bos_bar + 1]
        if direction == "bullish":
            for offset in range(len(window) - 1, -1, -1):
                value = self._to_float(window.iloc[offset].get("Sell_Side_Liquidity"))
                if value is not None:
                    return value
            return self._to_float(frame.iloc[bos_bar].get("Sell_Side_Liquidity"))
        for offset in range(len(window) - 1, -1, -1):
            value = self._to_float(window.iloc[offset].get("Buy_Side_Liquidity"))
            if value is not None:
                return value
        return self._to_float(frame.iloc[bos_bar].get("Buy_Side_Liquidity"))

    def _bos_reference_move(self, frame: pd.DataFrame, bos_bar: int, direction: str) -> float:
        if bos_bar >= len(frame) - 1:
            return 0.0
        origin = float(frame.iloc[bos_bar]["Close"])
        end = min(len(frame) - 1, bos_bar + FORWARD_BARS)
        if direction == "bullish":
            return round(float(frame.iloc[bos_bar + 1 : end + 1]["High"].max()) - origin, 2)
        return round(origin - float(frame.iloc[bos_bar + 1 : end + 1]["Low"].min()), 2)

    def _move_before_bar(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        target_bar: int,
        direction: str,
    ) -> float:
        if target_bar <= start_bar:
            return 0.0
        origin = float(frame.iloc[start_bar]["Close"])
        if direction == "bullish":
            return round(
                float(frame.iloc[start_bar + 1 : target_bar + 1]["High"].max()) - origin,
                2,
            )
        return round(
            origin - float(frame.iloc[start_bar + 1 : target_bar + 1]["Low"].min()),
            2,
        )

    def _resolve_entry(
        self,
        method_key: str,
        frame: pd.DataFrame,
        signal: TierSignal,
    ) -> EntryTrigger:
        bos_bar = signal.bos_bar
        direction = signal.direction
        bos_move = self._bos_reference_move(frame, bos_bar, direction)

        if method_key == "A_bos_close":
            price = round(float(frame.iloc[bos_bar]["Close"]), 2)
            return EntryTrigger(triggered=True, entry_bar=bos_bar, entry_price=price, missed_move=False)

        end = min(len(frame) - 1, bos_bar + ENTRY_SCAN_BARS)
        fvg = self._fvg_bounds(frame, bos_bar, direction)

        if method_key == "B_first_fvg_retest":
            if fvg is None:
                return EntryTrigger(
                    triggered=False,
                    entry_bar=None,
                    entry_price=None,
                    missed_move=bos_move >= MOVE_MISS_THRESHOLD,
                )
            bottom, top = fvg
            for index in range(bos_bar + 1, end + 1):
                row = frame.iloc[index]
                low = float(row["Low"])
                high = float(row["High"])
                close = float(row["Close"])
                if direction == "bullish":
                    if bottom <= low <= top and close >= bottom:
                        return EntryTrigger(
                            triggered=True,
                            entry_bar=index,
                            entry_price=round(close, 2),
                            missed_move=self._move_before_bar(frame, bos_bar, index, direction)
                            >= MOVE_MISS_THRESHOLD,
                        )
                elif bottom <= high <= top and close <= top:
                    return EntryTrigger(
                        triggered=True,
                        entry_bar=index,
                        entry_price=round(close, 2),
                        missed_move=self._move_before_bar(frame, bos_bar, index, direction)
                        >= MOVE_MISS_THRESHOLD,
                    )
            return EntryTrigger(
                triggered=False,
                entry_bar=None,
                entry_price=None,
                missed_move=bos_move >= MOVE_MISS_THRESHOLD,
            )

        if method_key == "C_fvg_50_percent":
            if fvg is None:
                return EntryTrigger(
                    triggered=False,
                    entry_bar=None,
                    entry_price=None,
                    missed_move=bos_move >= MOVE_MISS_THRESHOLD,
                )
            bottom, top = fvg
            midpoint = round((bottom + top) / 2.0, 2)
            for index in range(bos_bar + 1, end + 1):
                row = frame.iloc[index]
                low = float(row["Low"])
                high = float(row["High"])
                if low - ENTRY_TOLERANCE <= midpoint <= high + ENTRY_TOLERANCE:
                    return EntryTrigger(
                        triggered=True,
                        entry_bar=index,
                        entry_price=midpoint,
                        missed_move=self._move_before_bar(frame, bos_bar, index, direction)
                        >= MOVE_MISS_THRESHOLD,
                    )
            return EntryTrigger(
                triggered=False,
                entry_bar=None,
                entry_price=None,
                missed_move=bos_move >= MOVE_MISS_THRESHOLD,
            )

        if method_key == "D_order_block_retest":
            for index in range(bos_bar + 1, end + 1):
                row = frame.iloc[index]
                bounds = self._ob_bounds(frame, index, direction)
                if bounds is None:
                    continue
                ob_low, ob_high = bounds
                low = float(row["Low"])
                high = float(row["High"])
                close = float(row["Close"])
                if direction == "bullish":
                    if ob_low <= low <= ob_high and close >= ob_low:
                        entry_price = round((ob_low + ob_high) / 2.0, 2)
                        return EntryTrigger(
                            triggered=True,
                            entry_bar=index,
                            entry_price=entry_price,
                            missed_move=self._move_before_bar(frame, bos_bar, index, direction)
                            >= MOVE_MISS_THRESHOLD,
                        )
                elif ob_low <= high <= ob_high and close <= ob_high:
                    entry_price = round((ob_low + ob_high) / 2.0, 2)
                    return EntryTrigger(
                        triggered=True,
                        entry_bar=index,
                        entry_price=entry_price,
                        missed_move=self._move_before_bar(frame, bos_bar, index, direction)
                        >= MOVE_MISS_THRESHOLD,
                    )
            return EntryTrigger(
                triggered=False,
                entry_bar=None,
                entry_price=None,
                missed_move=bos_move >= MOVE_MISS_THRESHOLD,
            )

        if method_key == "E_liquidity_retest":
            level = self._liquidity_level(frame, bos_bar, direction)
            if level is None:
                return EntryTrigger(
                    triggered=False,
                    entry_bar=None,
                    entry_price=None,
                    missed_move=bos_move >= MOVE_MISS_THRESHOLD,
                )
            for index in range(bos_bar + 1, end + 1):
                row = frame.iloc[index]
                low = float(row["Low"])
                high = float(row["High"])
                close = float(row["Close"])
                if direction == "bullish":
                    if low <= level + ENTRY_TOLERANCE and close >= level - ENTRY_TOLERANCE:
                        return EntryTrigger(
                            triggered=True,
                            entry_bar=index,
                            entry_price=round(close, 2),
                            missed_move=self._move_before_bar(frame, bos_bar, index, direction)
                            >= MOVE_MISS_THRESHOLD,
                        )
                elif high >= level - ENTRY_TOLERANCE and close <= level + ENTRY_TOLERANCE:
                    return EntryTrigger(
                        triggered=True,
                        entry_bar=index,
                        entry_price=round(close, 2),
                        missed_move=self._move_before_bar(frame, bos_bar, index, direction)
                        >= MOVE_MISS_THRESHOLD,
                    )
            return EntryTrigger(
                triggered=False,
                entry_bar=None,
                entry_price=None,
                missed_move=bos_move >= MOVE_MISS_THRESHOLD,
            )

        raise Tier2EntryOptimizationError(f"Unknown entry method: {method_key}")

    def _risk_at_bar(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
    ) -> float:
        lookback = frame.iloc[max(0, entry_bar - RISK_LOOKBACK) : entry_bar + 1]
        if direction == "bullish":
            anchor = float(lookback["Low"].min())
            stop = anchor - SL_BUFFER_POINTS
            return max(entry_price - stop, MIN_RISK_POINTS)
        anchor = float(lookback["High"].max())
        stop = anchor + SL_BUFFER_POINTS
        return max(stop - entry_price, MIN_RISK_POINTS)

    def _simulate_from_entry(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
    ) -> tuple[float, float, float, float, float, bool]:
        risk = round(self._risk_at_bar(frame, entry_bar, entry_price, direction), 2)
        if direction == "bullish":
            stop = entry_price - risk
        else:
            stop = entry_price + risk

        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
        mfe = 0.0
        mae = 0.0
        stopped = False

        for index in range(entry_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])

            if direction == "bullish":
                if bar_low <= stop:
                    stopped = True
                    break
                mfe = max(mfe, bar_high - entry_price)
                mae = max(mae, entry_price - bar_low)
            else:
                if bar_high >= stop:
                    stopped = True
                    break
                mfe = max(mfe, entry_price - bar_low)
                mae = max(mae, bar_high - entry_price)

        mfe = round(max(mfe, 0.0), 2)
        mae = round(max(mae, 0.0), 2)

        if stopped:
            pnl = -risk
            rr = -1.0
            win = False
        else:
            pnl = mfe
            rr = round(mfe / risk, 2) if risk > 0 else 0.0
            win = mfe >= risk

        return risk, mfe, mae, pnl, rr, win

    def evaluate_method(
        self,
        method_key: str,
        frame: pd.DataFrame,
        signal: TierSignal,
    ) -> EntryMethodOutcome:
        trigger = self._resolve_entry(method_key, frame, signal)
        label = ENTRY_METHODS[method_key]

        if not trigger.triggered or trigger.entry_bar is None or trigger.entry_price is None:
            return EntryMethodOutcome(
                method_key=method_key,
                method_label=label,
                timeframe=signal.timeframe,
                direction=signal.direction,
                bos_timestamp=signal.bos_timestamp,
                entry_triggered=False,
                entry_bar=None,
                entry_price=None,
                risk_points=0.0,
                mfe_points=0.0,
                mae_points=0.0,
                realized_pnl_points=0.0,
                realized_rr=0.0,
                win=False,
                missed_move=trigger.missed_move,
            )

        risk, mfe, mae, pnl, rr, win = self._simulate_from_entry(
            frame,
            trigger.entry_bar,
            trigger.entry_price,
            signal.direction,
        )

        return EntryMethodOutcome(
            method_key=method_key,
            method_label=label,
            timeframe=signal.timeframe,
            direction=signal.direction,
            bos_timestamp=signal.bos_timestamp,
            entry_triggered=True,
            entry_bar=trigger.entry_bar,
            entry_price=trigger.entry_price,
            risk_points=risk,
            mfe_points=mfe,
            mae_points=mae,
            realized_pnl_points=round(pnl, 2),
            realized_rr=rr,
            win=win,
            missed_move=trigger.missed_move,
        )

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    def _method_metrics(
        self,
        method_key: str,
        outcomes: list[EntryMethodOutcome],
    ) -> EntryMethodMetrics:
        label = ENTRY_METHODS[method_key]
        triggered = [item for item in outcomes if item.entry_triggered]

        if not triggered:
            return EntryMethodMetrics(
                method_key=method_key,
                method_label=label,
                trades=0,
                entries_triggered=0,
                entries_missed_move=sum(1 for item in outcomes if item.missed_move),
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                maximum_drawdown_points=0.0,
                average_mae=0.0,
                average_mfe=0.0,
            )

        pnls = [item.realized_pnl_points for item in triggered]
        rrs = [item.realized_rr for item in triggered]
        wins = sum(1 for item in triggered if item.win)

        return EntryMethodMetrics(
            method_key=method_key,
            method_label=label,
            trades=len(triggered),
            entries_triggered=len(triggered),
            entries_missed_move=sum(1 for item in outcomes if item.missed_move),
            win_rate_pct=round(wins / len(triggered) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            average_mae=round(mean(item.mae_points for item in triggered), 2),
            average_mfe=round(mean(item.mfe_points for item in triggered), 2),
        )

    def run(self, metadata: dict[str, Any]) -> Tier2EntryOptimizationReport:
        """Run Tier-2 entry optimization research."""
        started = time.perf_counter()

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

        all_outcomes: dict[str, list[EntryMethodOutcome]] = {
            key: [] for key in ENTRY_METHODS
        }
        signal_count = 0

        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            signals = self.tier_engine._detect_tier2(frame, timeframe_label)
            signal_count += len(signals)

            for signal in signals:
                for method_key in ENTRY_METHODS:
                    all_outcomes[method_key].append(
                        self.evaluate_method(method_key, frame, signal),
                    )

        method_metrics = {
            key: self._method_metrics(key, outcomes).as_dict()
            for key, outcomes in all_outcomes.items()
        }

        eligible = {
            key: metrics
            for key, metrics in method_metrics.items()
            if metrics["trades"] >= MIN_METHOD_TRADES
        }

        best_entry = max(
            eligible.items(),
            key=lambda item: (item[1]["expectancy"], item[1]["profit_factor"] or 0),
        )[0] if eligible else "A_bos_close"

        best_rr = max(
            eligible.items(),
            key=lambda item: (item[1]["average_rr"], item[1]["expectancy"]),
        )[0] if eligible else "A_bos_close"

        best_accuracy = max(
            eligible.items(),
            key=lambda item: (item[1]["win_rate_pct"], item[1]["expectancy"]),
        )[0] if eligible else "A_bos_close"

        recommendations = {
            "best_entry_model": ENTRY_METHODS[best_entry],
            "best_rr_model": ENTRY_METHODS[best_rr],
            "best_accuracy_model": ENTRY_METHODS[best_accuracy],
        }

        conclusions = [
            f"Evaluated {signal_count} Tier-2 signals across {len(ENTRY_METHODS)} entry methods.",
            (
                f"Best entry model (expectancy): {recommendations['best_entry_model']} "
                f"(Exp {method_metrics[best_entry]['expectancy']}, "
                f"n={method_metrics[best_entry]['trades']})."
            ),
            (
                f"Best RR model: {recommendations['best_rr_model']} "
                f"(Avg RR {method_metrics[best_rr]['average_rr']})."
            ),
            (
                f"Best accuracy model: {recommendations['best_accuracy_model']} "
                f"(WR {method_metrics[best_accuracy]['win_rate_pct']}%)."
            ),
        ]
        for key, metrics in method_metrics.items():
            conclusions.append(
                f"{metrics['method_label']}: trades={metrics['trades']} "
                f"missed_move={metrics['entries_missed_move']} "
                f"WR={metrics['win_rate_pct']}% Exp={metrics['expectancy']} "
                f"RR={metrics['average_rr']} DD={metrics['maximum_drawdown_points']}."
            )

        return Tier2EntryOptimizationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            total_signals=signal_count,
            entry_methods=ENTRY_METHODS,
            method_metrics=method_metrics,
            best_entry_model=best_entry,
            best_rr_model=best_rr,
            best_accuracy_model=best_accuracy,
            recommendations=recommendations,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tier2_entry_optimization_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Tier2EntryOptimizationReport:
    """Run Tier-2 entry optimization research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Tier2EntryOptimizationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = Tier2EntryOptimizationResearch(
        symbol=metadata.get("symbol", "NIFTY50"),
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", ("5M", "15M", "1H"))),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info("Tier-2 entry optimization completed: best=%s", report.best_entry_model)
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tier2_entry_optimization_report()
        print("Tier-2 Entry Optimization Research Summary")
        print(f"Signals: {report.total_signals}")
        for key, metrics in report.method_metrics.items():
            print(
                f"{metrics['method_label']}: trades={metrics['trades']} "
                f"missed={metrics['entries_missed_move']} WR={metrics['win_rate_pct']}% "
                f"Exp={metrics['expectancy']} RR={metrics['average_rr']}"
            )
        print(f"Best Entry: {report.recommendations['best_entry_model']}")
        print(f"Best RR: {report.recommendations['best_rr_model']}")
        print(f"Best Accuracy: {report.recommendations['best_accuracy_model']}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except Tier2EntryOptimizationError as exc:
        logger.error("Tier-2 entry optimization error: %s", exc)
        print(f"Tier-2 entry optimization error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Tier-2 entry optimization failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
