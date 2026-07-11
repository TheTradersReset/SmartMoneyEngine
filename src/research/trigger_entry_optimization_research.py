"""
Trigger Entry Optimization research for SmartMoneyEngine.

Compares eight entry timings for each institutional trigger model without
changing trigger detection. Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import DisplacementStrength, LiquidityNarrativeEngine
from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.trigger_trade_validation_research import (
    DEFAULT_TRIGGER_REPORT_PATH,
    MIN_MODEL_SAMPLES,
    TriggerTradeValidationResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "trigger_entry_optimization.json"

ENTRY_SCAN_BARS = 40
ENTRY_TOLERANCE = 0.05

ENTRY_METHODS: dict[str, str] = {
    "trigger_close": "Trigger Close",
    "trigger_high_low_break": "Trigger High/Low Break",
    "trigger_candle_50pct": "50% Trigger Candle Retrace",
    "confirmation_candle_close": "Confirmation Candle Close",
    "displacement_close": "Displacement Close",
    "choch_confirmation": "CHOCH Confirmation",
    "bos_confirmation": "BOS Confirmation",
    "first_fvg_retest": "First FVG Retest",
}


class TriggerEntryOptimizationError(Exception):
    """Raised when trigger entry optimization fails."""


@dataclass(frozen=True)
class EntryResolution:
    """Resolved entry bar and price for one method."""

    triggered: bool
    entry_bar: int | None
    entry_price: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TriggerEntryOutcome:
    """Simulated trade for one trigger record and entry method."""

    trigger_model: str
    direction: str
    entry_method: str
    entry_label: str
    symbol: str
    timeframe: str
    trigger_bar: int
    entry_bar: int | None
    entry_price: float | None
    risk_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    entry_triggered: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntryMethodMetrics:
    """Aggregate metrics for trigger model + entry method."""

    trigger_model: str
    direction: str
    entry_method: str
    entry_label: str
    trades: int
    entries_triggered: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BestEntrySelection:
    """Best entry method for one trigger model."""

    trigger_model: str
    direction: str
    best_entry_method: str
    best_entry_label: str
    trades: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriggerEntryOptimizationReport:
    """Full trigger entry optimization output."""

    source_report: str
    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    total_trigger_records: int
    entry_methods: dict[str, str]
    stop_model: str
    target_model: str
    trigger_entry_metrics: list[dict[str, Any]]
    best_entry_per_trigger: list[dict[str, Any]]
    overall_entry_metrics: list[dict[str, Any]]
    best_overall_entry: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TriggerEntryOptimizationResearch(TriggerTradeValidationResearch):
    """Compare entry timings for institutional trigger models."""

    def __init__(
        self,
        trigger_report_path: Path | str | None = None,
        research_days: int = RESEARCH_DAYS,
    ) -> None:
        super().__init__(trigger_report_path=trigger_report_path, research_days=research_days)

    def _load_trigger_report(self) -> dict[str, Any]:
        if not self.trigger_report_path.exists():
            raise TriggerEntryOptimizationError(
                f"Institutional trigger validation report not found: {self.trigger_report_path}",
            )
        with self.trigger_report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _metrics_from_pnls(
        pnls: list[float],
        rrs: list[float],
        wins: int,
        total: int,
    ) -> tuple[float, float | None, float, float, float, float]:
        pf = TriggerEntryOptimizationResearch._profit_factor(pnls)
        exp = round(mean(pnls), 2) if pnls else 0.0
        avg_rr = round(mean(rrs), 2) if rrs else 0.0
        net = round(sum(pnls), 2)
        max_dd = TriggerEntryOptimizationResearch._maximum_drawdown(pnls)
        wr = round(wins / total * 100, 2) if total else 0.0
        return wr, pf, exp, avg_rr, net, max_dd

    def _resolve_trigger_close(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> EntryResolution:
        del direction
        price = round(float(frame.iloc[trigger_bar]["Close"]), 2)
        return EntryResolution(triggered=True, entry_bar=trigger_bar, entry_price=price)

    def _resolve_high_low_break(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> EntryResolution:
        trigger_row = frame.iloc[trigger_bar]
        trigger_high = float(trigger_row["High"])
        trigger_low = float(trigger_row["Low"])
        end = min(len(frame) - 1, trigger_bar + ENTRY_SCAN_BARS)

        for index in range(trigger_bar + 1, end + 1):
            row = frame.iloc[index]
            high = float(row["High"])
            low = float(row["Low"])
            if direction == "bullish" and high > trigger_high:
                return EntryResolution(
                    triggered=True,
                    entry_bar=index,
                    entry_price=round(trigger_high, 2),
                )
            if direction == "bearish" and low < trigger_low:
                return EntryResolution(
                    triggered=True,
                    entry_bar=index,
                    entry_price=round(trigger_low, 2),
                )
        return EntryResolution(triggered=False, entry_bar=None, entry_price=None)

    def _resolve_candle_50pct(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> EntryResolution:
        del direction
        trigger_row = frame.iloc[trigger_bar]
        midpoint = round((float(trigger_row["High"]) + float(trigger_row["Low"])) / 2.0, 2)
        end = min(len(frame) - 1, trigger_bar + ENTRY_SCAN_BARS)

        for index in range(trigger_bar + 1, end + 1):
            row = frame.iloc[index]
            low = float(row["Low"])
            high = float(row["High"])
            if low - ENTRY_TOLERANCE <= midpoint <= high + ENTRY_TOLERANCE:
                return EntryResolution(
                    triggered=True,
                    entry_bar=index,
                    entry_price=midpoint,
                )
        return EntryResolution(triggered=False, entry_bar=None, entry_price=None)

    def _resolve_confirmation_close(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> EntryResolution:
        del direction
        entry_bar = trigger_bar + 1
        if entry_bar >= len(frame):
            return EntryResolution(triggered=False, entry_bar=None, entry_price=None)
        price = round(float(frame.iloc[entry_bar]["Close"]), 2)
        return EntryResolution(triggered=True, entry_bar=entry_bar, entry_price=price)

    def _resolve_displacement_close(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> EntryResolution:
        end = min(len(frame) - 1, trigger_bar + ENTRY_SCAN_BARS)
        for index in range(trigger_bar, end + 1):
            row = frame.iloc[index]
            strength = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)
            if strength in {DisplacementStrength.MEDIUM, DisplacementStrength.STRONG}:
                return EntryResolution(
                    triggered=True,
                    entry_bar=index,
                    entry_price=round(float(row["Close"]), 2),
                )
        return EntryResolution(triggered=False, entry_bar=None, entry_price=None)

    def _resolve_structure_bar(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
        column: str,
    ) -> EntryResolution:
        end = min(len(frame) - 1, trigger_bar + ENTRY_SCAN_BARS)
        for index in range(trigger_bar, end + 1):
            if self._is_active(frame.iloc[index].get(column)):
                row = frame.iloc[index]
                return EntryResolution(
                    triggered=True,
                    entry_bar=index,
                    entry_price=round(float(row["Close"]), 2),
                )
        return EntryResolution(triggered=False, entry_bar=None, entry_price=None)

    def _resolve_choch(self, frame: pd.DataFrame, trigger_bar: int, direction: str) -> EntryResolution:
        column = "Bullish_CHOCH" if direction == "bullish" else "Bearish_CHOCH"
        return self._resolve_structure_bar(frame, trigger_bar, direction, column)

    def _resolve_bos(self, frame: pd.DataFrame, trigger_bar: int, direction: str) -> EntryResolution:
        column = "Bullish_BOS" if direction == "bullish" else "Bearish_BOS"
        return self._resolve_structure_bar(frame, trigger_bar, direction, column)

    def _fvg_bounds_after(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        direction: str,
    ) -> tuple[int, float, float] | None:
        end = min(len(frame) - 1, start_bar + ENTRY_SCAN_BARS)
        for index in range(start_bar, end + 1):
            row = frame.iloc[index]
            if direction == "bullish":
                top = self._to_float(row.get("Bullish_FVG_Top"))
                bottom = self._to_float(row.get("Bullish_FVG_Bottom"))
            else:
                top = self._to_float(row.get("Bearish_FVG_Top"))
                bottom = self._to_float(row.get("Bearish_FVG_Bottom"))
            if top is not None and bottom is not None:
                return index, bottom, top
        return None

    def _resolve_first_fvg_retest(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> EntryResolution:
        fvg = self._fvg_bounds_after(frame, trigger_bar, direction)
        if fvg is None:
            return EntryResolution(triggered=False, entry_bar=None, entry_price=None)

        fvg_bar, bottom, top = fvg
        end = min(len(frame) - 1, fvg_bar + ENTRY_SCAN_BARS)
        for index in range(fvg_bar + 1, end + 1):
            row = frame.iloc[index]
            low = float(row["Low"])
            high = float(row["High"])
            close = float(row["Close"])
            if direction == "bullish":
                if bottom <= low <= top and close >= bottom:
                    return EntryResolution(
                        triggered=True,
                        entry_bar=index,
                        entry_price=round(close, 2),
                    )
            elif bottom <= high <= top and close <= top:
                return EntryResolution(
                    triggered=True,
                    entry_bar=index,
                    entry_price=round(close, 2),
                )
        return EntryResolution(triggered=False, entry_bar=None, entry_price=None)

    def _resolve_entry(
        self,
        method_key: str,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> EntryResolution:
        resolvers = {
            "trigger_close": self._resolve_trigger_close,
            "trigger_high_low_break": self._resolve_high_low_break,
            "trigger_candle_50pct": self._resolve_candle_50pct,
            "confirmation_candle_close": self._resolve_confirmation_close,
            "displacement_close": self._resolve_displacement_close,
            "choch_confirmation": self._resolve_choch,
            "bos_confirmation": self._resolve_bos,
            "first_fvg_retest": self._resolve_first_fvg_retest,
        }
        resolver = resolvers.get(method_key)
        if resolver is None:
            raise TriggerEntryOptimizationError(f"Unknown entry method: {method_key}")
        return resolver(frame, trigger_bar, direction)

    def _evaluate_entry(
        self,
        frame: pd.DataFrame,
        record: dict[str, Any],
        trigger_bar: int,
        method_key: str,
    ) -> TriggerEntryOutcome:
        direction = str(record["direction"])
        label = ENTRY_METHODS[method_key]
        resolution = self._resolve_entry(method_key, frame, trigger_bar, direction)

        if (
            not resolution.triggered
            or resolution.entry_bar is None
            or resolution.entry_price is None
            or resolution.entry_bar >= len(frame) - 1
        ):
            return TriggerEntryOutcome(
                trigger_model=str(record["trigger_model"]),
                direction=direction,
                entry_method=method_key,
                entry_label=label,
                symbol=str(record["symbol"]),
                timeframe=str(record["timeframe"]),
                trigger_bar=trigger_bar,
                entry_bar=None,
                entry_price=None,
                risk_points=0.0,
                realized_pnl_points=0.0,
                realized_rr=0.0,
                win=False,
                entry_triggered=False,
            )

        entry_price = resolution.entry_price
        entry_bar = resolution.entry_bar
        stop, risk = self._structural_stop(frame, entry_bar, entry_price, direction)
        target = self._opposite_liquidity_target(frame, entry_bar, entry_price, direction, risk)
        pnl, rr, win, _ = self._simulate_trade(
            frame,
            entry_bar,
            entry_price,
            direction,
            stop,
            target,
            risk,
        )

        return TriggerEntryOutcome(
            trigger_model=str(record["trigger_model"]),
            direction=direction,
            entry_method=method_key,
            entry_label=label,
            symbol=str(record["symbol"]),
            timeframe=str(record["timeframe"]),
            trigger_bar=trigger_bar,
            entry_bar=entry_bar,
            entry_price=entry_price,
            risk_points=risk,
            realized_pnl_points=pnl,
            realized_rr=rr,
            win=win,
            entry_triggered=True,
        )

    def _aggregate_trigger_entry_metrics(
        self,
        outcomes: list[TriggerEntryOutcome],
    ) -> list[EntryMethodMetrics]:
        groups: dict[tuple[str, str, str], list[TriggerEntryOutcome]] = defaultdict(list)
        for outcome in outcomes:
            if not outcome.entry_triggered:
                continue
            key = (outcome.trigger_model, outcome.direction, outcome.entry_method)
            groups[key].append(outcome)

        metrics: list[EntryMethodMetrics] = []
        for (model, direction, method_key), bucket in groups.items():
            if len(bucket) < MIN_MODEL_SAMPLES:
                continue
            pnls = [item.realized_pnl_points for item in bucket]
            rrs = [item.realized_rr for item in bucket]
            wins = sum(1 for item in bucket if item.win)
            total = len(bucket)
            wr, pf, exp, avg_rr, net, max_dd = self._metrics_from_pnls(pnls, rrs, wins, total)
            metrics.append(
                EntryMethodMetrics(
                    trigger_model=model,
                    direction=direction,
                    entry_method=method_key,
                    entry_label=ENTRY_METHODS[method_key],
                    trades=total,
                    entries_triggered=total,
                    win_rate_pct=wr,
                    profit_factor=pf,
                    expectancy=exp,
                    average_rr=avg_rr,
                    net_points=net,
                    maximum_drawdown_points=max_dd,
                ),
            )
        return metrics

    def _aggregate_overall_metrics(
        self,
        outcomes: list[TriggerEntryOutcome],
    ) -> list[EntryMethodMetrics]:
        groups: dict[str, list[TriggerEntryOutcome]] = defaultdict(list)
        for outcome in outcomes:
            if outcome.entry_triggered:
                groups[outcome.entry_method].append(outcome)

        metrics: list[EntryMethodMetrics] = []
        for method_key, bucket in groups.items():
            pnls = [item.realized_pnl_points for item in bucket]
            rrs = [item.realized_rr for item in bucket]
            wins = sum(1 for item in bucket if item.win)
            total = len(bucket)
            wr, pf, exp, avg_rr, net, max_dd = self._metrics_from_pnls(pnls, rrs, wins, total)
            metrics.append(
                EntryMethodMetrics(
                    trigger_model="__ALL__",
                    direction="__ALL__",
                    entry_method=method_key,
                    entry_label=ENTRY_METHODS[method_key],
                    trades=total,
                    entries_triggered=total,
                    win_rate_pct=wr,
                    profit_factor=pf,
                    expectancy=exp,
                    average_rr=avg_rr,
                    net_points=net,
                    maximum_drawdown_points=max_dd,
                ),
            )
        metrics.sort(key=lambda item: (item.expectancy, item.win_rate_pct), reverse=True)
        return metrics

    @staticmethod
    def _best_entry_per_trigger(metrics: list[EntryMethodMetrics]) -> list[BestEntrySelection]:
        groups: dict[tuple[str, str], list[EntryMethodMetrics]] = defaultdict(list)
        for item in metrics:
            groups[(item.trigger_model, item.direction)].append(item)

        selections: list[BestEntrySelection] = []
        for (model, direction), bucket in groups.items():
            best = max(
                bucket,
                key=lambda item: (item.expectancy, item.win_rate_pct, item.trades),
            )
            selections.append(
                BestEntrySelection(
                    trigger_model=model,
                    direction=direction,
                    best_entry_method=best.entry_method,
                    best_entry_label=best.entry_label,
                    trades=best.trades,
                    win_rate_pct=best.win_rate_pct,
                    profit_factor=best.profit_factor,
                    expectancy=best.expectancy,
                    average_rr=best.average_rr,
                    net_points=best.net_points,
                    maximum_drawdown_points=best.maximum_drawdown_points,
                ),
            )
        selections.sort(key=lambda item: item.expectancy, reverse=True)
        return selections

    def run(self, metadata: dict[str, Any] | None = None) -> TriggerEntryOptimizationReport:
        started = time.perf_counter()
        trigger_report = self._load_trigger_report()

        if metadata is None:
            metadata = {
                "start_date": trigger_report.get("start_date", ""),
                "end_date": trigger_report.get("end_date", ""),
                "research_window_days": trigger_report.get("research_window_days", self.research_days),
            }

        records = trigger_report.get("trigger_records", [])
        outcomes: list[TriggerEntryOutcome] = []

        for record in records:
            frame = self._load_frame(
                str(record["symbol"]),
                str(record["timeframe"]),
                metadata,
            )
            if frame is None:
                continue
            trigger_bar = self._resolve_bar(frame, record)
            if trigger_bar is None:
                continue

            for method_key in ENTRY_METHODS:
                outcomes.append(self._evaluate_entry(frame, record, trigger_bar, method_key))

        trigger_metrics = self._aggregate_trigger_entry_metrics(outcomes)
        overall_metrics = self._aggregate_overall_metrics(outcomes)
        best_per_trigger = self._best_entry_per_trigger(trigger_metrics)
        best_overall = overall_metrics[0].as_dict() if overall_metrics else {}

        triggered_count = sum(1 for item in outcomes if item.entry_triggered)
        conclusions = [
            f"Evaluated {len(ENTRY_METHODS)} entry methods on {len(records)} trigger records.",
            f"Entry resolutions triggered: {triggered_count} of {len(outcomes)}.",
            f"Trigger-model entry combinations with metrics: {len(trigger_metrics)}.",
            (
                f"Best overall entry: {best_overall.get('entry_label', 'N/A')} "
                f"(Exp {best_overall.get('expectancy', 0)}, "
                f"WR {best_overall.get('win_rate_pct', 0)}%, "
                f"n={best_overall.get('trades', 0)})"
                if best_overall
                else "No overall entry metrics available."
            ),
            (
                f"Best per-trigger leader: {best_per_trigger[0].best_entry_label} on "
                f"{best_per_trigger[0].trigger_model[:60]} (Exp {best_per_trigger[0].expectancy})"
                if best_per_trigger
                else "No per-trigger best entries identified."
            ),
        ]

        return TriggerEntryOptimizationReport(
            source_report=str(self.trigger_report_path),
            symbols_analyzed=trigger_report.get("symbols_analyzed", []),
            research_window_days=trigger_report.get("research_window_days", self.research_days),
            start_date=trigger_report.get("start_date", ""),
            end_date=trigger_report.get("end_date", ""),
            timeframes_analyzed=trigger_report.get("timeframes_analyzed", []),
            total_trigger_records=len(records),
            entry_methods=dict(ENTRY_METHODS),
            stop_model="structural_swing",
            target_model="opposite_liquidity",
            trigger_entry_metrics=[item.as_dict() for item in trigger_metrics],
            best_entry_per_trigger=[item.as_dict() for item in best_per_trigger],
            overall_entry_metrics=[item.as_dict() for item in overall_metrics],
            best_overall_entry=best_overall,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_trigger_entry_optimization_report(
    report_path: Path | str | None = None,
    trigger_report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> TriggerEntryOptimizationReport:
    """Run trigger entry optimization and export JSON."""
    metadata: dict[str, Any] | None = None
    filter_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if filter_path.exists():
        with filter_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    engine = TriggerEntryOptimizationResearch(trigger_report_path=trigger_report_path)
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Trigger entry optimization completed: combinations=%s overall_best=%s",
        len(report.trigger_entry_metrics),
        report.best_overall_entry.get("entry_method"),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_trigger_entry_optimization_report()
        print("Trigger Entry Optimization Research Summary")
        print(f"Trigger records: {report.total_trigger_records}")
        print(f"Entry methods: {len(report.entry_methods)}")
        print(f"Trigger entry combinations: {len(report.trigger_entry_metrics)}")
        if report.best_overall_entry:
            best = report.best_overall_entry
            print(
                f"Best overall: {best['entry_label']} "
                f"(Exp {best['expectancy']}, WR {best['win_rate_pct']}%, n={best['trades']})",
            )
        if report.best_entry_per_trigger:
            top = report.best_entry_per_trigger[0]
            print(
                f"Best per-trigger: {top['best_entry_label']} on "
                f"{top['trigger_model'][:80]} (Exp {top['expectancy']})",
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except TriggerEntryOptimizationError as exc:
        logger.error("Trigger entry optimization error: %s", exc)
        print(f"Trigger entry optimization error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected trigger entry optimization error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
