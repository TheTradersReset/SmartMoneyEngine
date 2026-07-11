"""
Tier-2 exit optimization research for SmartMoneyEngine.

Evaluates partial and trailing exit models for Tier-2 BOS Close trades with
structural swing stop loss. Research-only; no production logic changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.tier2_entry_optimization_research import Tier2EntryOptimizationResearch
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch
from src.research.tiered_signal_framework_research import (
    FORWARD_BARS,
    RISK_LOOKBACK,
    SL_BUFFER_POINTS,
    TierSignal,
    TieredSignalFrameworkResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tier2_exit_optimization.json"

EXIT_MODELS: dict[str, str] = {
    "A": "Full exit at 1R",
    "B": "50% at 1R, 50% at 2R",
    "C": "50% at 1R, breakeven SL, remainder to opposite liquidity",
    "D": "33% at 1R, 33% at 2R, 33% at opposite liquidity",
    "E": "Trail swing structure after 1R",
}


class Tier2ExitOptimizationError(Exception):
    """Raised when Tier-2 exit optimization research fails."""


@dataclass(frozen=True)
class ExitTradeOutcome:
    """Simulated outcome for one exit model on one trade."""

    bos_timestamp: str
    timeframe: str
    direction: str
    exit_model: str
    entry_price: float
    stop_price: float
    risk_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExitModelMetrics:
    """Aggregate metrics for one exit model."""

    exit_model: str
    label: str
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
class Tier2ExitOptimizationReport:
    """Full Tier-2 exit optimization research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    entry_method: str
    stop_loss_model: str
    rr_reachability_baseline: dict[str, float]
    exit_models: dict[str, str]
    total_trades: int
    model_metrics: dict[str, dict[str, Any]]
    model_ranking: list[dict[str, Any]]
    best_profit_model: str
    best_risk_adjusted_model: str
    best_production_model: str
    production_recommendation: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tier2ExitOptimizationResearch:
    """Optimize exit models for Tier-2 BOS Close trades."""

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
        self.entry_engine = Tier2EntryOptimizationResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.construction_engine = TradeConstructionValidationResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _r_target(entry: float, risk: float, direction: str, multiple: float) -> float:
        if direction == "bullish":
            return round(entry + risk * multiple, 2)
        return round(entry - risk * multiple, 2)

    def _liquidity_target(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
        risk: float,
    ) -> float:
        return self.construction_engine._opposite_liquidity_target(
            frame,
            entry_bar,
            entry_price,
            direction,
            risk,
        )

    def _swing_trail_stop(
        self,
        frame: pd.DataFrame,
        bar_index: int,
        direction: str,
    ) -> float:
        window = frame.iloc[max(0, bar_index - RISK_LOOKBACK) : bar_index + 1]
        if direction == "bullish":
            anchor = float(window["Low"].min())
            return round(anchor - SL_BUFFER_POINTS, 2)
        anchor = float(window["High"].max())
        return round(anchor + SL_BUFFER_POINTS, 2)

    @staticmethod
    def _directional_pnl(entry: float, price: float, direction: str) -> float:
        if direction == "bullish":
            return price - entry
        return entry - price

    def _close_remaining(
        self,
        entry: float,
        close: float,
        direction: str,
        remaining: float,
    ) -> float:
        return round(remaining * self._directional_pnl(entry, close, direction), 2)

    def _stop_loss_pnl(
        self,
        entry: float,
        stop: float,
        direction: str,
        remaining: float,
    ) -> float:
        return round(remaining * self._directional_pnl(entry, stop, direction), 2)

    def _simulate_model_a(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry: float,
        stop: float,
        risk: float,
        direction: str,
    ) -> tuple[float, bool]:
        target = self._r_target(entry, risk, direction, 1.0)
        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)

        for index in range(entry_bar + 1, end + 1):
            high = float(frame.iloc[index]["High"])
            low = float(frame.iloc[index]["Low"])

            if direction == "bullish":
                if low <= stop:
                    return -risk, False
                if high >= target:
                    return round(risk, 2), True
            else:
                if high >= stop:
                    return -risk, False
                if low <= target:
                    return round(risk, 2), True

        close = float(frame.iloc[end]["Close"])
        pnl = round(self._directional_pnl(entry, close, direction), 2)
        return pnl, pnl > 0

    def _simulate_partial_legs(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry: float,
        stop: float,
        risk: float,
        direction: str,
        legs: list[tuple[float, float | str]],
        breakeven_after_first: bool = False,
    ) -> tuple[float, bool]:
        liquidity = self._liquidity_target(frame, entry_bar, entry, direction, risk)
        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
        remaining = 1.0
        pnl = 0.0
        current_stop = stop
        leg_index = 0

        for index in range(entry_bar + 1, end + 1):
            high = float(frame.iloc[index]["High"])
            low = float(frame.iloc[index]["Low"])

            if direction == "bullish":
                stop_hit = low <= current_stop
            else:
                stop_hit = high >= current_stop

            if stop_hit and remaining > 0:
                pnl += self._stop_loss_pnl(entry, current_stop, direction, remaining)
                remaining = 0.0
                break

            while leg_index < len(legs) and remaining > 0:
                fraction, leg_target = legs[leg_index]
                if isinstance(leg_target, str):
                    target_price = liquidity
                else:
                    target_price = self._r_target(entry, risk, direction, leg_target)

                hit = (
                    high >= target_price if direction == "bullish" else low <= target_price
                )
                if not hit:
                    break

                take = min(fraction, remaining)
                pnl += round(take * self._directional_pnl(entry, target_price, direction), 2)
                remaining = round(remaining - take, 4)
                leg_index += 1

                if breakeven_after_first and leg_index == 1:
                    current_stop = entry

            if remaining <= 0:
                break

        if remaining > 0:
            close = float(frame.iloc[end]["Close"])
            pnl += self._close_remaining(entry, close, direction, remaining)

        pnl = round(pnl, 2)
        return pnl, pnl > 0

    def _simulate_model_e(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry: float,
        stop: float,
        risk: float,
        direction: str,
    ) -> tuple[float, bool]:
        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
        target_1r = self._r_target(entry, risk, direction, 1.0)
        trail_active = False
        trail_stop = stop

        for index in range(entry_bar + 1, end + 1):
            high = float(frame.iloc[index]["High"])
            low = float(frame.iloc[index]["Low"])

            if not trail_active:
                if direction == "bullish":
                    if low <= stop:
                        return -risk, False
                    if high >= target_1r:
                        trail_active = True
                        trail_stop = self._swing_trail_stop(frame, index, direction)
                else:
                    if high >= stop:
                        return -risk, False
                    if low <= target_1r:
                        trail_active = True
                        trail_stop = self._swing_trail_stop(frame, index, direction)
                continue

            candidate = self._swing_trail_stop(frame, index, direction)
            if direction == "bullish":
                trail_stop = max(trail_stop, candidate)
                if low <= trail_stop:
                    pnl = round(self._directional_pnl(entry, trail_stop, direction), 2)
                    return pnl, pnl > 0
            else:
                trail_stop = min(trail_stop, candidate)
                if high >= trail_stop:
                    pnl = round(self._directional_pnl(entry, trail_stop, direction), 2)
                    return pnl, pnl > 0

        close = float(frame.iloc[end]["Close"])
        pnl = round(self._directional_pnl(entry, close, direction), 2)
        return pnl, pnl > 0

    def _simulate_exit(
        self,
        model_key: str,
        frame: pd.DataFrame,
        entry_bar: int,
        entry: float,
        stop: float,
        risk: float,
        direction: str,
    ) -> tuple[float, bool]:
        if model_key == "A":
            return self._simulate_model_a(frame, entry_bar, entry, stop, risk, direction)
        if model_key == "B":
            return self._simulate_partial_legs(
                frame,
                entry_bar,
                entry,
                stop,
                risk,
                direction,
                [(0.5, 1.0), (0.5, 2.0)],
            )
        if model_key == "C":
            return self._simulate_partial_legs(
                frame,
                entry_bar,
                entry,
                stop,
                risk,
                direction,
                [(0.5, 1.0), (0.5, "liquidity")],
                breakeven_after_first=True,
            )
        if model_key == "D":
            third = round(1.0 / 3.0, 4)
            return self._simulate_partial_legs(
                frame,
                entry_bar,
                entry,
                stop,
                risk,
                direction,
                [(third, 1.0), (third, 2.0), (third, "liquidity")],
            )
        if model_key == "E":
            return self._simulate_model_e(frame, entry_bar, entry, stop, risk, direction)
        raise Tier2ExitOptimizationError(f"Unknown exit model: {model_key}")

    def _evaluate_signal(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
        model_key: str,
    ) -> ExitTradeOutcome | None:
        trigger = self.entry_engine._resolve_entry("A_bos_close", frame, signal)
        if not trigger.triggered or trigger.entry_bar is None or trigger.entry_price is None:
            return None

        entry_bar = trigger.entry_bar
        entry = trigger.entry_price
        stop, risk = self.construction_engine._structural_stop(
            frame,
            entry_bar,
            entry,
            signal.direction,
        )
        pnl, win = self._simulate_exit(
            model_key,
            frame,
            entry_bar,
            entry,
            stop,
            risk,
            signal.direction,
        )
        rr = round(pnl / risk, 2) if risk > 0 else 0.0

        return ExitTradeOutcome(
            bos_timestamp=signal.bos_timestamp,
            timeframe=signal.timeframe,
            direction=signal.direction,
            exit_model=model_key,
            entry_price=entry,
            stop_price=stop,
            risk_points=risk,
            realized_pnl_points=pnl,
            realized_rr=rr,
            win=win,
        )

    def _collect_outcomes(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, list[ExitTradeOutcome]]:
        from datetime import timedelta

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        grouped: dict[str, list[ExitTradeOutcome]] = {key: [] for key in EXIT_MODELS}
        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)

            for signal in self.tier_engine._detect_tier2(frame, timeframe_label):
                for model_key in EXIT_MODELS:
                    outcome = self._evaluate_signal(frame, signal, model_key)
                    if outcome is not None:
                        grouped[model_key].append(outcome)

        for outcomes in grouped.values():
            outcomes.sort(key=lambda item: pd.Timestamp(item.bos_timestamp))
        return grouped

    def _metrics(self, model_key: str, outcomes: list[ExitTradeOutcome]) -> ExitModelMetrics:
        if not outcomes:
            return ExitModelMetrics(
                exit_model=model_key,
                label=EXIT_MODELS[model_key],
                trades=0,
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                net_points=0.0,
                maximum_drawdown_points=0.0,
            )

        pnls = [item.realized_pnl_points for item in outcomes]
        rrs = [item.realized_rr for item in outcomes]
        wins = sum(1 for item in outcomes if item.win)

        return ExitModelMetrics(
            exit_model=model_key,
            label=EXIT_MODELS[model_key],
            trades=len(outcomes),
            win_rate_pct=round(wins / len(outcomes) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
        )

    @staticmethod
    def _risk_adjusted_score(metrics: ExitModelMetrics) -> float:
        pf = metrics.profit_factor or 0.0
        if pf == float("inf"):
            pf = 10.0
        dd = max(metrics.maximum_drawdown_points, 1.0)
        return round(metrics.expectancy * pf / dd * 1000, 4)

    @staticmethod
    def _production_score(metrics: ExitModelMetrics) -> float:
        pf = metrics.profit_factor or 0.0
        if pf == float("inf"):
            pf = 10.0
        return round(
            metrics.expectancy * 0.45
            + pf * 15.0
            + metrics.win_rate_pct * 0.25
            + metrics.average_rr * 4.0
            - metrics.maximum_drawdown_points * 0.008,
            4,
        )

    def run(self, metadata: dict[str, Any]) -> Tier2ExitOptimizationReport:
        """Run Tier-2 exit optimization research."""
        started = time.perf_counter()
        grouped = self._collect_outcomes(metadata)

        if not grouped.get("A"):
            raise Tier2ExitOptimizationError("No Tier-2 BOS Close exit outcomes found.")

        all_metrics = {key: self._metrics(key, grouped[key]) for key in EXIT_MODELS}
        ranked = sorted(
            all_metrics.values(),
            key=lambda item: (item.net_points, item.expectancy, item.profit_factor or 0),
            reverse=True,
        )

        best_profit = max(all_metrics.values(), key=lambda item: (item.net_points, item.expectancy))
        best_risk_adjusted = max(
            all_metrics.values(),
            key=lambda item: (self._risk_adjusted_score(item), item.expectancy),
        )
        best_production = max(
            all_metrics.values(),
            key=lambda item: (self._production_score(item), item.net_points),
        )

        production = best_production
        production_recommendation = {
            "exit_model": production.exit_model,
            "label": production.label,
            "trades": production.trades,
            "win_rate_pct": production.win_rate_pct,
            "profit_factor": production.profit_factor,
            "expectancy": production.expectancy,
            "average_rr": production.average_rr,
            "net_points": production.net_points,
            "maximum_drawdown_points": production.maximum_drawdown_points,
            "recommendation": f"Deploy Tier-2 BOS Close with exit model {production.exit_model}: {production.label}.",
        }

        conclusions = [
            f"Evaluated {len(EXIT_MODELS)} exit models on {all_metrics['A'].trades} Tier-2 BOS Close trades.",
            (
                f"Best profit model: {best_profit.exit_model} ({best_profit.label}) — "
                f"net {best_profit.net_points} pts, expectancy {best_profit.expectancy}."
            ),
            (
                f"Best risk-adjusted model: {best_risk_adjusted.exit_model} ({best_risk_adjusted.label}) — "
                f"score {self._risk_adjusted_score(best_risk_adjusted)}, DD {best_risk_adjusted.maximum_drawdown_points}."
            ),
            (
                f"Best production model: {best_production.exit_model} ({best_production.label}) — "
                f"expectancy {best_production.expectancy}, PF {best_production.profit_factor}."
            ),
            production_recommendation["recommendation"],
        ]

        return Tier2ExitOptimizationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            entry_method="BOS Close Entry",
            stop_loss_model="Structural Swing SL",
            rr_reachability_baseline={
                "reached_1r_pct": 46.02,
                "reached_2r_pct": 24.1,
                "reached_3r_pct": 9.76,
            },
            exit_models=EXIT_MODELS,
            total_trades=all_metrics["A"].trades,
            model_metrics={key: metrics.as_dict() for key, metrics in all_metrics.items()},
            model_ranking=[item.as_dict() for item in ranked],
            best_profit_model=best_profit.exit_model,
            best_risk_adjusted_model=best_risk_adjusted.exit_model,
            best_production_model=best_production.exit_model,
            production_recommendation=production_recommendation,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tier2_exit_optimization_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Tier2ExitOptimizationReport:
    """Run Tier-2 exit optimization and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Tier2ExitOptimizationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = Tier2ExitOptimizationResearch(
        symbol=metadata.get("symbol", "NIFTY50"),
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", ("5M", "15M", "1H"))),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info("Tier-2 exit optimization completed: production=%s", report.best_production_model)
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tier2_exit_optimization_report()
        print("Tier-2 Exit Optimization Summary")
        print(f"Trades: {report.total_trades}")
        for item in report.model_ranking:
            print(
                f"  {item['exit_model']}: net={item['net_points']} "
                f"exp={item['expectancy']} WR={item['win_rate_pct']}% "
                f"DD={item['maximum_drawdown_points']}"
            )
        print(f"Best profit: {report.best_profit_model}")
        print(f"Best risk-adjusted: {report.best_risk_adjusted_model}")
        print(f"Best production: {report.best_production_model}")
        print(report.production_recommendation["recommendation"])
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except Tier2ExitOptimizationError as exc:
        logger.error("Tier-2 exit optimization error: %s", exc)
        print(f"Tier-2 exit optimization error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Tier-2 exit optimization failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
