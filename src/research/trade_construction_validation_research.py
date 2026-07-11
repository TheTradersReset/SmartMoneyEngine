"""
Trade construction validation research for SmartMoneyEngine.

Evaluates Entry + Stop Loss + Target combinations on existing Tier-2 BOS Close
signals. Research-only; no production logic or signal changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine
from src.research.filter_research_engine import FilterContextBuilder, RESEARCH_DAYS, _json_safe
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.tier2_entry_optimization_research import Tier2EntryOptimizationResearch
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.tiered_signal_framework_research import (
    FORWARD_BARS,
    MIN_RISK_POINTS,
    RISK_LOOKBACK,
    SL_BUFFER_POINTS,
    TIMEFRAME_MINUTES,
    TierSignal,
    TieredSignalFrameworkResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "trade_construction_validation.json"

ATR_MULTIPLIER = 1.5
DEALING_RANGE_LOOKBACK = 60
FALLBACK_TARGET_R = 2.0
MIN_COMBO_SIGNALS = 20

ENTRY_MODELS: dict[str, str] = {
    "A_bos_close": "BOS Close",
}

STOP_MODELS: dict[str, str] = {
    "A_structural_swing": "Structural Swing SL",
    "B_liquidity_sweep": "Liquidity Sweep SL",
    "C_atr": "ATR SL",
}

TARGET_MODELS: dict[str, str] = {
    "A_1r": "1R",
    "B_2r": "2R",
    "C_3r": "3R",
    "D_opposite_liquidity": "Opposite Liquidity Pool",
    "E_htf_supply_demand": "HTF Supply/Demand Zone",
}


class TradeConstructionValidationError(Exception):
    """Raised when trade construction validation fails."""


@dataclass(frozen=True)
class TradeConstructionOutcome:
    """Simulated outcome for one Entry + SL + Target combination."""

    bos_timestamp: str
    timeframe: str
    direction: str
    entry_model: str
    stop_model: str
    target_model: str
    entry_price: float
    stop_price: float
    target_price: float
    risk_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    holding_bars: int
    holding_minutes: float
    target_hit: bool
    stop_hit: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CombinationMetrics:
    """Aggregate metrics for one trade construction combination."""

    combination_key: str
    entry_model: str
    entry_label: str
    stop_model: str
    stop_label: str
    target_model: str
    target_label: str
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
class TradeConstructionValidationReport:
    """Full trade construction validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    entry_method: str
    entry_price_rule: str
    entry_models: dict[str, str]
    stop_models: dict[str, str]
    target_models: dict[str, str]
    total_tier2_signals: int
    combinations_evaluated: int
    combination_metrics: list[dict[str, Any]]
    best_accuracy_model: dict[str, Any]
    best_rr_model: dict[str, Any]
    best_net_profit_model: dict[str, Any]
    recommended_production_entry: str
    recommended_production_sl: str
    recommended_production_target: str
    production_recommendation: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TradeConstructionValidationResearch:
    """Validate optimal Entry + SL + Target construction for Tier-2 signals."""

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
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)

    @staticmethod
    def _is_active(value: Any) -> bool:
        return LiquidityNarrativeEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        return LiquidityNarrativeEngine._to_float(value)

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _combination_key(entry: str, stop: str, target: str) -> str:
        return f"{entry}|{stop}|{target}"

    @staticmethod
    def _minutes_per_bar(timeframe: str) -> int:
        return TIMEFRAME_MINUTES.get(timeframe, 5)

    def _structural_stop(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
    ) -> tuple[float, float]:
        lookback = frame.iloc[max(0, entry_bar - RISK_LOOKBACK) : entry_bar + 1]
        if direction == "bullish":
            anchor = float(lookback["Low"].min())
            stop = round(anchor - SL_BUFFER_POINTS, 2)
            risk = max(entry_price - stop, MIN_RISK_POINTS)
        else:
            anchor = float(lookback["High"].max())
            stop = round(anchor + SL_BUFFER_POINTS, 2)
            risk = max(stop - entry_price, MIN_RISK_POINTS)
        return stop, round(risk, 2)

    def _liquidity_sweep_stop(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
    ) -> tuple[float, float]:
        start = max(0, entry_bar - RISK_LOOKBACK)
        window = frame.iloc[start : entry_bar + 1]

        if direction == "bullish":
            sweep_lows: list[float] = []
            for _, row in window.iterrows():
                if self._is_active(row.get("Sell_Liquidity_Sweep")):
                    sweep_lows.append(float(row["Low"]))
            if sweep_lows:
                stop = round(min(sweep_lows) - SL_BUFFER_POINTS, 2)
                risk = max(entry_price - stop, MIN_RISK_POINTS)
                return stop, round(risk, 2)
        else:
            sweep_highs: list[float] = []
            for _, row in window.iterrows():
                if self._is_active(row.get("Buy_Liquidity_Sweep")):
                    sweep_highs.append(float(row["High"]))
            if sweep_highs:
                stop = round(max(sweep_highs) + SL_BUFFER_POINTS, 2)
                risk = max(stop - entry_price, MIN_RISK_POINTS)
                return stop, round(risk, 2)

        return self._structural_stop(frame, entry_bar, entry_price, direction)

    def _atr_stop(
        self,
        frame: pd.DataFrame,
        atr_series: pd.Series,
        entry_bar: int,
        entry_price: float,
        direction: str,
    ) -> tuple[float, float]:
        atr = float(atr_series.iloc[entry_bar]) if pd.notna(atr_series.iloc[entry_bar]) else None
        if atr is None or atr <= 0:
            return self._structural_stop(frame, entry_bar, entry_price, direction)

        risk = round(max(atr * ATR_MULTIPLIER, MIN_RISK_POINTS), 2)
        if direction == "bullish":
            stop = round(entry_price - risk, 2)
        else:
            stop = round(entry_price + risk, 2)
        return stop, risk

    def _resolve_stop(
        self,
        stop_model: str,
        frame: pd.DataFrame,
        atr_series: pd.Series,
        entry_bar: int,
        entry_price: float,
        direction: str,
    ) -> tuple[float, float]:
        if stop_model == "A_structural_swing":
            return self._structural_stop(frame, entry_bar, entry_price, direction)
        if stop_model == "B_liquidity_sweep":
            return self._liquidity_sweep_stop(frame, entry_bar, entry_price, direction)
        if stop_model == "C_atr":
            return self._atr_stop(frame, atr_series, entry_bar, entry_price, direction)
        raise TradeConstructionValidationError(f"Unknown stop model: {stop_model}")

    @staticmethod
    def _fixed_r_target(
        entry_price: float,
        risk: float,
        direction: str,
        multiple: float,
    ) -> float:
        if direction == "bullish":
            return round(entry_price + risk * multiple, 2)
        return round(entry_price - risk * multiple, 2)

    def _opposite_liquidity_target(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
        risk: float,
    ) -> float:
        row = frame.iloc[entry_bar]
        if direction == "bullish":
            pool = self._to_float(row.get("Buy_Side_Liquidity"))
            if pool is not None and pool > entry_price + MIN_RISK_POINTS:
                return round(pool, 2)
            end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
            pools = [
                level
                for level in (
                    self._to_float(frame.iloc[index].get("Buy_Side_Liquidity"))
                    for index in range(entry_bar, end + 1)
                )
                if level is not None and level > entry_price + MIN_RISK_POINTS
            ]
            if pools:
                return round(min(pools), 2)
        else:
            pool = self._to_float(row.get("Sell_Side_Liquidity"))
            if pool is not None and pool < entry_price - MIN_RISK_POINTS:
                return round(pool, 2)
            end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
            pools = [
                level
                for level in (
                    self._to_float(frame.iloc[index].get("Sell_Side_Liquidity"))
                    for index in range(entry_bar, end + 1)
                )
                if level is not None and level < entry_price - MIN_RISK_POINTS
            ]
            if pools:
                return round(max(pools), 2)

        return self._fixed_r_target(entry_price, risk, direction, FALLBACK_TARGET_R)

    def _htf_supply_demand_target(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
        risk: float,
    ) -> float:
        window = frame.iloc[max(0, entry_bar - DEALING_RANGE_LOOKBACK) : entry_bar + 1]
        swing_high = self._to_float(window["Swing_High"].dropna().iloc[-1]) if window[
            "Swing_High"
        ].notna().any() else None
        swing_low = self._to_float(window["Swing_Low"].dropna().iloc[-1]) if window[
            "Swing_Low"
        ].notna().any() else None

        if swing_high is None:
            swing_high = float(window["High"].astype(float).max())
        if swing_low is None:
            swing_low = float(window["Low"].astype(float).min())

        if direction == "bullish":
            target = round(swing_high, 2)
            if target > entry_price + MIN_RISK_POINTS:
                return target
        else:
            target = round(swing_low, 2)
            if target < entry_price - MIN_RISK_POINTS:
                return target

        return self._fixed_r_target(entry_price, risk, direction, FALLBACK_TARGET_R)

    def _resolve_target(
        self,
        target_model: str,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
        risk: float,
    ) -> float:
        multiples = {"A_1r": 1.0, "B_2r": 2.0, "C_3r": 3.0}
        if target_model in multiples:
            return self._fixed_r_target(entry_price, risk, direction, multiples[target_model])
        if target_model == "D_opposite_liquidity":
            return self._opposite_liquidity_target(frame, entry_bar, entry_price, direction, risk)
        if target_model == "E_htf_supply_demand":
            return self._htf_supply_demand_target(frame, entry_bar, entry_price, direction, risk)
        raise TradeConstructionValidationError(f"Unknown target model: {target_model}")

    def _simulate_trade(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
        stop: float,
        target: float,
        risk: float,
        timeframe: str,
    ) -> tuple[float, float, bool, int, bool, bool]:
        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
        stop_hit = False
        target_hit = False
        exit_bar = end

        for index in range(entry_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])

            if direction == "bullish":
                if bar_low <= stop:
                    stop_hit = True
                    exit_bar = index
                    return -risk, -1.0, False, exit_bar - entry_bar, stop_hit, target_hit
                if bar_high >= target:
                    target_hit = True
                    exit_bar = index
                    pnl = round(target - entry_price, 2)
                    rr = round(pnl / risk, 2) if risk > 0 else 0.0
                    return pnl, rr, True, exit_bar - entry_bar, stop_hit, target_hit
            else:
                if bar_high >= stop:
                    stop_hit = True
                    exit_bar = index
                    return -risk, -1.0, False, exit_bar - entry_bar, stop_hit, target_hit
                if bar_low <= target:
                    target_hit = True
                    exit_bar = index
                    pnl = round(entry_price - target, 2)
                    rr = round(pnl / risk, 2) if risk > 0 else 0.0
                    return pnl, rr, True, exit_bar - entry_bar, stop_hit, target_hit

        close = float(frame.iloc[end]["Close"])
        if direction == "bullish":
            pnl = round(close - entry_price, 2)
        else:
            pnl = round(entry_price - close, 2)
        rr = round(pnl / risk, 2) if risk > 0 else 0.0
        return pnl, rr, pnl > 0, end - entry_bar, stop_hit, target_hit

    def _evaluate_combination(
        self,
        frame: pd.DataFrame,
        atr_series: pd.Series,
        signal: TierSignal,
        entry_model: str,
        stop_model: str,
        target_model: str,
    ) -> TradeConstructionOutcome | None:
        trigger = self.entry_engine._resolve_entry(entry_model, frame, signal)
        if not trigger.triggered or trigger.entry_bar is None or trigger.entry_price is None:
            return None

        entry_bar = trigger.entry_bar
        entry_price = trigger.entry_price
        stop, risk = self._resolve_stop(
            stop_model,
            frame,
            atr_series,
            entry_bar,
            entry_price,
            signal.direction,
        )
        target = self._resolve_target(
            target_model,
            frame,
            entry_bar,
            entry_price,
            signal.direction,
            risk,
        )

        pnl, rr, win, holding_bars, stop_hit, target_hit = self._simulate_trade(
            frame,
            entry_bar,
            entry_price,
            signal.direction,
            stop,
            target,
            risk,
            signal.timeframe,
        )
        holding_minutes = round(holding_bars * self._minutes_per_bar(signal.timeframe), 1)

        return TradeConstructionOutcome(
            bos_timestamp=signal.bos_timestamp,
            timeframe=signal.timeframe,
            direction=signal.direction,
            entry_model=entry_model,
            stop_model=stop_model,
            target_model=target_model,
            entry_price=entry_price,
            stop_price=stop,
            target_price=target,
            risk_points=risk,
            realized_pnl_points=pnl,
            realized_rr=rr,
            win=win,
            holding_bars=holding_bars,
            holding_minutes=holding_minutes,
            target_hit=target_hit,
            stop_hit=stop_hit,
        )

    def _collect_outcomes(self, metadata: dict[str, Any]) -> dict[str, list[TradeConstructionOutcome]]:
        from datetime import timedelta

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        grouped: dict[str, list[TradeConstructionOutcome]] = {
            self._combination_key(entry, stop, target): []
            for entry, stop, target in product(
                ENTRY_MODELS,
                STOP_MODELS,
                TARGET_MODELS,
            )
        }
        tier2_count = 0

        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            enriched = FilterContextBuilder().enrich(frame)
            atr_series = enriched["_atr"]

            for signal in self.tier_engine._detect_tier2(frame, timeframe_label):
                tier2_count += 1
                for entry_model, stop_model, target_model in product(
                    ENTRY_MODELS,
                    STOP_MODELS,
                    TARGET_MODELS,
                ):
                    outcome = self._evaluate_combination(
                        frame,
                        atr_series,
                        signal,
                        entry_model,
                        stop_model,
                        target_model,
                    )
                    if outcome is not None:
                        key = self._combination_key(entry_model, stop_model, target_model)
                        grouped[key].append(outcome)

        self._tier2_signal_count = tier2_count
        return grouped

    def _metrics_for_outcomes(
        self,
        entry_model: str,
        stop_model: str,
        target_model: str,
        outcomes: list[TradeConstructionOutcome],
    ) -> CombinationMetrics:
        key = self._combination_key(entry_model, stop_model, target_model)
        if not outcomes:
            return CombinationMetrics(
                combination_key=key,
                entry_model=entry_model,
                entry_label=ENTRY_MODELS[entry_model],
                stop_model=stop_model,
                stop_label=STOP_MODELS[stop_model],
                target_model=target_model,
                target_label=TARGET_MODELS[target_model],
                trades=0,
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                net_points=0.0,
                maximum_drawdown_points=0.0,
            )

        pnls = [outcome.realized_pnl_points for outcome in outcomes]
        rrs = [outcome.realized_rr for outcome in outcomes]
        wins = sum(1 for outcome in outcomes if outcome.win)

        return CombinationMetrics(
            combination_key=key,
            entry_model=entry_model,
            entry_label=ENTRY_MODELS[entry_model],
            stop_model=stop_model,
            stop_label=STOP_MODELS[stop_model],
            target_model=target_model,
            target_label=TARGET_MODELS[target_model],
            trades=len(outcomes),
            win_rate_pct=round(wins / len(outcomes) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
        )

    @staticmethod
    def _model_summary(metrics: CombinationMetrics, rank_type: str) -> dict[str, Any]:
        return {
            "rank_type": rank_type,
            "combination_key": metrics.combination_key,
            "entry": metrics.entry_label,
            "stop_loss": metrics.stop_label,
            "target": metrics.target_label,
            "trades": metrics.trades,
            "win_rate_pct": metrics.win_rate_pct,
            "profit_factor": metrics.profit_factor,
            "expectancy": metrics.expectancy,
            "average_rr": metrics.average_rr,
            "net_points": metrics.net_points,
            "maximum_drawdown_points": metrics.maximum_drawdown_points,
        }

    def run(self, metadata: dict[str, Any]) -> TradeConstructionValidationReport:
        """Run trade construction validation research."""
        started = time.perf_counter()

        grouped = self._collect_outcomes(metadata)
        tier2_count = getattr(self, "_tier2_signal_count", 0)

        all_metrics: list[CombinationMetrics] = []
        for entry_model, stop_model, target_model in product(
            ENTRY_MODELS,
            STOP_MODELS,
            TARGET_MODELS,
        ):
            key = self._combination_key(entry_model, stop_model, target_model)
            all_metrics.append(
                self._metrics_for_outcomes(
                    entry_model,
                    stop_model,
                    target_model,
                    grouped[key],
                )
            )

        eligible = [metrics for metrics in all_metrics if metrics.trades >= MIN_COMBO_SIGNALS]
        if not eligible:
            eligible = all_metrics

        best_accuracy = max(eligible, key=lambda item: (item.win_rate_pct, item.expectancy))
        best_rr = max(eligible, key=lambda item: (item.average_rr, item.net_points))
        best_net = max(eligible, key=lambda item: (item.net_points, item.expectancy))
        production = best_net

        production_recommendation = {
            "entry": production.entry_label,
            "stop_loss": production.stop_label,
            "target": production.target_label,
            "combination_key": production.combination_key,
            "trades": production.trades,
            "win_rate_pct": production.win_rate_pct,
            "profit_factor": production.profit_factor,
            "expectancy": production.expectancy,
            "average_rr": production.average_rr,
            "net_points": production.net_points,
            "maximum_drawdown_points": production.maximum_drawdown_points,
            "recommendation": (
                f"Deploy Tier-2 with {production.entry_label} entry, "
                f"{production.stop_label}, and {production.target_label} target."
            ),
        }

        conclusions = [
            f"Evaluated {len(all_metrics)} Entry+SL+Target combinations on {tier2_count} Tier-2 BOS Close signals.",
            (
                f"Best accuracy: {best_accuracy.entry_label} + {best_accuracy.stop_label} + "
                f"{best_accuracy.target_label} (WR {best_accuracy.win_rate_pct}%)."
            ),
            (
                f"Best RR: {best_rr.entry_label} + {best_rr.stop_label} + "
                f"{best_rr.target_label} (avg RR {best_rr.average_rr})."
            ),
            (
                f"Best net profit: {best_net.entry_label} + {best_net.stop_label} + "
                f"{best_net.target_label} (net {best_net.net_points} pts)."
            ),
            production_recommendation["recommendation"],
        ]

        return TradeConstructionValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            entry_method="BOS Close Entry",
            entry_price_rule="BOS candle close",
            entry_models=ENTRY_MODELS,
            stop_models=STOP_MODELS,
            target_models=TARGET_MODELS,
            total_tier2_signals=tier2_count,
            combinations_evaluated=len(all_metrics),
            combination_metrics=[item.as_dict() for item in all_metrics],
            best_accuracy_model=self._model_summary(best_accuracy, "Best Accuracy"),
            best_rr_model=self._model_summary(best_rr, "Best RR"),
            best_net_profit_model=self._model_summary(best_net, "Best Net Profit"),
            recommended_production_entry=production.entry_label,
            recommended_production_sl=production.stop_label,
            recommended_production_target=production.target_label,
            production_recommendation=production_recommendation,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_trade_construction_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> TradeConstructionValidationReport:
    """Run trade construction validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise TradeConstructionValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = TradeConstructionValidationResearch(
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
        "Trade construction validation completed: %s + %s + %s",
        report.recommended_production_entry,
        report.recommended_production_sl,
        report.recommended_production_target,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_trade_construction_validation_report()
        print("Trade Construction Validation Summary")
        print(f"Tier-2 signals: {report.total_tier2_signals}")
        print(f"Combinations: {report.combinations_evaluated}")
        print(
            f"Production: {report.recommended_production_entry} | "
            f"{report.recommended_production_sl} | "
            f"{report.recommended_production_target}"
        )
        print(f"Best accuracy WR: {report.best_accuracy_model['win_rate_pct']}%")
        print(f"Best RR: {report.best_rr_model['average_rr']}")
        print(f"Best net: {report.best_net_profit_model['net_points']}")
        print(report.production_recommendation["recommendation"])
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except TradeConstructionValidationError as exc:
        logger.error("Trade construction validation error: %s", exc)
        print(f"Trade construction validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected trade construction validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
