"""
SmartMoneyEngine institutional target optimizer.

Discovers, ranks, and selects SMC-based profit targets from pipeline outputs
and multi-timeframe context — replacing mechanical fixed-R multiples.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.signals.decision_engine import DecisionEngineError, TradeDecision, evaluate_pipeline

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_MTF_REPORT = PROJECT_ROOT / "outputs" / "signals" / "multi_timeframe_report.json"
DEFAULT_TRADE_PLAN_V2 = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report_v2.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "signals" / "target_optimizer_report.json"

FORWARD_SCAN_BARS = 400
HTF_FORWARD_SCAN_BARS = 800
LOOKBACK_BARS = 200
MIN_RISK_POINTS = 1.0
MIN_TARGET_SEPARATION_R = 0.25
MAJOR_LIQUIDITY_STRENGTH = 0.5


class TargetOptimizerError(Exception):
    """Raised when target optimization fails."""


class TargetType(str, Enum):
    """Institutional target classification."""

    NEAREST_LIQUIDITY = "NEAREST_LIQUIDITY"
    MAJOR_LIQUIDITY = "MAJOR_LIQUIDITY"
    HTF_LIQUIDITY = "HTF_LIQUIDITY"
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"
    STRUCTURE_TARGET = "STRUCTURE_TARGET"


@dataclass
class OptimizedTarget:
    """Single optimized profit target."""

    target_price: float
    target_type: str
    target_probability: float
    expected_rr: float
    reasoning: str

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable target dictionary."""
        return asdict(self)


@dataclass
class SignalTargetAnalysis:
    """Target path analysis for one trade signal."""

    signal_index: int
    signal_date: str
    decision: str
    entry: float
    stop_loss: float
    risk_points: float
    target_path: list[dict[str, Any]]
    selected_targets: list[dict[str, Any]]
    average_probability: float
    average_expected_rr: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable analysis dictionary."""
        return asdict(self)


@dataclass
class TargetOptimizerReport:
    """Aggregate target optimization report."""

    symbol: str
    timeframe: str
    source_csv: str
    source_mtf_report: str
    source_trade_plan_report: str
    total_signals: int
    average_target_probability: float
    average_expected_rr: float
    execution_time_seconds: float
    signal_analyses: list[dict[str, Any]] = field(default_factory=list)
    top_targets: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return asdict(self)


class TargetOptimizer:
    """
    Discover and rank institutional targets from SMC pipeline data.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Timeframe label for reporting.
    lookback_bars : int, optional
        Bars to search backward for swing and structure context.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        lookback_bars: int = LOOKBACK_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars
        self._mtf_report: dict[str, Any] | None = None

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
    def _to_float(value: Any) -> float | None:
        if not TargetOptimizer._is_active(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _window_start(self, index: int) -> int:
        return max(0, index - self.lookback_bars)

    def load_mtf_report(self, path: Path | str | None = None) -> dict[str, Any]:
        """Load multi-timeframe alignment report."""
        report_path = Path(path) if path is not None else DEFAULT_MTF_REPORT
        if not report_path.exists():
            logger.warning("MTF report not found at %s; using neutral defaults.", report_path)
            self._mtf_report = {"overall_bias": "Neutral", "timeframes": []}
            return self._mtf_report
        with report_path.open("r", encoding="utf-8") as handle:
            self._mtf_report = json.load(handle)
        return self._mtf_report

    @staticmethod
    def load_trade_plans(path: Path | str | None = None) -> list[dict[str, Any]]:
        """Load trade plan V2 entries for signal context."""
        plan_path = Path(path) if path is not None else DEFAULT_TRADE_PLAN_V2
        if not plan_path.exists():
            return []
        with plan_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return list(payload.get("trade_plans", []))

    def _htf_trend_aligned(self, decision: str) -> bool:
        if self._mtf_report is None:
            return False
        trends = {
            item["timeframe"]: item.get("trend", "Neutral")
            for item in self._mtf_report.get("timeframes", [])
        }
        htf = [trends.get("1D", "Neutral"), trends.get("4H", "Neutral")]
        if decision == TradeDecision.BUY.value:
            return sum(1 for trend in htf if trend == "Bullish") >= 1
        return sum(1 for trend in htf if trend == "Bearish") >= 1

    def _htf_opposed(self, decision: str) -> bool:
        if self._mtf_report is None:
            return False
        trends = {
            item["timeframe"]: item.get("trend", "Neutral")
            for item in self._mtf_report.get("timeframes", [])
        }
        htf = [trends.get("1D", "Neutral"), trends.get("4H", "Neutral")]
        if decision == TradeDecision.BUY.value:
            return sum(1 for trend in htf if trend == "Bearish") >= 1
        return sum(1 for trend in htf if trend == "Bullish") >= 1

    @staticmethod
    def _compute_rr(entry: float, stop_loss: float, target: float, decision: str) -> float:
        risk = abs(entry - stop_loss)
        if risk <= 0:
            return 0.0
        reward = (target - entry) if decision == TradeDecision.BUY.value else (entry - target)
        return round(reward / risk, 2) if reward > 0 else 0.0

    def _collect_buy_candidates(
        self,
        frame: pd.DataFrame,
        index: int,
        entry: float,
    ) -> list[tuple[float, TargetType, str]]:
        start = self._window_start(index)
        end = min(len(frame), index + HTF_FORWARD_SCAN_BARS)
        forward = frame.iloc[index:end]
        lookback = frame.iloc[start:index + 1]
        candidates: list[tuple[float, TargetType, str]] = []

        for _, row in forward.iterrows():
            buy_liq = self._to_float(row.get("Buy_Side_Liquidity"))
            if buy_liq is not None and buy_liq > entry:
                candidates.append(
                    (buy_liq, TargetType.NEAREST_LIQUIDITY, "Buy-side liquidity pool above entry")
                )

            swing_high = self._to_float(row.get("Swing_High"))
            if swing_high is not None and swing_high > entry:
                candidates.append(
                    (swing_high, TargetType.SWING_HIGH, "Forward swing high resistance")
                )

        for _, row in lookback.iterrows():
            if self._is_active(row.get("HH")):
                hh = self._to_float(row.get("Swing_High"))
                if hh is not None and hh > entry:
                    candidates.append(
                        (hh, TargetType.STRUCTURE_TARGET, "Higher-high structure objective")
                    )
            if self._is_active(row.get("Bullish_BOS")):
                bos = self._to_float(row.get("Swing_High"))
                if bos is not None and bos > entry:
                    candidates.append(
                        (bos, TargetType.STRUCTURE_TARGET, "Bullish BOS continuation target")
                    )

        strength_levels: list[tuple[float, float]] = []
        for _, row in forward.iterrows():
            level = self._to_float(row.get("Buy_Side_Liquidity"))
            strength = self._to_float(row.get("Liquidity_Strength")) or 0.0
            if level is not None and level > entry:
                strength_levels.append((level, strength))
            if self._is_active(row.get("Equal_High")):
                eq = self._to_float(row.get("Buy_Side_Liquidity")) or self._to_float(
                    row.get("Swing_High")
                )
                if eq is not None and eq > entry:
                    candidates.append(
                        (eq, TargetType.MAJOR_LIQUIDITY, "Equal highs major liquidity pool")
                    )

        for level, strength in strength_levels:
            if strength >= MAJOR_LIQUIDITY_STRENGTH:
                candidates.append(
                    (
                        level,
                        TargetType.MAJOR_LIQUIDITY,
                        f"Major buy-side liquidity (strength {strength:.2f})",
                    )
                )

        htf_levels = sorted(
            {
                level
                for level, _, _ in candidates
                if level > entry + MIN_RISK_POINTS * 2
            },
            reverse=False,
        )
        if htf_levels:
            htf_level = htf_levels[-1]
            candidates.append(
                (
                    htf_level,
                    TargetType.HTF_LIQUIDITY,
                    "Extended HTF buy-side liquidity objective",
                )
            )

        return candidates

    def _collect_sell_candidates(
        self,
        frame: pd.DataFrame,
        index: int,
        entry: float,
    ) -> list[tuple[float, TargetType, str]]:
        start = self._window_start(index)
        end = min(len(frame), index + HTF_FORWARD_SCAN_BARS)
        forward = frame.iloc[index:end]
        lookback = frame.iloc[start:index + 1]
        candidates: list[tuple[float, TargetType, str]] = []

        for _, row in forward.iterrows():
            sell_liq = self._to_float(row.get("Sell_Side_Liquidity"))
            if sell_liq is not None and sell_liq < entry:
                candidates.append(
                    (sell_liq, TargetType.NEAREST_LIQUIDITY, "Sell-side liquidity pool below entry")
                )

            swing_low = self._to_float(row.get("Swing_Low"))
            if swing_low is not None and swing_low < entry:
                candidates.append(
                    (swing_low, TargetType.SWING_LOW, "Forward swing low support")
                )

        for _, row in lookback.iterrows():
            if self._is_active(row.get("LL")):
                ll = self._to_float(row.get("Swing_Low"))
                if ll is not None and ll < entry:
                    candidates.append(
                        (ll, TargetType.STRUCTURE_TARGET, "Lower-low structure objective")
                    )
            if self._is_active(row.get("Bearish_BOS")):
                bos = self._to_float(row.get("Swing_Low"))
                if bos is not None and bos < entry:
                    candidates.append(
                        (bos, TargetType.STRUCTURE_TARGET, "Bearish BOS continuation target")
                    )

        strength_levels: list[tuple[float, float]] = []
        for _, row in forward.iterrows():
            level = self._to_float(row.get("Sell_Side_Liquidity"))
            strength = self._to_float(row.get("Liquidity_Strength")) or 0.0
            if level is not None and level < entry:
                strength_levels.append((level, strength))
            if self._is_active(row.get("Equal_Low")):
                eq = self._to_float(row.get("Sell_Side_Liquidity")) or self._to_float(
                    row.get("Swing_Low")
                )
                if eq is not None and eq < entry:
                    candidates.append(
                        (eq, TargetType.MAJOR_LIQUIDITY, "Equal lows major liquidity pool")
                    )

        for level, strength in strength_levels:
            if strength >= MAJOR_LIQUIDITY_STRENGTH:
                candidates.append(
                    (
                        level,
                        TargetType.MAJOR_LIQUIDITY,
                        f"Major sell-side liquidity (strength {strength:.2f})",
                    )
                )

        htf_levels = sorted(
            {
                level
                for level, _, _ in candidates
                if level < entry - MIN_RISK_POINTS * 2
            },
            reverse=True,
        )
        if htf_levels:
            htf_level = htf_levels[-1]
            candidates.append(
                (
                    htf_level,
                    TargetType.HTF_LIQUIDITY,
                    "Extended HTF sell-side liquidity objective",
                )
            )

        return candidates

    def _estimate_probability(
        self,
        row: pd.Series,
        decision: str,
        target_type: TargetType,
        expected_rr: float,
    ) -> float:
        probability = 42.0

        trend_strength = self._to_float(row.get("Trend_Strength")) or 0.0
        probability += min(trend_strength * 4.0, 16.0)

        liquidity_strength = self._to_float(row.get("Liquidity_Strength")) or 0.0
        probability += liquidity_strength * 18.0

        if decision == TradeDecision.BUY.value:
            if self._is_active(row.get("Bullish_BOS")):
                probability += 8.0
            if self._is_active(row.get("Bullish_CHOCH")):
                probability += 6.0
            if self._is_active(row.get("Sell_Liquidity_Sweep")):
                probability += 10.0
        else:
            if self._is_active(row.get("Bearish_BOS")):
                probability += 8.0
            if self._is_active(row.get("Bearish_CHOCH")):
                probability += 6.0
            if self._is_active(row.get("Buy_Liquidity_Sweep")):
                probability += 10.0

        type_bonus = {
            TargetType.NEAREST_LIQUIDITY: 14.0,
            TargetType.MAJOR_LIQUIDITY: 10.0,
            TargetType.HTF_LIQUIDITY: 6.0,
            TargetType.SWING_HIGH: 8.0,
            TargetType.SWING_LOW: 8.0,
            TargetType.STRUCTURE_TARGET: 9.0,
        }
        probability += type_bonus.get(target_type, 0.0)

        if target_type == TargetType.HTF_LIQUIDITY:
            if self._htf_trend_aligned(decision):
                probability += 10.0
            if self._htf_opposed(decision):
                probability -= 8.0

        if expected_rr > 4.0:
            probability -= min((expected_rr - 4.0) * 4.0, 12.0)
        elif expected_rr < 1.0:
            probability -= 10.0

        confidence = self._to_float(row.get("Confidence")) or 0.0
        probability += confidence * 12.0

        return round(max(5.0, min(probability, 95.0)), 1)

    def _build_optimized_target(
        self,
        row: pd.Series,
        decision: str,
        entry: float,
        stop_loss: float,
        price: float,
        target_type: TargetType,
        detail: str,
    ) -> OptimizedTarget:
        expected_rr = self._compute_rr(entry, stop_loss, price, decision)
        probability = self._estimate_probability(row, decision, target_type, expected_rr)
        reasoning = (
            f"{target_type.value}: {detail}; price={price:.2f}; "
            f"expected_rr={expected_rr}; probability={probability}%"
        )
        return OptimizedTarget(
            target_price=round(price, 2),
            target_type=target_type.value,
            target_probability=probability,
            expected_rr=expected_rr,
            reasoning=reasoning,
        )

    def _dedupe_candidates(
        self,
        candidates: list[tuple[float, TargetType, str]],
        decision: str,
    ) -> list[tuple[float, TargetType, str]]:
        seen: dict[float, tuple[float, TargetType, str]] = {}
        type_rank = {
            TargetType.MAJOR_LIQUIDITY: 6,
            TargetType.STRUCTURE_TARGET: 5,
            TargetType.HTF_LIQUIDITY: 4,
            TargetType.NEAREST_LIQUIDITY: 3,
            TargetType.SWING_HIGH: 2,
            TargetType.SWING_LOW: 2,
        }
        for price, target_type, detail in candidates:
            key = round(price, 2)
            existing = seen.get(key)
            if existing is None or type_rank.get(target_type, 0) > type_rank.get(existing[1], 0):
                seen[key] = (price, target_type, detail)

        ordered = sorted(seen.values(), key=lambda item: item[0], reverse=(decision == "SELL"))
        return ordered

    def _rank_targets(
        self,
        row: pd.Series,
        decision: str,
        entry: float,
        stop_loss: float,
        candidates: list[tuple[float, TargetType, str]],
    ) -> list[OptimizedTarget]:
        ranked: list[OptimizedTarget] = []
        for price, target_type, detail in candidates:
            target = self._build_optimized_target(
                row, decision, entry, stop_loss, price, target_type, detail
            )
            if target.expected_rr > 0:
                ranked.append(target)

        ranked.sort(
            key=lambda item: item.target_probability * item.expected_rr,
            reverse=True,
        )
        return ranked

    def _select_target_path(
        self,
        ranked: list[OptimizedTarget],
        entry: float,
        stop_loss: float,
        decision: str,
    ) -> list[OptimizedTarget]:
        if not ranked:
            return []

        risk = max(abs(entry - stop_loss), MIN_RISK_POINTS)
        selected: list[OptimizedTarget] = []
        used_prices: list[float] = []

        for target in ranked:
            if len(selected) >= 3:
                break
            if any(abs(target.target_price - used) < risk * MIN_TARGET_SEPARATION_R for used in used_prices):
                continue
            selected.append(target)
            used_prices.append(target.target_price)

        if decision == TradeDecision.BUY.value:
            selected.sort(key=lambda item: item.target_price)
        else:
            selected.sort(key=lambda item: item.target_price, reverse=True)
        return selected

    def analyze_signal(
        self,
        frame: pd.DataFrame,
        index: int,
        decision: str,
        entry: float,
        stop_loss: float,
        signal_date: str,
    ) -> SignalTargetAnalysis:
        """Discover, rank, and select targets for one signal."""
        row = frame.iloc[index]
        if decision == TradeDecision.BUY.value:
            raw = self._collect_buy_candidates(frame, index, entry)
        else:
            raw = self._collect_sell_candidates(frame, index, entry)

        candidates = self._dedupe_candidates(raw, decision)
        ranked = self._rank_targets(row, decision, entry, stop_loss, candidates)
        selected = self._select_target_path(ranked, entry, stop_loss, decision)
        risk = round(max(abs(entry - stop_loss), MIN_RISK_POINTS), 2)

        avg_prob = (
            round(sum(item.target_probability for item in selected) / len(selected), 1)
            if selected
            else 0.0
        )
        avg_rr = (
            round(sum(item.expected_rr for item in selected) / len(selected), 2)
            if selected
            else 0.0
        )

        return SignalTargetAnalysis(
            signal_index=index,
            signal_date=signal_date,
            decision=decision,
            entry=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            risk_points=risk,
            target_path=[item.as_dict() for item in ranked[:10]],
            selected_targets=[item.as_dict() for item in selected],
            average_probability=avg_prob,
            average_expected_rr=avg_rr,
        )

    def optimize(
        self,
        frame: pd.DataFrame,
        trade_plans: list[dict[str, Any]] | None = None,
        mtf_report: dict[str, Any] | None = None,
    ) -> TargetOptimizerReport:
        """
        Run target optimization for all actionable trade signals.

        Parameters
        ----------
        frame : pd.DataFrame
            Evaluated pipeline dataframe with decision columns.
        trade_plans : list[dict[str, Any]] | None, optional
            Trade plan V2 entries providing entry and stop loss.
        mtf_report : dict[str, Any] | None, optional
            Multi-timeframe report for HTF weighting.

        Returns
        -------
        TargetOptimizerReport
            Aggregate optimization report.
        """
        started = time.perf_counter()
        if mtf_report is not None:
            self._mtf_report = mtf_report
        elif self._mtf_report is None:
            self.load_mtf_report()

        plans = trade_plans if trade_plans is not None else self.load_trade_plans()
        plan_by_index = {int(plan["signal_index"]): plan for plan in plans}

        analyses: list[SignalTargetAnalysis] = []
        for index, row in frame.iterrows():
            decision = str(row.get("Decision"))
            if decision not in {TradeDecision.BUY.value, TradeDecision.SELL.value}:
                continue

            plan = plan_by_index.get(int(index))
            if plan is not None:
                entry = float(plan["entry"])
                stop_loss = float(plan["stop_loss"])
                signal_date = str(plan["signal_date"])
            else:
                entry = self._to_float(row.get("Close")) or 0.0
                signal_date = str(row.get("Date"))
                if decision == TradeDecision.BUY.value:
                    swing = self._to_float(row.get("Swing_Low"))
                    stop_loss = (swing if swing is not None else entry * 0.995) - 5.0
                else:
                    swing = self._to_float(row.get("Swing_High"))
                    stop_loss = (swing if swing is not None else entry * 1.005) + 5.0

            analyses.append(
                self.analyze_signal(frame, int(index), decision, entry, stop_loss, signal_date)
            )

        all_targets: list[dict[str, Any]] = []
        for analysis in analyses:
            for target in analysis.target_path:
                enriched = dict(target)
                enriched["signal_index"] = analysis.signal_index
                enriched["signal_date"] = analysis.signal_date
                enriched["decision"] = analysis.decision
                enriched["score"] = round(
                    target["target_probability"] * target["expected_rr"],
                    2,
                )
                all_targets.append(enriched)

        top_targets = sorted(all_targets, key=lambda item: item["score"], reverse=True)[:10]

        avg_prob = (
            round(sum(item.average_probability for item in analyses) / len(analyses), 1)
            if analyses
            else 0.0
        )
        avg_rr = (
            round(sum(item.average_expected_rr for item in analyses) / len(analyses), 2)
            if analyses
            else 0.0
        )

        elapsed = time.perf_counter() - started
        return TargetOptimizerReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=str(DEFAULT_PIPELINE_CSV),
            source_mtf_report=str(DEFAULT_MTF_REPORT),
            source_trade_plan_report=str(DEFAULT_TRADE_PLAN_V2),
            total_signals=len(analyses),
            average_target_probability=avg_prob,
            average_expected_rr=avg_rr,
            execution_time_seconds=elapsed,
            signal_analyses=[item.as_dict() for item in analyses],
            top_targets=top_targets,
        )


def generate_target_optimizer_report(
    pipeline_csv: Path | str | None = None,
    mtf_report_path: Path | str | None = None,
    trade_plan_path: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
) -> TargetOptimizerReport:
    """Run target optimization and export the JSON report."""
    csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    mtf_path = Path(mtf_report_path) if mtf_report_path is not None else DEFAULT_MTF_REPORT
    plan_path = Path(trade_plan_path) if trade_plan_path is not None else DEFAULT_TRADE_PLAN_V2
    json_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    evaluated, _ = evaluate_pipeline(pipeline_csv=csv_path)
    optimizer = TargetOptimizer(symbol=symbol, timeframe=timeframe)
    mtf_report = optimizer.load_mtf_report(mtf_path)
    trade_plans = optimizer.load_trade_plans(plan_path)
    report = optimizer.optimize(evaluated, trade_plans=trade_plans, mtf_report=mtf_report)
    report.source_csv = str(csv_path)
    report.source_mtf_report = str(mtf_path)
    report.source_trade_plan_report = str(plan_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Target optimization completed: signals=%s avg_prob=%s avg_rr=%s",
        report.total_signals,
        report.average_target_probability,
        report.average_expected_rr,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_target_optimizer_report()
        print("Target Optimizer Summary")
        print(f"Total Signals: {report.total_signals}")
        print(f"Average Target Probability: {report.average_target_probability}%")
        print(f"Average Expected RR: {report.average_expected_rr}")
        print("Top Targets:")
        for rank, target in enumerate(report.top_targets[:5], start=1):
            print(
                f"  {rank}. {target['target_type']} @ {target['target_price']} | "
                f"prob={target['target_probability']}% rr={target['expected_rr']} "
                f"score={target['score']}"
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except (TargetOptimizerError, DecisionEngineError) as exc:
        logger.error("Target optimizer error: %s", exc)
        print(f"Target optimizer error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected target optimizer failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
