"""
SmartMoneyEngine trade plan engine.

Converts BUY/SELL decision signals into complete SMC-based trade plans
using pipeline outputs from the decision layer and detector columns.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.signals.decision_engine import (
    DecisionEngine,
    DecisionEngineError,
    TradeDecision,
    evaluate_pipeline,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report.json"

MIN_RISK_REWARD = 1.5
SL_BUFFER_POINTS = 5.0
LOOKBACK_BARS = 200
FORWARD_SCAN_BARS = 400


class TradePlanEngineError(Exception):
    """Raised when trade plan generation fails."""


@dataclass
class TradePlan:
    """Complete trade plan for a decision signal."""

    signal_index: int
    signal_date: str
    decision: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward_ratio: float
    confidence_pct: float
    trade_validity: bool
    reason: str
    invalid_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable trade plan dictionary."""
        return asdict(self)


@dataclass
class TradePlanReport:
    """Aggregate trade plan report."""

    symbol: str
    timeframe: str
    source_csv: str
    total_signals: int
    valid_trade_plans: int
    invalid_trade_plans: int
    average_rr: float
    average_confidence: float
    execution_time_seconds: float
    trade_plans: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return asdict(self)


class TradePlanEngine:
    """
    Build SMC trade plans from decision-engine outputs.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Timeframe label for reporting.
    min_risk_reward : float, optional
        Minimum acceptable risk-reward ratio for validity.
    sl_buffer_points : float, optional
        Stop-loss buffer below/above structure in index points.
    lookback_bars : int, optional
        Bars to search backward for OB/FVG/swing context.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        min_risk_reward: float = MIN_RISK_REWARD,
        sl_buffer_points: float = SL_BUFFER_POINTS,
        lookback_bars: int = LOOKBACK_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_risk_reward = min_risk_reward
        self.sl_buffer_points = sl_buffer_points
        self.lookback_bars = lookback_bars

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
        if not TradePlanEngine._is_active(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _window_start(self, index: int) -> int:
        return max(0, index - self.lookback_bars)

    def _find_recent_bullish_ob(
        self,
        frame: pd.DataFrame,
        index: int,
    ) -> tuple[float, float] | None:
        """Return the most recent active bullish order block zone."""
        start = self._window_start(index)
        window = frame.iloc[start:index + 1]
        candidates = window[
            window["Bullish_OB_High"].notna() & window["Bullish_OB_Low"].notna()
        ]
        if candidates.empty:
            return None

        for offset in range(len(candidates) - 1, -1, -1):
            row = candidates.iloc[offset]
            if self._is_active(row.get("Bullish_OB_Mitigated")):
                continue
            low = self._to_float(row.get("Bullish_OB_Low"))
            high = self._to_float(row.get("Bullish_OB_High"))
            if low is not None and high is not None:
                return low, high
        return None

    def _find_recent_bearish_ob(
        self,
        frame: pd.DataFrame,
        index: int,
    ) -> tuple[float, float] | None:
        """Return the most recent active bearish order block zone."""
        start = self._window_start(index)
        window = frame.iloc[start:index + 1]
        candidates = window[
            window["Bearish_OB_High"].notna() & window["Bearish_OB_Low"].notna()
        ]
        if candidates.empty:
            return None

        for offset in range(len(candidates) - 1, -1, -1):
            row = candidates.iloc[offset]
            if self._is_active(row.get("Bearish_OB_Mitigated")):
                continue
            low = self._to_float(row.get("Bearish_OB_Low"))
            high = self._to_float(row.get("Bearish_OB_High"))
            if low is not None and high is not None:
                return low, high
        return None

    def _find_recent_bullish_fvg(
        self,
        frame: pd.DataFrame,
        index: int,
    ) -> tuple[float, float] | None:
        """Return the most recent bullish fair value gap zone."""
        start = self._window_start(index)
        window = frame.iloc[start:index + 1]
        candidates = window[
            window["Bullish_FVG_Top"].notna() & window["Bullish_FVG_Bottom"].notna()
        ]
        if candidates.empty:
            return None
        row = candidates.iloc[-1]
        bottom = self._to_float(row.get("Bullish_FVG_Bottom"))
        top = self._to_float(row.get("Bullish_FVG_Top"))
        if bottom is not None and top is not None:
            return bottom, top
        return None

    def _find_recent_bearish_fvg(
        self,
        frame: pd.DataFrame,
        index: int,
    ) -> tuple[float, float] | None:
        """Return the most recent bearish fair value gap zone."""
        start = self._window_start(index)
        window = frame.iloc[start:index + 1]
        candidates = window[
            window["Bearish_FVG_Top"].notna() & window["Bearish_FVG_Bottom"].notna()
        ]
        if candidates.empty:
            return None
        row = candidates.iloc[-1]
        bottom = self._to_float(row.get("Bearish_FVG_Bottom"))
        top = self._to_float(row.get("Bearish_FVG_Top"))
        if bottom is not None and top is not None:
            return bottom, top
        return None

    def _nearest_swing_low(self, frame: pd.DataFrame, index: int) -> float | None:
        start = self._window_start(index)
        values = [
            self._to_float(value)
            for value in frame.iloc[start:index + 1]["Swing_Low"]
            if self._to_float(value) is not None
        ]
        return values[-1] if values else None

    def _nearest_swing_high(self, frame: pd.DataFrame, index: int) -> float | None:
        start = self._window_start(index)
        values = [
            self._to_float(value)
            for value in frame.iloc[start:index + 1]["Swing_High"]
            if self._to_float(value) is not None
        ]
        return values[-1] if values else None

    def _liquidity_levels_above(
        self,
        frame: pd.DataFrame,
        index: int,
        entry: float,
    ) -> list[float]:
        end = min(len(frame), index + FORWARD_SCAN_BARS)
        levels = {
            level
            for level in (
                self._to_float(value)
                for value in frame.iloc[index:end]["Buy_Side_Liquidity"]
            )
            if level is not None and level > entry
        }
        swing_highs = {
            level
            for level in (
                self._to_float(value)
                for value in frame.iloc[index:end]["Swing_High"]
            )
            if level is not None and level > entry
        }
        return sorted(levels.union(swing_highs))

    def _liquidity_levels_below(
        self,
        frame: pd.DataFrame,
        index: int,
        entry: float,
    ) -> list[float]:
        end = min(len(frame), index + FORWARD_SCAN_BARS)
        levels = {
            level
            for level in (
                self._to_float(value)
                for value in frame.iloc[index:end]["Sell_Side_Liquidity"]
            )
            if level is not None and level < entry
        }
        swing_lows = {
            level
            for level in (
                self._to_float(value)
                for value in frame.iloc[index:end]["Swing_Low"]
            )
            if level is not None and level < entry
        }
        return sorted(levels.union(swing_lows), reverse=True)

    def _build_buy_plan(self, frame: pd.DataFrame, index: int, row: pd.Series) -> TradePlan:
        close = self._to_float(row.get("Close")) or 0.0
        ob = self._find_recent_bullish_ob(frame, index)
        fvg = self._find_recent_bullish_fvg(frame, index)
        swing_low = self._nearest_swing_low(frame, index)

        entry_source = "close"
        if ob is not None:
            entry = ob[1]
            entry_source = "bullish order block retest"
            sl_anchor = ob[0]
        elif fvg is not None:
            entry = fvg[1]
            entry_source = "bullish FVG demand zone"
            sl_anchor = fvg[0]
        else:
            entry = close
            entry_source = "signal close"
            sl_anchor = swing_low if swing_low is not None else close * 0.995

        if swing_low is not None:
            sl_anchor = min(sl_anchor, swing_low)
        stop_loss = sl_anchor - self.sl_buffer_points

        targets = self._liquidity_levels_above(frame, index, entry)
        if len(targets) >= 2:
            target_1, target_2 = targets[0], targets[1]
        elif len(targets) == 1:
            target_1 = targets[0]
            target_2 = target_1 + max(entry - stop_loss, 1.0) * self.min_risk_reward
        else:
            target_1 = entry + max(entry - stop_loss, 1.0) * self.min_risk_reward
            target_2 = target_1 + max(entry - stop_loss, 1.0)

        risk = entry - stop_loss
        reward = target_1 - entry
        rr = round(reward / risk, 2) if risk > 0 else 0.0
        confidence_pct = round(float(row.get("Confidence", 0.0)) * 100, 1)

        valid = risk > 0 and reward > 0 and stop_loss < entry < target_1 < target_2
        invalid_reason = None
        if not valid:
            invalid_reason = "Invalid BUY price structure."
        elif rr < self.min_risk_reward:
            valid = False
            invalid_reason = f"Risk-reward {rr} below minimum {self.min_risk_reward}."

        reason = (
            f"BUY plan from {entry_source}; SL below structure at {stop_loss:.2f}; "
            f"T1 liquidity/resistance {target_1:.2f}; T2 extension {target_2:.2f}; "
            f"{row.get('Reason', '')}"
        )

        return TradePlan(
            signal_index=index,
            signal_date=str(row.get("Date")),
            decision=TradeDecision.BUY.value,
            entry_price=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            risk_reward_ratio=rr,
            confidence_pct=confidence_pct,
            trade_validity=valid,
            reason=reason,
            invalid_reason=invalid_reason,
        )

    def _build_sell_plan(self, frame: pd.DataFrame, index: int, row: pd.Series) -> TradePlan:
        close = self._to_float(row.get("Close")) or 0.0
        ob = self._find_recent_bearish_ob(frame, index)
        fvg = self._find_recent_bearish_fvg(frame, index)
        swing_high = self._nearest_swing_high(frame, index)

        entry_source = "close"
        if ob is not None:
            entry = ob[0]
            entry_source = "bearish order block retest"
            sl_anchor = ob[1]
        elif fvg is not None:
            entry = fvg[0]
            entry_source = "bearish FVG supply zone"
            sl_anchor = fvg[1]
        else:
            entry = close
            entry_source = "signal close"
            sl_anchor = swing_high if swing_high is not None else close * 1.005

        if swing_high is not None:
            sl_anchor = max(sl_anchor, swing_high)
        stop_loss = sl_anchor + self.sl_buffer_points

        targets = self._liquidity_levels_below(frame, index, entry)
        if len(targets) >= 2:
            target_1, target_2 = targets[0], targets[1]
        elif len(targets) == 1:
            target_1 = targets[0]
            target_2 = target_1 - max(stop_loss - entry, 1.0) * self.min_risk_reward
        else:
            target_1 = entry - max(stop_loss - entry, 1.0) * self.min_risk_reward
            target_2 = target_1 - max(stop_loss - entry, 1.0)

        risk = stop_loss - entry
        reward = entry - target_1
        rr = round(reward / risk, 2) if risk > 0 else 0.0
        confidence_pct = round(float(row.get("Confidence", 0.0)) * 100, 1)

        valid = risk > 0 and reward > 0 and target_2 < target_1 < entry < stop_loss
        invalid_reason = None
        if not valid:
            invalid_reason = "Invalid SELL price structure."
        elif rr < self.min_risk_reward:
            valid = False
            invalid_reason = f"Risk-reward {rr} below minimum {self.min_risk_reward}."

        reason = (
            f"SELL plan from {entry_source}; SL above structure at {stop_loss:.2f}; "
            f"T1 liquidity/support {target_1:.2f}; T2 extension {target_2:.2f}; "
            f"{row.get('Reason', '')}"
        )

        return TradePlan(
            signal_index=index,
            signal_date=str(row.get("Date")),
            decision=TradeDecision.SELL.value,
            entry_price=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            risk_reward_ratio=rr,
            confidence_pct=confidence_pct,
            trade_validity=valid,
            reason=reason,
            invalid_reason=invalid_reason,
        )

    def build_plans(self, evaluated: pd.DataFrame) -> list[TradePlan]:
        """Build trade plans for all BUY/SELL signals."""
        required = {"Decision", "Close", "Confidence", "Reason"}
        missing = required - set(evaluated.columns)
        if missing:
            raise TradePlanEngineError(f"Evaluated dataframe missing columns: {sorted(missing)}")

        plans: list[TradePlan] = []
        for index, row in evaluated.iterrows():
            decision = str(row.get("Decision"))
            if decision == TradeDecision.BUY.value:
                plans.append(self._build_buy_plan(evaluated, int(index), row))
            elif decision == TradeDecision.SELL.value:
                plans.append(self._build_sell_plan(evaluated, int(index), row))
        return plans

    def build_report(
        self,
        plans: list[TradePlan],
        source_csv: Path | str,
        execution_time_seconds: float,
    ) -> TradePlanReport:
        """Build aggregate trade plan report."""
        valid_plans = [plan for plan in plans if plan.trade_validity]
        average_rr = (
            round(sum(plan.risk_reward_ratio for plan in valid_plans) / len(valid_plans), 2)
            if valid_plans
            else 0.0
        )
        average_confidence = (
            round(sum(plan.confidence_pct for plan in plans) / len(plans), 2)
            if plans
            else 0.0
        )

        return TradePlanReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=str(source_csv),
            total_signals=len(plans),
            valid_trade_plans=len(valid_plans),
            invalid_trade_plans=len(plans) - len(valid_plans),
            average_rr=average_rr,
            average_confidence=average_confidence,
            execution_time_seconds=execution_time_seconds,
            trade_plans=[plan.as_dict() for plan in plans],
        )


def generate_trade_plans(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
) -> tuple[list[TradePlan], TradePlanReport]:
    """
    Evaluate decisions and generate trade plans from real pipeline data.

    Returns
    -------
    tuple[list[TradePlan], TradePlanReport]
        Trade plans and aggregate report.
    """
    csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    json_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    started = time.perf_counter()
    evaluated, _ = evaluate_pipeline(pipeline_csv=csv_path)
    engine = TradePlanEngine(symbol=symbol, timeframe=timeframe)
    plans = engine.build_plans(evaluated)
    elapsed = time.perf_counter() - started
    report = engine.build_report(plans, csv_path, elapsed)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Trade plan generation completed in %.3fs: signals=%s valid=%s avg_rr=%s",
        elapsed,
        report.total_signals,
        report.valid_trade_plans,
        report.average_rr,
    )
    return plans, report


def main() -> int:
    """CLI entry point."""
    try:
        _, report = generate_trade_plans()
        print("Trade Plan Engine Summary")
        print(f"Total Signals: {report.total_signals}")
        print(f"Valid Trade Plans: {report.valid_trade_plans}")
        print(f"Average RR: {report.average_rr}")
        print(f"Average Confidence: {report.average_confidence}%")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except (TradePlanEngineError, DecisionEngineError) as exc:
        logger.error("Trade plan engine error: %s", exc)
        print(f"Trade plan engine error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected trade plan engine failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
