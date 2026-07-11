"""
SmartMoneyEngine backtesting engine.

Evaluates trade plans from the signals layer against historical OHLCV candles
without modifying strategy logic or SMC detectors.
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

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRADE_PLAN_REPORT = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "backtesting"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "backtest_report.json"
DEFAULT_RESULTS_CSV = DEFAULT_OUTPUT_DIR / "trade_results.csv"

MAX_FORWARD_BARS = 400
ENTRY_TOLERANCE = 0.05


class BacktestEngineError(Exception):
    """Raised when backtesting fails."""


class TradeOutcome(str, Enum):
    """Closed trade classification."""

    WIN = "Win"
    LOSS = "Loss"
    BREAKEVEN = "Breakeven"
    OPEN = "Open"
    NO_ENTRY = "No Entry"


class ExitReason(str, Enum):
    """Reason a simulated trade stopped."""

    NO_ENTRY = "No Entry"
    STOP_LOSS = "Stop Loss"
    TARGET_1 = "Target 1"
    TARGET_2 = "Target 2"
    OPEN = "Open"


@dataclass
class TradeResult:
    """Per-trade backtest outcome."""

    signal_index: int
    signal_date: str
    decision: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    planned_risk_reward: float
    confidence_pct: float
    trade_validity: bool
    entry_hit: bool
    entry_timestamp: str | None
    sl_hit: bool
    target_1_hit: bool
    target_2_hit: bool
    exit_reason: str
    outcome: str
    trade_duration_bars: int
    trade_duration_minutes: int
    mfe_points: float
    mae_points: float
    realized_pnl_points: float
    realized_rr: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable trade result dictionary."""
        return asdict(self)


@dataclass
class BacktestReport:
    """Aggregate backtest summary."""

    symbol: str
    timeframe: str
    source_trade_plan_report: str
    source_data_start: str
    source_data_end: str
    total_trade_plans: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    open_trades: int
    no_entry_trades: int
    win_rate_pct: float
    average_rr: float
    profit_factor: float | None
    maximum_drawdown_points: float
    average_holding_time_minutes: float
    execution_time_seconds: float
    trade_results: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return asdict(self)


class BacktestEngine:
    """
    Simulate trade plans against historical OHLCV data.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting and data loading.
    timeframe : str, optional
        Candle timeframe for historical data.
    max_forward_bars : int, optional
        Maximum bars to scan for entry and trade management after signal.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        max_forward_bars: int = MAX_FORWARD_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_forward_bars = max_forward_bars

    @staticmethod
    def _load_trade_plan_report(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise BacktestEngineError(f"Trade plan report not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "trade_plans" not in payload:
            raise BacktestEngineError("Trade plan report missing 'trade_plans'.")
        return payload

    @staticmethod
    def _normalize_timeframe(value: str) -> str:
        normalized = value.strip().lower()
        replacements = {
            "5-minute": "5",
            "5min": "5",
            "5m": "5",
        }
        return replacements.get(normalized, value.strip())

    @staticmethod
    def _find_signal_index(frame: pd.DataFrame, signal_date: str) -> int | None:
        signal_ts = pd.Timestamp(signal_date)
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize("Asia/Kolkata")
        else:
            signal_ts = signal_ts.tz_convert("Asia/Kolkata")

        matches = frame.index[frame["timestamp"] == signal_ts].tolist()
        if matches:
            return int(matches[0])

        later = frame.index[frame["timestamp"] > signal_ts].tolist()
        if later:
            return int(later[0])
        return None

    @staticmethod
    def _entry_hit(decision: str, bar: pd.Series, entry_price: float) -> bool:
        if decision == "BUY":
            return float(bar["low"]) <= entry_price + ENTRY_TOLERANCE
        if decision == "SELL":
            return float(bar["high"]) >= entry_price - ENTRY_TOLERANCE
        return False

    @staticmethod
    def _risk_points(entry_price: float, stop_loss: float) -> float:
        risk = abs(entry_price - stop_loss)
        return risk if risk > 0 else 1e-9

    @staticmethod
    def _resolve_long_exit(
        bar: pd.Series,
        stop_loss: float,
        target_1: float,
        target_2: float,
    ) -> ExitReason | None:
        low = float(bar["low"])
        high = float(bar["high"])
        if low <= stop_loss:
            return ExitReason.STOP_LOSS
        if high >= target_2:
            return ExitReason.TARGET_2
        if high >= target_1:
            return ExitReason.TARGET_1
        return None

    @staticmethod
    def _resolve_short_exit(
        bar: pd.Series,
        stop_loss: float,
        target_1: float,
        target_2: float,
    ) -> ExitReason | None:
        high = float(bar["high"])
        low = float(bar["low"])
        if high >= stop_loss:
            return ExitReason.STOP_LOSS
        if low <= target_2:
            return ExitReason.TARGET_2
        if low <= target_1:
            return ExitReason.TARGET_1
        return None

    @staticmethod
    def _realized_pnl(
        decision: str,
        entry_price: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        exit_reason: ExitReason,
    ) -> float:
        if exit_reason == ExitReason.NO_ENTRY:
            return 0.0
        if exit_reason == ExitReason.OPEN:
            return 0.0
        if exit_reason == ExitReason.STOP_LOSS:
            return -(BacktestEngine._risk_points(entry_price, stop_loss))

        if decision == "BUY":
            if exit_reason == ExitReason.TARGET_2:
                return target_2 - entry_price
            if exit_reason == ExitReason.TARGET_1:
                return target_1 - entry_price
        elif decision == "SELL":
            if exit_reason == ExitReason.TARGET_2:
                return entry_price - target_2
            if exit_reason == ExitReason.TARGET_1:
                return entry_price - target_1
        return 0.0

    @staticmethod
    def _classify_outcome(exit_reason: ExitReason, realized_pnl: float) -> TradeOutcome:
        if exit_reason == ExitReason.NO_ENTRY:
            return TradeOutcome.NO_ENTRY
        if exit_reason == ExitReason.OPEN:
            return TradeOutcome.OPEN
        if exit_reason in {ExitReason.TARGET_1, ExitReason.TARGET_2}:
            return TradeOutcome.WIN
        if exit_reason == ExitReason.STOP_LOSS:
            if abs(realized_pnl) <= ENTRY_TOLERANCE:
                return TradeOutcome.BREAKEVEN
            return TradeOutcome.LOSS
        return TradeOutcome.BREAKEVEN

    def _simulate_trade(
        self,
        plan: dict[str, Any],
        frame: pd.DataFrame,
    ) -> TradeResult:
        signal_index = int(plan["signal_index"])
        signal_date = str(plan["signal_date"])
        decision = str(plan["decision"])
        entry_price = float(plan["entry_price"])
        stop_loss = float(plan["stop_loss"])
        target_1 = float(plan["target_1"])
        target_2 = float(plan["target_2"])
        planned_rr = float(plan["risk_reward_ratio"])
        confidence_pct = float(plan["confidence_pct"])
        trade_validity = bool(plan["trade_validity"])

        start_idx = self._find_signal_index(frame, signal_date)
        if start_idx is None:
            raise BacktestEngineError(
                f"Signal timestamp not found in historical data: {signal_date}"
            )

        end_idx = min(len(frame) - 1, start_idx + self.max_forward_bars)
        entry_idx: int | None = None
        entry_timestamp: str | None = None

        for idx in range(start_idx, end_idx + 1):
            if self._entry_hit(decision, frame.iloc[idx], entry_price):
                entry_idx = idx
                entry_timestamp = frame.iloc[idx]["timestamp"].isoformat()
                break

        if entry_idx is None:
            return TradeResult(
                signal_index=signal_index,
                signal_date=signal_date,
                decision=decision,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                planned_risk_reward=planned_rr,
                confidence_pct=confidence_pct,
                trade_validity=trade_validity,
                entry_hit=False,
                entry_timestamp=None,
                sl_hit=False,
                target_1_hit=False,
                target_2_hit=False,
                exit_reason=ExitReason.NO_ENTRY.value,
                outcome=TradeOutcome.NO_ENTRY.value,
                trade_duration_bars=0,
                trade_duration_minutes=0,
                mfe_points=0.0,
                mae_points=0.0,
                realized_pnl_points=0.0,
                realized_rr=0.0,
            )

        sl_hit = False
        target_1_hit = False
        target_2_hit = False
        exit_reason = ExitReason.OPEN
        exit_idx = end_idx
        mfe_points = 0.0
        mae_points = 0.0

        for idx in range(entry_idx, end_idx + 1):
            bar = frame.iloc[idx]
            high = float(bar["high"])
            low = float(bar["low"])

            if decision == "BUY":
                mfe_points = max(mfe_points, high - entry_price)
                mae_points = max(mae_points, entry_price - low)
                bar_exit = self._resolve_long_exit(bar, stop_loss, target_1, target_2)
            else:
                mfe_points = max(mfe_points, entry_price - low)
                mae_points = max(mae_points, high - entry_price)
                bar_exit = self._resolve_short_exit(bar, stop_loss, target_1, target_2)

            if bar_exit == ExitReason.STOP_LOSS:
                sl_hit = True
                exit_reason = ExitReason.STOP_LOSS
                exit_idx = idx
                break
            if bar_exit == ExitReason.TARGET_2:
                target_2_hit = True
                target_1_hit = True
                exit_reason = ExitReason.TARGET_2
                exit_idx = idx
                break
            if bar_exit == ExitReason.TARGET_1:
                target_1_hit = True
                exit_reason = ExitReason.TARGET_1
                exit_idx = idx
                break

        realized_pnl = self._realized_pnl(
            decision,
            entry_price,
            stop_loss,
            target_1,
            target_2,
            exit_reason,
        )
        risk = self._risk_points(entry_price, stop_loss)
        realized_rr = realized_pnl / risk if exit_reason not in {
            ExitReason.NO_ENTRY,
            ExitReason.OPEN,
        } else 0.0
        outcome = self._classify_outcome(exit_reason, realized_pnl)

        duration_bars = max(0, exit_idx - entry_idx)
        duration_minutes = duration_bars * int(self._normalize_timeframe(self.timeframe) or "5")

        return TradeResult(
            signal_index=signal_index,
            signal_date=signal_date,
            decision=decision,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            planned_risk_reward=planned_rr,
            confidence_pct=confidence_pct,
            trade_validity=trade_validity,
            entry_hit=True,
            entry_timestamp=entry_timestamp,
            sl_hit=sl_hit,
            target_1_hit=target_1_hit,
            target_2_hit=target_2_hit,
            exit_reason=exit_reason.value,
            outcome=outcome.value,
            trade_duration_bars=duration_bars,
            trade_duration_minutes=duration_minutes,
            mfe_points=round(mfe_points, 2),
            mae_points=round(mae_points, 2),
            realized_pnl_points=round(realized_pnl, 2),
            realized_rr=round(realized_rr, 2),
        )

    @staticmethod
    def _maximum_drawdown(pnl_series: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in pnl_series:
            equity += pnl
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
        return round(max_drawdown, 2)

    @staticmethod
    def _profit_factor(closed_pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in closed_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in closed_pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    def run(
        self,
        trade_plan_report: Path | str | None = None,
        ohlcv: pd.DataFrame | None = None,
    ) -> BacktestReport:
        """
        Backtest all trade plans in the report against OHLCV candles.

        Parameters
        ----------
        trade_plan_report : Path | str | None, optional
            Path to ``trade_plan_report.json``.
        ohlcv : pd.DataFrame | None, optional
            Preloaded OHLCV data. Loaded from ``HistoricalDataLoader`` when omitted.

        Returns
        -------
        BacktestReport
            Aggregate backtest report with per-trade analysis.
        """
        started = time.perf_counter()
        report_path = (
            Path(trade_plan_report)
            if trade_plan_report is not None
            else DEFAULT_TRADE_PLAN_REPORT
        )
        payload = self._load_trade_plan_report(report_path)
        plans = payload["trade_plans"]
        symbol = str(payload.get("symbol", self.symbol))
        timeframe = self._normalize_timeframe(str(payload.get("timeframe", self.timeframe)))

        if ohlcv is None:
            signal_dates = [pd.Timestamp(plan["signal_date"]) for plan in plans]
            start = min(signal_dates).date() - timedelta(days=1)
            end = max(signal_dates).date() + timedelta(days=10)
            loader = HistoricalDataLoader()
            ohlcv = loader.load(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start,
                end_date=end,
                prefer_parquet=True,
            )
        else:
            working = ohlcv.copy()
            if "timestamp" not in working.columns:
                raise BacktestEngineError("OHLCV data must include a timestamp column.")
            working["timestamp"] = pd.to_datetime(working["timestamp"])
            ohlcv = working.sort_values("timestamp").reset_index(drop=True)

        results: list[TradeResult] = []
        for plan in plans:
            logger.info(
                "Backtesting signal %s (%s) at %s",
                plan["signal_index"],
                plan["decision"],
                plan["signal_date"],
            )
            results.append(self._simulate_trade(plan, ohlcv))

        executed = [item for item in results if item.entry_hit]
        closed = [
            item
            for item in executed
            if item.outcome in {
                TradeOutcome.WIN.value,
                TradeOutcome.LOSS.value,
                TradeOutcome.BREAKEVEN.value,
            }
        ]

        winning = sum(1 for item in closed if item.outcome == TradeOutcome.WIN.value)
        losing = sum(1 for item in closed if item.outcome == TradeOutcome.LOSS.value)
        breakeven = sum(1 for item in closed if item.outcome == TradeOutcome.BREAKEVEN.value)
        open_trades = sum(1 for item in executed if item.outcome == TradeOutcome.OPEN.value)
        no_entry = sum(1 for item in results if item.outcome == TradeOutcome.NO_ENTRY.value)

        win_rate = round((winning / len(closed) * 100) if closed else 0.0, 2)
        average_rr = round(
            sum(item.realized_rr for item in closed) / len(closed) if closed else 0.0,
            2,
        )
        closed_pnls = [item.realized_pnl_points for item in closed]
        profit_factor = self._profit_factor(closed_pnls)
        max_drawdown = self._maximum_drawdown(closed_pnls)
        average_holding = round(
            sum(item.trade_duration_minutes for item in closed) / len(closed)
            if closed
            else 0.0,
            2,
        )

        elapsed = time.perf_counter() - started
        return BacktestReport(
            symbol=symbol,
            timeframe=timeframe,
            source_trade_plan_report=str(report_path),
            source_data_start=ohlcv["timestamp"].iloc[0].isoformat(),
            source_data_end=ohlcv["timestamp"].iloc[-1].isoformat(),
            total_trade_plans=len(plans),
            total_trades=len(executed),
            winning_trades=winning,
            losing_trades=losing,
            breakeven_trades=breakeven,
            open_trades=open_trades,
            no_entry_trades=no_entry,
            win_rate_pct=win_rate,
            average_rr=average_rr,
            profit_factor=profit_factor,
            maximum_drawdown_points=max_drawdown,
            average_holding_time_minutes=average_holding,
            execution_time_seconds=elapsed,
            trade_results=[item.as_dict() for item in results],
        )


def run_backtest(
    trade_plan_report: Path | str | None = None,
    output_dir: Path | str | None = None,
    ohlcv: pd.DataFrame | None = None,
) -> BacktestReport:
    """Run backtesting and export JSON/CSV outputs."""
    engine = BacktestEngine()
    report = engine.run(trade_plan_report=trade_plan_report, ohlcv=ohlcv)

    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / "backtest_report.json"
    csv_path = destination / "trade_results.csv"

    serializable = report.as_dict()
    if serializable["profit_factor"] == float("inf"):
        serializable["profit_factor"] = "inf"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    results_frame = pd.DataFrame(report.trade_results)
    results_frame.to_csv(csv_path, index=False)

    logger.info(
        "Backtest completed: trades=%s win_rate=%s avg_rr=%s profit_factor=%s",
        report.total_trades,
        report.win_rate_pct,
        report.average_rr,
        report.profit_factor,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = run_backtest()
        profit_factor_display = (
            "inf" if report.profit_factor == float("inf") else report.profit_factor
        )
        if report.profit_factor is None:
            profit_factor_display = "N/A"

        print("Backtesting Summary")
        print(f"Total Trades: {report.total_trades}")
        print(f"Win Rate: {report.win_rate_pct}%")
        print(f"Average RR: {report.average_rr}")
        print(f"Profit Factor: {profit_factor_display}")
        print(f"Max Drawdown: {report.maximum_drawdown_points}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        print(f"Trade Results: {DEFAULT_RESULTS_CSV}")
        return 0
    except BacktestEngineError as exc:
        logger.error("Backtest engine error: %s", exc)
        print(f"Backtest engine error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected backtest engine failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
