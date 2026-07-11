"""
Trigger-to-Trade Validation research for SmartMoneyEngine.

Converts institutional trigger models into simulated trade outcomes using
V1-aligned construction (Trigger Close entry, Structural Swing SL,
Opposite Liquidity target). Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine
from src.research.filter_research_engine import (
    DEFAULT_PIPELINE_DIR,
    FilterResearchEngine,
    RESEARCH_DAYS,
    _json_safe,
)
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS, TIMEFRAME_MINUTES
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tiered_signal_framework_research import (
    MIN_RISK_POINTS,
    RISK_LOOKBACK,
    SL_BUFFER_POINTS,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_TRIGGER_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "research" / "institutional_trigger_validation.json"
)
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "trigger_trade_validation.json"

FALLBACK_TARGET_R = 2.0
MIN_MODEL_SAMPLES = 5
MIN_PRODUCTION_SAMPLES = 50
TOP_RANK_COUNT = 20

ENTRY_MODEL = "trigger_close"
STOP_MODEL = "structural_swing"
TARGET_MODEL = "opposite_liquidity"


class TriggerTradeValidationError(Exception):
    """Raised when trigger-to-trade validation fails."""


@dataclass(frozen=True)
class TriggerTradeOutcome:
    """Simulated trade outcome for one trigger record."""

    symbol: str
    timeframe: str
    direction: str
    trigger_model: str
    trigger_timestamp: str
    trigger_bar: int
    entry_price: float
    stop_price: float
    target_price: float
    risk_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    holding_bars: int
    is_false_trigger: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriggerTradeMetrics:
    """Aggregate trade metrics for one trigger model."""

    trigger_model: str
    direction: str
    trades: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float
    false_trigger_rate_pct: float
    classification: str
    rank_profit: int = 0
    rank_accuracy: int = 0
    rank_danger: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriggerTradeValidationReport:
    """Full trigger-to-trade validation output."""

    source_report: str
    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    total_trigger_records: int
    trades_simulated: int
    trades_skipped: int
    entry_model: str
    stop_model: str
    target_model: str
    trigger_trade_metrics: list[dict[str, Any]]
    production_trigger_matrix: list[dict[str, Any]]
    most_profitable_triggers: list[dict[str, Any]]
    most_accurate_triggers: list[dict[str, Any]]
    most_dangerous_triggers: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TriggerTradeValidationResearch:
    """Convert institutional trigger models into trade-level validation."""

    def __init__(
        self,
        trigger_report_path: Path | str | None = None,
        research_days: int = RESEARCH_DAYS,
    ) -> None:
        self.trigger_report_path = Path(trigger_report_path or DEFAULT_TRIGGER_REPORT_PATH)
        self.research_days = research_days
        self._frame_cache: dict[tuple[str, str], pd.DataFrame] = {}

    @staticmethod
    def _is_active(value: Any) -> bool:
        return LiquidityNarrativeEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        return LiquidityNarrativeEngine._to_float(value)

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return Tier2ProductionValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _minutes_per_bar(timeframe: str) -> int:
        return TIMEFRAME_MINUTES.get(timeframe.upper(), 5)

    @staticmethod
    def _classify_trigger(
        trades: int,
        win_rate_pct: float,
        expectancy: float,
        profit_factor: float | None,
    ) -> str:
        if trades < MIN_MODEL_SAMPLES:
            return "Reject"
        if expectancy < 0:
            return "Reject"
        if (
            trades >= MIN_PRODUCTION_SAMPLES
            and win_rate_pct >= 40.0
            and expectancy >= 50.0
            and profit_factor is not None
            and profit_factor >= 1.5
        ):
            return "Production Ready"
        if trades >= MIN_MODEL_SAMPLES and expectancy > 0:
            return "Needs Validation"
        return "Reject"

    def _load_trigger_report(self) -> dict[str, Any]:
        if not self.trigger_report_path.exists():
            raise TriggerTradeValidationError(
                f"Institutional trigger validation report not found: {self.trigger_report_path}",
            )
        with self.trigger_report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_frame(
        self,
        symbol: str,
        timeframe: str,
        metadata: dict[str, Any],
    ) -> pd.DataFrame | None:
        key = (symbol, timeframe)
        if key in self._frame_cache:
            return self._frame_cache[key]

        end = date.fromisoformat(metadata["end_date"]) if metadata.get("end_date") else date.today()
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        engine = FilterResearchEngine(symbol=symbol, research_days=self.research_days)
        path = engine._pipeline_path(timeframe)
        if not path.exists():
            try:
                path = engine._ensure_pipeline(timeframe, start, end)
            except Exception as exc:
                logger.warning("Skipping %s/%s pipeline: %s", symbol, timeframe, exc)
                return None

        frame = pd.read_csv(path).reset_index(drop=True)
        self._frame_cache[key] = frame
        return frame

    @staticmethod
    def _resolve_bar(frame: pd.DataFrame, record: dict[str, Any]) -> int | None:
        trigger_bar = record.get("trigger_bar")
        if isinstance(trigger_bar, int) and 0 <= trigger_bar < len(frame):
            return trigger_bar

        timestamp = record.get("trigger_timestamp")
        if not timestamp:
            return None

        matches = frame.index[frame["Date"].astype(str) == str(timestamp)].tolist()
        if matches:
            return int(matches[0])

        target = pd.Timestamp(timestamp)
        parsed = pd.to_datetime(frame["Date"], errors="coerce")
        deltas = (parsed - target).abs()
        if deltas.notna().any():
            return int(deltas.idxmin())
        return None

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
            return round(entry_price + risk * FALLBACK_TARGET_R, 2)

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
        return round(entry_price - risk * FALLBACK_TARGET_R, 2)

    def _simulate_trade(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
        stop: float,
        target: float,
        risk: float,
    ) -> tuple[float, float, bool, int]:
        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)

        for index in range(entry_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])

            if direction == "bullish":
                if bar_low <= stop:
                    return -risk, -1.0, False, index - entry_bar
                if bar_high >= target:
                    pnl = round(target - entry_price, 2)
                    rr = round(pnl / risk, 2) if risk > 0 else 0.0
                    return pnl, rr, True, index - entry_bar
            else:
                if bar_high >= stop:
                    return -risk, -1.0, False, index - entry_bar
                if bar_low <= target:
                    pnl = round(entry_price - target, 2)
                    rr = round(pnl / risk, 2) if risk > 0 else 0.0
                    return pnl, rr, True, index - entry_bar

        close = float(frame.iloc[end]["Close"])
        if direction == "bullish":
            pnl = round(close - entry_price, 2)
        else:
            pnl = round(entry_price - close, 2)
        rr = round(pnl / risk, 2) if risk > 0 else 0.0
        return pnl, rr, pnl > 0, end - entry_bar

    def _simulate_trigger_trade(
        self,
        frame: pd.DataFrame,
        record: dict[str, Any],
    ) -> TriggerTradeOutcome | None:
        entry_bar = self._resolve_bar(frame, record)
        if entry_bar is None or entry_bar >= len(frame) - 1:
            return None

        direction = str(record["direction"])
        entry_price = round(float(frame.iloc[entry_bar]["Close"]), 2)
        stop, risk = self._structural_stop(frame, entry_bar, entry_price, direction)
        target = self._opposite_liquidity_target(frame, entry_bar, entry_price, direction, risk)
        pnl, rr, win, holding_bars = self._simulate_trade(
            frame,
            entry_bar,
            entry_price,
            direction,
            stop,
            target,
            risk,
        )

        return TriggerTradeOutcome(
            symbol=str(record["symbol"]),
            timeframe=str(record["timeframe"]),
            direction=direction,
            trigger_model=str(record["trigger_model"]),
            trigger_timestamp=str(record.get("trigger_timestamp", "")),
            trigger_bar=entry_bar,
            entry_price=entry_price,
            stop_price=stop,
            target_price=target,
            risk_points=risk,
            realized_pnl_points=pnl,
            realized_rr=rr,
            win=win,
            holding_bars=holding_bars,
            is_false_trigger=bool(record.get("is_false_trigger", False)),
        )

    def _aggregate_metrics(
        self,
        outcomes: list[TriggerTradeOutcome],
    ) -> list[TriggerTradeMetrics]:
        groups: dict[tuple[str, str], list[TriggerTradeOutcome]] = defaultdict(list)
        for outcome in outcomes:
            groups[(outcome.trigger_model, outcome.direction)].append(outcome)

        metrics: list[TriggerTradeMetrics] = []
        for (model, direction), bucket in groups.items():
            if len(bucket) < MIN_MODEL_SAMPLES:
                continue
            pnls = [item.realized_pnl_points for item in bucket]
            wins = sum(1 for item in bucket if item.win)
            total = len(bucket)
            pf = self._profit_factor(pnls)
            exp = round(mean(pnls), 2)
            metrics.append(
                TriggerTradeMetrics(
                    trigger_model=model,
                    direction=direction,
                    trades=total,
                    win_rate_pct=round(wins / total * 100, 2),
                    profit_factor=pf,
                    expectancy=exp,
                    average_rr=round(mean(item.realized_rr for item in bucket), 2),
                    net_points=round(sum(pnls), 2),
                    maximum_drawdown_points=self._maximum_drawdown(pnls),
                    false_trigger_rate_pct=round(
                        sum(1 for item in bucket if item.is_false_trigger) / total * 100,
                        2,
                    ),
                    classification=self._classify_trigger(
                        total,
                        round(wins / total * 100, 2),
                        exp,
                        pf,
                    ),
                ),
            )
        return metrics

    @staticmethod
    def _rank_metrics(
        metrics: list[TriggerTradeMetrics],
    ) -> tuple[
        list[TriggerTradeMetrics],
        list[TriggerTradeMetrics],
        list[TriggerTradeMetrics],
    ]:
        profitable = sorted(
            metrics,
            key=lambda item: (item.expectancy, item.profit_factor or 0, item.trades),
            reverse=True,
        )
        accurate = sorted(
            [item for item in metrics if item.trades >= MIN_MODEL_SAMPLES],
            key=lambda item: (item.win_rate_pct, item.expectancy, item.trades),
            reverse=True,
        )
        dangerous = sorted(
            metrics,
            key=lambda item: (
                item.expectancy,
                item.win_rate_pct,
                -item.maximum_drawdown_points,
            ),
        )

        for index, item in enumerate(profitable, start=1):
            item.rank_profit = index
        for index, item in enumerate(accurate, start=1):
            item.rank_accuracy = index
        for index, item in enumerate(dangerous, start=1):
            item.rank_danger = index

        return profitable, accurate, dangerous

    def run(self, metadata: dict[str, Any] | None = None) -> TriggerTradeValidationReport:
        started = time.perf_counter()
        trigger_report = self._load_trigger_report()

        if metadata is None:
            metadata = {
                "start_date": trigger_report.get("start_date", ""),
                "end_date": trigger_report.get("end_date", ""),
                "research_window_days": trigger_report.get("research_window_days", self.research_days),
            }

        records = trigger_report.get("trigger_records", [])
        outcomes: list[TriggerTradeOutcome] = []
        skipped = 0

        for record in records:
            frame = self._load_frame(
                str(record["symbol"]),
                str(record["timeframe"]),
                metadata,
            )
            if frame is None:
                skipped += 1
                continue
            outcome = self._simulate_trigger_trade(frame, record)
            if outcome is None:
                skipped += 1
                continue
            outcomes.append(outcome)

        metrics = self._aggregate_metrics(outcomes)
        profitable, accurate, dangerous = self._rank_metrics(metrics)

        production_matrix = sorted(
            metrics,
            key=lambda item: (
                0 if item.classification == "Production Ready" else 1,
                -item.expectancy,
                -item.win_rate_pct,
            ),
        )

        ready_count = sum(1 for item in metrics if item.classification == "Production Ready")
        validate_count = sum(1 for item in metrics if item.classification == "Needs Validation")
        reject_count = sum(1 for item in metrics if item.classification == "Reject")

        top_profit = profitable[0] if profitable else None
        top_accuracy = accurate[0] if accurate else None
        top_danger = dangerous[0] if dangerous else None

        conclusions = [
            f"Simulated {len(outcomes)} trades from {len(records)} trigger records.",
            f"Trigger models with trade metrics: {len(metrics)}.",
            (
                f"Most profitable: {top_profit.trigger_model[:80]} "
                f"(Exp {top_profit.expectancy}, n={top_profit.trades})"
                if top_profit
                else "No trigger models met minimum sample threshold."
            ),
            (
                f"Most accurate: {top_accuracy.trigger_model[:80]} "
                f"(WR {top_accuracy.win_rate_pct}%, n={top_accuracy.trades})"
                if top_accuracy
                else "No accurate trigger models identified."
            ),
            (
                f"Most dangerous: {top_danger.trigger_model[:80]} "
                f"(Exp {top_danger.expectancy}, WR {top_danger.win_rate_pct}%)"
                if top_danger
                else "No dangerous trigger models identified."
            ),
            (
                f"Production matrix: {ready_count} Ready, "
                f"{validate_count} Needs Validation, {reject_count} Reject."
            ),
        ]

        return TriggerTradeValidationReport(
            source_report=str(self.trigger_report_path),
            symbols_analyzed=trigger_report.get("symbols_analyzed", []),
            research_window_days=trigger_report.get("research_window_days", self.research_days),
            start_date=trigger_report.get("start_date", ""),
            end_date=trigger_report.get("end_date", ""),
            timeframes_analyzed=trigger_report.get("timeframes_analyzed", []),
            total_trigger_records=len(records),
            trades_simulated=len(outcomes),
            trades_skipped=skipped,
            entry_model=ENTRY_MODEL,
            stop_model=STOP_MODEL,
            target_model=TARGET_MODEL,
            trigger_trade_metrics=[item.as_dict() for item in metrics],
            production_trigger_matrix=[item.as_dict() for item in production_matrix],
            most_profitable_triggers=[item.as_dict() for item in profitable[:TOP_RANK_COUNT]],
            most_accurate_triggers=[item.as_dict() for item in accurate[:TOP_RANK_COUNT]],
            most_dangerous_triggers=[item.as_dict() for item in dangerous[:TOP_RANK_COUNT]],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_trigger_trade_validation_report(
    report_path: Path | str | None = None,
    trigger_report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> TriggerTradeValidationReport:
    """Run trigger-to-trade validation and export JSON."""
    metadata: dict[str, Any] | None = None
    filter_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if filter_path.exists():
        with filter_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    engine = TriggerTradeValidationResearch(trigger_report_path=trigger_report_path)
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Trigger-to-trade validation completed: trades=%s models=%s",
        report.trades_simulated,
        len(report.trigger_trade_metrics),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_trigger_trade_validation_report()
        print("Trigger-to-Trade Validation Research Summary")
        print(f"Trigger records: {report.total_trigger_records}")
        print(f"Trades simulated: {report.trades_simulated}")
        print(f"Trigger models: {len(report.trigger_trade_metrics)}")
        if report.most_profitable_triggers:
            top = report.most_profitable_triggers[0]
            print(
                f"Most profitable: {top['trigger_model'][:100]} "
                f"(Exp {top['expectancy']}, WR {top['win_rate_pct']}%)",
            )
        if report.most_dangerous_triggers:
            danger = report.most_dangerous_triggers[0]
            print(
                f"Most dangerous: {danger['trigger_model'][:100]} "
                f"(Exp {danger['expectancy']}, WR {danger['win_rate_pct']}%)",
            )
        ready = sum(
            1 for item in report.production_trigger_matrix
            if item.get("classification") == "Production Ready"
        )
        print(f"Production Ready triggers: {ready}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except TriggerTradeValidationError as exc:
        logger.error("Trigger-to-trade validation error: %s", exc)
        print(f"Trigger-to-trade validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected trigger-to-trade validation error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
