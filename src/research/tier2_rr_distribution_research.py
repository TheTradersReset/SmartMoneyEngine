"""
Tier-2 risk-reward distribution research for SmartMoneyEngine.

Analyzes MFE/MAE and R-multiple reachability for Tier-2 BOS Close trades with
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
from statistics import mean, median
from typing import Any

import pandas as pd

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.tier2_entry_optimization_research import Tier2EntryOptimizationResearch
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch
from src.research.tiered_signal_framework_research import (
    FORWARD_BARS,
    TIMEFRAME_MINUTES,
    TierSignal,
    TieredSignalFrameworkResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tier2_rr_distribution.json"

RR_LEVELS = (1, 2, 3, 4, 5)

RISK_BUCKETS = (
    ("0-30", 0, 30),
    ("30-60", 30, 60),
    ("60-90", 60, 90),
    ("90-120", 90, 120),
    ("120+", 120, float("inf")),
)

MFE_R_BUCKETS = (
    ("0-0.5R", 0, 0.5),
    ("0.5-1R", 0.5, 1),
    ("1-2R", 1, 2),
    ("2-3R", 2, 3),
    ("3-5R", 3, 5),
    ("5R+", 5, float("inf")),
)

MAE_R_BUCKETS = (
    ("0-0.25R", 0, 0.25),
    ("0.25-0.5R", 0.25, 0.5),
    ("0.5-0.75R", 0.5, 0.75),
    ("0.75-1R", 0.75, 1),
    ("1R+", 1, float("inf")),
)

MAX_R_OUTCOME_BUCKETS = ("0R", "1R", "2R", "3R", "4R", "5R+")


class Tier2RrDistributionError(Exception):
    """Raised when Tier-2 RR distribution research fails."""


@dataclass(frozen=True)
class RrTradeRecord:
    """RR profile for one Tier-2 BOS Close trade."""

    bos_timestamp: str
    timeframe: str
    direction: str
    entry_price: float
    stop_price: float
    risk_points: float
    mfe_points: float
    mae_points: float
    mfe_r: float
    mae_r: float
    max_r_reached: int
    reached_1r: bool
    reached_2r: bool
    reached_3r: bool
    reached_4r: bool
    reached_5r: bool
    stopped_out: bool
    bars_to_1r: int | None
    bars_to_2r: int | None
    bars_to_3r: int | None
    bars_to_4r: int | None
    bars_to_5r: int | None
    minutes_to_1r: float | None
    minutes_to_2r: float | None
    minutes_to_3r: float | None
    minutes_to_4r: float | None
    minutes_to_5r: float | None
    bars_to_stop: int | None
    minutes_to_stop: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2RrDistributionReport:
    """Full Tier-2 RR distribution research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    entry_method: str
    stop_loss_model: str
    total_signals: int
    rr_distribution_table: dict[str, dict[str, Any]]
    stop_loss_analysis: dict[str, Any]
    mfe_analysis: dict[str, Any]
    mae_analysis: dict[str, Any]
    trade_outcome_distribution: dict[str, Any]
    sample_trades: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tier2RrDistributionResearch:
    """Analyze RR reachability and excursion distribution for Tier-2 BOS Close trades."""

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
    def _minutes_per_bar(timeframe: str) -> int:
        return TIMEFRAME_MINUTES.get(timeframe, 5)

    @staticmethod
    def _bars_to_minutes(bars: int | None, timeframe: str) -> float | None:
        if bars is None:
            return None
        return round(bars * Tier2RrDistributionResearch._minutes_per_bar(timeframe), 1)

    @staticmethod
    def _bucket_label(
        value: float,
        buckets: tuple[tuple[str, float, float], ...],
    ) -> str:
        for label, lower, upper in buckets:
            if lower <= value < upper:
                return label
        return buckets[-1][0]

    @staticmethod
    def _summary_stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "mean": round(mean(values), 2),
            "median": round(median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    @staticmethod
    def _distribution(
        values: list[float],
        buckets: tuple[tuple[str, float, float], ...],
    ) -> dict[str, dict[str, Any]]:
        counts: dict[str, int] = {label: 0 for label, _, _ in buckets}
        for value in values:
            counts[Tier2RrDistributionResearch._bucket_label(value, buckets)] += 1
        total = len(values)
        return {
            label: {
                "count": counts[label],
                "pct": round(counts[label] / total * 100, 2) if total else 0.0,
            }
            for label, _, _ in buckets
        }

    def _simulate_rr_profile(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        stop_price: float,
        risk: float,
        direction: str,
        timeframe: str,
    ) -> dict[str, Any]:
        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
        mfe = 0.0
        mae = 0.0
        stopped_out = False
        bars_to_stop: int | None = None
        reached = {level: False for level in RR_LEVELS}
        bars_to_level: dict[int, int | None] = {level: None for level in RR_LEVELS}

        for index in range(entry_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])
            offset = index - entry_bar

            if direction == "bullish":
                stop_hit = bar_low <= stop_price
                favorable = bar_high - entry_price
                adverse = entry_price - bar_low
            else:
                stop_hit = bar_high >= stop_price
                favorable = entry_price - bar_low
                adverse = bar_high - entry_price

            mfe = max(mfe, max(favorable, 0.0))
            mae = max(mae, max(adverse, 0.0))

            if stop_hit:
                stopped_out = True
                bars_to_stop = offset
                break

            for level in RR_LEVELS:
                threshold = risk * level
                if not reached[level] and favorable >= threshold:
                    reached[level] = True
                    bars_to_level[level] = offset

        mfe = round(mfe, 2)
        mae = round(mae, 2)
        mfe_r = round(mfe / risk, 2) if risk > 0 else 0.0
        mae_r = round(mae / risk, 2) if risk > 0 else 0.0

        max_r = 0
        for level in reversed(RR_LEVELS):
            if reached[level]:
                max_r = level
                break
        if not stopped_out and mfe_r >= 5:
            max_r = 5

        return {
            "mfe_points": mfe,
            "mae_points": mae,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "max_r_reached": max_r,
            "reached": reached,
            "bars_to_level": bars_to_level,
            "stopped_out": stopped_out,
            "bars_to_stop": bars_to_stop,
            "minutes_to_stop": self._bars_to_minutes(bars_to_stop, timeframe),
            "minutes_to_level": {
                level: self._bars_to_minutes(bars_to_level[level], timeframe)
                for level in RR_LEVELS
            },
        }

    def _analyze_signal(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
    ) -> RrTradeRecord | None:
        trigger = self.entry_engine._resolve_entry("A_bos_close", frame, signal)
        if not trigger.triggered or trigger.entry_bar is None or trigger.entry_price is None:
            return None

        entry_bar = trigger.entry_bar
        entry_price = trigger.entry_price
        stop_price, risk = self.construction_engine._structural_stop(
            frame,
            entry_bar,
            entry_price,
            signal.direction,
        )

        profile = self._simulate_rr_profile(
            frame,
            entry_bar,
            entry_price,
            stop_price,
            risk,
            signal.direction,
            signal.timeframe,
        )

        return RrTradeRecord(
            bos_timestamp=signal.bos_timestamp,
            timeframe=signal.timeframe,
            direction=signal.direction,
            entry_price=entry_price,
            stop_price=stop_price,
            risk_points=risk,
            mfe_points=profile["mfe_points"],
            mae_points=profile["mae_points"],
            mfe_r=profile["mfe_r"],
            mae_r=profile["mae_r"],
            max_r_reached=profile["max_r_reached"],
            reached_1r=profile["reached"][1],
            reached_2r=profile["reached"][2],
            reached_3r=profile["reached"][3],
            reached_4r=profile["reached"][4],
            reached_5r=profile["reached"][5],
            stopped_out=profile["stopped_out"],
            bars_to_1r=profile["bars_to_level"][1],
            bars_to_2r=profile["bars_to_level"][2],
            bars_to_3r=profile["bars_to_level"][3],
            bars_to_4r=profile["bars_to_level"][4],
            bars_to_5r=profile["bars_to_level"][5],
            minutes_to_1r=profile["minutes_to_level"][1],
            minutes_to_2r=profile["minutes_to_level"][2],
            minutes_to_3r=profile["minutes_to_level"][3],
            minutes_to_4r=profile["minutes_to_level"][4],
            minutes_to_5r=profile["minutes_to_level"][5],
            bars_to_stop=profile["bars_to_stop"],
            minutes_to_stop=profile["minutes_to_stop"],
        )

    def _collect_records(self, metadata: dict[str, Any]) -> list[RrTradeRecord]:
        from datetime import timedelta

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        records: list[RrTradeRecord] = []
        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)

            for signal in self.tier_engine._detect_tier2(frame, timeframe_label):
                record = self._analyze_signal(frame, signal)
                if record is not None:
                    records.append(record)

        records.sort(key=lambda item: pd.Timestamp(item.bos_timestamp))
        return records

    def _rr_distribution_table(self, records: list[RrTradeRecord]) -> dict[str, dict[str, Any]]:
        total = len(records)
        table: dict[str, dict[str, Any]] = {}

        level_fields = {
            1: ("reached_1r", "minutes_to_1r"),
            2: ("reached_2r", "minutes_to_2r"),
            3: ("reached_3r", "minutes_to_3r"),
            4: ("reached_4r", "minutes_to_4r"),
            5: ("reached_5r", "minutes_to_5r"),
        }

        for level in RR_LEVELS:
            reached_field, time_field = level_fields[level]
            hitters = [record for record in records if getattr(record, reached_field)]
            times = [
                getattr(record, time_field)
                for record in hitters
                if getattr(record, time_field) is not None
            ]

            table[f"{level}R"] = {
                "signals": total,
                "hits": len(hitters),
                "hit_rate_pct": round(len(hitters) / total * 100, 2) if total else 0.0,
                "average_time_to_hit_minutes": round(mean(times), 1) if times else None,
                "average_mfe_points": round(mean(record.mfe_points for record in records), 2)
                if records
                else 0.0,
                "average_mae_points": round(mean(record.mae_points for record in records), 2)
                if records
                else 0.0,
                "average_mfe_r": round(mean(record.mfe_r for record in records), 2)
                if records
                else 0.0,
                "average_mae_r": round(mean(record.mae_r for record in records), 2)
                if records
                else 0.0,
            }

        return table

    def _stop_loss_analysis(self, records: list[RrTradeRecord]) -> dict[str, Any]:
        risks = [record.risk_points for record in records]
        stopped = [record for record in records if record.stopped_out]
        stop_times = [
            record.minutes_to_stop for record in stopped if record.minutes_to_stop is not None
        ]

        return {
            "stop_loss_model": "Structural Swing SL",
            "risk_points": self._summary_stats(risks),
            "risk_distribution": self._distribution(risks, RISK_BUCKETS),
            "stopped_out_count": len(stopped),
            "stopped_out_pct": round(len(stopped) / len(records) * 100, 2) if records else 0.0,
            "average_time_to_stop_minutes": round(mean(stop_times), 1) if stop_times else None,
            "average_risk_points": round(mean(risks), 2) if risks else 0.0,
        }

    def _mfe_analysis(self, records: list[RrTradeRecord]) -> dict[str, Any]:
        mfe_points = [record.mfe_points for record in records]
        mfe_r = [record.mfe_r for record in records]
        return {
            "mfe_points": self._summary_stats(mfe_points),
            "mfe_r": self._summary_stats(mfe_r),
            "mfe_r_distribution": self._distribution(mfe_r, MFE_R_BUCKETS),
            "reached_1r_pct": round(sum(1 for record in records if record.reached_1r) / len(records) * 100, 2)
            if records
            else 0.0,
            "reached_2r_pct": round(sum(1 for record in records if record.reached_2r) / len(records) * 100, 2)
            if records
            else 0.0,
            "reached_3r_or_more_pct": round(
                sum(1 for record in records if record.reached_3r) / len(records) * 100,
                2,
            )
            if records
            else 0.0,
        }

    def _mae_analysis(self, records: list[RrTradeRecord]) -> dict[str, Any]:
        mae_points = [record.mae_points for record in records]
        mae_r = [record.mae_r for record in records]
        full_1r_mae = sum(1 for record in records if record.mae_r >= 1.0)
        return {
            "mae_points": self._summary_stats(mae_points),
            "mae_r": self._summary_stats(mae_r),
            "mae_r_distribution": self._distribution(mae_r, MAE_R_BUCKETS),
            "touched_full_risk_pct": round(full_1r_mae / len(records) * 100, 2) if records else 0.0,
            "stopped_out_before_1r_mfe_pct": round(
                sum(
                    1
                    for record in records
                    if record.stopped_out and not record.reached_1r
                )
                / len(records)
                * 100,
                2,
            )
            if records
            else 0.0,
        }

    @staticmethod
    def _max_r_outcome_label(max_r: int, mfe_r: float) -> str:
        if max_r >= 5 or mfe_r >= 5:
            return "5R+"
        if max_r == 0:
            return "0R"
        return f"{max_r}R"

    def _trade_outcome_distribution(self, records: list[RrTradeRecord]) -> dict[str, Any]:
        total = len(records)
        outcome_counts: dict[str, int] = {label: 0 for label in MAX_R_OUTCOME_BUCKETS}

        for record in records:
            label = self._max_r_outcome_label(record.max_r_reached, record.mfe_r)
            outcome_counts[label] += 1

        stopped_before_1r = sum(
            1 for record in records if record.stopped_out and not record.reached_1r
        )
        reached_1r_stopped = sum(
            1 for record in records if record.stopped_out and record.reached_1r
        )
        not_stopped = sum(1 for record in records if not record.stopped_out)

        return {
            "max_r_reached_distribution": {
                label: {
                    "count": outcome_counts[label],
                    "pct": round(outcome_counts[label] / total * 100, 2) if total else 0.0,
                }
                for label in MAX_R_OUTCOME_BUCKETS
            },
            "stopped_out_before_1r": {
                "count": stopped_before_1r,
                "pct": round(stopped_before_1r / total * 100, 2) if total else 0.0,
            },
            "stopped_out_after_1r": {
                "count": reached_1r_stopped,
                "pct": round(reached_1r_stopped / total * 100, 2) if total else 0.0,
            },
            "not_stopped_within_window": {
                "count": not_stopped,
                "pct": round(not_stopped / total * 100, 2) if total else 0.0,
            },
        }

    def run(self, metadata: dict[str, Any]) -> Tier2RrDistributionReport:
        """Run Tier-2 RR distribution research."""
        started = time.perf_counter()
        records = self._collect_records(metadata)
        if not records:
            raise Tier2RrDistributionError("No Tier-2 BOS Close RR records found.")

        rr_table = self._rr_distribution_table(records)
        stop_analysis = self._stop_loss_analysis(records)
        mfe_analysis = self._mfe_analysis(records)
        mae_analysis = self._mae_analysis(records)
        outcome_dist = self._trade_outcome_distribution(records)

        conclusions = [
            f"Analyzed {len(records)} Tier-2 BOS Close trades with Structural Swing SL.",
            (
                f"1R hit rate: {rr_table['1R']['hit_rate_pct']}% "
                f"(avg time {rr_table['1R']['average_time_to_hit_minutes']} min)."
            ),
            (
                f"2R hit rate: {rr_table['2R']['hit_rate_pct']}%; "
                f"3R hit rate: {rr_table['3R']['hit_rate_pct']}%."
            ),
            (
                f"Average MFE: {mfe_analysis['mfe_points']['mean']} pts "
                f"({mfe_analysis['mfe_r']['mean']}R); "
                f"Average MAE: {mae_analysis['mae_points']['mean']} pts "
                f"({mae_analysis['mae_r']['mean']}R)."
            ),
            (
                f"Stopped out: {stop_analysis['stopped_out_pct']}% | "
                f"Stopped before 1R: {mae_analysis['stopped_out_before_1r_mfe_pct']}%."
            ),
        ]

        return Tier2RrDistributionReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            entry_method="BOS Close Entry",
            stop_loss_model="Structural Swing SL",
            total_signals=len(records),
            rr_distribution_table=rr_table,
            stop_loss_analysis=stop_analysis,
            mfe_analysis=mfe_analysis,
            mae_analysis=mae_analysis,
            trade_outcome_distribution=outcome_dist,
            sample_trades=[record.as_dict() for record in records[:12]],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tier2_rr_distribution_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Tier2RrDistributionReport:
    """Run Tier-2 RR distribution research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Tier2RrDistributionError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = Tier2RrDistributionResearch(
        symbol=metadata.get("symbol", "NIFTY50"),
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", ("5M", "15M", "1H"))),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info("Tier-2 RR distribution research completed: %s signals", report.total_signals)
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tier2_rr_distribution_report()
        print("Tier-2 RR Distribution Summary")
        print(f"Signals: {report.total_signals} | Entry: {report.entry_method}")
        for level, metrics in report.rr_distribution_table.items():
            print(
                f"  {level}: hit={metrics['hit_rate_pct']}% "
                f"time={metrics['average_time_to_hit_minutes']}min "
                f"MFE={metrics['average_mfe_r']}R"
            )
        print(f"Stopped out: {report.stop_loss_analysis['stopped_out_pct']}%")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except Tier2RrDistributionError as exc:
        logger.error("Tier-2 RR distribution error: %s", exc)
        print(f"Tier-2 RR distribution error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Tier-2 RR distribution failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
