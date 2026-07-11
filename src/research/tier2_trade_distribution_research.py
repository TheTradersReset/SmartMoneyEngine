"""
Tier-2 trade distribution research for SmartMoneyEngine.

Analyzes Tier-2 setup outcomes (Displacement + CHOCH + BOS + FVG Reclaim):
winner/loser distributions, MFE/MAE, timing, streaks, monthly breakdown,
and equity curve simulation. Research-only; no production logic or entries.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.tiered_signal_framework_research import (
    FORWARD_BARS,
    TierSignal,
    TieredSignalFrameworkResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tier2_trade_distribution.json"

WINNER_BUCKETS = (
    ("0-50", 0, 50),
    ("50-100", 50, 100),
    ("100-150", 100, 150),
    ("150-200", 150, 200),
    ("200+", 200, float("inf")),
)

LOSER_BUCKETS = (
    ("0-20", 0, 20),
    ("20-40", 20, 40),
    ("40-60", 40, 60),
    ("60+", 60, float("inf")),
)

TIER2_DEFINITION = [
    "Displacement",
    "CHOCH",
    "BOS",
    "FVG Reclaim",
]


class Tier2TradeDistributionError(Exception):
    """Raised when tier-2 trade distribution research fails."""


@dataclass(frozen=True)
class Tier2TradeDetail:
    """Detailed research outcome for one Tier-2 signal."""

    timeframe: str
    direction: str
    bos_timestamp: str
    risk_points: float
    mfe_points: float
    mae_points: float
    realized_pnl_points: float
    win: bool
    bars_to_target: int | None
    bars_to_stop: int | None
    minutes_to_target: float | None
    minutes_to_stop: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2TradeDistributionReport:
    """Full Tier-2 trade distribution research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    total_signals: int
    winners: int
    losers: int
    win_rate_pct: float
    winner_distribution: dict[str, dict[str, Any]]
    loser_distribution: dict[str, dict[str, Any]]
    mfe_summary: dict[str, float]
    mae_summary: dict[str, float]
    time_to_target_summary: dict[str, float | None]
    time_to_stop_summary: dict[str, float | None]
    consecutive_streaks: dict[str, Any]
    consecutive_loss_probability_pct: dict[str, float]
    monthly_breakdown: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    expected_monthly_points: float
    expected_yearly_points: float
    worst_drawdown_points: float
    best_drawdown_points: float
    max_equity_peak_points: float
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tier2TradeDistributionResearch:
    """Analyze Tier-2 setup trade distribution and equity simulation."""

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
    def _distribution(
        values: list[float],
        buckets: tuple[tuple[str, float, float], ...],
    ) -> dict[str, dict[str, Any]]:
        counts: dict[str, int] = {label: 0 for label, _, _ in buckets}
        for value in values:
            counts[Tier2TradeDistributionResearch._bucket_label(value, buckets)] += 1
        total = len(values)
        return {
            label: {
                "count": counts[label],
                "pct": round(counts[label] / total * 100, 2) if total else 0.0,
            }
            for label, _, _ in buckets
        }

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

    def _simulate_detailed(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
    ) -> Tier2TradeDetail | None:
        anchor_bar = signal.bos_bar
        if anchor_bar >= len(frame) - 1:
            return None

        entry, risk = self.tier_engine._risk_points(frame, anchor_bar, signal.direction)
        end = min(len(frame) - 1, anchor_bar + FORWARD_BARS)
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)

        if signal.direction == "bullish":
            stop = entry - risk
            target = entry + risk
        else:
            stop = entry + risk
            target = entry - risk

        mfe = 0.0
        mae = 0.0
        bars_to_target: int | None = None
        bars_to_stop: int | None = None
        stopped = False

        for index in range(anchor_bar + 1, end + 1):
            bar_high = float(highs.iloc[index])
            bar_low = float(lows.iloc[index])
            offset = index - anchor_bar

            if signal.direction == "bullish":
                adverse = entry - bar_low
                favorable = bar_high - entry
                stop_hit = bar_low <= stop
                target_hit = bar_high >= target
            else:
                adverse = bar_high - entry
                favorable = entry - bar_low
                stop_hit = bar_high >= stop
                target_hit = bar_low <= target

            mae = max(mae, max(adverse, 0.0))
            mfe = max(mfe, max(favorable, 0.0))

            if target_hit and bars_to_target is None:
                bars_to_target = offset
            if stop_hit:
                bars_to_stop = offset
                stopped = True
                break

        mfe = round(mfe, 2)
        mae = round(mae, 2)

        if stopped:
            pnl = -risk
            win = False
        else:
            pnl = round(mfe, 2)
            win = mfe >= risk

        return Tier2TradeDetail(
            timeframe=signal.timeframe,
            direction=signal.direction,
            bos_timestamp=signal.bos_timestamp,
            risk_points=risk,
            mfe_points=mfe,
            mae_points=mae,
            realized_pnl_points=pnl,
            win=win,
            bars_to_target=bars_to_target,
            bars_to_stop=bars_to_stop,
            minutes_to_target=self.tier_engine._bars_to_minutes(
                bars_to_target,
                signal.timeframe,
            ),
            minutes_to_stop=self.tier_engine._bars_to_minutes(
                bars_to_stop,
                signal.timeframe,
            ),
        )

    @staticmethod
    def _streak_stats(wins: list[bool]) -> dict[str, Any]:
        if not wins:
            return {
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "average_win_streak": 0.0,
                "average_loss_streak": 0.0,
                "win_streak_distribution": {},
                "loss_streak_distribution": {},
            }

        win_streaks: list[int] = []
        loss_streaks: list[int] = []
        current = 1
        for index in range(1, len(wins)):
            if wins[index] == wins[index - 1]:
                current += 1
            else:
                (win_streaks if wins[index - 1] else loss_streaks).append(current)
                current = 1
        (win_streaks if wins[-1] else loss_streaks).append(current)

        win_dist: dict[int, int] = defaultdict(int)
        loss_dist: dict[int, int] = defaultdict(int)
        for streak in win_streaks:
            win_dist[streak] += 1
        for streak in loss_streaks:
            loss_dist[streak] += 1

        return {
            "max_win_streak": max(win_streaks) if win_streaks else 0,
            "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
            "average_win_streak": round(mean(win_streaks), 2) if win_streaks else 0.0,
            "average_loss_streak": round(mean(loss_streaks), 2) if loss_streaks else 0.0,
            "win_streak_distribution": {str(k): v for k, v in sorted(win_dist.items())},
            "loss_streak_distribution": {str(k): v for k, v in sorted(loss_dist.items())},
        }

    @staticmethod
    def _consecutive_loss_probability(wins: list[bool], streak_length: int) -> float:
        if len(wins) < streak_length:
            return 0.0
        windows = len(wins) - streak_length + 1
        hits = sum(
            1
            for start in range(windows)
            if all(not wins[start + offset] for offset in range(streak_length))
        )
        return round(hits / windows * 100, 2)

    @staticmethod
    def _month_key(timestamp: str) -> str:
        parsed = pd.Timestamp(timestamp)
        return parsed.strftime("%Y-%m")

    def _monthly_breakdown(self, trades: list[Tier2TradeDetail]) -> list[dict[str, Any]]:
        buckets: dict[str, list[Tier2TradeDetail]] = defaultdict(list)
        for trade in trades:
            buckets[self._month_key(trade.bos_timestamp)].append(trade)

        rows: list[dict[str, Any]] = []
        for month in sorted(buckets.keys()):
            bucket = buckets[month]
            pnls = [item.realized_pnl_points for item in bucket]
            wins = sum(1 for item in bucket if item.win)
            rows.append(
                {
                    "month": month,
                    "signals": len(bucket),
                    "winners": wins,
                    "losers": len(bucket) - wins,
                    "win_rate_pct": round(wins / len(bucket) * 100, 2) if bucket else 0.0,
                    "total_pnl_points": round(sum(pnls), 2),
                    "average_pnl_points": round(mean(pnls), 2) if pnls else 0.0,
                }
            )
        return rows

    @staticmethod
    def _equity_curve(trades: list[Tier2TradeDetail]) -> list[dict[str, Any]]:
        cumulative = 0.0
        curve: list[dict[str, Any]] = []
        for index, trade in enumerate(trades, start=1):
            cumulative = round(cumulative + trade.realized_pnl_points, 2)
            curve.append(
                {
                    "trade_number": index,
                    "timestamp": trade.bos_timestamp,
                    "trade_pnl_points": trade.realized_pnl_points,
                    "cumulative_pnl_points": cumulative,
                }
            )
        return curve

    @staticmethod
    def _drawdown_metrics(equity_values: list[float]) -> tuple[float, float, float]:
        if not equity_values:
            return 0.0, 0.0, 0.0

        peak = equity_values[0]
        max_drawdown = 0.0
        min_drawdown = float("inf")
        max_peak = equity_values[0]

        for value in equity_values:
            peak = max(peak, value)
            max_peak = max(max_peak, value)
            drawdown = round(peak - value, 2)
            max_drawdown = max(max_drawdown, drawdown)
            if drawdown > 0:
                min_drawdown = min(min_drawdown, drawdown)

        if min_drawdown == float("inf"):
            min_drawdown = 0.0

        return round(max_drawdown, 2), round(min_drawdown, 2), round(max_peak, 2)

    def _conclusions(
        self,
        trades: list[Tier2TradeDetail],
        expected_monthly: float,
        worst_dd: float,
        loss_probs: dict[str, float],
    ) -> list[str]:
        winners = [item for item in trades if item.win]
        notes = [
            f"Tier-2 signals analyzed: {len(trades)} ({TIER2_DEFINITION}).",
            f"Win rate: {round(len(winners) / len(trades) * 100, 2) if trades else 0}% "
            f"({len(winners)} winners, {len(trades) - len(winners)} losers).",
            f"Expected monthly points: {expected_monthly}.",
            f"Worst drawdown: {worst_dd} points.",
            (
                f"Consecutive loss probability: 3={loss_probs['3']}% "
                f"5={loss_probs['5']}% 7={loss_probs['7']}%."
            ),
        ]
        return notes

    def run(self, metadata: dict[str, Any]) -> Tier2TradeDistributionReport:
        """Run Tier-2 trade distribution research."""
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
        research_months = self.tier_engine._research_months(start, end)

        all_trades: list[Tier2TradeDetail] = []

        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            signals = self.tier_engine._detect_tier2(frame, timeframe_label)
            for signal in signals:
                detail = self._simulate_detailed(frame, signal)
                if detail:
                    all_trades.append(detail)

        all_trades.sort(key=lambda item: pd.Timestamp(item.bos_timestamp))

        winners = [item for item in all_trades if item.win]
        losers = [item for item in all_trades if not item.win]
        win_flags = [item.win for item in all_trades]

        winner_sizes = [item.realized_pnl_points for item in winners]
        loser_sizes = [abs(item.realized_pnl_points) for item in losers]

        target_times = [
            item.minutes_to_target for item in all_trades if item.minutes_to_target is not None
        ]
        stop_times = [
            item.minutes_to_stop for item in losers if item.minutes_to_stop is not None
        ]

        equity_curve = self._equity_curve(all_trades)
        equity_values = [point["cumulative_pnl_points"] for point in equity_curve]
        worst_dd, best_dd, max_peak = self._drawdown_metrics(equity_values)
        total_pnl = equity_values[-1] if equity_values else 0.0
        expected_monthly = round(total_pnl / research_months, 2)

        loss_probs = {
            "3": self._consecutive_loss_probability(win_flags, 3),
            "5": self._consecutive_loss_probability(win_flags, 5),
            "7": self._consecutive_loss_probability(win_flags, 7),
        }

        return Tier2TradeDistributionReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            total_signals=len(all_trades),
            winners=len(winners),
            losers=len(losers),
            win_rate_pct=round(len(winners) / len(all_trades) * 100, 2) if all_trades else 0.0,
            winner_distribution=self._distribution(winner_sizes, WINNER_BUCKETS),
            loser_distribution=self._distribution(loser_sizes, LOSER_BUCKETS),
            mfe_summary=self._summary_stats([item.mfe_points for item in all_trades]),
            mae_summary=self._summary_stats([item.mae_points for item in all_trades]),
            time_to_target_summary={
                **self._summary_stats(target_times),
                "hit_rate_pct": round(len(target_times) / len(all_trades) * 100, 2)
                if all_trades
                else 0.0,
            },
            time_to_stop_summary={
                **self._summary_stats(stop_times),
                "hit_rate_pct": round(len(stop_times) / len(losers) * 100, 2) if losers else 0.0,
            },
            consecutive_streaks=self._streak_stats(win_flags),
            consecutive_loss_probability_pct=loss_probs,
            monthly_breakdown=self._monthly_breakdown(all_trades),
            equity_curve=equity_curve,
            expected_monthly_points=expected_monthly,
            expected_yearly_points=round(expected_monthly * 12, 2),
            worst_drawdown_points=worst_dd,
            best_drawdown_points=best_dd,
            max_equity_peak_points=max_peak,
            conclusions=self._conclusions(all_trades, expected_monthly, worst_dd, loss_probs),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tier2_trade_distribution_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Tier2TradeDistributionReport:
    """Run Tier-2 trade distribution research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Tier2TradeDistributionError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = Tier2TradeDistributionResearch(
        symbol=metadata.get("symbol", "NIFTY50"),
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", ("5M", "15M", "1H"))),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info("Tier-2 trade distribution completed: signals=%s", report.total_signals)
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tier2_trade_distribution_report()
        print("Tier-2 Trade Distribution Research Summary")
        print(f"Signals: {report.total_signals} | WR: {report.win_rate_pct}%")
        print(f"Expected monthly: {report.expected_monthly_points} pts")
        print(f"Expected yearly: {report.expected_yearly_points} pts")
        print(f"Worst drawdown: {report.worst_drawdown_points} pts")
        print(f"Best drawdown: {report.best_drawdown_points} pts")
        probs = report.consecutive_loss_probability_pct
        print(f"P(3 losses): {probs['3']}% | P(5): {probs['5']}% | P(7): {probs['7']}%")
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except Tier2TradeDistributionError as exc:
        logger.error("Tier-2 trade distribution error: %s", exc)
        print(f"Tier-2 trade distribution error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Tier-2 trade distribution failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
