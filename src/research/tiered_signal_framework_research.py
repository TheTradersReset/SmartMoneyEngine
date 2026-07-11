"""
Tiered signal framework research for SmartMoneyEngine.

Converts the validated institutional sequence into three research tiers and
compares accuracy, frequency, and risk-reward balance. Research-only; no
production logic, entries, or trades.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import (
    DisplacementStrength,
    FvgContext,
    LiquidityNarrativeEngine,
)
from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tiered_signal_framework.json"

FORWARD_BARS = 80
SEQUENCE_LOOKBACK = 40
SETUP_LOOKBACK_BARS = 20
MIN_EVENT_SEPARATION = 15
EXPANSION_THRESHOLD = 50
RISK_LOOKBACK = 20
SL_BUFFER_POINTS = 5.0
MIN_RISK_POINTS = 1.0

TIMEFRAME_MINUTES = {"5M": 5, "15M": 15, "1H": 60}

EVENT_MAP = {
    "bullish": {
        "sweep": "Sell_Liquidity_Sweep",
        "choch": "Bullish_CHOCH",
        "bos": "Bullish_BOS",
        "fvg_bias": "bullish",
        "opposing_choch": "Bearish_CHOCH",
    },
    "bearish": {
        "sweep": "Buy_Liquidity_Sweep",
        "choch": "Bearish_CHOCH",
        "bos": "Bearish_BOS",
        "fvg_bias": "bearish",
        "opposing_choch": "Bullish_CHOCH",
    },
}

TIER_DEFINITIONS = {
    "tier_1": {
        "label": "Tier 1 (Highest Quality)",
        "components": [
            "Liquidity Sweep",
            "Displacement",
            "CHOCH",
            "BOS",
            "FVG Reclaim",
        ],
    },
    "tier_2": {
        "label": "Tier 2 (Balanced)",
        "components": [
            "Displacement",
            "CHOCH",
            "BOS",
            "FVG Reclaim",
        ],
    },
    "tier_3": {
        "label": "Tier 3 (Higher Frequency)",
        "components": [
            "Continuation BOS",
            "FVG Reclaim",
        ],
    },
}


class TieredSignalFrameworkError(Exception):
    """Raised when tiered signal framework research fails."""


@dataclass(frozen=True)
class TierSignal:
    """One detected tier signal anchored at the BOS confirmation bar."""

    tier: str
    timeframe: str
    direction: str
    bos_bar: int
    bos_timestamp: str
    choch_bar: int | None = None
    sweep_bar: int | None = None
    displacement_bar: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TierSignalOutcome:
    """Research outcome for one tier signal using forward structural simulation."""

    tier: str
    timeframe: str
    direction: str
    bos_bar: int
    bos_timestamp: str
    risk_points: float
    forward_move_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    expansion_bar: int | None
    time_to_expansion_minutes: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TierMetrics:
    """Aggregate metrics for one signal tier."""

    tier: str
    label: str
    components: list[str]
    signals: int
    signals_per_month: float
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    average_move_size: float
    average_time_to_expansion_minutes: float | None
    win_rate_by_direction: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TierComparison:
    """Cross-tier comparison and balance recommendation."""

    best_accuracy_tier: str
    best_frequency_tier: str
    best_risk_reward_tier: str
    best_balanced_tier: str
    balance_scores: dict[str, float]
    ranking_by_expectancy: list[str]
    conclusions: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TieredSignalFrameworkReport:
    """Full tiered signal framework research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definitions: dict[str, dict[str, Any]]
    tiers: dict[str, dict[str, Any]]
    comparison: dict[str, Any]
    execution_time_seconds: float
    sample_signals: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TieredSignalFrameworkResearch:
    """Detect tiered institutional signals and measure research outcomes."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.filter_engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _minutes_per_bar(timeframe_label: str) -> int:
        return TIMEFRAME_MINUTES.get(timeframe_label, 5)

    @staticmethod
    def _bars_to_minutes(bars: int | None, timeframe_label: str) -> float | None:
        if bars is None:
            return None
        return round(bars * TieredSignalFrameworkResearch._minutes_per_bar(timeframe_label), 1)

    @staticmethod
    def _normalize_trend(value: Any) -> str:
        text = str(value or "").strip().upper()
        if text in {"BULLISH", "BEARISH"}:
            return text
        return "NEUTRAL"

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _research_months(start: date, end: date) -> float:
        return max((end - start).days / 30.44, 1.0)

    def _find_last_event_bar(
        self,
        frame: pd.DataFrame,
        column: str,
        before_index: int,
        lookback: int = SEQUENCE_LOOKBACK,
    ) -> int | None:
        start = max(0, before_index - lookback)
        for index in range(before_index, start - 1, -1):
            if self._is_active(frame.iloc[index].get(column)):
                return index
        return None

    def _displacement_at_bar(
        self,
        frame: pd.DataFrame,
        index: int,
        direction: str,
    ) -> DisplacementStrength:
        return LiquidityNarrativeEngine._displacement_strength_for_bar(
            frame.iloc[index],
            direction,
        )

    def _has_displacement_between(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        direction: str,
    ) -> tuple[bool, int]:
        for index in range(start_bar, end_bar + 1):
            strength = self._displacement_at_bar(frame, index, direction)
            if strength in {DisplacementStrength.MEDIUM, DisplacementStrength.STRONG}:
                return True, index
        return False, end_bar

    def _fvg_reclaimed_at_bar(
        self,
        frame: pd.DataFrame,
        index: int,
        direction: str,
    ) -> bool:
        window = self.narrative_engine._window(frame, index)
        fvg_context, fvg_bias = self.narrative_engine._fvg_context_state(frame, index, window)
        expected_bias = EVENT_MAP[direction]["fvg_bias"]
        return fvg_context == FvgContext.RECLAIMED and fvg_bias == expected_bias

    def _column_active_in_window(self, window: pd.DataFrame, column: str) -> bool:
        if column not in window.columns:
            return False
        return any(self._is_active(value) for value in window[column])

    @staticmethod
    def _dedupe_signals(signals: list[TierSignal]) -> list[TierSignal]:
        ranked = sorted(signals, key=lambda item: item.bos_bar)
        kept: list[TierSignal] = []
        last_bar = -MIN_EVENT_SEPARATION
        for signal in ranked:
            if signal.bos_bar - last_bar < MIN_EVENT_SEPARATION:
                continue
            kept.append(signal)
            last_bar = signal.bos_bar
        return kept

    def _detect_tier1(self, frame: pd.DataFrame, timeframe_label: str) -> list[TierSignal]:
        signals: list[TierSignal] = []
        for direction, mapping in EVENT_MAP.items():
            for bos_bar in range(len(frame)):
                if not self._is_active(frame.iloc[bos_bar].get(mapping["bos"])):
                    continue

                sweep_bar = self._find_last_event_bar(frame, mapping["sweep"], bos_bar - 1)
                choch_bar = self._find_last_event_bar(frame, mapping["choch"], bos_bar - 1)
                if sweep_bar is None or choch_bar is None:
                    continue
                if not (sweep_bar < choch_bar < bos_bar):
                    continue

                has_displacement, displacement_bar = self._has_displacement_between(
                    frame,
                    sweep_bar,
                    bos_bar,
                    direction,
                )
                if not has_displacement or not self._fvg_reclaimed_at_bar(frame, bos_bar, direction):
                    continue

                signals.append(
                    TierSignal(
                        tier="tier_1",
                        timeframe=timeframe_label,
                        direction=direction,
                        bos_bar=bos_bar,
                        bos_timestamp=str(frame.iloc[bos_bar]["Date"]),
                        choch_bar=choch_bar,
                        sweep_bar=sweep_bar,
                        displacement_bar=displacement_bar,
                    )
                )
        return self._dedupe_signals(signals)

    def _detect_tier2(self, frame: pd.DataFrame, timeframe_label: str) -> list[TierSignal]:
        signals: list[TierSignal] = []
        for direction, mapping in EVENT_MAP.items():
            for bos_bar in range(len(frame)):
                if not self._is_active(frame.iloc[bos_bar].get(mapping["bos"])):
                    continue

                choch_bar = self._find_last_event_bar(frame, mapping["choch"], bos_bar - 1)
                if choch_bar is None or not (choch_bar < bos_bar):
                    continue

                has_displacement, displacement_bar = self._has_displacement_between(
                    frame,
                    choch_bar,
                    bos_bar,
                    direction,
                )
                if not has_displacement or not self._fvg_reclaimed_at_bar(frame, bos_bar, direction):
                    continue

                signals.append(
                    TierSignal(
                        tier="tier_2",
                        timeframe=timeframe_label,
                        direction=direction,
                        bos_bar=bos_bar,
                        bos_timestamp=str(frame.iloc[bos_bar]["Date"]),
                        choch_bar=choch_bar,
                        displacement_bar=displacement_bar,
                    )
                )
        return self._dedupe_signals(signals)

    def _detect_tier3(self, frame: pd.DataFrame, timeframe_label: str) -> list[TierSignal]:
        signals: list[TierSignal] = []
        for index in range(len(frame)):
            row = frame.iloc[index]
            window = frame.iloc[max(0, index - SETUP_LOOKBACK_BARS) : index + 1]
            trend = self._normalize_trend(row.get("Trend"))

            for direction, mapping in EVENT_MAP.items():
                expected_trend = "BULLISH" if direction == "bullish" else "BEARISH"
                if trend != expected_trend:
                    continue
                if not self._is_active(row.get(mapping["bos"])):
                    continue
                if self._column_active_in_window(window, mapping["opposing_choch"]):
                    continue
                if not self._fvg_reclaimed_at_bar(frame, index, direction):
                    continue

                signals.append(
                    TierSignal(
                        tier="tier_3",
                        timeframe=timeframe_label,
                        direction=direction,
                        bos_bar=index,
                        bos_timestamp=str(row["Date"]),
                    )
                )
        return self._dedupe_signals(signals)

    def _risk_points(
        self,
        frame: pd.DataFrame,
        anchor_bar: int,
        direction: str,
    ) -> tuple[float, float]:
        row = frame.iloc[anchor_bar]
        entry = float(row["Close"])
        lookback = frame.iloc[max(0, anchor_bar - RISK_LOOKBACK) : anchor_bar + 1]

        if direction == "bullish":
            anchor = float(lookback["Low"].min())
            stop = anchor - SL_BUFFER_POINTS
            risk = max(entry - stop, MIN_RISK_POINTS)
        else:
            anchor = float(lookback["High"].max())
            stop = anchor + SL_BUFFER_POINTS
            risk = max(stop - entry, MIN_RISK_POINTS)

        return round(entry, 2), round(risk, 2)

    def _simulate_outcome(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
    ) -> TierSignalOutcome | None:
        anchor_bar = signal.bos_bar
        if anchor_bar >= len(frame) - 1:
            return None

        entry, risk = self._risk_points(frame, anchor_bar, signal.direction)
        end = min(len(frame) - 1, anchor_bar + FORWARD_BARS)

        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)

        if signal.direction == "bullish":
            stop = entry - risk
        else:
            stop = entry + risk

        mfe = 0.0
        expansion_bar: int | None = None
        stopped = False

        for index in range(anchor_bar + 1, end + 1):
            bar_high = float(highs.iloc[index])
            bar_low = float(lows.iloc[index])

            if signal.direction == "bullish":
                if bar_low <= stop:
                    stopped = True
                    break
                move = bar_high - entry
            else:
                if bar_high >= stop:
                    stopped = True
                    break
                move = entry - bar_low

            mfe = max(mfe, move)
            if move >= EXPANSION_THRESHOLD and expansion_bar is None:
                expansion_bar = index

        mfe = round(max(mfe, 0.0), 2)
        if stopped:
            pnl = -risk
            rr = -1.0
            win = False
        else:
            pnl = round(mfe, 2)
            rr = round(mfe / risk, 2) if risk > 0 else 0.0
            win = mfe >= risk

        expansion_offset = expansion_bar - anchor_bar if expansion_bar is not None else None

        return TierSignalOutcome(
            tier=signal.tier,
            timeframe=signal.timeframe,
            direction=signal.direction,
            bos_bar=anchor_bar,
            bos_timestamp=signal.bos_timestamp,
            risk_points=risk,
            forward_move_points=mfe,
            realized_pnl_points=pnl,
            realized_rr=rr,
            win=win,
            expansion_bar=expansion_bar,
            time_to_expansion_minutes=self._bars_to_minutes(
                expansion_offset,
                signal.timeframe,
            ),
        )

    def _tier_metrics(
        self,
        tier_key: str,
        outcomes: list[TierSignalOutcome],
        research_months: float,
    ) -> TierMetrics:
        definition = TIER_DEFINITIONS[tier_key]
        if not outcomes:
            return TierMetrics(
                tier=tier_key,
                label=definition["label"],
                components=definition["components"],
                signals=0,
                signals_per_month=0.0,
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                average_move_size=0.0,
                average_time_to_expansion_minutes=None,
            )

        pnls = [item.realized_pnl_points for item in outcomes]
        rrs = [item.realized_rr for item in outcomes]
        bullish = [item for item in outcomes if item.direction == "bullish"]
        bearish = [item for item in outcomes if item.direction == "bearish"]
        expansion_times = [
            item.time_to_expansion_minutes
            for item in outcomes
            if item.time_to_expansion_minutes is not None
        ]

        def direction_win_rate(bucket: list[TierSignalOutcome]) -> float:
            if not bucket:
                return 0.0
            return round(sum(1 for item in bucket if item.win) / len(bucket) * 100, 2)

        wins = sum(1 for item in outcomes if item.win)
        return TierMetrics(
            tier=tier_key,
            label=definition["label"],
            components=definition["components"],
            signals=len(outcomes),
            signals_per_month=round(len(outcomes) / research_months, 2),
            win_rate_pct=round(wins / len(outcomes) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            average_move_size=round(mean(item.forward_move_points for item in outcomes), 2),
            average_time_to_expansion_minutes=round(mean(expansion_times), 1)
            if expansion_times
            else None,
            win_rate_by_direction={
                "bullish": direction_win_rate(bullish),
                "bearish": direction_win_rate(bearish),
            },
        )

    def _balance_scores(self, tiers: dict[str, TierMetrics]) -> dict[str, float]:
        if not tiers:
            return {}

        max_frequency = max(item.signals_per_month for item in tiers.values()) or 1.0
        max_rr = max(item.average_rr for item in tiers.values()) or 1.0
        max_expectancy = max(item.expectancy for item in tiers.values()) or 1.0

        scores: dict[str, float] = {}
        for tier_key, metrics in tiers.items():
            accuracy = metrics.win_rate_pct / 100.0
            frequency = metrics.signals_per_month / max_frequency if max_frequency else 0.0
            rr = metrics.average_rr / max_rr if max_rr else 0.0
            expectancy = metrics.expectancy / max_expectancy if max_expectancy > 0 else 0.0
            scores[tier_key] = round(
                0.35 * accuracy + 0.25 * frequency + 0.20 * rr + 0.20 * expectancy,
                4,
            )
        return scores

    def _comparison(self, tiers: dict[str, TierMetrics]) -> TierComparison:
        balance_scores = self._balance_scores(tiers)
        ranked_expectancy = sorted(
            tiers.keys(),
            key=lambda key: (tiers[key].expectancy, tiers[key].win_rate_pct),
            reverse=True,
        )

        best_accuracy = max(tiers.keys(), key=lambda key: tiers[key].win_rate_pct)
        best_frequency = max(tiers.keys(), key=lambda key: tiers[key].signals_per_month)
        best_rr = max(tiers.keys(), key=lambda key: tiers[key].average_rr)
        best_balanced = max(balance_scores.keys(), key=lambda key: balance_scores[key])

        t1, t2, t3 = tiers["tier_1"], tiers["tier_2"], tiers["tier_3"]
        conclusions = [
            (
                f"{t1.label}: {t1.signals} signals ({t1.signals_per_month}/mo), "
                f"WR {t1.win_rate_pct}%, PF {t1.profit_factor}, "
                f"expectancy {t1.expectancy}, avg RR {t1.average_rr}."
            ),
            (
                f"{t2.label}: {t2.signals} signals ({t2.signals_per_month}/mo), "
                f"WR {t2.win_rate_pct}%, PF {t2.profit_factor}, "
                f"expectancy {t2.expectancy}, avg RR {t2.average_rr}."
            ),
            (
                f"{t3.label}: {t3.signals} signals ({t3.signals_per_month}/mo), "
                f"WR {t3.win_rate_pct}%, PF {t3.profit_factor}, "
                f"expectancy {t3.expectancy}, avg RR {t3.average_rr}."
            ),
            (
                f"Best accuracy: {best_accuracy} ({tiers[best_accuracy].win_rate_pct}%). "
                f"Best frequency: {best_frequency} ({tiers[best_frequency].signals_per_month}/mo). "
                f"Best RR: {best_rr} ({tiers[best_rr].average_rr}). "
                f"Best balanced: {best_balanced} (score {balance_scores[best_balanced]})."
            ),
        ]

        if best_balanced == "tier_2":
            conclusions.append(
                "Tier 2 provides the best balance: higher frequency than Tier 1 with "
                "stronger accuracy and RR than Tier 3."
            )
        elif best_balanced == "tier_1":
            conclusions.append(
                "Tier 1 provides the best balance despite lower frequency due to superior "
                "accuracy and risk-reward."
            )
        else:
            conclusions.append(
                "Tier 3 provides the best balance via high signal frequency; accept lower "
                "per-signal accuracy vs higher tiers."
            )

        return TierComparison(
            best_accuracy_tier=best_accuracy,
            best_frequency_tier=best_frequency,
            best_risk_reward_tier=best_rr,
            best_balanced_tier=best_balanced,
            balance_scores=balance_scores,
            ranking_by_expectancy=ranked_expectancy,
            conclusions=conclusions,
        )

    def run(self, metadata: dict[str, Any]) -> TieredSignalFrameworkReport:
        """Run tiered signal framework research."""
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
        research_months = self._research_months(start, end)

        tier_signals: dict[str, list[TierSignal]] = {
            "tier_1": [],
            "tier_2": [],
            "tier_3": [],
        }
        tier_outcomes: dict[str, list[TierSignalOutcome]] = {
            "tier_1": [],
            "tier_2": [],
            "tier_3": [],
        }

        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)

            detected = {
                "tier_1": self._detect_tier1(frame, timeframe_label),
                "tier_2": self._detect_tier2(frame, timeframe_label),
                "tier_3": self._detect_tier3(frame, timeframe_label),
            }

            for tier_key, signals in detected.items():
                tier_signals[tier_key].extend(signals)
                for signal in signals:
                    outcome = self._simulate_outcome(frame, signal)
                    if outcome:
                        tier_outcomes[tier_key].append(outcome)

        tier_metrics = {
            tier_key: self._tier_metrics(tier_key, tier_outcomes[tier_key], research_months)
            for tier_key in tier_signals
        }
        comparison = self._comparison(tier_metrics)

        return TieredSignalFrameworkReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definitions=TIER_DEFINITIONS,
            tiers={key: value.as_dict() for key, value in tier_metrics.items()},
            comparison=comparison.as_dict(),
            sample_signals={
                tier_key: [signal.as_dict() for signal in signals[:10]]
                for tier_key, signals in tier_signals.items()
            },
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tiered_signal_framework_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> TieredSignalFrameworkReport:
    """Run tiered signal framework research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise TieredSignalFrameworkError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = TieredSignalFrameworkResearch(
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
        "Tiered signal framework research completed: tier1=%s tier2=%s tier3=%s",
        report.tiers["tier_1"]["signals"],
        report.tiers["tier_2"]["signals"],
        report.tiers["tier_3"]["signals"],
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tiered_signal_framework_report()
        print("Tiered Signal Framework Research Summary")
        for tier_key in ("tier_1", "tier_2", "tier_3"):
            metrics = report.tiers[tier_key]
            print(
                f"{metrics['label']}: {metrics['signals']} signals "
                f"({metrics['signals_per_month']}/mo) "
                f"WR={metrics['win_rate_pct']}% PF={metrics['profit_factor']} "
                f"Exp={metrics['expectancy']} RR={metrics['average_rr']}"
            )
        comparison = report.comparison
        print(f"Best balanced tier: {comparison['best_balanced_tier']}")
        for note in comparison["conclusions"]:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except TieredSignalFrameworkError as exc:
        logger.error("Tiered signal framework error: %s", exc)
        print(f"Tiered signal framework error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected tiered signal framework failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
