"""
Tier-2 regime classification research for SmartMoneyEngine.

Classifies existing Tier-2 signals into market regimes and compares performance
by regime. Research-only; no signal logic, setups, or entry changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import DisplacementStrength, LiquidityNarrativeEngine
from src.research.filter_research_engine import (
    FilterContextBuilder,
    FilterResearchEngine,
    RESEARCH_DAYS,
    _json_safe,
)
from src.research.institutional_quality_validation_research import InstitutionalQualityValidationResearch
from src.research.signal_funnel_analyzer import SignalFunnelAnalyzer
from src.research.tier2_entry_optimization_research import Tier2EntryOptimizationResearch
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.tiered_signal_framework_research import TierSignal, TieredSignalFrameworkResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tier2_regime_classification.json"

MIN_PRODUCTION_SIGNALS = 20
PRIOR_DAY_LOOKBACK_BARS = 500


class Tier2Regime(str, Enum):
    """Market regime labels for Tier-2 signal classification."""

    TREND_CONTINUATION = "Trend Continuation"
    LIQUIDITY_REVERSAL = "Liquidity Reversal"
    RANGE_EXPANSION = "Range Expansion"
    SESSION_BREAKOUT = "Session Breakout"
    HTF_REVERSAL = "HTF Reversal"


REGIME_ORDER = tuple(regime.value for regime in Tier2Regime)


class Tier2RegimeClassificationError(Exception):
    """Raised when Tier-2 regime classification research fails."""


@dataclass(frozen=True)
class ClassifiedTier2Signal:
    """Tier-2 signal with regime label and BOS close outcome."""

    bos_timestamp: str
    timeframe: str
    direction: str
    regime: str
    session: str
    ltf_trend: str
    htf_1h_trend: str
    realized_pnl_points: float
    realized_rr: float
    win: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeMetrics:
    """Performance metrics for one market regime."""

    regime: str
    rank: int
    signals: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2RegimeClassificationReport:
    """Full Tier-2 regime classification research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    entry_method: str
    classification_rules: dict[str, str]
    total_signals: int
    regime_metrics: dict[str, dict[str, Any]]
    regime_ranking: list[dict[str, Any]]
    highest_frequency_regime: str
    highest_expectancy_regime: str
    best_production_regime: str
    production_recommendation: dict[str, Any]
    regime_distribution: dict[str, int]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tier2RegimeClassificationResearch:
    """Classify Tier-2 signals into market regimes and compare outcomes."""

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
        self.funnel_analyzer = SignalFunnelAnalyzer(symbol=symbol)
        self.filter_engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self._htf_1h_lookup: pd.DataFrame | None = None

    @staticmethod
    def classification_rules_definition() -> dict[str, str]:
        return {
            Tier2Regime.SESSION_BREAKOUT.value: (
                "Opening session (09:15-10:30 IST) and BOS close breaks prior "
                "session day high (bullish) or low (bearish)."
            ),
            Tier2Regime.HTF_REVERSAL.value: (
                "Higher-timeframe trend opposes signal direction on 1H/4H/1D "
                "(for 1H signals, 4H and 1D only). Requires HTF opposition "
                "without HTF alignment."
            ),
            Tier2Regime.LIQUIDITY_REVERSAL.value: (
                "Opposing liquidity swept within narrative lookback: sell-side "
                "sweep before bullish signal or buy-side sweep before bearish."
            ),
            Tier2Regime.RANGE_EXPANSION.value: (
                "LTF trend SIDEWAYS or strength <= 1 with medium/strong "
                "displacement at BOS."
            ),
            Tier2Regime.TREND_CONTINUATION.value: (
                "Default when LTF trend aligns with signal direction "
                "(BULLISH+bullish or BEARISH+bearish)."
            ),
            "priority_order": (
                "Session Breakout > HTF Reversal > Liquidity Reversal > "
                "Range Expansion > Trend Continuation"
            ),
        }

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    def _ensure_htf_1h_lookup(self, start: date, end: date) -> pd.DataFrame:
        if self._htf_1h_lookup is not None:
            return self._htf_1h_lookup

        path = self.tier_engine.filter_engine._ensure_pipeline("1H", start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        lookup = frame[["Date", "Trend"]].copy()
        lookup["timestamp"] = pd.to_datetime(lookup["Date"], errors="coerce")
        if lookup["timestamp"].dt.tz is None:
            lookup["timestamp"] = lookup["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            lookup["timestamp"] = lookup["timestamp"].dt.tz_convert("Asia/Kolkata")
        lookup["HTF_1H_Trend"] = lookup["Trend"].astype(str).str.upper()
        self._htf_1h_lookup = lookup[["timestamp", "HTF_1H_Trend"]].sort_values("timestamp")
        return self._htf_1h_lookup

    @staticmethod
    def _htf_1h_trend_at(
        lookup: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> str:
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Kolkata")
        else:
            ts = ts.tz_convert("Asia/Kolkata")
        merged = pd.merge_asof(
            pd.DataFrame({"timestamp": [ts]}),
            lookup.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        return str(merged.iloc[0].get("HTF_1H_Trend", "SIDEWAYS")).upper()

    @staticmethod
    def _htf_opposes(direction: str, htf_1h_trend: str) -> bool:
        if direction == "bullish":
            return htf_1h_trend == "BEARISH"
        if direction == "bearish":
            return htf_1h_trend == "BULLISH"
        return False

    @staticmethod
    def _ltf_trend_aligned(direction: str, ltf_trend: str) -> bool:
        if direction == "bullish":
            return ltf_trend == "BULLISH"
        if direction == "bearish":
            return ltf_trend == "BEARISH"
        return False

    def _breaks_prior_session_day(
        self,
        frame: pd.DataFrame,
        bos_bar: int,
        direction: str,
    ) -> bool:
        timestamps = pd.to_datetime(frame["Date"], errors="coerce")
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("Asia/Kolkata")
        else:
            timestamps = timestamps.dt.tz_convert("Asia/Kolkata")

        current_day = timestamps.iloc[bos_bar].date()
        prior_mask = timestamps.dt.date < current_day
        if not prior_mask.any():
            return False

        prior_frame = frame.loc[prior_mask].tail(PRIOR_DAY_LOOKBACK_BARS)
        if prior_frame.empty:
            return False

        prior_day = timestamps.loc[prior_frame.index[-1]].date()
        day_mask = timestamps.dt.date == prior_day
        day_frame = frame.loc[day_mask]
        if day_frame.empty:
            return False

        prior_high = float(day_frame["High"].astype(float).max())
        prior_low = float(day_frame["Low"].astype(float).min())
        close = float(frame.iloc[bos_bar]["Close"])

        if direction == "bullish":
            return close > prior_high
        return close < prior_low

    def _liquidity_reversal(
        self,
        frame: pd.DataFrame,
        bos_bar: int,
        direction: str,
    ) -> bool:
        window = self.narrative_engine._window(frame, bos_bar)
        liquidity = self.narrative_engine._liquidity_events(frame, bos_bar, window)
        if direction == "bullish":
            return liquidity.sell_side_liquidity_taken
        return liquidity.buy_side_liquidity_taken

    def _range_expansion(
        self,
        frame: pd.DataFrame,
        bos_bar: int,
    ) -> bool:
        row = frame.iloc[bos_bar]
        trend = str(row.get("Trend", "SIDEWAYS")).upper()
        strength = LiquidityNarrativeEngine._to_float(row.get("Trend_Strength")) or 0.0
        window = self.narrative_engine._window(frame, bos_bar)
        displacement = self.narrative_engine._displacement_strength(window)
        sideways = trend == "SIDEWAYS" or strength <= 1
        expanding = displacement in {
            DisplacementStrength.MEDIUM,
            DisplacementStrength.STRONG,
        }
        return sideways and expanding

    def _htf_reversal(
        self,
        direction: str,
        timeframe: str,
        htf_1h: str,
        htf_4h: str,
        htf_1d: str,
    ) -> bool:
        if timeframe == "1H":
            trends = [htf_4h, htf_1d]
        else:
            trends = [htf_1h, htf_4h, htf_1d]

        labels = [self.funnel_analyzer._htf_trend_label(trend) for trend in trends]
        if direction == "bullish":
            opposed = any(label == "Bearish" for label in labels)
            aligned = (
                sum(1 for label in labels if label == "Bullish") >= 1
                and not any(label == "Bearish" for label in labels)
            )
        else:
            opposed = any(label == "Bullish" for label in labels)
            aligned = (
                sum(1 for label in labels if label == "Bearish") >= 1
                and not any(label == "Bullish" for label in labels)
            )
        return opposed and not aligned

    def classify_regime(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
        session: str,
        htf_1h: str,
        htf_4h: str,
        htf_1d: str,
    ) -> str:
        """Assign one regime using priority-ordered rules."""
        bos_bar = signal.bos_bar
        direction = signal.direction
        row = frame.iloc[bos_bar]
        ltf_trend = str(row.get("Trend", "SIDEWAYS")).upper()

        if session == "Opening" and self._breaks_prior_session_day(frame, bos_bar, direction):
            return Tier2Regime.SESSION_BREAKOUT.value

        if self._htf_reversal(direction, signal.timeframe, htf_1h, htf_4h, htf_1d):
            return Tier2Regime.HTF_REVERSAL.value

        if self._liquidity_reversal(frame, bos_bar, direction):
            return Tier2Regime.LIQUIDITY_REVERSAL.value

        if self._range_expansion(frame, bos_bar):
            return Tier2Regime.RANGE_EXPANSION.value

        if self._ltf_trend_aligned(direction, ltf_trend):
            return Tier2Regime.TREND_CONTINUATION.value

        return Tier2Regime.RANGE_EXPANSION.value

    def _collect_classified_signals(
        self,
        metadata: dict[str, Any],
    ) -> list[ClassifiedTier2Signal]:
        from datetime import timedelta

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        htf_lookup = self._ensure_htf_1h_lookup(start, end)
        classified: list[ClassifiedTier2Signal] = []

        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            timestamps = pd.to_datetime(frame["Date"], errors="coerce")
            htf_frame_lookup = self.funnel_analyzer._build_htf_lookup(frame)

            for signal in self.tier_engine._detect_tier2(frame, timeframe_label):
                outcome = self.entry_engine.evaluate_method("A_bos_close", frame, signal)
                if not outcome.entry_triggered:
                    continue

                bar = signal.bos_bar
                ts = pd.Timestamp(timestamps.iloc[bar])
                session = FilterContextBuilder._session_label(ts)
                htf_1h = self._htf_1h_trend_at(htf_lookup, ts)
                htf_4h = str(htf_frame_lookup.iloc[bar].get("HTF_4H_Trend", "SIDEWAYS")).upper()
                htf_1d = str(htf_frame_lookup.iloc[bar].get("HTF_1D_Trend", "SIDEWAYS")).upper()
                row = frame.iloc[bar]
                regime = self.classify_regime(
                    frame,
                    signal,
                    session,
                    htf_1h,
                    htf_4h,
                    htf_1d,
                )

                classified.append(
                    ClassifiedTier2Signal(
                        bos_timestamp=signal.bos_timestamp,
                        timeframe=signal.timeframe,
                        direction=signal.direction,
                        regime=regime,
                        session=session,
                        ltf_trend=str(row.get("Trend", "SIDEWAYS")).upper(),
                        htf_1h_trend=htf_1h,
                        realized_pnl_points=outcome.realized_pnl_points,
                        realized_rr=outcome.realized_rr,
                        win=outcome.win,
                    )
                )

        classified.sort(key=lambda item: pd.Timestamp(item.bos_timestamp))
        return classified

    def _regime_metrics(self, regime: str, signals: list[ClassifiedTier2Signal]) -> RegimeMetrics:
        if not signals:
            return RegimeMetrics(
                regime=regime,
                rank=0,
                signals=0,
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                net_points=0.0,
                maximum_drawdown_points=0.0,
            )

        pnls = [signal.realized_pnl_points for signal in signals]
        rrs = [signal.realized_rr for signal in signals]
        wins = sum(1 for signal in signals if signal.win)

        return RegimeMetrics(
            regime=regime,
            rank=0,
            signals=len(signals),
            win_rate_pct=round(wins / len(signals) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
        )

    @staticmethod
    def _production_score(metrics: RegimeMetrics) -> float:
        pf = metrics.profit_factor or 0.0
        if pf == float("inf"):
            pf = 10.0
        return round(
            metrics.expectancy * 0.45
            + pf * 18.0
            + metrics.win_rate_pct * 0.35
            - metrics.maximum_drawdown_points * 0.015,
            4,
        )

    def _rank_regimes(
        self,
        metrics_by_regime: dict[str, RegimeMetrics],
    ) -> list[RegimeMetrics]:
        ranked = sorted(
            metrics_by_regime.values(),
            key=lambda item: (
                item.expectancy,
                item.profit_factor or 0,
                item.win_rate_pct,
                -item.maximum_drawdown_points,
                item.net_points,
            ),
            reverse=True,
        )
        for index, metrics in enumerate(ranked, start=1):
            metrics.rank = index
        return ranked

    def _best_production_regime(self, ranked: list[RegimeMetrics]) -> RegimeMetrics:
        eligible = [item for item in ranked if item.signals >= MIN_PRODUCTION_SIGNALS]
        if not eligible:
            return ranked[0]

        return max(
            eligible,
            key=lambda item: (
                self._production_score(item),
                item.expectancy,
                item.profit_factor or 0,
            ),
        )

    def run(self, metadata: dict[str, Any]) -> Tier2RegimeClassificationReport:
        """Run Tier-2 regime classification research."""
        started = time.perf_counter()

        signals = self._collect_classified_signals(metadata)
        if not signals:
            raise Tier2RegimeClassificationError("No classified Tier-2 signals found.")

        groups: dict[str, list[ClassifiedTier2Signal]] = {
            regime: [] for regime in REGIME_ORDER
        }
        for signal in signals:
            groups[signal.regime].append(signal)

        metrics_by_regime = {
            regime: self._regime_metrics(regime, group)
            for regime, group in groups.items()
        }
        ranked = self._rank_regimes(metrics_by_regime)

        frequency_leader = max(ranked, key=lambda item: item.signals)
        expectancy_leader = ranked[0]
        production_leader = self._best_production_regime(ranked)

        distribution = {regime: metrics_by_regime[regime].signals for regime in REGIME_ORDER}

        production_recommendation = {
            "best_production_regime": production_leader.regime,
            "production_score": self._production_score(production_leader),
            "signals": production_leader.signals,
            "expectancy": production_leader.expectancy,
            "profit_factor": production_leader.profit_factor,
            "win_rate_pct": production_leader.win_rate_pct,
            "maximum_drawdown_points": production_leader.maximum_drawdown_points,
            "recommendation": (
                f"Prioritize Tier-2 BOS Close signals classified as "
                f"'{production_leader.regime}' for production deployment."
            ),
        }

        conclusions = [
            f"Classified {len(signals)} Tier-2 BOS Close signals across {len(REGIME_ORDER)} regimes.",
            (
                f"Highest frequency: {frequency_leader.regime} "
                f"({frequency_leader.signals} signals, {frequency_leader.signals / len(signals) * 100:.1f}%)."
            ),
            (
                f"Highest expectancy: {expectancy_leader.regime} "
                f"(expectancy {expectancy_leader.expectancy}, PF {expectancy_leader.profit_factor})."
            ),
            (
                f"Best production regime: {production_leader.regime} "
                f"(score {self._production_score(production_leader)}, "
                f"n={production_leader.signals})."
            ),
            production_recommendation["recommendation"],
        ]

        return Tier2RegimeClassificationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            entry_method="BOS Close Entry",
            classification_rules=self.classification_rules_definition(),
            total_signals=len(signals),
            regime_metrics={
                regime: metrics_by_regime[regime].as_dict() for regime in REGIME_ORDER
            },
            regime_ranking=[item.as_dict() for item in ranked],
            highest_frequency_regime=frequency_leader.regime,
            highest_expectancy_regime=expectancy_leader.regime,
            best_production_regime=production_leader.regime,
            production_recommendation=production_recommendation,
            regime_distribution=distribution,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tier2_regime_classification_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Tier2RegimeClassificationReport:
    """Run Tier-2 regime classification and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Tier2RegimeClassificationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = Tier2RegimeClassificationResearch(
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
        "Tier-2 regime classification completed: best=%s",
        report.best_production_regime,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tier2_regime_classification_report()
        print("Tier-2 Regime Classification Summary")
        print(f"Signals: {report.total_signals} | Entry: {report.entry_method}")
        for item in report.regime_ranking:
            print(
                f"  #{item['rank']} {item['regime']}: n={item['signals']} "
                f"WR={item['win_rate_pct']}% Exp={item['expectancy']} "
                f"Net={item['net_points']}"
            )
        print(f"Highest frequency: {report.highest_frequency_regime}")
        print(f"Highest expectancy: {report.highest_expectancy_regime}")
        print(f"Best production: {report.best_production_regime}")
        print(report.production_recommendation["recommendation"])
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except Tier2RegimeClassificationError as exc:
        logger.error("Tier-2 regime classification error: %s", exc)
        print(f"Tier-2 regime classification error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Tier-2 regime classification failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
