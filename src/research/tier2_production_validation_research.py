"""
Tier-2 production validation research for SmartMoneyEngine.

Validates whether Tier-2 (Displacement + CHOCH + BOS + FVG Reclaim) should
become the production signal engine by comparing raw and filtered variants.
Research-only; no production logic or entries.
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
from typing import Any, Callable

import pandas as pd

from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.signal_funnel_analyzer import SignalFunnelAnalyzer
from src.research.tier2_trade_distribution_research import (
    TIER2_DEFINITION,
    Tier2TradeDistributionResearch,
)
from src.research.tiered_signal_framework_research import TierSignal, TieredSignalFrameworkResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tier2_production_validation.json"

MIN_INTELLIGENCE_SCORE = 65

VARIANTS: dict[str, dict[str, Any]] = {
    "raw_tier_2": {
        "label": "Raw Tier-2",
        "filters": [],
    },
    "tier_2_htf_alignment": {
        "label": "Tier-2 + HTF Alignment",
        "filters": ["HTF Alignment"],
    },
    "tier_2_mi_65": {
        "label": "Tier-2 + Market Intelligence >= 65",
        "filters": ["Market Intelligence >= 65"],
    },
    "tier_2_htf_mi_65": {
        "label": "Tier-2 + HTF Alignment + MI >= 65",
        "filters": ["HTF Alignment", "Market Intelligence >= 65"],
    },
}


class Tier2ProductionValidationError(Exception):
    """Raised when Tier-2 production validation fails."""


@dataclass(frozen=True)
class Tier2ProductionSignal:
    """Tier-2 signal with research outcome and filter context."""

    timeframe: str
    direction: str
    bos_bar: int
    bos_timestamp: str
    risk_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    htf_aligned: bool
    intelligence_score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VariantMetrics:
    """Performance metrics for one Tier-2 production variant."""

    variant_key: str
    label: str
    filters: list[str]
    signals: int
    signals_per_month: float
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    maximum_drawdown_points: float
    net_points: float
    streak_analysis: dict[str, Any]
    balance_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2ProductionValidationReport:
    """Full Tier-2 production validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    variants: dict[str, dict[str, Any]]
    comparison: dict[str, Any]
    recommended_production_version: str
    recommendation_rationale: list[str]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tier2ProductionValidationResearch:
    """Validate Tier-2 production variants against frequency, profit, drawdown."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
        min_intelligence_score: float = MIN_INTELLIGENCE_SCORE,
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.min_intelligence_score = min_intelligence_score
        self.tier_engine = TieredSignalFrameworkResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.distribution_engine = Tier2TradeDistributionResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.intelligence_engine = MarketIntelligenceEngine(symbol=symbol)
        self.funnel_analyzer = SignalFunnelAnalyzer(symbol=symbol)

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in pnls:
            cumulative = round(cumulative + pnl, 2)
            peak = max(peak, cumulative)
            max_drawdown = max(max_drawdown, round(peak - cumulative, 2))
        return max_drawdown

    def _htf_aligned(self, direction: str, htf_1d: str, htf_4h: str) -> bool:
        values = [
            self.funnel_analyzer._htf_trend_label(htf_1d),
            self.funnel_analyzer._htf_trend_label(htf_4h),
        ]
        if direction == "bullish":
            return (
                sum(1 for value in values if value == "Bullish") >= 1
                and not any(value == "Bearish" for value in values)
            )
        return (
            sum(1 for value in values if value == "Bearish") >= 1
            and not any(value == "Bullish" for value in values)
        )

    def _collect_signals(self, metadata: dict[str, Any]) -> list[Tier2ProductionSignal]:
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

        collected: list[Tier2ProductionSignal] = []

        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            htf_lookup = self.funnel_analyzer._build_htf_lookup(frame)
            intel_frame = self.intelligence_engine.enrich(frame)

            for signal in self.tier_engine._detect_tier2(frame, timeframe_label):
                detail = self.distribution_engine._simulate_detailed(frame, signal)
                if detail is None:
                    continue

                bar = signal.bos_bar
                htf_1d = str(htf_lookup.iloc[bar].get("HTF_1D_Trend", "SIDEWAYS"))
                htf_4h = str(htf_lookup.iloc[bar].get("HTF_4H_Trend", "SIDEWAYS"))
                intelligence = self.intelligence_engine.evaluate_bar(intel_frame, bar)

                risk = detail.risk_points
                rr = round(detail.realized_pnl_points / risk, 2) if risk > 0 else 0.0

                collected.append(
                    Tier2ProductionSignal(
                        timeframe=signal.timeframe,
                        direction=signal.direction,
                        bos_bar=bar,
                        bos_timestamp=signal.bos_timestamp,
                        risk_points=risk,
                        realized_pnl_points=detail.realized_pnl_points,
                        realized_rr=rr,
                        win=detail.win,
                        htf_aligned=self._htf_aligned(signal.direction, htf_1d, htf_4h),
                        intelligence_score=round(intelligence.intelligence_score, 2),
                    )
                )

        collected.sort(key=lambda item: pd.Timestamp(item.bos_timestamp))
        return collected

    @staticmethod
    def _variant_filter(variant_key: str) -> Callable[[Tier2ProductionSignal], bool]:
        if variant_key == "raw_tier_2":
            return lambda signal: True
        if variant_key == "tier_2_htf_alignment":
            return lambda signal: signal.htf_aligned
        if variant_key == "tier_2_mi_65":
            return lambda signal: signal.intelligence_score >= MIN_INTELLIGENCE_SCORE
        if variant_key == "tier_2_htf_mi_65":
            return lambda signal: (
                signal.htf_aligned and signal.intelligence_score >= MIN_INTELLIGENCE_SCORE
            )
        raise Tier2ProductionValidationError(f"Unknown variant: {variant_key}")

    def _variant_metrics(
        self,
        variant_key: str,
        signals: list[Tier2ProductionSignal],
        research_months: float,
        balance_inputs: dict[str, VariantMetrics] | None = None,
    ) -> VariantMetrics:
        definition = VARIANTS[variant_key]
        if not signals:
            metrics = VariantMetrics(
                variant_key=variant_key,
                label=definition["label"],
                filters=definition["filters"],
                signals=0,
                signals_per_month=0.0,
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                maximum_drawdown_points=0.0,
                net_points=0.0,
                streak_analysis=Tier2TradeDistributionResearch._streak_stats([]),
            )
            if balance_inputs is not None:
                metrics.balance_score = 0.0
                balance_inputs[variant_key] = metrics
            return metrics

        pnls = [signal.realized_pnl_points for signal in signals]
        rrs = [signal.realized_rr for signal in signals]
        wins = sum(1 for signal in signals if signal.win)

        metrics = VariantMetrics(
            variant_key=variant_key,
            label=definition["label"],
            filters=definition["filters"],
            signals=len(signals),
            signals_per_month=round(len(signals) / research_months, 2),
            win_rate_pct=round(wins / len(signals) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            net_points=round(sum(pnls), 2),
            streak_analysis=Tier2TradeDistributionResearch._streak_stats(
                [signal.win for signal in signals]
            ),
        )

        if balance_inputs is not None:
            balance_inputs[variant_key] = metrics
            self._apply_balance_scores(balance_inputs)

        return metrics

    @staticmethod
    def _apply_balance_scores(metrics_by_key: dict[str, VariantMetrics]) -> None:
        if not metrics_by_key:
            return

        max_frequency = max(item.signals_per_month for item in metrics_by_key.values()) or 1.0
        max_expectancy = max(item.expectancy for item in metrics_by_key.values()) or 1.0
        max_drawdown = max(item.maximum_drawdown_points for item in metrics_by_key.values()) or 1.0

        for metrics in metrics_by_key.values():
            frequency = metrics.signals_per_month / max_frequency if max_frequency else 0.0
            profitability = metrics.expectancy / max_expectancy if max_expectancy > 0 else 0.0
            drawdown_control = (
                1.0 - (metrics.maximum_drawdown_points / max_drawdown) if max_drawdown else 1.0
            )
            accuracy = metrics.win_rate_pct / 100.0
            metrics.balance_score = round(
                0.25 * frequency
                + 0.30 * profitability
                + 0.25 * drawdown_control
                + 0.20 * accuracy,
                4,
            )

    def _comparison(self, variants: dict[str, VariantMetrics]) -> dict[str, Any]:
        ranked_balance = sorted(
            variants.keys(),
            key=lambda key: (
                variants[key].balance_score,
                variants[key].expectancy,
                variants[key].signals_per_month,
            ),
            reverse=True,
        )
        ranked_expectancy = sorted(
            variants.keys(),
            key=lambda key: (variants[key].expectancy, variants[key].net_points),
            reverse=True,
        )
        ranked_frequency = sorted(
            variants.keys(),
            key=lambda key: variants[key].signals_per_month,
            reverse=True,
        )
        ranked_drawdown = sorted(
            variants.keys(),
            key=lambda key: variants[key].maximum_drawdown_points,
        )

        return {
            "best_frequency": ranked_frequency[0] if ranked_frequency else None,
            "best_expectancy": ranked_expectancy[0] if ranked_expectancy else None,
            "lowest_drawdown": ranked_drawdown[0] if ranked_drawdown else None,
            "best_balanced": ranked_balance[0] if ranked_balance else None,
            "ranking_by_balance_score": ranked_balance,
            "ranking_by_expectancy": ranked_expectancy,
            "ranking_by_frequency": ranked_frequency,
            "ranking_by_drawdown": ranked_drawdown,
        }

    def _recommendation(
        self,
        variants: dict[str, VariantMetrics],
        comparison: dict[str, Any],
    ) -> tuple[str, list[str]]:
        recommended = comparison["best_balanced"] or "raw_tier_2"
        best = variants[recommended]
        raw = variants["raw_tier_2"]

        rationale = [
            f"Recommended production version: {best.label} ({recommended}).",
            (
                f"Balance score {best.balance_score} vs raw {raw.balance_score}; "
                f"signals {best.signals_per_month}/mo vs {raw.signals_per_month}/mo; "
                f"expectancy {best.expectancy} vs {raw.expectancy}; "
                f"max DD {best.maximum_drawdown_points} vs {raw.maximum_drawdown_points}."
            ),
        ]

        if recommended == "tier_2_htf_mi_65":
            rationale.append(
                "Full stack (HTF + MI >= 65) delivers best risk-adjusted profile for production."
            )
        elif recommended == "tier_2_htf_alignment":
            rationale.append(
                "HTF alignment filter improves quality without requiring MI threshold."
            )
        elif recommended == "tier_2_mi_65":
            rationale.append(
                "Market Intelligence >= 65 filter provides best balance without HTF gating."
            )
        else:
            rationale.append(
                "Raw Tier-2 already outperforms filtered variants on the composite balance score."
            )

        return recommended, rationale

    def run(self, metadata: dict[str, Any]) -> Tier2ProductionValidationReport:
        """Run Tier-2 production validation research."""
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

        all_signals = self._collect_signals(metadata)
        balance_inputs: dict[str, VariantMetrics] = {}
        variant_metrics: dict[str, VariantMetrics] = {}

        for variant_key in VARIANTS:
            predicate = self._variant_filter(variant_key)
            filtered = [signal for signal in all_signals if predicate(signal)]
            variant_metrics[variant_key] = self._variant_metrics(
                variant_key,
                filtered,
                research_months,
                balance_inputs,
            )

        comparison = self._comparison(variant_metrics)
        recommended, rationale = self._recommendation(variant_metrics, comparison)

        conclusions = [
            f"Tier-2 production validation on {len(all_signals)} base signals.",
            *rationale,
        ]
        for key in comparison["ranking_by_balance_score"]:
            metrics = variant_metrics[key]
            conclusions.append(
                f"{metrics.label}: {metrics.signals_per_month}/mo, WR {metrics.win_rate_pct}%, "
                f"PF {metrics.profit_factor}, Exp {metrics.expectancy}, "
                f"DD {metrics.maximum_drawdown_points}, Net {metrics.net_points}."
            )

        return Tier2ProductionValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            variants={key: value.as_dict() for key, value in variant_metrics.items()},
            comparison=comparison,
            recommended_production_version=recommended,
            recommendation_rationale=rationale,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tier2_production_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Tier2ProductionValidationReport:
    """Run Tier-2 production validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Tier2ProductionValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = Tier2ProductionValidationResearch(
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
        "Tier-2 production validation completed: recommended=%s",
        report.recommended_production_version,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tier2_production_validation_report()
        print("Tier-2 Production Validation Summary")
        for key, metrics in report.variants.items():
            print(
                f"{metrics['label']}: {metrics['signals_per_month']}/mo "
                f"WR={metrics['win_rate_pct']}% PF={metrics['profit_factor']} "
                f"Exp={metrics['expectancy']} DD={metrics['maximum_drawdown_points']} "
                f"Net={metrics['net_points']} Balance={metrics['balance_score']}"
            )
        print(f"Recommended: {report.recommended_production_version}")
        for note in report.recommendation_rationale:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except Tier2ProductionValidationError as exc:
        logger.error("Tier-2 production validation error: %s", exc)
        print(f"Tier-2 production validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Tier-2 production validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
