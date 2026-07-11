"""
Tier-2 signal ranking validation research for SmartMoneyEngine.

Validates whether Institutional Quality Score improves Tier-2 BOS Close
entries. Research-only; no new setups, entries, or production changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.institutional_quality_validation_research import (
    SCORE_BUCKETS,
    InstitutionalQualityValidationResearch,
)
from src.research.tier2_entry_optimization_research import Tier2EntryOptimizationResearch
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.tiered_signal_framework_research import TieredSignalFrameworkResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tier2_signal_ranking_validation.json"

COHORT_FRACTION = 0.20
MIN_THRESHOLD_SIGNALS = 20
THRESHOLD_CANDIDATES = (0, 40, 60, 70, 80)


class Tier2SignalRankingValidationError(Exception):
    """Raised when Tier-2 signal ranking validation fails."""


@dataclass(frozen=True)
class RankedBosCloseSignal:
    """Tier-2 BOS close signal with quality score and outcome."""

    bos_timestamp: str
    timeframe: str
    direction: str
    quality_score: int
    realized_pnl_points: float
    realized_rr: float
    win: bool
    mfe_points: float
    mae_points: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RankingCohortMetrics:
    """Performance metrics for one ranked cohort."""

    label: str
    signals: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float
    average_mfe: float
    average_mae: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2SignalRankingValidationReport:
    """Full Tier-2 signal ranking validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    entry_method: str
    scoring_model: dict[str, Any]
    total_signals: int
    score_buckets: dict[str, dict[str, Any]]
    top_20_pct_quality: dict[str, Any]
    bottom_20_pct_quality: dict[str, Any]
    quality_score_improves_outcomes: bool
    optimal_production_threshold: int
    expected_signals_per_month: float
    expected_yearly_signals: float
    threshold_analysis: dict[str, dict[str, Any]]
    production_recommendation: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tier2SignalRankingValidationResearch:
    """Validate institutional quality score ranking for BOS close entries."""

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
        self.quality_engine = InstitutionalQualityValidationResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.entry_engine = Tier2EntryOptimizationResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.edge_engine = self.quality_engine.edge_engine

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _research_months(metadata: dict[str, Any]) -> float:
        start = date.fromisoformat(metadata["start_date"])
        end = date.fromisoformat(metadata["end_date"])
        return max((end - start).days / 30.44, 1.0)

    @staticmethod
    def _bucket_for_score(score: int) -> str:
        return InstitutionalQualityValidationResearch._bucket_for_score(score)

    def _collect_ranked_signals(self, metadata: dict[str, Any]) -> list[RankedBosCloseSignal]:
        from datetime import timedelta

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        ranked: list[RankedBosCloseSignal] = []

        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)

            for signal in self.tier_engine._detect_tier2(frame, timeframe_label):
                outcome = self.entry_engine.evaluate_method("A_bos_close", frame, signal)
                if not outcome.entry_triggered:
                    continue

                record = self.edge_engine._extract_features(
                    frame,
                    signal,
                    outcome.realized_pnl_points,
                    outcome.win,
                    outcome.risk_points,
                )
                if record is None:
                    continue

                score, _, _ = self.quality_engine.compute_quality_score(record)

                ranked.append(
                    RankedBosCloseSignal(
                        bos_timestamp=signal.bos_timestamp,
                        timeframe=signal.timeframe,
                        direction=signal.direction,
                        quality_score=score,
                        realized_pnl_points=outcome.realized_pnl_points,
                        realized_rr=outcome.realized_rr,
                        win=outcome.win,
                        mfe_points=outcome.mfe_points,
                        mae_points=outcome.mae_points,
                    )
                )

        ranked.sort(key=lambda item: pd.Timestamp(item.bos_timestamp))
        return ranked

    def _metrics(self, label: str, signals: list[RankedBosCloseSignal]) -> RankingCohortMetrics:
        if not signals:
            return RankingCohortMetrics(
                label=label,
                signals=0,
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                net_points=0.0,
                maximum_drawdown_points=0.0,
                average_mfe=0.0,
                average_mae=0.0,
            )

        pnls = [signal.realized_pnl_points for signal in signals]
        rrs = [signal.realized_rr for signal in signals]
        wins = sum(1 for signal in signals if signal.win)

        return RankingCohortMetrics(
            label=label,
            signals=len(signals),
            win_rate_pct=round(wins / len(signals) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            average_mfe=round(mean(signal.mfe_points for signal in signals), 2),
            average_mae=round(mean(signal.mae_points for signal in signals), 2),
        )

    def _optimal_threshold(
        self,
        signals: list[RankedBosCloseSignal],
        baseline: RankingCohortMetrics,
    ) -> tuple[int, dict[str, RankingCohortMetrics]]:
        threshold_metrics: dict[str, RankingCohortMetrics] = {}
        for threshold in THRESHOLD_CANDIDATES:
            filtered = [s for s in signals if s.quality_score >= threshold]
            threshold_metrics[str(threshold)] = self._metrics(f"Score >= {threshold}", filtered)

        ranked = sorted(
            [t for t in THRESHOLD_CANDIDATES if t > 0],
            key=lambda threshold: (
                threshold_metrics[str(threshold)].expectancy,
                threshold_metrics[str(threshold)].profit_factor or 0,
                threshold_metrics[str(threshold)].win_rate_pct,
                -threshold_metrics[str(threshold)].maximum_drawdown_points,
            ),
            reverse=True,
        )

        for threshold in ranked:
            metrics = threshold_metrics[str(threshold)]
            if metrics.signals < MIN_THRESHOLD_SIGNALS:
                continue
            if (
                metrics.expectancy >= baseline.expectancy
                and (metrics.profit_factor or 0) >= (baseline.profit_factor or 0)
                and metrics.maximum_drawdown_points <= baseline.maximum_drawdown_points
            ):
                return threshold, threshold_metrics

        for threshold in ranked:
            if threshold_metrics[str(threshold)].signals >= MIN_THRESHOLD_SIGNALS:
                return threshold, threshold_metrics

        return 60, threshold_metrics

    def _production_recommendation(
        self,
        threshold: int,
        threshold_metrics: dict[str, RankingCohortMetrics],
        research_months: float,
        quality_improves: bool,
        top: RankingCohortMetrics,
        bottom: RankingCohortMetrics,
    ) -> dict[str, Any]:
        at_threshold = threshold_metrics.get(str(threshold)) or threshold_metrics[str(60)]
        monthly = round(at_threshold.signals / research_months, 2)
        yearly = round(monthly * 12, 1)

        if quality_improves and at_threshold.expectancy > 0:
            action = (
                f"Deploy Tier-2 BOS Close with Institutional Quality Score >= {threshold}."
            )
        elif threshold > 0:
            action = (
                f"Use quality score >= {threshold} as a research filter; "
                "monitor live frequency before full production."
            )
        else:
            action = "Use raw Tier-2 BOS Close without quality filter."

        return {
            "optimal_production_threshold": threshold,
            "expected_signals_per_month": monthly,
            "expected_yearly_signals": yearly,
            "filtered_expectancy": at_threshold.expectancy,
            "filtered_win_rate_pct": at_threshold.win_rate_pct,
            "filtered_profit_factor": at_threshold.profit_factor,
            "filtered_max_drawdown_points": at_threshold.maximum_drawdown_points,
            "top_20_pct_expectancy": top.expectancy,
            "bottom_20_pct_expectancy": bottom.expectancy,
            "expectancy_delta_top_vs_bottom": round(top.expectancy - bottom.expectancy, 2),
            "recommendation": action,
        }

    def run(self, metadata: dict[str, Any]) -> Tier2SignalRankingValidationReport:
        """Run Tier-2 signal ranking validation."""
        started = time.perf_counter()
        research_months = self._research_months(metadata)

        signals = self._collect_ranked_signals(metadata)
        if not signals:
            raise Tier2SignalRankingValidationError("No ranked Tier-2 BOS close signals found.")

        baseline = self._metrics("All Tier-2 BOS Close", signals)

        bucket_groups: dict[str, list[RankedBosCloseSignal]] = {
            label: [] for label, _, _ in SCORE_BUCKETS
        }
        for signal in signals:
            bucket_groups[self._bucket_for_score(signal.quality_score)].append(signal)

        score_buckets = {
            label: self._metrics(label, group).as_dict()
            for label, group in bucket_groups.items()
        }

        cohort_size = max(1, int(len(signals) * COHORT_FRACTION))
        by_score = sorted(signals, key=lambda item: item.quality_score, reverse=True)
        top_signals = by_score[:cohort_size]
        bottom_signals = by_score[-cohort_size:]

        top_metrics = self._metrics("Top 20% Quality", top_signals)
        bottom_metrics = self._metrics("Bottom 20% Quality", bottom_signals)

        quality_improves = (
            top_metrics.expectancy > bottom_metrics.expectancy
            and top_metrics.win_rate_pct > bottom_metrics.win_rate_pct
            and (top_metrics.profit_factor or 0) >= (bottom_metrics.profit_factor or 0)
        )

        optimal_threshold, threshold_analysis_raw = self._optimal_threshold(signals, baseline)
        threshold_analysis = {
            key: value.as_dict() for key, value in threshold_analysis_raw.items()
        }

        production = self._production_recommendation(
            optimal_threshold,
            threshold_analysis_raw,
            research_months,
            quality_improves,
            top_metrics,
            bottom_metrics,
        )

        conclusions = [
            (
                f"Validated {len(signals)} Tier-2 BOS Close signals with Institutional Quality Score."
            ),
            (
                f"Top 20% quality: expectancy {top_metrics.expectancy}, WR {top_metrics.win_rate_pct}%, "
                f"net {top_metrics.net_points}."
            ),
            (
                f"Bottom 20% quality: expectancy {bottom_metrics.expectancy}, WR {bottom_metrics.win_rate_pct}%, "
                f"net {bottom_metrics.net_points}."
            ),
            f"Quality score improves outcomes: {quality_improves}.",
            (
                f"Optimal production threshold: Score >= {optimal_threshold} "
                f"({production['expected_signals_per_month']}/mo, "
                f"{production['expected_yearly_signals']}/yr)."
            ),
            production["recommendation"],
        ]

        return Tier2SignalRankingValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            entry_method="BOS Close Entry",
            scoring_model=self.quality_engine.scoring_model_definition(),
            total_signals=len(signals),
            score_buckets=score_buckets,
            top_20_pct_quality=top_metrics.as_dict(),
            bottom_20_pct_quality=bottom_metrics.as_dict(),
            quality_score_improves_outcomes=quality_improves,
            optimal_production_threshold=optimal_threshold,
            expected_signals_per_month=production["expected_signals_per_month"],
            expected_yearly_signals=production["expected_yearly_signals"],
            threshold_analysis=threshold_analysis,
            production_recommendation=production,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tier2_signal_ranking_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Tier2SignalRankingValidationReport:
    """Run Tier-2 signal ranking validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Tier2SignalRankingValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = Tier2SignalRankingValidationResearch(
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
        "Tier-2 signal ranking validation completed: threshold>=%s",
        report.optimal_production_threshold,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tier2_signal_ranking_validation_report()
        print("Tier-2 Signal Ranking Validation Summary")
        print(f"Signals: {report.total_signals} | Entry: {report.entry_method}")
        for label, metrics in report.score_buckets.items():
            print(
                f"  {label}: n={metrics['signals']} WR={metrics['win_rate_pct']}% "
                f"Exp={metrics['expectancy']} Net={metrics['net_points']}"
            )
        print(f"Top 20% Exp={report.top_20_pct_quality['expectancy']}")
        print(f"Bottom 20% Exp={report.bottom_20_pct_quality['expectancy']}")
        print(f"Optimal threshold: >= {report.optimal_production_threshold}")
        print(f"Expected: {report.expected_signals_per_month}/mo, {report.expected_yearly_signals}/yr")
        print(report.production_recommendation["recommendation"])
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except Tier2SignalRankingValidationError as exc:
        logger.error("Tier-2 signal ranking validation error: %s", exc)
        print(f"Tier-2 signal ranking validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Tier-2 signal ranking validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
