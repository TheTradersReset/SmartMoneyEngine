"""
Institutional quality score validation research for SmartMoneyEngine.

Validates the Tier-2 Institutional Quality Score across all Raw Tier-2 signals
by score bucket and production threshold. Research-only; no production logic,
indicators, or entries.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.institutional_edge_extraction_research import (
    EdgeFeatureRecord,
    InstitutionalEdgeExtractionResearch,
)
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_quality_validation.json"

SCORE_COMPONENT_POINTS = 20
THRESHOLD_CANDIDATES = (60, 70, 80)

SCORE_BUCKETS = (
    ("0-20", 0, 20),
    ("20-40", 20, 40),
    ("40-60", 40, 60),
    ("60-80", 60, 80),
    ("80-100", 80, 101),
)


class InstitutionalQualityValidationError(Exception):
    """Raised when institutional quality validation fails."""


@dataclass(frozen=True)
class ScoredTier2Signal:
    """Tier-2 signal with institutional quality score and outcome."""

    bos_timestamp: str
    timeframe: str
    direction: str
    quality_score: int
    component_hits: dict[str, bool]
    component_points: dict[str, int]
    realized_pnl_points: float
    realized_rr: float
    win: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CohortMetrics:
    """Performance metrics for one score cohort."""

    label: str
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
class InstitutionalQualityValidationReport:
    """Full institutional quality score validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    scoring_model: dict[str, Any]
    total_signals: int
    score_buckets: dict[str, dict[str, Any]]
    threshold_comparison: dict[str, dict[str, Any]]
    unfiltered_baseline: dict[str, Any]
    higher_score_improves_outcomes: bool
    recommended_production_threshold: int
    recommendation_rationale: list[str]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalQualityValidationResearch:
    """Validate Tier-2 institutional quality score predictive power."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.edge_engine = InstitutionalEdgeExtractionResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _bucket_for_score(score: int) -> str:
        for label, lower, upper in SCORE_BUCKETS:
            if lower <= score < upper:
                return label
        return SCORE_BUCKETS[-1][0]

    @staticmethod
    def scoring_model_definition() -> dict[str, Any]:
        return {
            "model_name": "Tier-2 Institutional Quality Score",
            "max_score": 100,
            "components": [
                {
                    "name": "Strong Displacement",
                    "points": SCORE_COMPONENT_POINTS,
                    "condition": "Displacement strength is Strong",
                },
                {
                    "name": "CHOCH to BOS Timing",
                    "points": SCORE_COMPONENT_POINTS,
                    "condition": "CHOCH->BOS between 90 and 240 minutes (inclusive lower, exclusive upper at 240)",
                },
                {
                    "name": "FVG Retests",
                    "points": SCORE_COMPONENT_POINTS,
                    "condition": "Exactly 1 FVG retest before BOS",
                },
                {
                    "name": "FVG Freshness",
                    "points": SCORE_COMPONENT_POINTS,
                    "condition": "FVG age between 6 and 15 bars inclusive",
                },
                {
                    "name": "Swing Distance",
                    "points": SCORE_COMPONENT_POINTS,
                    "condition": "Distance from swing high/low under 20 points",
                },
            ],
        }

    def _component_hits(self, record: EdgeFeatureRecord) -> dict[str, bool]:
        return {
            "strong_displacement": record.displacement_strength == "Strong",
            "choch_to_bos_timing": 90 <= record.choch_to_bos_minutes < 240,
            "fvg_retests_one": record.fvg_retests == 1,
            "fvg_freshness_recent": 6 <= record.fvg_freshness_bars <= 15,
            "swing_distance_close": record.distance_from_swing_points < 20,
        }

    def compute_quality_score(self, record: EdgeFeatureRecord) -> tuple[int, dict[str, bool], dict[str, int]]:
        hits = self._component_hits(record)
        points = {
            key: SCORE_COMPONENT_POINTS if active else 0 for key, active in hits.items()
        }
        return sum(points.values()), hits, points

    def _metrics(self, label: str, signals: list[ScoredTier2Signal]) -> CohortMetrics:
        if not signals:
            return CohortMetrics(
                label=label,
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

        return CohortMetrics(
            label=label,
            signals=len(signals),
            win_rate_pct=round(wins / len(signals) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
        )

    def _collect_scored_signals(self, metadata: dict[str, Any]) -> list[ScoredTier2Signal]:
        records = self.edge_engine._collect_records(metadata)
        scored: list[ScoredTier2Signal] = []

        for record in records:
            score, hits, points = self.compute_quality_score(record)
            risk = record.risk_points
            rr = round(record.realized_pnl_points / risk, 2) if risk > 0 else 0.0

            scored.append(
                ScoredTier2Signal(
                    bos_timestamp=record.bos_timestamp,
                    timeframe=record.timeframe,
                    direction=record.direction,
                    quality_score=score,
                    component_hits=hits,
                    component_points=points,
                    realized_pnl_points=record.realized_pnl_points,
                    realized_rr=rr,
                    win=record.win,
                )
            )

        scored.sort(key=lambda item: pd.Timestamp(item.bos_timestamp))
        return scored

    def _threshold_metrics(
        self,
        signals: list[ScoredTier2Signal],
        threshold: int,
    ) -> CohortMetrics:
        filtered = [signal for signal in signals if signal.quality_score >= threshold]
        return self._metrics(f"Score >= {threshold}", filtered)

    @staticmethod
    def _improves_over_baseline(baseline: CohortMetrics, candidate: CohortMetrics) -> bool:
        if candidate.signals == 0:
            return False
        return (
            candidate.expectancy >= baseline.expectancy
            and (candidate.profit_factor or 0) >= (baseline.profit_factor or 0)
            and candidate.win_rate_pct >= baseline.win_rate_pct
            and candidate.maximum_drawdown_points <= baseline.maximum_drawdown_points
        )

    def _recommend_threshold(
        self,
        baseline: CohortMetrics,
        threshold_metrics: dict[int, CohortMetrics],
    ) -> tuple[int, list[str]]:
        ranked = sorted(
            THRESHOLD_CANDIDATES,
            key=lambda threshold: (
                threshold_metrics[threshold].expectancy,
                threshold_metrics[threshold].profit_factor or 0,
                threshold_metrics[threshold].win_rate_pct,
                -threshold_metrics[threshold].maximum_drawdown_points,
            ),
            reverse=True,
        )

        for threshold in ranked:
            metrics = threshold_metrics[threshold]
            if metrics.signals < 20:
                continue
            if self._improves_over_baseline(baseline, metrics):
                rationale = [
                    f"Recommended production threshold: Score >= {threshold}.",
                    (
                        f"Expectancy {metrics.expectancy} vs baseline {baseline.expectancy}; "
                        f"PF {metrics.profit_factor} vs {baseline.profit_factor}; "
                        f"WR {metrics.win_rate_pct}% vs {baseline.win_rate_pct}%; "
                        f"DD {metrics.maximum_drawdown_points} vs {baseline.maximum_drawdown_points}; "
                        f"signals {metrics.signals} vs {baseline.signals}."
                    ),
                ]
                return threshold, rationale

        best = ranked[0]
        metrics = threshold_metrics[best]
        rationale = [
            f"Recommended production threshold: Score >= {best} (best available trade-off).",
            (
                f"No threshold improved all baseline metrics; selected highest expectancy "
                f"({metrics.expectancy}) with {metrics.signals} signals."
            ),
        ]
        return best, rationale

    def run(self, metadata: dict[str, Any]) -> InstitutionalQualityValidationReport:
        """Run institutional quality score validation."""
        started = time.perf_counter()

        signals = self._collect_scored_signals(metadata)
        if not signals:
            raise InstitutionalQualityValidationError("No scored Tier-2 signals found.")

        baseline = self._metrics("All Tier-2 (unfiltered)", signals)

        bucket_groups: dict[str, list[ScoredTier2Signal]] = {
            label: [] for label, _, _ in SCORE_BUCKETS
        }
        for signal in signals:
            bucket_groups[self._bucket_for_score(signal.quality_score)].append(signal)

        score_buckets = {
            label: self._metrics(label, group).as_dict()
            for label, group in bucket_groups.items()
        }

        threshold_metrics = {
            threshold: self._threshold_metrics(signals, threshold)
            for threshold in THRESHOLD_CANDIDATES
        }
        threshold_comparison = {
            str(threshold): metrics.as_dict() for threshold, metrics in threshold_metrics.items()
        }

        high_bucket = score_buckets["80-100"]
        low_bucket = score_buckets["0-20"]
        higher_score_improves = (
            high_bucket["expectancy"] > low_bucket["expectancy"]
            and high_bucket["win_rate_pct"] > low_bucket["win_rate_pct"]
            and (high_bucket["profit_factor"] or 0) >= (low_bucket["profit_factor"] or 0)
        )

        recommended, rationale = self._recommend_threshold(baseline, threshold_metrics)

        conclusions = [
            f"Validated institutional quality score on {len(signals)} Raw Tier-2 signals.",
            (
                f"Score 80-100: expectancy {high_bucket['expectancy']}, WR {high_bucket['win_rate_pct']}% "
                f"({high_bucket['signals']} signals)."
            ),
            (
                f"Score 0-20: expectancy {low_bucket['expectancy']}, WR {low_bucket['win_rate_pct']}% "
                f"({low_bucket['signals']} signals)."
            ),
            f"Higher quality scores improve outcomes: {higher_score_improves}.",
            *rationale,
        ]

        return InstitutionalQualityValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            scoring_model=self.scoring_model_definition(),
            total_signals=len(signals),
            score_buckets=score_buckets,
            threshold_comparison=threshold_comparison,
            unfiltered_baseline=baseline.as_dict(),
            higher_score_improves_outcomes=higher_score_improves,
            recommended_production_threshold=recommended,
            recommendation_rationale=rationale,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_quality_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> InstitutionalQualityValidationReport:
    """Run institutional quality validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalQualityValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalQualityValidationResearch(
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
        "Institutional quality validation completed: recommended>=%s",
        report.recommended_production_threshold,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_quality_validation_report()
        print("Institutional Quality Score Validation Summary")
        print(f"Signals: {report.total_signals}")
        for label, metrics in report.score_buckets.items():
            print(
                f"  {label}: n={metrics['signals']} WR={metrics['win_rate_pct']}% "
                f"Exp={metrics['expectancy']} Net={metrics['net_points']}"
            )
        print("Threshold comparison:")
        for threshold, metrics in report.threshold_comparison.items():
            print(
                f"  >={threshold}: n={metrics['signals']} WR={metrics['win_rate_pct']}% "
                f"Exp={metrics['expectancy']} DD={metrics['maximum_drawdown_points']}"
            )
        print(f"Recommended threshold: Score >= {report.recommended_production_threshold}")
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalQualityValidationError as exc:
        logger.error("Institutional quality validation error: %s", exc)
        print(f"Institutional quality validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional quality validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
