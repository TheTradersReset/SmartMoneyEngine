"""
Tier-2 composite edge validation research for SmartMoneyEngine.

Validates whether the strongest winning traits from winner-loser research
improve performance when combined. Research-only.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.tier2_winner_loser_comparison_research import (
    ComparativeTradeRecord,
    Tier2WinnerLoserComparisonResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tier2_composite_edge_validation.json"

MIN_PRODUCTION_SIGNALS = 20
TOP_COMBINATION_COUNT = 20

WINNING_TRAITS: dict[str, str] = {
    "rsi_below_40": "RSI < 40",
    "near_support": "Market Location = Near Support",
    "midday_session": "Session = Midday",
    "strong_displacement": "Strong Displacement",
    "slow_choch_bos": "CHOCH->BOS Slow (90-240 min)",
}


class Tier2CompositeEdgeValidationError(Exception):
    """Raised when Tier-2 composite edge validation fails."""


@dataclass
class CombinationMetrics:
    """Performance metrics for one trait combination filter."""

    combination_key: str
    combination_label: str
    trait_count: int
    traits: list[str]
    signals: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float
    production_score: float
    meets_minimum_signals: bool
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2CompositeEdgeValidationReport:
    """Full Tier-2 composite edge validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    entry_method: str
    stop_loss_model: str
    winning_traits: dict[str, str]
    minimum_signals_required: int
    total_signals: int
    baseline_unfiltered: dict[str, Any]
    individual_traits: list[dict[str, Any]]
    two_trait_combinations: list[dict[str, Any]]
    three_trait_combinations: list[dict[str, Any]]
    four_trait_combinations: list[dict[str, Any]]
    five_trait_combination: dict[str, Any] | None
    all_combinations: list[dict[str, Any]]
    rejected_combinations: list[dict[str, Any]]
    eligible_combinations: list[dict[str, Any]]
    top_20_combinations: list[dict[str, Any]]
    best_production_ready_filter: dict[str, Any]
    production_recommendation: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tier2CompositeEdgeValidationResearch:
    """Validate composite filters built from winner-loser winning traits."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.comparison_engine = Tier2WinnerLoserComparisonResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _production_score(metrics: CombinationMetrics) -> float:
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

    @staticmethod
    def _trait_checks(record: ComparativeTradeRecord) -> dict[str, bool]:
        return {
            "rsi_below_40": record.rsi < 40,
            "near_support": record.market_location == "Near Support",
            "midday_session": record.session == "Midday",
            "strong_displacement": record.displacement_strength == "Strong",
            "slow_choch_bos": 90 <= record.choch_to_bos_minutes < 240,
        }

    @staticmethod
    def _combination_label(trait_keys: tuple[str, ...]) -> str:
        return " + ".join(WINNING_TRAITS[key] for key in trait_keys)

    def _matches_combination(
        self,
        record: ComparativeTradeRecord,
        trait_keys: tuple[str, ...],
    ) -> bool:
        checks = self._trait_checks(record)
        return all(checks[key] for key in trait_keys)

    def _filter_records(
        self,
        records: list[ComparativeTradeRecord],
        trait_keys: tuple[str, ...],
    ) -> list[ComparativeTradeRecord]:
        return [record for record in records if self._matches_combination(record, trait_keys)]

    def _metrics(
        self,
        trait_keys: tuple[str, ...],
        records: list[ComparativeTradeRecord],
    ) -> CombinationMetrics:
        label = self._combination_label(trait_keys) if trait_keys else "Baseline (Unfiltered)"
        key = "+".join(trait_keys) if trait_keys else "baseline"

        if not records:
            empty = CombinationMetrics(
                combination_key=key,
                combination_label=label,
                trait_count=len(trait_keys),
                traits=[WINNING_TRAITS[k] for k in trait_keys],
                signals=0,
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                net_points=0.0,
                maximum_drawdown_points=0.0,
                production_score=0.0,
                meets_minimum_signals=False,
            )
            return empty

        pnls = [record.realized_pnl_points for record in records]
        rrs = [record.realized_rr for record in records]
        wins = sum(1 for record in records if record.win)

        metrics = CombinationMetrics(
            combination_key=key,
            combination_label=label,
            trait_count=len(trait_keys),
            traits=[WINNING_TRAITS[k] for k in trait_keys],
            signals=len(records),
            win_rate_pct=round(wins / len(records) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            production_score=0.0,
            meets_minimum_signals=len(records) >= MIN_PRODUCTION_SIGNALS,
        )
        metrics.production_score = self._production_score(metrics)
        return metrics

    def _evaluate_all_combinations(
        self,
        records: list[ComparativeTradeRecord],
    ) -> list[CombinationMetrics]:
        trait_keys = tuple(WINNING_TRAITS.keys())
        evaluated: list[CombinationMetrics] = []

        for size in range(1, len(trait_keys) + 1):
            for combo in combinations(trait_keys, size):
                filtered = self._filter_records(records, combo)
                evaluated.append(self._metrics(combo, filtered))

        return evaluated

    @staticmethod
    def _group_by_trait_count(
        evaluated: list[CombinationMetrics],
    ) -> dict[int, list[CombinationMetrics]]:
        grouped: dict[int, list[CombinationMetrics]] = {1: [], 2: [], 3: [], 4: [], 5: []}
        for item in evaluated:
            grouped[item.trait_count].append(item)
        return grouped

    @staticmethod
    def _sort_combinations(items: list[CombinationMetrics]) -> list[CombinationMetrics]:
        return sorted(
            items,
            key=lambda item: (
                item.production_score,
                item.expectancy,
                item.profit_factor or 0,
                item.win_rate_pct,
                -item.maximum_drawdown_points,
                item.net_points,
            ),
            reverse=True,
        )

    def _best_production_filter(
        self,
        eligible: list[CombinationMetrics],
        baseline: CombinationMetrics,
    ) -> CombinationMetrics:
        if not eligible:
            return baseline
        return self._sort_combinations(eligible)[0]

    def run(self, metadata: dict[str, Any]) -> Tier2CompositeEdgeValidationReport:
        """Run Tier-2 composite edge validation research."""
        started = time.perf_counter()
        records = self.comparison_engine._collect_records(metadata)
        if not records:
            raise Tier2CompositeEdgeValidationError("No Tier-2 composite edge records found.")

        baseline = self._metrics((), records)
        evaluated = self._evaluate_all_combinations(records)
        grouped = self._group_by_trait_count(evaluated)

        eligible = [item for item in evaluated if item.meets_minimum_signals]
        rejected = [item for item in evaluated if not item.meets_minimum_signals]
        ranked_eligible = self._sort_combinations(eligible)

        for index, item in enumerate(ranked_eligible, start=1):
            item.rank = index

        top_20 = ranked_eligible[:TOP_COMBINATION_COUNT]
        best = self._best_production_filter(eligible, baseline)

        five_trait = next((item for item in evaluated if item.trait_count == 5), None)

        research_months = max(
            (
                metadata.get("research_window_days", self.research_days) / 30.44
            ),
            1.0,
        )
        expected_monthly = round(best.signals / research_months, 2)

        production_recommendation = {
            "filter": best.combination_label,
            "trait_count": best.trait_count,
            "traits": best.traits,
            "signals": best.signals,
            "win_rate_pct": best.win_rate_pct,
            "profit_factor": best.profit_factor,
            "expectancy": best.expectancy,
            "average_rr": best.average_rr,
            "net_points": best.net_points,
            "maximum_drawdown_points": best.maximum_drawdown_points,
            "production_score": best.production_score,
            "expected_signals_per_month": expected_monthly,
            "meets_minimum_signals": best.meets_minimum_signals,
        }

        conclusions = [
            f"Validated {len(evaluated)} trait combinations across {len(records)} Tier-2 BOS Close signals.",
            (
                f"Baseline unfiltered: n={baseline.signals}, expectancy {baseline.expectancy}, "
                f"PF {baseline.profit_factor}, win rate {baseline.win_rate_pct}%."
            ),
            (
                f"Eligible combinations (n>={MIN_PRODUCTION_SIGNALS}): {len(eligible)}; "
                f"rejected: {len(rejected)}."
            ),
        ]
        if top_20:
            leader = top_20[0]
            conclusions.append(
                f"Top ranked combination: {leader.combination_label} "
                f"(n={leader.signals}, expectancy {leader.expectancy}, PF {leader.profit_factor})."
            )
        conclusions.append(
            f"Best production-ready filter: {best.combination_label} "
            f"(score {best.production_score}, n={best.signals})."
        )

        return Tier2CompositeEdgeValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            entry_method="BOS Close Entry",
            stop_loss_model="Structural Swing SL",
            winning_traits=WINNING_TRAITS,
            minimum_signals_required=MIN_PRODUCTION_SIGNALS,
            total_signals=len(records),
            baseline_unfiltered=baseline.as_dict(),
            individual_traits=[item.as_dict() for item in grouped[1]],
            two_trait_combinations=[item.as_dict() for item in grouped[2]],
            three_trait_combinations=[item.as_dict() for item in grouped[3]],
            four_trait_combinations=[item.as_dict() for item in grouped[4]],
            five_trait_combination=five_trait.as_dict() if five_trait else None,
            all_combinations=[item.as_dict() for item in evaluated],
            rejected_combinations=[item.as_dict() for item in rejected],
            eligible_combinations=[item.as_dict() for item in ranked_eligible],
            top_20_combinations=[item.as_dict() for item in top_20],
            best_production_ready_filter=best.as_dict(),
            production_recommendation=production_recommendation,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tier2_composite_edge_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Tier2CompositeEdgeValidationReport:
    """Run Tier-2 composite edge validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Tier2CompositeEdgeValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = Tier2CompositeEdgeValidationResearch(
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
        "Tier-2 composite edge validation completed: best=%s",
        report.best_production_ready_filter.get("combination_label"),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tier2_composite_edge_validation_report()
        print("Tier-2 Composite Edge Validation Summary")
        print(f"Total signals: {report.total_signals}")
        print(f"Eligible combinations: {len(report.eligible_combinations)}")
        print("Top combinations:")
        for item in report.top_20_combinations[:5]:
            print(
                f"  #{item['rank']} {item['combination_label']}: "
                f"n={item['signals']} exp={item['expectancy']} PF={item['profit_factor']}"
            )
        best = report.best_production_ready_filter
        print(
            f"Best production filter: {best['combination_label']} "
            f"(n={best['signals']}, score={best['production_score']})"
        )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except Tier2CompositeEdgeValidationError as exc:
        logger.error("Tier-2 composite edge validation error: %s", exc)
        print(f"Tier-2 composite edge validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Tier-2 composite edge validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
