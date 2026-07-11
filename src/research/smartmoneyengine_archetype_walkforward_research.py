"""
SmartMoneyEngine Archetype Walk-Forward Validation research.

Validates whether top-ranked V2 archetypes survive on unseen 70/30 data.
Research-only; no production modifications.
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
    InstitutionalQualityValidationResearch,
)
from src.research.smartmoneyengine_v2_signal_ranking_research import (
    RankedV2Signal,
    SmartMoneyEngineV2SignalRankingResearch,
)
from src.research.smartmoneyengine_walkforward_validation_research import (
    TRAIN_FRACTION,
    TEST_FRACTION,
    SmartMoneyEngineWalkForwardValidationResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_V2_RANKING_PATH = RESEARCH_DIR / "smartmoneyengine_v2_signal_ranking.json"
DEFAULT_V2_OPTIMIZATION_PATH = RESEARCH_DIR / "smartmoneyengine_v2_frequency_optimization.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_archetype_walkforward.json"

SURVIVES_MIN_PF = 1.5
SURVIVES_MIN_EXPECTANCY = 75.0
SURVIVES_MIN_WIN_RATE = 50.0
TOP_ROBUST_COUNT = 20
MIN_TEST_SAMPLES = 5


class ArchetypeWalkForwardError(Exception):
    """Raised when archetype walk-forward validation fails."""


@dataclass
class PeriodMetrics:
    """Metrics for one train or test period."""

    sample_size: int
    signals_per_month: float
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    hit_1r_rate_pct: float
    hit_2r_rate_pct: float
    hit_3r_rate_pct: float
    net_points: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchetypeWalkForwardResult:
    """Walk-forward validation for one V2 archetype."""

    archetype_key: str
    signal_side: str
    ranking_signal_quality_score: float
    ranking_tier: str
    train: PeriodMetrics
    test: PeriodMetrics
    edge_decay_pct: float | None
    walkforward_classification: str
    robustness_score: float
    production_candidate: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train"] = self.train.as_dict()
        payload["test"] = self.test.as_dict()
        return payload


@dataclass
class ArchetypeWalkForwardReport:
    """Full archetype walk-forward validation output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    train_fraction: float
    test_fraction: float
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    source_ranking_export: str
    archetypes_validated: int
    classification_summary: dict[str, int]
    archetype_walkforward_results: list[dict[str, Any]]
    top_20_robust_buy_archetypes: list[dict[str, Any]]
    top_20_robust_sell_archetypes: list[dict[str, Any]]
    production_candidate_list: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineArchetypeWalkForwardResearch:
    """Validate top V2 archetypes on temporal train/test split."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
        v2_ranking_path: Path | str | None = None,
        v2_optimization_path: Path | str | None = None,
    ) -> None:
        self.symbols = symbols or ("NIFTY50", "BANKNIFTY", "FINNIFTY")
        self.research_days = research_days
        self.timeframes = timeframes
        self.v2_ranking_path = Path(v2_ranking_path or DEFAULT_V2_RANKING_PATH)
        self.v2_optimization_path = Path(v2_optimization_path or DEFAULT_V2_OPTIMIZATION_PATH)
        self._ranking_engine = SmartMoneyEngineV2SignalRankingResearch(
            symbols=self.symbols,
            research_days=research_days,
            timeframes=timeframes,
            v2_optimization_path=self.v2_optimization_path,
        )
        self._walkforward_engine = SmartMoneyEngineWalkForwardValidationResearch(
            symbols=self.symbols,
            research_days=research_days,
            timeframes=timeframes,
        )

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _parse_archetype_key(key: str) -> dict[str, str]:
        criteria: dict[str, str] = {}
        for part in key.split(" | "):
            if "=" not in part:
                continue
            dimension, value = part.split("=", 1)
            criteria[dimension.strip()] = value.strip()
        return criteria

    @staticmethod
    def _matches_archetype(signal: RankedV2Signal, criteria: dict[str, str]) -> bool:
        dimensions = signal.dimension_values()
        return all(dimensions.get(dim) == value for dim, value in criteria.items())

    @staticmethod
    def _in_period(timestamp: str, period_start: date, period_end: date) -> bool:
        trade_date = pd.to_datetime(timestamp).date()
        return period_start <= trade_date <= period_end

    def _load_top_archetypes(self) -> list[dict[str, Any]]:
        if not self.v2_ranking_path.exists():
            raise ArchetypeWalkForwardError(
                f"V2 signal ranking export not found: {self.v2_ranking_path}",
            )
        with self.v2_ranking_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        archetypes = payload.get("top_50_signal_archetypes", [])
        if not archetypes:
            raise ArchetypeWalkForwardError("No top_50_signal_archetypes in ranking export.")
        return archetypes

    def _aggregate_period(
        self,
        signals: list[RankedV2Signal],
        period_days: int,
    ) -> PeriodMetrics:
        total = len(signals)
        pnls = [item.realized_pnl_points for item in signals]
        wins = sum(1 for item in signals if item.win)
        months = max(period_days / 30.4375, 1.0)
        return PeriodMetrics(
            sample_size=total,
            signals_per_month=round(total / months, 2) if total else 0.0,
            win_rate_pct=round(wins / total * 100, 2) if total else 0.0,
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2) if pnls else 0.0,
            hit_1r_rate_pct=round(sum(1 for item in signals if item.hit_1r) / total * 100, 2)
            if total
            else 0.0,
            hit_2r_rate_pct=round(sum(1 for item in signals if item.hit_2r) / total * 100, 2)
            if total
            else 0.0,
            hit_3r_rate_pct=round(sum(1 for item in signals if item.hit_3r) / total * 100, 2)
            if total
            else 0.0,
            net_points=round(sum(pnls), 2),
        )

    @staticmethod
    def _edge_decay(train: PeriodMetrics, test: PeriodMetrics) -> float | None:
        if train.expectancy in (0, 0.0):
            return None
        return round((test.expectancy - train.expectancy) / abs(train.expectancy) * 100, 2)

    @staticmethod
    def _classify(test: PeriodMetrics) -> str:
        pf = test.profit_factor
        if test.expectancy < 0 or pf is None or pf < 1.0:
            return "FAILS"
        if (
            pf >= SURVIVES_MIN_PF
            and test.expectancy >= SURVIVES_MIN_EXPECTANCY
            and test.win_rate_pct >= SURVIVES_MIN_WIN_RATE
        ):
            return "SURVIVES"
        return "DEGRADES"

    @staticmethod
    def _robustness_score(
        train: PeriodMetrics,
        test: PeriodMetrics,
        classification: str,
        edge_decay_pct: float | None,
    ) -> float:
        if classification == "FAILS":
            return round(max(0.0, min(test.expectancy / 10.0, 15.0)), 2)

        pf = test.profit_factor or 0.0
        pf_score = min(max(pf - 1.0, 0.0) / 2.0, 1.0) * 25.0
        exp_score = min(max(test.expectancy, 0.0) / 150.0, 1.0) * 25.0
        wr_score = min(test.win_rate_pct / 70.0, 1.0) * 25.0

        if edge_decay_pct is None:
            retention_score = 12.5
        else:
            retention = max(0.0, 1.0 - min(abs(edge_decay_pct), 100.0) / 100.0)
            retention_score = retention * 25.0

        sample_bonus = min(test.sample_size / 30.0, 1.0) * 5.0
        classification_bonus = {"SURVIVES": 20.0, "DEGRADES": 8.0, "FAILS": 0.0}[classification]
        return round(
            min(pf_score + exp_score + wr_score + retention_score + sample_bonus + classification_bonus, 100.0),
            2,
        )

    def _validate_archetype(
        self,
        archetype: dict[str, Any],
        all_signals: list[RankedV2Signal],
        train_start: date,
        train_end: date,
        test_start: date,
        test_end: date,
        train_days: int,
        test_days: int,
    ) -> ArchetypeWalkForwardResult:
        criteria = self._parse_archetype_key(archetype["archetype_key"])
        matched = [item for item in all_signals if self._matches_archetype(item, criteria)]

        train_signals = [
            item for item in matched if self._in_period(item.bos_timestamp, train_start, train_end)
        ]
        test_signals = [
            item for item in matched if self._in_period(item.bos_timestamp, test_start, test_end)
        ]

        train_metrics = self._aggregate_period(train_signals, train_days)
        test_metrics = self._aggregate_period(test_signals, test_days)
        decay = self._edge_decay(train_metrics, test_metrics)

        if test_metrics.sample_size < MIN_TEST_SAMPLES:
            classification = "FAILS" if test_metrics.expectancy <= 0 else "DEGRADES"
        else:
            classification = self._classify(test_metrics)

        robustness = self._robustness_score(train_metrics, test_metrics, classification, decay)
        production_candidate = classification == "SURVIVES" or (
            classification == "DEGRADES" and robustness >= 65.0 and test_metrics.profit_factor is not None
        )

        return ArchetypeWalkForwardResult(
            archetype_key=archetype["archetype_key"],
            signal_side=str(archetype.get("signal_side", matched[0].signal_side if matched else "")),
            ranking_signal_quality_score=float(archetype.get("signal_quality_score", 0.0)),
            ranking_tier=str(archetype.get("tier", "")),
            train=train_metrics,
            test=test_metrics,
            edge_decay_pct=decay,
            walkforward_classification=classification,
            robustness_score=robustness,
            production_candidate=production_candidate and test_metrics.sample_size >= MIN_TEST_SAMPLES,
        )

    @staticmethod
    def _production_row(result: ArchetypeWalkForwardResult) -> dict[str, Any]:
        test = result.test.as_dict()
        train = result.train.as_dict()
        return {
            "archetype_key": result.archetype_key,
            "signal_side": result.signal_side,
            "walkforward_classification": result.walkforward_classification,
            "robustness_score": result.robustness_score,
            "edge_decay_pct": result.edge_decay_pct,
            "train_sample_size": train["sample_size"],
            "test_sample_size": test["sample_size"],
            "train_win_rate_pct": train["win_rate_pct"],
            "test_win_rate_pct": test["win_rate_pct"],
            "train_profit_factor": train["profit_factor"],
            "test_profit_factor": test["profit_factor"],
            "train_expectancy": train["expectancy"],
            "test_expectancy": test["expectancy"],
            "train_hit_1r_rate_pct": train["hit_1r_rate_pct"],
            "train_hit_2r_rate_pct": train["hit_2r_rate_pct"],
            "train_hit_3r_rate_pct": train["hit_3r_rate_pct"],
            "test_hit_1r_rate_pct": test["hit_1r_rate_pct"],
            "test_hit_2r_rate_pct": test["hit_2r_rate_pct"],
            "test_hit_3r_rate_pct": test["hit_3r_rate_pct"],
            "signals_per_month": test["signals_per_month"],
            "win_rate_pct": test["win_rate_pct"],
            "profit_factor": test["profit_factor"],
            "expectancy": test["expectancy"],
            "hit_1r_rate_pct": test["hit_1r_rate_pct"],
            "hit_2r_rate_pct": test["hit_2r_rate_pct"],
            "hit_3r_rate_pct": test["hit_3r_rate_pct"],
        }

    def run(self, metadata: dict[str, Any]) -> ArchetypeWalkForwardReport:
        started = time.perf_counter()
        top_archetypes = self._load_top_archetypes()
        v2_card = self._ranking_engine._load_v2_card()
        all_signals = self._ranking_engine._collect_v2_signals(metadata, v2_card)

        start, train_end, test_start, end = self._walkforward_engine._split_dates(metadata)
        train_days = max((train_end - start).days + 1, 1)
        test_days = max((end - test_start).days + 1, 1)

        results: list[ArchetypeWalkForwardResult] = []
        for archetype in top_archetypes:
            results.append(
                self._validate_archetype(
                    archetype,
                    all_signals,
                    start,
                    train_end,
                    test_start,
                    end,
                    train_days,
                    test_days,
                ),
            )

        classification_summary = {
            "SURVIVES": sum(1 for item in results if item.walkforward_classification == "SURVIVES"),
            "DEGRADES": sum(1 for item in results if item.walkforward_classification == "DEGRADES"),
            "FAILS": sum(1 for item in results if item.walkforward_classification == "FAILS"),
        }

        ranked = sorted(results, key=lambda item: (-item.robustness_score, -item.test.sample_size))
        top_buy = [
            self._production_row(item)
            for item in ranked
            if item.signal_side == "BUY"
        ][:TOP_ROBUST_COUNT]
        top_sell = [
            self._production_row(item)
            for item in ranked
            if item.signal_side == "SELL"
        ][:TOP_ROBUST_COUNT]

        production_candidates = [
            self._production_row(item)
            for item in ranked
            if item.production_candidate
        ]

        conclusions = [
            f"Validated {len(results)} top-ranked V2 archetypes on {int(TRAIN_FRACTION * 100)}/{int(TEST_FRACTION * 100)} walk-forward split.",
            f"Train window: {start.isoformat()} -> {train_end.isoformat()}; "
            f"Test window: {test_start.isoformat()} -> {end.isoformat()}.",
            f"Classification: SURVIVES={classification_summary['SURVIVES']}, "
            f"DEGRADES={classification_summary['DEGRADES']}, FAILS={classification_summary['FAILS']}.",
            f"Production candidates: {len(production_candidates)} archetypes.",
            (
                f"Top robust SELL: score={top_sell[0]['robustness_score'] if top_sell else 'N/A'} "
                f"class={top_sell[0]['walkforward_classification'] if top_sell else 'N/A'}."
            ),
            (
                f"Top robust BUY: score={top_buy[0]['robustness_score'] if top_buy else 'N/A'} "
                f"class={top_buy[0]['walkforward_classification'] if top_buy else 'N/A'}."
            ),
        ]

        return ArchetypeWalkForwardReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            train_fraction=TRAIN_FRACTION,
            test_fraction=TEST_FRACTION,
            train_start_date=start.isoformat(),
            train_end_date=train_end.isoformat(),
            test_start_date=test_start.isoformat(),
            test_end_date=end.isoformat(),
            source_ranking_export=str(self.v2_ranking_path),
            archetypes_validated=len(results),
            classification_summary=classification_summary,
            archetype_walkforward_results=[item.as_dict() for item in results],
            top_20_robust_buy_archetypes=top_buy,
            top_20_robust_sell_archetypes=top_sell,
            production_candidate_list=production_candidates,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_archetype_walkforward_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    v2_ranking_path: Path | str | None = None,
    v2_optimization_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> ArchetypeWalkForwardReport:
    """Run archetype walk-forward validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise ArchetypeWalkForwardError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineArchetypeWalkForwardResearch(
        symbols=symbols,
        v2_ranking_path=v2_ranking_path,
        v2_optimization_path=v2_optimization_path,
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Archetype walk-forward completed: survives=%s production_candidates=%s",
        report.classification_summary.get("SURVIVES"),
        len(report.production_candidate_list),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_archetype_walkforward_report()
        print("SmartMoneyEngine Archetype Walk-Forward Validation Summary")
        print(f"Archetypes validated: {report.archetypes_validated}")
        print(f"SURVIVES: {report.classification_summary['SURVIVES']}")
        print(f"DEGRADES: {report.classification_summary['DEGRADES']}")
        print(f"FAILS: {report.classification_summary['FAILS']}")
        print(f"Production candidates: {len(report.production_candidate_list)}")
        if report.top_20_robust_sell_archetypes:
            top = report.top_20_robust_sell_archetypes[0]
            print(
                f"Top robust SELL: score={top['robustness_score']} "
                f"test PF={top['test_profit_factor']} test Exp={top['test_expectancy']}",
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except ArchetypeWalkForwardError as exc:
        logger.error("Archetype walk-forward error: %s", exc)
        print(f"Archetype walk-forward error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected archetype walk-forward error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
