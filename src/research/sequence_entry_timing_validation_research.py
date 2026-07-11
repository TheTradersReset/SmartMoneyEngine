"""
Sequence entry timing validation research for SmartMoneyEngine.

Determines the best entry stage in the institutional sequence
(Sweep -> Displacement -> CHOCH -> BOS -> FVG Reclaim). Research-only.
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
from src.research.institutional_move_validation_research import (
    FORWARD_BARS,
    InstitutionalMoveValidationResearch,
    SequenceInstance,
)
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "sequence_entry_timing_validation.json"

FVG_RECLAIM_SCAN_BARS = 20

ENTRY_STAGES: dict[str, str] = {
    "sweep_close": "Sweep Close",
    "displacement_close": "Displacement Close",
    "choch_confirmation": "CHOCH Confirmation",
    "bos_confirmation": "BOS Confirmation",
    "fvg_reclaim": "FVG Reclaim",
}


class SequenceEntryTimingValidationError(Exception):
    """Raised when sequence entry timing validation fails."""


@dataclass(frozen=True)
class EnrichedSequence:
    """Institutional sequence with FVG reclaim bar."""

    sequence: SequenceInstance
    fvg_reclaim_bar: int

    def as_dict(self) -> dict[str, Any]:
        payload = self.sequence.as_dict()
        payload["fvg_reclaim_bar"] = self.fvg_reclaim_bar
        return payload


@dataclass(frozen=True)
class StageTradeOutcome:
    """Simulated trade for one entry stage on one sequence."""

    stage_key: str
    stage_label: str
    sequence_bos_timestamp: str
    timeframe: str
    direction: str
    entry_bar: int
    entry_timestamp: str
    entry_price: float
    stop_price: float
    risk_points: float
    target_price: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    hit_1r_before_sl: bool
    hit_2r_before_sl: bool
    hit_3r_before_sl: bool
    stopped_out: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageMetrics:
    """Aggregate metrics for one entry stage."""

    stage_key: str
    stage_label: str
    trades: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float
    hit_1r_rate_pct: float
    hit_2r_rate_pct: float
    hit_3r_rate_pct: float
    composite_score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceEntryTimingValidationReport:
    """Full sequence entry timing validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    institutional_sequence_definition: list[str]
    entry_stages: dict[str, str]
    stop_loss_model: str
    exit_target_model: str
    total_sequences: int
    stage_metrics: dict[str, dict[str, Any]]
    stage_rankings: dict[str, str]
    recommended_institutional_entry_stage: str
    production_recommendation: dict[str, Any]
    comparison_summary: dict[str, dict[str, Any]]
    trade_outcomes: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SequenceEntryTimingValidationResearch:
    """Validate institutional sequence entry timing across five stages."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.move_engine = InstitutionalMoveValidationResearch(
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
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    def _find_fvg_reclaim_bar(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        direction: str,
    ) -> int:
        end = min(len(frame) - 1, start_bar + FVG_RECLAIM_SCAN_BARS)
        for index in range(start_bar, end + 1):
            if self.move_engine._fvg_reclaimed_at_bar(frame, index, direction):
                return index
        return start_bar

    def _enrich_sequences(
        self,
        frame: pd.DataFrame,
        sequences: list[SequenceInstance],
    ) -> list[EnrichedSequence]:
        enriched: list[EnrichedSequence] = []
        for sequence in sequences:
            reclaim_bar = self._find_fvg_reclaim_bar(frame, sequence.bos_bar, sequence.direction)
            enriched.append(EnrichedSequence(sequence=sequence, fvg_reclaim_bar=reclaim_bar))
        return enriched

    def _entry_bar_for_stage(self, enriched: EnrichedSequence, stage_key: str) -> int:
        sequence = enriched.sequence
        mapping = {
            "sweep_close": sequence.sweep_bar,
            "displacement_close": sequence.displacement_bar,
            "choch_confirmation": sequence.choch_bar,
            "bos_confirmation": sequence.bos_bar,
            "fvg_reclaim": enriched.fvg_reclaim_bar,
        }
        return mapping[stage_key]

    def _simulate_stage_trade(
        self,
        frame: pd.DataFrame,
        enriched: EnrichedSequence,
        stage_key: str,
    ) -> StageTradeOutcome | None:
        sequence = enriched.sequence
        entry_bar = self._entry_bar_for_stage(enriched, stage_key)
        if entry_bar >= len(frame) - 1:
            return None

        direction = sequence.direction
        entry_price = round(float(frame.iloc[entry_bar]["Close"]), 2)
        stop, risk = self.construction_engine._structural_stop(
            frame,
            entry_bar,
            entry_price,
            direction,
        )
        target = self.construction_engine._opposite_liquidity_target(
            frame,
            entry_bar,
            entry_price,
            direction,
            risk,
        )

        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
        hit_1r = hit_2r = hit_3r = False
        stopped_out = False
        pnl = 0.0
        rr = 0.0
        win = False

        for index in range(entry_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])

            if direction == "bullish":
                favorable = bar_high - entry_price
                stop_hit = bar_low <= stop
                target_hit = bar_high >= target
            else:
                favorable = entry_price - bar_low
                stop_hit = bar_high >= stop
                target_hit = bar_low <= target

            if not stopped_out:
                if favorable >= risk:
                    hit_1r = True
                if favorable >= risk * 2:
                    hit_2r = True
                if favorable >= risk * 3:
                    hit_3r = True

            if stop_hit:
                stopped_out = True
                pnl = -risk
                rr = -1.0
                break

            if target_hit:
                pnl = round(abs(target - entry_price), 2)
                rr = round(pnl / risk, 2) if risk > 0 else 0.0
                win = pnl > 0
                break
        else:
            close = float(frame.iloc[end]["Close"])
            if direction == "bullish":
                pnl = round(close - entry_price, 2)
            else:
                pnl = round(entry_price - close, 2)
            rr = round(pnl / risk, 2) if risk > 0 else 0.0
            win = pnl > 0

        return StageTradeOutcome(
            stage_key=stage_key,
            stage_label=ENTRY_STAGES[stage_key],
            sequence_bos_timestamp=sequence.bos_timestamp,
            timeframe=sequence.timeframe,
            direction=direction,
            entry_bar=entry_bar,
            entry_timestamp=str(frame.iloc[entry_bar]["Date"]),
            entry_price=entry_price,
            stop_price=round(stop, 2),
            risk_points=round(risk, 2),
            target_price=round(target, 2),
            realized_pnl_points=pnl,
            realized_rr=rr,
            win=win,
            hit_1r_before_sl=hit_1r,
            hit_2r_before_sl=hit_2r,
            hit_3r_before_sl=hit_3r,
            stopped_out=stopped_out,
        )

    def _collect_outcomes(self, metadata: dict[str, Any]) -> list[StageTradeOutcome]:
        from datetime import timedelta

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        outcomes: list[StageTradeOutcome] = []
        for timeframe_label in self.timeframes:
            path = self.move_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            sequences = self.move_engine._detect_institutional_sequences(frame, timeframe_label)
            enriched_sequences = self._enrich_sequences(frame, sequences)

            for enriched in enriched_sequences:
                for stage_key in ENTRY_STAGES:
                    outcome = self._simulate_stage_trade(frame, enriched, stage_key)
                    if outcome is not None:
                        outcomes.append(outcome)

        return outcomes

    @staticmethod
    def _pct_true(outcomes: list[StageTradeOutcome], field: str) -> float:
        if not outcomes:
            return 0.0
        return round(sum(1 for item in outcomes if getattr(item, field)) / len(outcomes) * 100, 2)

    @staticmethod
    def _composite_score(metrics: StageMetrics) -> float:
        pf = metrics.profit_factor or 0.0
        if pf == float("inf"):
            pf = 10.0
        return round(
            metrics.expectancy * 0.40
            + pf * 12.0
            + metrics.win_rate_pct * 0.25
            + metrics.average_rr * 8.0
            + metrics.hit_2r_rate_pct * 0.10
            - metrics.maximum_drawdown_points * 0.008,
            4,
        )

    def _metrics_for_stage(
        self,
        stage_key: str,
        outcomes: list[StageTradeOutcome],
    ) -> StageMetrics:
        if not outcomes:
            return StageMetrics(
                stage_key=stage_key,
                stage_label=ENTRY_STAGES[stage_key],
                trades=0,
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                net_points=0.0,
                maximum_drawdown_points=0.0,
                hit_1r_rate_pct=0.0,
                hit_2r_rate_pct=0.0,
                hit_3r_rate_pct=0.0,
                composite_score=0.0,
            )

        pnls = [item.realized_pnl_points for item in outcomes]
        rrs = [item.realized_rr for item in outcomes]
        wins = sum(1 for item in outcomes if item.win)

        metrics = StageMetrics(
            stage_key=stage_key,
            stage_label=ENTRY_STAGES[stage_key],
            trades=len(outcomes),
            win_rate_pct=round(wins / len(outcomes) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            hit_1r_rate_pct=self._pct_true(outcomes, "hit_1r_before_sl"),
            hit_2r_rate_pct=self._pct_true(outcomes, "hit_2r_before_sl"),
            hit_3r_rate_pct=self._pct_true(outcomes, "hit_3r_before_sl"),
            composite_score=0.0,
        )
        metrics.composite_score = self._composite_score(metrics)
        return metrics

    def _stage_rankings(self, metrics_by_stage: dict[str, StageMetrics]) -> dict[str, str]:
        if not metrics_by_stage:
            return {}

        return {
            "highest_accuracy": max(
                metrics_by_stage.values(),
                key=lambda item: item.win_rate_pct,
            ).stage_label,
            "highest_rr": max(
                metrics_by_stage.values(),
                key=lambda item: item.average_rr,
            ).stage_label,
            "best_expectancy": max(
                metrics_by_stage.values(),
                key=lambda item: item.expectancy,
            ).stage_label,
            "best_net_profit": max(
                metrics_by_stage.values(),
                key=lambda item: item.net_points,
            ).stage_label,
            "best_composite_score": max(
                metrics_by_stage.values(),
                key=lambda item: item.composite_score,
            ).stage_label,
        }

    def run(self, metadata: dict[str, Any]) -> SequenceEntryTimingValidationReport:
        """Run sequence entry timing validation research."""
        started = time.perf_counter()
        all_outcomes = self._collect_outcomes(metadata)
        if not all_outcomes:
            raise SequenceEntryTimingValidationError(
                "No institutional sequence entry outcomes found.",
            )

        sequences_count = len(
            {item.sequence_bos_timestamp for item in all_outcomes if item.stage_key == "bos_confirmation"}
        )

        metrics_by_stage: dict[str, StageMetrics] = {}
        grouped: dict[str, list[StageTradeOutcome]] = {
            stage_key: [] for stage_key in ENTRY_STAGES
        }
        for outcome in all_outcomes:
            grouped[outcome.stage_key].append(outcome)

        for stage_key in ENTRY_STAGES:
            metrics_by_stage[stage_key] = self._metrics_for_stage(stage_key, grouped[stage_key])

        rankings = self._stage_rankings(metrics_by_stage)
        recommended = rankings["best_composite_score"]
        leader = max(metrics_by_stage.values(), key=lambda item: item.composite_score)

        comparison_summary = {
            stage_key: {
                "win_rate_pct": metrics.win_rate_pct,
                "average_rr": metrics.average_rr,
                "expectancy": metrics.expectancy,
                "net_points": metrics.net_points,
                "hit_1r_rate_pct": metrics.hit_1r_rate_pct,
                "hit_2r_rate_pct": metrics.hit_2r_rate_pct,
                "hit_3r_rate_pct": metrics.hit_3r_rate_pct,
                "composite_score": metrics.composite_score,
            }
            for stage_key, metrics in metrics_by_stage.items()
        }

        production_recommendation = {
            "recommended_stage": recommended,
            "stage_key": leader.stage_key,
            "trades": leader.trades,
            "win_rate_pct": leader.win_rate_pct,
            "profit_factor": leader.profit_factor,
            "expectancy": leader.expectancy,
            "average_rr": leader.average_rr,
            "net_points": leader.net_points,
            "composite_score": leader.composite_score,
            "rankings": rankings,
        }

        conclusions = [
            f"Validated {sequences_count} institutional sequences across {len(ENTRY_STAGES)} entry stages.",
            (
                f"Highest accuracy: {rankings.get('highest_accuracy')} | "
                f"Highest RR: {rankings.get('highest_rr')} | "
                f"Best expectancy: {rankings.get('best_expectancy')} | "
                f"Best net profit: {rankings.get('best_net_profit')}."
            ),
            f"Recommended institutional entry stage: {recommended} (composite score {leader.composite_score}).",
        ]

        bos_metrics = metrics_by_stage.get("bos_confirmation")
        fvg_metrics = metrics_by_stage.get("fvg_reclaim")
        if bos_metrics and fvg_metrics:
            conclusions.append(
                f"BOS vs FVG Reclaim: WR {bos_metrics.win_rate_pct}% vs {fvg_metrics.win_rate_pct}%, "
                f"exp {bos_metrics.expectancy} vs {fvg_metrics.expectancy}."
            )

        return SequenceEntryTimingValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            institutional_sequence_definition=[
                "Liquidity Sweep",
                "Displacement",
                "CHOCH",
                "BOS",
                "FVG Reclaim",
            ],
            entry_stages=ENTRY_STAGES,
            stop_loss_model="Structural Swing SL",
            exit_target_model="Opposite Liquidity Pool",
            total_sequences=sequences_count,
            stage_metrics={
                stage_key: metrics.as_dict() for stage_key, metrics in metrics_by_stage.items()
            },
            stage_rankings=rankings,
            recommended_institutional_entry_stage=recommended,
            production_recommendation=production_recommendation,
            comparison_summary=comparison_summary,
            trade_outcomes=[item.as_dict() for item in all_outcomes],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_sequence_entry_timing_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> SequenceEntryTimingValidationReport:
    """Run sequence entry timing validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise SequenceEntryTimingValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SequenceEntryTimingValidationResearch(
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
        "Sequence entry timing validation completed: recommended=%s",
        report.recommended_institutional_entry_stage,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_sequence_entry_timing_validation_report()
        print("Sequence Entry Timing Validation Summary")
        print(f"Sequences: {report.total_sequences}")
        print("Stage comparison:")
        for stage_key, metrics in report.stage_metrics.items():
            print(
                f"  {metrics['stage_label']}: WR={metrics['win_rate_pct']}% "
                f"exp={metrics['expectancy']} RR={metrics['average_rr']} net={metrics['net_points']}"
            )
        print(f"Recommended: {report.recommended_institutional_entry_stage}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except SequenceEntryTimingValidationError as exc:
        logger.error("Sequence entry timing validation error: %s", exc)
        print(f"Sequence entry timing validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected sequence entry timing validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
