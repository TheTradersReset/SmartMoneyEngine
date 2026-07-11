"""
Robust filter validation for profitable setup combinations.

Validates filter combinations from filter research using chronological
train/validation splits. Does not introduce new setups or indicators.
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.research.filter_research_engine import (
    FILTER_DIMENSIONS,
    PROFITABLE_SETUPS,
    RESEARCH_DAYS,
    FilterResearchEngine,
    FilteredTradeRecord,
    _json_safe,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "robust_filter_report.json"

MIN_TRADES = 100
MIN_PROFIT_FACTOR = 1.1
TRAIN_RATIO = 0.7
VALIDATION_PF_MIN = 1.0
MAX_WIN_RATE_DEGRADATION_PCT = 10.0


class RobustFilterValidatorError(Exception):
    """Raised when robust filter validation fails."""


@dataclass
class PeriodMetrics:
    """Performance metrics for one time period."""

    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    max_drawdown: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidatedCombination:
    """One filter combination with train/validation robustness metrics."""

    label: str
    setup_type: str | None
    filters: dict[str, str]
    trades: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    max_drawdown: float
    stability_score: float
    is_valid: bool
    validation_passed: bool
    win_rate_degradation_pct: float
    train: PeriodMetrics
    validation: PeriodMetrics

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train"] = self.train.as_dict()
        payload["validation"] = self.validation.as_dict()
        return payload


@dataclass
class RobustFilterReport:
    """Aggregate robust filter validation output."""

    input_report_path: str
    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    setups_analyzed: list[str]
    train_split_pct: float
    validation_split_pct: float
    min_trades: int
    min_profit_factor: float
    candidates_from_input: int
    candidates_evaluated: int
    candidates_after_trade_filter: int
    removed_low_sample_count: int
    top_10_robust_combinations: list[dict[str, Any]]
    top_10_overfit_combinations: list[dict[str, Any]]
    best_production_ready_filter_stack: dict[str, Any] | None
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RobustFilterValidator:
    """
    Validate statistical robustness of filter combinations.

    Parameters
    ----------
    input_report_path : Path | str, optional
        Path to filter research JSON report.
    min_trades : int, optional
        Minimum trades required for ranking.
    min_profit_factor : float, optional
        Minimum profit factor required for ranking.
    """

    def __init__(
        self,
        input_report_path: Path | str = DEFAULT_INPUT_PATH,
        min_trades: int = MIN_TRADES,
        min_profit_factor: float = MIN_PROFIT_FACTOR,
    ) -> None:
        self.input_report_path = Path(input_report_path)
        self.min_trades = min_trades
        self.min_profit_factor = min_profit_factor

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _max_drawdown(pnls: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return round(max_dd, 2)

    @staticmethod
    def _entry_trades(trades: list[FilteredTradeRecord]) -> list[FilteredTradeRecord]:
        return [trade for trade in trades if trade.entry_hit]

    @staticmethod
    def _sort_trades(trades: list[FilteredTradeRecord]) -> list[FilteredTradeRecord]:
        return sorted(trades, key=lambda trade: trade.trigger_timestamp)

    def _split_trades(
        self,
        trades: list[FilteredTradeRecord],
        train_ratio: float = TRAIN_RATIO,
    ) -> tuple[list[FilteredTradeRecord], list[FilteredTradeRecord]]:
        ordered = self._sort_trades(trades)
        split_index = int(len(ordered) * train_ratio)
        if split_index <= 0 or split_index >= len(ordered):
            return ordered, []
        return ordered[:split_index], ordered[split_index:]

    def _period_metrics(self, trades: list[FilteredTradeRecord]) -> PeriodMetrics:
        entries = self._entry_trades(trades)
        pnls = [trade.realized_pnl_points for trade in entries]
        rrs = [trade.realized_rr for trade in entries]
        wins = sum(1 for trade in entries if trade.outcome == "Win")
        losses = sum(1 for trade in entries if trade.outcome == "Loss")
        return PeriodMetrics(
            trades=len(entries),
            wins=wins,
            losses=losses,
            win_rate_pct=round((wins / len(entries)) * 100, 2) if entries else 0.0,
            profit_factor=self._profit_factor(pnls),
            expectancy=round(sum(pnls) / len(entries), 2) if entries else 0.0,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            max_drawdown=self._max_drawdown(pnls),
        )

    @staticmethod
    def _setup_type_from_label(label: str) -> str | None:
        for setup_type in PROFITABLE_SETUPS:
            if label.startswith(f"{setup_type}:"):
                return setup_type
        return None

    @staticmethod
    def _matches_filters(
        trade: FilteredTradeRecord,
        setup_type: str | None,
        filters: dict[str, str],
    ) -> bool:
        if setup_type and trade.setup_type != setup_type:
            return False
        for dimension, value in filters.items():
            if getattr(trade.filters, dimension) != value:
                return False
        return True

    def _filter_trades(
        self,
        trades: list[FilteredTradeRecord],
        setup_type: str | None,
        filters: dict[str, str],
    ) -> list[FilteredTradeRecord]:
        return [
            trade
            for trade in trades
            if self._matches_filters(trade, setup_type, filters)
        ]

    @staticmethod
    def _candidate_key(setup_type: str | None, filters: dict[str, str]) -> tuple[Any, ...]:
        return (setup_type, tuple(sorted(filters.items())))

    def _build_label(self, setup_type: str | None, filters: dict[str, str]) -> str:
        filter_text = " | ".join(f"{key}={value}" for key, value in sorted(filters.items()))
        if setup_type:
            return f"{setup_type}: {filter_text}"
        return filter_text

    def _stability_score(
        self,
        train: PeriodMetrics,
        validation: PeriodMetrics,
        is_valid: bool,
        win_rate_degradation_pct: float,
    ) -> float:
        if not is_valid or validation.trades == 0:
            return 0.0

        pf_component = 0.0
        if train.profit_factor and validation.profit_factor:
            pf_component = min(validation.profit_factor / train.profit_factor, 1.0)

        exp_component = 0.0
        if train.expectancy > 0 and validation.expectancy > 0:
            exp_component = min(validation.expectancy / train.expectancy, 1.0)

        wr_degradation_penalty = max(win_rate_degradation_pct, 0.0)
        wr_component = max(
            0.0,
            1.0 - (wr_degradation_penalty / MAX_WIN_RATE_DEGRADATION_PCT),
        )
        sample_component = min(validation.trades / 100.0, 1.0)

        score = (
            pf_component * 0.30
            + exp_component * 0.30
            + wr_component * 0.25
            + sample_component * 0.15
        ) * 100.0
        return round(min(score, 100.0), 2)

    def _validation_result(
        self,
        train: PeriodMetrics,
        validation: PeriodMetrics,
    ) -> tuple[bool, float]:
        if validation.trades == 0:
            return False, 100.0

        win_rate_degradation = train.win_rate_pct - validation.win_rate_pct
        pf_ok = (validation.profit_factor or 0) > VALIDATION_PF_MIN
        expectancy_ok = validation.expectancy > 0
        win_rate_ok = win_rate_degradation < MAX_WIN_RATE_DEGRADATION_PCT
        return pf_ok and expectancy_ok and win_rate_ok, round(win_rate_degradation, 2)

    def _evaluate_candidate(
        self,
        trades: list[FilteredTradeRecord],
        setup_type: str | None,
        filters: dict[str, str],
        label: str | None = None,
    ) -> ValidatedCombination | None:
        matched = self._filter_trades(trades, setup_type, filters)
        full = self._period_metrics(matched)

        if full.trades < self.min_trades:
            return None
        if (full.profit_factor or 0) <= self.min_profit_factor:
            return None

        train_trades, validation_trades = self._split_trades(matched)
        train = self._period_metrics(train_trades)
        validation = self._period_metrics(validation_trades)
        is_valid, win_rate_degradation = self._validation_result(train, validation)
        full_pnls = [
            trade.realized_pnl_points for trade in self._entry_trades(matched)
        ]

        return ValidatedCombination(
            label=label or self._build_label(setup_type, filters),
            setup_type=setup_type,
            filters=filters,
            trades=full.trades,
            win_rate_pct=full.win_rate_pct,
            profit_factor=full.profit_factor,
            expectancy=full.expectancy,
            average_rr=full.average_rr,
            max_drawdown=self._max_drawdown(full_pnls),
            stability_score=self._stability_score(
                train,
                validation,
                is_valid,
                win_rate_degradation,
            ),
            is_valid=is_valid,
            validation_passed=is_valid,
            win_rate_degradation_pct=win_rate_degradation,
            train=train,
            validation=validation,
        )

    def _load_input_report(self) -> dict[str, Any]:
        if not self.input_report_path.exists():
            raise RobustFilterValidatorError(
                f"Input report not found: {self.input_report_path}"
            )
        with self.input_report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _candidates_from_input(
        self,
        report: dict[str, Any],
    ) -> list[tuple[str | None, dict[str, str], str]]:
        candidates: list[tuple[str | None, dict[str, str], str]] = []
        seen: set[tuple[Any, ...]] = set()

        def add_candidate(
            setup_type: str | None,
            filters: dict[str, str],
            label: str,
        ) -> None:
            if not filters:
                return
            key = self._candidate_key(setup_type, filters)
            if key in seen:
                return
            seen.add(key)
            candidates.append((setup_type, filters, label))

        for combo in report.get("top_20_combinations", []):
            label = combo.get("label", "")
            filters = combo.get("filters", {})
            setup_type = self._setup_type_from_label(label)
            add_candidate(setup_type, filters, label)

        single_analysis = report.get("single_filter_analysis", {})
        for setup_key, dimensions in single_analysis.items():
            setup_type = setup_key if setup_key in PROFITABLE_SETUPS else None
            for entries in dimensions.values():
                for entry in entries:
                    filters = entry.get("filters", {})
                    label = self._build_label(setup_type, filters)
                    if entry.get("label"):
                        label = f"{setup_key}: {entry['label']}" if setup_type else entry["label"]
                    add_candidate(setup_type, filters, label)

        return candidates

    def _candidates_from_trade_scan(
        self,
        trades: list[FilteredTradeRecord],
    ) -> list[tuple[str | None, dict[str, str], str]]:
        candidates: list[tuple[str | None, dict[str, str], str]] = []
        seen: set[tuple[Any, ...]] = set()
        entries = self._entry_trades(trades)

        def add(setup_type: str | None, filters: dict[str, str]) -> None:
            key = self._candidate_key(setup_type, filters)
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                (setup_type, filters, self._build_label(setup_type, filters))
            )

        for setup_type in (None, *PROFITABLE_SETUPS):
            scoped = [
                trade
                for trade in entries
                if setup_type is None or trade.setup_type == setup_type
            ]
            for size in range(1, len(FILTER_DIMENSIONS) + 1):
                for dimensions in itertools.combinations(FILTER_DIMENSIONS, size):
                    grouped: dict[tuple[tuple[str, str], ...], int] = {}
                    for trade in scoped:
                        key = tuple(
                            (dimension, getattr(trade.filters, dimension))
                            for dimension in dimensions
                        )
                        grouped[key] = grouped.get(key, 0) + 1

                    for key, count in grouped.items():
                        if count < self.min_trades:
                            continue
                        add(setup_type, dict(key))

        return candidates

    def _collect_trades(self, report: dict[str, Any]) -> list[FilteredTradeRecord]:
        symbol = report.get("symbol", "NIFTY50")
        research_days = report.get("research_window_days", RESEARCH_DAYS)
        timeframes = tuple(report.get("timeframes_analyzed", ("5M", "15M", "1H")))
        end = (
            date.fromisoformat(report["end_date"])
            if report.get("end_date")
            else date.today()
        )
        start = (
            date.fromisoformat(report["start_date"])
            if report.get("start_date")
            else end - timedelta(days=research_days)
        )

        engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        all_trades: list[FilteredTradeRecord] = []
        for timeframe_label in timeframes:
            path = engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            logger.info(
                "Collecting trades for validation from %s (%s rows).",
                path.name,
                len(frame),
            )
            all_trades.extend(engine._collect_trades(frame, timeframe_label))
        return all_trades

    def run(self) -> RobustFilterReport:
        """Validate filter combinations from the input research report."""
        started = time.perf_counter()
        input_report = self._load_input_report()
        trades = self._collect_trades(input_report)

        input_candidates = self._candidates_from_input(input_report)
        scanned_candidates = self._candidates_from_trade_scan(trades)

        merged: dict[tuple[Any, ...], tuple[str | None, dict[str, str], str]] = {}
        for setup_type, filters, label in input_candidates + scanned_candidates:
            merged[self._candidate_key(setup_type, filters)] = (setup_type, filters, label)

        candidates_evaluated = len(merged)
        removed_low_sample = 0
        ranked: list[ValidatedCombination] = []

        for setup_type, filters, label in merged.values():
            result = self._evaluate_candidate(trades, setup_type, filters, label)
            if result is None:
                removed_low_sample += 1
                continue
            ranked.append(result)

        ranked.sort(key=lambda item: (item.expectancy, item.trades), reverse=True)

        robust = [
            combo
            for combo in ranked
            if combo.is_valid
        ]
        robust.sort(key=lambda item: (item.stability_score, item.validation.expectancy), reverse=True)

        overfit = [
            combo
            for combo in ranked
            if not combo.is_valid
        ]
        overfit.sort(
            key=lambda item: (
                item.train.expectancy,
                item.train.profit_factor or 0,
            ),
            reverse=True,
        )

        top_10_robust = [combo.as_dict() for combo in robust[:10]]
        top_10_overfit = [combo.as_dict() for combo in overfit[:10]]

        production_candidates = [
            combo
            for combo in robust
            if combo.train.expectancy > 0 and combo.validation.expectancy > 0
        ]
        production_candidates.sort(
            key=lambda item: (
                item.setup_type is not None,
                item.stability_score,
                item.validation.expectancy,
            ),
            reverse=True,
        )
        best_combo = production_candidates[0] if production_candidates else (robust[0] if robust else None)
        best_stack = best_combo.as_dict() if best_combo else None
        if best_stack:
            best_stack = {
                **best_stack,
                "recommendation": (
                    "Deploy with monitoring: validation PF, expectancy, and win rate "
                    "remained within robustness thresholds on the last 30% of data."
                ),
            }

        return RobustFilterReport(
            input_report_path=str(self.input_report_path),
            symbol=input_report.get("symbol", "NIFTY50"),
            research_window_days=input_report.get("research_window_days", RESEARCH_DAYS),
            start_date=input_report.get("start_date", ""),
            end_date=input_report.get("end_date", ""),
            timeframes_analyzed=list(input_report.get("timeframes_analyzed", [])),
            setups_analyzed=sorted(PROFITABLE_SETUPS),
            train_split_pct=TRAIN_RATIO * 100,
            validation_split_pct=round((1 - TRAIN_RATIO) * 100, 1),
            min_trades=self.min_trades,
            min_profit_factor=self.min_profit_factor,
            candidates_from_input=len(input_candidates),
            candidates_evaluated=candidates_evaluated,
            candidates_after_trade_filter=len(ranked),
            removed_low_sample_count=removed_low_sample,
            top_10_robust_combinations=top_10_robust,
            top_10_overfit_combinations=top_10_overfit,
            best_production_ready_filter_stack=best_stack,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_robust_filter_report(
    input_report_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> RobustFilterReport:
    """Run robust filter validation and export JSON report."""
    validator = RobustFilterValidator(
        input_report_path=input_report_path or DEFAULT_INPUT_PATH,
    )
    report = validator.run()

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Robust filter validation completed: ranked=%s robust=%s overfit=%s",
        report.candidates_after_trade_filter,
        len(report.top_10_robust_combinations),
        len(report.top_10_overfit_combinations),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_robust_filter_report()
        print("Robust Filter Validation Summary")
        print(f"Input: {report.input_report_path}")
        print(f"Candidates evaluated: {report.candidates_evaluated}")
        print(f"After filters (trades>={report.min_trades}, PF>{report.min_profit_factor}): "
              f"{report.candidates_after_trade_filter}")
        print(f"Robust combinations: {len(report.top_10_robust_combinations)}")
        print(f"Overfit combinations: {len(report.top_10_overfit_combinations)}")
        if report.best_production_ready_filter_stack:
            best = report.best_production_ready_filter_stack
            print("Best Production Stack:")
            print(f"  {best['label']}")
            print(f"  Stability={best['stability_score']} "
                  f"Val PF={best['validation']['profit_factor']} "
                  f"Val Exp={best['validation']['expectancy']}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except RobustFilterValidatorError as exc:
        logger.error("Robust filter validation error: %s", exc)
        print(f"Robust filter validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected robust filter validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
