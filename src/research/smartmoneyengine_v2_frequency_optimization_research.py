"""
SmartMoneyEngine V2 Frequency Optimization research.

Increases signal frequency while preserving profitability by analyzing filter
removal impact against the frozen V1 production card. Research-only.
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
from typing import Any, Iterable

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.smartmoneyengine_production_candidate_research import (
    FEATURE_DEFINITIONS,
    ProductionCandidateTrade,
    SmartMoneyEngineProductionCandidateResearch,
)
from src.research.smartmoneyengine_walkforward_validation_research import (
    FILTER_LABEL_TO_KEY,
    MANDATORY_CORE,
    SmartMoneyEngineWalkForwardValidationResearch,
    TRAIN_FRACTION,
    TEST_FRACTION,
)
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_winner_loser_comparison_research import Tier2WinnerLoserComparisonResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_PRODUCTION_CARD_PATH = RESEARCH_DIR / "smartmoneyengine_final_production_validation.json"
DEFAULT_WALKFORWARD_PATH = RESEARCH_DIR / "smartmoneyengine_walkforward_validation.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v2_frequency_optimization.json"

MIN_PROFIT_FACTOR = 1.5
MIN_EXPECTANCY = 75.0
MIN_WIN_RATE = 50.0
FREQUENCY_TARGETS = (20, 30, 40)


class V2FrequencyOptimizationError(Exception):
    """Raised when V2 frequency optimization cannot complete."""


@dataclass(frozen=True)
class EnrichedTier2Trade:
    """Tier-2 candidate with symbol and trait metadata for filter analysis."""

    symbol: str
    trade: ProductionCandidateTrade
    trait_tags: tuple[str, ...]

    @property
    def signal_side(self) -> str:
        return self.trade.signal_side

    @property
    def bos_timestamp(self) -> str:
        return self.trade.bos_timestamp


@dataclass
class FilterScenarioMetrics:
    """Metrics for one filter configuration scenario."""

    label: str
    buy_filter_stack: list[str]
    sell_filter_stack: list[str]
    no_trade_rules: list[str]
    sample_size: int
    signals_per_month: float
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    maximum_drawdown_points: float
    hit_1r_rate_pct: float
    hit_2r_rate_pct: float
    hit_3r_rate_pct: float
    net_points: float
    meets_quality_thresholds: bool
    frequency_targets_met: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FilterRemovalImpact:
    """Impact of removing one filter from the V1 baseline."""

    filter_name: str
    filter_category: str
    signals_removed: int
    win_rate_change: float
    profit_factor_change: float | None
    expectancy_change: float
    signals_per_month_change: float
    scenario_metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class V2FrequencyOptimizationReport:
    """Full V2 frequency optimization output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    walkforward_reference: dict[str, Any]
    v1_production_card: dict[str, Any]
    v1_baseline_metrics: dict[str, Any]
    total_tier2_signals: int
    single_filter_removal_analysis: dict[str, Any]
    filter_combination_analysis: list[dict[str, Any]]
    minimum_filter_sets: list[dict[str, Any]]
    frequency_target_analysis: dict[str, Any]
    smartmoneyengine_v2_production_card: dict[str, Any]
    v1_vs_v2_comparison: dict[str, Any]
    walkforward_v2_validation: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineV2FrequencyOptimizationResearch:
    """Optimize V1 filter stacks for higher frequency while preserving edge."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
        production_card_path: Path | str | None = None,
        walkforward_path: Path | str | None = None,
    ) -> None:
        self.symbols = symbols or ("NIFTY50", "BANKNIFTY", "FINNIFTY")
        self.research_days = research_days
        self.timeframes = timeframes
        self.production_card_path = Path(production_card_path or DEFAULT_PRODUCTION_CARD_PATH)
        self.walkforward_path = Path(walkforward_path or DEFAULT_WALKFORWARD_PATH)
        self._walkforward_engine = SmartMoneyEngineWalkForwardValidationResearch(
            symbols=self.symbols,
            research_days=research_days,
            timeframes=timeframes,
            production_card_path=self.production_card_path,
        )

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _keys_from_labels(labels: Iterable[str]) -> tuple[str, ...]:
        keys: list[str] = []
        for label in labels:
            key = FILTER_LABEL_TO_KEY.get(label)
            if key is None:
                raise V2FrequencyOptimizationError(f"Unknown filter label: {label}")
            keys.append(key)
        return tuple(keys)

    @staticmethod
    def _labels_from_keys(keys: Iterable[str]) -> list[str]:
        return [FEATURE_DEFINITIONS[key] for key in keys]

    def _load_v1_card(self) -> dict[str, Any]:
        if not self.production_card_path.exists():
            raise V2FrequencyOptimizationError(
                f"Production card not found: {self.production_card_path}",
            )
        with self.production_card_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        card = payload.get("smartmoneyengine_v1_final_production_card")
        if not card:
            raise V2FrequencyOptimizationError("V1 production card missing from export.")
        return card

    def _load_walkforward_reference(self) -> dict[str, Any]:
        if not self.walkforward_path.exists():
            logger.warning("Walk-forward export missing; baseline metrics computed from trades.")
            return {}
        with self.walkforward_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _no_trade_blocked(trait_tags: tuple[str, ...], rules: list[str]) -> bool:
        return SmartMoneyEngineWalkForwardValidationResearch._no_trade_blocked(trait_tags, rules)

    @staticmethod
    def _matches_side(trade: EnrichedTier2Trade, side: str, feature_keys: tuple[str, ...]) -> bool:
        if trade.signal_side != side:
            return False
        return SmartMoneyEngineProductionCandidateResearch._matches(trade.trade, feature_keys)

    def _collect_enriched_trades(self, metadata: dict[str, Any]) -> tuple[list[EnrichedTier2Trade], int]:
        enriched: list[EnrichedTier2Trade] = []
        tier2_total = 0
        for symbol in self.symbols:
            candidate_engine = SmartMoneyEngineProductionCandidateResearch(
                symbol=symbol,
                research_days=self.research_days,
                timeframes=self.timeframes,
            )
            comparison_engine = Tier2WinnerLoserComparisonResearch(
                symbol=symbol,
                research_days=self.research_days,
                timeframes=self.timeframes,
            )
            trait_lookup = {
                (record.timeframe, record.bos_timestamp): record.trait_tags
                for record in comparison_engine._collect_records(metadata)
            }
            for trade in candidate_engine._collect_candidates(metadata):
                tier2_total += 1
                trait_tags = trait_lookup.get((trade.timeframe, trade.bos_timestamp), ())
                enriched.append(EnrichedTier2Trade(symbol=symbol, trade=trade, trait_tags=trait_tags))
        return enriched, tier2_total

    def _apply_configuration(
        self,
        trades: list[EnrichedTier2Trade],
        buy_keys: tuple[str, ...],
        sell_keys: tuple[str, ...],
        no_trade_rules: list[str],
    ) -> list[EnrichedTier2Trade]:
        selected: list[EnrichedTier2Trade] = []
        for item in trades:
            if self._no_trade_blocked(item.trait_tags, no_trade_rules):
                continue
            if item.signal_side == "BUY" and self._matches_side(item, "BUY", buy_keys):
                selected.append(item)
            elif item.signal_side == "SELL" and self._matches_side(item, "SELL", sell_keys):
                selected.append(item)
        return selected

    def _aggregate_trades(
        self,
        trades: list[EnrichedTier2Trade],
        research_days: int,
    ) -> dict[str, Any]:
        pnls = [item.trade.realized_pnl_points for item in trades]
        total = len(trades)
        wins = sum(1 for item in trades if item.trade.win)
        months = max(research_days / 30.4375, 1.0)
        pf = self._profit_factor(pnls)
        expectancy = round(mean(pnls), 2) if pnls else 0.0
        wr = round(wins / total * 100, 2) if total else 0.0
        return {
            "sample_size": total,
            "signals_per_month": round(total / months, 2) if total else 0.0,
            "win_rate_pct": wr,
            "profit_factor": pf,
            "expectancy": expectancy,
            "maximum_drawdown_points": self._maximum_drawdown(pnls),
            "hit_1r_rate_pct": round(
                sum(1 for item in trades if item.trade.hit_1r_before_sl) / total * 100,
                2,
            )
            if total
            else 0.0,
            "hit_2r_rate_pct": round(
                sum(1 for item in trades if item.trade.hit_2r_before_sl) / total * 100,
                2,
            )
            if total
            else 0.0,
            "hit_3r_rate_pct": round(
                sum(1 for item in trades if item.trade.hit_3r_before_sl) / total * 100,
                2,
            )
            if total
            else 0.0,
            "net_points": round(sum(pnls), 2),
        }

    @staticmethod
    def _meets_thresholds(metrics: dict[str, Any]) -> bool:
        pf = metrics.get("profit_factor")
        if pf is None or pf < MIN_PROFIT_FACTOR:
            return False
        return (
            metrics.get("expectancy", 0.0) > MIN_EXPECTANCY
            and metrics.get("win_rate_pct", 0.0) > MIN_WIN_RATE
        )

    @staticmethod
    def _frequency_targets(signals_per_month: float) -> dict[str, bool]:
        return {f"{target}+": signals_per_month >= target for target in FREQUENCY_TARGETS}

    def _scenario_metrics(
        self,
        *,
        label: str,
        buy_labels: list[str],
        sell_labels: list[str],
        no_trade_rules: list[str],
        trades: list[EnrichedTier2Trade],
        research_days: int,
    ) -> FilterScenarioMetrics:
        buy_keys = self._keys_from_labels(buy_labels)
        sell_keys = self._keys_from_labels(sell_labels)
        selected = self._apply_configuration(trades, buy_keys, sell_keys, no_trade_rules)
        metrics = self._aggregate_trades(selected, research_days)
        return FilterScenarioMetrics(
            label=label,
            buy_filter_stack=buy_labels,
            sell_filter_stack=sell_labels,
            no_trade_rules=no_trade_rules,
            meets_quality_thresholds=self._meets_thresholds(metrics),
            frequency_targets_met=self._frequency_targets(metrics["signals_per_month"]),
            **metrics,
        )

    def _signals_blocked_by_filter(
        self,
        trades: list[EnrichedTier2Trade],
        side: str,
        filter_keys: tuple[str, ...],
        target_key: str,
        no_trade_rules: list[str],
    ) -> int:
        other_keys = tuple(key for key in filter_keys if key != target_key)
        count = 0
        for item in trades:
            if self._no_trade_blocked(item.trait_tags, no_trade_rules):
                continue
            if item.signal_side != side:
                continue
            if other_keys and not SmartMoneyEngineProductionCandidateResearch._matches(
                item.trade,
                other_keys,
            ):
                continue
            if not item.trade.feature_flags.get(target_key, False):
                count += 1
        return count

    def _single_filter_removal(
        self,
        trades: list[EnrichedTier2Trade],
        card: dict[str, Any],
        baseline: FilterScenarioMetrics,
        research_days: int,
    ) -> dict[str, Any]:
        buy_labels = list(card["buy_rules"]["filter_stack"])
        sell_labels = list(card["sell_rules"]["filter_stack"])
        no_trade_rules = list(card.get("no_trade_rules", []))
        buy_keys = self._keys_from_labels(buy_labels)
        sell_keys = self._keys_from_labels(sell_labels)

        buy_impacts: list[FilterRemovalImpact] = []
        for key in buy_keys:
            reduced = [label for label in buy_labels if FILTER_LABEL_TO_KEY[label] != key]
            scenario = self._scenario_metrics(
                label=f"BUY minus {FEATURE_DEFINITIONS[key]}",
                buy_labels=reduced,
                sell_labels=sell_labels,
                no_trade_rules=no_trade_rules,
                trades=trades,
                research_days=research_days,
            )
            buy_impacts.append(
                FilterRemovalImpact(
                    filter_name=FEATURE_DEFINITIONS[key],
                    filter_category="buy_filter",
                    signals_removed=self._signals_blocked_by_filter(
                        trades,
                        "BUY",
                        buy_keys,
                        key,
                        no_trade_rules,
                    ),
                    win_rate_change=round(
                        scenario.win_rate_pct - baseline.win_rate_pct,
                        2,
                    ),
                    profit_factor_change=(
                        round(scenario.profit_factor - baseline.profit_factor, 2)
                        if scenario.profit_factor is not None and baseline.profit_factor is not None
                        else None
                    ),
                    expectancy_change=round(scenario.expectancy - baseline.expectancy, 2),
                    signals_per_month_change=round(
                        scenario.signals_per_month - baseline.signals_per_month,
                        2,
                    ),
                    scenario_metrics=scenario.as_dict(),
                ),
            )

        sell_impacts: list[FilterRemovalImpact] = []
        for key in sell_keys:
            reduced = [label for label in sell_labels if FILTER_LABEL_TO_KEY[label] != key]
            scenario = self._scenario_metrics(
                label=f"SELL minus {FEATURE_DEFINITIONS[key]}",
                buy_labels=buy_labels,
                sell_labels=reduced,
                no_trade_rules=no_trade_rules,
                trades=trades,
                research_days=research_days,
            )
            sell_impacts.append(
                FilterRemovalImpact(
                    filter_name=FEATURE_DEFINITIONS[key],
                    filter_category="sell_filter",
                    signals_removed=self._signals_blocked_by_filter(
                        trades,
                        "SELL",
                        sell_keys,
                        key,
                        no_trade_rules,
                    ),
                    win_rate_change=round(
                        scenario.win_rate_pct - baseline.win_rate_pct,
                        2,
                    ),
                    profit_factor_change=(
                        round(scenario.profit_factor - baseline.profit_factor, 2)
                        if scenario.profit_factor is not None and baseline.profit_factor is not None
                        else None
                    ),
                    expectancy_change=round(scenario.expectancy - baseline.expectancy, 2),
                    signals_per_month_change=round(
                        scenario.signals_per_month - baseline.signals_per_month,
                        2,
                    ),
                    scenario_metrics=scenario.as_dict(),
                ),
            )

        no_trade_impacts: list[FilterRemovalImpact] = []
        for rule in no_trade_rules:
            reduced_rules = [item for item in no_trade_rules if item != rule]
            scenario = self._scenario_metrics(
                label=f"NO-TRADE minus {rule}",
                buy_labels=buy_labels,
                sell_labels=sell_labels,
                no_trade_rules=reduced_rules,
                trades=trades,
                research_days=research_days,
            )
            baseline_selected = self._apply_configuration(
                trades,
                buy_keys,
                sell_keys,
                no_trade_rules,
            )
            without_rule = self._apply_configuration(
                trades,
                buy_keys,
                sell_keys,
                reduced_rules,
            )
            baseline_keys = {
                (item.symbol, item.trade.timeframe, item.bos_timestamp) for item in baseline_selected
            }
            added = sum(
                1
                for item in without_rule
                if (item.symbol, item.trade.timeframe, item.bos_timestamp) not in baseline_keys
            )
            no_trade_impacts.append(
                FilterRemovalImpact(
                    filter_name=rule,
                    filter_category="no_trade_rule",
                    signals_removed=added,
                    win_rate_change=round(
                        scenario.win_rate_pct - baseline.win_rate_pct,
                        2,
                    ),
                    profit_factor_change=(
                        round(scenario.profit_factor - baseline.profit_factor, 2)
                        if scenario.profit_factor is not None and baseline.profit_factor is not None
                        else None
                    ),
                    expectancy_change=round(scenario.expectancy - baseline.expectancy, 2),
                    signals_per_month_change=round(
                        scenario.signals_per_month - baseline.signals_per_month,
                        2,
                    ),
                    scenario_metrics=scenario.as_dict(),
                ),
            )

        def _sort_impacts(items: list[FilterRemovalImpact]) -> list[dict[str, Any]]:
            return [
                item.as_dict()
                for item in sorted(items, key=lambda row: row.signals_removed, reverse=True)
            ]

        return {
            "buy_filters": _sort_impacts(buy_impacts),
            "sell_filters": _sort_impacts(sell_impacts),
            "no_trade_rules": _sort_impacts(no_trade_impacts),
            "largest_signal_removers": _sort_impacts(
                buy_impacts + sell_impacts + no_trade_impacts,
            )[:10],
        }

    @staticmethod
    def _subsets(items: list[str]) -> list[list[str]]:
        result: list[list[str]] = []
        for size in range(len(items) + 1):
            for combo in combinations(items, size):
                result.append(list(combo))
        return result

    def _combination_analysis(
        self,
        trades: list[EnrichedTier2Trade],
        card: dict[str, Any],
        research_days: int,
    ) -> list[dict[str, Any]]:
        buy_labels = list(card["buy_rules"]["filter_stack"])
        sell_labels = list(card["sell_rules"]["filter_stack"])
        no_trade_rules = list(card.get("no_trade_rules", []))

        scenarios: list[FilterScenarioMetrics] = []
        for buy_subset in self._subsets(buy_labels):
            for sell_subset in self._subsets(sell_labels):
                for removed_count in range(len(no_trade_rules) + 1):
                    for removed in combinations(no_trade_rules, removed_count):
                        remaining = [rule for rule in no_trade_rules if rule not in removed]
                        removed_names = list(removed)
                        label_parts = [
                            f"BUY[{len(buy_subset)}/{len(buy_labels)}]",
                            f"SELL[{len(sell_subset)}/{len(sell_labels)}]",
                            f"NO-TRADE[-{len(removed_names)}]",
                        ]
                        scenarios.append(
                            self._scenario_metrics(
                                label=" ".join(label_parts),
                                buy_labels=buy_subset,
                                sell_labels=sell_subset,
                                no_trade_rules=remaining,
                                trades=trades,
                                research_days=research_days,
                            ),
                        )

        qualifying = [item for item in scenarios if item.meets_quality_thresholds]
        qualifying.sort(
            key=lambda item: (item.signals_per_month, item.expectancy),
            reverse=True,
        )
        return [item.as_dict() for item in qualifying[:50]]

    def _minimum_filter_sets(
        self,
        combination_analysis: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not combination_analysis:
            return []
        ranked: list[dict[str, Any]] = []
        for item in combination_analysis:
            filter_count = (
                len(item["buy_filter_stack"])
                + len(item["sell_filter_stack"])
                + len(item.get("no_trade_rules", []))
            )
            ranked.append(
                {
                    **item,
                    "total_active_filters": filter_count,
                    "filters_removed_from_v1": {
                        "buy_filters_removed": item.get("buy_filter_stack"),
                        "sell_filters_removed": item.get("sell_filter_stack"),
                        "no_trade_rules_remaining": len(item.get("no_trade_rules", [])),
                    },
                },
            )
        ranked.sort(
            key=lambda row: (
                -row["signals_per_month"],
                row["total_active_filters"],
                -row["expectancy"],
            ),
        )
        return ranked[:20]

    def _select_v2_card(
        self,
        card: dict[str, Any],
        minimum_sets: list[dict[str, Any]],
        baseline: FilterScenarioMetrics,
    ) -> dict[str, Any]:
        best = minimum_sets[0] if minimum_sets else baseline.as_dict()
        buy_stack = best["buy_filter_stack"]
        sell_stack = best["sell_filter_stack"]
        no_trade = best["no_trade_rules"]

        return {
            "card_name": "SMARTMONEYENGINE_V2_FREQUENCY_OPTIMIZED_CARD",
            "derived_from": "SMARTMONEYENGINE_V1_FINAL_PRODUCTION_CARD",
            "optimization_objective": (
                f"Maximize signals/month while maintaining PF>={MIN_PROFIT_FACTOR}, "
                f"Expectancy>{MIN_EXPECTANCY}, WR>{MIN_WIN_RATE}%"
            ),
            "mandatory_signal_core": list(MANDATORY_CORE),
            "buy_rules": {
                "mandatory_core": list(MANDATORY_CORE),
                "filter_stack": buy_stack,
                "v1_filter_stack": list(card["buy_rules"]["filter_stack"]),
                "filters_removed": [
                    label
                    for label in card["buy_rules"]["filter_stack"]
                    if label not in buy_stack
                ],
            },
            "sell_rules": {
                "mandatory_core": list(MANDATORY_CORE),
                "filter_stack": sell_stack,
                "v1_filter_stack": list(card["sell_rules"]["filter_stack"]),
                "filters_removed": [
                    label
                    for label in card["sell_rules"]["filter_stack"]
                    if label not in sell_stack
                ],
            },
            "no_trade_rules": no_trade,
            "v1_no_trade_rules": list(card.get("no_trade_rules", [])),
            "no_trade_rules_removed": [
                rule for rule in card.get("no_trade_rules", []) if rule not in no_trade
            ],
            "entry_method": card.get("entry_method"),
            "stop_loss_method": card.get("stop_loss_method"),
            "target_1": card.get("target_1", "1R"),
            "target_2": card.get("target_2", "2R"),
            "target_3": card.get("target_3"),
            "recommended_symbols": card.get("recommended_symbols", list(self.symbols)),
            "recommended_timeframes": card.get("recommended_timeframes"),
            "expected_metrics": {
                "signals_per_month": best.get("signals_per_month"),
                "win_rate_pct": best.get("win_rate_pct"),
                "profit_factor": best.get("profit_factor"),
                "expectancy": best.get("expectancy"),
                "maximum_drawdown_points": best.get("maximum_drawdown_points"),
            },
            "frequency_targets_met": best.get("frequency_targets_met", {}),
        }

    @staticmethod
    def _comparison_row(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "signals_per_month": metrics.get("signals_per_month"),
            "win_rate_pct": metrics.get("win_rate_pct"),
            "profit_factor": metrics.get("profit_factor"),
            "expectancy": metrics.get("expectancy"),
            "maximum_drawdown_points": metrics.get("maximum_drawdown_points"),
            "sample_size": metrics.get("sample_size"),
        }

    def _walkforward_v2_validation(
        self,
        metadata: dict[str, Any],
        v2_card: dict[str, Any],
        enriched: list[EnrichedTier2Trade],
    ) -> dict[str, Any]:
        buy_keys = self._keys_from_labels(v2_card["buy_rules"]["filter_stack"])
        sell_keys = self._keys_from_labels(v2_card["sell_rules"]["filter_stack"])
        no_trade_rules = list(v2_card["no_trade_rules"])
        selected = self._apply_configuration(enriched, buy_keys, sell_keys, no_trade_rules)

        start, train_end, test_start, end = self._walkforward_engine._split_dates(metadata)
        train_days = max((train_end - start).days + 1, 1)
        test_days = max((end - test_start).days + 1, 1)
        full_days = max((end - start).days + 1, 1)

        def _period_metrics(period_start, period_end, scope: str, period_days: int) -> dict[str, Any]:
            bucket = [
                item
                for item in selected
                if self._walkforward_engine._in_period(item.bos_timestamp, period_start, period_end)
            ]
            return self._aggregate_trades(bucket, period_days)

        return {
            "train_fraction": TRAIN_FRACTION,
            "test_fraction": TEST_FRACTION,
            "in_sample": _period_metrics(start, train_end, "in_sample", train_days),
            "out_of_sample": _period_metrics(test_start, end, "out_of_sample", test_days),
            "full_sample": _period_metrics(start, end, "full", full_days),
        }

    def run(self, metadata: dict[str, Any]) -> V2FrequencyOptimizationReport:
        started = time.perf_counter()
        card = self._load_v1_card()
        walkforward_ref = self._load_walkforward_reference()
        research_days = metadata.get("research_window_days", self.research_days)

        enriched, tier2_total = self._collect_enriched_trades(metadata)
        buy_labels = list(card["buy_rules"]["filter_stack"])
        sell_labels = list(card["sell_rules"]["filter_stack"])
        no_trade_rules = list(card.get("no_trade_rules", []))

        baseline = self._scenario_metrics(
            label="V1 Baseline",
            buy_labels=buy_labels,
            sell_labels=sell_labels,
            no_trade_rules=no_trade_rules,
            trades=enriched,
            research_days=research_days,
        )

        single_removal = self._single_filter_removal(
            enriched,
            card,
            baseline,
            research_days,
        )
        combinations = self._combination_analysis(enriched, card, research_days)
        minimum_sets = self._minimum_filter_sets(combinations)
        v2_card = self._select_v2_card(card, minimum_sets, baseline)

        v1_metrics = (
            walkforward_ref.get("in_sample_metrics", {}).get("overall")
            if walkforward_ref
            else baseline.as_dict()
        )
        if not v1_metrics:
            v1_metrics = baseline.as_dict()

        v2_full = (
            minimum_sets[0]
            if minimum_sets
            else baseline.as_dict()
        )
        v1_vs_v2 = {
            "v1": self._comparison_row(v1_metrics),
            "v2": self._comparison_row(v2_full),
            "delta": {
                "signals_per_month": round(
                    v2_full.get("signals_per_month", 0) - v1_metrics.get("signals_per_month", 0),
                    2,
                ),
                "win_rate_pct": round(
                    v2_full.get("win_rate_pct", 0) - v1_metrics.get("win_rate_pct", 0),
                    2,
                ),
                "profit_factor_change": (
                    round(v2_full.get("profit_factor", 0) - (v1_metrics.get("profit_factor") or 0), 2)
                    if v2_full.get("profit_factor") is not None
                    else None
                ),
                "expectancy": round(
                    v2_full.get("expectancy", 0) - v1_metrics.get("expectancy", 0),
                    2,
                ),
                "maximum_drawdown_points": round(
                    v2_full.get("maximum_drawdown_points", 0)
                    - v1_metrics.get("maximum_drawdown_points", 0),
                    2,
                ),
            },
        }

        wf_v2 = self._walkforward_v2_validation(metadata, v2_card, enriched)

        frequency_target_analysis = {
            "targets": list(FREQUENCY_TARGETS),
            "v1_baseline": baseline.frequency_targets_met,
            "v2_selected": v2_card.get("frequency_targets_met", {}),
            "qualifying_scenarios_by_target": {
                f"{target}+": sum(
                    1 for item in combinations if item.get("frequency_targets_met", {}).get(f"{target}+")
                )
                for target in FREQUENCY_TARGETS
            },
        }

        top_removers = single_removal.get("largest_signal_removers", [])[:3]
        conclusions = [
            "V2 frequency optimization uses frozen V1 card and walk-forward validation baseline.",
            (
                f"V1 baseline: {baseline.sample_size} signals, "
                f"{baseline.signals_per_month}/month, WR {baseline.win_rate_pct}%, "
                f"PF {baseline.profit_factor}, expectancy {baseline.expectancy}."
            ),
            (
                f"Top signal removers: "
                + ", ".join(
                    f"{item['filter_name']} ({item['signals_removed']} signals)"
                    for item in top_removers
                )
                + "."
            ),
            (
                f"Qualifying filter combinations (PF>={MIN_PROFIT_FACTOR}, "
                f"Exp>{MIN_EXPECTANCY}, WR>{MIN_WIN_RATE}%): {len(combinations)}."
            ),
            (
                f"V2 card: BUY [{', '.join(v2_card['buy_rules']['filter_stack']) or 'none'}], "
                f"SELL [{', '.join(v2_card['sell_rules']['filter_stack']) or 'none'}], "
                f"{len(v2_card['no_trade_rules'])} NO-TRADE rules."
            ),
            (
                f"V2 vs V1: signals/month {v1_vs_v2['v1']['signals_per_month']} -> "
                f"{v1_vs_v2['v2']['signals_per_month']}, "
                f"expectancy {v1_vs_v2['v1']['expectancy']} -> {v1_vs_v2['v2']['expectancy']}."
            ),
            (
                f"Frequency targets — 20+: {v2_card.get('frequency_targets_met', {}).get('20+')}, "
                f"30+: {v2_card.get('frequency_targets_met', {}).get('30+')}, "
                f"40+: {v2_card.get('frequency_targets_met', {}).get('40+')}."
            ),
        ]

        return V2FrequencyOptimizationReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=research_days,
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            walkforward_reference={
                "path": str(self.walkforward_path),
                "survival_verdict": walkforward_ref.get("survival_verdict"),
                "in_sample_overall": walkforward_ref.get("in_sample_metrics", {}).get("overall"),
                "out_of_sample_overall": walkforward_ref.get("out_of_sample_metrics", {}).get("overall"),
            },
            v1_production_card=card,
            v1_baseline_metrics=baseline.as_dict(),
            total_tier2_signals=tier2_total,
            single_filter_removal_analysis=single_removal,
            filter_combination_analysis=combinations,
            minimum_filter_sets=minimum_sets,
            frequency_target_analysis=frequency_target_analysis,
            smartmoneyengine_v2_production_card=v2_card,
            v1_vs_v2_comparison=v1_vs_v2,
            walkforward_v2_validation=wf_v2,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_v2_frequency_optimization_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    production_card_path: Path | str | None = None,
    walkforward_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> V2FrequencyOptimizationReport:
    """Run V2 frequency optimization and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise V2FrequencyOptimizationError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineV2FrequencyOptimizationResearch(
        symbols=symbols,
        production_card_path=production_card_path,
        walkforward_path=walkforward_path,
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "V2 frequency optimization completed: v2_signals_per_month=%s qualifying=%s",
        report.v1_vs_v2_comparison["v2"]["signals_per_month"],
        len(report.filter_combination_analysis),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_v2_frequency_optimization_report()
        print("SmartMoneyEngine V2 Frequency Optimization Summary")
        print(f"V1 signals/month: {report.v1_vs_v2_comparison['v1']['signals_per_month']}")
        print(f"V2 signals/month: {report.v1_vs_v2_comparison['v2']['signals_per_month']}")
        print(f"V1 expectancy: {report.v1_vs_v2_comparison['v1']['expectancy']}")
        print(f"V2 expectancy: {report.v1_vs_v2_comparison['v2']['expectancy']}")
        print(f"Qualifying combinations: {len(report.filter_combination_analysis)}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except V2FrequencyOptimizationError as exc:
        logger.error("V2 frequency optimization error: %s", exc)
        print(f"V2 frequency optimization error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected V2 frequency optimization error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
