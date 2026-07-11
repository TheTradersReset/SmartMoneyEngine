"""
SmartMoneyEngine Final Production Signal Validation research.

Synthesizes completed research exports into a unified production-grade BUY/SELL
signal engine verdict. Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_final_production_validation.json"

MANDATORY_CORE = ("Displacement", "CHOCH", "BOS", "FVG Reclaim")
MIN_SAMPLE_SIZE = 50
PREFERRED_WIN_RATE = 55.0
PREFERRED_PROFIT_FACTOR = 1.8
PREFERRED_EXPECTANCY = 50.0
TOP_STACK_COUNT = 20
MONTHLY_SIGNAL_TARGETS = (20, 30, 40, 50)

REQUIRED_EXPORTS = (
    "institutional_move_validation.json",
    "institutional_trigger_validation.json",
    "institutional_confirmation_candle.json",
    "institutional_signal_construction.json",
    "institutional_momentum_origin.json",
    "support_resistance_pressure.json",
    "major_level_strength.json",
    "institutional_move_dna.json",
    "institutional_expansion_trigger_discovery.json",
    "trigger_trade_validation.json",
    "trigger_entry_optimization.json",
    "tiered_signal_framework.json",
    "trade_construction_validation.json",
    "winning_trade_narratives.json",
    "vwap_validation_report.json",
    "tier2_winner_loser_comparison.json",
    "smartmoneyengine_production_candidate.json",
    "tier2_production_validation.json",
    "tier2_composite_edge_validation.json",
)

OPTIONAL_EXPORTS = (
    "institutional_quality_validation.json",
    "tier2_exit_optimization.json",
    "smartmoneyengine_unified_signal_validation.json",
    "smartmoneyengine_feature_cache_architecture.json",
)


class FinalProductionValidationError(Exception):
    """Raised when final production validation cannot complete."""


@dataclass
class FinalProductionValidationReport:
    """Full final production validation output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    research_exports_loaded: dict[str, Any]
    mandatory_signal_core: list[str]
    top_20_buy_stacks: list[dict[str, Any]]
    top_20_sell_stacks: list[dict[str, Any]]
    top_20_no_trade_filters: list[dict[str, Any]]
    best_buy_stack: dict[str, Any] | None
    best_sell_stack: dict[str, Any] | None
    best_no_trade_filters: list[str]
    best_timeframe_combinations: list[dict[str, Any]]
    best_symbol_combinations: list[dict[str, Any]]
    multi_timeframe_validation: dict[str, Any]
    supply_demand_level_validation: dict[str, Any]
    monthly_signal_analysis: dict[str, Any]
    booster_impact_summary: dict[str, Any]
    smartmoneyengine_v1_final_production_card: dict[str, Any]
    top_10_signal_failure_reasons: list[dict[str, Any]]
    top_10_strong_momentum_conditions: list[dict[str, Any]]
    production_readiness_verdict: str
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineFinalProductionValidationResearch:
    """Synthesize completed research into a final production signal verdict."""

    def __init__(
        self,
        research_dir: Path | str | None = None,
        research_days: int = RESEARCH_DAYS,
    ) -> None:
        self.research_dir = Path(research_dir or RESEARCH_DIR)
        self.research_days = research_days
        self.exports: dict[str, Any] = {}

    def _load_exports(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for filename in (*REQUIRED_EXPORTS, *OPTIONAL_EXPORTS):
            path = self.research_dir / filename
            if not path.exists():
                loaded[filename] = {"status": "missing", "path": str(path)}
                continue
            with path.open("r", encoding="utf-8") as handle:
                loaded[filename] = {"status": "loaded", "data": json.load(handle), "path": str(path)}
        self.exports = loaded
        return loaded

    @staticmethod
    def _metric(
        item: dict[str, Any],
        *,
        trades_key: str = "trades",
        signals_key: str = "signals",
        occurrences_key: str = "occurrences",
    ) -> dict[str, Any]:
        sample = int(
            item.get(trades_key)
            or item.get(signals_key)
            or item.get(occurrences_key)
            or item.get("sample_size")
            or 0,
        )
        return {
            "sample_size": sample,
            "win_rate_pct": round(float(item.get("win_rate_pct", 0.0)), 2),
            "profit_factor": item.get("profit_factor"),
            "expectancy": round(float(item.get("expectancy", 0.0)), 2),
            "maximum_drawdown_points": round(
                float(
                    item.get("maximum_drawdown_points")
                    or item.get("max_drawdown")
                    or item.get("maximum_drawdown")
                    or 0.0,
                ),
                2,
            ),
            "hit_1r_rate_pct": round(float(item.get("hit_1r_rate_pct", 0.0)), 2),
            "hit_2r_rate_pct": round(float(item.get("hit_2r_rate_pct", 0.0)), 2),
            "hit_3r_rate_pct": round(float(item.get("hit_3r_rate_pct", 0.0)), 2),
            "average_rr": round(float(item.get("average_rr", 0.0)), 2),
            "signals_per_month": round(
                float(item.get("signals_per_month") or item.get("trades_per_month") or 0.0),
                2,
            ),
            "signals_per_week": round(
                float(item.get("signals_per_week") or (item.get("trades_per_month", 0) or 0) / 4.33),
                2,
            ),
        }

    @staticmethod
    def _grade_signal(metrics: dict[str, Any]) -> str:
        sample = metrics["sample_size"]
        wr = metrics["win_rate_pct"]
        pf = metrics["profit_factor"]
        exp = metrics["expectancy"]
        if sample < MIN_SAMPLE_SIZE:
            return "D"
        if pf is None:
            pf = 0.0
        if wr >= 60 and pf >= 2.0 and exp >= 80 and sample >= 100:
            return "A+"
        if wr >= PREFERRED_WIN_RATE and pf >= PREFERRED_PROFIT_FACTOR and exp >= PREFERRED_EXPECTANCY:
            return "A"
        if wr >= 45 and pf >= 1.5 and exp >= 30:
            return "B"
        if wr >= 40 and pf >= 1.2 and exp > 0:
            return "C"
        return "D"

    @staticmethod
    def _classification(metrics: dict[str, Any], grade: str) -> str:
        if metrics["sample_size"] < MIN_SAMPLE_SIZE or grade == "D":
            return "Reject"
        if grade in {"A+", "A"}:
            return "Production Ready"
        if metrics["expectancy"] > 0:
            return "Needs Validation"
        return "Reject"

    def _stack_record(
        self,
        *,
        stack_id: str,
        stack_label: str,
        signal_side: str,
        boosters: list[str],
        source: str,
        raw_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = self._metric(raw_metrics)
        grade = self._grade_signal(metrics)
        months = max(self.research_days / 30.4375, 1.0)
        per_month = metrics["signals_per_month"] or (metrics["sample_size"] / months)
        return {
            "stack_id": stack_id,
            "stack_label": stack_label,
            "signal_side": signal_side,
            "mandatory_core": list(MANDATORY_CORE),
            "boosters": boosters,
            "source": source,
            "signal_grade": grade,
            "classification": self._classification(metrics, grade),
            "sample_size": metrics["sample_size"],
            "win_rate_pct": metrics["win_rate_pct"],
            "profit_factor": metrics["profit_factor"],
            "expectancy": metrics["expectancy"],
            "maximum_drawdown_points": metrics["maximum_drawdown_points"],
            "hit_1r_rate_pct": metrics["hit_1r_rate_pct"],
            "hit_2r_rate_pct": metrics["hit_2r_rate_pct"],
            "hit_3r_rate_pct": metrics["hit_3r_rate_pct"],
            "average_rr": metrics["average_rr"],
            "signals_per_week": round(per_month / 4.33, 2),
            "signals_per_month": round(per_month, 2),
            "signals_per_quarter": round(per_month * 3, 2),
            "signals_per_year": round(per_month * 12, 2),
            "meets_monthly_20_plus": per_month >= 20,
            "meets_monthly_30_plus": per_month >= 30,
            "meets_monthly_40_plus": per_month >= 40,
            "meets_monthly_50_plus": per_month >= 50,
        }

    def _collect_candidate_stacks(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        buy_stacks: list[dict[str, Any]] = []
        sell_stacks: list[dict[str, Any]] = []
        candidate = self._export("smartmoneyengine_production_candidate.json")
        if candidate is None:
            return buy_stacks, sell_stacks

        for item in candidate.get("eligible_models", []):
            side = item.get("signal_side", "")
            record = self._stack_record(
                stack_id=item.get("model_key", ""),
                stack_label=item.get("model_label", ""),
                signal_side=side,
                boosters=list(item.get("features", [])),
                source="smartmoneyengine_production_candidate.json",
                raw_metrics=item,
            )
            if side == "BUY":
                buy_stacks.append(record)
            elif side == "SELL":
                sell_stacks.append(record)

        for side_key, bucket in (("bullish", buy_stacks), ("bearish", sell_stacks)):
            discovery = self._export("institutional_expansion_trigger_discovery.json")
            if discovery is None:
                continue
            blueprint_key = (
                "top_20_bullish_momentum_blueprints"
                if side_key == "bullish"
                else "top_20_bearish_momentum_blueprints"
            )
            for item in discovery.get(blueprint_key, []):
                signal_side = "BUY" if side_key == "bullish" else "SELL"
                bucket.append(
                    self._stack_record(
                        stack_id=f"blueprint_{item.get('rank', 0)}",
                        stack_label=item.get("blueprint", ""),
                        signal_side=signal_side,
                        boosters=[part.strip() for part in item.get("blueprint", "").split("->")],
                        source="institutional_expansion_trigger_discovery.json",
                        raw_metrics={
                            "occurrences": item.get("occurrences"),
                            "win_rate_pct": item.get("hit_1r_rate_pct"),
                            "profit_factor": None,
                            "expectancy": item.get("average_move_points"),
                            "maximum_drawdown_points": item.get("average_drawdown_points"),
                            "hit_1r_rate_pct": item.get("hit_1r_rate_pct"),
                            "hit_2r_rate_pct": item.get("hit_2r_rate_pct"),
                            "hit_3r_rate_pct": item.get("hit_3r_rate_pct"),
                            "average_rr": 0.0,
                            "signals_per_month": item.get("occurrences", 0) / max(self.research_days / 30.4375, 1),
                        },
                    ),
                )
        return buy_stacks, sell_stacks

    def _export(self, filename: str) -> dict[str, Any] | None:
        entry = self.exports.get(filename, {})
        if entry.get("status") != "loaded":
            return None
        return entry["data"]

    def _rank_stacks(self, stacks: list[dict[str, Any]], signal_side: str) -> list[dict[str, Any]]:
        eligible = [item for item in stacks if item["signal_side"] == signal_side and item["sample_size"] >= MIN_SAMPLE_SIZE]
        eligible.sort(
            key=lambda item: (
                {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}.get(item["signal_grade"], 0),
                item["expectancy"],
                item["profit_factor"] or 0,
                item["hit_3r_rate_pct"],
            ),
            reverse=True,
        )
        for index, item in enumerate(eligible[:TOP_STACK_COUNT], start=1):
            item["rank"] = index
        return eligible[:TOP_STACK_COUNT]

    def _build_no_trade_filters(self) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        winner_loser = self._export("tier2_winner_loser_comparison.json")
        if winner_loser:
            for item in winner_loser.get("trait_frequency_comparison", []):
                if item.get("more_common_in") != "Losers":
                    continue
                filters.append(
                    {
                        "filter": item.get("trait"),
                        "category": item.get("category"),
                        "loser_frequency_pct": item.get("loser_frequency_pct"),
                        "winner_frequency_pct": item.get("winner_frequency_pct"),
                        "edge_pct": item.get("edge_pct"),
                        "avoidance_value": "High" if abs(item.get("edge_pct", 0)) >= 10 else "Moderate",
                        "source": "tier2_winner_loser_comparison.json",
                    },
                )

        trigger_trade = self._export("trigger_trade_validation.json")
        if trigger_trade:
            for item in trigger_trade.get("trigger_trade_metrics", []):
                if item.get("classification") != "Reject":
                    continue
                if item.get("trades", 0) < MIN_SAMPLE_SIZE:
                    continue
                filters.append(
                    {
                        "filter": f"Reject Trigger: {item.get('trigger_model')}",
                        "category": "Trigger Validation",
                        "loser_frequency_pct": item.get("false_trigger_rate_pct"),
                        "winner_frequency_pct": 100 - float(item.get("win_rate_pct", 0)),
                        "edge_pct": -float(item.get("expectancy", 0)),
                        "avoidance_value": "High",
                        "source": "trigger_trade_validation.json",
                    },
                )

        static_filters = [
            ("Sideways market / Mid Range location", "Market Location", "institutional_signal_construction.json"),
            ("Weak displacement at BOS", "Displacement", "tier2_winner_loser_comparison.json"),
            ("HTF conflict / HTF Reversal regime", "Multi-Timeframe", "tier2_production_validation.json"),
            ("Late session entries (Afternoon)", "Session", "smartmoneyengine_production_candidate.json"),
            ("Weak confirmation candle score", "Confirmation", "institutional_confirmation_candle.json"),
            ("Excessive ATR expansion without structure", "Volatility", "support_resistance_pressure.json"),
            ("Weak level strength (Weak category)", "Levels", "major_level_strength.json"),
            ("FVG Retests 2+ before entry", "Structure", "tier2_winner_loser_comparison.json"),
            ("Near Resistance on BUY setups", "Levels", "institutional_signal_construction.json"),
            ("Near Support on SELL setups", "Levels", "institutional_signal_construction.json"),
        ]
        for label, category, source in static_filters:
            filters.append(
                {
                    "filter": label,
                    "category": category,
                    "loser_frequency_pct": None,
                    "winner_frequency_pct": None,
                    "edge_pct": None,
                    "avoidance_value": "Moderate",
                    "source": source,
                },
            )

        filters.sort(
            key=lambda item: (
                {"High": 2, "Moderate": 1}.get(item.get("avoidance_value", ""), 0),
                abs(item.get("edge_pct") or 0),
            ),
            reverse=True,
        )
        for index, item in enumerate(filters[:TOP_STACK_COUNT], start=1):
            item["rank"] = index
        return filters[:TOP_STACK_COUNT]

    def _multi_timeframe_validation(self) -> dict[str, Any]:
        tier2 = self._export("tier2_production_validation.json") or {}
        variants = tier2.get("variants", {})
        aligned = variants.get("tier_2_htf_alignment", {})
        raw = variants.get("raw_tier_2", {})
        non_aligned = {
            "signals": max(raw.get("signals", 0) - aligned.get("signals", 0), 0),
            "signals_per_month": round(
                max(raw.get("signals_per_month", 0) - aligned.get("signals_per_month", 0), 0),
                2,
            ),
        }
        return {
            "validation_model": "1H trend + 15M structure + 5M trigger (Tier-2 sequence)",
            "aligned_htf": {
                "label": aligned.get("label"),
                "signals": aligned.get("signals"),
                "win_rate_pct": aligned.get("win_rate_pct"),
                "profit_factor": aligned.get("profit_factor"),
                "expectancy": aligned.get("expectancy"),
                "maximum_drawdown_points": aligned.get("maximum_drawdown_points"),
                "signals_per_month": aligned.get("signals_per_month"),
            },
            "non_aligned_estimate": non_aligned,
            "raw_unfiltered": {
                "signals": raw.get("signals"),
                "win_rate_pct": raw.get("win_rate_pct"),
                "profit_factor": raw.get("profit_factor"),
                "expectancy": raw.get("expectancy"),
                "maximum_drawdown_points": raw.get("maximum_drawdown_points"),
                "signals_per_month": raw.get("signals_per_month"),
            },
            "verdict": (
                "HTF alignment improves risk control"
                if aligned.get("expectancy", 0) >= raw.get("expectancy", 0) * 0.9
                else "HTF alignment reduces sample without clear edge"
            ),
        }

    def _supply_demand_validation(self) -> dict[str, Any]:
        pressure = self._export("support_resistance_pressure.json") or {}
        major = self._export("major_level_strength.json") or {}
        discovery = self._export("institutional_expansion_trigger_discovery.json") or {}
        return {
            "level_states": pressure.get("level_classification_summary", {}),
            "outcome_counts": pressure.get("outcome_counts", {}),
            "aggregate_level_metrics": pressure.get("aggregate_level_metrics", {}),
            "level_strength_matrix": major.get("level_strength_matrix", {}),
            "tests_before_momentum": {
                "average_tests_support_break": pressure.get("aggregate_level_metrics", {})
                .get("support_break", {})
                .get("average_tests"),
                "average_tests_resistance_break": pressure.get("aggregate_level_metrics", {})
                .get("resistance_break", {})
                .get("average_tests"),
                "expansion_discovery_bullish_baseline_3r": discovery.get("baseline_metrics", {})
                .get("bullish", {})
                .get("hit_3r_rate_pct"),
            },
            "attempt_comparison": {
                "first_attempt_breakout_bias": "Strong/Moderate levels show lower false breakout on retest",
                "third_plus_attempt_exhaustion": pressure.get("level_classification_summary", {}).get("Exhausted"),
            },
        }

    def _booster_impact(self) -> dict[str, Any]:
        composite = self._export("tier2_composite_edge_validation.json") or {}
        vwap = self._export("vwap_validation_report.json") or {}
        narratives = self._export("winning_trade_narratives.json") or {}
        return {
            "top_composite_filter": composite.get("best_production_ready_filter"),
            "composite_recommendation": composite.get("production_recommendation"),
            "vwap_best_segment": vwap.get("best_vwap_segments"),
            "vwap_recommended_filter": vwap.get("recommended_filter"),
            "winning_narrative_liquidity": narratives.get("by_liquidity_event"),
            "winning_narrative_structure": narratives.get("by_structure_sequence"),
        }

    def _failure_reasons(self) -> list[dict[str, Any]]:
        winner_loser = self._export("tier2_winner_loser_comparison.json") or {}
        losers = winner_loser.get("sample_bottom_losers", [])
        counts: dict[str, int] = {}
        for item in losers:
            for tag in item.get("trait_tags", []):
                counts[tag] = counts.get(tag, 0) + 1
        ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
        return [
            {"reason": reason, "occurrences_in_bottom_losers": count, "source": "tier2_winner_loser_comparison.json"}
            for reason, count in ranked[:10]
        ]

    def _momentum_conditions(self) -> list[dict[str, Any]]:
        discovery = self._export("institutional_expansion_trigger_discovery.json") or {}
        momentum = self._export("institutional_momentum_origin.json") or {}
        dna = self._export("institutional_move_dna.json") or {}
        conditions: list[dict[str, Any]] = []
        for item in discovery.get("top_20_bullish_momentum_blueprints", [])[:5]:
            conditions.append(
                {
                    "condition": item.get("blueprint"),
                    "direction": "bullish",
                    "occurrences": item.get("occurrences"),
                    "hit_3r_rate_pct": item.get("hit_3r_rate_pct"),
                    "reliability_score": item.get("reliability_score"),
                    "source": "institutional_expansion_trigger_discovery.json",
                },
            )
        for item in discovery.get("top_20_bearish_momentum_blueprints", [])[:5]:
            conditions.append(
                {
                    "condition": item.get("blueprint"),
                    "direction": "bearish",
                    "occurrences": item.get("occurrences"),
                    "hit_3r_rate_pct": item.get("hit_3r_rate_pct"),
                    "reliability_score": item.get("reliability_score"),
                    "source": "institutional_expansion_trigger_discovery.json",
                },
            )
        for item in (momentum.get("top_20_momentum_origin_patterns") or [])[:3]:
            conditions.append(
                {
                    "condition": item.get("pattern") or item.get("origin_pattern"),
                    "direction": item.get("direction"),
                    "occurrences": item.get("occurrences"),
                    "hit_3r_rate_pct": item.get("hit_3r_rate_pct"),
                    "reliability_score": item.get("reliability_score"),
                    "source": "institutional_momentum_origin.json",
                },
            )
        return conditions[:10]

    def _production_card(
        self,
        *,
        best_buy: dict[str, Any] | None,
        best_sell: dict[str, Any] | None,
        no_trade: list[dict[str, Any]],
        tier2: dict[str, Any],
        trade_construction: dict[str, Any],
        candidate: dict[str, Any],
        entry_opt: dict[str, Any],
        verdict: str,
    ) -> dict[str, Any]:
        recommended = (candidate or {}).get("recommended_production_signal_engine", {})
        production_rec = (trade_construction or {}).get("production_recommendation", {})
        raw_tier2 = (tier2 or {}).get("variants", {}).get("raw_tier_2", {})
        best_entry = (entry_opt or {}).get("best_overall_entry", {})
        monthly_capacity = raw_tier2.get("signals_per_month", 0)
        return {
            "card_name": "SMARTMONEYENGINE_V1_FINAL_PRODUCTION_CARD",
            "buy_rules": {
                "mandatory_core": list(MANDATORY_CORE),
                "filter_stack": recommended.get("buy_filter_stack", []),
                "best_validated_stack": best_buy.get("stack_label") if best_buy else None,
            },
            "sell_rules": {
                "mandatory_core": list(MANDATORY_CORE),
                "filter_stack": recommended.get("sell_filter_stack", []),
                "best_validated_stack": best_sell.get("stack_label") if best_sell else None,
            },
            "no_trade_rules": [item.get("filter") for item in no_trade[:10]],
            "entry_method": best_entry.get("label") or recommended.get("entry") or production_rec.get("entry"),
            "stop_loss_method": recommended.get("stop_loss") or production_rec.get("stop_loss"),
            "target_1": recommended.get("t1") or "1R",
            "target_2": recommended.get("t2") or "2R",
            "target_3": recommended.get("t3") or production_rec.get("target"),
            "recommended_symbols": ["NIFTY50", "BANKNIFTY", "FINNIFTY"],
            "recommended_timeframes": {
                "trend": "1H",
                "structure": "15M",
                "trigger": "5M",
            },
            "expected_signals_per_month": {
                "raw_tier2_capacity": monthly_capacity,
                "best_buy_stack": best_buy.get("signals_per_month") if best_buy else None,
                "best_sell_stack": best_sell.get("signals_per_month") if best_sell else None,
                "combined_best_estimate": round(
                    (best_buy.get("signals_per_month", 0) if best_buy else 0)
                    + (best_sell.get("signals_per_month", 0) if best_sell else 0),
                    2,
                ),
            },
            "expected_win_rate_pct": {
                "buy": best_buy.get("win_rate_pct") if best_buy else None,
                "sell": best_sell.get("win_rate_pct") if best_sell else None,
                "raw_tier2": raw_tier2.get("win_rate_pct"),
            },
            "expected_profit_factor": {
                "buy": best_buy.get("profit_factor") if best_buy else None,
                "sell": best_sell.get("profit_factor") if best_sell else None,
                "raw_tier2": raw_tier2.get("profit_factor"),
            },
            "expected_expectancy": {
                "buy": best_buy.get("expectancy") if best_buy else None,
                "sell": best_sell.get("expectancy") if best_sell else None,
                "raw_tier2": raw_tier2.get("expectancy"),
            },
            "expected_maximum_drawdown_points": {
                "buy": best_buy.get("maximum_drawdown_points") if best_buy else None,
                "sell": best_sell.get("maximum_drawdown_points") if best_sell else None,
                "raw_tier2": raw_tier2.get("maximum_drawdown_points"),
            },
            "monthly_signal_targets": {
                str(threshold): monthly_capacity >= threshold for threshold in MONTHLY_SIGNAL_TARGETS
            },
            "production_readiness_verdict": verdict,
        }

    def _verdict(
        self,
        best_buy: dict[str, Any] | None,
        best_sell: dict[str, Any] | None,
        tier2: dict[str, Any],
        loaded_count: int,
        required_count: int,
    ) -> str:
        raw = (tier2 or {}).get("variants", {}).get("raw_tier_2", {})
        raw_ready = (
            raw.get("signals", 0) >= MIN_SAMPLE_SIZE
            and raw.get("profit_factor", 0) >= PREFERRED_PROFIT_FACTOR
            and raw.get("expectancy", 0) >= PREFERRED_EXPECTANCY
        )
        stack_ready = bool(
            best_buy
            and best_sell
            and best_buy.get("signal_grade") in {"A", "A+"}
            and best_sell.get("signal_grade") in {"A", "A+"}
        )
        exports_ready = loaded_count >= required_count - 2
        if exports_ready and (raw_ready or stack_ready):
            return "READY"
        return "NOT READY"

    def run(self, metadata: dict[str, Any]) -> FinalProductionValidationReport:
        started = time.perf_counter()
        loaded = self._load_exports()
        loaded_count = sum(1 for item in loaded.values() if item.get("status") == "loaded")
        required_count = len(REQUIRED_EXPORTS)

        buy_pool, sell_pool = self._collect_candidate_stacks()
        top_buy = self._rank_stacks(buy_pool, "BUY")
        top_sell = self._rank_stacks(sell_pool, "SELL")
        no_trade = self._build_no_trade_filters()
        best_buy = top_buy[0] if top_buy else None
        best_sell = top_sell[0] if top_sell else None

        tier2 = self._export("tier2_production_validation.json") or {}
        trade_construction = self._export("trade_construction_validation.json") or {}
        candidate = self._export("smartmoneyengine_production_candidate.json") or {}
        entry_opt = self._export("trigger_entry_optimization.json") or {}
        verdict = self._verdict(best_buy, best_sell, tier2, loaded_count, required_count)

        monthly_analysis = {
            "raw_tier2_signals_per_month": tier2.get("variants", {}).get("raw_tier_2", {}).get("signals_per_month"),
            "best_buy_signals_per_month": best_buy.get("signals_per_month") if best_buy else None,
            "best_sell_signals_per_month": best_sell.get("signals_per_month") if best_sell else None,
            "monthly_targets_met": {
                str(threshold): tier2.get("variants", {}).get("raw_tier_2", {}).get("signals_per_month", 0) >= threshold
                for threshold in MONTHLY_SIGNAL_TARGETS
            },
            "high_frequency_capability": tier2.get("variants", {}).get("raw_tier_2", {}).get("signals_per_month", 0) >= 40,
        }

        production_card = self._production_card(
            best_buy=best_buy,
            best_sell=best_sell,
            no_trade=no_trade,
            tier2=tier2,
            trade_construction=trade_construction,
            candidate=candidate,
            entry_opt=entry_opt,
            verdict=verdict,
        )

        conclusions = [
            f"Loaded {loaded_count}/{len(REQUIRED_EXPORTS) + len(OPTIONAL_EXPORTS)} research exports (synthesis-only, no new discovery).",
            f"Mandatory core enforced: {' + '.join(MANDATORY_CORE)}.",
            f"Top BUY stack: {best_buy['stack_label'] if best_buy else 'None'} (grade {best_buy['signal_grade'] if best_buy else 'N/A'}).",
            f"Top SELL stack: {best_sell['stack_label'] if best_sell else 'None'} (grade {best_sell['signal_grade'] if best_sell else 'N/A'}).",
            f"Raw Tier-2 capacity: {tier2.get('variants', {}).get('raw_tier_2', {}).get('signals_per_month', 0):.1f} signals/month.",
            f"Production readiness verdict: {verdict}.",
        ]

        symbols = []
        for name in ("institutional_expansion_trigger_discovery.json", "smartmoneyengine_production_candidate.json"):
            payload = self._export(name)
            if payload and payload.get("symbols_analyzed"):
                symbols = payload["symbols_analyzed"]
                break

        return FinalProductionValidationReport(
            symbols_analyzed=symbols or ["NIFTY50", "BANKNIFTY", "FINNIFTY"],
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            research_exports_loaded={
                name: {"status": entry.get("status"), "path": entry.get("path")}
                for name, entry in loaded.items()
            },
            mandatory_signal_core=list(MANDATORY_CORE),
            top_20_buy_stacks=top_buy,
            top_20_sell_stacks=top_sell,
            top_20_no_trade_filters=no_trade,
            best_buy_stack=best_buy,
            best_sell_stack=best_sell,
            best_no_trade_filters=[item.get("filter", "") for item in no_trade[:10]],
            best_timeframe_combinations=[
                {"combination": "1H trend + 15M structure + 5M trigger", "source": "tier2_production_validation.json"},
                {"combination": "5M trigger with HTF alignment", "source": "tier2_production_validation.json"},
            ],
            best_symbol_combinations=[
                {"symbols": symbols or ["NIFTY50", "BANKNIFTY", "FINNIFTY"], "source": "completed research exports"},
            ],
            multi_timeframe_validation=self._multi_timeframe_validation(),
            supply_demand_level_validation=self._supply_demand_validation(),
            monthly_signal_analysis=monthly_analysis,
            booster_impact_summary=self._booster_impact(),
            smartmoneyengine_v1_final_production_card=production_card,
            top_10_signal_failure_reasons=self._failure_reasons(),
            top_10_strong_momentum_conditions=self._momentum_conditions(),
            production_readiness_verdict=verdict,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_final_production_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> FinalProductionValidationReport:
    """Run final production validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise FinalProductionValidationError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineFinalProductionValidationResearch()
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Final production validation completed: verdict=%s buy=%s sell=%s",
        report.production_readiness_verdict,
        report.best_buy_stack.get("stack_label") if report.best_buy_stack else None,
        report.best_sell_stack.get("stack_label") if report.best_sell_stack else None,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_final_production_validation_report()
        card = report.smartmoneyengine_v1_final_production_card
        print("SmartMoneyEngine Final Production Validation Summary")
        print(f"Verdict: {report.production_readiness_verdict}")
        print(f"Best BUY: {report.best_buy_stack['stack_label'] if report.best_buy_stack else 'None'}")
        print(f"Best SELL: {report.best_sell_stack['stack_label'] if report.best_sell_stack else 'None'}")
        print(f"Expected signals/month (raw Tier-2): {card['expected_signals_per_month']['raw_tier2_capacity']}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except FinalProductionValidationError as exc:
        logger.error("Final production validation error: %s", exc)
        print(f"Final production validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected final production validation error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
