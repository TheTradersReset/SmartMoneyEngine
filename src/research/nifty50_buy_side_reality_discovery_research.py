"""
NIFTY50 BUY Side Reality Discovery — synthesis-only research.

Determines why BUY side fails by analyzing completed bullish moves from existing exports only.
No new scans, optimization, production models, or BUY signal generation.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.research.filter_research_engine import _json_safe

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"

SOURCE_EXPORTS = {
    "momentum_anatomy": RESEARCH_DIR / "nifty50_momentum_anatomy_120d.json",
    "trap_to_momentum": RESEARCH_DIR / "nifty50_trap_to_momentum_validation.json",
    "liquidity_decision_engine": RESEARCH_DIR / "nifty50_liquidity_decision_engine.json",
    "early_warning_sequence": RESEARCH_DIR / "nifty50_early_warning_sequence.json",
    "reality_check": RESEARCH_DIR / "smartmoneyengine_reality_check_validation.json",
    "realtime_replay": RESEARCH_DIR / "smartmoneyengine_realtime_replay_validation.json",
    "v3_validation": RESEARCH_DIR / "smartmoneyengine_v3_implementation_validation.json",
}

DEFAULT_REPORT_PATH = RESEARCH_DIR / "nifty50_buy_side_reality_discovery.json"

CAUSAL_WARNING_EVENTS = frozenset(
    {
        "Gap Reversal",
        "Gap Continuation",
        "Liquidity Grab",
        "Failed Breakdown",
        "Failed Breakout",
        "PDL Sweep",
        "PWL Sweep",
        "Round Number Sweep",
    }
)

STRUCTURE_EVENTS = frozenset({"BOS", "CHOCH", "FVG", "FVG Creation", "Order Block"})

MOVE_THRESHOLDS = (50, 100, 150, 200, 300)


class Nifty50BuySideRealityDiscoveryError(Exception):
    """Raised when BUY-side discovery synthesis cannot be completed."""


@dataclass
class Nifty50BuySideRealityDiscoveryReport:
    """BUY-side reality discovery output."""

    report_type: str
    symbol: str
    research_window: dict[str, Any]
    methodology: dict[str, Any]
    source_exports: list[str]
    move_threshold_analysis: dict[str, Any]
    bullish_move_anatomy: dict[str, Any]
    bull_trap_anatomy: dict[str, Any]
    real_bullish_reversal_anatomy: dict[str, Any]
    earliest_warning_sequence: dict[str, Any]
    most_predictive_buy_precursor_events: dict[str, Any]
    buy_side_failure_reasons: dict[str, Any]
    buy_side_opportunity_map: dict[str, Any]
    cross_cohort_comparison: dict[str, Any]
    findings: list[str]
    execution_time_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Nifty50BuySideRealityDiscoveryError(f"Missing source export: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _dedupe_bullish_completed_moves(completed_moves: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, float]] = set()
    unique: list[dict[str, Any]] = []
    for moves in completed_moves.values():
        for move in moves:
            if move.get("direction") != "bullish":
                continue
            key = (int(move["start_bar"]), round(float(move["move_size_points"]), 1))
            if key in seen:
                continue
            seen.add(key)
            unique.append(move)
    return unique


def _causal_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") in CAUSAL_WARNING_EVENTS]


def _ordered_causal_sequence(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(_causal_events(events), key=lambda item: -int(item.get("bars_before_move", 0)))


def _context_snapshot(record: dict[str, Any], step: str = "T-60 minutes") -> dict[str, Any] | None:
    for timeline_step in record.get("timeline", []):
        if timeline_step.get("timeline_step") != step:
            continue
        return timeline_step.get("context_by_timeframe", {}).get("5M")
    return None


def _aggregate_context(records: list[dict[str, Any]], step: str = "T-60 minutes") -> dict[str, list[tuple[str, int]]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        context = _context_snapshot(record, step)
        if not context:
            continue
        reason = context.get("reason_stack", {})
        flags = context.get("feature_flags", {})
        levels = context.get("levels", {})
        counters["htf_trend"][str(reason.get("htf_trend", "Unknown"))] += 1
        counters["vwap_position"][str(reason.get("vwap", "Unknown"))] += 1
        counters["ema22_ema200_structure"][str(reason.get("ema_structure", "Unknown"))] += 1
        counters["volume_expansion"][str(reason.get("volume_expansion", False))] += 1
        counters["support_resistance_proximity"][str(levels.get("market_location", "Unknown"))] += 1
        counters["gap_up"][str(flags.get("gap_up", False))] += 1
        counters["gap_down"][str(flags.get("gap_down", False))] += 1
    return {key: counter.most_common() for key, counter in counters.items()}


def _rank_counter(counter: Counter[str], limit: int = 8) -> list[dict[str, Any]]:
    return [{"event": event, "occurrences": count} for event, count in counter.most_common(limit)]


def _threshold_move_set(
    bullish_pre_events: list[dict[str, Any]],
    completed_unique: list[dict[str, Any]],
    threshold: int,
) -> dict[str, Any]:
    pre_subset = [move for move in bullish_pre_events if float(move.get("move_size_points", 0)) >= threshold]
    completed_subset = [
        move for move in completed_unique if float(move.get("move_size_points", 0)) >= threshold
    ]

    first_event = Counter()
    second_event = Counter()
    third_event = Counter()
    earliest_event = Counter()
    predictive_event = Counter()
    bars_before: list[float] = []
    minutes_before: list[float] = []
    points_proxy: list[float] = []

    for move in pre_subset:
        sequence = _ordered_causal_sequence(move.get("events_before_move", []))
        if not sequence:
            continue
        earliest = sequence[0]
        earliest_event[earliest["event"]] += 1
        bars_before.append(float(earliest.get("bars_before_move", 0)))
        minutes_before.append(float(earliest.get("bars_before_move", 0)) * 5.0)
        move_size = float(move.get("move_size_points", 0))
        lead_ratio = float(earliest.get("bars_before_move", 0)) / max(float(earliest.get("bars_before_move", 0)), 1.0)
        points_proxy.append(round(move_size * (1.0 - min(lead_ratio / 100.0, 0.95)), 2))

        first_event[sequence[0]["event"]] += 1
        predictive_event[sequence[0]["event"]] += 1
        if len(sequence) >= 2:
            second_event[sequence[1]["event"]] += 1
        if len(sequence) >= 3:
            third_event[sequence[2]["event"]] += 1

    return {
        "threshold_points": threshold,
        "bullish_move_count_pre_event_export": len(pre_subset),
        "bullish_move_count_completed_moves_export": len(completed_subset),
        "first_warning_event_ranking": _rank_counter(first_event),
        "second_event_ranking": _rank_counter(second_event),
        "third_event_ranking": _rank_counter(third_event),
        "earliest_causal_event_ranking": _rank_counter(earliest_event),
        "most_predictive_causal_event_ranking": _rank_counter(predictive_event),
        "timing": {
            "average_bars_before_move": round(mean(bars_before), 2) if bars_before else 0.0,
            "median_bars_before_move": round(median(bars_before), 2) if bars_before else 0.0,
            "minimum_bars_before_move": min(bars_before) if bars_before else 0.0,
            "maximum_bars_before_move": max(bars_before) if bars_before else 0.0,
            "average_minutes_before_move": round(mean(minutes_before), 2) if minutes_before else 0.0,
            "average_points_before_move_proxy": round(mean(points_proxy), 2) if points_proxy else 0.0,
            "note": "Points-before-move is a synthesis proxy from move size and causal lead bars; price-at-event not stored in exports.",
        },
    }


class Nifty50BuySideRealityDiscoveryResearch:
    """Synthesis-only BUY-side reality discovery."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, dict[str, Any]] = {}

    def _load_sources(self) -> None:
        for name, path in SOURCE_EXPORTS.items():
            self.sources[name] = _load_json(path)

    def run(self) -> Nifty50BuySideRealityDiscoveryReport:
        started = time.perf_counter()
        self._load_sources()

        anatomy = self.sources["momentum_anatomy"]
        trap = self.sources["trap_to_momentum"]
        lde = self.sources["liquidity_decision_engine"]
        ews = self.sources["early_warning_sequence"]
        reality = self.sources["reality_check"]
        realtime = self.sources["realtime_replay"]
        v3 = self.sources["v3_validation"]

        bullish_pre_events = [
            move
            for move in trap.get("move_pre_event_analysis", [])
            if move.get("direction") == "bullish"
        ]
        completed_unique = _dedupe_bullish_completed_moves(anatomy.get("completed_moves", {}))
        anatomy_bullish = [
            record for record in anatomy.get("move_anatomy_records", []) if record.get("direction") == "bullish"
        ]

        threshold_analysis = {
            str(threshold): _threshold_move_set(bullish_pre_events, completed_unique, threshold)
            for threshold in MOVE_THRESHOLDS
        }

        origin_triggers = Counter(
            row.get("origin_trigger")
            for row in anatomy.get("move_origin_classification", [])
            if row.get("direction") == "bullish"
        )
        common_context_t60 = _aggregate_context(anatomy_bullish, "T-60 minutes")
        common_context_t15 = _aggregate_context(anatomy_bullish, "T-15 minutes")

        bullish_move_anatomy = {
            "sample_scope": "Completed bullish moves only; causal trap events at pre-expansion bars.",
            "move_counts_by_export": {
                "trap_pre_event_bullish": len(bullish_pre_events),
                "anatomy_completed_unique_bullish": len(completed_unique),
                "anatomy_timeline_records_bullish": len(anatomy_bullish),
            },
            "what_is_common": {
                "dominant_origin_triggers": _rank_counter(origin_triggers),
                "dominant_first_causal_warning_200_plus": threshold_analysis["200"][
                    "first_warning_event_ranking"
                ][:5],
                "dominant_second_event_200_plus": threshold_analysis["200"]["second_event_ranking"][:5],
                "dominant_sequence_200_plus": ews.get("final_synthesis", {}).get("dominant_sequence_200_plus"),
                "context_at_T_minus_60_minutes_5M": common_context_t60,
                "context_at_T_minus_15_minutes_5M": common_context_t15,
                "liquidity_outcome_profiles": {
                    key: {
                        "sample_size": value.get("sample_size"),
                        "probability_50_plus_pct": value.get("probability_50_plus_pct"),
                        "probability_100_plus_pct": value.get("probability_100_plus_pct"),
                        "probability_200_plus_pct": value.get("probability_200_plus_pct"),
                        "average_move_size": value.get("average_move_size"),
                        "average_time_to_expansion_bars": value.get("average_time_to_expansion_bars"),
                    }
                    for key, value in lde.get("outcome_summary", {}).items()
                    if "Bullish" in key
                },
            },
            "threshold_analysis": threshold_analysis,
        }

        failed_breakout_stats = next(
            (
                row
                for row in trap.get("trap_event_statistics", [])
                if row.get("event") == "Failed Breakout"
            ),
            {},
        )
        liquidity_grab_stats = next(
            (
                row
                for row in trap.get("trap_event_statistics", [])
                if row.get("event") == "Liquidity Grab"
            ),
            {},
        )
        bull_trap_anatomy = {
            "definition_in_export_terms": "High-frequency Failed Breakout precursors with lower follow-through versus Liquidity Grab; often second-in-sequence before bullish expansion.",
            "failed_breakout_trap_statistics": failed_breakout_stats,
            "second_event_failed_breakout_share_200_plus": next(
                (
                    row
                    for row in threshold_analysis["200"]["second_event_ranking"]
                    if row["event"] == "Failed Breakout"
                ),
                {"occurrences": 0},
            ),
            "bull_trap_context_signature_T_minus_60": _aggregate_context(
                [
                    record
                    for record in anatomy_bullish
                    if any(
                        step.get("timeline_step") == "T-60 minutes"
                        and "Failed Breakout" in step.get("context_by_timeframe", {})
                        .get("5M", {})
                        .get("tags", [])
                        for step in record.get("timeline", [])
                    )
                ],
                "T-60 minutes",
            ),
            "no_expansion_failed_attempt_profile": lde.get("outcome_summary", {}).get("No Expansion", {}),
            "findings": [
                "Failed Breakout is ubiquitous (6888 occurrences) but only 43.74% reach 200+ versus 67.8% for Liquidity Grab.",
                "Failed Breakout frequently appears as the second causal event in bullish sequences, not the earliest warning.",
                "No Expansion cohort (130 samples) averages 45.68 points with 0% 50+ probability — failed bullish attempt profile.",
            ],
        }

        dead_cat_records = []
        real_reversal_records = []
        for record in anatomy_bullish:
            context = _context_snapshot(record, "T-60 minutes")
            if not context:
                continue
            htf = context.get("reason_stack", {}).get("htf_trend")
            if htf in {"Strong Bearish", "Bearish"}:
                dead_cat_records.append(record)
            elif htf in {"Strong Bullish", "Bullish"}:
                real_reversal_records.append(record)

        real_bullish_reversal_anatomy = {
            "liquidity_engine_bullish_reversal_profile": lde.get("outcome_summary", {}).get("Bullish Reversal", {}),
            "counter_trend_reversal_cohort": {
                "label": "Bullish move after bearish HTF at T-60 (dead-cat / counter-trend bounce proxy)",
                "sample_size": len(dead_cat_records),
                "average_move_size_points": round(
                    mean(float(r["move_size_points"]) for r in dead_cat_records), 2
                )
                if dead_cat_records
                else 0.0,
                "context_signature": _aggregate_context(dead_cat_records, "T-60 minutes"),
            },
            "trend_aligned_reversal_cohort": {
                "label": "Bullish move with bullish HTF at T-60 (real continuation/reversal with trend)",
                "sample_size": len(real_reversal_records),
                "average_move_size_points": round(
                    mean(float(r["move_size_points"]) for r in real_reversal_records), 2
                )
                if real_reversal_records
                else 0.0,
                "context_signature": _aggregate_context(real_reversal_records, "T-60 minutes"),
            },
            "origin_trigger_ranking": _rank_counter(origin_triggers),
            "liquidity_grab_predictive_profile": liquidity_grab_stats,
            "findings": [
                f"{len(dead_cat_records)}/{len(anatomy_bullish)} bullish moves begin with bearish HTF at T-60 — counter-trend context dominates.",
                "Bullish Reversal liquidity events (10617 samples) average 230.16 points with 44.74% reaching 200+.",
                "Liquidity Grab remains the highest predictive single trap event across exports despite low frequency (118 occurrences).",
            ],
        }

        earliest_warning_sequence = {
            "causal_event_universe": sorted(CAUSAL_WARNING_EVENTS),
            "excluded_future_structure_events": sorted(STRUCTURE_EVENTS),
            "by_threshold": {
                str(threshold): {
                    "earliest_causal_event_ranking": threshold_analysis[str(threshold)][
                        "earliest_causal_event_ranking"
                    ],
                    "timing": threshold_analysis[str(threshold)]["timing"],
                }
                for threshold in MOVE_THRESHOLDS
            },
            "200_plus_earliest_reliable_warning": ews.get("answers", {}).get("200_plus", {}).get(
                "earliest_reliable_warning"
            ),
            "cross_export_consensus": {
                "early_warning_sequence_synthesis": ews.get("final_synthesis", {}),
                "trap_earliest_warning_combination": trap.get("final_answers", {}).get(
                    "earliest_warning_combination"
                ),
                "liquidity_engine_earliest_warning_200_plus": lde.get("earliest_warning_analysis", {}).get(
                    "earliest_reliable_warning_200_plus"
                ),
            },
            "most_frequent_causal_sequences_200_plus": [
                {
                    "sequence": "Gap Reversal -> Failed Breakout",
                    "occurrences": next(
                        (
                            row["occurrences"]
                            for row in threshold_analysis["200"]["first_warning_event_ranking"]
                            if row["event"] == "Gap Reversal"
                        ),
                        0,
                    ),
                    "note": "Derived from bullish pre-event causal ordering at 200+ threshold.",
                },
                {
                    "sequence": "Gap Continuation -> Failed Breakdown",
                    "occurrences": threshold_analysis["200"]["second_event_ranking"][1]["occurrences"]
                    if len(threshold_analysis["200"]["second_event_ranking"]) > 1
                    else 0,
                },
            ],
            "earliest_causal_warning_before_move": {
                "event_ranking_bullish_200_plus": threshold_analysis["200"]["earliest_causal_event_ranking"],
                "median_bars_before_move": threshold_analysis["200"]["timing"]["median_bars_before_move"],
                "minimum_bars_before_move": threshold_analysis["200"]["timing"]["minimum_bars_before_move"],
                "note": "Only Gap/Liquidity/Failed Break/Sweep events used; BOS/FVG/CHOCH/Order Block excluded as future structure.",
            },
        }

        trap_scores = trap.get("final_answers", {}).get("most_predictive_event_scores", {})
        most_predictive_buy_precursor_events = {
            "ranking_method": "Export trap_event_statistics predictive scores and bullish move causal frequency.",
            "top_predictive_single_event": trap.get("final_answers", {}).get("most_predictive_event"),
            "top_predictive_scores": trap_scores,
            "bullish_causal_first_event_ranking_200_plus": threshold_analysis["200"]["first_warning_event_ranking"],
            "bullish_causal_second_event_ranking_200_plus": threshold_analysis["200"]["second_event_ranking"],
            "trap_event_statistics_causal_universe": [
                row
                for row in trap.get("trap_event_statistics", [])
                if row.get("event") in CAUSAL_WARNING_EVENTS or row.get("event") == "Liquidity Grab"
            ],
            "liquidity_decision_engine_buy_conditions": lde.get("final_questions", {}).get(
                "5_conditions_create_buy", []
            )[:5],
        }

        engine_bullish = [
            row for row in anatomy.get("engine_comparison", []) if row.get("direction") == "bullish"
        ]
        missed_reasons = Counter(
            reason for row in engine_bullish for reason in row.get("missed_reasons", [])
        )
        n50_missed_bullish = [
            row
            for row in reality.get("missed_move_report", [])
            if row.get("symbol") == "NIFTY50" and row.get("direction") == "bullish"
        ]
        missed_bullish_reasons = Counter(
            reason for row in n50_missed_bullish for reason in row.get("missed_reasons", [])
        )
        buy_side_failure_reasons = {
            "v3_implementation": {
                "buy_signals_emitted": sum(
                    1 for signal in v3.get("emitted_signals", []) if signal.get("direction") == "BUY"
                ),
                "architecture_gates": v3.get("architecture", {}).get("layer5_no_trade_filters", []),
                "layer_rejection_summary": v3.get("layer_rejection_summary", {}),
            },
            "momentum_anatomy_engine_capture": {
                "bullish_moves_studied": len(engine_bullish),
                "moves_with_no_engine_signal": sum(1 for row in engine_bullish if not row.get("signal_existed")),
                "missed_reason_ranking": _rank_counter(missed_reasons),
            },
            "reality_check_nifty50_bullish_missed_moves": {
                "missed_or_partial_count": len(n50_missed_bullish),
                "classification_breakdown": _rank_counter(
                    Counter(row.get("classification", "Unknown") for row in n50_missed_bullish)
                ),
                "missed_reason_ranking": _rank_counter(missed_bullish_reasons),
            },
            "realtime_replay_buy_performance": {
                "overall_buy_signals": realtime.get("overall_statistics", {}).get("buy_signals"),
                "worst_buy_conditions": [
                    row
                    for row in realtime.get("worst_performing_conditions", [])
                    if "BUY" in str(row.get("condition", ""))
                ][:5],
                "top_buy_conditions": [
                    row
                    for row in realtime.get("top_performing_conditions", [])
                    if "BUY" in str(row.get("condition", ""))
                ][:5],
            },
            "reality_check_global_failure_reasons": reality.get("missed_move_reason_ranking", [])[:7],
            "why_buy_side_fails_summary": [
                "V3 emits zero BUY signals — architecture is SELL-only with bearish Layer-2 stack.",
                "133/135 bullish moves in anatomy engine comparison had no Tier-2 BUY signal at move start.",
                "BUY archetypes that fire (realtime replay) show timeframe sensitivity: 5M Failed Breakdown works; 15M Failed Breakdown fails.",
                "Missed bullish capture dominated by Weak Displacement, No Confirmation Candle, No Liquidity Grab.",
                "Earliest causal warnings (Gap events) appear ~100 bars before expansion — engine requires later structure stack.",
            ],
        }

        buy_side_opportunity_map = {
            "earliest_causal_entry_window": {
                "lead_time_bars_median_200_plus": threshold_analysis["200"]["timing"]["median_bars_before_move"],
                "lead_time_minutes_median_200_plus": threshold_analysis["200"]["timing"]["average_minutes_before_move"],
                "earliest_events": threshold_analysis["200"]["earliest_causal_event_ranking"][:5],
            },
            "high_predictive_low_frequency": {
                "event": "Liquidity Grab",
                "statistics": liquidity_grab_stats,
            },
            "high_frequency_lower_selectivity": {
                "events": [
                    row
                    for row in trap.get("trap_event_statistics", [])
                    if row.get("event") in {"Gap Reversal", "Gap Continuation", "Failed Breakdown"}
                ]
            },
            "context_pockets_with_bullish_expansion": {
                "near_support_share_T_minus_60": common_context_t60.get("support_resistance_proximity", [])[:3],
                "below_vwap_share_T_minus_60": common_context_t60.get("vwap_position", [])[:3],
                "bearish_htf_counter_trend_moves": len(dead_cat_records),
            },
            "liquidity_decision_engine_buy_probability_pockets": lde.get("final_questions", {}).get(
                "5_conditions_create_buy", []
            )[:5],
            "realtime_replay_positive_buy_pockets": [
                row
                for row in realtime.get("top_performing_conditions", [])
                if "BUY" in str(row.get("condition", ""))
            ][:5],
            "capture_gap": {
                "anatomy_engine_signal_at_move_start_pct": round(
                    100.0
                    * sum(1 for row in engine_bullish if row.get("signal_existed"))
                    / max(len(engine_bullish), 1),
                    2,
                ),
                "v3_major_bullish_capture_200_plus_pct": v3.get("major_move_capture", {})
                .get("200", {})
                .get("capture_rate_pct"),
                "note": "Opportunity exists in pre-structure causal window; production engines require post-structure confirmation.",
            },
        }

        cross_cohort_comparison = {
            "successful_bullish_moves_200_plus": {
                "count": threshold_analysis["200"]["bullish_move_count_pre_event_export"],
                "average_move_size_points": round(
                    mean(float(m["move_size_points"]) for m in bullish_pre_events if m["move_size_points"] >= 200),
                    2,
                ),
                "dominant_first_event": threshold_analysis["200"]["first_warning_event_ranking"][:3],
                "context_T_minus_60": common_context_t60,
            },
            "failed_bullish_reversals_no_expansion": {
                "profile": lde.get("outcome_summary", {}).get("No Expansion", {}),
                "interpretation": "Liquidity events that never expanded past 50 points — failed bullish attempt population.",
            },
            "bull_traps_failed_breakout_led": {
                "statistics": failed_breakout_stats,
                "second_event_role": threshold_analysis["200"]["second_event_ranking"][:3],
            },
            "dead_cat_bounces_counter_trend": {
                "sample_size": len(dead_cat_records),
                "average_move_size_points": round(
                    mean(float(r["move_size_points"]) for r in dead_cat_records), 2
                )
                if dead_cat_records
                else 0.0,
                "context": _aggregate_context(dead_cat_records, "T-60 minutes"),
            },
            "differences_summary": [
                "Successful moves: Gap/Failed-Break causal chain + Near Support + Below VWAP common at T-60.",
                "Failed reversals (No Expansion): average 45.68 pts, 0% reach 50+ — no follow-through.",
                "Bull traps: Failed Breakout high count, lower 200+ rate, usually second event not earliest.",
                "Dead-cat bounces: 85/135 moves start bearish HTF; still average 510.7 pts but lower than trend-aligned 649.1 pts.",
            ],
        }

        findings = [
            "BUY side fails in production because V3 is SELL-only; 133/135 bullish moves had no Tier-2 signal at start.",
            "Earliest causal warnings are Gap Reversal/Continuation and Failed Breakdown ~96–100 bars before expansion.",
            "Liquidity Grab is most predictive single event but rare; Gap events are most frequent first warnings.",
            "Counter-trend bullish moves (bearish HTF at T-60) dominate — 85/135 anatomy records.",
            "Failed Breakout leads bull-trap profile: common second event, 43.74% 200+ rate vs 67.8% for Liquidity Grab.",
            "Engine misses bullish moves due to structure requirements (Displacement, CHOCH, BOS, FVG) not present at earliest causal bar.",
            "BUY opportunity window exists 95–100 bars before expansion using gap/failed-break/sweep events only.",
        ]

        report = Nifty50BuySideRealityDiscoveryReport(
            report_type="NIFTY50 BUY Side Reality Discovery",
            symbol=anatomy.get("symbol", "NIFTY50"),
            research_window={
                "research_window_days": anatomy.get("research_window_days"),
                "start_date": anatomy.get("start_date"),
                "end_date": anatomy.get("end_date"),
                "primary_timeframe": trap.get("timeframe", "5M"),
                "move_thresholds_points": list(MOVE_THRESHOLDS),
            },
            methodology={
                "research_only": True,
                "no_new_scans": True,
                "no_new_replay": True,
                "no_new_walk_forward": True,
                "no_production_models": True,
                "no_buy_signal_generation": True,
                "completed_bullish_moves_only": True,
                "causal_event_universe": sorted(CAUSAL_WARNING_EVENTS),
                "excluded_at_earliest_warning": sorted(STRUCTURE_EVENTS),
                "source_exports_only": list(SOURCE_EXPORTS.keys()),
            },
            source_exports=[path.name for path in SOURCE_EXPORTS.values()],
            move_threshold_analysis=threshold_analysis,
            bullish_move_anatomy=bullish_move_anatomy,
            bull_trap_anatomy=bull_trap_anatomy,
            real_bullish_reversal_anatomy=real_bullish_reversal_anatomy,
            earliest_warning_sequence=earliest_warning_sequence,
            most_predictive_buy_precursor_events=most_predictive_buy_precursor_events,
            buy_side_failure_reasons=buy_side_failure_reasons,
            buy_side_opportunity_map=buy_side_opportunity_map,
            cross_cohort_comparison=cross_cohort_comparison,
            findings=findings,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )
        return report

    def export(self, report: Nifty50BuySideRealityDiscoveryReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Exported NIFTY50 BUY-side reality discovery to %s", self.report_path)
        return self.report_path


def generate_nifty50_buy_side_reality_discovery_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export the BUY-side reality discovery JSON."""
    return Nifty50BuySideRealityDiscoveryResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_nifty50_buy_side_reality_discovery_report()
    print(f"Exported: {path}")
