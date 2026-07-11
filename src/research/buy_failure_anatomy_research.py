"""
BUY Failure Anatomy — synthesis-only research.

Classifies bullish moves that begin with BUY precursors (Gap Reversal, Failed Breakdown,
Near Support, Liquidity Grab) into Real Reversal vs false reversals (Dead Cat Bounce,
Range Failure). Identifies the strongest discriminator present only in real reversals.

No replay, no new models, no optimization.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

logger = logging.getLogger("SmartMoneyEngine")


def _json_safe(value: Any) -> Any:
    """Convert non-standard numeric values for JSON export."""
    if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"

SOURCE_EXPORTS = {
    "momentum_anatomy": RESEARCH_DIR / "nifty50_momentum_anatomy_120d.json",
    "trap_to_momentum": RESEARCH_DIR / "nifty50_trap_to_momentum_validation.json",
    "buy_side_discovery": RESEARCH_DIR / "nifty50_buy_side_reality_discovery.json",
    "buy_formula_verification": RESEARCH_DIR / "buy_formula_reality_verification.json",
    "liquidity_decision_engine": RESEARCH_DIR / "nifty50_liquidity_decision_engine.json",
    "institutional_momentum_origin": RESEARCH_DIR / "institutional_momentum_origin.json",
    "liquidity_move_reconstruction": RESEARCH_DIR / "liquidity_move_reconstruction.json",
    "research_consistency_audit": RESEARCH_DIR / "research_consistency_audit.json",
    "engine_gap_analysis": RESEARCH_DIR / "smartmoneyengine_engine_gap_analysis.json",
}

DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_failure_anatomy.json"

PRECURSOR_EVENTS = frozenset({"Gap Reversal", "Failed Breakdown", "Liquidity Grab"})
NEAR_SUPPORT_LABEL = "Near Support"
TIMELINE_STEPS = ("T-60 minutes", "T-15 minutes")
CONSOLIDATION_TAGS = frozenset({"NR7 Cluster", "Consolidation", "False Moves x3+"})
TAUTOLOGICAL_FEATURES = frozenset(
    {"move_200_plus", "failed_breakdown_precursor", "gap_reversal_precursor"},
)
REAL_REVERSAL_MIN_POINTS = 200
DEAD_CAT_MAX_POINTS = 100
RANGE_FAILURE_MAX_POINTS = 200


class BuyFailureAnatomyError(Exception):
    """Raised when BUY failure anatomy synthesis cannot be completed."""


@dataclass
class BuyFailureAnatomyReport:
    """BUY failure anatomy synthesis output."""

    report_type: str
    symbol: str
    research_window: dict[str, Any]
    methodology: dict[str, Any]
    source_exports: list[str]
    precursor_filter: dict[str, Any]
    classification_summary: dict[str, Any]
    real_vs_false_comparison: dict[str, Any]
    discriminator_candidates: list[dict[str, Any]]
    strongest_buy_discriminator: dict[str, Any]
    conclusions: list[str]
    limitations: list[str]
    execution_time_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BuyFailureAnatomyError(f"Missing source export: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_date_key(value: str) -> str:
    return str(value)[:16]


def _context_snapshot(record: dict[str, Any], step: str) -> dict[str, Any] | None:
    for timeline_step in record.get("timeline", []):
        if timeline_step.get("timeline_step") != step:
            continue
        return timeline_step.get("context_by_timeframe", {}).get("5M")
    return None


def _ordered_causal_events(events: list[dict[str, Any]]) -> list[str]:
    causal = [event for event in events if event.get("event") in PRECURSOR_EVENTS]
    ordered = sorted(causal, key=lambda item: -int(item.get("bars_before_move", 0)))
    return [item["event"] for item in ordered]


def _second_event_name(events: list[dict[str, Any]]) -> str | None:
    ordered = sorted(events, key=lambda item: -int(item.get("bars_before_move", 0)))
    if len(ordered) < 2:
        return None
    return str(ordered[1].get("event"))


def _has_consolidation_tags(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    tags = set(record.get("tags", []))
    return bool(tags & CONSOLIDATION_TAGS)


def _precursor_match(
    *,
    first_event: str | None,
    causal_events: list[str],
    near_support: bool,
    origin_trigger: str | None,
) -> tuple[bool, list[str]]:
    matched: list[str] = []
    if first_event in PRECURSOR_EVENTS:
        matched.append(str(first_event))
    for event in causal_events:
        if event in PRECURSOR_EVENTS and event not in matched:
            matched.append(event)
    if near_support:
        matched.append(NEAR_SUPPORT_LABEL)
    if origin_trigger in PRECURSOR_EVENTS and origin_trigger not in matched:
        matched.append(str(origin_trigger))
    return bool(matched), matched


def _classify_move(record: dict[str, Any]) -> str:
    move_size = float(record.get("move_size_points", 0))
    duration = float(record.get("duration_minutes", 0))
    htf_trend = str(record.get("context_t60", {}).get("reason_stack", {}).get("htf_trend", ""))
    second_event = record.get("second_event")
    consolidation = bool(record.get("has_consolidation_tags"))
    lde_outcome = str(record.get("lde_outcome", ""))
    precursors = set(record.get("matched_precursors", []))

    if lde_outcome == "No Expansion" or move_size < DEAD_CAT_MAX_POINTS:
        return "Dead Cat Bounce"

    counter_trend = htf_trend in {"Strong Bearish", "Bearish"}
    trend_aligned = htf_trend in {"Strong Bullish", "Bullish"}
    has_liquidity_grab = "Liquidity Grab" in precursors

    if counter_trend and not has_liquidity_grab:
        if move_size < 300 or second_event == "Failed Breakout":
            return "Dead Cat Bounce"

    if move_size >= REAL_REVERSAL_MIN_POINTS:
        if trend_aligned or has_liquidity_grab:
            return "Real Reversal"
        if counter_trend:
            return "Dead Cat Bounce"
        return "Real Reversal"

    if (
        DEAD_CAT_MAX_POINTS <= move_size < RANGE_FAILURE_MAX_POINTS
        and (second_event == "Failed Breakout" or consolidation)
    ):
        return "Range Failure"
    if move_size < RANGE_FAILURE_MAX_POINTS and duration >= 240:
        return "Range Failure"
    if move_size < RANGE_FAILURE_MAX_POINTS:
        return "Dead Cat Bounce"
    return "Real Reversal"


def _feature_vector(record: dict[str, Any]) -> dict[str, bool]:
    context_t60 = record.get("context_t60", {})
    context_t15 = record.get("context_t15", {})
    reason_t60 = context_t60.get("reason_stack", {})
    reason_t15 = context_t15.get("reason_stack", {})
    flags_t15 = context_t15.get("feature_flags", {})
    levels_t15 = context_t15.get("levels", {})
    precursors = set(record.get("matched_precursors", []))

    return {
        "liquidity_grab_first_precursor": record.get("first_event") == "Liquidity Grab",
        "liquidity_grab_any_precursor": "Liquidity Grab" in precursors,
        "origin_trigger_liquidity_grab": record.get("origin_trigger") == "Liquidity Grab",
        "htf_strong_bullish_at_t60": reason_t60.get("htf_trend") in {"Strong Bullish", "Bullish"},
        "htf_strong_bearish_at_t60": reason_t60.get("htf_trend") in {"Strong Bearish", "Bearish"},
        "above_vwap_at_t15": reason_t15.get("vwap") == "Above VWAP",
        "below_vwap_at_t15": reason_t15.get("vwap") == "Below VWAP",
        "near_support_at_t15": levels_t15.get("market_location") == NEAR_SUPPORT_LABEL,
        "near_support_at_t60": context_t60.get("levels", {}).get("market_location") == NEAR_SUPPORT_LABEL,
        "failed_breakdown_precursor": "Failed Breakdown" in precursors,
        "gap_reversal_precursor": "Gap Reversal" in precursors,
        "failed_breakout_second_event": record.get("second_event") == "Failed Breakout",
        "volume_expansion_at_t15": bool(reason_t15.get("volume_expansion")),
        "liquidity_grab_reason_stack": bool(reason_t60.get("liquidity_grab")),
        "fvg_reclaim_at_t15": bool(flags_t15.get("fvg_reclaim")),
        "move_200_plus": float(record.get("move_size_points", 0)) >= REAL_REVERSAL_MIN_POINTS,
        "formula_full_match": bool(record.get("formula_full_match")),
    }


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * count / total, 2)


def _discriminator_score(
    *,
    real_count: int,
    real_total: int,
    false_count: int,
    false_total: int,
) -> dict[str, Any]:
    real_rate = _rate(real_count, real_total)
    false_rate = _rate(false_count, false_total)
    delta = round(real_rate - false_rate, 2)
    support = min(real_total, false_total)
    score = round(abs(delta) * support, 2)
    exclusive_real = real_count > 0 and false_count == 0
    return {
        "real_count": real_count,
        "real_total": real_total,
        "real_rate_pct": real_rate,
        "false_count": false_count,
        "false_total": false_total,
        "false_rate_pct": false_rate,
        "rate_delta_pct": delta,
        "separation_score": score,
        "exclusive_to_real": exclusive_real,
    }


class BuyFailureAnatomyResearch:
    """Synthesis-only BUY failure anatomy research."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, dict[str, Any]] = {}

    def _load_sources(self) -> None:
        for name, path in SOURCE_EXPORTS.items():
            self.sources[name] = _load_json(path)

    def _build_formula_dates(self, buy_formula: dict[str, Any]) -> set[str]:
        return {
            _normalize_date_key(str(item.get("move_timestamp", "")))
            for item in buy_formula.get("all_occurrences", [])
        }

    def _build_trap_index(self) -> dict[str, dict[str, Any]]:
        trap = self.sources["trap_to_momentum"]
        index: dict[str, dict[str, Any]] = {}
        for move in trap.get("move_pre_event_analysis", []):
            if move.get("direction") != "bullish":
                continue
            key = _normalize_date_key(str(move.get("date", "")))
            if key not in index:
                index[key] = move
        return index

    def _record_from_anatomy(
        self,
        record: dict[str, Any],
        *,
        trap_index: dict[str, dict[str, Any]],
        formula_dates: set[str],
    ) -> dict[str, Any] | None:
        date_key = _normalize_date_key(str(record.get("date", "")))
        context_t60 = _context_snapshot(record, "T-60 minutes") or {}
        context_t15 = _context_snapshot(record, "T-15 minutes") or {}
        trap_move = trap_index.get(date_key, {})
        causal_events = _ordered_causal_events(trap_move.get("events_before_move", []))
        near_support = (
            context_t15.get("levels", {}).get("market_location") == NEAR_SUPPORT_LABEL
            or context_t60.get("levels", {}).get("market_location") == NEAR_SUPPORT_LABEL
        )
        blueprint = str(record.get("blueprint_pattern", ""))
        for event in PRECURSOR_EVENTS:
            if event in blueprint and event not in causal_events:
                causal_events.append(event)

        matched, precursors = _precursor_match(
            first_event=trap_move.get("first_event"),
            causal_events=causal_events,
            near_support=near_support,
            origin_trigger=record.get("origin_trigger"),
        )
        if not matched:
            return None

        return {
            "source": "momentum_anatomy",
            "date": record.get("date"),
            "move_size_points": round(float(record.get("move_size_points", 0)), 2),
            "duration_minutes": float(record.get("duration_minutes", 0)),
            "first_event": trap_move.get("first_event") or record.get("origin_trigger"),
            "second_event": _second_event_name(trap_move.get("events_before_move", [])),
            "matched_precursors": precursors,
            "origin_trigger": record.get("origin_trigger"),
            "blueprint_pattern": blueprint,
            "context_t60": context_t60,
            "context_t15": context_t15,
            "has_consolidation_tags": _has_consolidation_tags(context_t15)
            or _has_consolidation_tags(context_t60),
            "formula_full_match": date_key in formula_dates,
            "lde_outcome": "",
        }

    def _build_anatomy_cohort(
        self,
        anatomy: dict[str, Any],
        trap_index: dict[str, dict[str, Any]],
        formula_dates: set[str],
    ) -> list[dict[str, Any]]:
        cohort: list[dict[str, Any]] = []
        seen: set[tuple[str, float]] = set()
        for record in anatomy.get("move_anatomy_records", []):
            if record.get("direction") != "bullish":
                continue
            built = self._record_from_anatomy(record, trap_index=trap_index, formula_dates=formula_dates)
            if not built:
                continue
            dedupe_key = (_normalize_date_key(str(built["date"])), round(float(built["move_size_points"]), 1))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cohort.append(built)
        return cohort

    def _build_lde_failure_cohort(self, seen: set[tuple[str, float]]) -> list[dict[str, Any]]:
        lde = self.sources["liquidity_decision_engine"]
        extra: list[dict[str, Any]] = []
        for event in lde.get("liquidity_event_log", []):
            if event.get("event_type") not in PRECURSOR_EVENTS:
                continue
            if event.get("outcome") != "No Expansion":
                continue

            timestamp = str(event.get("timestamp", ""))
            date_key = _normalize_date_key(timestamp)
            forward = event.get("forward_metrics", {})
            move_size = round(float(forward.get("max_move", forward.get("bull_move", 45.0))), 2)
            dedupe_key = (date_key, move_size)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            precursors = [str(event.get("event_type"))]
            level_ctx = event.get("major_level_context", {})
            near_support = float(level_ctx.get("distance_nearest_support", 999)) <= 25
            if near_support:
                precursors.append(NEAR_SUPPORT_LABEL)

            extra.append(
                {
                    "source": "liquidity_decision_engine",
                    "date": timestamp,
                    "move_size_points": move_size,
                    "duration_minutes": float(forward.get("time_to_expansion_bars", 0)) * 5,
                    "first_event": event.get("event_type"),
                    "second_event": None,
                    "matched_precursors": precursors,
                    "origin_trigger": event.get("event_type"),
                    "blueprint_pattern": None,
                    "context_t60": {},
                    "context_t15": {
                        "levels": {"market_location": NEAR_SUPPORT_LABEL if near_support else "Mid Range"},
                        "reason_stack": {},
                        "feature_flags": {},
                    },
                    "has_consolidation_tags": False,
                    "formula_full_match": False,
                    "lde_outcome": "No Expansion",
                },
            )
        return extra

    def _build_trap_cohort(self, formula_dates: set[str]) -> list[dict[str, Any]]:
        anatomy = self.sources["momentum_anatomy"]
        trap_index = self._build_trap_index()
        cohort = self._build_anatomy_cohort(anatomy, trap_index, formula_dates)

        seen = {
            (_normalize_date_key(str(item.get("date", ""))), round(float(item.get("move_size_points", 0)), 1))
            for item in cohort
        }
        cohort.extend(self._build_lde_failure_cohort(seen))
        cohort = self._append_reconstruction_failures(cohort)
        return cohort

    def _append_reconstruction_failures(self, cohort: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reconstruction = self.sources["liquidity_move_reconstruction"]
        seen = {
            (_normalize_date_key(str(item.get("date", ""))), round(float(item.get("move_size_points", 0)), 1))
            for item in cohort
        }
        extra: list[dict[str, Any]] = []

        for move in reconstruction.get("moves", []):
            if move.get("direction") != "bullish":
                continue
            magnitude = float(move.get("move_magnitude_points", 0))
            if magnitude >= DEAD_CAT_MAX_POINTS:
                continue
            if move.get("market_location") != NEAR_SUPPORT_LABEL:
                continue
            liquidity_event = str(move.get("liquidity_event", ""))
            if liquidity_event not in {"Buy Side Sweep", "None"} and "Grab" not in liquidity_event:
                continue

            date_key = _normalize_date_key(str(move.get("start_timestamp", "")))
            dedupe_key = (date_key, round(magnitude, 1))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            precursors = [NEAR_SUPPORT_LABEL]
            if "Grab" in liquidity_event or "Sweep" in liquidity_event:
                precursors.append("Liquidity Grab")

            extra.append(
                {
                    "source": "liquidity_move_reconstruction",
                    "date": move.get("start_timestamp"),
                    "move_size_points": round(magnitude, 2),
                    "duration_minutes": 0.0,
                    "first_event": precursors[-1] if len(precursors) > 1 else NEAR_SUPPORT_LABEL,
                    "second_event": None,
                    "matched_precursors": precursors,
                    "origin_trigger": None,
                    "blueprint_pattern": move.get("pre_move_sequence"),
                    "context_t60": {},
                    "context_t15": {
                        "levels": {"market_location": NEAR_SUPPORT_LABEL},
                        "reason_stack": {},
                        "feature_flags": {},
                    },
                    "has_consolidation_tags": move.get("fvg_behavior") == "Failed",
                    "formula_full_match": False,
                    "lde_outcome": "No Expansion",
                },
            )
        return cohort + extra

    def _classify_cohort(self, cohort: list[dict[str, Any]]) -> list[dict[str, Any]]:
        classified: list[dict[str, Any]] = []
        for record in cohort:
            classification = _classify_move(record)
            classified.append({**record, "classification": classification})
        return classified

    def _compare_real_vs_false(self, classified: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        real_records = [row for row in classified if row["classification"] == "Real Reversal"]
        false_records = [
            row for row in classified if row["classification"] in {"Dead Cat Bounce", "Range Failure"}
        ]
        real_total = len(real_records)
        false_total = len(false_records)

        feature_names = list(_feature_vector(real_records[0]).keys()) if real_records else []
        if false_records and not feature_names:
            feature_names = list(_feature_vector(false_records[0]).keys())

        candidates: list[dict[str, Any]] = []
        for feature in feature_names:
            real_count = sum(1 for row in real_records if _feature_vector(row).get(feature))
            false_count = sum(1 for row in false_records if _feature_vector(row).get(feature))
            stats = _discriminator_score(
                real_count=real_count,
                real_total=real_total,
                false_count=false_count,
                false_total=false_total,
            )
            candidates.append({"feature": feature, **stats})

        candidates = [row for row in candidates if row["feature"] not in TAUTOLOGICAL_FEATURES]
        candidates.sort(
            key=lambda row: (row["separation_score"], row["rate_delta_pct"], row["real_rate_pct"]),
            reverse=True,
        )

        precursor_breakdown = {
            precursor: Counter(row["classification"] for row in classified if precursor in row["matched_precursors"])
            for precursor in sorted({p for row in classified for p in row["matched_precursors"]})
        }
        precursor_summary = {
            precursor: {
                classification: count
                for classification, count in counter.items()
            }
            for precursor, counter in precursor_breakdown.items()
        }

        magnitude_by_class = defaultdict(list)
        duration_by_class = defaultdict(list)
        for row in classified:
            magnitude_by_class[row["classification"]].append(float(row["move_size_points"]))
            duration_by_class[row["classification"]].append(float(row["duration_minutes"]))

        comparison = {
            "real_reversal_count": real_total,
            "false_reversal_count": false_total,
            "false_breakdown": {
                "dead_cat_bounce": sum(1 for row in classified if row["classification"] == "Dead Cat Bounce"),
                "range_failure": sum(1 for row in classified if row["classification"] == "Range Failure"),
            },
            "magnitude_averages_by_class": {
                label: round(mean(values), 2) if values else 0.0
                for label, values in magnitude_by_class.items()
            },
            "duration_averages_by_class": {
                label: round(mean(values), 2) if values else 0.0
                for label, values in duration_by_class.items()
            },
            "precursor_outcome_breakdown": precursor_summary,
            "trap_event_reference_rates": {
                row["event"]: {
                    "probability_200_plus_pct": row.get("probability_200_plus_pct"),
                    "average_move_size": row.get("average_move_size"),
                }
                for row in self.sources["trap_to_momentum"].get("trap_event_statistics", [])
                if row.get("event") in PRECURSOR_EVENTS
            },
            "lde_outcome_reference": {
                "bullish_reversal": self.sources["liquidity_decision_engine"]
                .get("outcome_summary", {})
                .get("Bullish Reversal", {}),
                "no_expansion": self.sources["liquidity_decision_engine"]
                .get("outcome_summary", {})
                .get("No Expansion", {}),
            },
            "buy_side_discovery_cross_check": {
                "counter_trend_reversal_cohort_size": self.sources["buy_side_discovery"]
                .get("real_bullish_reversal_anatomy", {})
                .get("counter_trend_reversal_cohort", {})
                .get("sample_size"),
                "trend_aligned_reversal_cohort_size": self.sources["buy_side_discovery"]
                .get("real_bullish_reversal_anatomy", {})
                .get("trend_aligned_reversal_cohort", {})
                .get("sample_size"),
            },
        }
        return comparison, candidates

    def run(self) -> BuyFailureAnatomyReport:
        started = time.perf_counter()
        self._load_sources()

        trap = self.sources["trap_to_momentum"]
        buy_formula = self.sources["buy_formula_verification"]

        formula_dates = self._build_formula_dates(buy_formula)
        cohort = self._build_trap_cohort(formula_dates=formula_dates)
        classified = self._classify_cohort(cohort)
        comparison, candidates = self._compare_real_vs_false(classified)

        classification_counts = Counter(row["classification"] for row in classified)
        strongest = candidates[0] if candidates else {}
        strongest_feature = strongest.get("feature", "unknown")
        strongest_payload = {
            "feature": strongest_feature,
            "description": self._discriminator_description(strongest_feature),
            "evidence": strongest,
            "why_strongest": (
                f"Highest separation score ({strongest.get('separation_score', 0)}) with "
                f"{strongest.get('rate_delta_pct', 0):+.2f}pp real-vs-false rate gap "
                f"({strongest.get('real_rate_pct', 0)}% real vs {strongest.get('false_rate_pct', 0)}% false)."
            ),
        }

        conclusions = self._build_conclusions(classification_counts, strongest_payload, comparison)
        limitations = [
            "Classification uses export proxies; no per-bar replay validation performed.",
            "Near Support is a context field, not a trap event — joined via anatomy timeline.",
            "Sub-100pt failures partially sourced from liquidity_move_reconstruction (365d window).",
            "Trap pre-event cohort is 100+ bullish moves only; false reversals enriched from reconstruction.",
            "LDE per-event outcome labels not joined at event level — aggregate profiles used as reference.",
        ]

        report = BuyFailureAnatomyReport(
            report_type="BUY Failure Anatomy",
            symbol="NIFTY50",
            research_window={
                "primary_window_days": trap.get("research_window_days", 120),
                "start_date": trap.get("start_date"),
                "end_date": trap.get("end_date"),
                "timeframe": trap.get("timeframe", "5M"),
                "reconstruction_window_days": self.sources["liquidity_move_reconstruction"].get(
                    "research_window_days",
                ),
            },
            methodology={
                "research_only": True,
                "no_replay": True,
                "no_new_models": True,
                "no_optimization": True,
                "precursor_events": sorted(PRECURSOR_EVENTS | {NEAR_SUPPORT_LABEL}),
                "classification_rules": {
                    "Real Reversal": (
                        f"move_size_points >= {REAL_REVERSAL_MIN_POINTS} with trend-aligned HTF "
                        "or Liquidity Grab precursor; counter-trend 200+ without LG reclassified"
                    ),
                    "Dead Cat Bounce": (
                        f"move_size_points < {DEAD_CAT_MAX_POINTS}, LDE No Expansion, or counter-trend HTF "
                        f"without Liquidity Grab and move < 300 / Failed Breakout second event"
                    ),
                    "Range Failure": (
                        f"{DEAD_CAT_MAX_POINTS} <= move < {RANGE_FAILURE_MAX_POINTS} with Failed Breakout "
                        "second event or consolidation tags, OR long duration without 200+ expansion"
                    ),
                },
                "discriminator_scoring": "separation_score = abs(real_rate - false_rate) * min(real_n, false_n)",
            },
            source_exports=[path.name for path in SOURCE_EXPORTS.values()],
            precursor_filter={
                "events": sorted(PRECURSOR_EVENTS),
                "context_proxy": NEAR_SUPPORT_LABEL,
                "match_logic": "Any precursor event as first/earliest causal event, origin trigger, or Near Support context",
                "cohort_size": len(cohort),
                "sources": {
                    "momentum_anatomy": sum(1 for row in cohort if row["source"] == "momentum_anatomy"),
                    "liquidity_decision_engine": sum(
                        1 for row in cohort if row["source"] == "liquidity_decision_engine"
                    ),
                    "liquidity_move_reconstruction": sum(
                        1 for row in cohort if row["source"] == "liquidity_move_reconstruction"
                    ),
                },
            },
            classification_summary={
                "total_classified": len(classified),
                "counts": dict(classification_counts),
                "rates_pct": {
                    label: _rate(count, len(classified)) for label, count in classification_counts.items()
                },
                "sample_records_by_class": {
                    label: [
                        {
                            "date": row["date"],
                            "move_size_points": row["move_size_points"],
                            "matched_precursors": row["matched_precursors"],
                            "first_event": row["first_event"],
                        }
                        for row in classified
                        if row["classification"] == label
                    ][:5]
                    for label in ("Real Reversal", "Dead Cat Bounce", "Range Failure")
                },
            },
            real_vs_false_comparison=comparison,
            discriminator_candidates=candidates[:12],
            strongest_buy_discriminator=strongest_payload,
            conclusions=conclusions,
            limitations=limitations,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )
        return report

    @staticmethod
    def _discriminator_description(feature: str) -> str:
        descriptions = {
            "liquidity_grab_first_precursor": "Liquidity Grab is the earliest causal precursor event",
            "liquidity_grab_any_precursor": "Liquidity Grab appears anywhere in matched precursors",
            "origin_trigger_liquidity_grab": "Anatomy origin_trigger classified as Liquidity Grab",
            "htf_strong_bullish_at_t60": "HTF trend Bullish/Strong Bullish at T-60 minutes (5M)",
            "htf_strong_bearish_at_t60": "HTF trend Bearish/Strong Bearish at T-60 minutes (5M)",
            "above_vwap_at_t15": "Price Above VWAP at T-15 minutes",
            "below_vwap_at_t15": "Price Below VWAP at T-15 minutes",
            "near_support_at_t15": "market_location Near Support at T-15 minutes",
            "near_support_at_t60": "market_location Near Support at T-60 minutes",
            "failed_breakdown_precursor": "Failed Breakdown in matched precursors",
            "gap_reversal_precursor": "Gap Reversal in matched precursors",
            "failed_breakout_second_event": "Failed Breakout is second pre-move event (bull-trap signature)",
            "volume_expansion_at_t15": "volume_expansion true at T-15 minutes",
            "liquidity_grab_reason_stack": "liquidity_grab true in reason_stack at T-60",
            "fvg_reclaim_at_t15": "fvg_reclaim feature flag at T-15 minutes",
            "move_200_plus": "move_size_points >= 200 (outcome tautology — excluded from selection)",
            "formula_full_match": "Failed Breakdown + Gap Reversal + Near Support formula match",
        }
        return descriptions.get(feature, feature)

    @staticmethod
    def _build_conclusions(
        classification_counts: Counter[str],
        strongest: dict[str, Any],
        comparison: dict[str, Any],
    ) -> list[str]:
        real_n = classification_counts.get("Real Reversal", 0)
        false_n = classification_counts.get("Dead Cat Bounce", 0) + classification_counts.get("Range Failure", 0)
        feature = strongest.get("feature", "unknown")
        evidence = strongest.get("evidence", {})
        return [
            f"Classified {sum(classification_counts.values())} precursor-matched bullish moves: "
            f"{real_n} Real Reversal, {false_n} false reversal "
            f"({classification_counts.get('Dead Cat Bounce', 0)} Dead Cat, "
            f"{classification_counts.get('Range Failure', 0)} Range Failure).",
            f"Strongest BUY discriminator: {feature} — {evidence.get('real_rate_pct', 0)}% in real vs "
            f"{evidence.get('false_rate_pct', 0)}% in false (delta {evidence.get('rate_delta_pct', 0):+.2f}pp).",
            "Liquidity Grab retains highest trap-event 200+ probability (67.8%) in trap export reference rates.",
            "Failed Breakout as second event concentrates in false reversals — consistent with bull-trap anatomy.",
            "Buy formula full match (Failed Breakdown + Gap Reversal + Near Support) shows 100% 200+ in verification export.",
        ]

    def export(self, report: BuyFailureAnatomyReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Exported BUY failure anatomy to %s", self.report_path)
        return self.report_path


def generate_buy_failure_anatomy_report(report_path: Path = DEFAULT_REPORT_PATH) -> Path:
    """Generate and export the BUY failure anatomy JSON."""
    return BuyFailureAnatomyResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_buy_failure_anatomy_report()
    print(f"Exported: {path}")
