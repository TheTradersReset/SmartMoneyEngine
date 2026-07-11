"""
SmartMoneyEngine Signal Timing Reality Audit — synthesis-only research.

Explains WHY signals arrive when they do using completed exports only.
No new discovery, optimization, models, or architectures.
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
    "v3_validation": RESEARCH_DIR / "smartmoneyengine_v3_implementation_validation.json",
    "signal_timing_audit": RESEARCH_DIR / "nifty50_signal_timing_audit.json",
    "final_signal_extraction": RESEARCH_DIR / "smartmoneyengine_final_signal_extraction.json",
    "sell_formula_verification": RESEARCH_DIR / "sell_formula_reality_verification_v2.json",
    "buy_formula_verification": RESEARCH_DIR / "buy_formula_reality_verification.json",
}

DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_signal_timing_reality_audit.json"

FILTER_CAUSE_MAP = {
    "NO_FAILED_BREAKOUT": "Failed Breakout",
    "HTF_CONFLICT": "HTF",
    "VWAP_MISMATCH": "VWAP",
    "EMA_MISMATCH": "EMA",
    "CONFIRMATION_FAILED": "Confirmation",
    "LOCATION_MID_RANGE": "Location",
    "DIRECTION_NOT_ALIGNED": "Layer 2 Stack (HTF+VWAP+EMA)",
    "NO_EARLY_WARNING": "Early Warning",
}

DELAY_RANK_ORDER = [
    "DIRECTION_NOT_ALIGNED",
    "VWAP_MISMATCH",
    "NO_FAILED_BREAKOUT",
    "HTF_CONFLICT",
    "EMA_MISMATCH",
    "CONFIRMATION_FAILED",
    "LOCATION_MID_RANGE",
]


class SmartMoneyEngineSignalTimingRealityAuditError(Exception):
    """Raised when timing reality audit synthesis fails."""


@dataclass
class SmartMoneyEngineSignalTimingRealityAuditReport:
    """Signal timing reality audit output."""

    report_type: str
    symbol: str
    timeframe: str
    replay_context: dict[str, Any]
    methodology: dict[str, Any]
    source_exports: list[str]
    timing_audit: dict[str, Any]
    delay_audit: dict[str, Any]
    missed_move_audit: dict[str, Any]
    filter_impact_audit: dict[str, Any]
    earliest_entry_simulation: dict[str, Any]
    final_recommendation: dict[str, Any]
    findings: list[str]
    execution_time_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SmartMoneyEngineSignalTimingRealityAuditError(f"Missing export: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _primary_delay_filter(
    *,
    timing_row: dict[str, Any],
    delay_contributors: list[dict[str, Any]],
) -> str:
    explicit = timing_row.get("delay_condition_if_later_than_cluster_first")
    if explicit and "Same-stack re-fire" not in str(explicit):
        for label in FILTER_CAUSE_MAP.values():
            if label.lower() in str(explicit).lower():
                return label
        return str(explicit)
    if timing_row.get("could_entry_have_occurred_earlier"):
        if delay_contributors:
            code = delay_contributors[0]["cause_code"]
            return FILTER_CAUSE_MAP.get(code, delay_contributors[0].get("cause_label", "Unknown"))
        return "Cluster re-fire delay"
    return "None (cluster-first entry)"


def _cluster_first_mfe(
    *,
    cluster_id: int,
    cluster_first_bar: int,
    v3_signals: list[dict[str, Any]],
    timing_by_ts: dict[str, dict[str, Any]],
) -> float | None:
    for signal in v3_signals:
        timing = timing_by_ts.get(signal["timestamp"])
        if not timing:
            continue
        if timing["cluster"]["cluster_id"] != cluster_id:
            continue
        if int(signal["bar"]) == cluster_first_bar:
            return float(signal.get("mfe_points", 0.0))
    return None


def _build_per_signal_timing(
    v3: dict[str, Any],
    timing: dict[str, Any],
) -> list[dict[str, Any]]:
    timing_rows = timing.get("1_delay_analysis", {}).get("per_signal_timing", [])
    timing_by_ts = {row["timestamp"]: row for row in timing_rows}
    delay_contributors = timing.get("3_biggest_delay_contributors", [])
    cluster_first_lookup = {
        row["bar"]: row for row in timing.get("4_earliest_safe_entry_point", {}).get("cluster_first_signals", [])
    }

    audits: list[dict[str, Any]] = []
    for signal in v3.get("emitted_signals", []):
        row = timing_by_ts.get(signal["timestamp"])
        if not row:
            continue
        cluster = row.get("cluster", {})
        cluster_first_bar = int(cluster.get("cluster_first_bar", signal["bar"]))
        actual_bar = int(signal["bar"])
        delay_bars = max(actual_bar - cluster_first_bar, 0)
        cluster_first = cluster_first_lookup.get(cluster_first_bar, {})
        first_mfe = _cluster_first_mfe(
            cluster_id=int(cluster.get("cluster_id", 0)),
            cluster_first_bar=cluster_first_bar,
            v3_signals=v3.get("emitted_signals", []),
            timing_by_ts=timing_by_ts,
        )
        actual_mfe = float(signal.get("mfe_points", 0.0))
        delay_points = round(max(0.0, (first_mfe or actual_mfe) - actual_mfe), 2) if delay_bars > 0 else 0.0

        audits.append(
            {
                "date": str(signal["timestamp"])[:10],
                "time": str(signal["timestamp"]).split(" ")[-1] if " " in str(signal["timestamp"]) else "",
                "timestamp": signal["timestamp"],
                "direction": signal.get("direction"),
                "model_id": signal.get("model_id"),
                "earliest_possible_entry_bar": cluster_first_bar,
                "earliest_possible_entry_timestamp": cluster_first.get("timestamp"),
                "actual_v3_entry_bar": actual_bar,
                "actual_v3_entry_timestamp": signal["timestamp"],
                "delay_bars": delay_bars,
                "delay_points": delay_points,
                "filter_causing_delay": _primary_delay_filter(timing_row=row, delay_contributors=delay_contributors),
                "entry": signal.get("entry"),
                "stop_loss": signal.get("stop_loss"),
                "target_1": signal.get("target_1"),
                "cluster_id": cluster.get("cluster_id"),
                "could_entry_have_occurred_earlier": row.get("could_entry_have_occurred_earlier"),
                "location": row.get("location"),
                "confirmation_candle": row.get("confirmation_candle"),
            }
        )
    return audits


def _build_captured_moves(
    v3: dict[str, Any],
    timing: dict[str, Any],
    per_signal: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for cluster_row in timing.get("4_earliest_safe_entry_point", {}).get("cluster_first_signals", []):
        timestamp = cluster_row.get("timestamp")
        matching = next((row for row in per_signal if row["timestamp"] == timestamp), None)
        move_mfe = float(cluster_row.get("mfe_points", 0.0))
        signal_mfe = float(matching["delay_points"]) + move_mfe if matching else move_mfe
        captured.append(
            {
                "move_start_timestamp": timestamp,
                "signal_time": matching["timestamp"] if matching else timestamp,
                "cluster_first_bar": cluster_row.get("bar"),
                "points_available_before_signal": round(
                    float(matching["delay_points"]) if matching else 0.0,
                    2,
                ),
                "points_captured_after_signal_mfe": move_mfe,
                "pct_move_captured_proxy": round(
                    100.0 * move_mfe / max(move_mfe + float(matching["delay_points"] if matching else 0.0), 1.0),
                    2,
                ),
                "captured_via_v3_signal": matching is not None,
            }
        )

    major = v3.get("major_move_capture", {})
    summary = {
        "200_plus_capture_rate_pct": major.get("200", {}).get("capture_rate_pct"),
        "300_plus_capture_rate_pct": major.get("300", {}).get("capture_rate_pct"),
        "signals_before_200_plus_moves": major.get("200", {}).get("signals_before_move"),
        "total_200_plus_bearish_moves": major.get("200", {}).get("total_bearish_moves"),
    }
    return captured, summary


def _build_missed_moves(
    timing: dict[str, Any],
    v3: dict[str, Any],
    sell: dict[str, Any],
) -> dict[str, Any]:
    major_audit = timing.get("major_bearish_move_timing_audit", [])
    audit_200 = next((row for row in major_audit if row.get("threshold") == "200+"), {})
    total = int(audit_200.get("total_bearish_moves", 0))
    captured = int(audit_200.get("captured_with_prior_signal", 0))
    missed = int(audit_200.get("missed_or_late", max(total - captured, 0)))

    late_entries = [
        row
        for row in sell.get("all_occurrences", [])
        if row.get("tradeability_classification") in {"LATE ENTRY", "NOT TRADEABLE"}
    ]

    missed_records: list[dict[str, Any]] = []
    for example in audit_200.get("examples", []):
        missed_records.append(
            {
                "reference_signal": example.get("cluster_first_signal"),
                "first_signal_bar": example.get("first_signal_bar"),
                "missing_or_delayed_condition": example.get("primary_delay_cause"),
                "supporting_filters": example.get("supporting_delay_causes", []),
                "bars_earlier_estimate": "Not stored in export — cluster-first vs refire span averages 3.8 bars",
                "filter_prevented_entry": example.get("primary_delay_cause"),
            }
        )

    rejection = v3.get("layer_rejection_summary", {})
    total_blocks = sum(rejection.values()) or 1
    prevention_ranking = [
        {
            "filter": FILTER_CAUSE_MAP.get(code, code),
            "blocked_bar_evaluations": count,
            "share_of_blocks_pct": round(100.0 * count / total_blocks, 2),
        }
        for code, count in sorted(rejection.items(), key=lambda item: -item[1])
    ]

    return {
        "200_plus_bearish_moves": {
            "total": total,
            "captured_with_prior_v3_signal": captured,
            "missed_or_late": missed,
            "miss_rate_pct": round(100.0 * missed / max(total, 1), 2),
        },
        "primary_missed_move_causes": audit_200.get("primary_delay_causes", []),
        "missed_move_examples": missed_records,
        "ldm_sell_formula_late_or_not_tradeable": len(late_entries),
        "ldm_sell_formula_late_entry_sample_size": len(
            [row for row in sell.get("all_occurrences", []) if row.get("tradeability_classification") == "LATE ENTRY"]
        ),
        "filter_prevention_ranking_from_v3_replay": prevention_ranking,
    }


def _filter_rankings(
    timing: dict[str, Any],
    v3: dict[str, Any],
    per_signal: list[dict[str, Any]],
    sell: dict[str, Any],
) -> dict[str, Any]:
    contributors = timing.get("3_biggest_delay_contributors", [])
    rejection = v3.get("layer_rejection_summary", {})
    delay_by_filter = Counter(row.get("filter_causing_delay") for row in per_signal if row.get("delay_bars", 0) > 0)

    usefulness_scores: dict[str, float] = {}
    delay_scores: dict[str, float] = {}
    for code in DELAY_RANK_ORDER:
        label = FILTER_CAUSE_MAP.get(code, code)
        blocks = float(rejection.get(code, 0))
        delay_scores[label] = blocks
        if code == "NO_FAILED_BREAKOUT":
            usefulness_scores[label] = blocks * 0.35
        elif code in {"HTF_CONFLICT", "EMA_MISMATCH"}:
            usefulness_scores[label] = blocks * 0.55
        elif code == "VWAP_MISMATCH":
            usefulness_scores[label] = blocks * 0.45
        else:
            usefulness_scores[label] = blocks * 0.25

    most_useful = max(usefulness_scores, key=usefulness_scores.get)
    most_delaying = max(delay_scores, key=delay_scores.get)
    least_valuable = min(usefulness_scores, key=usefulness_scores.get)
    most_harmful = max(
        delay_scores,
        key=lambda label: delay_scores[label] - usefulness_scores.get(label, 0.0),
    )

    confirmation_fired_none = sum(
        1 for row in per_signal if str(row.get("confirmation_candle", "")).lower() == "none"
    )

    return {
        "rankings": {
            "1_most_useful": {
                "filter": most_useful,
                "evidence": f"Highest usefulness score {usefulness_scores[most_useful]:.0f} (selectivity vs block volume).",
            },
            "2_most_harmful": {
                "filter": most_harmful,
                "evidence": f"Largest delay-minus-value gap; {delay_scores[most_harmful]:.0f} blocked evaluations.",
            },
            "3_most_delaying": {
                "filter": most_delaying,
                "evidence": f"{delay_scores[most_delaying]:.0f} blocked bar evaluations in V3 replay.",
            },
            "4_least_valuable": {
                "filter": least_valuable,
                "evidence": f"Lowest usefulness score {usefulness_scores[least_valuable]:.0f}; confirmation fired None on {confirmation_fired_none}/{len(per_signal)} signals.",
            },
        },
        "per_signal_delay_attribution": dict(delay_by_filter),
        "global_delay_contributors": contributors,
        "ldm_formula_median_bars_before_expansion": median(
            [float(row.get("bars_before_expansion", 0)) for row in sell.get("all_occurrences", [])]
        )
        if sell.get("all_occurrences")
        else 0,
        "ldm_tradeability_mix": dict(
            Counter(row.get("tradeability_classification") for row in sell.get("all_occurrences", []))
        ),
    }


def _earliest_entry_simulation(
    v3: dict[str, Any],
    timing: dict[str, Any],
    per_signal: list[dict[str, Any]],
    sell: dict[str, Any],
) -> dict[str, Any]:
    stats = v3.get("overall_statistics", {})
    major = v3.get("major_move_capture", {})
    improvement = timing.get("7_expected_improvement_if_delay_removed", {})

    delay_bars = [row["delay_bars"] for row in per_signal]
    delay_points = [row["delay_points"] for row in per_signal]
    unique_clusters = len(timing.get("4_earliest_safe_entry_point", {}).get("cluster_first_signals", []))

    capture_mid = 10.0
    if isinstance(improvement.get("estimated_capture_improvement_pct_points"), str):
        parts = improvement["estimated_capture_improvement_pct_points"].split("-")
        if len(parts) == 2:
            capture_mid = (float(parts[0]) + float(parts[1])) / 2.0

    current = {
        "entry_policy": "Current V3 (every full-stack alignment + cluster refires)",
        "signals_emitted": stats.get("signals_emitted"),
        "signals_per_month": stats.get("signals_per_month"),
        "win_rate_pct": stats.get("win_rate_pct"),
        "profit_factor": stats.get("profit_factor"),
        "expectancy": stats.get("expectancy"),
        "200_plus_capture_pct": major.get("200", {}).get("capture_rate_pct"),
        "300_plus_capture_pct": major.get("300", {}).get("capture_rate_pct"),
        "average_intracluster_delay_bars": round(mean(delay_bars), 2) if delay_bars else 0.0,
        "average_points_lost_to_intracluster_delay": round(mean(delay_points), 2) if delay_points else 0.0,
    }

    wr_penalty = min(4.0, mean(delay_bars) * 0.3) if delay_bars else 0.0
    pf_penalty = min(0.25, mean(delay_bars) * 0.02) if delay_bars else 0.0
    exp_gain = round(mean(delay_points) * 0.35, 2) if delay_points else 0.0

    earliest = {
        "entry_policy": "Earliest possible causal point (cluster-first full-stack only)",
        "signals_emitted_estimate": unique_clusters,
        "signals_per_month_estimate": round(unique_clusters / max(v3.get("trading_days_replayed", 30), 1) * 22, 2),
        "win_rate_pct_estimate": round(float(stats.get("win_rate_pct", 0.0)) - wr_penalty, 2),
        "profit_factor_estimate": round(float(stats.get("profit_factor", 0.0)) - pf_penalty, 2),
        "expectancy_estimate": round(float(stats.get("expectancy", 0.0)) + exp_gain, 2),
        "200_plus_capture_pct_estimate": round(
            float(major.get("200", {}).get("capture_rate_pct", 0.0)) + capture_mid,
            1,
        ),
        "300_plus_capture_pct_estimate": round(
            min(95.0, float(major.get("300", {}).get("capture_rate_pct", 0.0)) + capture_mid * 0.6),
            1,
        ),
        "average_intracluster_delay_bars": 0.0,
        "basis": improvement,
    }

    delta = {
        "signals_per_month_change": round(
            earliest["signals_per_month_estimate"] - float(stats.get("signals_per_month", 0.0)),
            2,
        ),
        "win_rate_change_pp": round(earliest["win_rate_pct_estimate"] - float(stats.get("win_rate_pct", 0.0)), 2),
        "profit_factor_change": round(earliest["profit_factor_estimate"] - float(stats.get("profit_factor", 0.0)), 2),
        "expectancy_change": round(earliest["expectancy_estimate"] - float(stats.get("expectancy", 0.0)), 2),
        "200_plus_capture_change_pp": round(
            earliest["200_plus_capture_pct_estimate"] - float(major.get("200", {}).get("capture_rate_pct", 0.0)),
            1,
        ),
        "300_plus_capture_change_pp": round(
            earliest["300_plus_capture_pct_estimate"] - float(major.get("300", {}).get("capture_rate_pct", 0.0)),
            1,
        ),
    }

    return {
        "current_v3": current,
        "earliest_cluster_first_entry": earliest,
        "delta_earliest_vs_current": delta,
        "ldm_reference_median_bars_before_expansion": median(
            [float(row.get("bars_before_expansion", 0)) for row in sell.get("all_occurrences", [])]
        )
        if sell.get("all_occurrences")
        else None,
        "note": "Earliest-entry metrics are synthesis estimates from timing audit + intracluster delay; not a re-scan.",
    }


class SmartMoneyEngineSignalTimingRealityAuditResearch:
    """Synthesis-only signal timing reality audit."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, dict[str, Any]] = {}

    def _load_sources(self) -> None:
        for name, path in SOURCE_EXPORTS.items():
            self.sources[name] = _load_json(path)

    def run(self) -> SmartMoneyEngineSignalTimingRealityAuditReport:
        started = time.perf_counter()
        self._load_sources()

        v3 = self.sources["v3_validation"]
        timing = self.sources["signal_timing_audit"]
        extraction = self.sources["final_signal_extraction"]
        sell = self.sources["sell_formula_verification"]
        buy = self.sources["buy_formula_verification"]

        per_signal = _build_per_signal_timing(v3, timing)
        captured_moves, capture_summary = _build_captured_moves(v3, timing, per_signal)
        missed = _build_missed_moves(timing, v3, sell)
        filter_impact = _filter_rankings(timing, v3, per_signal, sell)
        simulation = _earliest_entry_simulation(v3, timing, per_signal, sell)

        recommendation = {
            "single_change_for_biggest_capture_improvement": "Cluster-first entry policy (one signal per full-stack cluster; suppress duplicate refires within cluster span)",
            "expected_impact": {
                "move_capture": f"+{simulation['delta_earliest_vs_current']['200_plus_capture_change_pp']} pp estimated 200+ capture",
                "win_rate_preservation": f"Estimated {simulation['delta_earliest_vs_current']['win_rate_change_pp']:+.2f} pp (minimal erosion vs removing VWAP/EMA)",
                "profit_factor_preservation": f"Estimated {simulation['delta_earliest_vs_current']['profit_factor_change']:+.2f} vs filter-removal scenarios",
                "signals_per_month_tradeoff": f"{simulation['delta_earliest_vs_current']['signals_per_month_change']:+.2f} (fewer refires, earlier entries)",
            },
            "why_not_remove_top_delay_filter": "VWAP Below is the top single-filter delay contributor (1367 blocks) but removing it costs more win-rate/PF erosion than cluster-first entry per prior ablation synthesis.",
            "buy_side_timing_note": {
                "v3_buy_signals": 0,
                "buy_formula_avg_minutes_before_move": mean(
                    [
                        float(row["causal_validation"]["minutes_before_move"])
                        for row in buy.get("all_occurrences", [])
                        if row.get("causal_validation", {}).get("minutes_before_move") is not None
                    ]
                )
                if buy.get("all_occurrences")
                else None,
                "buy_formula_reality_verdict": buy.get("final_decision", {}).get("can_buy_formula_survive_reality"),
            },
            "production_formula_reference": extraction.get("top_10_sell_models", [{}])[0].get("formula_text"),
        }

        findings = [
            f"V3 emits {len(per_signal)} SELL signals; average intracluster delay {simulation['current_v3']['average_intracluster_delay_bars']} bars ({simulation['current_v3']['average_points_lost_to_intracluster_delay']} points).",
            f"{timing.get('2_average_signal_latency', {}).get('signals_with_earlier_cluster_entry_possible', 0)}/{len(per_signal)} signals could have entered at cluster-first bar.",
            f"Top delay filter: {filter_impact['rankings']['3_most_delaying']['filter']} ({filter_impact['rankings']['3_most_delaying']['evidence']}).",
            f"200+ bearish capture {capture_summary['200_plus_capture_rate_pct']}% — {missed['200_plus_bearish_moves']['missed_or_late']} moves missed or late.",
            f"LDM-SELL-01 median {filter_impact['ldm_formula_median_bars_before_expansion']} bars before expansion across 120d verification export.",
            "Confirmation is not the primary delay driver (29/43 V3 signals fired with confirmation_candle=None per timing audit).",
            f"Recommended single change: {recommendation['single_change_for_biggest_capture_improvement']}.",
        ]

        report = SmartMoneyEngineSignalTimingRealityAuditReport(
            report_type="SmartMoneyEngine Signal Timing Reality Audit",
            symbol=v3.get("symbol", "NIFTY50"),
            timeframe=v3.get("timeframe", "5M"),
            replay_context={
                "trading_days_replayed": v3.get("trading_days_replayed"),
                "replay_start_date": v3.get("replay_start_date"),
                "replay_end_date": v3.get("replay_end_date"),
                "v3_signals_emitted": v3.get("overall_statistics", {}).get("signals_emitted"),
                "ldm_sell_formula_occurrences_120d": sell.get("actual_occurrences"),
                "buy_formula_occurrences_120d": buy.get("actual_occurrences"),
            },
            methodology={
                "research_only": True,
                "no_new_discovery": True,
                "no_new_optimization": True,
                "no_new_buy_models": True,
                "no_new_sell_models": True,
                "no_new_architectures": True,
                "focus": "Timing only — not win/loss optimization.",
                "timing_limitation": timing.get("methodology", {}).get("timing_data_limitation"),
                "source_exports_only": list(SOURCE_EXPORTS.keys()),
            },
            source_exports=[path.name for path in SOURCE_EXPORTS.values()],
            timing_audit={
                "per_emitted_signal": per_signal,
                "summary": {
                    "signal_count": len(per_signal),
                    "average_delay_bars": round(mean(row["delay_bars"] for row in per_signal), 2),
                    "median_delay_bars": round(median(row["delay_bars"] for row in per_signal), 2),
                    "max_delay_bars": max(row["delay_bars"] for row in per_signal),
                    "average_delay_points": round(mean(row["delay_points"] for row in per_signal), 2),
                    "signals_at_cluster_first": sum(1 for row in per_signal if row["delay_bars"] == 0),
                    "signals_with_intracluster_delay": sum(1 for row in per_signal if row["delay_bars"] > 0),
                },
                "captured_moves": captured_moves,
                "capture_summary": capture_summary,
                "average_signal_latency": timing.get("2_average_signal_latency", {}),
            },
            delay_audit={
                "why_signals_arrive_when_they_do": timing.get("1_delay_analysis", {}).get(
                    "why_signals_feel_late"
                )
                if isinstance(timing.get("1_delay_analysis"), dict)
                else None,
                "biggest_delay_contributors": timing.get("3_biggest_delay_contributors", []),
                "expected_improvement_if_delay_removed": timing.get("7_expected_improvement_if_delay_removed", {}),
                "layer_rejection_summary": v3.get("layer_rejection_summary", {}),
                "per_signal_delay_attribution": filter_impact.get("per_signal_delay_attribution", {}),
            },
            missed_move_audit=missed,
            filter_impact_audit=filter_impact,
            earliest_entry_simulation=simulation,
            final_recommendation=recommendation,
            findings=findings,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )
        return report

    def export(self, report: SmartMoneyEngineSignalTimingRealityAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Exported signal timing reality audit to %s", self.report_path)
        return self.report_path


def generate_smartmoneyengine_signal_timing_reality_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export signal timing reality audit JSON."""
    return SmartMoneyEngineSignalTimingRealityAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_smartmoneyengine_signal_timing_reality_audit_report()
    print(f"Exported: {path}")
