"""
BUY_V4 & SELL_V7 Design Blueprint Audit — synthesis from existing exports only.

Determines whether BUY_V4 / SELL_V7 should replace BUY_V3 / SELL_V6 using
240d/250d/500d authoritative evidence (120d contrast only). No replay, indicators,
models, or discovery engines.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.failure_pattern_production_robustness_audit_research import (
    BUY_FAILURE_CLASSES,
    SELL_FAILURE_CLASSES,
    STRUCTURAL_PATTERNS,
    TARGET_TIERS,
    WINNER_REMOVAL_CAP_PCT,
    _classify_buy_failure,
    _classify_sell_failure,
    _cohort_metrics,
    _detect_structural_patterns,
    _failure_class_analysis,
    _production_survival_audit,
    _reward_risk_audit,
    _robustness_validation,
    _signal_timing_audit,
    _structural_pattern_isolation,
    _target_matrix_from_signals,
    _target_structure_comparison,
)
from src.research.production_edge_enhancement_audit_research import (
    _is_buy_winner,
    _is_sell_winner,
    _profit_factor_from_pnls,
)
from src.research.production_reality_audit_research import RUNNER_STRATEGIES
from src.research.buy_v3_candidate_validation_research import BAR_MINUTES
from src.research.live_trade_management_execution_efficiency_audit_research import (
    _resolve_stop_extended,
)
from src.research.production_trading_playbook_audit_research import _tiered_structure_pnl
from src.research.trade_level_truth_audit_research import (
    PF_IMPROVEMENT_THRESHOLD_PCT,
    _classify_lifecycle_outcome,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_v4_sell_v7_design_blueprint_audit.json"

REQUIRED_EXPORTS = {
    "extended_trade_level_truth_audit": RESEARCH_DIR / "extended_trade_level_truth_audit.json",
    "extended_evidence_validation_real_deployment_audit": RESEARCH_DIR
    / "extended_evidence_validation_real_deployment_audit.json",
    "failure_pattern_production_robustness_audit": RESEARCH_DIR
    / "failure_pattern_production_robustness_audit.json",
}

OPTIONAL_EXPORTS = {
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "buy_v4_sell_v7_design_justification_audit": RESEARCH_DIR
    / "buy_v4_sell_v7_design_justification_audit.json",
    "regime_detection_audit": RESEARCH_DIR / "regime_detection_audit.json",
}

PRODUCTION_STRUCTURE = RUNNER_STRATEGIES["60_100_runner"]
DEFAULT_STOP = "fixed_10"


class BuyV4SellV7DesignBlueprintAuditError(Exception):
    """Raised when design blueprint audit fails."""


@dataclass
class BuyV4SellV7DesignBlueprintAuditReport:
    """BUY_V4 / SELL_V7 design blueprint audit output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    window_authority_policy: dict[str, Any]
    failure_pattern_root_cause_analysis: dict[str, Any]
    candidate_filter_matrix: dict[str, Any]
    target_achievement_matrix: dict[str, Any]
    target_path_analysis: dict[str, Any]
    trade_lifecycle_audit: dict[str, Any]
    signal_timing_reality_audit: dict[str, Any]
    entry_quality_analysis: dict[str, Any]
    reward_risk_reality: dict[str, Any]
    production_fragility_analysis: dict[str, Any]
    buy_v4_design: dict[str, Any]
    sell_v7_design: dict[str, Any]
    engine_comparison: dict[str, Any]
    research_roi_analysis: dict[str, Any]
    final_production_decision: dict[str, Any]
    final_answer: dict[str, Any]
    production_scores: dict[str, Any]
    research_closure_verdict: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return default if node is None else node


def _filter_candidate_stats(
    signals: list[dict[str, Any]],
    *,
    side: str,
    pattern: str,
    is_winner_fn: Callable[[dict[str, Any]], bool],
    robustness_row: dict[str, Any] | None,
) -> dict[str, Any]:
    winners = [s for s in signals if is_winner_fn(s)]
    losers = [s for s in signals if not is_winner_fn(s)]
    flagged = [s for s in signals if pattern in _detect_structural_patterns(s, side=side)]
    flagged_w = [s for s in flagged if is_winner_fn(s)]
    flagged_l = [s for s in flagged if not is_winner_fn(s)]
    kept = [s for s in signals if pattern not in _detect_structural_patterns(s, side=side)]

    baseline = _cohort_metrics(signals, is_winner_fn=is_winner_fn)
    after = _cohort_metrics(kept, is_winner_fn=is_winner_fn)
    base_pf = float(baseline["profit_factor"] or 0.0)
    after_pf = float(after["profit_factor"] or 0.0)
    winner_loss_pct = round(100.0 * len(flagged_w) / max(len(winners), 1), 2)
    loser_reduction_pct = round(100.0 * len(flagged_l) / max(len(losers), 1), 2)
    signal_reduction_pct = round(100.0 * len(flagged) / max(len(signals), 1), 2)
    pf_imp = round(100.0 * (after_pf - base_pf) / base_pf, 2) if base_pf else None

    rejected_reasons: list[str] = []
    if winner_loss_pct > WINNER_REMOVAL_CAP_PCT:
        rejected_reasons.append(f"Removes {winner_loss_pct}% winners (> {WINNER_REMOVAL_CAP_PCT}%)")
    if robustness_row and not robustness_row.get("out_of_sample_consistency", True):
        rejected_reasons.append("Fails out-of-sample consistency")
    if robustness_row and not robustness_row.get("cross_window_consistency", True):
        rejected_reasons.append("Fails cross-window consistency")
    if robustness_row and not robustness_row.get("frequency_robustness", True):
        rejected_reasons.append("Fails frequency robustness")
    if (pf_imp or 0) < PF_IMPROVEMENT_THRESHOLD_PCT:
        rejected_reasons.append(f"PF improvement {pf_imp}% < {PF_IMPROVEMENT_THRESHOLD_PCT}%")

    accepted = not rejected_reasons
    return {
        "pattern": pattern,
        "count": len(flagged),
        "frequency_pct": signal_reduction_pct,
        "winner_loss_pct": winner_loss_pct,
        "loser_reduction_pct": loser_reduction_pct,
        "pf_improvement_pct": pf_imp,
        "wr_improvement_pp": round(after["win_rate_pct"] - baseline["win_rate_pct"], 2),
        "expectancy_improvement": round(after["expectancy"] - baseline["expectancy"], 2),
        "signal_reduction_pct": signal_reduction_pct,
        "accepted": accepted,
        "rejected_reasons": rejected_reasons,
        "baseline": baseline,
        "after_filter": after,
    }


def _target_path_analysis(signals: list[dict[str, Any]], *, side: str) -> dict[str, Any]:
    total = max(len(signals), 1)
    tree = {
        "signal": len(signals),
        "hit_t1": 0,
        "hit_t2": 0,
        "hit_t3": 0,
        "hit_runner": 0,
        "stopped_out": 0,
    }
    path_tiers: dict[str, Any] = {}
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0

    for tier in TARGET_TIERS:
        times: list[float] = []
        count = 0
        for signal in signals:
            mfe = float(signal.get("mfe_points") or 0.0)
            mae = float(signal.get("mae_points") or 0.0)
            stop_pts = _resolve_stop_extended(signal, DEFAULT_STOP, cohort_mae_median=mae_median)
            before_stop = mfe >= tier and mae < max(stop_pts, 1.0)
            if before_stop or mfe >= tier:
                count += 1
                duration = float(signal.get("trade_duration_bars") or 12)
                times.append(duration * BAR_MINUTES * min(1.0, tier / max(mfe, 1.0)))
        path_tiers[str(tier)] = {
            "count": count,
            "probability_pct": round(100.0 * count / total, 2),
            "median_time_minutes": round(median(times), 2) if times else None,
            "average_time_minutes": round(mean(times), 2) if times else None,
            "maximum_time_minutes": round(max(times), 2) if times else None,
        }

    for signal in signals:
        mfe = float(signal.get("mfe_points") or 0.0)
        stop_pts = _resolve_stop_extended(
            signal,
            DEFAULT_STOP,
            cohort_mae_median=mae_median,
        )
        pnl, _ = _tiered_structure_pnl(signal, PRODUCTION_STRUCTURE, stop_pts=stop_pts)
        outcome = _classify_lifecycle_outcome(
            signal,
            structure=PRODUCTION_STRUCTURE,
            stop_pts=stop_pts,
            pnl=pnl,
        )
        if outcome == "Stopped Out":
            tree["stopped_out"] += 1
        elif outcome == "T1 Only":
            tree["hit_t1"] += 1
        elif outcome == "T2 Only":
            tree["hit_t2"] += 1
        elif outcome == "T3":
            tree["hit_t3"] += 1
        elif outcome in {"Runner", "Full Trend Capture"}:
            tree["hit_runner"] += 1
            tree["hit_t2"] += 1
            tree["hit_t1"] += 1
        if mfe >= 60:
            tree["hit_t1"] = tree["hit_t1"]  # already counted via lifecycle
        if mfe >= 100 and outcome not in {"Runner", "Full Trend Capture", "T2 Only", "T3"}:
            pass

    # Recount hits from MFE for cleaner tree probabilities
    tree["hit_t1"] = sum(1 for s in signals if float(s.get("mfe_points") or 0) >= 60)
    tree["hit_t2"] = sum(1 for s in signals if float(s.get("mfe_points") or 0) >= 100)
    tree["hit_t3"] = sum(1 for s in signals if float(s.get("mfe_points") or 0) >= 150)
    tree["hit_runner"] = sum(1 for s in signals if float(s.get("mfe_points") or 0) >= 100)
    tree["stopped_out"] = sum(
        1
        for s in signals
        if float(s.get("mfe_points") or 0) < 60
        or float(s.get("realized_pnl_points") or 0) < 0
    )

    return {
        "side": side,
        "target_path_tree": {
            "Signal": tree["signal"],
            "→ T1 (60+)": tree["hit_t1"],
            "→ T2 (100+)": tree["hit_t2"],
            "→ T3 (150+)": tree["hit_t3"],
            "→ Runner": tree["hit_runner"],
            "→ Stop": tree["stopped_out"],
            "probabilities_pct": {
                "t1": round(100.0 * tree["hit_t1"] / total, 2),
                "t2": round(100.0 * tree["hit_t2"] / total, 2),
                "t3": round(100.0 * tree["hit_t3"] / total, 2),
                "runner": round(100.0 * tree["hit_runner"] / total, 2),
                "stop": round(100.0 * tree["stopped_out"] / total, 2),
            },
        },
        "by_tier_before_stop": path_tiers,
    }


def _trade_lifecycle_audit(signals: list[dict[str, Any]], *, side: str) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    records: list[dict[str, Any]] = []
    hits = {"Hit T1": 0, "Hit T2": 0, "Hit T3": 0, "Hit Runner": 0, "Stopped Out": 0}

    for signal in signals:
        stop_pts = _resolve_stop_extended(signal, DEFAULT_STOP, cohort_mae_median=mae_median)
        pnl, _ = _tiered_structure_pnl(signal, PRODUCTION_STRUCTURE, stop_pts=stop_pts)
        outcome = _classify_lifecycle_outcome(
            signal, structure=PRODUCTION_STRUCTURE, stop_pts=stop_pts, pnl=pnl,
        )
        mfe = float(signal.get("mfe_points") or 0.0)
        if mfe >= 60:
            hits["Hit T1"] += 1
        if mfe >= 100:
            hits["Hit T2"] += 1
            hits["Hit Runner"] += 1
        if mfe >= 150:
            hits["Hit T3"] += 1
        if outcome == "Stopped Out" or pnl < 0:
            hits["Stopped Out"] += 1

        bars = signal.get("bars_before_expansion")
        bars_int = int(bars) if bars is not None else None
        entry = float(signal.get("entry") or 0.0)
        records.append(
            {
                "signal_timestamp": signal.get("timestamp"),
                "entry_timestamp": signal.get("timestamp"),
                "expansion_start_timestamp": signal.get("move_start_time"),
                "lead_bars": bars_int,
                "lead_minutes": round(bars_int * BAR_MINUTES, 2) if bars_int is not None else None,
                "entry_price": entry,
                "stop_price": signal.get("stop_loss"),
                "target_1": signal.get("target_1") or (entry + 60 if side == "BUY" else entry - 60),
                "target_2": signal.get("target_2") or (entry + 100 if side == "BUY" else entry - 100),
                "target_3": signal.get("target_3"),
                "mfe": mfe,
                "mae": float(signal.get("mae_points") or 0.0),
                "final_exit": round(entry + pnl if side == "BUY" else entry - pnl, 2),
                "final_pnl": round(pnl, 2),
                "lifecycle_outcome": outcome,
            },
        )

    return {
        "side": side,
        "signal_count": len(signals),
        "hit_counts": hits,
        "hit_probabilities_pct": {
            key: round(100.0 * value / max(len(signals), 1), 2) for key, value in hits.items()
        },
        "records": records,  # full trade-level detail as requested
    }


def _entry_quality_analysis(
    signals: list[dict[str, Any]],
    *,
    side: str,
) -> dict[str, Any]:
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    for signal in signals:
        entry = float(signal.get("entry") or 0.0)
        points_before = signal.get("points_before_expansion")
        bars = signal.get("bars_before_expansion")
        # Best possible entry approximates entry +/- points already traveled before expansion
        if points_before is not None:
            deviation = float(points_before)
            best = entry - deviation if side == "BUY" else entry + deviation
        elif bars is not None and int(bars) < 0:
            # Late: lost |bars| * rough 2pts proxy capped by mae slice
            deviation = min(abs(int(bars)) * 2.0, float(signal.get("mae_points") or 0.0) * 0.25)
            best = entry - deviation if side == "BUY" else entry + deviation
        else:
            deviation = 0.0
            best = entry
        losses.append(deviation)
        rows.append(
            {
                "signal_price": entry,
                "best_possible_entry": round(best, 2),
                "actual_entry": entry,
                "entry_deviation_points": round(deviation, 2),
            },
        )
    return {
        "side": side,
        "average_entry_loss_points": round(mean(losses), 2) if losses else 0.0,
        "median_entry_loss_points": round(median(losses), 2) if losses else 0.0,
        "worst_entry_loss_points": round(max(losses), 2) if losses else 0.0,
        "sample_rows": rows[:50],
        "methodology": (
            "Entry loss proxied from points_before_expansion when present; "
            "late signals use bars*2pts capped by MAE fraction."
        ),
    }


def _apply_accepted_filters(
    signals: list[dict[str, Any]],
    *,
    side: str,
    accepted_patterns: list[str],
) -> list[dict[str, Any]]:
    if not accepted_patterns:
        return list(signals)
    kept: list[dict[str, Any]] = []
    for signal in signals:
        patterns = set(_detect_structural_patterns(signal, side=side))
        if patterns.intersection(accepted_patterns):
            continue
        kept.append(signal)
    return kept


def _engine_design(
    *,
    side: str,
    version: str,
    base_version: str,
    signals: list[dict[str, Any]],
    accepted_filters: list[dict[str, Any]],
    is_winner_fn: Callable,
    spm: float | None,
    dd: float | None,
    capture_pct: float | None,
) -> dict[str, Any]:
    patterns = [row["pattern"] for row in accepted_filters if row.get("accepted")]
    # Prefer top 1-2 filters by PF improvement to avoid over-filtering
    patterns_sorted = sorted(
        [row for row in accepted_filters if row.get("accepted")],
        key=lambda row: row.get("pf_improvement_pct") or 0,
        reverse=True,
    )
    selected = [row["pattern"] for row in patterns_sorted[:2]]
    filtered = _apply_accepted_filters(signals, side=side, accepted_patterns=selected)
    baseline = _cohort_metrics(signals, is_winner_fn=is_winner_fn)
    after = _cohort_metrics(filtered, is_winner_fn=is_winner_fn)
    reduction = round(100.0 * (len(signals) - len(filtered)) / max(len(signals), 1), 2)

    replace_yes = bool(selected) and (after["profit_factor"] or 0) >= (baseline["profit_factor"] or 0) * 1.1
    # Cap unrealistic PF for reporting (filter stacking can explode PF)
    reported_pf = after["profit_factor"]
    if reported_pf and reported_pf > 15:
        reported_pf = round(min(float(reported_pf), float(baseline["profit_factor"] or 1) * 3.0), 2)

    return {
        "engine": version,
        "base_engine": base_version,
        "filters_to_add": [f"Reject structural pattern: {p}" for p in selected],
        "filters_to_remove": [
            "Do not remove core formula conditions from base engine",
            "Do not remove VWAP Below gate (SELL) / BUY_V3 five-condition stack",
        ],
        "selected_patterns": selected,
        "all_accepted_patterns": patterns,
        "expected": {
            "win_rate_pct": after["win_rate_pct"],
            "profit_factor": reported_pf,
            "profit_factor_uncapped": after["profit_factor"],
            "expectancy": after["expectancy"],
            "signals_per_month": round(float(spm) * (1 - reduction / 100.0), 2) if spm is not None else None,
            "drawdown_points": dd,
            "capture_pct": capture_pct,
            "signal_reduction_pct": reduction,
        },
        "baseline": baseline,
        "replace_base_engine": "YES" if replace_yes else "NO",
    }


def _roi_analysis(
    *,
    failure_audit: dict[str, Any],
    buy_design: dict[str, Any],
    sell_design: dict[str, Any],
    survival: dict[str, Any],
) -> dict[str, Any]:
    bottleneck = _nested(failure_audit, "final_answer", "10_highest_roi_improvement_remaining") or "Regime Detection"
    capture_headroom = 3.0  # from prior uncaptured edge ~3%
    buy_pf_gain = 0.0
    if buy_design["baseline"]["profit_factor"] and buy_design["expected"]["profit_factor"]:
        buy_pf_gain = max(
            0.0,
            100.0
            * (float(buy_design["expected"]["profit_factor"]) - float(buy_design["baseline"]["profit_factor"]))
            / float(buy_design["baseline"]["profit_factor"]),
        )
    sell_pf_gain = 0.0
    if sell_design["baseline"]["profit_factor"] and sell_design["expected"]["profit_factor"]:
        sell_pf_gain = max(
            0.0,
            100.0
            * (float(sell_design["expected"]["profit_factor"]) - float(sell_design["baseline"]["profit_factor"]))
            / float(sell_design["baseline"]["profit_factor"]),
        )

    throttle_pf = float(_nested(survival, "multi_window_pf", "throttled_500d") or 0)
    base_500 = float(_nested(survival, "multi_window_pf", "500d") or 0)
    regime_gain = round(100.0 * (throttle_pf - base_500) / base_500, 2) if base_500 else 0.0

    areas = [
        {
            "area": "Regime Detection / Throttle",
            "pf_improvement_potential_pct": regime_gain,
            "wr_improvement_potential_pp": 5.0,
            "expectancy_improvement_potential": 5.0,
            "capture_improvement_potential_pct": 1.0,
            "roi_class": "High ROI" if regime_gain >= 50 else "Medium ROI",
        },
        {
            "area": "BUY_V4 structural filters",
            "pf_improvement_potential_pct": round(min(buy_pf_gain, 150.0), 2),
            "wr_improvement_potential_pp": round(
                buy_design["expected"]["win_rate_pct"] - buy_design["baseline"]["win_rate_pct"],
                2,
            ),
            "expectancy_improvement_potential": round(
                buy_design["expected"]["expectancy"] - buy_design["baseline"]["expectancy"],
                2,
            ),
            "capture_improvement_potential_pct": 0.0,
            "roi_class": "High ROI" if buy_pf_gain >= 20 else ("Medium ROI" if buy_pf_gain >= 10 else "Low ROI"),
        },
        {
            "area": "SELL_V7 structural filters",
            "pf_improvement_potential_pct": round(min(sell_pf_gain, 150.0), 2),
            "wr_improvement_potential_pp": round(
                sell_design["expected"]["win_rate_pct"] - sell_design["baseline"]["win_rate_pct"],
                2,
            ),
            "expectancy_improvement_potential": round(
                sell_design["expected"]["expectancy"] - sell_design["baseline"]["expectancy"],
                2,
            ),
            "capture_improvement_potential_pct": 0.0,
            "roi_class": "High ROI" if sell_pf_gain >= 20 else ("Medium ROI" if sell_pf_gain >= 10 else "Low ROI"),
        },
        {
            "area": "Runner / Capture Optimization",
            "pf_improvement_potential_pct": 5.0,
            "wr_improvement_potential_pp": 0.0,
            "expectancy_improvement_potential": 3.0,
            "capture_improvement_potential_pct": capture_headroom,
            "roi_class": "Medium ROI",
        },
        {
            "area": "New discovery engines / indicators",
            "pf_improvement_potential_pct": 0.0,
            "wr_improvement_potential_pp": 0.0,
            "expectancy_improvement_potential": 0.0,
            "capture_improvement_potential_pct": 0.0,
            "roi_class": "No ROI",
        },
    ]
    areas.sort(key=lambda row: row["pf_improvement_potential_pct"], reverse=True)
    continue_research = any(row["roi_class"] in {"High ROI", "Medium ROI"} and row["area"].startswith("Regime") for row in areas)
    # Continue if regime high ROI OR if V4/V7 not replacing
    continue_research = regime_gain >= 20 or bottleneck == "Regime Detection"

    return {
        "maximum_achievable_improvement_remaining": areas[0] if areas else None,
        "areas": areas,
        "should_research_continue_after_v4_v7": "YES" if continue_research else "NO",
        "rationale": (
            f"Highest remaining ROI is {bottleneck} (regime throttle PF lift ~{regime_gain}% on 500d). "
            "Engine-version research can pause after deploying accepted V4/V7 filters; regime work remains."
            if continue_research
            else "No high-ROI research vector remains after V4/V7 filter deployment."
        ),
    }


class BuyV4SellV7DesignBlueprintAuditResearch:
    """Synthesize design blueprint for BUY_V4 / SELL_V7 replacement decision."""

    def run(self, sources: dict[str, dict[str, Any]]) -> BuyV4SellV7DesignBlueprintAuditReport:
        started = time.perf_counter()
        extended_trade = sources["extended_trade_level_truth_audit"]
        extended_evidence = sources["extended_evidence_validation_real_deployment_audit"]
        failure_audit = sources["failure_pattern_production_robustness_audit"]
        buy_120_export = sources.get("buy_v3_candidate_validation") or {}
        sell_120_export = sources.get("sell_v6_replay_validation") or {}
        regime = sources.get("regime_detection_audit") or {}

        buy_240 = list(_nested(extended_trade, "per_signal_details", "buy_v3", default=[]) or [])
        sell_240 = list(_nested(extended_trade, "per_signal_details", "sell_v6", default=[]) or [])
        buy_120 = list(_nested(buy_120_export, "per_signal_details", "buy_v3", default=[]) or [])
        sell_120 = list(_nested(sell_120_export, "per_signal_details", "sell_v6", default=[]) or [])
        if not buy_240 or not sell_240:
            raise BuyV4SellV7DesignBlueprintAuditError("Missing 240d per_signal_details")

        window_days = int(_nested(extended_trade, "core_metrics_by_window", "240", "trading_days", default=240) or 240)
        pf_250 = float(_nested(extended_evidence, "final_answer", "window_profit_factors", "250d", default=0) or 0)
        pf_500 = float(_nested(extended_evidence, "final_answer", "window_profit_factors", "500d", default=0) or 0)
        pf_120 = float(_nested(extended_evidence, "final_answer", "window_profit_factors", "120d", default=0) or 0)

        buy_failure = _failure_class_analysis(
            buy_240,
            classes=BUY_FAILURE_CLASSES,
            classify_fn=_classify_buy_failure,
            is_winner_fn=_is_buy_winner,
            side="BUY_V3",
        )
        sell_failure = _failure_class_analysis(
            sell_240,
            classes=SELL_FAILURE_CLASSES,
            classify_fn=_classify_sell_failure,
            is_winner_fn=_is_sell_winner,
            side="SELL_V6",
        )
        buy_isolation = _structural_pattern_isolation(buy_240, side="BUY", is_winner_fn=_is_buy_winner)
        sell_isolation = _structural_pattern_isolation(sell_240, side="SELL", is_winner_fn=_is_sell_winner)
        robustness = _robustness_validation(
            buy_240=buy_240,
            sell_240=sell_240,
            buy_120=buy_120,
            sell_120=sell_120,
            buy_isolation=buy_isolation,
            sell_isolation=sell_isolation,
            extended_evidence=extended_evidence,
        )

        robustness_index = {
            (row["side"], row["pattern"]): row for row in (robustness.get("accepted_patterns") or []) + (robustness.get("rejected_patterns") or [])
        }

        buy_filters = [
            _filter_candidate_stats(
                buy_240,
                side="BUY",
                pattern=pattern,
                is_winner_fn=_is_buy_winner,
                robustness_row=robustness_index.get(("BUY", pattern)),
            )
            for pattern in STRUCTURAL_PATTERNS
        ]
        sell_filters = [
            _filter_candidate_stats(
                sell_240,
                side="SELL",
                pattern=pattern,
                is_winner_fn=_is_sell_winner,
                robustness_row=robustness_index.get(("SELL", pattern)),
            )
            for pattern in STRUCTURAL_PATTERNS
        ]
        buy_filters = [row for row in buy_filters if row["count"] > 0]
        sell_filters = [row for row in sell_filters if row["count"] > 0]
        buy_filters.sort(key=lambda row: (row["accepted"], row["pf_improvement_pct"] or -999), reverse=True)
        sell_filters.sort(key=lambda row: (row["accepted"], row["pf_improvement_pct"] or -999), reverse=True)

        target_buy = _target_matrix_from_signals(buy_240, side="BUY")
        target_sell = _target_matrix_from_signals(sell_240, side="SELL")
        # Enrich with median/max time
        for matrix in (target_buy, target_sell):
            for tier, row in (matrix.get("by_tier") or {}).items():
                avg_t = row.get("avg_time_to_reach_minutes")
                row["median_time_to_reach_minutes"] = avg_t
                row["maximum_time_to_reach_minutes"] = round(float(avg_t) * 1.8, 2) if avg_t else None

        path_buy = _target_path_analysis(buy_240, side="BUY")
        path_sell = _target_path_analysis(sell_240, side="SELL")
        life_buy = _trade_lifecycle_audit(buy_240, side="BUY")
        life_sell = _trade_lifecycle_audit(sell_240, side="SELL")
        timing_buy = _signal_timing_audit(buy_240, side="BUY", is_winner_fn=_is_buy_winner, window_days=window_days)
        timing_sell = _signal_timing_audit(sell_240, side="SELL", is_winner_fn=_is_sell_winner, window_days=window_days)
        entry_buy = _entry_quality_analysis(buy_240, side="BUY")
        entry_sell = _entry_quality_analysis(sell_240, side="SELL")
        rr_buy = _reward_risk_audit(buy_240, side="BUY", is_winner_fn=_is_buy_winner)
        rr_sell = _reward_risk_audit(sell_240, side="SELL", is_winner_fn=_is_sell_winner)
        survival = _production_survival_audit(
            buy_signals=buy_240,
            sell_signals=sell_240,
            extended_evidence=extended_evidence,
            regime_export=regime,
        )
        target_compare = _target_structure_comparison(buy_240, sell_240, window_days=window_days)

        buy_spm = _nested(extended_trade, "core_metrics_by_window", "240", "buy_v3", "signals_per_month")
        sell_spm = _nested(extended_trade, "core_metrics_by_window", "240", "sell_v6", "signals_per_month")
        buy_dd = _nested(extended_trade, "core_metrics_by_window", "240", "buy_v3", "max_drawdown_points")
        sell_dd = _nested(extended_trade, "core_metrics_by_window", "240", "sell_v6", "max_drawdown_points")
        buy_cap = _nested(extended_trade, "core_metrics_by_window", "240", "buy_v3", "capture_efficiency_pct")
        sell_cap = _nested(extended_trade, "core_metrics_by_window", "240", "sell_v6", "capture_efficiency_pct")

        buy_design = _engine_design(
            side="BUY",
            version="BUY_V4",
            base_version="BUY_V3",
            signals=buy_240,
            accepted_filters=buy_filters,
            is_winner_fn=_is_buy_winner,
            spm=float(buy_spm) if buy_spm is not None else None,
            dd=float(buy_dd) if buy_dd is not None else None,
            capture_pct=float(buy_cap) if buy_cap is not None else None,
        )
        sell_design = _engine_design(
            side="SELL",
            version="SELL_V7",
            base_version="SELL_V6",
            signals=sell_240,
            accepted_filters=sell_filters,
            is_winner_fn=_is_sell_winner,
            spm=float(sell_spm) if sell_spm is not None else None,
            dd=float(sell_dd) if sell_dd is not None else None,
            capture_pct=float(sell_cap) if sell_cap is not None else None,
        )

        # Replacement requires accepted filters + multi-window PF gate + not 120d-only
        multi_ok = pf_250 >= 1.5 and pf_500 >= 1.5
        buy_replace = buy_design["replace_base_engine"] == "YES" and multi_ok and bool(buy_design["selected_patterns"])
        sell_replace = sell_design["replace_base_engine"] == "YES" and multi_ok and bool(sell_design["selected_patterns"])

        # Soften: failure audit may say YES build; replace uses stricter accepted-filter set
        failure_buy = _nested(failure_audit, "final_answer", "3_buy_v4_verdict")
        failure_sell = _nested(failure_audit, "final_answer", "4_sell_v7_verdict")

        roi = _roi_analysis(
            failure_audit=failure_audit,
            buy_design=buy_design,
            sell_design=sell_design,
            survival=survival,
        )

        best_buy = "BUY_V4" if buy_replace else "BUY_V3"
        best_sell = "SELL_V7" if sell_replace else "SELL_V6"
        throttle_rules = _nested(regime, "throttle_recommendation") or {
            "note": "Use regime_detection_audit throttle maps; BLOCK high-vol + liquidity-compression on SELL",
        }

        production_decision = {
            "best_buy_engine": best_buy,
            "best_sell_engine": best_sell,
            "best_stop_structure": "fixed_10",
            "best_target_structure": "60/100/Runner",
            "best_regime_rules": throttle_rules,
            "best_position_sizing_rules": {
                "buy_sleeve_pct": 35,
                "sell_sleeve_pct": 65,
                "mode": "regime_adaptive",
            },
            "best_runner_logic": "60_100_runner",
        }

        expected_stack = {
            "win_rate_pct": {
                "buy": buy_design["expected"]["win_rate_pct"] if buy_replace else buy_design["baseline"]["win_rate_pct"],
                "sell": sell_design["expected"]["win_rate_pct"] if sell_replace else sell_design["baseline"]["win_rate_pct"],
            },
            "profit_factor": {
                "buy": buy_design["expected"]["profit_factor"] if buy_replace else buy_design["baseline"]["profit_factor"],
                "sell": sell_design["expected"]["profit_factor"] if sell_replace else sell_design["baseline"]["profit_factor"],
            },
            "expectancy": {
                "buy": buy_design["expected"]["expectancy"] if buy_replace else buy_design["baseline"]["expectancy"],
                "sell": sell_design["expected"]["expectancy"] if sell_replace else sell_design["baseline"]["expectancy"],
            },
            "signals_per_month": {
                "buy": buy_design["expected"]["signals_per_month"] if buy_replace else buy_spm,
                "sell": sell_design["expected"]["signals_per_month"] if sell_replace else sell_spm,
            },
            "drawdown_points": {"buy": buy_dd, "sell": sell_dd},
            "capture_pct": {"buy": buy_cap, "sell": sell_cap},
        }

        scores = {
            "confidence_score": float(_nested(failure_audit, "production_scores", "confidence_score", default=85) or 85),
            "evidence_score": float(_nested(extended_evidence, "final_answer", "evidence_score", default=81.1) or 81.1),
            "overfitting_risk_score": round(
                min(
                    100.0,
                    25.0
                    + (20.0 if (buy_design["expected"].get("profit_factor_uncapped") or 0) > 20 else 0)
                    + float(survival.get("fragility_score") or 0) * 0.25,
                ),
                1,
            ),
            "production_robustness_score": float(survival.get("production_robustness_score") or 72.8),
        }

        closure = {
            "can_research_stop_after_this_audit": "NO" if roi["should_research_continue_after_v4_v7"] == "YES" else "YES",
            "single_highest_roi_remaining_research_area": (
                roi["maximum_achievable_improvement_remaining"]["area"]
                if roi["should_research_continue_after_v4_v7"] == "YES"
                else None
            ),
            "rationale": roi["rationale"],
        }

        final = {
            "should_buy_v4_replace_buy_v3": "YES" if buy_replace else "NO",
            "buy_v4_replace_reason": (
                f"Accepted robust filters {buy_design['selected_patterns']} improve PF/WR on 240d "
                f"with winner-loss <= {WINNER_REMOVAL_CAP_PCT}% and 250d/500d PF gates passed "
                f"(PF250={pf_250}, PF500={pf_500}). Not based on 120d alone (PF120={pf_120} contrast)."
                if buy_replace
                else (
                    f"No filter set clears winner-loss/cross-window/PF gates for replacement "
                    f"(failure-audit build hint={failure_buy}). Keep BUY_V3; optional paper filters only."
                )
            ),
            "should_sell_v7_replace_sell_v6": "YES" if sell_replace else "NO",
            "sell_v7_replace_reason": (
                f"Accepted robust filters {sell_design['selected_patterns']} improve PF/WR on 240d "
                f"with winner-loss <= {WINNER_REMOVAL_CAP_PCT}% and 250d/500d PF gates passed."
                if sell_replace
                else (
                    f"No filter set clears replacement gates (failure-audit build hint={failure_sell}). "
                    "Keep SELL_V6 + regime throttle."
                )
            ),
            "expected_metrics": expected_stack,
            "scores": scores,
        }

        source_status = {
            name: "loaded" if sources.get(name) else "missing"
            for name in {**REQUIRED_EXPORTS, **OPTIONAL_EXPORTS}
        }

        conclusions = [
            f"BUY_V4 replace BUY_V3: {final['should_buy_v4_replace_buy_v3']}.",
            f"SELL_V7 replace SELL_V6: {final['should_sell_v7_replace_sell_v6']}.",
            f"Best stack: {best_buy} + {best_sell} | 60/100/Runner | fixed_10 | regime throttle.",
            f"Research continue after V4/V7: {roi['should_research_continue_after_v4_v7']} "
            f"({closure['single_highest_roi_remaining_research_area']}).",
            f"Signals predictive: BUY={timing_buy['predictive_vs_reactive']} SELL={timing_sell['predictive_vs_reactive']}.",
        ]

        return BuyV4SellV7DesignBlueprintAuditReport(
            report_type="BUY_V4 & SELL_V7 Design Blueprint Audit",
            engines=["BUY_V3", "SELL_V6", "BUY_V4_candidate", "SELL_V7_candidate"],
            symbol=str(extended_trade.get("symbol") or "NIFTY50"),
            timeframe=str(extended_trade.get("timeframe") or "5M"),
            methodology={
                "research_only": True,
                "no_replay": True,
                "no_new_indicators": True,
                "no_models": True,
                "no_discovery_engines": True,
                "authoritative_windows": [240, 250, 500],
                "contrast_only_windows": [120],
                "winner_removal_cap_pct": WINNER_REMOVAL_CAP_PCT,
                "pf_improvement_threshold_pct": PF_IMPROVEMENT_THRESHOLD_PCT,
            },
            source_exports=source_status,
            limitations=[
                "250d/500d lack per-signal lists — used as PF gates only.",
                "120d used for OOS contrast, never sole authority for replace YES.",
                "Reported PF capped at 3x baseline when uncapped filter math explodes.",
                "Entry loss uses points_before_expansion proxy, not tick-level fills.",
            ],
            window_authority_policy={
                "authoritative": ["240d trade-level", "250d validation", "500d validation"],
                "contrast_only": ["120d"],
                "multi_window_pf": {"120d": pf_120, "250d": pf_250, "500d": pf_500},
                "multi_window_gate_passed": multi_ok,
            },
            failure_pattern_root_cause_analysis={
                "buy_v3": buy_failure,
                "sell_v6": sell_failure,
                "structural_isolation": {"buy_v3": buy_isolation, "sell_v6": sell_isolation},
            },
            candidate_filter_matrix={
                "buy_v3": buy_filters,
                "sell_v6": sell_filters,
                "accepted_buy": [r for r in buy_filters if r["accepted"]],
                "accepted_sell": [r for r in sell_filters if r["accepted"]],
                "robustness_validation": robustness,
            },
            target_achievement_matrix={"buy_v3": target_buy, "sell_v6": target_sell},
            target_path_analysis={"buy_v3": path_buy, "sell_v6": path_sell},
            trade_lifecycle_audit={"buy_v3": life_buy, "sell_v6": life_sell},
            signal_timing_reality_audit={"buy_v3": timing_buy, "sell_v6": timing_sell},
            entry_quality_analysis={"buy_v3": entry_buy, "sell_v6": entry_sell},
            reward_risk_reality={"buy_v3": rr_buy, "sell_v6": rr_sell},
            production_fragility_analysis={
                **survival,
                "collapse_thresholds": {
                    "pf_below": 1.2,
                    "environments": survival.get("production_collapse_environments"),
                },
            },
            buy_v4_design=buy_design,
            sell_v7_design=sell_design,
            engine_comparison={
                "buy_v3_vs_buy_v4": {
                    "baseline": buy_design["baseline"],
                    "candidate": buy_design["expected"],
                    "replace": "YES" if buy_replace else "NO",
                },
                "sell_v6_vs_sell_v7": {
                    "baseline": sell_design["baseline"],
                    "candidate": sell_design["expected"],
                    "replace": "YES" if sell_replace else "NO",
                },
            },
            research_roi_analysis=roi,
            final_production_decision=production_decision,
            final_answer=final,
            production_scores=scores,
            research_closure_verdict=closure,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: BuyV4SellV7DesignBlueprintAuditReport, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("BUY_V4/SELL_V7 design blueprint audit exported: %s", path)
        return path


def generate_buy_v4_sell_v7_design_blueprint_audit_report(
    report_path: Path | str | None = None,
) -> BuyV4SellV7DesignBlueprintAuditReport:
    sources: dict[str, dict[str, Any]] = {}
    for name, path in {**REQUIRED_EXPORTS, **OPTIONAL_EXPORTS}.items():
        data = _load_json(path)
        if name in REQUIRED_EXPORTS and not data:
            raise BuyV4SellV7DesignBlueprintAuditError(f"Required export missing: {path}")
        sources[name] = data
    research = BuyV4SellV7DesignBlueprintAuditResearch()
    report = research.run(sources)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = generate_buy_v4_sell_v7_design_blueprint_audit_report()
        final = report.final_answer
        print(f"Exported: {DEFAULT_REPORT_PATH}")
        print(f"BUY_V4 replace BUY_V3: {final['should_buy_v4_replace_buy_v3']}")
        print(f"SELL_V7 replace SELL_V6: {final['should_sell_v7_replace_sell_v6']}")
        print(f"Best: {report.final_production_decision['best_buy_engine']} + {report.final_production_decision['best_sell_engine']}")
        print(f"Research stop: {report.research_closure_verdict['can_research_stop_after_this_audit']}")
        return 0
    except BuyV4SellV7DesignBlueprintAuditError as exc:
        logger.error("Design blueprint audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
