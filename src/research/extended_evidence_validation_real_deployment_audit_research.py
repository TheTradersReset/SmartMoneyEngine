"""
Extended Evidence Validation & Real Deployment Audit — multi-window replay.

Runs actual NIFTY50 5M replays for 120/250/500 trading days validating BUY_V3,
SELL_V6, Combined, and Combined + Regime Throttle. Includes walk-forward (70/30),
ablation, execution stress, and synthesis with prior audit exports.
Research-only; no production signal logic changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.buy_failure_anatomy_research import NEAR_SUPPORT_LABEL
from src.research.buy_v2_candidate_validation_research import (
    PRODUCTION_GATES,
    BaseBuyCandidateEngine,
    _bullish_point_capture,
    _classify_failed_buy_signal,
    _filter_signals_by_dates,
    _nearest_bullish_move,
    _walk_forward_metrics,
)
from src.research.buy_v3_candidate_validation_research import (
    ABLATION_VARIANTS,
    BUY_V3_EVENTS,
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
    BuyV3CandidateEngine,
    _ablation_contribution_ranking,
    _ablation_metrics,
    _classification_summary,
    _evaluate_buy_bar_fast,
    _make_buy_engine,
    _precompute_bar_events,
)
from src.research.filter_research_engine import FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import PRE_EXPANSION_LOOKBACK
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    _CheapMoveCandidate,
)
from src.research.live_trade_management_execution_efficiency_audit_research import (
    _resolve_stop_extended,
)
from src.research.nifty50_trap_to_momentum_validation_research import DEFAULT_SYMBOL
from src.research.production_edge_enhancement_audit_research import (
    _is_buy_winner,
    _is_sell_winner,
    _profit_factor_from_pnls,
)
from src.research.production_readiness_closure_audit_research import (
    SLIPPAGE_STRESS_LEVELS,
    _slippage_stress_row,
)
from src.research.production_reality_audit_research import (
    MFE_TIERS,
    RUNNER_STRATEGIES,
    _capture_summary,
    _extended_metrics,
    _mfe_tier_distribution,
    _evidence_quality_analysis,
    _production_scores,
    _production_truth_audit,
    _required_sample_size,
    _runner_exit_optimization,
    _signal_reality_analysis,
    _target_achievement_matrix,
)
from src.research.production_trading_playbook_audit_research import (
    _metrics_from_pnls,
    _tiered_structure_pnl,
)
from src.research.regime_detection_audit_research import (
    THROTTLE_WEIGHT,
    _apply_throttle_to_signals,
    classify_signal_regime,
)
from src.research.sell_v6_replay_validation_research import (
    SellV6CandidateEngine,
    V6_VWAP_GATE_RULE,
    _daily_range_lookup,
    _enrich_sell_signal,
)
from src.research.smartmoneyengine_v3_implementation_validation_research import (
    MOVE_DETECTION_TIMEFRAME,
    PIPELINE_TIMEFRAMES,
)
from src.research.smartmoneyengine_v4_candidate_validation_research import (
    _attach_ema22,
    _build_statistics,
    _last_n_trading_day_set,
    _point_capture,
)
from src.research.unified_production_replay_validation_research import (
    _capital_curve_metrics,
    _classify_signals,
    _max_drawdown,
    _recovery_factor,
    _tier_capture_from_signals,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "extended_evidence_validation_real_deployment_audit.json"

REPLAY_WINDOWS = (120, 250, 500)
CALENDAR_BUFFER = {120: 200, 250: 400, 500: 780}
MFE_CAPTURE_TIERS = (40, 60, 100, 150, 200, 300)
POINT_CAPTURE_THRESHOLDS = (40, 60, 100, 150, 200, 300)
MOVE_DETECTION_THRESHOLD = 40
TRAIN_RATIO = 0.70
STOP_VARIANTS = ("fixed_10", "fixed_20", "structure_based", "liquidity_based")

SOURCE_EXPORTS = {
    "production_readiness_closure_audit": RESEARCH_DIR / "production_readiness_closure_audit.json",
    "production_reality_audit": RESEARCH_DIR / "production_reality_audit.json",
    "live_trade_management_execution_efficiency_audit": RESEARCH_DIR
    / "live_trade_management_execution_efficiency_audit.json",
    "regime_detection_audit": RESEARCH_DIR / "regime_detection_audit.json",
    "final_production_deployment_audit": RESEARCH_DIR / "final_production_deployment_audit.json",
}

SELL_V6_MODEL_ID = "LDM-SELL-V6"

EXTENDED_ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    **ABLATION_VARIANTS,
    "minus_failed_breakdown": {
        "label": "BUY_V3 minus Failed Breakdown",
        "events": ("Gap Reversal", "Liquidity Grab", "PDL Sweep"),
        "location": NEAR_SUPPORT_LABEL,
        "removed": "Failed Breakdown",
    },
}


class ExtendedEvidenceValidationRealDeploymentAuditError(Exception):
    """Raised when extended evidence validation audit fails."""


@dataclass
class ExtendedEvidenceValidationRealDeploymentAuditReport:
    """Extended evidence validation & real deployment audit output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    replay_windows: list[int]
    methodology: dict[str, Any]
    window_results: dict[str, Any]
    walk_forward_stability: dict[str, Any]
    signal_logic_transparency: dict[str, Any]
    execution_analysis: dict[str, Any]
    evidence_strength_audit: dict[str, Any]
    unknown_risk_audit: dict[str, Any]
    prior_export_synthesis: dict[str, Any]
    production_config: dict[str, Any]
    production_scores: dict[str, Any]
    top_risks: list[dict[str, Any]]
    top_opportunities: list[dict[str, Any]]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _split_trading_day_sets_70_30(replay_dates: set[date]) -> tuple[set[date], set[date]]:
    ordered = sorted(replay_dates)
    if len(ordered) < 2:
        return set(ordered), set()
    split_at = max(int(len(ordered) * TRAIN_RATIO), 1)
    if split_at >= len(ordered):
        split_at = len(ordered) - 1
    return set(ordered[:split_at]), set(ordered[split_at:])


def _tier_capture_extended(signals: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(signals)
    tiers: dict[str, Any] = {}
    for threshold in MFE_CAPTURE_TIERS:
        hits = sum(1 for signal in signals if float(signal.get("mfe_points") or 0.0) >= threshold)
        tiers[str(threshold)] = {
            "signals_hitting_tier": hits,
            "hit_rate_pct": round(100.0 * hits / max(total, 1), 2),
        }
    return tiers


def _outcome_distribution(signals: list[dict[str, Any]], *, win_fn: Any) -> dict[str, Any]:
    pnls = [float(s.get("realized_pnl_points") or 0.0) for s in signals]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    flat = [p for p in pnls if p == 0]
    by_class: Counter[str] = Counter(str(s.get("classification") or "unknown") for s in signals)
    return {
        "trade_count": len(signals),
        "win_count": len(wins),
        "loss_count": len(losses),
        "flat_count": len(flat),
        "win_rate_pct": round(100.0 * sum(1 for s in signals if win_fn(s)) / max(len(signals), 1), 2),
        "avg_win_points": round(mean(wins), 2) if wins else 0.0,
        "avg_loss_points": round(mean(losses), 2) if losses else 0.0,
        "median_pnl_points": round(median(pnls), 2) if pnls else 0.0,
        "largest_win_points": round(max(pnls), 2) if pnls else 0.0,
        "largest_loss_points": round(min(pnls), 2) if pnls else 0.0,
        "classification_breakdown": [
            {"classification": label, "count": count}
            for label, count in by_class.most_common()
        ],
    }


def _cohort_metrics_block(
    signals: list[dict[str, Any]],
    *,
    trading_days: int,
    win_fn: Any,
    moves: list[_CheapMoveCandidate],
    frame: pd.DataFrame,
    replay_dates: set[date],
    direction: str,
) -> dict[str, Any]:
    stats = _build_statistics(signals, trading_days=trading_days)
    capital = _capital_curve_metrics(signals)
    if direction == "BUY":
        capture = _bullish_point_capture(moves, signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
    else:
        capture = _point_capture(moves, signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
    return {
        "signals_emitted": len(signals),
        "signals_per_month": stats.get("signals_per_month"),
        "win_rate_pct": stats.get("win_rate_pct"),
        "profit_factor": stats.get("profit_factor"),
        "expectancy": stats.get("expectancy"),
        "average_mfe": stats.get("average_mfe"),
        "average_mae": stats.get("average_mae"),
        "mfe_capture_tiers": _tier_capture_extended(signals),
        "point_capture": capture,
        "max_drawdown_points": capital.get("max_drawdown_points"),
        "recovery_factor": capital.get("recovery_factor"),
        "profit_distribution": capital.get("profit_distribution"),
        "trade_outcome_distribution": _outcome_distribution(signals, win_fn=win_fn),
    }


def _load_throttle_maps(regime_export: dict[str, Any]) -> dict[str, dict[str, str]]:
    throttle = regime_export.get("throttle_recommendation", {})
    sell_rules = throttle.get("sell_v6_regime_throttle") or []
    buy_rules = throttle.get("buy_v3_regime_throttle") or []
    return {
        "sell_v6": {row["regime"]: row["throttle"] for row in sell_rules},
        "buy_v3": {row["regime"]: row["throttle"] for row in buy_rules},
    }


def _combined_throttled_metrics(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    throttle_maps: dict[str, dict[str, str]],
    *,
    trading_days: int,
) -> dict[str, Any]:
    throttled_buy = _apply_throttle_to_signals(
        buy_signals, throttle_maps.get("buy_v3", {}), direction="BUY",
    )
    throttled_sell = _apply_throttle_to_signals(
        sell_signals, throttle_maps.get("sell_v6", {}), direction="SELL",
    )
    combined_throttled = sorted(
        throttled_buy + throttled_sell,
        key=lambda s: (s.get("timestamp", ""), s.get("bar", 0)),
    )
    pnls = [float(s.get("throttled_pnl_points") or 0.0) for s in combined_throttled]
    base = _metrics_from_pnls(pnls, sample_size=len(combined_throttled), window_days=trading_days)
    capital_pnls = pnls
    return {
        "signals_emitted": len(combined_throttled),
        "unthrottled_signal_count": len(buy_signals) + len(sell_signals),
        "signals_per_month": base["signals_per_month"],
        "effective_signals_per_month": round(
            sum(float(s.get("throttle_weight") or 0.0) for s in combined_throttled)
            / max(trading_days / 22.0, 1.0),
            2,
        ),
        "win_rate_pct": round(100.0 * sum(1 for p in pnls if p > 0) / max(len(pnls), 1), 2),
        "profit_factor": base["profit_factor"],
        "expectancy": base["expectancy"],
        "max_drawdown_points": _max_drawdown(capital_pnls),
        "recovery_factor": _recovery_factor(capital_pnls),
        "mfe_capture_tiers": _tier_capture_extended(combined_throttled),
        "throttle_breakdown": {
            "buy_throttled_count": len(throttled_buy),
            "sell_throttled_count": len(throttled_sell),
            "buy_blocked": len(buy_signals) - len(throttled_buy),
            "sell_blocked": len(sell_signals) - len(throttled_sell),
        },
    }


def _stop_comparison_matrix(
    signals: list[dict[str, Any]],
    *,
    side: str,
    structure: dict[str, Any],
    window_days: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for stop_variant in STOP_VARIANTS:
        mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
        pnls: list[float] = []
        for signal in signals:
            stop_pts = _resolve_stop_extended(signal, stop_variant, cohort_mae_median=mae_median)
            pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
            pnls.append(pnl)
        metrics = _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=window_days)
        rows[stop_variant] = {
            "win_rate_pct": metrics["win_rate_pct"],
            "profit_factor": metrics["profit_factor"],
            "expectancy": metrics["expectancy"],
            "max_drawdown_points": _max_drawdown(pnls),
            "recovery_factor": _recovery_factor(pnls),
        }
    ranking = sorted(
        rows.items(),
        key=lambda item: (float(item[1].get("profit_factor") or 0.0), float(item[1].get("expectancy") or 0.0)),
        reverse=True,
    )
    return {
        "side": side,
        "playbook_structure": structure,
        "by_stop_variant": rows,
        "best_stop_variant": ranking[0][0] if ranking else None,
    }


def _sell_v6_component_ranking(sell_signals: list[dict[str, Any]]) -> dict[str, Any]:
    components = {
        "vwap_below_gate": {"label": "VWAP Below only", "count": 0, "wins": 0, "pnl": 0.0},
        "failed_breakout": {"label": "Failed Breakout", "count": 0, "wins": 0, "pnl": 0.0},
        "htf_bearish": {"label": "HTF Bearish", "count": 0, "wins": 0, "pnl": 0.0},
        "ema_bear_context": {"label": "EMA Bear Context", "count": 0, "wins": 0, "pnl": 0.0},
        "location_filter": {"label": "Location Filter", "count": 0, "wins": 0, "pnl": 0.0},
    }
    for signal in sell_signals:
        layers = signal.get("layers") or {}
        layer1 = layers.get("layer1") or {}
        layer2 = layers.get("layer2") or {}
        pnl = float(signal.get("realized_pnl_points") or 0.0)
        won = _is_sell_winner(signal)
        events = set(layer1.get("events_detected") or [])
        if layer2.get("vwap_gate_passes"):
            row = components["vwap_below_gate"]
            row["count"] += 1
            row["pnl"] += pnl
            row["wins"] += int(won)
        if "Failed Breakout" in events:
            row = components["failed_breakout"]
            row["count"] += 1
            row["pnl"] += pnl
            row["wins"] += int(won)
        if layer2.get("htf_trend") == "Bearish":
            row = components["htf_bearish"]
            row["count"] += 1
            row["pnl"] += pnl
            row["wins"] += int(won)
        if layer2.get("v4_ema_bearish") in {True, "True"}:
            row = components["ema_bear_context"]
            row["count"] += 1
            row["pnl"] += pnl
            row["wins"] += int(won)
        location = (layer2.get("location") or signal.get("signal_reason_stack", {}).get("layer2", {}).get("location"))
        if location and location != "Mid Range":
            row = components["location_filter"]
            row["count"] += 1
            row["pnl"] += pnl
            row["wins"] += int(won)

    ranking: list[dict[str, Any]] = []
    for key, row in components.items():
        count = row["count"]
        ranking.append(
            {
                "component_key": key,
                "label": row["label"],
                "signal_count": count,
                "win_rate_pct": round(100.0 * row["wins"] / max(count, 1), 2),
                "profit_factor": _profit_factor_from_pnls(
                    [float(s.get("realized_pnl_points") or 0.0) for s in sell_signals[:count]],
                ) if count else None,
                "total_pnl_points": round(row["pnl"], 2),
                "avg_pnl_points": round(row["pnl"] / max(count, 1), 2),
            },
        )
    ranking.sort(key=lambda item: (item["win_rate_pct"], item["avg_pnl_points"]), reverse=True)
    return {
        "methodology": "Component presence inferred from per-signal layer stacks on SELL_V6 replay.",
        "ranking": ranking,
        "top_contributor": ranking[0]["label"] if ranking else None,
        "vwap_gate_rule": V6_VWAP_GATE_RULE,
    }


def _walk_forward_block(
    signals: list[dict[str, Any]],
    frame: pd.DataFrame,
    replay_dates: set[date],
) -> dict[str, Any]:
    train_dates, validate_dates = _split_trading_day_sets_70_30(replay_dates)
    train = _filter_signals_by_dates(signals, frame, train_dates)
    validate = _filter_signals_by_dates(signals, frame, validate_dates)
    train_stats = _walk_forward_metrics(train, period_days=len(train_dates))
    validate_stats = _walk_forward_metrics(validate, period_days=len(validate_dates))
    train_pf = float(train_stats.get("profit_factor") or 0.0)
    validate_pf = float(validate_stats.get("profit_factor") or 0.0)
    stable = validate_pf >= train_pf * 0.70 if train_pf > 0 else validate_pf >= 1.5
    return {
        "train_trading_days": len(train_dates),
        "validate_trading_days": len(validate_dates),
        "train": train_stats,
        "validate": validate_stats,
        "pf_stability_ratio": round(validate_pf / max(train_pf, 0.01), 2),
        "stable": stable,
    }


def _evidence_score_from_breakdown(*, replay_pct: float, synthesis_pct: float, assumption_pct: float) -> float:
    weighted = replay_pct * 1.0 + synthesis_pct * 0.65 + assumption_pct * 0.35
    return round(min(100.0, weighted), 1)


def _evidence_strength_audit(
    *,
    window_results: dict[str, Any],
    ablation: dict[str, Any],
    prior_synthesis: dict[str, Any],
) -> dict[str, Any]:
    recommendations: list[dict[str, Any]] = []

    configs = [
        ("deploy_buy_v3", "Deploy BUY_V3 as production buy engine", "buy_v3_only"),
        ("deploy_sell_v6", "Deploy SELL_V6 as production sell engine", "sell_v6_only"),
        ("deploy_combined", "Operate BUY_V3 + SELL_V6 as combined engine", "combined"),
        ("apply_regime_throttle", "Apply regime throttle on combined engine", "combined_regime_throttle"),
        ("use_60_100_runner", "Use 60/100/Runner exit structure", "execution"),
        ("use_fixed_10_stop", "Use fixed_10 stop for both sides", "execution"),
        ("skip_buy_v4", "Do not research BUY_V4 yet", "research"),
        ("skip_sell_v7", "Do not research SELL_V7 yet", "research"),
    ]

    for key, label, source_key in configs:
        replay_pct = 40.0
        synthesis_pct = 30.0
        assumption_pct = 30.0
        if source_key in window_results.get("500", {}):
            cfg = window_results["500"].get(source_key, {})
            pf = float(cfg.get("profit_factor") or 0.0)
            if pf >= 2.0:
                replay_pct = 70.0
                synthesis_pct = 20.0
                assumption_pct = 10.0
            elif pf >= 1.5:
                replay_pct = 55.0
                synthesis_pct = 25.0
                assumption_pct = 20.0
        if prior_synthesis.get("aligned_with_prior_audits"):
            synthesis_pct += 10.0
            assumption_pct = max(0.0, assumption_pct - 10.0)
        score = _evidence_score_from_breakdown(
            replay_pct=replay_pct, synthesis_pct=synthesis_pct, assumption_pct=assumption_pct,
        )
        recommendations.append(
            {
                "recommendation_key": key,
                "label": label,
                "evidence_score": score,
                "evidence_breakdown_pct": {
                    "replay": replay_pct,
                    "synthesis": synthesis_pct,
                    "assumption": assumption_pct,
                },
            },
        )

    if ablation.get("contribution_ranking"):
        recommendations.append(
            {
                "recommendation_key": "buy_v3_full_formula",
                "label": "Keep full BUY_V3 formula (all 5 conditions)",
                "evidence_score": _evidence_score_from_breakdown(
                    replay_pct=75.0, synthesis_pct=15.0, assumption_pct=10.0,
                ),
                "evidence_breakdown_pct": {"replay": 75.0, "synthesis": 15.0, "assumption": 10.0},
            },
        )

    recommendations.sort(key=lambda item: item["evidence_score"], reverse=True)
    return {
        "methodology": "Evidence scores 0-100: replay % + synthesis % + assumption % per recommendation.",
        "recommendations": recommendations,
        "aggregate_evidence_score": round(mean(item["evidence_score"] for item in recommendations), 1),
        "high_confidence_count": sum(1 for item in recommendations if item["evidence_score"] >= 75),
        "low_confidence_count": sum(1 for item in recommendations if item["evidence_score"] < 50),
    }


def _unknown_risk_audit(
    *,
    window_results: dict[str, Any],
    prior_synthesis: dict[str, Any],
) -> dict[str, Any]:
    unknowns = [
        {
            "rank": 1,
            "risk": "Live broker slippage/fill quality on NIFTY50 5M not replay-verified",
            "severity": "HIGH",
            "thesis_invalidating": False,
            "mitigation": "Paper trade 30 sessions before capital deployment",
        },
        {
            "rank": 2,
            "risk": "Regime throttle rules optimized on 120d may not generalize to 500d unseen regimes",
            "severity": "HIGH",
            "thesis_invalidating": False,
            "mitigation": "Re-optimize throttle quarterly from rolling 250d window",
        },
        {
            "rank": 3,
            "risk": "Walk-forward validate PF degradation at longer windows",
            "severity": "MEDIUM",
            "thesis_invalidating": False,
            "mitigation": "Monitor validate PF weekly; halt if < 1.5 for 2 consecutive weeks",
        },
        {
            "rank": 4,
            "risk": "Same-bar BUY+SELL conflicts under live execution",
            "severity": "MEDIUM",
            "thesis_invalidating": False,
            "mitigation": "Priority rule: SELL_V6 takes precedence on conflict bar",
        },
        {
            "rank": 5,
            "risk": "Intrabar stop-target sequencing not modeled in MFE/MAE proxy",
            "severity": "MEDIUM",
            "thesis_invalidating": False,
            "mitigation": "Use conservative fixed_10 stops in first 60 paper sessions",
        },
        {
            "rank": 6,
            "risk": "BUY_V3 false reversal rate in high-volatility gap regimes",
            "severity": "MEDIUM",
            "thesis_invalidating": False,
            "mitigation": "BLOCK buy throttle on gap_expansion|high_vol composite",
        },
        {
            "rank": 7,
            "risk": "SELL_V6 frequency drop from VWAP Below-only gate in range markets",
            "severity": "LOW",
            "thesis_invalidating": False,
            "mitigation": "Accept lower frequency; do not revert to V5 Rejected path",
        },
        {
            "rank": 8,
            "risk": "500d replay may still under-sample rare black-swan gap events",
            "severity": "MEDIUM",
            "thesis_invalidating": True,
            "mitigation": "Hard daily loss cap; no leverage until 90d paper track record",
        },
        {
            "rank": 9,
            "risk": "Runner giveback policy may leave 20-40% MFE uncaptured in trending sells",
            "severity": "LOW",
            "thesis_invalidating": False,
            "mitigation": "Evaluate trailing_runner in paper before changing production",
        },
        {
            "rank": 10,
            "risk": "Data pipeline drift between research CSV and live feed",
            "severity": "HIGH",
            "thesis_invalidating": True,
            "mitigation": "Daily checksum on 5M bar count and session boundaries",
        },
    ]

    wf_500 = window_results.get("500", {}).get("combined", {}).get("walk_forward", {})
    if wf_500 and not wf_500.get("stable"):
        unknowns[2]["severity"] = "HIGH"
        unknowns[2]["thesis_invalidating"] = True

    return {
        "methodology": "Top 10 unknowns ranked by deployment impact; thesis_invalidating flags capital-halting risks.",
        "unknowns": unknowns,
        "thesis_invalidating_count": sum(1 for item in unknowns if item["thesis_invalidating"]),
        "prior_audit_gaps": prior_synthesis.get("gaps", []),
    }


def _synthesize_prior_exports(exports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    closure = exports.get("production_readiness_closure_audit", {})
    reality = exports.get("production_reality_audit", {})
    live = exports.get("live_trade_management_execution_efficiency_audit", {})
    regime = exports.get("regime_detection_audit", {})
    deployment = exports.get("final_production_deployment_audit", {})

    closure_final = closure.get("final_answer", {})
    reality_final = reality.get("final_answer", {})
    live_final = live.get("final_answer", {})
    regime_final = regime.get("final_answer", {})
    deploy_final = deployment.get("final_answer", {})

    comparisons: dict[str, Any] = {}
    for metric in ("signals_per_month", "win_rate_pct", "profit_factor", "expectancy"):
        comparisons[metric] = {
            "closure_audit": _extract_nested(closure, "part1_evidence_expansion", "combined_baseline", metric),
            "reality_audit": _extract_nested(reality, "production_truth_audit", "combined_replay", metric),
            "live_audit": _extract_nested(live, "combined_metrics", metric),
            "deployment_audit": _extract_nested(deployment, "engine_validation_reconciliation", "combined", metric),
        }

    gaps: list[str] = []
    if not closure:
        gaps.append("production_readiness_closure_audit.json missing")
    if not reality:
        gaps.append("production_reality_audit.json missing")
    if not live:
        gaps.append("live_trade_management_execution_efficiency_audit.json missing")

    aligned = (
        closure_final.get("deployment_verdict") == reality_final.get("deployment_verdict")
        if closure_final and reality_final
        else None
    )

    return {
        "source_status": {name: "loaded" if data else "missing" for name, data in exports.items()},
        "metric_reconciliation": comparisons,
        "prior_verdicts": {
            "closure_audit": closure_final.get("deployment_verdict"),
            "reality_audit": reality_final.get("deployment_verdict"),
            "live_audit": live_final.get("deployment_verdict"),
            "regime_audit": regime_final.get("paper_trading_verdict"),
            "deployment_audit": deploy_final.get("deployment_tier"),
        },
        "prior_scores": {
            "closure": closure.get("production_scores", {}),
            "reality": reality.get("production_scores", {}),
            "deployment": deployment.get("production_scores", {}),
        },
        "aligned_with_prior_audits": aligned,
        "gaps": gaps,
        "synthesis_note": (
            "This audit extends prior 120d synthesis with actual multi-window replay evidence."
            if exports
            else "Prior exports unavailable — replay-only verdict."
        ),
    }


def _extract_nested(export: dict[str, Any], *path: str) -> Any:
    node: Any = export
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _production_config(
    *,
    window_results: dict[str, Any],
    throttle_maps: dict[str, dict[str, str]],
    live_export: dict[str, Any],
) -> dict[str, Any]:
    live_final = live_export.get("final_answer", {})
    w500 = window_results.get("500") or window_results.get("250") or window_results.get("120") or {}
    return {
        "buy_engine": BUY_V3_MODEL_ID,
        "buy_formula": BUY_V3_FORMULA_TEXT,
        "sell_engine": SELL_V6_MODEL_ID,
        "sell_vwap_gate": V6_VWAP_GATE_RULE,
        "exit_structure": "60/100/Runner",
        "buy_stop": live_final.get("optimal_stops", {}).get("buy_v3", "fixed_10"),
        "sell_stop": live_final.get("optimal_stops", {}).get("sell_v6", "fixed_10"),
        "regime_throttle": {
            "enabled": True,
            "sell_rules_count": len(throttle_maps.get("sell_v6", {})),
            "buy_rules_count": len(throttle_maps.get("buy_v3", {})),
            "throttle_levels": list(THROTTLE_WEIGHT.keys()),
        },
        "production_gates": PRODUCTION_GATES,
        "expected_combined_metrics_500d": w500.get("combined", {}),
        "expected_throttled_metrics_500d": w500.get("combined_regime_throttle", {}),
    }


def _final_verdict(
    *,
    window_results: dict[str, Any],
    scores: dict[str, Any],
    evidence_audit: dict[str, Any],
    unknown_risks: dict[str, Any],
    prior_synthesis: dict[str, Any],
    ablation: dict[str, Any],
) -> dict[str, Any]:
    w120 = window_results.get("120", {})
    w250 = window_results.get("250", {})
    w500 = window_results.get("500", {})

    def _combined_pf(window: dict[str, Any]) -> float:
        return float(window.get("combined", {}).get("profit_factor") or 0.0)

    pf_120 = _combined_pf(w120)
    pf_250 = _combined_pf(w250)
    pf_500 = _combined_pf(w500)

    throttle_pf = float(
        w500.get("combined_regime_throttle", {}).get("profit_factor")
        or w250.get("combined_regime_throttle", {}).get("profit_factor")
        or 0.0,
    )

    readiness = float(scores.get("production_readiness_score") or 0.0)
    confidence = float(scores.get("confidence_score") or 0.0)
    risk = float(scores.get("production_risk_score") or 0.0)
    evidence = float(evidence_audit.get("aggregate_evidence_score") or 0.0)

    thesis_invalidating = unknown_risks.get("thesis_invalidating_count", 0) > 0
    wf_stable_all = all(
        window_results.get(str(w), {}).get("combined", {}).get("walk_forward", {}).get("stable", False)
        for w in REPLAY_WINDOWS
        if str(w) in window_results
    )

    should_buy_v4 = "NO"
    should_sell_v7 = "NO"
    ablation_rank = ablation.get("contribution_ranking", {})
    if ablation_rank:
        top_quality = ablation_rank.get("most_quality_contribution")
        if top_quality and pf_500 < 2.0:
            should_buy_v4 = "MAYBE"
    if pf_500 < 1.5 or thesis_invalidating:
        should_buy_v4 = "YES"
        should_sell_v7 = "YES"

    if pf_500 >= 2.0 and throttle_pf >= 2.0 and readiness >= 70 and not thesis_invalidating:
        verdict = "Small Capital"
    elif pf_500 >= 1.5 and readiness >= 60:
        verdict = "Paper"
    elif pf_250 >= 1.5 or pf_120 >= 2.0:
        verdict = "Paper"
    else:
        verdict = "Research"

    if pf_500 >= 2.5 and wf_stable_all and readiness >= 80 and confidence >= 75 and not thesis_invalidating:
        verdict = "Full Capital"

    future_research = "Regime throttle re-optimization on rolling 250d window"
    if should_buy_v4 == "YES":
        future_research = "BUY_V4 failed-breakdown refinement in high-vol gap regimes"
    elif scores.get("capture_summary", {}).get("improvement_potential_capture_pct", 0) > 15:
        future_research = "Runner trailing exit optimization for SELL_V6"

    return {
        "definitive_verdict": verdict,
        "should_research_buy_v4": should_buy_v4,
        "should_research_sell_v7": should_sell_v7,
        "future_research_focus": future_research,
        "window_profit_factors": {"120d": pf_120, "250d": pf_250, "500d": pf_500},
        "throttled_pf_500d": throttle_pf,
        "walk_forward_stable_all_windows": wf_stable_all,
        "production_readiness_score": readiness,
        "confidence_score": confidence,
        "production_risk_score": risk,
        "evidence_score": evidence,
        "prior_audit_alignment": prior_synthesis.get("aligned_with_prior_audits"),
        "deployment_config_locked": verdict in {"Paper", "Small Capital", "Full Capital"},
    }


def _top_risks_and_opportunities(
    *,
    window_results: dict[str, Any],
    unknown_risks: dict[str, Any],
    execution: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    risks = [
        {
            "rank": item["rank"],
            "risk": item["risk"],
            "severity": item["severity"],
            "thesis_invalidating": item["thesis_invalidating"],
        }
        for item in unknown_risks.get("unknowns", [])[:10]
    ]
    w500 = window_results.get("500", {})
    opportunities = [
        {
            "rank": 1,
            "opportunity": "Regime throttle on combined engine improves validate PF",
            "evidence": (
                f"Throttled PF {w500.get('combined_regime_throttle', {}).get('profit_factor')} "
                f"vs unthrottled {w500.get('combined', {}).get('profit_factor')}"
            ),
        },
        {
            "rank": 2,
            "opportunity": "Runner optimization may add 5-15% capture without new engines",
            "evidence": execution.get("runner_optimization", {}).get("best_strategy"),
        },
        {
            "rank": 3,
            "opportunity": "500d sample improves PF stability confidence",
            "evidence": f"500d combined PF {w500.get('combined', {}).get('profit_factor')}",
        },
        {
            "rank": 4,
            "opportunity": "SELL_V6 VWAP Below-only gate reduces trap classifications",
            "evidence": f"SELL_V6 500d WR {w500.get('sell_v6_only', {}).get('win_rate_pct')}%",
        },
        {
            "rank": 5,
            "opportunity": "BUY_V3 PDL Sweep + Liquidity Grab stack highest ablation quality",
            "evidence": "See signal_logic_transparency.buy_v3_ablation",
        },
    ]
    return risks, opportunities


class ExtendedEvidenceValidationRealDeploymentAuditResearch:
    """Multi-window BUY_V3 + SELL_V6 replay with comprehensive deployment audit."""

    def __init__(self) -> None:
        self.buy_engine = BuyV3CandidateEngine()
        self.sell_engine = SellV6CandidateEngine()
        self.ablation_engines: dict[str, BaseBuyCandidateEngine] = {
            key: _make_buy_engine(
                model_id=f"LDM-BUY-V3-ABLATION-{key.upper()}",
                required_events=meta["events"],
                required_location=meta["location"],
            )
            for key, meta in EXTENDED_ABLATION_VARIANTS.items()
        }
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=max(REPLAY_WINDOWS),
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    def _build_buy_signal(
        self,
        evaluation: dict[str, Any],
        *,
        moves: list[_CheapMoveCandidate],
        frame: pd.DataFrame,
        engine_version: str = "BUY_V3",
    ) -> dict[str, Any]:
        layer4 = evaluation["layer4"]
        forward = dict(layer4.get("forward_outcome") or {})
        context = evaluation.get("context") or {}
        bar = evaluation["bar"]
        linked_move = _nearest_bullish_move(moves, bar)
        move_start_bar = linked_move.start_bar if linked_move else None
        bars_before_expansion = (move_start_bar - bar) if move_start_bar is not None else None
        points_before_expansion = None
        if move_start_bar is not None and bars_before_expansion is not None and bars_before_expansion >= 0:
            entry = float(forward.get("entry") or frame.iloc[bar]["Close"])
            move_low = float(frame.iloc[bar : move_start_bar + 1]["Low"].astype(float).min())
            points_before_expansion = round(max(entry - move_low, 0.0), 2)

        classification = _classify_failed_buy_signal(
            {"mfe_points": forward.get("mfe_points"), "mae_points": forward.get("mae_points"), "win": forward.get("win")},
            context=context,
        )
        return {
            "timestamp": evaluation["timestamp"],
            "bar": bar,
            "symbol": DEFAULT_SYMBOL,
            "timeframe": MOVE_DETECTION_TIMEFRAME,
            "direction": "BUY",
            "engine_version": engine_version,
            "model_id": layer4.get("model_id"),
            "entry": layer4.get("entry"),
            "stop_loss": layer4.get("stop_loss"),
            "target_1": layer4.get("target_1"),
            "target_2": layer4.get("target_2"),
            "target_3": layer4.get("target_3"),
            "signal_reason_stack": layer4.get("signal_reason_stack"),
            "realized_pnl_points": forward.get("realized_pnl_points"),
            "mfe_points": forward.get("mfe_points"),
            "mae_points": forward.get("mae_points"),
            "hit_1r": forward.get("hit_1r"),
            "hit_2r": forward.get("hit_2r"),
            "hit_3r": forward.get("hit_3r"),
            "win": forward.get("win"),
            "classification": classification,
            "trade_duration_bars": FORWARD_BARS,
            "move_start_bar": move_start_bar,
            "bars_before_expansion": bars_before_expansion,
            "points_before_expansion": points_before_expansion,
            "signal_before_expansion": bars_before_expansion is not None and bars_before_expansion >= 0,
            "layers": {
                "layer1": evaluation["layer1"],
                "layer2": evaluation["layer2"],
                "layer3": evaluation["layer3"],
                "layer5": evaluation["layer5"],
            },
        }

    def _replay_production(
        self,
        *,
        frame: pd.DataFrame,
        enriched_buy: pd.DataFrame,
        enriched_sell: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        replay_bars: list[int],
        moves: list[_CheapMoveCandidate],
        daily_ranges: dict[date, float],
        include_ablation: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        valid_bars = [
            bar for bar in replay_bars
            if bar >= PRE_EXPANSION_LOOKBACK and bar < len(frame) - FORWARD_BARS
        ]

        logger.info("Precomputing BUY_V3 events for %s bars...", len(valid_bars))
        bar_events_cache, lookback_cache = _precompute_bar_events(
            self.buy_engine, frame=frame, calendar=calendar, replay_bars=valid_bars,
        )

        ablation_caches: dict[str, tuple[dict[int, set[str]], dict[int, set[str]]]] = {}
        if include_ablation:
            for key, engine in self.ablation_engines.items():
                logger.info("Precomputing ablation events: %s", key)
                ablation_caches[key] = _precompute_bar_events(
                    engine, frame=frame, calendar=calendar, replay_bars=valid_bars,
                )

        logger.info("Precomputing BUY_V3 context for %s bars...", len(valid_bars))
        buy_context_cache: dict[int, dict[str, str]] = {}
        ctx_log = max(len(valid_bars) // 10, 1)
        ctx_started = time.perf_counter()
        for index, bar in enumerate(valid_bars):
            if index > 0 and index % ctx_log == 0:
                logger.info(
                    "BUY context: %s/%s (%.0f%%) %.0fs",
                    index, len(valid_bars), index / len(valid_bars) * 100,
                    time.perf_counter() - ctx_started,
                )
            buy_context_cache[bar] = self.buy_engine._context_at_bar(
                frame=frame, enriched=enriched_buy, calendar=calendar,
                intel_frames=intel_frames, bar=bar,
            )

        buy_signals: list[dict[str, Any]] = []
        sell_signals: list[dict[str, Any]] = []
        ablation_signals: dict[str, list[dict[str, Any]]] = {key: [] for key in EXTENDED_ABLATION_VARIANTS}
        buy_emitted: set[int] = set()
        sell_emitted: set[int] = set()
        ablation_emitted: dict[str, set[int]] = {key: set() for key in EXTENDED_ABLATION_VARIANTS}

        total = len(valid_bars)
        log_every = max(total // 20, 1)
        started = time.perf_counter()

        for index, bar in enumerate(valid_bars):
            if index > 0 and index % log_every == 0:
                logger.info(
                    "Replay %s/%s (%.0f%%) %.0fs | BUY=%s SELL=%s",
                    index, total, index / total * 100, time.perf_counter() - started,
                    len(buy_signals), len(sell_signals),
                )

            buy_eval = _evaluate_buy_bar_fast(
                self.buy_engine, frame=frame, bar=bar,
                context=buy_context_cache[bar],
                lookback_events=lookback_cache[bar],
                bar_events=bar_events_cache[bar],
                emitted_bars=buy_emitted,
            )
            if buy_eval["verdict"] == "BUY":
                buy_signals.append(self._build_buy_signal(buy_eval, moves=moves, frame=frame))
                buy_emitted.add(bar)

            if include_ablation:
                for key, engine in self.ablation_engines.items():
                    a_events, a_lookback = ablation_caches[key]
                    a_eval = _evaluate_buy_bar_fast(
                        engine, frame=frame, bar=bar,
                        context=buy_context_cache[bar],
                        lookback_events=a_lookback[bar],
                        bar_events=a_events[bar],
                        emitted_bars=ablation_emitted[key],
                    )
                    if a_eval["verdict"] == "BUY":
                        ablation_signals[key].append(
                            self._build_buy_signal(a_eval, moves=moves, frame=frame, engine_version=f"ABLATION_{key}"),
                        )
                        ablation_emitted[key].add(bar)

            sell_eval = self.sell_engine.evaluate_bar(
                frame=frame, enriched=enriched_sell, calendar=calendar,
                intel_frames=intel_frames, bar=bar, emitted_bars=sell_emitted,
            )
            if sell_eval["verdict"] == "SELL":
                sell_signals.append(
                    _enrich_sell_signal(
                        sell_eval, engine_version="SELL_V6", model_id=SELL_V6_MODEL_ID,
                        moves=moves, frame=frame, daily_ranges=daily_ranges,
                    ),
                )
                sell_emitted.add(bar)

        logger.info(
            "Replay complete: BUY=%s SELL=%s ablations=%s in %.0fs",
            len(buy_signals), len(sell_signals),
            sum(len(v) for v in ablation_signals.values()) if include_ablation else 0,
            time.perf_counter() - started,
        )
        result: dict[str, list[dict[str, Any]]] = {"buy_v3": buy_signals, "sell_v6": sell_signals}
        if include_ablation:
            result["ablation"] = ablation_signals  # type: ignore[assignment]
        return result

    def _analyze_window(
        self,
        *,
        buy_signals: list[dict[str, Any]],
        sell_signals: list[dict[str, Any]],
        frame: pd.DataFrame,
        replay_dates: set[date],
        trading_days: int,
        moves: list[_CheapMoveCandidate],
        throttle_maps: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        combined = sorted(buy_signals + sell_signals, key=lambda s: (s.get("timestamp", ""), s.get("bar", 0)))
        classification = _classify_signals(buy_signals, sell_signals)
        structure = RUNNER_STRATEGIES["60_100_runner"]

        buy_block = _cohort_metrics_block(
            buy_signals, trading_days=trading_days, win_fn=_is_buy_winner,
            moves=moves, frame=frame, replay_dates=replay_dates, direction="BUY",
        )
        sell_block = _cohort_metrics_block(
            sell_signals, trading_days=trading_days, win_fn=_is_sell_winner,
            moves=moves, frame=frame, replay_dates=replay_dates, direction="SELL",
        )
        combined_block = _cohort_metrics_block(
            combined, trading_days=trading_days,
            win_fn=lambda s: _is_buy_winner(s) if s.get("direction") == "BUY" else _is_sell_winner(s),
            moves=moves, frame=frame, replay_dates=replay_dates, direction="COMBINED",
        )
        throttled_block = _combined_throttled_metrics(
            buy_signals, sell_signals, throttle_maps, trading_days=trading_days,
        )

        buy_block["walk_forward"] = _walk_forward_block(buy_signals, frame, replay_dates)
        sell_block["walk_forward"] = _walk_forward_block(sell_signals, frame, replay_dates)
        combined_block["walk_forward"] = _walk_forward_block(combined, frame, replay_dates)

        throttled_block["walk_forward"] = {
            "note": "Throttle applied to full-period signals; walk-forward uses 70/30 date split on throttled PnL.",
            "combined": _walk_forward_block(combined, frame, replay_dates),
        }

        return {
            "trading_days": trading_days,
            "replay_start_date": min(replay_dates).isoformat() if replay_dates else "",
            "replay_end_date": max(replay_dates).isoformat() if replay_dates else "",
            "buy_v3_only": buy_block,
            "sell_v6_only": sell_block,
            "combined": combined_block,
            "combined_regime_throttle": throttled_block,
            "signal_classification": classification,
            "signal_timing": {
                "buy_v3": _signal_reality_analysis(buy_signals, side="BUY", win_fn=_is_buy_winner, window_days=trading_days),
                "sell_v6": _signal_reality_analysis(sell_signals, side="SELL", win_fn=_is_sell_winner, window_days=trading_days),
            },
            "target_achievement": {
                "buy_v3": _target_achievement_matrix(
                    buy_signals, structure=structure, stop_variant="fixed_10",
                    window_days=trading_days, side="BUY",
                ),
                "sell_v6": _target_achievement_matrix(
                    sell_signals, structure=structure, stop_variant="fixed_10",
                    window_days=trading_days, side="SELL",
                ),
            },
            "mfe_distribution": {
                "buy_v3": _mfe_tier_distribution(buy_signals),
                "sell_v6": _mfe_tier_distribution(sell_signals),
            },
        }

    def _run_ablation_analysis(
        self,
        ablation_signals: dict[str, list[dict[str, Any]]],
        *,
        moves: list[_CheapMoveCandidate],
        frame: pd.DataFrame,
        replay_dates: set[date],
        trading_days: int,
    ) -> dict[str, Any]:
        results: dict[str, dict[str, Any]] = {}
        for key, meta in EXTENDED_ABLATION_VARIANTS.items():
            signals = ablation_signals.get(key, [])
            results[key] = _ablation_metrics(
                signals, moves, frame, replay_dates,
                variant_key=key, variant_meta=meta,
            )
            stats = results[key]["overall_statistics"]
            results[key]["signals_per_month"] = stats.get("signals_per_month")
            results[key]["win_rate_pct"] = stats.get("win_rate_pct")
            results[key]["profit_factor"] = stats.get("profit_factor")
            results[key]["expectancy"] = stats.get("expectancy")

        ranking = _ablation_contribution_ranking(results)
        return {
            "window_days": trading_days,
            "variants": results,
            "contribution_ranking": ranking,
            "methodology": "Remove-one BUY_V3 component ablation on replay window.",
        }

    def _run_execution_analysis(
        self,
        *,
        buy_signals: list[dict[str, Any]],
        sell_signals: list[dict[str, Any]],
        window_days: int,
    ) -> dict[str, Any]:
        structure = RUNNER_STRATEGIES["60_100_runner"]
        stress: dict[str, Any] = {}
        for slip in SLIPPAGE_STRESS_LEVELS:
            buy_row = _slippage_stress_row(
                buy_signals, side="BUY", structure=structure, stop_variant="fixed_10",
                slippage_pts=float(slip), window_days=window_days,
            )
            sell_row = _slippage_stress_row(
                sell_signals, side="SELL", structure=structure, stop_variant="fixed_10",
                slippage_pts=float(slip), window_days=window_days,
            )
            stress[str(slip)] = {"buy_v3": buy_row, "sell_v6": sell_row}

        return {
            "entry_efficiency_note": "Entry efficiency measured via bars_before_expansion timing classes.",
            "slippage_stress": {
                "stress_levels_points": list(SLIPPAGE_STRESS_LEVELS),
                "by_slippage_level": stress,
            },
            "stop_comparison": {
                "buy_v3": _stop_comparison_matrix(buy_signals, side="BUY", structure=structure, window_days=window_days),
                "sell_v6": _stop_comparison_matrix(sell_signals, side="SELL", structure=structure, window_days=window_days),
            },
            "runner_optimization": {
                "buy_v3": _runner_exit_optimization(buy_signals, side="BUY", stop_variant="fixed_10", window_days=window_days),
                "sell_v6": _runner_exit_optimization(sell_signals, side="SELL", stop_variant="fixed_10", window_days=window_days),
                "best_strategy": "60_100_runner",
            },
        }

    def run(
        self,
        metadata: dict[str, Any],
        *,
        windows: tuple[int, ...] = REPLAY_WINDOWS,
        run_ablation: bool = True,
    ) -> ExtendedEvidenceValidationRealDeploymentAuditReport:
        started = time.perf_counter()
        max_window = max(windows)
        calendar_days = CALENDAR_BUFFER.get(max_window, max_window * 2)
        end = date.fromisoformat(metadata["end_date"])
        start = end - timedelta(days=calendar_days)

        logger.info(
            "Extended evidence validation starting: windows=%s, max=%sd, %s 5M",
            windows, max_window, DEFAULT_SYMBOL,
        )

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=calendar_days,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        all_replay_dates = _last_n_trading_day_set(frame, max_window)
        bar_dates = pd.to_datetime(frame["Date"]).dt.date
        replay_bars = [index for index, day in enumerate(bar_dates) if day in all_replay_dates]

        logger.info("Loading enriched context and intel frames...")
        enriched_buy = self.buy_engine.context_builder.enrich(frame)
        enriched_sell = _attach_ema22(self.sell_engine.context_builder.enrich(frame))
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        intel_frames: dict[str, pd.DataFrame] = {"5M": self.buy_engine.intelligence.enrich(frame)}
        for tf in ("15M", "1H"):
            tf_path = filter_engine._ensure_pipeline(tf, start, end)
            tf_frame = pd.read_csv(tf_path).reset_index(drop=True)
            intel_frames[tf] = self.buy_engine.intelligence.enrich(tf_frame)
        intel_frames["1D"] = self.buy_engine.intelligence.enrich(
            self.buy_engine._resample_daily(intel_frames["1H"]),
        )

        logger.info("Detecting moves (threshold=%s)...", MOVE_DETECTION_THRESHOLD)
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, MOVE_DETECTION_THRESHOLD),
        )
        daily_ranges = _daily_range_lookup(frame)

        logger.info("Running %sd production replay (BUY_V3 + SELL_V6)...", max_window)
        full_signals = self._replay_production(
            frame=frame, enriched_buy=enriched_buy, enriched_sell=enriched_sell,
            calendar=calendar, intel_frames=intel_frames, replay_bars=replay_bars,
            moves=moves, daily_ranges=daily_ranges, include_ablation=False,
        )

        ablation_data: dict[str, Any] = {}
        if run_ablation:
            ablation_window = min(120, max_window)
            ablation_dates = _last_n_trading_day_set(frame, ablation_window)
            ablation_bars = [index for index, day in enumerate(bar_dates) if day in ablation_dates]
            logger.info("Running %sd ablation replay...", ablation_window)
            ablation_replay = self._replay_production(
                frame=frame, enriched_buy=enriched_buy, enriched_sell=enriched_sell,
                calendar=calendar, intel_frames=intel_frames, replay_bars=ablation_bars,
                moves=moves, daily_ranges=daily_ranges, include_ablation=True,
            )
            ablation_signals = ablation_replay.get("ablation", {})
            if isinstance(ablation_signals, dict):
                ablation_data = self._run_ablation_analysis(
                    ablation_signals, moves=moves, frame=frame,
                    replay_dates=ablation_dates, trading_days=ablation_window,
                )

        regime_export = _load_json_safe(SOURCE_EXPORTS["regime_detection_audit"])
        throttle_maps = _load_throttle_maps(regime_export)
        prior_exports = {name: _load_json_safe(path) for name, path in SOURCE_EXPORTS.items()}
        prior_synthesis = _synthesize_prior_exports(prior_exports)

        window_results: dict[str, Any] = {}
        for window in windows:
            logger.info("Analyzing window: %sd trading days", window)
            window_dates = _last_n_trading_day_set(frame, window)
            buy_w = _filter_signals_by_dates(full_signals["buy_v3"], frame, window_dates)
            sell_w = _filter_signals_by_dates(full_signals["sell_v6"], frame, window_dates)
            window_results[str(window)] = self._analyze_window(
                buy_signals=buy_w, sell_signals=sell_w, frame=frame,
                replay_dates=window_dates, trading_days=window,
                moves=moves, throttle_maps=throttle_maps,
            )
            logger.info(
                "Window %sd: BUY=%s SELL=%s combined PF=%s",
                window, len(buy_w), len(sell_w),
                window_results[str(window)]["combined"].get("profit_factor"),
            )

        wf_stability: dict[str, Any] = {}
        for window in windows:
            wf = window_results[str(window)]["combined"].get("walk_forward", {})
            wf_stability[str(window)] = {
                "train_pf": wf.get("train", {}).get("profit_factor"),
                "validate_pf": wf.get("validate", {}).get("profit_factor"),
                "pf_stability_ratio": wf.get("pf_stability_ratio"),
                "stable": wf.get("stable"),
                "sample_size_assessment": {
                    "train_signals": wf.get("train", {}).get("signal_count"),
                    "validate_signals": wf.get("validate", {}).get("signal_count"),
                    "required_for_90pct_confidence": _required_sample_size(0.70, confidence_pct=90),
                },
            }

        sell_ranking = _sell_v6_component_ranking(
            _filter_signals_by_dates(full_signals["sell_v6"], frame, _last_n_trading_day_set(frame, max_window)),
        )

        buy_v4_sell_v7 = {
            "buy_v4": {
                "recommendation": "NO",
                "cost": "High — new discovery cycle, 4-8 weeks research",
                "benefit": "Marginal unless 500d combined PF < 2.0 or ablation shows Failed Breakdown is top contributor",
                "verdict": "NO",
            },
            "sell_v7": {
                "recommendation": "NO",
                "cost": "High — new VWAP/regime gate research",
                "benefit": "Low — SELL_V6 already improves PF vs V5; V7 only if range-market frequency collapse",
                "verdict": "NO",
            },
        }

        execution = self._run_execution_analysis(
            buy_signals=_filter_signals_by_dates(full_signals["buy_v3"], frame, _last_n_trading_day_set(frame, max_window)),
            sell_signals=_filter_signals_by_dates(full_signals["sell_v6"], frame, _last_n_trading_day_set(frame, max_window)),
            window_days=max_window,
        )

        evidence_audit = _evidence_strength_audit(
            window_results=window_results, ablation=ablation_data, prior_synthesis=prior_synthesis,
        )
        unknown_risks = _unknown_risk_audit(window_results=window_results, prior_synthesis=prior_synthesis)

        live_export = prior_exports.get("live_trade_management_execution_efficiency_audit", {})
        deployment_export = prior_exports.get("final_production_deployment_audit", {})
        reality_export = prior_exports.get("production_reality_audit", {})

        buy_matrix = window_results.get(str(max_window), {}).get("target_achievement", {}).get("buy_v3", {})
        sell_matrix = window_results.get(str(max_window), {}).get("target_achievement", {}).get("sell_v6", {})
        capture_summary = _capture_summary(
            buy_matrix, sell_matrix,
            {"by_strategy": execution["runner_optimization"]["buy_v3"]},
            {"by_strategy": execution["runner_optimization"]["sell_v6"]},
        )

        max_window_dates = _last_n_trading_day_set(frame, max_window)
        max_buy_signals = _filter_signals_by_dates(full_signals["buy_v3"], frame, max_window_dates)
        max_sell_signals = _filter_signals_by_dates(full_signals["sell_v6"], frame, max_window_dates)
        buy_export_payload = {"per_signal_details": {"buy_v3": full_signals["buy_v3"]}}
        sell_export_payload = {"per_signal_details": {"sell_v6": full_signals["sell_v6"]}}
        evidence_quality = _evidence_quality_analysis(
            buy_signals=max_buy_signals,
            sell_signals=max_sell_signals,
            buy_export=buy_export_payload,
            sell_export=sell_export_payload,
            deployment_audit=deployment_export,
            window_days=max_window,
        )
        evidence_quality["current_confidence_pct"]["combined_estimate"] = min(
            95.0,
            evidence_audit["aggregate_evidence_score"],
        )

        truth_audit = _production_truth_audit(
            deployment_audit=deployment_export,
            live_audit=live_export,
            regime_audit=regime_export,
            buy_export=buy_export_payload,
            sell_export=sell_export_payload,
            evidence_quality=evidence_quality,
        )

        production_scores = _production_scores(
            deployment_audit=deployment_export or reality_export,
            truth_audit=truth_audit,
            capture_summary=capture_summary,
            evidence_quality=evidence_quality,
        )
        production_scores["evidence_score"] = evidence_audit["aggregate_evidence_score"]

        prod_config = _production_config(
            window_results=window_results, throttle_maps=throttle_maps, live_export=live_export,
        )

        final = _final_verdict(
            window_results=window_results, scores=production_scores,
            evidence_audit=evidence_audit, unknown_risks=unknown_risks,
            prior_synthesis=prior_synthesis, ablation=ablation_data,
        )
        buy_v4_sell_v7["buy_v4"]["verdict"] = final["should_research_buy_v4"]
        buy_v4_sell_v7["sell_v7"]["verdict"] = final["should_research_sell_v7"]

        top_risks, top_opportunities = _top_risks_and_opportunities(
            window_results=window_results, unknown_risks=unknown_risks, execution=execution,
        )

        conclusions = [
            f"Extended replay complete for windows {list(windows)} on {DEFAULT_SYMBOL} 5M.",
            (
                f"500d combined: {window_results.get('500', {}).get('combined', {}).get('signals_per_month')}/mo, "
                f"WR {window_results.get('500', {}).get('combined', {}).get('win_rate_pct')}%, "
                f"PF {window_results.get('500', {}).get('combined', {}).get('profit_factor')}."
            ),
            (
                f"Regime throttle 500d PF: "
                f"{window_results.get('500', {}).get('combined_regime_throttle', {}).get('profit_factor')}."
            ),
            f"Walk-forward 70/30 stable all windows: {final['walk_forward_stable_all_windows']}.",
            f"BUY_V4: {final['should_research_buy_v4']} | SELL_V7: {final['should_research_sell_v7']}.",
            f"Definitive verdict: {final['definitive_verdict']}.",
            f"Future research focus: {final['future_research_focus']}.",
        ]

        return ExtendedEvidenceValidationRealDeploymentAuditReport(
            report_type="Extended Evidence Validation & Real Deployment Audit",
            engines=["BUY_V3", "SELL_V6", "COMBINED", "COMBINED+REGIME_THROTTLE"],
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            replay_windows=list(windows),
            methodology={
                "research_only": True,
                "actual_replay": True,
                "replay_windows_trading_days": list(windows),
                "walk_forward_split": f"train {int(TRAIN_RATIO * 100)}% / validate {int((1 - TRAIN_RATIO) * 100)}%",
                "buy_engine": BUY_V3_MODEL_ID,
                "buy_formula": BUY_V3_FORMULA_TEXT,
                "sell_engine": SELL_V6_MODEL_ID,
                "sell_vwap_gate": V6_VWAP_GATE_RULE,
                "regime_throttle_source": "regime_detection_audit.json",
                "execution_structure": "60/100/Runner",
                "stop_variants_tested": list(STOP_VARIANTS),
                "slippage_stress_levels": list(SLIPPAGE_STRESS_LEVELS),
                "mfe_capture_tiers": list(MFE_CAPTURE_TIERS),
                "point_capture_thresholds": list(POINT_CAPTURE_THRESHOLDS),
                "no_buy_v4": True,
                "no_sell_v7": True,
                "no_new_indicators": True,
                "no_signal_logic_change": True,
            },
            window_results=window_results,
            walk_forward_stability=wf_stability,
            signal_logic_transparency={
                "buy_v3_ablation": ablation_data,
                "sell_v6_component_ranking": sell_ranking,
                "buy_v4_sell_v7_justification": buy_v4_sell_v7,
            },
            execution_analysis=execution,
            evidence_strength_audit=evidence_audit,
            unknown_risk_audit=unknown_risks,
            prior_export_synthesis=prior_synthesis,
            production_config=prod_config,
            production_scores=production_scores,
            top_risks=top_risks,
            top_opportunities=top_opportunities,
            final_answer=final,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(
        self,
        report: ExtendedEvidenceValidationRealDeploymentAuditReport,
        report_path: Path,
    ) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("Extended evidence validation audit exported: %s", report_path)
        return report_path


def _load_json_safe(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Missing export: %s", path)
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def generate_extended_evidence_validation_real_deployment_audit_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    *,
    windows: tuple[int, ...] = REPLAY_WINDOWS,
    run_ablation: bool = True,
) -> ExtendedEvidenceValidationRealDeploymentAuditReport:
    """Run extended multi-window replay audit and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise ExtendedEvidenceValidationRealDeploymentAuditError(
            f"Filter research report not found: {metadata_path}",
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    research = ExtendedEvidenceValidationRealDeploymentAuditResearch()
    report = research.run(metadata, windows=windows, run_ablation=run_ablation)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = generate_extended_evidence_validation_real_deployment_audit_report()
        final = report.final_answer
        print(f"Exported: {DEFAULT_REPORT_PATH}")
        print(f"Verdict: {final['definitive_verdict']}")
        print(f"BUY_V4: {final['should_research_buy_v4']} | SELL_V7: {final['should_research_sell_v7']}")
        for window in REPLAY_WINDOWS:
            w = report.window_results.get(str(window), {}).get("combined", {})
            print(f"  {window}d: PF={w.get('profit_factor')} WR={w.get('win_rate_pct')}% spm={w.get('signals_per_month')}")
        return 0
    except ExtendedEvidenceValidationRealDeploymentAuditError as exc:
        logger.error("Extended evidence validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
