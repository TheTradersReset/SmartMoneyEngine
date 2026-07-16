"""
SELL_V6 Replay Validation — actual combined replay vs SELL_V5.

SELL_V6 = SELL_V5 with one change: VWAP gate accepts Below only (V4 rule).
Removes the V5 VWAP Rejected path. Research-only; does not modify BUY_V3.
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
from src.research.buy_v2_candidate_validation_research import (
    PRODUCTION_GATES,
    TRAIN_TRADING_DAYS,
    TRADING_DAYS_REPLAY,
    VALIDATE_TRADING_DAYS,
    _filter_signals_by_dates,
    _split_trading_day_sets,
    _walk_forward_metrics,
)
from src.research.filter_research_engine import FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import PRE_EXPANSION_LOOKBACK
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    _CheapMoveCandidate,
)
from src.research.nifty50_trap_to_momentum_validation_research import DEFAULT_SYMBOL
from src.research.production_edge_enhancement_audit_research import (
    _classify_sell_signal,
    _extract_sell_conditions,
)
from src.research.smartmoneyengine_v3_implementation_validation_research import (
    MOVE_DETECTION_TIMEFRAME,
    PIPELINE_TIMEFRAMES,
)
from src.research.smartmoneyengine_v4_candidate_validation_research import (
    V4_EMA_BEAR_CONTEXT,
    V4_EMA22_RULE,
    _attach_ema22,
    _build_statistics,
    _last_n_trading_day_set,
    _point_capture,
    _profit_factor,
    _signal_before_move,
)
from src.research.smartmoneyengine_v5_candidate_validation_research import (
    V5CandidateEngine,
    V5_VWAP_GATE_RULE,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "sell_v6_replay_validation.json"
DEFAULT_AUDIT_PATH = RESEARCH_DIR / "production_edge_enhancement_audit.json"
DEFAULT_V5_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v5_candidate_validation.json"

POINT_CAPTURE_THRESHOLDS = (40, 60, 100, 200)
MFE_CAPTURE_TIERS = (40, 60, 100, 200)
MOVE_DETECTION_THRESHOLD = 40
SELL_MIN_SIGNALS_PER_MONTH = 60.0

AUDIT_V5_PF = 3.37
AUDIT_V6_PF = 4.09
AUDIT_PF_TOLERANCE = 0.10

V6_VWAP_GATE_RULE = "VWAP Below only"
V6_ALLOWED_VWAP_STATES = frozenset({"Below"})

REMOVED_CLASSIFICATIONS = (
    "Winner",
    "Bear Trap",
    "Range Failure",
    "Liquidity Failure",
    "No Expansion",
    "Trend Reversal",
    "Gap Failure",
)


class SellV6ReplayValidationError(Exception):
    """Raised when SELL_V6 replay validation fails."""


def _v6_vwap_gate_passes(vwap_state: str | None) -> bool:
    return vwap_state == "Below"


def _map_removed_classification(label: str) -> str:
    if label == "Trend Exhaustion":
        return "Trend Reversal"
    if label in REMOVED_CLASSIFICATIONS:
        return label
    return "Liquidity Failure"


class SellV6CandidateEngine(V5CandidateEngine):
    """SELL_V6: V5 stack with VWAP Below only (V4 gate)."""

    MODEL_ID = "LDM-SELL-V6"

    def _layer2_directional_filter(self, context: dict[str, str]) -> dict[str, Any]:
        ema_bearish = context.get("v4_ema_bearish") == "True"
        vwap = context.get("vwap")
        vwap_ok = _v6_vwap_gate_passes(vwap)
        aligned = (
            context.get("htf_trend") == "Bearish"
            and vwap_ok
            and ema_bearish
        )
        return {
            "direction": "SELL" if aligned else "NO_TRADE",
            "htf_trend": context.get("htf_trend"),
            "vwap_state": vwap,
            "vwap_gate_rule": V6_VWAP_GATE_RULE,
            "vwap_gate_passes": vwap_ok,
            "ema_structure": context.get("v4_ema_structure"),
            "v4_ema_rule": V4_EMA22_RULE,
            "v4_ema_bearish": ema_bearish,
            "aligned": aligned,
        }


def _nearest_bearish_move(
    moves: list[_CheapMoveCandidate],
    signal_bar: int,
    *,
    forward_bars: int = FORWARD_BARS,
) -> _CheapMoveCandidate | None:
    candidates = [
        move
        for move in moves
        if move.direction == "bearish"
        and signal_bar <= move.start_bar <= signal_bar + forward_bars
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.start_bar)


def _tier_hits(signal: dict[str, Any]) -> dict[str, bool]:
    mfe = float(signal.get("mfe_points") or 0.0)
    return {str(tier): mfe >= tier for tier in MFE_CAPTURE_TIERS}


def _daily_range_lookup(frame: pd.DataFrame) -> dict[date, float]:
    day_frame = frame.copy()
    day_frame["_day"] = pd.to_datetime(day_frame["Date"]).dt.date
    grouped = day_frame.groupby("_day").agg(
        day_high=("High", "max"),
        day_low=("Low", "min"),
    )
    return {
        day: float(row.day_high - row.day_low)
        for day, row in grouped.iterrows()
    }


def _classify_market_regime(
    frame: pd.DataFrame,
    bar: int,
    *,
    layer2: dict[str, Any],
    layer1: dict[str, Any],
    daily_ranges: dict[date, float],
) -> dict[str, str]:
    signal_day = pd.to_datetime(frame.iloc[bar]["Date"]).date()
    htf = layer2.get("htf_trend") or "Neutral"
    events = set(layer1.get("events_detected") or [])

    if htf == "Bearish":
        trend = "trending"
    elif htf == "Bullish":
        trend = "counter_trend"
    else:
        trend = "range"

    day_range = daily_ranges.get(signal_day)
    if day_range is not None:
        median_range = median(daily_ranges.values()) if daily_ranges else day_range
        vol = "high_vol" if day_range >= median_range else "low_vol"
    else:
        vol = "unknown_vol"

    gap = "no_gap"
    if bar > 0:
        prev_close = float(frame.iloc[bar - 1]["Close"])
        open_price = float(frame.iloc[bar]["Open"])
        gap_pct = (open_price - prev_close) / max(abs(prev_close), 1.0) * 100.0
        if gap_pct >= 0.5:
            gap = "gap_up"
        elif gap_pct <= -0.5:
            gap = "gap_down"

    if "Gap Reversal" in events or "Gap Continuation" in events:
        if gap == "no_gap":
            gap = "gap_event"

    return {
        "trend_regime": trend,
        "vol_regime": vol,
        "gap_regime": gap,
        "composite": f"{trend}|{vol}|{gap}",
    }


def _enrich_sell_signal(
    evaluation: dict[str, Any],
    *,
    engine_version: str,
    model_id: str,
    moves: list[_CheapMoveCandidate],
    frame: pd.DataFrame,
    daily_ranges: dict[date, float],
) -> dict[str, Any]:
    layer4 = evaluation["layer4"]
    forward = dict(layer4.get("forward_outcome") or {})
    bar = evaluation["bar"]
    linked_move = _nearest_bearish_move(moves, bar)
    move_start_bar = linked_move.start_bar if linked_move else None
    bars_before_expansion = (move_start_bar - bar) if move_start_bar is not None else None

    layer1 = evaluation["layer1"]
    layer2 = evaluation["layer2"]
    regime = _classify_market_regime(
        frame,
        bar,
        layer2=layer2,
        layer1=layer1,
        daily_ranges=daily_ranges,
    )

    signal = {
        "timestamp": evaluation["timestamp"],
        "bar": bar,
        "symbol": DEFAULT_SYMBOL,
        "timeframe": MOVE_DETECTION_TIMEFRAME,
        "direction": "SELL",
        "engine_version": engine_version,
        "model_id": model_id,
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
        "move_start_bar": move_start_bar,
        "move_start_time": str(frame.iloc[move_start_bar]["Date"]) if move_start_bar is not None else None,
        "bars_before_expansion": bars_before_expansion,
        "mfe_capture_tiers": _tier_hits(forward),
        "regime": regime,
        "layers": {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": evaluation["layer3"],
            "layer5": evaluation["layer5"],
        },
    }
    signal["conditions"] = _extract_sell_conditions(signal)
    signal["classification"] = _map_removed_classification(_classify_sell_signal(signal))
    return signal


def _comparison_row(stats: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    return {
        "signals_emitted": stats.get("signals_emitted"),
        "signals_per_week": stats.get("signals_per_week"),
        "signals_per_month": stats.get("signals_per_month"),
        "win_rate_pct": stats.get("win_rate_pct"),
        "profit_factor": stats.get("profit_factor"),
        "expectancy": stats.get("expectancy"),
        "average_mfe": stats.get("average_mfe"),
        "average_mae": stats.get("average_mae"),
        "capture_40_pct": capture.get("40", {}).get("capture_rate_pct"),
        "capture_60_pct": capture.get("60", {}).get("capture_rate_pct"),
        "capture_100_pct": capture.get("100", {}).get("capture_rate_pct"),
        "capture_200_pct": capture.get("200", {}).get("capture_rate_pct"),
    }


def _delta_row(v5: dict[str, Any], v6: dict[str, Any]) -> dict[str, Any]:
    pf_v5 = v5.get("profit_factor")
    pf_v6 = v6.get("profit_factor")
    return {
        "signals_delta": (v6.get("signals_emitted") or 0) - (v5.get("signals_emitted") or 0),
        "signals_per_month_delta": round(
            (v6.get("signals_per_month") or 0) - (v5.get("signals_per_month") or 0),
            2,
        ),
        "wr_delta_pp": round((v6.get("win_rate_pct") or 0) - (v5.get("win_rate_pct") or 0), 2),
        "pf_delta": round(pf_v6 - pf_v5, 2) if pf_v5 is not None and pf_v6 is not None else None,
        "expectancy_delta": round((v6.get("expectancy") or 0) - (v5.get("expectancy") or 0), 2),
        "mae_delta": round((v6.get("average_mae") or 0) - (v5.get("average_mae") or 0), 2),
        "mfe_delta": round((v6.get("average_mfe") or 0) - (v5.get("average_mfe") or 0), 2),
        "capture_200_delta_pp": round(
            (v6.get("capture_200_pct") or 0) - (v5.get("capture_200_pct") or 0),
            2,
        ),
    }


def _removed_trade_analysis(
    v5_signals: list[dict[str, Any]],
    v6_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    v6_bars = {signal["bar"] for signal in v6_signals}
    removed = [signal for signal in v5_signals if signal["bar"] not in v6_bars]
    removed_winners = [signal for signal in removed if signal.get("win")]
    removed_losers = [signal for signal in removed if not signal.get("win")]

    winner_pnls = sum(float(signal.get("realized_pnl_points") or 0.0) for signal in removed_winners)
    loser_pnls = sum(float(signal.get("realized_pnl_points") or 0.0) for signal in removed_losers)
    net_pnl = winner_pnls + loser_pnls

    classification_counts = Counter(signal.get("classification", "Unknown") for signal in removed)
    vwap_states = Counter(
        signal.get("layers", {}).get("layer2", {}).get("vwap_state") for signal in removed
    )

    removed_rows = []
    for signal in removed:
        removed_rows.append(
            {
                "timestamp": signal.get("timestamp"),
                "bar": signal.get("bar"),
                "outcome": "win" if signal.get("win") else "loss",
                "realized_pnl_points": signal.get("realized_pnl_points"),
                "mfe_points": signal.get("mfe_points"),
                "mae_points": signal.get("mae_points"),
                "mfe_capture_tiers": signal.get("mfe_capture_tiers"),
                "classification": signal.get("classification"),
                "vwap_state": signal.get("layers", {}).get("layer2", {}).get("vwap_state"),
                "regime": signal.get("regime"),
            },
        )

    bad_only = all(not signal.get("win") for signal in removed) if removed else True
    high_value_winners_lost = [
        signal
        for signal in removed_winners
        if float(signal.get("mfe_points") or 0.0) >= 100.0
    ]

    return {
        "removed_count": len(removed),
        "removed_winners": len(removed_winners),
        "removed_losers": len(removed_losers),
        "net_winners_lost": len(removed_winners),
        "net_losers_removed": len(removed_losers),
        "net_pnl_removed_points": round(net_pnl, 2),
        "winner_pnl_removed": round(winner_pnls, 2),
        "loser_pnl_removed": round(loser_pnls, 2),
        "bad_trades_only": bad_only,
        "high_value_winners_lost_count": len(high_value_winners_lost),
        "high_value_winners_lost_mfe_avg": round(
            mean(float(s.get("mfe_points") or 0.0) for s in high_value_winners_lost),
            2,
        )
        if high_value_winners_lost
        else None,
        "removal_composition": "bad_trades_only" if bad_only else "bad_trades_and_high_value_winners",
        "vwap_state_breakdown": dict(vwap_states),
        "classification_breakdown": dict(classification_counts),
        "removed_signal_details": removed_rows,
        "summary": (
            f"Removed {len(removed)} V5-only signals ({len(removed_losers)} losers, "
            f"{len(removed_winners)} winners); net PnL impact {round(net_pnl, 2)} pts."
        ),
    }


def _regime_analysis(
    v5_signals: list[dict[str, Any]],
    v6_signals: list[dict[str, Any]],
    removed: list[dict[str, Any]],
) -> dict[str, Any]:
    v6_bars = {signal["bar"] for signal in v6_signals}
    rejected_v5 = [
        signal
        for signal in v5_signals
        if signal.get("layers", {}).get("layer2", {}).get("vwap_state") == "Rejected"
    ]
    rejected_kept = [signal for signal in rejected_v5 if signal["bar"] in v6_bars]
    rejected_removed = [signal for signal in rejected_v5 if signal["bar"] not in v6_bars]

    def _cohort_stats(signals: list[dict[str, Any]]) -> dict[str, Any]:
        if not signals:
            return {"count": 0, "win_rate_pct": 0.0, "profit_factor": None, "expectancy": 0.0}
        pnls = [float(s.get("realized_pnl_points") or 0.0) for s in signals]
        return {
            "count": len(signals),
            "win_rate_pct": round(sum(1 for s in signals if s.get("win")) / len(signals) * 100, 2),
            "profit_factor": _profit_factor(pnls),
            "expectancy": round(mean(pnls), 2),
            "average_mfe": round(mean(float(s.get("mfe_points") or 0.0) for s in signals), 2),
            "average_mae": round(mean(float(s.get("mae_points") or 0.0) for s in signals), 2),
        }

    by_axis: dict[str, dict[str, dict[str, Any]]] = {
        "trend_regime": defaultdict(dict),
        "vol_regime": defaultdict(dict),
        "gap_regime": defaultdict(dict),
    }

    for axis in by_axis:
        keys = set()
        for signal in v5_signals + removed:
            keys.add(signal.get("regime", {}).get(axis, "unknown"))
        for key in sorted(keys):
            v5_cohort = [s for s in v5_signals if s.get("regime", {}).get(axis) == key]
            v6_cohort = [s for s in v6_signals if s.get("regime", {}).get(axis) == key]
            removed_cohort = [s for s in removed if s.get("regime", {}).get(axis) == key]
            by_axis[axis][key] = {
                "sell_v5": _cohort_stats(v5_cohort),
                "sell_v6": _cohort_stats(v6_cohort),
                "removed_by_v6": _cohort_stats(removed_cohort),
            }

    rejected_removed_stats = _cohort_stats(rejected_removed)
    rejected_kept_stats = _cohort_stats(rejected_kept)

    return {
        "by_trend_regime": dict(by_axis["trend_regime"]),
        "by_vol_regime": dict(by_axis["vol_regime"]),
        "by_gap_regime": dict(by_axis["gap_regime"]),
        "vwap_rejected_cohort": {
            "v5_rejected_total": len(rejected_v5),
            "kept_in_v6": len(rejected_kept),
            "removed_by_v6": len(rejected_removed),
            "rejected_kept_stats": rejected_kept_stats,
            "rejected_removed_stats": rejected_removed_stats,
            "value_vs_damage_verdict": (
                "damage"
                if rejected_removed_stats.get("expectancy", 0) < 0
                else (
                    "value"
                    if rejected_removed_stats.get("expectancy", 0) > 0
                    else "neutral"
                )
            ),
            "note": (
                "V6 removes all VWAP Rejected entries; this cohort measures whether "
                "Rejected path added edge (value) or diluted PF (damage)."
            ),
        },
    }


def _walk_forward_stability(
    v5_train: dict[str, Any],
    v5_validate: dict[str, Any],
    v6_train: dict[str, Any],
    v6_validate: dict[str, Any],
) -> dict[str, Any]:
    def _stable(train: dict[str, Any], validate: dict[str, Any]) -> bool:
        train_wr = float(train.get("win_rate_pct") or 0.0)
        val_wr = float(validate.get("win_rate_pct") or 0.0)
        train_pf = float(train.get("profit_factor") or 0.0)
        val_pf = float(validate.get("profit_factor") or 0.0)
        return val_wr >= train_wr * 0.85 and val_pf >= train_pf * 0.70

    v5_stable = _stable(v5_train, v5_validate)
    v6_stable = _stable(v6_train, v6_validate)

    return {
        "split": f"train {TRAIN_TRADING_DAYS} / validate {VALIDATE_TRADING_DAYS} trading days",
        "train_days": TRAIN_TRADING_DAYS,
        "validate_days": VALIDATE_TRADING_DAYS,
        "sell_v5_stable": v5_stable,
        "sell_v6_stable": v6_stable,
        "both_stable": v5_stable and v6_stable,
        "v6_improves_validate_pf": (
            float(v6_validate.get("profit_factor") or 0.0)
            >= float(v5_validate.get("profit_factor") or 0.0)
        ),
    }


def _audit_pf_reconciliation(v6_pf: float | None) -> dict[str, Any]:
    if v6_pf is None:
        survives = False
        delta_vs_audit = None
    else:
        delta_vs_audit = round(v6_pf - AUDIT_V6_PF, 2)
        survives = v6_pf >= AUDIT_V6_PF * (1.0 - AUDIT_PF_TOLERANCE)
    return {
        "audit_baseline_v5_pf": AUDIT_V5_PF,
        "audit_expected_v6_pf": AUDIT_V6_PF,
        "replay_v6_pf": v6_pf,
        "replay_v5_pf_baseline": AUDIT_V5_PF,
        "pf_delta_audit_predicted": round(AUDIT_V6_PF - AUDIT_V5_PF, 2),
        "replay_delta_vs_audit_expected": delta_vs_audit,
        "pf_survives_replay": survives,
        "tolerance_pct": AUDIT_PF_TOLERANCE * 100,
        "verdict": "YES" if survives else "NO",
        "note": (
            f"Production edge audit predicted PF {AUDIT_V5_PF}→{AUDIT_V6_PF} via "
            f"'require VWAP Below' filter simulation on unified export."
        ),
    }


def _production_readiness_score(
    *,
    v6_stats: dict[str, Any],
    v6_capture: dict[str, Any],
    walk_forward: dict[str, Any],
    pf_reconciliation: dict[str, Any],
) -> dict[str, Any]:
    score = 0
    components: dict[str, Any] = {}

    wr = float(v6_stats.get("win_rate_pct") or 0.0)
    wr_score = 25 if wr >= 70 else (20 if wr >= 65 else (10 if wr >= 60 else 0))
    components["win_rate"] = {"score": wr_score, "win_rate_pct": wr}
    score += wr_score

    pf = float(v6_stats.get("profit_factor") or 0.0)
    pf_score = 25 if pf >= 4.0 else (20 if pf >= 3.0 else (12 if pf >= 2.0 else 0))
    components["profit_factor"] = {"score": pf_score, "profit_factor": pf}
    score += pf_score

    spm = float(v6_stats.get("signals_per_month") or 0.0)
    freq_score = 15 if spm >= SELL_MIN_SIGNALS_PER_MONTH else (10 if spm >= 50 else 5)
    components["frequency"] = {"score": freq_score, "signals_per_month": spm}
    score += freq_score

    wf_score = 15 if walk_forward.get("sell_v6_stable") else 5
    components["walk_forward"] = {"score": wf_score, "stable": walk_forward.get("sell_v6_stable")}
    score += wf_score

    capture_40 = float(v6_capture.get("40", {}).get("capture_rate_pct") or 0.0)
    capture_score = 10 if capture_40 >= 5 else (6 if capture_40 >= 2 else 2)
    components["capture"] = {"score": capture_score, "capture_40_pct": capture_40}
    score += capture_score

    audit_score = 10 if pf_reconciliation.get("pf_survives_replay") else 0
    components["audit_pf_survival"] = {
        "score": audit_score,
        "survives": pf_reconciliation.get("pf_survives_replay"),
    }
    score += audit_score

    if score >= 80:
        tier = "Production Candidate"
    elif score >= 65:
        tier = "Paper Trading"
    elif score >= 45:
        tier = "Dry Run"
    else:
        tier = "Research"

    return {
        "score": min(100, score),
        "components": components,
        "recommendation_tier": tier,
        "production_gates": {
            "win_rate_above_65_pct": wr >= PRODUCTION_GATES["win_rate_min_pct"],
            "profit_factor_above_2": pf >= PRODUCTION_GATES["profit_factor_min"],
            "signals_per_month_60_plus": spm >= SELL_MIN_SIGNALS_PER_MONTH,
            "all_pass": (
                wr >= PRODUCTION_GATES["win_rate_min_pct"]
                and pf >= PRODUCTION_GATES["profit_factor_min"]
                and spm >= SELL_MIN_SIGNALS_PER_MONTH
            ),
        },
    }


def _final_verdict(
    *,
    v5_stats: dict[str, Any],
    v6_stats: dict[str, Any],
    removed_analysis: dict[str, Any],
    walk_forward: dict[str, Any],
    pf_reconciliation: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    pf_v5 = float(v5_stats.get("profit_factor") or 0.0)
    pf_v6 = float(v6_stats.get("profit_factor") or 0.0)
    spm_v6 = float(v6_stats.get("signals_per_month") or 0.0)
    wr_v6 = float(v6_stats.get("win_rate_pct") or 0.0)

    pf_improves = pf_v6 > pf_v5
    gates_pass = readiness.get("production_gates", {}).get("all_pass", False)
    wf_ok = walk_forward.get("sell_v6_stable", False)
    audit_ok = pf_reconciliation.get("pf_survives_replay", False)
    net_pnl_removed = float(removed_analysis.get("net_pnl_removed_points") or 0.0)

    if pf_improves and gates_pass and wf_ok and audit_ok and net_pnl_removed >= 0:
        verdict = "YES"
        rationale = (
            "V6 improves PF vs V5, passes production frequency/WR/PF gates, walk-forward stable, "
            "and audit PF 3.37→4.09 survives actual replay."
        )
    elif pf_improves and gates_pass and (wf_ok or audit_ok):
        verdict = "PARTIAL"
        rationale = (
            "V6 improves PF and passes gates but walk-forward or audit reconciliation is not fully confirmed."
        )
    elif pf_improves and spm_v6 >= 50 and wr_v6 >= 65:
        verdict = "PARTIAL"
        rationale = "V6 improves quality metrics but misses one or more production readiness thresholds."
    else:
        verdict = "NO"
        rationale = "V6 does not sufficiently improve or stabilize SELL leg vs V5 on replay evidence."

    return {
        "can_sell_v6_replace_sell_v5": verdict,
        "rationale": rationale,
        "pf_improves_vs_v5": pf_improves,
        "audit_pf_survives": audit_ok,
        "walk_forward_stable": wf_ok,
        "production_gates_pass": gates_pass,
        "readiness_score": readiness.get("score"),
        "recommendation_tier": readiness.get("recommendation_tier"),
    }


@dataclass
class SellV6ReplayValidationReport:
    """SELL_V5 vs SELL_V6 actual replay validation output."""

    report_type: str
    engines_compared: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    v6_change_summary: dict[str, Any]
    methodology: dict[str, Any]
    comparison_table: dict[str, Any]
    walk_forward: dict[str, Any]
    pf_audit_reconciliation: dict[str, Any]
    removed_trade_analysis: dict[str, Any]
    regime_analysis: dict[str, Any]
    trap_and_mae_impact: dict[str, Any]
    production_readiness: dict[str, Any]
    final_verdict: dict[str, Any]
    per_signal_details: dict[str, list[dict[str, Any]]]
    conclusions: list[str]
    execution_time_seconds: float


class SellV6ReplayValidationResearch:
    """Combined single-pass SELL_V5 + SELL_V6 replay on 120-day NIFTY50 5M."""

    def __init__(self) -> None:
        self.v5_engine = V5CandidateEngine()
        self.v6_engine = SellV6CandidateEngine()
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=TRADING_DAYS_REPLAY,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    def _replay_combined(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        replay_bars: list[int],
        moves: list[_CheapMoveCandidate],
        daily_ranges: dict[date, float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, int]]:
        v5_signals: list[dict[str, Any]] = []
        v6_signals: list[dict[str, Any]] = []
        v5_emitted: set[int] = set()
        v6_emitted: set[int] = set()
        v5_rejections: dict[str, int] = {}
        v6_rejections: dict[str, int] = {}
        total = len(replay_bars)
        log_every = max(total // 20, 1)
        started = time.perf_counter()

        for index, bar in enumerate(replay_bars):
            if index > 0 and index % log_every == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "SELL V5+V6 replay: %s/%s bars (%.0f%%) elapsed %.0fs | V5=%s V6=%s",
                    index,
                    total,
                    index / total * 100,
                    elapsed,
                    len(v5_signals),
                    len(v6_signals),
                )

            if bar < PRE_EXPANSION_LOOKBACK or bar >= len(frame) - FORWARD_BARS:
                continue

            v5_eval = self.v5_engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=v5_emitted,
            )
            v6_eval = self.v6_engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=v6_emitted,
            )

            if v5_eval["verdict"] == "SELL":
                v5_signals.append(
                    _enrich_sell_signal(
                        v5_eval,
                        engine_version="SELL_V5",
                        model_id="LDM-SELL-V5",
                        moves=moves,
                        frame=frame,
                        daily_ranges=daily_ranges,
                    ),
                )
                v5_emitted.add(bar)
            else:
                for reason in v5_eval["layer5"]["reason_codes"]:
                    v5_rejections[reason] = v5_rejections.get(reason, 0) + 1

            if v6_eval["verdict"] == "SELL":
                v6_signals.append(
                    _enrich_sell_signal(
                        v6_eval,
                        engine_version="SELL_V6",
                        model_id="LDM-SELL-V6",
                        moves=moves,
                        frame=frame,
                        daily_ranges=daily_ranges,
                    ),
                )
                v6_emitted.add(bar)
            else:
                for reason in v6_eval["layer5"]["reason_codes"]:
                    v6_rejections[reason] = v6_rejections.get(reason, 0) + 1

        logger.info(
            "SELL replay complete: V5=%s V6=%s signals in %.0fs",
            len(v5_signals),
            len(v6_signals),
            time.perf_counter() - started,
        )
        return v5_signals, v6_signals, v5_rejections, v6_rejections

    def run(self, metadata: dict[str, Any]) -> SellV6ReplayValidationReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = end - timedelta(days=120)

        logger.info(
            "SELL_V6 replay validation starting: %s days, %s 5M",
            TRADING_DAYS_REPLAY,
            DEFAULT_SYMBOL,
        )

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=TRADING_DAYS_REPLAY,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        replay_dates = _last_n_trading_day_set(frame, TRADING_DAYS_REPLAY)
        train_dates, validate_dates = _split_trading_day_sets(replay_dates)
        bar_dates = pd.to_datetime(frame["Date"]).dt.date
        replay_bars = [index for index, day in enumerate(bar_dates) if day in replay_dates]
        daily_ranges = _daily_range_lookup(frame)

        logger.info("Loading enriched context and intel frames...")
        enriched = _attach_ema22(self.v5_engine.context_builder.enrich(frame))
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        intel_frames: dict[str, pd.DataFrame] = {"5M": self.v5_engine.intelligence.enrich(frame)}
        for tf in ("15M", "1H"):
            tf_path = filter_engine._ensure_pipeline(tf, start, end)
            tf_frame = pd.read_csv(tf_path).reset_index(drop=True)
            intel_frames[tf] = self.v5_engine.intelligence.enrich(tf_frame)
        intel_frames["1D"] = self.v5_engine.intelligence.enrich(
            self.v5_engine._resample_daily(intel_frames["1H"]),
        )

        logger.info("Detecting bearish moves (threshold=%s)...", MOVE_DETECTION_THRESHOLD)
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, MOVE_DETECTION_THRESHOLD),
        )

        v5_signals, v6_signals, v5_rej, v6_rej = self._replay_combined(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            replay_bars=replay_bars,
            moves=moves,
            daily_ranges=daily_ranges,
        )

        v5_stats = _build_statistics(v5_signals, trading_days=TRADING_DAYS_REPLAY)
        v6_stats = _build_statistics(v6_signals, trading_days=TRADING_DAYS_REPLAY)
        v5_capture = _point_capture(moves, v5_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        v6_capture = _point_capture(moves, v6_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)

        v5_train = _filter_signals_by_dates(v5_signals, frame, train_dates)
        v6_train = _filter_signals_by_dates(v6_signals, frame, train_dates)
        v5_validate = _filter_signals_by_dates(v5_signals, frame, validate_dates)
        v6_validate = _filter_signals_by_dates(v6_signals, frame, validate_dates)

        v5_train_stats = _walk_forward_metrics(v5_train, period_days=len(train_dates))
        v6_train_stats = _walk_forward_metrics(v6_train, period_days=len(train_dates))
        v5_validate_stats = _walk_forward_metrics(v5_validate, period_days=len(validate_dates))
        v6_validate_stats = _walk_forward_metrics(v6_validate, period_days=len(validate_dates))

        walk_forward = {
            "train": {
                "sell_v5": v5_train_stats,
                "sell_v6": v6_train_stats,
            },
            "validate": {
                "sell_v5": v5_validate_stats,
                "sell_v6": v6_validate_stats,
            },
            **_walk_forward_stability(v5_train_stats, v5_validate_stats, v6_train_stats, v6_validate_stats),
        }

        v5_row = _comparison_row(v5_stats, v5_capture)
        v6_row = _comparison_row(v6_stats, v6_capture)
        delta_row = _delta_row(v5_row, v6_row)

        removed_analysis = _removed_trade_analysis(v5_signals, v6_signals)
        removed_signals = [
            signal for signal in v5_signals if signal["bar"] not in {s["bar"] for s in v6_signals}
        ]
        regime_analysis = _regime_analysis(v5_signals, v6_signals, removed_signals)

        pf_reconciliation = _audit_pf_reconciliation(v6_stats.get("profit_factor"))

        trap_v5 = sum(
            1 for signal in v5_signals if signal.get("classification") in {"Bear Trap", "Gap Failure"}
        )
        trap_v6 = sum(
            1 for signal in v6_signals if signal.get("classification") in {"Bear Trap", "Gap Failure"}
        )
        trap_and_mae = {
            "trap_count_v5": trap_v5,
            "trap_count_v6": trap_v6,
            "trap_reduction": trap_v5 - trap_v6,
            "trap_reduction_pct": round((trap_v5 - trap_v6) / max(trap_v5, 1) * 100, 2),
            "average_mae_v5": v5_stats.get("average_mae"),
            "average_mae_v6": v6_stats.get("average_mae"),
            "mae_reduction_points": round(
                (v5_stats.get("average_mae") or 0) - (v6_stats.get("average_mae") or 0),
                2,
            ),
            "expectancy_v5": v5_stats.get("expectancy"),
            "expectancy_v6": v6_stats.get("expectancy"),
            "expectancy_delta": delta_row.get("expectancy_delta"),
        }

        readiness = _production_readiness_score(
            v6_stats=v6_stats,
            v6_capture=v6_capture,
            walk_forward=walk_forward,
            pf_reconciliation=pf_reconciliation,
        )
        final_verdict = _final_verdict(
            v5_stats=v5_stats,
            v6_stats=v6_stats,
            removed_analysis=removed_analysis,
            walk_forward=walk_forward,
            pf_reconciliation=pf_reconciliation,
            readiness=readiness,
        )

        replay_start = min(replay_dates).isoformat() if replay_dates else ""
        replay_end = max(replay_dates).isoformat() if replay_dates else ""

        reference: dict[str, Any] = {}
        if DEFAULT_AUDIT_PATH.exists():
            audit = json.loads(DEFAULT_AUDIT_PATH.read_text(encoding="utf-8"))
            sell_best = (
                audit.get("proposed_filters", {})
                .get("sell_v5", {})
                .get("best_gate_passing_filter", {})
            )
            reference["production_edge_audit"] = {
                "source": DEFAULT_AUDIT_PATH.name,
                "predicted_v6_pf": sell_best.get("profit_factor"),
                "predicted_v5_pf": audit.get("sell_v5_winner_loser_analysis", {})
                .get("baseline", {})
                .get("profit_factor"),
                "filter_label": sell_best.get("label"),
            }
        if DEFAULT_V5_REPORT_PATH.exists():
            v5_ref = json.loads(DEFAULT_V5_REPORT_PATH.read_text(encoding="utf-8"))
            reference["v5_candidate_validation"] = {
                "source": DEFAULT_V5_REPORT_PATH.name,
                "v5_signals": v5_ref.get("comparison", {})
                .get("v5_candidate", {})
                .get("signals_emitted_count"),
            }

        conclusions = [
            f"SELL_V5={len(v5_signals)} vs SELL_V6={len(v6_signals)} signals over {TRADING_DAYS_REPLAY} days.",
            (
                f"WR V5 {v5_stats.get('win_rate_pct')}% → V6 {v6_stats.get('win_rate_pct')}% "
                f"({delta_row.get('wr_delta_pp')}pp)."
            ),
            (
                f"PF V5 {v5_stats.get('profit_factor')} → V6 {v6_stats.get('profit_factor')} "
                f"(delta {delta_row.get('pf_delta')}); audit survival {pf_reconciliation.get('verdict')}."
            ),
            (
                f"Frequency V5 {v5_stats.get('signals_per_month')}/mo → "
                f"V6 {v6_stats.get('signals_per_month')}/mo ({delta_row.get('signals_per_month_delta')})."
            ),
            removed_analysis.get("summary", ""),
            (
                f"Walk-forward V6 stable={walk_forward.get('sell_v6_stable')}; "
                f"validate PF V5 {v5_validate_stats.get('profit_factor')} vs "
                f"V6 {v6_validate_stats.get('profit_factor')}."
            ),
            f"Final verdict: {final_verdict.get('can_sell_v6_replace_sell_v5')} — {final_verdict.get('rationale')}",
        ]

        return SellV6ReplayValidationReport(
            report_type="SELL_V6 Replay Validation",
            engines_compared=["SELL_V5", "SELL_V6"],
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            trading_days_replayed=TRADING_DAYS_REPLAY,
            replay_start_date=replay_start,
            replay_end_date=replay_end,
            v6_change_summary={
                "base_engine": "SELL_V5 (V4 stack + VWAP Below OR Rejected)",
                "v6_engine": "SELL_V6 (V5 stack + VWAP Below only)",
                "unchanged_from_v5": [
                    "Failed Breakout",
                    "HTF Bearish",
                    "EMA22 + EMA200 Context",
                    "Location filters",
                    "Confirmation optional",
                    "Volume bucket gate",
                ],
                "modified": {
                    "vwap_gate": {
                        "v5": V5_VWAP_GATE_RULE,
                        "v6": V6_VWAP_GATE_RULE,
                        "allowed_vwap_states_v6": sorted(V6_ALLOWED_VWAP_STATES),
                        "removed_vwap_states": ["Rejected"],
                    },
                },
            },
            methodology={
                "research_only": True,
                "single_pass_replay": True,
                "combined_replay": "SELL_V5 + SELL_V6 evaluated once per bar",
                "walk_forward_split": f"train {TRAIN_TRADING_DAYS} / validate {VALIDATE_TRADING_DAYS} trading days",
                "no_look_ahead": True,
                "no_buy_v3_modification": True,
                "move_detection_threshold": MOVE_DETECTION_THRESHOLD,
                "point_capture_thresholds": list(POINT_CAPTURE_THRESHOLDS),
                "audit_pf_targets": {"v5_baseline": AUDIT_V5_PF, "v6_expected": AUDIT_V6_PF},
                "reference_exports": reference,
            },
            comparison_table={
                "sell_v5": v5_row,
                "sell_v6": v6_row,
                "delta_v6_minus_v5": delta_row,
                "layer_rejection_summary": {
                    "sell_v5": v5_rej,
                    "sell_v6": v6_rej,
                },
                "overall_statistics": {
                    "sell_v5": v5_stats,
                    "sell_v6": v6_stats,
                },
                "point_capture": {
                    "sell_v5": v5_capture,
                    "sell_v6": v6_capture,
                },
            },
            walk_forward=walk_forward,
            pf_audit_reconciliation=pf_reconciliation,
            removed_trade_analysis=removed_analysis,
            regime_analysis=regime_analysis,
            trap_and_mae_impact=trap_and_mae,
            production_readiness=readiness,
            final_verdict=final_verdict,
            per_signal_details={
                "sell_v5": v5_signals,
                "sell_v6": v6_signals,
            },
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: SellV6ReplayValidationReport, report_path: Path) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("SELL_V6 replay validation exported: %s", report_path)
        return report_path


def generate_sell_v6_replay_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> SellV6ReplayValidationReport:
    """Run SELL_V5 vs SELL_V6 replay validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise SellV6ReplayValidationError(f"Filter research report not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    research = SellV6ReplayValidationResearch()
    report = research.run(metadata)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_sell_v6_replay_validation_report()
    except SellV6ReplayValidationError as exc:
        logger.error("SELL_V6 replay validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected SELL_V6 replay validation error")
        return 1

    v5 = report.comparison_table["sell_v5"]
    v6 = report.comparison_table["sell_v6"]
    pf_audit = report.pf_audit_reconciliation
    verdict = report.final_verdict

    print("SELL_V6 Replay Validation Summary")
    print(f"V5 signals: {v5['signals_emitted']} | V6 signals: {v6['signals_emitted']}")
    print(f"V5 WR: {v5['win_rate_pct']}% | V6 WR: {v6['win_rate_pct']}%")
    print(f"V5 PF: {v5['profit_factor']} | V6 PF: {v6['profit_factor']}")
    print(f"Audit PF survives replay: {pf_audit.get('verdict')} (expected {AUDIT_V6_PF})")
    print(f"Replace V5 with V6: {verdict.get('can_sell_v6_replace_sell_v5')}")
    print(f"Production readiness: {report.production_readiness.get('score')}/100")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
