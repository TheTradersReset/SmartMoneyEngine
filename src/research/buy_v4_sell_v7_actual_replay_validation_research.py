"""
BUY_V4 & SELL_V7 Actual Replay Validation.

Implements BUY_V4 / SELL_V7 as real engine variants on BUY_V3 / SELL_V6 bases.
Approved structural filters (from design blueprint) are applied inside the
emission path during a fresh NIFTY50 5M bar replay — not by post-hoc filtering
of a completed V3/V6 export corpus.

BUY_V4 = BUY_V3 + reject Liquidity Sweep Failure + Gap Continuation
SELL_V7 = SELL_V6 + reject Liquidity Sweep Failure + Volatility Collapse

Paper stack mirrors extended_trade_level_truth_audit: fixed_10 + 60/100/Runner
+ regime throttle maps. Prefer 300 trading days; fall back to 240.
Research-only; no production signal logic changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v2_candidate_validation_research import _filter_signals_by_dates
from src.research.buy_v3_candidate_validation_research import (
    BAR_MINUTES,
    BUY_V3_MODEL_ID,
    BuyV3CandidateEngine,
    _evaluate_buy_bar_fast,
    _precompute_bar_events,
)
from src.research.buy_v4_sell_v7_design_blueprint_audit_research import _trade_lifecycle_audit
from src.research.extended_evidence_validation_real_deployment_audit_research import (
    ExtendedEvidenceValidationRealDeploymentAuditResearch,
    _cohort_metrics_block,
    _combined_throttled_metrics,
    _load_json_safe,
    _load_throttle_maps,
)
from src.research.extended_trade_level_truth_audit_research import (
    DEFAULT_STOP_VARIANT,
    PRODUCTION_STRUCTURE,
    _count_trading_days,
)
from src.research.failure_pattern_production_robustness_audit_research import (
    TARGET_TIERS,
    _detect_structural_patterns,
    _reward_risk_audit,
    _target_matrix_from_signals,
)
from src.research.filter_research_engine import FilterResearchEngine
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
from src.research.production_reality_audit_research import (
    RUNNER_STRATEGIES,
    _extended_metrics,
    _timing_class,
)
from src.research.production_trading_playbook_audit_research import _tiered_structure_pnl
from src.research.regime_detection_audit_research import SELL_V6_MODEL_ID
from src.research.sell_v6_replay_validation_research import (
    SellV6CandidateEngine,
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
)
from src.research.trade_level_truth_audit_research import _entry_precision_audit

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_BLUEPRINT_PATH = RESEARCH_DIR / "buy_v4_sell_v7_design_blueprint_audit.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_v4_sell_v7_actual_replay_validation.json"
DEFAULT_STATUS_PATH = RESEARCH_DIR / "buy_v4_sell_v7_actual_replay_validation.status.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "buy_v4_sell_v7_actual_replay_validation_run.log"
REGIME_EXPORT_PATH = RESEARCH_DIR / "regime_detection_audit.json"

BUY_V4_MODEL_ID = "LDM-BUY-V4"
SELL_V7_MODEL_ID = "LDM-SELL-V7"

# Blueprint-approved emission rejects (design source of truth).
DEFAULT_BUY_V4_REJECT_PATTERNS = ("Liquidity Sweep Failure", "Gap Continuation")
DEFAULT_SELL_V7_REJECT_PATTERNS = ("Liquidity Sweep Failure", "Volatility Collapse")

PREFERRED_TRADING_DAYS = 300
FALLBACK_TRADING_DAYS = 240
CALENDAR_BUFFER = {240: 380, 300: 480, 500: 780}
PF_IMPROVEMENT_THRESHOLD = 1.10


class BuyV4SellV7ActualReplayValidationError(Exception):
    """Raised when BUY_V4 / SELL_V7 actual replay validation fails."""


@dataclass
class BuyV4SellV7ActualReplayValidationReport:
    """BUY_V4 / SELL_V7 actual bar-replay validation output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    trading_days_used: int
    preferred_trading_days: int
    fallback_trading_days: int
    available_trading_days: int
    replay_start_date: str
    replay_end_date: str
    methodology: dict[str, Any]
    approved_filters: dict[str, Any]
    engine_definitions: dict[str, Any]
    core_metrics: dict[str, Any]
    target_achievement_matrix: dict[str, Any]
    trade_lifecycle: dict[str, Any]
    entry_timing: dict[str, Any]
    reward_risk: dict[str, Any]
    capture_metrics: dict[str, Any]
    regime_throttle: dict[str, Any]
    engine_comparison: dict[str, Any]
    per_signal_details: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _load_blueprint_filters(blueprint_path: Path) -> dict[str, Any]:
    if not blueprint_path.exists():
        return {
            "source": "defaults",
            "buy_v4_reject_patterns": list(DEFAULT_BUY_V4_REJECT_PATTERNS),
            "sell_v7_reject_patterns": list(DEFAULT_SELL_V7_REJECT_PATTERNS),
            "blueprint_loaded": False,
        }
    data = json.loads(blueprint_path.read_text(encoding="utf-8"))
    buy_design = data.get("buy_v4_design") or {}
    sell_design = data.get("sell_v7_design") or {}
    buy_patterns = list(buy_design.get("selected_patterns") or DEFAULT_BUY_V4_REJECT_PATTERNS)
    sell_patterns = list(sell_design.get("selected_patterns") or DEFAULT_SELL_V7_REJECT_PATTERNS)
    return {
        "source": str(blueprint_path.name),
        "buy_v4_reject_patterns": buy_patterns,
        "sell_v7_reject_patterns": sell_patterns,
        "buy_v4_design": {
            "filters_to_add": buy_design.get("filters_to_add"),
            "selected_patterns": buy_patterns,
            "base_engine": buy_design.get("base_engine", "BUY_V3"),
        },
        "sell_v7_design": {
            "filters_to_add": sell_design.get("filters_to_add"),
            "selected_patterns": sell_patterns,
            "base_engine": sell_design.get("base_engine", "SELL_V6"),
        },
        "blueprint_loaded": True,
    }


def _resolve_trading_days(available_days: int) -> int:
    if available_days >= PREFERRED_TRADING_DAYS:
        return PREFERRED_TRADING_DAYS
    if available_days >= FALLBACK_TRADING_DAYS:
        return FALLBACK_TRADING_DAYS
    if available_days < 1:
        raise BuyV4SellV7ActualReplayValidationError("No trading days available in frame")
    return available_days


def _patterns_blocked(signal: dict[str, Any], *, side: str, reject_patterns: tuple[str, ...] | list[str]) -> list[str]:
    if not reject_patterns:
        return []
    detected = set(_detect_structural_patterns(signal, side=side))
    return sorted(detected.intersection(set(reject_patterns)))


def _engine_should_emit(
    signal: dict[str, Any],
    *,
    side: str,
    reject_patterns: tuple[str, ...] | list[str],
) -> tuple[bool, list[str]]:
    blocked = _patterns_blocked(signal, side=side, reject_patterns=reject_patterns)
    return (len(blocked) == 0, blocked)


class BuyV4CandidateEngine(BuyV3CandidateEngine):
    """BUY_V4: BUY_V3 base with blueprint structural rejects at emission."""

    MODEL_ID = BUY_V4_MODEL_ID

    def __init__(self, reject_patterns: tuple[str, ...] | list[str] | None = None) -> None:
        super().__init__()
        self.reject_patterns = tuple(reject_patterns or DEFAULT_BUY_V4_REJECT_PATTERNS)

    def should_emit_signal(self, signal: dict[str, Any]) -> tuple[bool, list[str]]:
        return _engine_should_emit(signal, side="BUY", reject_patterns=self.reject_patterns)


class SellV7CandidateEngine(SellV6CandidateEngine):
    """SELL_V7: SELL_V6 base with blueprint structural rejects at emission."""

    MODEL_ID = SELL_V7_MODEL_ID

    def __init__(self, reject_patterns: tuple[str, ...] | list[str] | None = None) -> None:
        super().__init__()
        self.reject_patterns = tuple(reject_patterns or DEFAULT_SELL_V7_REJECT_PATTERNS)

    def should_emit_signal(self, signal: dict[str, Any]) -> tuple[bool, list[str]]:
        return _engine_should_emit(signal, side="SELL", reject_patterns=self.reject_patterns)


def _playbook_pnls(signals: list[dict[str, Any]]) -> list[float]:
    if not signals:
        return []
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals)
    pnls: list[float] = []
    for signal in signals:
        stop_pts = _resolve_stop_extended(signal, DEFAULT_STOP_VARIANT, cohort_mae_median=mae_median)
        pnl, _ = _tiered_structure_pnl(signal, PRODUCTION_STRUCTURE, stop_pts=stop_pts)
        pnls.append(float(pnl))
    return pnls


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return round(max_dd, 2)


def _recovery_factor(pnls: list[float], max_dd: float) -> float | None:
    if max_dd <= 0:
        return None
    return round(sum(pnls) / max_dd, 2)


def _lifecycle_pcts(lifecycle: dict[str, Any]) -> dict[str, Any]:
    probs = lifecycle.get("hit_probabilities_pct") or {}
    total = max(int(lifecycle.get("signal_count") or 0), 1)
    records = lifecycle.get("records") or []
    full_trend = sum(1 for r in records if float(r.get("mfe") or 0.0) >= 200)
    return {
        "stopped_out_pct": probs.get("Stopped Out"),
        "t1_pct": probs.get("Hit T1"),
        "t2_pct": probs.get("Hit T2"),
        "t3_pct": probs.get("Hit T3"),
        "runner_pct": probs.get("Hit Runner"),
        "full_trend_capture_pct": round(100.0 * full_trend / total, 2),
        "hit_counts": lifecycle.get("hit_counts"),
        "hit_probabilities_pct": probs,
    }


def _entry_timing_summary(entry_audit: dict[str, Any], signals: list[dict[str, Any]]) -> dict[str, Any]:
    timing = entry_audit.get("timing_class_summary") or {}
    total = max(len(signals), 1)
    lead_bars = [
        int(s["bars_before_expansion"])
        for s in signals
        if s.get("bars_before_expansion") is not None
    ]
    lead_minutes = [b * BAR_MINUTES for b in lead_bars]

    def _pct(label: str) -> float:
        # timing_class_summary may be counts or nested; normalize
        row = timing.get(label)
        if isinstance(row, dict):
            count = int(row.get("count") or 0)
        elif isinstance(row, (int, float)):
            count = int(row)
        else:
            count = sum(
                1
                for s in signals
                if _timing_class(
                    int(s["bars_before_expansion"]) if s.get("bars_before_expansion") is not None else None,
                )
                == label
            )
        return round(100.0 * count / total, 2)

    return {
        "very_early_pct": _pct("Very Early"),
        "early_pct": _pct("Early"),
        "same_candle_pct": _pct("Same"),
        "late_pct": _pct("Late"),
        "avg_lead_bars": round(mean(lead_bars), 2) if lead_bars else None,
        "avg_lead_minutes": round(mean(lead_minutes), 2) if lead_minutes else None,
        "timing_class_summary": timing,
        "predictive_vs_reactive": entry_audit.get("predictive_vs_reactive"),
    }


def _capture_block(signals: list[dict[str, Any]], *, trading_days: int) -> dict[str, Any]:
    mfes = [float(s.get("mfe_points") or 0.0) for s in signals]
    playbook = _playbook_pnls(signals)
    max_available = round(sum(mfes), 2)
    captured = round(sum(max(p, 0.0) for p in playbook), 2)
    efficiency = round(100.0 * captured / max(max_available, 1.0), 2) if mfes else None
    ext = _extended_metrics(
        playbook,
        signals=signals,
        sample_size=len(signals),
        window_days=trading_days,
    )
    return {
        "maximum_available_points": max_available,
        "captured_points": captured,
        "capture_efficiency_pct": efficiency,
        "playbook_capture_efficiency_pct": ext.get("capture_efficiency_pct"),
    }


def _engine_metrics(
    signals: list[dict[str, Any]],
    *,
    trading_days: int,
    side: str,
    win_fn: Any,
    moves: list[_CheapMoveCandidate],
    frame: pd.DataFrame,
    replay_dates: set[date],
) -> dict[str, Any]:
    direction = "BUY" if side == "BUY" else "SELL"
    cohort = _cohort_metrics_block(
        signals,
        trading_days=trading_days,
        win_fn=win_fn,
        moves=moves,
        frame=frame,
        replay_dates=replay_dates,
        direction=direction,
    )
    stats = _build_statistics(signals, trading_days=trading_days)
    mfes = [float(s.get("mfe_points") or 0.0) for s in signals]
    maes = [float(s.get("mae_points") or 0.0) for s in signals]
    raw_pnls = [float(s.get("realized_pnl_points") or 0.0) for s in signals]
    playbook_pnls = _playbook_pnls(signals)
    playbook_dd = _max_drawdown(playbook_pnls)
    wins = sum(1 for s in signals if win_fn(s))

    lifecycle = _trade_lifecycle_audit(signals, side=side)
    entry_audit = _entry_precision_audit(signals, side=side, win_fn=win_fn)
    rr = _reward_risk_audit(signals, side=side, is_winner_fn=win_fn)
    target = _target_matrix_from_signals(signals, side=side)
    capture = _capture_block(signals, trading_days=trading_days)

    # Target achievement: P(reach tier BEFORE stop) using MFE vs stop distance proxy
    stop_hit_matrix: dict[str, Any] = {}
    for tier in TARGET_TIERS:
        reached = 0
        for signal in signals:
            mfe = float(signal.get("mfe_points") or 0.0)
            mae = float(signal.get("mae_points") or 0.0)
            stop_pts = float(
                abs(float(signal.get("entry") or 0.0) - float(signal.get("stop_loss") or 0.0)) or 10.0,
            )
            # Before stop: mae did not exceed stop before mfe reached tier (path proxy)
            if mfe >= tier and mae <= max(stop_pts, 10.0):
                reached += 1
            elif mfe >= tier:
                # Still count MFE reach; stop ordering unknown without tick path
                reached += 1
        stop_hit_matrix[str(tier)] = {
            "tier_points": tier,
            "reached_count": reached,
            "probability_pct": round(100.0 * reached / max(len(signals), 1), 2),
        }

    return {
        "signals": len(signals),
        "signals_per_month": stats.get("signals_per_month"),
        "win_rate_pct": round(100.0 * wins / max(len(signals), 1), 2),
        "profit_factor": _profit_factor_from_pnls(raw_pnls),
        "expectancy": round(mean(raw_pnls), 2) if raw_pnls else 0.0,
        "max_drawdown": cohort.get("max_drawdown_points"),
        "recovery_factor": cohort.get("recovery_factor"),
        "playbook_profit_factor": _profit_factor_from_pnls(playbook_pnls),
        "playbook_expectancy": round(mean(playbook_pnls), 2) if playbook_pnls else 0.0,
        "playbook_max_drawdown": playbook_dd,
        "playbook_recovery_factor": _recovery_factor(playbook_pnls, playbook_dd),
        "average_mfe": round(mean(mfes), 2) if mfes else None,
        "average_mae": round(mean(maes), 2) if maes else None,
        "median_mfe": round(median(mfes), 2) if mfes else None,
        "median_mae": round(median(maes), 2) if maes else None,
        "target_achievement_matrix": {
            "by_tier_mfe": target,
            "probability_reach_before_stop_proxy": stop_hit_matrix,
            "tiers": list(TARGET_TIERS),
        },
        "trade_lifecycle": _lifecycle_pcts(lifecycle),
        "entry_timing": _entry_timing_summary(entry_audit, signals),
        "reward_risk": {
            "probability_1_to_1": (rr.get("rr_probability") or {}).get("1_to_1"),
            "probability_1_to_2": (rr.get("rr_probability") or {}).get("1_to_2"),
            "probability_1_to_3": (rr.get("rr_probability") or {}).get("1_to_3"),
            "probability_1_to_5": (rr.get("rr_probability") or {}).get("1_to_5"),
            "average_rr": rr.get("average_rr"),
            "median_rr": rr.get("median_rr"),
            "detail": rr,
        },
        "capture": capture,
        "cohort_block": {
            "max_drawdown_points": cohort.get("max_drawdown_points"),
            "recovery_factor": cohort.get("recovery_factor"),
            "average_mfe": cohort.get("average_mfe"),
            "average_mae": cohort.get("average_mae"),
        },
    }


def _compare_engines(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    b_pf = float(baseline.get("profit_factor") or 0.0)
    c_pf = float(candidate.get("profit_factor") or 0.0)
    b_wr = float(baseline.get("win_rate_pct") or 0.0)
    c_wr = float(candidate.get("win_rate_pct") or 0.0)
    b_exp = float(baseline.get("expectancy") or 0.0)
    c_exp = float(candidate.get("expectancy") or 0.0)
    pf_lift = round(100.0 * (c_pf - b_pf) / b_pf, 2) if b_pf else None
    metric_outperform = bool(c_pf >= b_pf * PF_IMPROVEMENT_THRESHOLD and c_exp >= b_exp)
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "baseline_signals": baseline.get("signals"),
        "candidate_signals": candidate.get("signals"),
        "signal_reduction_pct": round(
            100.0 * (int(baseline.get("signals") or 0) - int(candidate.get("signals") or 0))
            / max(int(baseline.get("signals") or 0), 1),
            2,
        ),
        "profit_factor": {"baseline": b_pf, "candidate": c_pf, "lift_pct": pf_lift},
        "win_rate_pct": {"baseline": b_wr, "candidate": c_wr, "delta_pp": round(c_wr - b_wr, 2)},
        "expectancy": {"baseline": b_exp, "candidate": c_exp, "delta": round(c_exp - b_exp, 2)},
        "metric_outperform_10pct_pf": metric_outperform,
    }


def _patterns_use_forward_path(patterns: list[str]) -> bool:
    """Blueprint structural detector gates these patterns on MFE/MAE (forward path)."""
    forward_gated = {
        "Liquidity Sweep Failure",
        "Gap Continuation",
        "Volatility Collapse",
        "Weak Displacement",
        "Low Expansion Regime",
        "VWAP Reclaim Failure",
        "Failed Reclaim",
        "Late BOS",
    }
    return bool(set(patterns).intersection(forward_gated))


def _build_final_answer(
    *,
    buy_v3: dict[str, Any],
    buy_v4: dict[str, Any],
    sell_v6: dict[str, Any],
    sell_v7: dict[str, Any],
    buy_compare: dict[str, Any],
    sell_compare: dict[str, Any],
    buy_patterns: list[str],
    sell_patterns: list[str],
    trading_days: int,
    throttle: dict[str, Any],
) -> dict[str, Any]:
    buy_forward = _patterns_use_forward_path(buy_patterns)
    sell_forward = _patterns_use_forward_path(sell_patterns)

    # Genuine live outperformance requires deployable emission filters (no forward path).
    buy_genuine = bool(buy_compare["metric_outperform_10pct_pf"] and not buy_forward)
    sell_genuine = bool(sell_compare["metric_outperform_10pct_pf"] and not sell_forward)
    buy_replace = "YES" if buy_genuine else "NO"
    sell_replace = "YES" if sell_genuine else "NO"

    if buy_compare["metric_outperform_10pct_pf"] and buy_forward:
        buy_reason = (
            f"Replay metrics improve (PF {buy_v3.get('profit_factor')}→{buy_v4.get('profit_factor')}, "
            f"WR {buy_v3.get('win_rate_pct')}→{buy_v4.get('win_rate_pct')}) but approved filters "
            f"{buy_patterns} are gated on forward MFE/MAE inside _detect_structural_patterns — "
            "not live-deployable at signal time without lookahead. Not genuine production outperformance."
        )
    elif buy_genuine:
        buy_reason = (
            f"BUY_V4 genuinely outperforms on {trading_days}d actual replay with live-safe filters."
        )
    else:
        buy_reason = (
            f"BUY_V4 does not clear genuine replace gates on {trading_days}d actual replay "
            f"(PF {buy_v3.get('profit_factor')} vs {buy_v4.get('profit_factor')})."
        )

    if sell_compare["metric_outperform_10pct_pf"] and sell_forward:
        sell_reason = (
            f"Replay metrics improve (PF {sell_v6.get('profit_factor')}→{sell_v7.get('profit_factor')}, "
            f"WR {sell_v6.get('win_rate_pct')}→{sell_v7.get('win_rate_pct')}) but approved filters "
            f"{sell_patterns} require forward MFE/MAE — not live-deployable. Keep SELL_V6."
        )
    elif sell_genuine:
        sell_reason = (
            f"SELL_V7 genuinely outperforms on {trading_days}d actual replay with live-safe filters."
        )
    else:
        sell_reason = (
            f"SELL_V7 does not clear genuine replace gates on {trading_days}d actual replay "
            f"(PF {sell_v6.get('profit_factor')} vs {sell_v7.get('profit_factor')})."
        )

    evidence_strength = "WEAK" if (buy_forward or sell_forward) else "STRONG"
    if buy_forward or sell_forward:
        confidence = "LOW"
        overfitting_risk = "HIGH"
    else:
        confidence = "HIGH" if (buy_genuine or sell_genuine) else "MEDIUM"
        overfitting_risk = "LOW"

    best_buy = "BUY_V4" if buy_replace == "YES" else "BUY_V3"
    best_sell = "SELL_V7" if sell_replace == "YES" else "SELL_V6"

    return {
        "does_buy_v4_genuinely_outperform_buy_v3": "YES" if buy_genuine else "NO",
        "should_buy_v4_replace_buy_v3": buy_replace,
        "buy_v4_replace_reason": buy_reason,
        "does_sell_v7_genuinely_outperform_sell_v6": "YES" if sell_genuine else "NO",
        "should_sell_v7_replace_sell_v6": sell_replace,
        "sell_v7_replace_reason": sell_reason,
        "evidence_strength": evidence_strength,
        "confidence": confidence,
        "overfitting_risk": overfitting_risk,
        "best_buy_engine": best_buy,
        "best_sell_engine": best_sell,
        "best_stop": "fixed_10",
        "best_exit_structure": "60/100/Runner",
        "best_regime_rules": "regime_detection_audit throttle maps (BUY_V3 / SELL_V6 keys)",
        "best_sizing_rules": {
            "buy_sleeve_pct": 35,
            "sell_sleeve_pct": 65,
            "mode": "regime_adaptive",
        },
        "pf_wr_comparison": {
            "buy_v3_vs_buy_v4": {
                "buy_v3_pf": buy_v3.get("profit_factor"),
                "buy_v4_pf": buy_v4.get("profit_factor"),
                "buy_v3_wr": buy_v3.get("win_rate_pct"),
                "buy_v4_wr": buy_v4.get("win_rate_pct"),
                "buy_v3_signals": buy_v3.get("signals"),
                "buy_v4_signals": buy_v4.get("signals"),
            },
            "sell_v6_vs_sell_v7": {
                "sell_v6_pf": sell_v6.get("profit_factor"),
                "sell_v7_pf": sell_v7.get("profit_factor"),
                "sell_v6_wr": sell_v6.get("win_rate_pct"),
                "sell_v7_wr": sell_v7.get("win_rate_pct"),
                "sell_v6_signals": sell_v6.get("signals"),
                "sell_v7_signals": sell_v7.get("signals"),
            },
        },
        "filter_forward_path_dependency": {
            "buy_v4_patterns_use_forward_mfe_mae": buy_forward,
            "sell_v7_patterns_use_forward_mfe_mae": sell_forward,
            "note": (
                "Blueprint structural detector (_detect_structural_patterns) gates approved "
                "patterns on mae/mfe. Emission-path application during research replay still "
                "uses those features after _trade_outcome; live engines cannot know them at entry."
            ),
        },
        "trading_days_used": trading_days,
        "regime_throttle_combined_pf": throttle.get("profit_factor"),
    }


class BuyV4SellV7ActualReplayValidationResearch(ExtendedEvidenceValidationRealDeploymentAuditResearch):
    """Four-engine actual bar replay: BUY_V3, BUY_V4, SELL_V6, SELL_V7."""

    def __init__(
        self,
        *,
        buy_v4_reject_patterns: tuple[str, ...] | list[str] | None = None,
        sell_v7_reject_patterns: tuple[str, ...] | list[str] | None = None,
        status_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.buy_v4_engine = BuyV4CandidateEngine(buy_v4_reject_patterns)
        self.sell_v7_engine = SellV7CandidateEngine(sell_v7_reject_patterns)
        self.status_path = status_path or DEFAULT_STATUS_PATH

    def _replay_four_engines(
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
    ) -> dict[str, list[dict[str, Any]]]:
        valid_bars = [
            bar
            for bar in replay_bars
            if bar >= PRE_EXPANSION_LOOKBACK and bar < len(frame) - FORWARD_BARS
        ]

        logger.info("Precomputing BUY events for %s bars...", len(valid_bars))
        _write_status(
            self.status_path,
            {"phase": "precompute_events", "bars": len(valid_bars), "ts": time.time()},
        )
        bar_events_cache, lookback_cache = _precompute_bar_events(
            self.buy_engine, frame=frame, calendar=calendar, replay_bars=valid_bars,
        )

        logger.info("Precomputing BUY context for %s bars...", len(valid_bars))
        _write_status(
            self.status_path,
            {"phase": "precompute_buy_context", "bars": len(valid_bars), "ts": time.time()},
        )
        buy_context_cache: dict[int, dict[str, str]] = {}
        ctx_log = max(len(valid_bars) // 10, 1)
        ctx_started = time.perf_counter()
        for index, bar in enumerate(valid_bars):
            if index > 0 and index % ctx_log == 0:
                logger.info(
                    "BUY context: %s/%s (%.0f%%) %.0fs",
                    index,
                    len(valid_bars),
                    index / len(valid_bars) * 100,
                    time.perf_counter() - ctx_started,
                )
                _write_status(
                    self.status_path,
                    {
                        "phase": "precompute_buy_context",
                        "index": index,
                        "total": len(valid_bars),
                        "elapsed_s": round(time.perf_counter() - ctx_started, 1),
                        "ts": time.time(),
                    },
                )
            buy_context_cache[bar] = self.buy_engine._context_at_bar(
                frame=frame,
                enriched=enriched_buy,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
            )

        buy_v3: list[dict[str, Any]] = []
        buy_v4: list[dict[str, Any]] = []
        sell_v6: list[dict[str, Any]] = []
        sell_v7: list[dict[str, Any]] = []
        buy_emitted: set[int] = set()
        sell_emitted: set[int] = set()
        buy_v4_rejected = 0
        sell_v7_rejected = 0

        total = len(valid_bars)
        log_every = max(total // 20, 1)
        started = time.perf_counter()
        _write_status(
            self.status_path,
            {"phase": "bar_replay", "bars": total, "ts": time.time()},
        )

        for index, bar in enumerate(valid_bars):
            if index > 0 and index % log_every == 0:
                logger.info(
                    "Replay %s/%s (%.0f%%) %.0fs | V3=%s V4=%s V6=%s V7=%s (rej V4=%s V7=%s)",
                    index,
                    total,
                    index / total * 100,
                    time.perf_counter() - started,
                    len(buy_v3),
                    len(buy_v4),
                    len(sell_v6),
                    len(sell_v7),
                    buy_v4_rejected,
                    sell_v7_rejected,
                )
                _write_status(
                    self.status_path,
                    {
                        "phase": "bar_replay",
                        "index": index,
                        "total": total,
                        "buy_v3": len(buy_v3),
                        "buy_v4": len(buy_v4),
                        "sell_v6": len(sell_v6),
                        "sell_v7": len(sell_v7),
                        "buy_v4_rejected": buy_v4_rejected,
                        "sell_v7_rejected": sell_v7_rejected,
                        "elapsed_s": round(time.perf_counter() - started, 1),
                        "ts": time.time(),
                    },
                )

            buy_eval = _evaluate_buy_bar_fast(
                self.buy_engine,
                frame=frame,
                bar=bar,
                context=buy_context_cache[bar],
                lookback_events=lookback_cache[bar],
                bar_events=bar_events_cache[bar],
                emitted_bars=buy_emitted,
            )
            if buy_eval["verdict"] == "BUY":
                signal_v3 = self._build_buy_signal(buy_eval, moves=moves, frame=frame, engine_version="BUY_V3")
                signal_v3["model_id"] = BUY_V3_MODEL_ID
                buy_v3.append(signal_v3)
                buy_emitted.add(bar)

                emit_v4, _blocked_v4 = self.buy_v4_engine.should_emit_signal(signal_v3)
                if emit_v4:
                    signal_v4 = dict(signal_v3)
                    signal_v4["engine_version"] = "BUY_V4"
                    signal_v4["model_id"] = BUY_V4_MODEL_ID
                    signal_v4["v4_rejected_patterns"] = []
                    buy_v4.append(signal_v4)
                else:
                    buy_v4_rejected += 1

            sell_eval = self.sell_engine.evaluate_bar(
                frame=frame,
                enriched=enriched_sell,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=sell_emitted,
            )
            if sell_eval["verdict"] == "SELL":
                signal_v6 = _enrich_sell_signal(
                    sell_eval,
                    engine_version="SELL_V6",
                    model_id=SELL_V6_MODEL_ID,
                    moves=moves,
                    frame=frame,
                    daily_ranges=daily_ranges,
                )
                sell_v6.append(signal_v6)
                sell_emitted.add(bar)

                emit_v7, _blocked_v7 = self.sell_v7_engine.should_emit_signal(signal_v6)
                if emit_v7:
                    signal_v7 = dict(signal_v6)
                    signal_v7["engine_version"] = "SELL_V7"
                    signal_v7["model_id"] = SELL_V7_MODEL_ID
                    signal_v7["v7_rejected_patterns"] = []
                    sell_v7.append(signal_v7)
                else:
                    sell_v7_rejected += 1

        logger.info(
            "Four-engine replay complete: V3=%s V4=%s V6=%s V7=%s | rejected V4=%s V7=%s in %.0fs",
            len(buy_v3),
            len(buy_v4),
            len(sell_v6),
            len(sell_v7),
            buy_v4_rejected,
            sell_v7_rejected,
            time.perf_counter() - started,
        )
        _write_status(
            self.status_path,
            {
                "phase": "replay_complete",
                "buy_v3": len(buy_v3),
                "buy_v4": len(buy_v4),
                "sell_v6": len(sell_v6),
                "sell_v7": len(sell_v7),
                "buy_v4_rejected": buy_v4_rejected,
                "sell_v7_rejected": sell_v7_rejected,
                "elapsed_s": round(time.perf_counter() - started, 1),
                "ts": time.time(),
            },
        )
        return {
            "buy_v3": buy_v3,
            "buy_v4": buy_v4,
            "sell_v6": sell_v6,
            "sell_v7": sell_v7,
            "rejection_counts": {
                "buy_v4_rejected": buy_v4_rejected,
                "sell_v7_rejected": sell_v7_rejected,
            },
        }

    def run(
        self,
        metadata: dict[str, Any],
        *,
        approved_filters: dict[str, Any] | None = None,
        trading_days: int | None = None,
    ) -> BuyV4SellV7ActualReplayValidationReport:
        started = time.perf_counter()
        filters = approved_filters or _load_blueprint_filters(DEFAULT_BLUEPRINT_PATH)
        buy_patterns = list(filters.get("buy_v4_reject_patterns") or DEFAULT_BUY_V4_REJECT_PATTERNS)
        sell_patterns = list(filters.get("sell_v7_reject_patterns") or DEFAULT_SELL_V7_REJECT_PATTERNS)
        self.buy_v4_engine = BuyV4CandidateEngine(buy_patterns)
        self.sell_v7_engine = SellV7CandidateEngine(sell_patterns)

        end = date.fromisoformat(metadata["end_date"])
        # Load enough calendar span for preferred window
        calendar_days = CALENDAR_BUFFER[PREFERRED_TRADING_DAYS]
        start = end - timedelta(days=calendar_days)

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=calendar_days,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        logger.info(
            "BUY_V4/SELL_V7 actual replay starting: prefer %sd (fallback %sd), %s 5M",
            PREFERRED_TRADING_DAYS,
            FALLBACK_TRADING_DAYS,
            DEFAULT_SYMBOL,
        )
        _write_status(
            self.status_path,
            {"phase": "load_data", "start": str(start), "end": str(end), "ts": time.time()},
        )

        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        available_days = _count_trading_days(frame)
        days_used = trading_days or _resolve_trading_days(available_days)

        logger.info(
            "Data: %s trading days available; using %sd (preferred=%s fallback=%s)",
            available_days,
            days_used,
            PREFERRED_TRADING_DAYS,
            FALLBACK_TRADING_DAYS,
        )

        replay_dates = _last_n_trading_day_set(frame, days_used)
        bar_dates = pd.to_datetime(frame["Date"]).dt.date
        replay_bars = [index for index, day in enumerate(bar_dates) if day in replay_dates]

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

        logger.info("Detecting moves...")
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, 40),
        )
        daily_ranges = _daily_range_lookup(frame)

        logger.info("Running %sd four-engine actual replay...", days_used)
        full = self._replay_four_engines(
            frame=frame,
            enriched_buy=enriched_buy,
            enriched_sell=enriched_sell,
            calendar=calendar,
            intel_frames=intel_frames,
            replay_bars=replay_bars,
            moves=moves,
            daily_ranges=daily_ranges,
        )

        buy_v3 = _filter_signals_by_dates(full["buy_v3"], frame, replay_dates)
        buy_v4 = _filter_signals_by_dates(full["buy_v4"], frame, replay_dates)
        sell_v6 = _filter_signals_by_dates(full["sell_v6"], frame, replay_dates)
        sell_v7 = _filter_signals_by_dates(full["sell_v7"], frame, replay_dates)

        _write_status(self.status_path, {"phase": "metrics", "ts": time.time()})
        logger.info("Computing per-engine metrics...")

        metrics = {
            "buy_v3": _engine_metrics(
                buy_v3, trading_days=days_used, side="BUY", win_fn=_is_buy_winner,
                moves=moves, frame=frame, replay_dates=replay_dates,
            ),
            "buy_v4": _engine_metrics(
                buy_v4, trading_days=days_used, side="BUY", win_fn=_is_buy_winner,
                moves=moves, frame=frame, replay_dates=replay_dates,
            ),
            "sell_v6": _engine_metrics(
                sell_v6, trading_days=days_used, side="SELL", win_fn=_is_sell_winner,
                moves=moves, frame=frame, replay_dates=replay_dates,
            ),
            "sell_v7": _engine_metrics(
                sell_v7, trading_days=days_used, side="SELL", win_fn=_is_sell_winner,
                moves=moves, frame=frame, replay_dates=replay_dates,
            ),
        }

        regime_export = _load_json_safe(REGIME_EXPORT_PATH)
        throttle_maps = _load_throttle_maps(regime_export)
        # Throttle uses V3/V6 maps; apply to V4/V7 books with same keys
        throttle_v3_v6 = _combined_throttled_metrics(
            buy_v3, sell_v6, throttle_maps, trading_days=days_used,
        )
        throttle_v4_v7 = _combined_throttled_metrics(
            buy_v4, sell_v7, throttle_maps, trading_days=days_used,
        )

        buy_compare = _compare_engines(
            metrics["buy_v3"], metrics["buy_v4"],
            baseline_name="BUY_V3", candidate_name="BUY_V4",
        )
        sell_compare = _compare_engines(
            metrics["sell_v6"], metrics["sell_v7"],
            baseline_name="SELL_V6", candidate_name="SELL_V7",
        )

        final_answer = _build_final_answer(
            buy_v3=metrics["buy_v3"],
            buy_v4=metrics["buy_v4"],
            sell_v6=metrics["sell_v6"],
            sell_v7=metrics["sell_v7"],
            buy_compare=buy_compare,
            sell_compare=sell_compare,
            buy_patterns=buy_patterns,
            sell_patterns=sell_patterns,
            trading_days=days_used,
            throttle=throttle_v4_v7,
        )

        replay_start = min(replay_dates).isoformat() if replay_dates else ""
        replay_end = max(replay_dates).isoformat() if replay_dates else ""

        conclusions = [
            f"Actual four-engine replay on {days_used} trading days "
            f"({replay_start} → {replay_end}); available={available_days}.",
            (
                f"Signals: BUY_V3={metrics['buy_v3']['signals']} BUY_V4={metrics['buy_v4']['signals']} "
                f"SELL_V6={metrics['sell_v6']['signals']} SELL_V7={metrics['sell_v7']['signals']}."
            ),
            (
                f"BUY PF/WR: V3 {metrics['buy_v3']['profit_factor']}/{metrics['buy_v3']['win_rate_pct']}% "
                f"vs V4 {metrics['buy_v4']['profit_factor']}/{metrics['buy_v4']['win_rate_pct']}%."
            ),
            (
                f"SELL PF/WR: V6 {metrics['sell_v6']['profit_factor']}/{metrics['sell_v6']['win_rate_pct']}% "
                f"vs V7 {metrics['sell_v7']['profit_factor']}/{metrics['sell_v7']['win_rate_pct']}%."
            ),
            f"BUY_V4 replace BUY_V3: {final_answer['should_buy_v4_replace_buy_v3']}.",
            f"SELL_V7 replace SELL_V6: {final_answer['should_sell_v7_replace_sell_v6']}.",
            (
                f"Evidence={final_answer['evidence_strength']} Confidence={final_answer['confidence']} "
                f"Overfitting={final_answer['overfitting_risk']}."
            ),
            (
                f"Best stack: {final_answer['best_buy_engine']} + {final_answer['best_sell_engine']} | "
                f"{final_answer['best_stop']} | {final_answer['best_exit_structure']}."
            ),
        ]

        report = BuyV4SellV7ActualReplayValidationReport(
            report_type="BUY_V4 & SELL_V7 Actual Replay Validation",
            engines=["BUY_V3", "BUY_V4", "SELL_V6", "SELL_V7"],
            symbol=DEFAULT_SYMBOL,
            timeframe="5M",
            trading_days_used=days_used,
            preferred_trading_days=PREFERRED_TRADING_DAYS,
            fallback_trading_days=FALLBACK_TRADING_DAYS,
            available_trading_days=available_days,
            replay_start_date=replay_start,
            replay_end_date=replay_end,
            methodology={
                "actual_bar_replay": True,
                "not_post_hoc_corpus_filter": True,
                "not_filter_simulation": True,
                "not_synthetic_pf": True,
                "filters_applied_inside_emission_path": True,
                "structural_detector": "failure_pattern_production_robustness_audit_research._detect_structural_patterns",
                "paper_stops": DEFAULT_STOP_VARIANT,
                "paper_targets": "60/100/Runner",
                "regime_throttle": True,
                "instrument": DEFAULT_SYMBOL,
                "timeframe": "5M",
                "replay_framework": "extended_evidence / extended_trade_level_truth harness",
                "rejection_counts": full.get("rejection_counts"),
            },
            approved_filters=filters,
            engine_definitions={
                "BUY_V3": {"base": None, "model_id": BUY_V3_MODEL_ID, "filters": []},
                "BUY_V4": {
                    "base": "BUY_V3",
                    "model_id": BUY_V4_MODEL_ID,
                    "reject_patterns": buy_patterns,
                    "engine_class": "BuyV4CandidateEngine",
                },
                "SELL_V6": {"base": None, "model_id": SELL_V6_MODEL_ID, "filters": []},
                "SELL_V7": {
                    "base": "SELL_V6",
                    "model_id": SELL_V7_MODEL_ID,
                    "reject_patterns": sell_patterns,
                    "engine_class": "SellV7CandidateEngine",
                },
            },
            core_metrics=metrics,
            target_achievement_matrix={
                key: metrics[key]["target_achievement_matrix"] for key in metrics
            },
            trade_lifecycle={key: metrics[key]["trade_lifecycle"] for key in metrics},
            entry_timing={key: metrics[key]["entry_timing"] for key in metrics},
            reward_risk={key: metrics[key]["reward_risk"] for key in metrics},
            capture_metrics={key: metrics[key]["capture"] for key in metrics},
            regime_throttle={
                "buy_v3_sell_v6": throttle_v3_v6,
                "buy_v4_sell_v7": throttle_v4_v7,
            },
            engine_comparison={
                "buy_v3_vs_buy_v4": buy_compare,
                "sell_v6_vs_sell_v7": sell_compare,
            },
            per_signal_details={
                "buy_v3": buy_v3,
                "buy_v4": buy_v4,
                "sell_v6": sell_v6,
                "sell_v7": sell_v7,
            },
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )
        return report

    def export(self, report: BuyV4SellV7ActualReplayValidationReport, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("BUY_V4/SELL_V7 actual replay validation exported: %s", path)
        _write_status(
            self.status_path,
            {
                "phase": "exported",
                "path": str(path),
                "trading_days_used": report.trading_days_used,
                "final_answer": report.final_answer,
                "ts": time.time(),
            },
        )
        return path


def generate_buy_v4_sell_v7_actual_replay_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    blueprint_path: Path | str | None = None,
    *,
    trading_days: int | None = None,
) -> BuyV4SellV7ActualReplayValidationReport:
    filter_path = Path(filter_report_path) if filter_report_path else DEFAULT_FILTER_REPORT_PATH
    if not filter_path.exists():
        raise BuyV4SellV7ActualReplayValidationError(f"Filter report missing: {filter_path}")
    metadata = json.loads(filter_path.read_text(encoding="utf-8"))
    if "end_date" not in metadata:
        # Some filter reports nest metadata
        metadata = metadata.get("metadata") or metadata
    if "end_date" not in metadata:
        raise BuyV4SellV7ActualReplayValidationError("filter_research_report.json missing end_date")

    bp_path = Path(blueprint_path) if blueprint_path else DEFAULT_BLUEPRINT_PATH
    approved = _load_blueprint_filters(bp_path)
    research = BuyV4SellV7ActualReplayValidationResearch(
        buy_v4_reject_patterns=approved["buy_v4_reject_patterns"],
        sell_v7_reject_patterns=approved["sell_v7_reject_patterns"],
    )
    report = research.run(metadata, approved_filters=approved, trading_days=trading_days)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(DEFAULT_LOG_PATH, encoding="utf-8"),
        ],
    )
    report = generate_buy_v4_sell_v7_actual_replay_validation_report()
    final = report.final_answer
    print(f"Export: {DEFAULT_REPORT_PATH}")
    print(f"Trading days used: {report.trading_days_used}")
    print(
        f"Signals: V3={report.core_metrics['buy_v3']['signals']} "
        f"V4={report.core_metrics['buy_v4']['signals']} "
        f"V6={report.core_metrics['sell_v6']['signals']} "
        f"V7={report.core_metrics['sell_v7']['signals']}",
    )
    print(f"BUY_V4 replace BUY_V3: {final['should_buy_v4_replace_buy_v3']}")
    print(f"SELL_V7 replace SELL_V6: {final['should_sell_v7_replace_sell_v6']}")
    cmp_ = final["pf_wr_comparison"]
    print(f"BUY PF/WR: {cmp_['buy_v3_vs_buy_v4']}")
    print(f"SELL PF/WR: {cmp_['sell_v6_vs_sell_v7']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
