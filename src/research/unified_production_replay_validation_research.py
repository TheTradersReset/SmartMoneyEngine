"""
Unified Production Replay Validation — SELL_V5 + BUY_V3 combined engine.

Single-pass 120-day NIFTY50 5M replay with walk-forward (train 80 / validate 40 days).
Measures combined production metrics, overlap/conflict classification, capital curve,
monthly breakdown, and reconciles against buy_v3_tradeability_production_validation.json.
Research-only; no production modifications.
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
from statistics import mean, median, pstdev
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.buy_v2_candidate_validation_research import (
    PRODUCTION_GATES,
    TRADING_DAYS_REPLAY,
    TRAIN_TRADING_DAYS,
    VALIDATE_TRADING_DAYS,
    _bullish_point_capture,
    _classify_failed_buy_signal,
    _filter_signals_by_dates,
    _nearest_bullish_move,
    _split_trading_day_sets,
    _walk_forward_metrics,
)
from src.research.buy_v3_candidate_validation_research import (
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
    BuyV3CandidateEngine,
    _evaluate_buy_bar_fast,
    _precompute_bar_events,
)
from src.research.filter_research_engine import FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import PRE_EXPANSION_LOOKBACK
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    _CheapMoveCandidate,
)
from src.research.nifty50_trap_to_momentum_validation_research import DEFAULT_SYMBOL
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
from src.research.smartmoneyengine_v5_candidate_validation_research import (
    V5CandidateEngine,
    V5_VWAP_GATE_RULE,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "unified_production_replay_validation.json"
DEFAULT_SYNTHESIS_PATH = RESEARCH_DIR / "buy_v3_tradeability_production_validation.json"

SELL_V5_MODEL_ID = "LDM-SELL-V5"
POINT_CAPTURE_THRESHOLDS = (40, 60, 100, 200)
MFE_CAPTURE_TIERS = (40, 60, 100, 200)
MOVE_DETECTION_THRESHOLD = 40
SYNTHESIS_TOLERANCE_PCT = 15.0


class UnifiedProductionReplayValidationError(Exception):
    """Raised when unified production replay validation fails."""


@dataclass
class UnifiedProductionReplayValidationReport:
    """Unified SELL_V5 + BUY_V3 production replay validation output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    methodology: dict[str, Any]
    walk_forward: dict[str, Any]
    engine_comparison: dict[str, Any]
    combined_metrics: dict[str, Any]
    signal_classification: dict[str, Any]
    capital_curve: dict[str, Any]
    monthly_breakdown: dict[str, Any]
    human_tradeability: dict[str, Any]
    synthesis_reconciliation: dict[str, Any]
    production_readiness: dict[str, Any]
    final_answer: dict[str, Any]
    per_signal_details: dict[str, list[dict[str, Any]]]
    conclusions: list[str]
    execution_time_seconds: float


def _profit_factor(pnls: list[float]) -> float | None:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return None if gross_profit == 0 else round(gross_profit, 2)
    return round(gross_profit / gross_loss, 2)


def _signal_date(timestamp: str) -> str:
    return str(timestamp)[:10]


def _signal_month(timestamp: str) -> str:
    return str(timestamp)[:7]


def _tier_capture_from_signals(signals: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(signals)
    tiers: dict[str, Any] = {}
    for threshold in MFE_CAPTURE_TIERS:
        hits = sum(1 for signal in signals if float(signal.get("mfe_points") or 0.0) >= threshold)
        tiers[str(threshold)] = {
            "signals_hitting_tier": hits,
            "hit_rate_pct": round(100.0 * hits / max(total, 1), 2),
        }
    return tiers


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return round(max_dd, 2)


def _recovery_factor(pnls: list[float]) -> float | None:
    net = sum(pnls)
    dd = _max_drawdown(pnls)
    if dd <= 0:
        return None if net <= 0 else round(net, 2)
    return round(net / dd, 2)


def _sharpe_proxy(pnls: list[float]) -> float | None:
    if len(pnls) < 2:
        return None
    avg = mean(pnls)
    std = pstdev(pnls)
    if std == 0:
        return None
    return round(avg / std, 2)


def _capital_curve_metrics(signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(signals, key=lambda s: (s.get("timestamp", ""), s.get("bar", 0)))
    pnls = [float(s.get("realized_pnl_points") or 0.0) for s in ordered]
    equity_points: list[float] = []
    running = 0.0
    for pnl in pnls:
        running += pnl
        equity_points.append(round(running, 2))

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "trade_count": len(pnls),
        "net_points": round(sum(pnls), 2),
        "max_drawdown_points": _max_drawdown(pnls),
        "recovery_factor": _recovery_factor(pnls),
        "sharpe_proxy": _sharpe_proxy(pnls),
        "profit_distribution": {
            "win_count": len(wins),
            "loss_count": len(losses),
            "avg_win_points": round(mean(wins), 2) if wins else 0.0,
            "avg_loss_points": round(mean(losses), 2) if losses else 0.0,
            "median_pnl_points": round(median(pnls), 2) if pnls else 0.0,
            "largest_win_points": round(max(pnls), 2) if pnls else 0.0,
            "largest_loss_points": round(min(pnls), 2) if pnls else 0.0,
        },
        "equity_stability": {
            "positive_equity_pct": round(
                100.0 * sum(1 for value in equity_points if value > 0) / max(len(equity_points), 1),
                2,
            ),
            "final_equity_points": equity_points[-1] if equity_points else 0.0,
            "equity_curve_sample": equity_points[:50],
        },
    }


def _human_tradeability(signals: list[dict[str, Any]]) -> dict[str, Any]:
    by_day: Counter[str] = Counter()
    for signal in signals:
        by_day[_signal_date(str(signal.get("timestamp", "")))] += 1
    counts = list(by_day.values())
    if not counts:
        return {
            "trading_days_with_signals": 0,
            "avg_signals_per_day": 0.0,
            "median_signals_per_day": 0.0,
            "max_signals_per_day": 0,
            "days_over_3_signals": 0,
            "days_over_5_signals": 0,
        }
    return {
        "trading_days_with_signals": len(counts),
        "avg_signals_per_day": round(mean(counts), 2),
        "median_signals_per_day": round(median(counts), 2),
        "max_signals_per_day": max(counts),
        "days_over_3_signals": sum(1 for count in counts if count > 3),
        "days_over_5_signals": sum(1 for count in counts if count > 5),
    }


def _classify_signals(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    buy_bars = {s["bar"]: s for s in buy_signals}
    sell_bars = {s["bar"]: s for s in sell_signals}
    overlap_bars = set(buy_bars) & set(sell_bars)

    buy_dates = {_signal_date(str(s["timestamp"])) for s in buy_signals}
    sell_dates = {_signal_date(str(s["timestamp"])) for s in sell_signals}
    same_day_both = buy_dates & sell_dates

    conflict_details: list[dict[str, Any]] = []
    for bar in sorted(overlap_bars):
        buy = buy_bars[bar]
        sell = sell_bars[bar]
        buy_pnl = float(buy.get("realized_pnl_points") or 0.0)
        sell_pnl = float(sell.get("realized_pnl_points") or 0.0)
        conflict_details.append(
            {
                "bar": bar,
                "timestamp": buy.get("timestamp"),
                "buy_pnl": buy_pnl,
                "sell_pnl": sell_pnl,
                "net_pnl": round(buy_pnl + sell_pnl, 2),
                "both_win": bool(buy.get("win")) and bool(sell.get("win")),
                "both_lose": (not buy.get("win")) and (not sell.get("win")),
                "opposite_outcome": bool(buy.get("win")) != bool(sell.get("win")),
            },
        )

    session_analysis: list[dict[str, Any]] = []
    buy_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sell_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signal in buy_signals:
        buy_by_date[_signal_date(str(signal["timestamp"]))].append(signal)
    for signal in sell_signals:
        sell_by_date[_signal_date(str(signal["timestamp"]))].append(signal)

    for day in sorted(same_day_both):
        day_buys = buy_by_date[day]
        day_sells = sell_by_date[day]
        buy_pnl = sum(float(s.get("realized_pnl_points") or 0.0) for s in day_buys)
        sell_pnl = sum(float(s.get("realized_pnl_points") or 0.0) for s in day_sells)
        net_pnl = buy_pnl + sell_pnl
        buy_wins = sum(1 for s in day_buys if s.get("win"))
        sell_wins = sum(1 for s in day_sells if s.get("win"))
        session_analysis.append(
            {
                "date": day,
                "buy_signals": len(day_buys),
                "sell_signals": len(day_sells),
                "buy_pnl": round(buy_pnl, 2),
                "sell_pnl": round(sell_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "same_direction_outcome": (buy_wins > 0 and sell_wins > 0) or (
                    buy_wins == 0 and sell_wins == 0 and len(day_buys) > 0 and len(day_sells) > 0
                ),
                "opposite_direction_outcome": (buy_wins > 0 and sell_wins == 0) or (
                    buy_wins == 0 and sell_wins > 0
                ),
            },
        )

    session_pnls = [row["net_pnl"] for row in session_analysis]
    conflict_pnls = [row["net_pnl"] for row in conflict_details]

    return {
        "buy_only_count": len(buy_signals) - len(overlap_bars),
        "sell_only_count": len(sell_signals) - len(overlap_bars),
        "same_bar_overlap_count": len(overlap_bars),
        "same_bar_conflict_rate_pct": round(
            100.0 * len(overlap_bars) / max(len(buy_signals) + len(sell_signals), 1),
            2,
        ),
        "same_day_both_engines_count": len(same_day_both),
        "same_day_overlap_rate_pct": round(
            100.0 * len(same_day_both) / max(len(buy_dates | sell_dates), 1),
            2,
        ),
        "conflict_details_sample": conflict_details[:25],
        "session_dual_engine_analysis": {
            "sessions_with_both": len(session_analysis),
            "avg_net_pnl": round(mean(session_pnls), 2) if session_pnls else 0.0,
            "median_net_pnl": round(median(session_pnls), 2) if session_pnls else 0.0,
            "worst_session_net_pnl": round(min(session_pnls), 2) if session_pnls else 0.0,
            "best_session_net_pnl": round(max(session_pnls), 2) if session_pnls else 0.0,
            "same_direction_sessions": sum(
                1 for row in session_analysis if row["same_direction_outcome"]
            ),
            "opposite_direction_sessions": sum(
                1 for row in session_analysis if row["opposite_direction_outcome"]
            ),
            "session_details_sample": session_analysis[:25],
        },
        "conflict_bar_summary": {
            "count": len(conflict_details),
            "avg_net_pnl": round(mean(conflict_pnls), 2) if conflict_pnls else 0.0,
            "worst_case_net_pnl": round(min(conflict_pnls), 2) if conflict_pnls else 0.0,
            "both_win_count": sum(1 for row in conflict_details if row["both_win"]),
            "both_lose_count": sum(1 for row in conflict_details if row["both_lose"]),
            "opposite_outcome_count": sum(1 for row in conflict_details if row["opposite_outcome"]),
        },
    }


def _monthly_breakdown(
    signals: list[dict[str, Any]],
    *,
    moves: list[_CheapMoveCandidate],
    frame: pd.DataFrame,
    replay_dates: set[date],
) -> dict[str, Any]:
    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        by_month[_signal_month(str(signal.get("timestamp", "")))].append(signal)

    months: dict[str, Any] = {}
    for month, month_signals in sorted(by_month.items()):
        pnls = [float(s.get("realized_pnl_points") or 0.0) for s in month_signals]
        wins = [p for p in pnls if p > 0]
        buy_count = sum(1 for s in month_signals if s.get("direction") == "BUY")
        sell_count = sum(1 for s in month_signals if s.get("direction") == "SELL")
        months[month] = {
            "signals": len(month_signals),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "win_rate_pct": round(100.0 * len(wins) / max(len(pnls), 1), 2),
            "profit_factor": _profit_factor(pnls),
            "expectancy": round(mean(pnls), 2) if pnls else 0.0,
            "net_points": round(sum(pnls), 2),
            "max_drawdown_points": _max_drawdown(pnls),
            "mfe_capture_tiers": _tier_capture_from_signals(month_signals),
        }

    if not months:
        return {"by_month": {}, "best_month": None, "worst_month": None}

    best = max(months.items(), key=lambda item: item[1]["net_points"])
    worst = min(months.items(), key=lambda item: item[1]["net_points"])

    def _regime_note(month_key: str, stats: dict[str, Any]) -> str:
        buy_ratio = stats["buy_signals"] / max(stats["signals"], 1)
        if buy_ratio > 0.65:
            regime = "bullish-reversal dominated"
        elif buy_ratio < 0.35:
            regime = "bearish-expansion dominated"
        else:
            regime = "mixed regime"
        return (
            f"{month_key}: {stats['signals']} signals ({stats['buy_signals']} BUY / "
            f"{stats['sell_signals']} SELL), net {stats['net_points']} pts, {regime}."
        )

    return {
        "by_month": months,
        "best_month": {
            "month": best[0],
            "metrics": best[1],
            "regime_explanation": _regime_note(best[0], best[1]),
        },
        "worst_month": {
            "month": worst[0],
            "metrics": worst[1],
            "regime_explanation": _regime_note(worst[0], worst[1]),
        },
    }


def _engine_comparison_three_way(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    combined_signals: list[dict[str, Any]],
    *,
    moves: list[_CheapMoveCandidate],
    frame: pd.DataFrame,
    replay_dates: set[date],
    trading_days: int,
) -> dict[str, Any]:
    buy_stats = _build_statistics(buy_signals, trading_days=trading_days)
    sell_stats = _build_statistics(sell_signals, trading_days=trading_days)
    combined_stats = _build_statistics(combined_signals, trading_days=trading_days)

    buy_capture = _bullish_point_capture(moves, buy_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
    sell_capture = _point_capture(moves, sell_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)

    def _incremental(base: dict[str, Any], combined: dict[str, Any], key: str) -> float | None:
        base_val = base.get(key)
        comb_val = combined.get(key)
        if base_val is None or comb_val is None:
            return None
        return round(float(comb_val) - float(base_val), 2)

    return {
        "sell_v5_only": {
            "overall_statistics": sell_stats,
            "point_capture_bearish": sell_capture,
            "mfe_capture_tiers": _tier_capture_from_signals(sell_signals),
            "capital_curve": _capital_curve_metrics(sell_signals),
        },
        "buy_v3_only": {
            "overall_statistics": buy_stats,
            "point_capture_bullish": buy_capture,
            "mfe_capture_tiers": _tier_capture_from_signals(buy_signals),
            "capital_curve": _capital_curve_metrics(buy_signals),
        },
        "combined": {
            "overall_statistics": combined_stats,
            "mfe_capture_tiers": _tier_capture_from_signals(combined_signals),
            "capital_curve": _capital_curve_metrics(combined_signals),
            "bullish_move_capture": buy_capture,
            "bearish_move_capture": sell_capture,
        },
        "incremental_benefit": {
            "signals_per_month_delta_vs_sell_only": _incremental(
                sell_stats,
                combined_stats,
                "signals_per_month",
            ),
            "signals_per_month_delta_vs_buy_only": _incremental(
                buy_stats,
                combined_stats,
                "signals_per_month",
            ),
            "expectancy_delta_vs_sell_only": _incremental(sell_stats, combined_stats, "expectancy"),
            "expectancy_delta_vs_buy_only": _incremental(buy_stats, combined_stats, "expectancy"),
            "max_drawdown_combined_vs_sell_only": round(
                _capital_curve_metrics(combined_signals)["max_drawdown_points"]
                - _capital_curve_metrics(sell_signals)["max_drawdown_points"],
                2,
            ),
            "max_drawdown_combined_vs_buy_only": round(
                _capital_curve_metrics(combined_signals)["max_drawdown_points"]
                - _capital_curve_metrics(buy_signals)["max_drawdown_points"],
                2,
            ),
            "headline": (
                f"Combined adds "
                f"{round((combined_stats.get('signals_per_month') or 0) - (sell_stats.get('signals_per_month') or 0), 2)} "
                f"signals/mo vs SELL-only and "
                f"{round((combined_stats.get('signals_per_month') or 0) - (buy_stats.get('signals_per_month') or 0), 2)} "
                f"vs BUY-only; combined expectancy {combined_stats.get('expectancy')}."
            ),
        },
    }


def _reconcile_synthesis(
    replay_combined: dict[str, Any],
    synthesis_path: Path,
) -> dict[str, Any]:
    if not synthesis_path.exists():
        return {
            "synthesis_available": False,
            "synthesis_match": "no",
            "note": f"Missing synthesis export: {synthesis_path}",
        }

    synthesis = json.loads(synthesis_path.read_text(encoding="utf-8"))
    synth_metrics = synthesis.get("combined_engine_simulation", {}).get("combined_metrics", {})
    replay_stats = replay_combined.get("overall_statistics", {})

    comparisons: dict[str, Any] = {}
    for key in ("signals_per_month", "win_rate_pct", "profit_factor", "expectancy"):
        synth_val = synth_metrics.get(key)
        replay_val = replay_stats.get(key)
        if synth_val is None or replay_val is None:
            comparisons[key] = {"synthesis": synth_val, "replay": replay_val, "match": None}
            continue
        synth_f = float(synth_val)
        replay_f = float(replay_val)
        if synth_f == 0:
            match = abs(replay_f) < 1.0
        else:
            match = abs(replay_f - synth_f) / abs(synth_f) * 100 <= SYNTHESIS_TOLERANCE_PCT
        comparisons[key] = {
            "synthesis": synth_val,
            "replay": replay_val,
            "delta": round(replay_f - synth_f, 2),
            "delta_pct": round(abs(replay_f - synth_f) / max(abs(synth_f), 0.01) * 100, 2),
            "match": match,
        }

    match_count = sum(1 for row in comparisons.values() if row.get("match") is True)
    total_checks = sum(1 for row in comparisons.values() if row.get("match") is not None)
    synthesis_match = "yes" if total_checks > 0 and match_count >= total_checks * 0.75 else "no"

    return {
        "synthesis_available": True,
        "synthesis_source": str(synthesis_path.name),
        "synthesis_basis": synthesis.get("combined_engine_simulation", {}).get("simulation_basis"),
        "synthesis_limitations": synthesis.get("combined_engine_simulation", {}).get("limitations"),
        "metric_comparisons": comparisons,
        "synthesis_match": synthesis_match,
        "match_count": match_count,
        "total_checks": total_checks,
        "note": (
            "Replay uses full per-signal SELL_V5 + BUY_V3 merge; synthesis used aggregate SELL proxy."
            if synthesis_match == "no"
            else "Replay metrics align with synthesis within tolerance."
        ),
    }


def _production_readiness_score(
    *,
    buy_stats: dict[str, Any],
    sell_stats: dict[str, Any],
    combined_stats: dict[str, Any],
    classification: dict[str, Any],
    walk_forward_stable: bool,
    capital_curve: dict[str, Any],
) -> dict[str, Any]:
    score = 0
    components: dict[str, Any] = {}

    buy_wr = float(buy_stats.get("win_rate_pct") or 0.0)
    sell_wr = float(sell_stats.get("win_rate_pct") or 0.0)
    combined_wr = float(combined_stats.get("win_rate_pct") or 0.0)
    wr_score = min(25, int(combined_wr * 0.35))
    if buy_wr >= 65 and sell_wr >= 65:
        wr_score = min(25, wr_score + 5)
    components["win_rate"] = {"score": wr_score, "combined_wr_pct": combined_wr}
    score += wr_score

    buy_pf = float(buy_stats.get("profit_factor") or 0.0)
    sell_pf = float(sell_stats.get("profit_factor") or 0.0)
    combined_pf = float(combined_stats.get("profit_factor") or 0.0)
    pf_score = 0
    if combined_pf >= 3:
        pf_score = 25
    elif combined_pf >= 2:
        pf_score = 20
    elif combined_pf >= 1.5:
        pf_score = 12
    if buy_pf >= 2 and sell_pf >= 2:
        pf_score = min(25, pf_score + 3)
    components["profit_factor"] = {"score": pf_score, "combined_pf": combined_pf}
    score += pf_score

    spm = float(combined_stats.get("signals_per_month") or 0.0)
    freq_score = min(15, int(spm / 6))
    components["frequency"] = {"score": freq_score, "signals_per_month": spm}
    score += freq_score

    wf_score = 15 if walk_forward_stable else 5
    components["walk_forward"] = {"score": wf_score, "stable": walk_forward_stable}
    score += wf_score

    conflict_rate = float(classification.get("same_bar_conflict_rate_pct") or 0.0)
    conflict_score = 10 if conflict_rate < 1.0 else (6 if conflict_rate < 3.0 else 2)
    components["conflict_rate"] = {"score": conflict_score, "same_bar_conflict_rate_pct": conflict_rate}
    score += conflict_score

    recovery = capital_curve.get("recovery_factor")
    dd = float(capital_curve.get("max_drawdown_points") or 0.0)
    dd_score = 5
    if recovery is not None and float(recovery) >= 2.0:
        dd_score = 10
    elif recovery is not None and float(recovery) >= 1.0:
        dd_score = 7
    components["drawdown_recovery"] = {
        "score": dd_score,
        "max_drawdown_points": dd,
        "recovery_factor": recovery,
    }
    score += dd_score

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
    }


def _final_answer(
    *,
    buy_stats: dict[str, Any],
    sell_stats: dict[str, Any],
    combined_stats: dict[str, Any],
    classification: dict[str, Any],
    walk_forward_stable: bool,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    buy_wr = float(buy_stats.get("win_rate_pct") or 0.0)
    sell_wr = float(sell_stats.get("win_rate_pct") or 0.0)
    buy_pf = float(buy_stats.get("profit_factor") or 0.0)
    sell_pf = float(sell_stats.get("profit_factor") or 0.0)
    combined_spm = float(combined_stats.get("signals_per_month") or 0.0)
    conflict_rate = float(classification.get("same_bar_conflict_rate_pct") or 0.0)

    legs_pass = (
        buy_wr >= PRODUCTION_GATES["win_rate_min_pct"]
        and sell_wr >= PRODUCTION_GATES["win_rate_min_pct"]
        and buy_pf >= PRODUCTION_GATES["profit_factor_min"]
        and sell_pf >= PRODUCTION_GATES["profit_factor_min"]
    )

    if (
        legs_pass
        and walk_forward_stable
        and conflict_rate < 3.0
        and combined_spm >= 50
        and readiness["score"] >= 70
    ):
        verdict = "YES"
    elif legs_pass and combined_spm >= 30 and readiness["score"] >= 50:
        verdict = "PARTIAL"
    else:
        verdict = "NO"

    return {
        "can_operate_as_single_production_engine": verdict,
        "production_readiness_score": readiness["score"],
        "recommendation_tier": readiness["recommendation_tier"],
        "evidence": {
            "buy_v3_wr_pct": buy_wr,
            "buy_v3_pf": buy_pf,
            "sell_v5_wr_pct": sell_wr,
            "sell_v5_pf": sell_pf,
            "combined_signals_per_month": combined_spm,
            "same_bar_conflict_rate_pct": conflict_rate,
            "walk_forward_stable": walk_forward_stable,
            "both_legs_pass_production_gates": legs_pass,
        },
    }


class UnifiedProductionReplayValidationResearch:
    """Combined SELL_V5 + BUY_V3 single-pass production replay validation."""

    def __init__(self) -> None:
        self.buy_engine = BuyV3CandidateEngine()
        self.sell_engine = V5CandidateEngine()
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=TRADING_DAYS_REPLAY,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    def _build_buy_signal(
        self,
        evaluation: dict[str, Any],
        *,
        moves: list[_CheapMoveCandidate],
        frame: pd.DataFrame,
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
            {
                "mfe_points": forward.get("mfe_points"),
                "mae_points": forward.get("mae_points"),
                "win": forward.get("win"),
            },
            context=context,
        )

        return {
            "timestamp": evaluation["timestamp"],
            "bar": bar,
            "symbol": DEFAULT_SYMBOL,
            "timeframe": MOVE_DETECTION_TIMEFRAME,
            "direction": "BUY",
            "engine_version": "BUY_V3",
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
            "move_start_time": str(frame.iloc[move_start_bar]["Date"]) if move_start_bar is not None else None,
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

    def _build_sell_signal(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        layer4 = evaluation["layer4"]
        forward = dict(layer4.get("forward_outcome") or {})
        return {
            "timestamp": evaluation["timestamp"],
            "bar": evaluation["bar"],
            "symbol": DEFAULT_SYMBOL,
            "timeframe": MOVE_DETECTION_TIMEFRAME,
            "direction": "SELL",
            "engine_version": "SELL_V5",
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
            "layers": {
                "layer1": evaluation["layer1"],
                "layer2": evaluation["layer2"],
                "layer3": evaluation["layer3"],
                "layer5": evaluation["layer5"],
            },
        }

    def _replay_unified(
        self,
        *,
        frame: pd.DataFrame,
        enriched_buy: pd.DataFrame,
        enriched_sell: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        replay_bars: list[int],
        moves: list[_CheapMoveCandidate],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        valid_bars = [
            bar
            for bar in replay_bars
            if bar >= PRE_EXPANSION_LOOKBACK and bar < len(frame) - FORWARD_BARS
        ]

        logger.info("Precomputing BUY_V3 event detection for %s bars...", len(valid_bars))
        bar_events_cache, lookback_cache = _precompute_bar_events(
            self.buy_engine,
            frame=frame,
            calendar=calendar,
            replay_bars=valid_bars,
        )

        logger.info("Precomputing BUY_V3 context for %s bars...", len(valid_bars))
        buy_context_cache: dict[int, dict[str, str]] = {}
        context_log_every = max(len(valid_bars) // 10, 1)
        context_started = time.perf_counter()
        for index, bar in enumerate(valid_bars):
            if index > 0 and index % context_log_every == 0:
                logger.info(
                    "BUY context precompute: %s/%s (%.0f%%) elapsed %.0fs",
                    index,
                    len(valid_bars),
                    index / len(valid_bars) * 100,
                    time.perf_counter() - context_started,
                )
            buy_context_cache[bar] = self.buy_engine._context_at_bar(
                frame=frame,
                enriched=enriched_buy,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
            )
        logger.info(
            "BUY context precompute complete in %.0fs",
            time.perf_counter() - context_started,
        )

        buy_signals: list[dict[str, Any]] = []
        sell_signals: list[dict[str, Any]] = []
        buy_emitted: set[int] = set()
        sell_emitted: set[int] = set()

        total = len(valid_bars)
        log_every = max(total // 20, 1)
        started = time.perf_counter()

        for index, bar in enumerate(valid_bars):
            if index > 0 and index % log_every == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "Unified replay: %s/%s bars (%.0f%%) elapsed %.0fs | BUY=%s SELL=%s conflicts=%s",
                    index,
                    total,
                    index / total * 100,
                    elapsed,
                    len(buy_signals),
                    len(sell_signals),
                    len(set(s["bar"] for s in buy_signals) & set(s["bar"] for s in sell_signals)),
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
                buy_signals.append(self._build_buy_signal(buy_eval, moves=moves, frame=frame))
                buy_emitted.add(bar)

            sell_eval = self.sell_engine.evaluate_bar(
                frame=frame,
                enriched=enriched_sell,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=sell_emitted,
            )
            if sell_eval["verdict"] == "SELL":
                sell_signals.append(self._build_sell_signal(sell_eval))
                sell_emitted.add(bar)

        logger.info(
            "Unified replay complete: BUY=%s SELL=%s in %.0fs",
            len(buy_signals),
            len(sell_signals),
            time.perf_counter() - started,
        )
        return buy_signals, sell_signals

    def run(self, metadata: dict[str, Any]) -> UnifiedProductionReplayValidationReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = end - timedelta(days=TRADING_DAYS_REPLAY)

        logger.info(
            "Unified production replay starting: %s days, %s 5M",
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

        logger.info("Loading enriched context and intel frames...")
        enriched_buy = self.buy_engine.context_builder.enrich(frame)
        enriched_sell = _attach_ema22(self.sell_engine.context_builder.enrich(frame))
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        intel_frames: dict[str, pd.DataFrame] = {
            "5M": self.buy_engine.intelligence.enrich(frame),
        }
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
        logger.info("Detected %s deduped moves", len(moves))

        buy_signals, sell_signals = self._replay_unified(
            frame=frame,
            enriched_buy=enriched_buy,
            enriched_sell=enriched_sell,
            calendar=calendar,
            intel_frames=intel_frames,
            replay_bars=replay_bars,
            moves=moves,
        )

        combined_signals = sorted(
            buy_signals + sell_signals,
            key=lambda s: (s.get("timestamp", ""), s.get("bar", 0)),
        )

        engine_comparison = _engine_comparison_three_way(
            buy_signals,
            sell_signals,
            combined_signals,
            moves=moves,
            frame=frame,
            replay_dates=replay_dates,
            trading_days=TRADING_DAYS_REPLAY,
        )

        classification = _classify_signals(buy_signals, sell_signals)
        capital_curve = _capital_curve_metrics(combined_signals)
        monthly = _monthly_breakdown(
            combined_signals,
            moves=moves,
            frame=frame,
            replay_dates=replay_dates,
        )
        tradeability = _human_tradeability(combined_signals)

        combined_train = _filter_signals_by_dates(combined_signals, frame, train_dates)
        combined_validate = _filter_signals_by_dates(combined_signals, frame, validate_dates)
        buy_train = _filter_signals_by_dates(buy_signals, frame, train_dates)
        buy_validate = _filter_signals_by_dates(buy_signals, frame, validate_dates)
        sell_train = _filter_signals_by_dates(sell_signals, frame, train_dates)
        sell_validate = _filter_signals_by_dates(sell_signals, frame, validate_dates)

        train_stats = _walk_forward_metrics(combined_train, period_days=len(train_dates))
        validate_stats = _walk_forward_metrics(combined_validate, period_days=len(validate_dates))
        walk_forward_stable = (
            float(validate_stats.get("win_rate_pct") or 0.0)
            >= float(train_stats.get("win_rate_pct") or 0.0) * 0.85
            and float(validate_stats.get("profit_factor") or 0.0)
            >= float(train_stats.get("profit_factor") or 0.0) * 0.70
        )

        walk_forward = {
            "train_trading_days": len(train_dates),
            "validate_trading_days": len(validate_dates),
            "train_start_date": min(train_dates).isoformat() if train_dates else "",
            "train_end_date": max(train_dates).isoformat() if train_dates else "",
            "validate_start_date": min(validate_dates).isoformat() if validate_dates else "",
            "validate_end_date": max(validate_dates).isoformat() if validate_dates else "",
            "train": {
                "combined": train_stats,
                "buy_v3": _walk_forward_metrics(buy_train, period_days=len(train_dates)),
                "sell_v5": _walk_forward_metrics(sell_train, period_days=len(train_dates)),
            },
            "validate": {
                "combined": validate_stats,
                "buy_v3": _walk_forward_metrics(buy_validate, period_days=len(validate_dates)),
                "sell_v5": _walk_forward_metrics(sell_validate, period_days=len(validate_dates)),
            },
            "stable": walk_forward_stable,
        }

        synthesis = _reconcile_synthesis(
            engine_comparison["combined"],
            DEFAULT_SYNTHESIS_PATH,
        )

        readiness = _production_readiness_score(
            buy_stats=engine_comparison["buy_v3_only"]["overall_statistics"],
            sell_stats=engine_comparison["sell_v5_only"]["overall_statistics"],
            combined_stats=engine_comparison["combined"]["overall_statistics"],
            classification=classification,
            walk_forward_stable=walk_forward_stable,
            capital_curve=capital_curve,
        )

        final = _final_answer(
            buy_stats=engine_comparison["buy_v3_only"]["overall_statistics"],
            sell_stats=engine_comparison["sell_v5_only"]["overall_statistics"],
            combined_stats=engine_comparison["combined"]["overall_statistics"],
            classification=classification,
            walk_forward_stable=walk_forward_stable,
            readiness=readiness,
        )

        replay_start = min(replay_dates).isoformat() if replay_dates else ""
        replay_end = max(replay_dates).isoformat() if replay_dates else ""

        combined_stats = engine_comparison["combined"]["overall_statistics"]
        conclusions = [
            (
                f"Unified replay: BUY_V3 {len(buy_signals)} + SELL_V5 {len(sell_signals)} = "
                f"{len(combined_signals)} signals over {TRADING_DAYS_REPLAY} days."
            ),
            (
                f"Combined: {combined_stats.get('signals_per_month')}/mo, WR "
                f"{combined_stats.get('win_rate_pct')}%, PF {combined_stats.get('profit_factor')}, "
                f"expectancy {combined_stats.get('expectancy')}."
            ),
            (
                f"Same-bar conflicts: {classification['same_bar_overlap_count']} "
                f"({classification['same_bar_conflict_rate_pct']}%)."
            ),
            f"Production readiness: {readiness['score']}/100 — {readiness['recommendation_tier']}.",
            f"Single engine verdict: {final['can_operate_as_single_production_engine']}.",
            f"Synthesis reconciliation: {synthesis.get('synthesis_match', 'no')}.",
        ]

        return UnifiedProductionReplayValidationReport(
            report_type="Unified Production Replay Validation",
            engines=["SELL_V5", "BUY_V3"],
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            trading_days_replayed=TRADING_DAYS_REPLAY,
            replay_start_date=replay_start,
            replay_end_date=replay_end,
            methodology={
                "research_only": True,
                "actual_replay": True,
                "single_pass_combined_replay": True,
                "sell_engine": "V5CandidateEngine (LDM-SELL-V5)",
                "buy_engine": "BuyV3CandidateEngine (LDM-BUY-V3)",
                "sell_vwap_gate": V5_VWAP_GATE_RULE,
                "buy_formula": BUY_V3_FORMULA_TEXT,
                "walk_forward": f"train {TRAIN_TRADING_DAYS} / validate {VALIDATE_TRADING_DAYS} trading days",
                "no_lookahead": True,
                "move_detection_threshold": MOVE_DETECTION_THRESHOLD,
                "point_capture_thresholds": list(POINT_CAPTURE_THRESHOLDS),
                "mfe_capture_tiers": list(MFE_CAPTURE_TIERS),
            },
            walk_forward=walk_forward,
            engine_comparison=engine_comparison,
            combined_metrics={
                "overall_statistics": combined_stats,
                "signals_per_week": combined_stats.get("signals_per_week"),
                "signals_per_month": combined_stats.get("signals_per_month"),
                "win_rate_pct": combined_stats.get("win_rate_pct"),
                "profit_factor": combined_stats.get("profit_factor"),
                "expectancy": combined_stats.get("expectancy"),
                "mfe_capture_tiers": engine_comparison["combined"]["mfe_capture_tiers"],
                "bullish_move_capture": engine_comparison["combined"]["bullish_move_capture"],
                "bearish_move_capture": engine_comparison["combined"]["bearish_move_capture"],
            },
            signal_classification=classification,
            capital_curve=capital_curve,
            monthly_breakdown=monthly,
            human_tradeability=tradeability,
            synthesis_reconciliation=synthesis,
            production_readiness=readiness,
            final_answer=final,
            per_signal_details={
                "buy_v3": buy_signals,
                "sell_v5": sell_signals,
                "combined": combined_signals,
            },
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: UnifiedProductionReplayValidationReport, report_path: Path) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("Unified production replay validation exported: %s", report_path)
        return report_path


def generate_unified_production_replay_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> UnifiedProductionReplayValidationReport:
    """Run unified SELL_V5 + BUY_V3 replay validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise UnifiedProductionReplayValidationError(f"Filter research report not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    research = UnifiedProductionReplayValidationResearch()
    report = research.run(metadata)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_unified_production_replay_validation_report()
    except UnifiedProductionReplayValidationError as exc:
        logger.error("Unified production replay validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected unified production replay validation error")
        return 1

    final = report.final_answer
    print("Unified Production Replay Validation Summary")
    print(f"Verdict: {final['can_operate_as_single_production_engine']}")
    print(f"Production readiness: {final['production_readiness_score']}/100")
    print(f"Recommendation: {final['recommendation_tier']}")
    print(f"Synthesis match: {report.synthesis_reconciliation.get('synthesis_match', 'no')}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
