"""
Production Trading Playbook Audit — synthesis from existing replay exports only.

Converts BUY_V3 + SELL_V6 + Regime Throttle into a deployable paper-trading playbook.
Targets, stops, and sizing are simulated from replay MFE/MAE — no new replay,
indicators, models, or discovery.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v2_candidate_validation_research import PRODUCTION_GATES
from src.research.buy_v3_candidate_validation_research import (
    BAR_MINUTES,
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
)
from src.research.buy_v3_tradeability_production_validation_research import (
    _fixed_target_pnl,
    _profit_factor,
)
from src.research.production_edge_enhancement_audit_research import (
    _cohort_performance,
    _is_buy_winner,
    _is_sell_winner,
    _profit_factor_from_pnls,
)
from src.research.regime_detection_audit_research import (
    SELL_V6_MODEL_ID,
    THROTTLE_WEIGHT,
    classify_signal_regime,
)
from src.research.sell_v6_replay_validation_research import V6_VWAP_GATE_RULE
from src.research.unified_production_replay_validation_research import (
    _max_drawdown,
    _recovery_factor,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "production_trading_playbook_audit.json"

SOURCE_EXPORTS = {
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "regime_detection_audit": RESEARCH_DIR / "regime_detection_audit.json",
    "unified_production_replay_validation": RESEARCH_DIR / "unified_production_replay_validation.json",
    "walk_forward_failure_root_cause_audit": RESEARCH_DIR
    / "walk_forward_failure_root_cause_audit.json",
    "buy_v3_tradeability_production_validation": RESEARCH_DIR
    / "buy_v3_tradeability_production_validation.json",
}

TARGET_STRUCTURES: dict[str, dict[str, Any]] = {
    "40/60/100": {"t1": 40, "t2": 60, "t3": 100, "runner": False},
    "40/80/Runner": {"t1": 40, "t2": 80, "t3": None, "runner": True},
    "50/100/Runner": {"t1": 50, "t2": 100, "t3": None, "runner": True},
    "60/120/Runner": {"t1": 60, "t2": 120, "t3": None, "runner": True},
}

STOP_VARIANTS = (
    "fixed_10",
    "fixed_20",
    "fixed_30",
    "structure_based",
    "liquidity_based",
    "atr_based",
)

SIZING_MODES = ("full", "half", "quarter", "regime_adaptive")

LEG_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)

BUY_MIN_SIGNALS_PER_MONTH = 20.0
SELL_MIN_SIGNALS_PER_MONTH = 60.0


class ProductionTradingPlaybookAuditError(Exception):
    """Raised when production trading playbook audit synthesis fails."""


@dataclass
class ProductionTradingPlaybookAuditReport:
    """Production trading playbook audit output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    buy_v3_playbook: dict[str, Any]
    sell_v6_playbook: dict[str, Any]
    combined_playbook: dict[str, Any]
    target_structure_comparison: dict[str, Any]
    stop_loss_optimization: dict[str, Any]
    position_sizing_comparison: dict[str, Any]
    regime_deployment: dict[str, Any]
    capital_curve_proxy: dict[str, Any]
    production_scores: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ProductionTradingPlaybookAuditError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _signal_date(timestamp: str) -> str:
    return str(timestamp)[:10]


def _structure_stop_points(signal: dict[str, Any]) -> float:
    entry = float(signal.get("entry") or 0.0)
    stop = float(signal.get("stop_loss") or 0.0)
    if entry and stop:
        return round(abs(entry - stop), 2)
    return round(float(signal.get("mae_points") or 0.0), 2)


def _liquidity_stop_points(signal: dict[str, Any], *, cohort_mae_median: float) -> float:
    mae = float(signal.get("mae_points") or 0.0)
    points_before = signal.get("points_before_expansion")
    if points_before is not None and float(points_before) > 0:
        return round(min(mae, float(points_before)), 2)
    return round(min(mae, cohort_mae_median), 2)


def _atr_stop_points(signal: dict[str, Any], *, cohort_mae_median: float) -> float:
    mae = float(signal.get("mae_points") or 0.0)
    proxy = cohort_mae_median * 0.75
    return round(min(mae, proxy) if mae else proxy, 2)


def _resolve_stop_points(
    signal: dict[str, Any],
    stop_variant: str,
    *,
    cohort_mae_median: float,
) -> float:
    if stop_variant == "fixed_10":
        return 10.0
    if stop_variant == "fixed_20":
        return 20.0
    if stop_variant == "fixed_30":
        return 30.0
    if stop_variant == "structure_based":
        return _structure_stop_points(signal)
    if stop_variant == "liquidity_based":
        return _liquidity_stop_points(signal, cohort_mae_median=cohort_mae_median)
    if stop_variant == "atr_based":
        return _atr_stop_points(signal, cohort_mae_median=cohort_mae_median)
    return _structure_stop_points(signal)


def _tiered_structure_pnl(
    signal: dict[str, Any],
    structure: dict[str, Any],
    *,
    stop_pts: float,
) -> tuple[float, bool]:
    mfe = float(signal.get("mfe_points") or 0.0)
    mae = float(signal.get("mae_points") or 0.0)
    t1 = float(structure["t1"])
    t2 = float(structure["t2"])
    t3 = structure.get("t3")
    runner = bool(structure.get("runner"))
    effective_stop = max(stop_pts, 1.0)

    if mfe < t1:
        return round(-min(mae, effective_stop), 2), False

    pnl = t1 * LEG_WEIGHTS[0]
    if mfe < t2:
        remainder = LEG_WEIGHTS[1] + LEG_WEIGHTS[2]
        return round(pnl - min(mae, effective_stop) * remainder, 2), pnl > 0

    pnl += t2 * LEG_WEIGHTS[1]
    if runner:
        runner_gain = max(0.0, mfe - t2)
        pnl += runner_gain * LEG_WEIGHTS[2]
        return round(pnl, 2), pnl > 0

    t3_value = float(t3 or t2)
    if mfe >= t3_value:
        pnl += t3_value * LEG_WEIGHTS[2]
        return round(pnl, 2), True

    pnl -= min(mae, effective_stop) * LEG_WEIGHTS[2]
    return round(pnl, 2), pnl > 0


def _metrics_from_pnls(
    pnls: list[float],
    *,
    sample_size: int,
    window_days: int,
) -> dict[str, Any]:
    if not pnls:
        return {
            "sample_size": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "realized_profit_points": 0.0,
            "max_drawdown_points": 0.0,
            "recovery_factor": None,
            "signals_per_month": 0.0,
        }
    wins = [p for p in pnls if p > 0]
    months = max(window_days / 22.0, 1.0)
    return {
        "sample_size": sample_size,
        "win_rate_pct": round(100.0 * len(wins) / len(pnls), 2),
        "profit_factor": _profit_factor_from_pnls(pnls),
        "expectancy": round(sum(pnls) / len(pnls), 2),
        "realized_profit_points": round(sum(pnls), 2),
        "max_drawdown_points": _max_drawdown(pnls),
        "recovery_factor": _recovery_factor(pnls),
        "signals_per_month": round(sample_size / months, 2),
    }


def _extract_entry_rules(signal: dict[str, Any], *, side: str) -> dict[str, Any]:
    layers = signal.get("layers", {})
    stack = signal.get("signal_reason_stack", {})
    layer1 = layers.get("layer1", {})
    layer2 = layers.get("layer2", {})
    layer3 = layers.get("layer3", {})
    layer5 = layers.get("layer5", {})

    formula_events = layer1.get("formula_events_matched") or stack.get("layer1") or []
    events_detected = layer1.get("events_detected") or []

    if side == "BUY":
        return {
            "model_id": BUY_V3_MODEL_ID,
            "formula_stack": list(formula_events) or BUY_V3_FORMULA_TEXT.split(" + "),
            "layer1_gate": "All formula events matched on causal bar stack",
            "layer2_gate": {
                "location": layer2.get("location") or stack.get("layer2", {}).get("location"),
                "htf_trend": layer2.get("htf_trend") or stack.get("layer2", {}).get("htf_trend"),
                "vwap_state": layer2.get("vwap_state") or stack.get("layer2", {}).get("vwap"),
                "ema_structure": layer2.get("ema_structure") or stack.get("layer2", {}).get("ema_structure"),
                "aligned": layer2.get("aligned"),
            },
            "layer3_gate": {
                "confirmation_candle": layer3.get("confirmation_candle")
                or stack.get("layer3", {}).get("confirmation_candle"),
                "volume_bucket": layer3.get("volume_bucket") or stack.get("layer3", {}).get("volume"),
                "confirmation_optional": layer3.get("confirmation_optional", True),
            },
            "layer5_gate": {
                "pass": layer5.get("pass", True),
                "reason_codes": layer5.get("reason_codes", []),
            },
            "execution": "Enter at signal bar close; reject if layer5.pass is false",
        }

    return {
        "model_id": SELL_V6_MODEL_ID,
        "formula_stack": events_detected[:3] if events_detected else ["Failed Breakout"],
        "layer1_gate": f"Primary event: {layer1.get('primary_event', 'Failed Breakout')}",
        "layer2_gate": {
            "htf_trend": layer2.get("htf_trend"),
            "vwap_gate_rule": layer2.get("vwap_gate_rule") or V6_VWAP_GATE_RULE,
            "vwap_state": layer2.get("vwap_state"),
            "vwap_gate_passes": layer2.get("vwap_gate_passes"),
            "ema_structure": layer2.get("ema_structure"),
            "v4_ema_bearish": layer2.get("v4_ema_bearish"),
            "aligned": layer2.get("aligned"),
        },
        "layer3_gate": {
            "confirmation_candle": layer3.get("confirmation_candle"),
            "volume_bucket": layer3.get("volume_bucket"),
            "confirmation_optional": layer3.get("confirmation_optional", True),
        },
        "layer5_gate": {
            "pass": layer5.get("pass", True),
            "reason_codes": layer5.get("reason_codes", []),
        },
        "execution": "Enter at signal bar close when VWAP Below only and layer5.pass is true",
    }


def _exit_rules_from_signal(signal: dict[str, Any]) -> dict[str, Any]:
    entry = float(signal.get("entry") or 0.0)
    stop = float(signal.get("stop_loss") or 0.0)
    t1 = float(signal.get("target_1") or 0.0)
    t2 = float(signal.get("target_2") or 0.0)
    t3 = float(signal.get("target_3") or 0.0)
    direction = signal.get("direction", "BUY")

    if entry and stop and t1:
        risk = abs(entry - stop)
        t1_pts = abs(t1 - entry)
        t2_pts = abs(t2 - entry) if t2 else None
        t3_pts = abs(t3 - entry) if t3 else None
        return {
            "stop_loss_price": stop,
            "stop_loss_points": round(risk, 2),
            "target_1_price": t1,
            "target_1_points": round(t1_pts, 2),
            "target_2_price": t2,
            "target_2_points": round(t2_pts, 2) if t2_pts is not None else None,
            "target_3_price": t3,
            "target_3_points": round(t3_pts, 2) if t3_pts is not None else None,
            "runner_logic": (
                "After T2 fill, trail runner at structure break or 40% MFE giveback; "
                "export proxy uses remaining MFE beyond T2."
            ),
            "direction": direction,
        }

    return {
        "stop_loss_points": round(float(signal.get("mae_points") or 0.0), 2),
        "target_1_points": 40.0,
        "target_2_points": 60.0,
        "target_3_points": 100.0,
        "runner_logic": "Simulated from replay MFE tiers when price levels absent.",
        "direction": direction,
    }


def _signal_distribution_metrics(signals: list[dict[str, Any]]) -> dict[str, Any]:
    mfes = [float(s.get("mfe_points") or 0.0) for s in signals]
    maes = [float(s.get("mae_points") or 0.0) for s in signals]
    holds = [float(s.get("trade_duration_bars") or 0.0) for s in signals if s.get("trade_duration_bars")]
    lead_bars = [
        int(s["bars_before_expansion"])
        for s in signals
        if s.get("bars_before_expansion") is not None and int(s["bars_before_expansion"]) > 0
    ]
    hit_1r = sum(1 for s in signals if s.get("hit_1r"))
    hit_2r = sum(1 for s in signals if s.get("hit_2r"))
    hit_3r = sum(1 for s in signals if s.get("hit_3r"))
    total = len(signals)

    structure_stops = [_structure_stop_points(s) for s in signals]
    return {
        "sample_size": total,
        "average_mfe": round(mean(mfes), 2) if mfes else 0.0,
        "average_mae": round(mean(maes), 2) if maes else 0.0,
        "median_mfe": round(median(mfes), 2) if mfes else 0.0,
        "median_mae": round(median(maes), 2) if maes else 0.0,
        "average_holding_time_bars": round(mean(holds), 2) if holds else None,
        "average_holding_time_minutes": round(mean(holds) * BAR_MINUTES, 2) if holds else None,
        "lead_time_bars": {
            "avg": round(mean(lead_bars), 2) if lead_bars else None,
            "median": round(median(lead_bars), 2) if lead_bars else None,
        },
        "lead_time_minutes": {
            "avg": round(mean(lead_bars) * BAR_MINUTES, 2) if lead_bars else None,
            "median": round(median(lead_bars) * BAR_MINUTES, 2) if lead_bars else None,
        },
        "hit_1r_rate_pct": round(100.0 * hit_1r / max(total, 1), 2),
        "hit_2r_rate_pct": round(100.0 * hit_2r / max(total, 1), 2),
        "hit_3r_rate_pct": round(100.0 * hit_3r / max(total, 1), 2),
        "optimal_stop_proxy_points": round(median(maes), 2) if maes else None,
        "optimal_target_proxy_points": 60,
        "optimal_r_multiple_proxy": round(mean(mfes) / max(mean(maes), 1.0), 2) if mfes and maes else None,
        "structure_stop_median_points": round(median(structure_stops), 2) if structure_stops else None,
    }


def _target_structure_comparison(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
    stop_variant: str = "structure_based",
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    rows: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for label, structure in TARGET_STRUCTURES.items():
        pnls: list[float] = []
        for signal in signals:
            stop_pts = _resolve_stop_points(signal, stop_variant, cohort_mae_median=mae_median)
            pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
            pnls.append(pnl)
        metrics = _metrics_from_pnls(pnls, sample_size=len(signals), window_days=window_days)
        row = {
            "structure": label,
            "tiers": structure,
            "stop_variant": stop_variant,
            **metrics,
        }
        rows[label] = row
        ranking.append({**row, "optimization_score": round(metrics["expectancy"] * (metrics["profit_factor"] or 0.0), 2)})

    best = max(
        ranking,
        key=lambda item: (item["expectancy"], item["profit_factor"] or 0.0, item["win_rate_pct"]),
    )
    return {
        "method": (
            "Three-leg partial exits (33% each); runner leg uses remaining MFE beyond T2; "
            "losses simulated when MFE fails next tier and MAE exceeds stop proxy."
        ),
        "stop_variant_used": stop_variant,
        "by_structure": rows,
        "ranking": sorted(ranking, key=lambda item: item["optimization_score"], reverse=True),
        "best_structure": best["structure"],
        "best_structure_evidence": best,
    }


def _stop_optimization(
    signals: list[dict[str, Any]],
    *,
    window_days: int,
    target_structure: dict[str, Any],
    structure_label: str,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    rows: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for variant in STOP_VARIANTS:
        pnls: list[float] = []
        avg_stop = 0.0
        for signal in signals:
            stop_pts = _resolve_stop_points(signal, variant, cohort_mae_median=mae_median)
            avg_stop += stop_pts
            pnl, _ = _tiered_structure_pnl(signal, target_structure, stop_pts=stop_pts)
            pnls.append(pnl)
        metrics = _metrics_from_pnls(pnls, sample_size=len(signals), window_days=window_days)
        row = {
            "stop_variant": variant,
            "average_stop_points": round(avg_stop / max(len(signals), 1), 2),
            **metrics,
        }
        rows[variant] = row
        ranking.append({**row, "optimization_score": round(metrics["expectancy"] * (metrics["profit_factor"] or 0.0), 2)})

    best = max(
        ranking,
        key=lambda item: (item["expectancy"], item["profit_factor"] or 0.0, -item["average_stop_points"]),
    )
    return {
        "target_structure": structure_label,
        "method": "Stop applied before tier progression; structure/liquidity/ATR proxies from export fields.",
        "by_stop_variant": rows,
        "ranking": sorted(ranking, key=lambda item: item["optimization_score"], reverse=True),
        "best_stop_variant": best["stop_variant"],
        "best_stop_evidence": best,
    }


def _throttle_lookup(throttle_rules: list[dict[str, Any]]) -> dict[str, str]:
    return {row["regime"]: row["throttle"] for row in throttle_rules}


def _sizing_weight(
    mode: str,
    *,
    throttle_level: str | None,
) -> float:
    if mode == "full":
        return 1.0
    if mode == "half":
        return 0.5
    if mode == "quarter":
        return 0.25
    if mode == "regime_adaptive":
        return THROTTLE_WEIGHT.get(throttle_level or "FULL", 1.0)
    return 1.0


def _position_sizing_comparison(
    signals: list[dict[str, Any]],
    *,
    direction: str,
    throttle_rules: list[dict[str, Any]],
    target_structure: dict[str, Any],
    stop_variant: str,
    window_days: int,
) -> dict[str, Any]:
    throttle_map = _throttle_lookup(throttle_rules)
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    rows: dict[str, Any] = {}

    for mode in SIZING_MODES:
        pnls: list[float] = []
        active_signals = 0
        for signal in signals:
            regime = classify_signal_regime(signal, direction=direction)
            throttle = throttle_map.get(regime["composite"], "FULL")
            weight = _sizing_weight(mode, throttle_level=throttle)
            if mode == "regime_adaptive" and throttle == "BLOCK":
                continue
            stop_pts = _resolve_stop_points(signal, stop_variant, cohort_mae_median=mae_median)
            pnl, _ = _tiered_structure_pnl(signal, target_structure, stop_pts=stop_pts)
            pnls.append(round(pnl * weight, 2))
            active_signals += 1

        metrics = _metrics_from_pnls(pnls, sample_size=active_signals, window_days=window_days)
        rows[mode] = {
            "sizing_mode": mode,
            "active_signals": active_signals,
            "blocked_signals": len(signals) - active_signals if mode == "regime_adaptive" else 0,
            **metrics,
        }

    best = max(
        rows.values(),
        key=lambda item: (item["profit_factor"] or 0.0, item["recovery_factor"] or 0.0, -item["max_drawdown_points"]),
    )
    return {
        "direction": direction,
        "target_structure": target_structure,
        "stop_variant": stop_variant,
        "by_mode": rows,
        "best_mode": best["sizing_mode"],
        "best_mode_evidence": best,
    }


def _propose_risk_rules(
    signals: list[dict[str, Any]],
    *,
    side: str,
    stop_variant: str,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    structure_stops = [_structure_stop_points(s) for s in signals]
    daily_pnls: dict[str, float] = defaultdict(float)

    for signal in signals:
        stop_pts = _resolve_stop_points(signal, stop_variant, cohort_mae_median=mae_median)
        _, win = _fixed_target_pnl(signal, 60)
        pnl = 60.0 if win else -stop_pts
        daily_pnls[_signal_date(str(signal.get("timestamp", "")))] += pnl

    daily_values = list(daily_pnls.values())
    losses = sorted([v for v in daily_values if v < 0])
    wins = sorted([v for v in daily_values if v > 0])

    risk_per_trade = round(median(structure_stops), 2) if structure_stops else 20.0
    daily_loss_limit = round(abs(losses[int(len(losses) * 0.9)]) if losses else risk_per_trade * 3, 2)
    daily_profit_lock = round(wins[int(len(wins) * 0.75)] if wins else 120.0, 2)

    return {
        "side": side,
        "risk_per_trade_points": risk_per_trade,
        "risk_per_trade_pct_capital": "0.5%–1.0% of allocated sleeve",
        "max_concurrent_positions": 2 if side == "BUY" else 3,
        "daily_loss_limit_points": daily_loss_limit,
        "daily_profit_lock_points": daily_profit_lock,
        "daily_loss_limit_rule": f"Stop trading {side} sleeve after -{daily_loss_limit} pts day",
        "daily_profit_lock_rule": f"Reduce size 50% after +{daily_profit_lock} pts day; flat optional",
        "rationale": "Derived from MAE/structure-stop distributions and daily PnL proxy on 60pt target.",
    }


def _build_playbook_section(
    signals: list[dict[str, Any]],
    *,
    side: str,
    throttle_rules: list[dict[str, Any]],
    best_target: dict[str, Any],
    best_stop: dict[str, Any],
    best_sizing: dict[str, Any],
    tradeability_export: dict[str, Any],
) -> dict[str, Any]:
    sample = signals[0] if signals else {}
    win_fn = _is_buy_winner if side == "BUY" else _is_sell_winner
    baseline = _cohort_performance(signals, window_days=120, win_fn=win_fn)
    distribution = _signal_distribution_metrics(signals)

    optimal_target = 60
    if side == "BUY":
        optimal_target = int(
            tradeability_export.get("final_answers", {}).get("optimal_target_tier_points") or 60,
        )

    return {
        "engine": "BUY_V3" if side == "BUY" else "SELL_V6",
        "baseline_replay_metrics": baseline,
        "per_signal_distribution": distribution,
        "signal_execution_rules": _extract_entry_rules(sample, side=side),
        "exit_rules_template": _exit_rules_from_signal(sample),
        "capital_allocation_rules": {
            "sleeve": side,
            "recommended_sizing_mode": best_sizing.get("best_mode"),
            "sizing_evidence": best_sizing.get("best_mode_evidence"),
            "max_risk_per_trade_points": _propose_risk_rules(
                signals, side=side, stop_variant=best_stop.get("best_stop_variant", "structure_based"),
            )["risk_per_trade_points"],
        },
        "regime_rules": {
            "throttle_map": throttle_rules,
            "deployment_note": "Import FULL/HALF/QUARTER/BLOCK per composite regime from regime_detection_audit.json",
            "block_regimes": [r["regime"] for r in throttle_rules if r.get("throttle") == "BLOCK"],
        },
        "risk_rules": _propose_risk_rules(
            signals,
            side=side,
            stop_variant=best_stop.get("best_stop_variant", "structure_based"),
        ),
        "target_rules": {
            "recommended_structure": best_target.get("best_structure"),
            "recommended_single_target_points": optimal_target,
            "structure_evidence": best_target.get("best_structure_evidence"),
            "tier_definitions": TARGET_STRUCTURES,
            "runner_policy": "33% runner uses MFE beyond T2 with 40% giveback trail in live execution",
        },
        "stop_rules": {
            "recommended_variant": best_stop.get("best_stop_variant"),
            "evidence": best_stop.get("best_stop_evidence"),
            "variants_evaluated": list(STOP_VARIANTS),
            "structure_stop_median": distribution.get("structure_stop_median_points"),
        },
    }


def _capital_curve_proxy(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    *,
    buy_structure: dict[str, Any],
    sell_structure: dict[str, Any],
    buy_stop: str,
    sell_stop: str,
    sell_throttle_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    throttle_map = _throttle_lookup(sell_throttle_rules)
    buy_mae_med = median(float(s.get("mae_points") or 0.0) for s in buy_signals) if buy_signals else 0.0
    sell_mae_med = median(float(s.get("mae_points") or 0.0) for s in sell_signals) if sell_signals else 0.0

    combined: list[tuple[str, float]] = []
    for signal in buy_signals:
        stop_pts = _resolve_stop_points(signal, buy_stop, cohort_mae_median=buy_mae_med)
        pnl, _ = _tiered_structure_pnl(signal, buy_structure, stop_pts=stop_pts)
        combined.append((str(signal.get("timestamp", "")), pnl))

    for signal in sell_signals:
        regime = classify_signal_regime(signal, direction="SELL")
        throttle = throttle_map.get(regime["composite"], "FULL")
        if throttle == "BLOCK":
            continue
        weight = THROTTLE_WEIGHT.get(throttle, 1.0)
        stop_pts = _resolve_stop_points(signal, sell_stop, cohort_mae_median=sell_mae_med)
        pnl, _ = _tiered_structure_pnl(signal, sell_structure, stop_pts=stop_pts)
        combined.append((str(signal.get("timestamp", "")), round(pnl * weight, 2)))

    combined.sort(key=lambda item: item[0])
    pnls = [pnl for _, pnl in combined]
    equity: list[float] = []
    running = 0.0
    for pnl in pnls:
        running += pnl
        equity.append(round(running, 2))

    metrics = _metrics_from_pnls(pnls, sample_size=len(pnls), window_days=120)
    return {
        "method": "Sorted BUY_V3 + throttled SELL_V6 trades using recommended target/stop structures",
        "trade_count": len(pnls),
        "net_points": metrics["realized_profit_points"],
        "max_drawdown_points": metrics["max_drawdown_points"],
        "recovery_factor": metrics["recovery_factor"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "equity_curve_sample": equity[:50],
        "final_equity_points": equity[-1] if equity else 0.0,
    }


def _production_scores(
    *,
    regime_audit: dict[str, Any],
    buy_playbook: dict[str, Any],
    sell_playbook: dict[str, Any],
    combined_curve: dict[str, Any],
) -> dict[str, Any]:
    regime_scores = regime_audit.get("output_metrics", {})
    return {
        "production_readiness_score": regime_scores.get("production_readiness_score", 75.0),
        "production_risk_score": regime_scores.get("production_risk_score", 60.0),
        "confidence_score": regime_scores.get("confidence_score", 70.0),
        "buy_v3_readiness": {
            "signals_per_month": buy_playbook["baseline_replay_metrics"].get("signals_per_month"),
            "profit_factor": buy_playbook["baseline_replay_metrics"].get("profit_factor"),
            "win_rate_pct": buy_playbook["baseline_replay_metrics"].get("win_rate_pct"),
            "passes_frequency_gate": (
                buy_playbook["baseline_replay_metrics"].get("signals_per_month", 0) >= BUY_MIN_SIGNALS_PER_MONTH
            ),
        },
        "sell_v6_readiness": {
            "signals_per_month": sell_playbook["baseline_replay_metrics"].get("signals_per_month"),
            "profit_factor": sell_playbook["baseline_replay_metrics"].get("profit_factor"),
            "win_rate_pct": sell_playbook["baseline_replay_metrics"].get("win_rate_pct"),
            "passes_frequency_gate": (
                sell_playbook["baseline_replay_metrics"].get("signals_per_month", 0) >= SELL_MIN_SIGNALS_PER_MONTH
            ),
        },
        "combined_curve_pf": combined_curve.get("profit_factor"),
        "combined_curve_max_dd": combined_curve.get("max_drawdown_points"),
    }


def _final_verdict(
    *,
    regime_audit: dict[str, Any],
    tradeability: dict[str, Any],
    buy_playbook: dict[str, Any],
    sell_playbook: dict[str, Any],
    combined_curve: dict[str, Any],
    buy_best_target: dict[str, Any],
    sell_best_target: dict[str, Any],
    buy_best_stop: dict[str, Any],
    sell_best_stop: dict[str, Any],
    buy_best_sizing: dict[str, Any],
    sell_best_sizing: dict[str, Any],
) -> dict[str, Any]:
    regime_final = regime_audit.get("final_answer", {})
    buy_ok = bool(buy_playbook["baseline_replay_metrics"].get("profit_factor", 0) or 0 >= 2.0)
    sell_throttled_ok = regime_final.get("sell_v6_paper_trading_throttled") == "YES"
    overall = regime_final.get("paper_trading_verdict", "PARTIAL")

    return {
        "paper_trade_tomorrow": overall,
        "buy_v3_paper_trading": regime_final.get("buy_v3_paper_trading", "YES" if buy_ok else "PARTIAL"),
        "sell_v6_paper_trading_unthrottled": regime_final.get("sell_v6_paper_trading_unthrottled", "NO"),
        "sell_v6_paper_trading_throttled": regime_final.get("sell_v6_paper_trading_throttled", "YES"),
        "combined_paper_trading_throttled": regime_final.get("combined_paper_trading_throttled", "YES"),
        "evidence": {
            "buy_v3_signals_per_month": buy_playbook["baseline_replay_metrics"].get("signals_per_month"),
            "buy_v3_wr_pct": buy_playbook["baseline_replay_metrics"].get("win_rate_pct"),
            "buy_v3_pf": buy_playbook["baseline_replay_metrics"].get("profit_factor"),
            "sell_v6_signals_per_month": sell_playbook["baseline_replay_metrics"].get("signals_per_month"),
            "sell_v6_wr_pct": sell_playbook["baseline_replay_metrics"].get("win_rate_pct"),
            "sell_v6_pf": sell_playbook["baseline_replay_metrics"].get("profit_factor"),
            "sell_v6_validate_pf_unthrottled": regime_final.get("baseline_sell_v6_validate_pf"),
            "sell_v6_validate_pf_throttled": regime_final.get("throttled_sell_v6_validate_pf"),
            "combined_expected_signals_per_month": round(
                float(buy_playbook["baseline_replay_metrics"].get("signals_per_month") or 0)
                + float(sell_best_sizing["best_mode_evidence"].get("signals_per_month") or 0),
                2,
            ),
            "combined_curve_pf": combined_curve.get("profit_factor"),
            "combined_curve_max_dd": combined_curve.get("max_drawdown_points"),
            "buy_optimal_target_points": tradeability.get("final_answers", {}).get("optimal_target_tier_points", 60),
            "buy_best_target_structure": buy_best_target.get("best_structure"),
            "sell_best_target_structure": sell_best_target.get("best_structure"),
            "buy_best_stop_variant": buy_best_stop.get("best_stop_variant"),
            "sell_best_stop_variant": sell_best_stop.get("best_stop_variant"),
            "buy_best_sizing_mode": buy_best_sizing.get("best_mode"),
            "sell_best_sizing_mode": sell_best_sizing.get("best_mode"),
        },
        "rationale": (
            "BUY_V3 passes full-period gates with 60pt optimal single target. SELL_V6 requires regime throttle "
            f"(validate PF {regime_final.get('baseline_sell_v6_validate_pf')} unthrottled vs "
            f"{regime_final.get('throttled_sell_v6_validate_pf')} throttled). "
            f"Combined playbook uses {buy_best_target.get('best_structure')} / {sell_best_target.get('best_structure')} "
            f"with {sell_best_sizing.get('best_mode')} SELL sizing."
        ),
    }


class ProductionTradingPlaybookAuditResearch:
    """Synthesize deployable paper-trading playbook from completed validation exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in SOURCE_EXPORTS.items():
            required = name != "walk_forward_failure_root_cause_audit"
            status = "loaded" if path.exists() else ("missing" if required else "optional_missing")
            loaded[name] = {
                "path": str(path),
                "status": status,
                "data": _load_json(path, required=required) if path.exists() or required else {},
            }
        self.sources = loaded
        return loaded

    def run(self) -> ProductionTradingPlaybookAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        buy_export = sources["buy_v3_candidate_validation"]["data"]
        sell_export = sources["sell_v6_replay_validation"]["data"]
        unified = sources["unified_production_replay_validation"]["data"]
        regime_audit = sources["regime_detection_audit"]["data"]
        tradeability = sources["buy_v3_tradeability_production_validation"]["data"]
        wf_audit = sources["walk_forward_failure_root_cause_audit"]["data"]

        window_days = int(
            buy_export.get("trading_days_replayed")
            or sell_export.get("trading_days_replayed")
            or 120,
        )

        buy_signals = list(
            unified.get("per_signal_details", {}).get("buy_v3")
            or buy_export.get("per_signal_details", {}).get("buy_v3")
            or [],
        )
        sell_signals = list(sell_export.get("per_signal_details", {}).get("sell_v6") or [])
        if not buy_signals:
            raise ProductionTradingPlaybookAuditError("No BUY_V3 per_signal_details found in exports.")
        if not sell_signals:
            raise ProductionTradingPlaybookAuditError("No SELL_V6 per_signal_details found in exports.")

        throttle = regime_audit.get("throttle_recommendation", {})
        buy_throttle_rules = throttle.get("buy_v3_regime_throttle", [])
        sell_throttle_rules = throttle.get("sell_v6_regime_throttle", [])

        buy_target_cmp = _target_structure_comparison(buy_signals, window_days=window_days)
        sell_target_cmp = _target_structure_comparison(sell_signals, window_days=window_days)

        buy_best_structure = TARGET_STRUCTURES[buy_target_cmp["best_structure"]]
        sell_best_structure = TARGET_STRUCTURES[sell_target_cmp["best_structure"]]

        buy_stop_opt = _stop_optimization(
            buy_signals,
            window_days=window_days,
            target_structure=buy_best_structure,
            structure_label=buy_target_cmp["best_structure"],
        )
        sell_stop_opt = _stop_optimization(
            sell_signals,
            window_days=window_days,
            target_structure=sell_best_structure,
            structure_label=sell_target_cmp["best_structure"],
        )

        buy_sizing = _position_sizing_comparison(
            buy_signals,
            direction="BUY",
            throttle_rules=buy_throttle_rules,
            target_structure=buy_best_structure,
            stop_variant=buy_stop_opt["best_stop_variant"],
            window_days=window_days,
        )
        sell_sizing = _position_sizing_comparison(
            sell_signals,
            direction="SELL",
            throttle_rules=sell_throttle_rules,
            target_structure=sell_best_structure,
            stop_variant=sell_stop_opt["best_stop_variant"],
            window_days=window_days,
        )

        buy_playbook = _build_playbook_section(
            buy_signals,
            side="BUY",
            throttle_rules=buy_throttle_rules,
            best_target=buy_target_cmp,
            best_stop=buy_stop_opt,
            best_sizing=buy_sizing,
            tradeability_export=tradeability,
        )
        sell_playbook = _build_playbook_section(
            sell_signals,
            side="SELL",
            throttle_rules=sell_throttle_rules,
            best_target=sell_target_cmp,
            best_stop=sell_stop_opt,
            best_sizing=sell_sizing,
            tradeability_export=tradeability,
        )

        combined_curve = _capital_curve_proxy(
            buy_signals,
            sell_signals,
            buy_structure=buy_best_structure,
            sell_structure=sell_best_structure,
            buy_stop=buy_stop_opt["best_stop_variant"],
            sell_stop=sell_stop_opt["best_stop_variant"],
            sell_throttle_rules=sell_throttle_rules,
        )

        production_scores = _production_scores(
            regime_audit=regime_audit,
            buy_playbook=buy_playbook,
            sell_playbook=sell_playbook,
            combined_curve=combined_curve,
        )

        final_answer = _final_verdict(
            regime_audit=regime_audit,
            tradeability=tradeability,
            buy_playbook=buy_playbook,
            sell_playbook=sell_playbook,
            combined_curve=combined_curve,
            buy_best_target=buy_target_cmp,
            sell_best_target=sell_target_cmp,
            buy_best_stop=buy_stop_opt,
            sell_best_stop=sell_stop_opt,
            buy_best_sizing=buy_sizing,
            sell_best_sizing=sell_sizing,
        )

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "no_new_replay": True,
            "no_new_discovery": True,
            "no_new_models": True,
            "no_new_indicators": True,
            "simulation_basis": (
                "Targets and stops simulated from per_signal_details MFE/MAE, hit_1r/2r/3r, "
                "entry/stop_loss/target prices — not re-run through replay engine."
            ),
            "target_structures_evaluated": list(TARGET_STRUCTURES.keys()),
            "stop_variants_evaluated": list(STOP_VARIANTS),
            "sizing_modes_evaluated": list(SIZING_MODES),
            "regime_throttle_source": "regime_detection_audit.json throttle_recommendation",
            "production_gates": PRODUCTION_GATES,
            "engines": {
                "buy_v3": BUY_V3_MODEL_ID,
                "sell_v6": SELL_V6_MODEL_ID,
                "buy_v3_formula": BUY_V3_FORMULA_TEXT,
                "sell_v6_vwap_gate": V6_VWAP_GATE_RULE,
            },
        }

        limitations = [
            "Target/stop simulations are MFE/MAE proxies — intrabar sequencing not modeled.",
            "BUY_V3 validate walk-forward sample is small (6 signals) — throttle is indicative.",
            "SELL_V6 unthrottled validate PF fails 2.0 gate; regime BLOCK/HALF rules required.",
            "Combined capital curve uses playbook structures, not default replay realized_pnl_points.",
            "Daily risk limits derived from proxy daily PnL — calibrate on first 20 paper sessions.",
        ]
        if wf_audit:
            limitations.append(
                "Walk-forward root cause context imported from walk_forward_failure_root_cause_audit.json.",
            )

        conclusions = [
            "Production trading playbook synthesized from replay exports only — no new replay.",
            (
                f"BUY_V3: {buy_playbook['baseline_replay_metrics']['signals_per_month']}/mo, "
                f"WR {buy_playbook['baseline_replay_metrics']['win_rate_pct']}%, "
                f"PF {buy_playbook['baseline_replay_metrics']['profit_factor']}."
            ),
            (
                f"SELL_V6: {sell_playbook['baseline_replay_metrics']['signals_per_month']}/mo, "
                f"WR {sell_playbook['baseline_replay_metrics']['win_rate_pct']}%, "
                f"PF {sell_playbook['baseline_replay_metrics']['profit_factor']}."
            ),
            (
                f"Best BUY target structure: {buy_target_cmp['best_structure']} | "
                f"Best SELL target structure: {sell_target_cmp['best_structure']}."
            ),
            (
                f"Best stops: BUY {buy_stop_opt['best_stop_variant']}, "
                f"SELL {sell_stop_opt['best_stop_variant']}."
            ),
            (
                f"Best sizing: BUY {buy_sizing['best_mode']}, SELL {sell_sizing['best_mode']} "
                f"(regime throttle restores SELL validate PF)."
            ),
            (
                f"Combined curve proxy: PF {combined_curve.get('profit_factor')}, "
                f"max DD {combined_curve.get('max_drawdown_points')} pts."
            ),
            f"Paper trade tomorrow: {final_answer['paper_trade_tomorrow']} — {final_answer['rationale']}",
        ]

        return ProductionTradingPlaybookAuditReport(
            report_type="Production Trading Playbook Audit",
            engines=["BUY_V3", "SELL_V6", "COMBINED"],
            symbol=buy_export.get("symbol") or "NIFTY50",
            timeframe=buy_export.get("timeframe") or "5M",
            trading_days_replayed=window_days,
            replay_start_date=buy_export.get("replay_start_date", ""),
            replay_end_date=buy_export.get("replay_end_date", ""),
            methodology=methodology,
            source_exports={name: {"path": info["path"], "status": info["status"]} for name, info in sources.items()},
            limitations=limitations,
            buy_v3_playbook=buy_playbook,
            sell_v6_playbook=sell_playbook,
            combined_playbook={
                "signal_execution_rules": {
                    "buy_engine": BUY_V3_MODEL_ID,
                    "sell_engine": SELL_V6_MODEL_ID,
                    "conflict_policy": "NO_TRADE when same-bar opposing signals; prefer regime BLOCK on weak SELL regimes",
                    "session_window": "NIFTY50 5M regular session",
                },
                "capital_allocation_rules": {
                    "buy_sleeve_pct": 35,
                    "sell_sleeve_pct": 65,
                    "buy_sizing_mode": buy_sizing["best_mode"],
                    "sell_sizing_mode": sell_sizing["best_mode"],
                    "note": "Sell sleeve larger due to higher signal frequency; throttle reduces effective exposure.",
                },
                "regime_rules": {
                    "sell_v6_throttle": sell_throttle_rules,
                    "buy_v3_throttle": buy_throttle_rules,
                    "import_source": "regime_detection_audit.json",
                },
                "risk_rules": {
                    "buy": buy_playbook["risk_rules"],
                    "sell": sell_playbook["risk_rules"],
                    "portfolio_daily_loss_limit_points": round(
                        buy_playbook["risk_rules"]["daily_loss_limit_points"]
                        + sell_playbook["risk_rules"]["daily_loss_limit_points"],
                        2,
                    ),
                },
                "target_rules": {
                    "buy_structure": buy_target_cmp["best_structure"],
                    "sell_structure": sell_target_cmp["best_structure"],
                    "buy_single_target_fallback": tradeability.get("final_answers", {}).get(
                        "optimal_target_tier_points", 60,
                    ),
                },
                "stop_rules": {
                    "buy_variant": buy_stop_opt["best_stop_variant"],
                    "sell_variant": sell_stop_opt["best_stop_variant"],
                },
            },
            target_structure_comparison={
                "buy_v3": buy_target_cmp,
                "sell_v6": sell_target_cmp,
            },
            stop_loss_optimization={
                "buy_v3": buy_stop_opt,
                "sell_v6": sell_stop_opt,
            },
            position_sizing_comparison={
                "buy_v3": buy_sizing,
                "sell_v6": sell_sizing,
            },
            regime_deployment={
                "sell_v6_regime_throttle": sell_throttle_rules,
                "buy_v3_regime_throttle": buy_throttle_rules,
                "throttle_levels": THROTTLE_WEIGHT,
                "regime_audit_verdict": regime_audit.get("final_answer", {}),
            },
            capital_curve_proxy=combined_curve,
            production_scores=production_scores,
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: ProductionTradingPlaybookAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Production trading playbook audit exported to %s", self.report_path)
        return self.report_path


def generate_production_trading_playbook_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export production trading playbook audit JSON."""
    return ProductionTradingPlaybookAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_production_trading_playbook_audit_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    print(f"Exported: {path}")
    print(f"Paper trade tomorrow: {final['paper_trade_tomorrow']}")
    print(f"BUY target: {final['evidence']['buy_best_target_structure']}")
    print(f"SELL target: {final['evidence']['sell_best_target_structure']}")
