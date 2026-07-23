"""
Signal Monetization & Target Probability Audit — synthesis from existing exports only.

Answers: if a BUY_V3 / SELL_V6 signal fires, what is the probability of reaching
20/40/60/80/100/150/200/300 points before stop, which target structure monetizes
best, and the concrete production playbook (T1/T2/runner/SL).

No new replay, BUY_V4, SELL_V7, indicators, models, or discovery.
Primary window: extended_trade_level_truth_audit (240d); 120d truth audits supplement.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v3_candidate_validation_research import BAR_MINUTES, BUY_V3_MODEL_ID
from src.research.buy_v3_tradeability_production_validation_research import _fixed_target_pnl
from src.research.production_reality_audit_research import _extended_metrics
from src.research.production_trading_playbook_audit_research import (
    LEG_WEIGHTS,
    _tiered_structure_pnl,
)
from src.research.regime_detection_audit_research import SELL_V6_MODEL_ID
from src.research.trade_level_truth_audit_research import _estimate_time_to_tier, _tier_reached
from src.research.live_trade_management_execution_efficiency_audit_research import (
    _resolve_stop_extended,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "signal_monetization_target_probability_audit.json"

REQUIRED_EXPORTS = {
    "extended_trade_level_truth_audit": RESEARCH_DIR / "extended_trade_level_truth_audit.json",
    "trade_level_truth_audit": RESEARCH_DIR / "trade_level_truth_audit.json",
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "extended_evidence_validation_real_deployment_audit": RESEARCH_DIR
    / "extended_evidence_validation_real_deployment_audit.json",
}

DEFAULT_STOP_VARIANT = "fixed_10"
DEFAULT_STOP_POINTS = 10.0
TRADING_DAYS_PER_MONTH = 22.0
RR_MULTIPLES = (1, 2, 3, 5)
PATH_LEVELS = (20, 40, 60, 80, 100, 150, 200, 300)

MONETIZATION_STRUCTURES: dict[str, dict[str, Any]] = {
    "40 Fixed": {"kind": "fixed", "target": 40, "t1": 40, "t2": 40, "t3": 40, "runner": False},
    "60 Fixed": {"kind": "fixed", "target": 60, "t1": 60, "t2": 60, "t3": 60, "runner": False},
    "100 Fixed": {"kind": "fixed", "target": 100, "t1": 100, "t2": 100, "t3": 100, "runner": False},
    "40/80/Runner": {"kind": "tiered", "t1": 40, "t2": 80, "t3": None, "runner": True},
    "60/100/Runner": {"kind": "tiered", "t1": 60, "t2": 100, "t3": None, "runner": True},
    "100/Runner": {"kind": "tiered", "t1": 100, "t2": 100, "t3": None, "runner": True},
}


class SignalMonetizationTargetProbabilityAuditError(Exception):
    """Raised when signal monetization audit synthesis fails."""


@dataclass
class SignalMonetizationTargetProbabilityAuditReport:
    """Signal monetization and target probability audit output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    primary_window_days: int
    supplement_window_days: int
    replay_start_date: str
    replay_end_date: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    target_probability_before_stop: dict[str, Any]
    target_path_analysis: dict[str, Any]
    reward_risk_analysis: dict[str, Any]
    target_structure_comparison: dict[str, Any]
    production_playbook: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise SignalMonetizationTargetProbabilityAuditError(f"Missing export: {path}")
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


def _extract_signals(export: dict[str, Any], engine_key: str) -> list[dict[str, Any]]:
    """Pull per-signal rows from candidate/replay or extended truth exports."""
    details = export.get("per_signal_details")
    if isinstance(details, dict):
        rows = details.get(engine_key)
        if isinstance(rows, list) and rows:
            return list(rows)
        # Window-nested variant (rare)
        for value in details.values():
            if isinstance(value, dict) and isinstance(value.get(engine_key), list):
                return list(value[engine_key])
    records = _nested(export, "per_signal_records", engine_key, "records")
    if isinstance(records, list) and records:
        # Truth-audit records use mfe/mae aliases — normalize for path math
        normalized: list[dict[str, Any]] = []
        for row in records:
            item = dict(row)
            if "mfe_points" not in item and "mfe" in item:
                item["mfe_points"] = item["mfe"]
            if "mae_points" not in item and "mae" in item:
                item["mae_points"] = item["mae"]
            normalized.append(item)
        return normalized
    return []


def _reached_before_stop(
    signal: dict[str, Any],
    threshold: int,
    *,
    stop_pts: float = DEFAULT_STOP_POINTS,
) -> bool:
    """Conservative MFE/MAE proxy: tier hit and MAE stayed inside stop."""
    if not _tier_reached(signal, threshold):
        return False
    mae = float(signal.get("mae_points") or 0.0)
    return mae < stop_pts


def _time_to_level_minutes(signal: dict[str, Any], threshold: int) -> tuple[float | None, str]:
    """Prefer explicit per-signal timing fields; else duration×(tier/mfe) proxy."""
    if not _tier_reached(signal, threshold):
        return None, "UNAVAILABLE"

    for key in (
        f"time_to_{threshold}",
        f"time_to_{threshold}_minutes",
        f"time_to_target_{threshold}",
        f"minutes_to_{threshold}",
    ):
        value = signal.get(key)
        if value is not None:
            try:
                return round(float(value), 2), "measured"
            except (TypeError, ValueError):
                pass

    for key in (f"bars_to_{threshold}", f"bars_to_target_{threshold}", f"bars_to_{threshold}_points"):
        value = signal.get(key)
        if value is not None:
            try:
                return round(float(value) * BAR_MINUTES, 2), "measured_bars"
            except (TypeError, ValueError):
                pass

    proxied = _estimate_time_to_tier(signal, threshold)
    if proxied is None:
        return None, "UNAVAILABLE"
    return proxied, "derived_from_path"


def _target_probability_matrix(
    signals: list[dict[str, Any]],
    *,
    side: str,
    window_days: int,
    provenance: str,
) -> dict[str, Any]:
    total = len(signals)
    by_tier: dict[str, Any] = {}

    for threshold in PATH_LEVELS:
        reached_signals = [s for s in signals if _reached_before_stop(s, threshold)]
        count = len(reached_signals)
        frequency = round(count / max(total, 1), 4)
        probability_pct = round(100.0 * frequency, 2)

        # Fixed-target monetization at this tier (stop = fixed_10 MAE clip)
        wins = 0
        losses = 0
        for signal in signals:
            win, pnl = _fixed_target_pnl(signal, threshold)
            # Cap loss at stop distance for fixed_10 context
            if win:
                wins += 1
            else:
                losses += 1
                _ = pnl  # mae-based loss retained in structure sim elsewhere

        by_tier[str(threshold)] = {
            "count": count,
            "frequency": frequency,
            "probability_pct": probability_pct,
            "win_pct": round(100.0 * wins / max(total, 1), 2),
            "loss_pct": round(100.0 * losses / max(total, 1), 2),
            "label": f"P(reach {threshold}+ before stop | signal fires)",
        }

    return {
        "side": side,
        "sample_size": total,
        "window_days": window_days,
        "stop_context": DEFAULT_STOP_VARIANT,
        "stop_points": DEFAULT_STOP_POINTS,
        "provenance": provenance,
        "methodology": (
            "Reach-before-stop: mfe_points >= tier AND mae_points < fixed_10 stop. "
            "Win%/Loss% from fixed-target simulation at that tier (mfe>=tier win else loss). "
            "derived_from_path — not a new bar replay."
        ),
        "by_tier": by_tier,
    }


def _path_node_stats(
    signals: list[dict[str, Any]],
    threshold: int | str,
) -> dict[str, Any]:
    if threshold == "Stop":
        stopped = sum(
            1
            for s in signals
            if float(s.get("mae_points") or 0.0) >= DEFAULT_STOP_POINTS
            and float(s.get("mfe_points") or 0.0) < 20
        )
        return {
            "level": "Stop",
            "count": stopped,
            "probability_pct": round(100.0 * stopped / max(len(signals), 1), 2),
            "avg_time_to_reach_minutes": None,
            "median_time_to_reach_minutes": None,
            "max_time_to_reach_minutes": None,
            "time_provenance": "UNAVAILABLE",
            "note": "Early stop cohort: MAE>=10 and MFE<20 (proxy)",
        }

    tier = int(threshold)
    times: list[float] = []
    provenances: list[str] = []
    for signal in signals:
        if not _reached_before_stop(signal, tier):
            continue
        minutes, prov = _time_to_level_minutes(signal, tier)
        if minutes is not None:
            times.append(minutes)
            provenances.append(prov)

    reached = sum(1 for s in signals if _reached_before_stop(s, tier))
    time_prov = "UNAVAILABLE"
    if provenances:
        if all(p == "measured" for p in provenances):
            time_prov = "measured"
        elif any(p.startswith("measured") for p in provenances):
            time_prov = "mixed_measured_derived"
        else:
            time_prov = "derived_from_path"

    return {
        "level": tier,
        "count": reached,
        "probability_pct": round(100.0 * reached / max(len(signals), 1), 2),
        "avg_time_to_reach_minutes": round(mean(times), 2) if times else None,
        "median_time_to_reach_minutes": round(median(times), 2) if times else None,
        "max_time_to_reach_minutes": round(max(times), 2) if times else None,
        "time_provenance": time_prov,
        "time_sample_size": len(times),
    }


def _target_path_analysis(
    signals: list[dict[str, Any]],
    *,
    side: str,
    window_days: int,
    provenance: str,
) -> dict[str, Any]:
    path_labels: list[Any] = ["Signal", *PATH_LEVELS, "Stop"]
    nodes: dict[str, Any] = {"Signal": {
        "level": "Signal",
        "count": len(signals),
        "probability_pct": 100.0,
        "avg_time_to_reach_minutes": 0.0,
        "median_time_to_reach_minutes": 0.0,
        "max_time_to_reach_minutes": 0.0,
        "time_provenance": "measured",
    }}
    for level in PATH_LEVELS:
        nodes[str(level)] = _path_node_stats(signals, level)
    nodes["Stop"] = _path_node_stats(signals, "Stop")

    transitions: list[dict[str, Any]] = []
    ordered = ["Signal", *[str(t) for t in PATH_LEVELS], "Stop"]
    for idx in range(len(ordered) - 1):
        left = ordered[idx]
        right = ordered[idx + 1]
        left_n = nodes[left]["count"]
        right_n = nodes[right]["count"] if right != "Stop" else nodes[right]["count"]
        if left == "Signal":
            cond = round(100.0 * nodes[right]["count"] / max(left_n, 1), 2) if right != "Stop" else None
        elif right == "Stop":
            cond = None
        else:
            cond = round(100.0 * right_n / max(left_n, 1), 2)
        transitions.append(
            {
                "from": left,
                "to": right,
                "conditional_probability_pct": cond,
                "label": f"{left} → {right}",
            },
        )

    return {
        "side": side,
        "sample_size": len(signals),
        "window_days": window_days,
        "provenance": provenance,
        "path": "Signal → 20 → 40 → 60 → 80 → 100 → 150 → 200 → 300 → Stop",
        "nodes": nodes,
        "transitions": transitions,
        "path_labels": path_labels,
        "methodology": (
            "Times: prefer time_to_*/bars_to_* fields when present; else "
            "trade_duration_bars*(tier/mfe)*BAR_MINUTES. Missing → null/UNAVAILABLE."
        ),
    }


def _reward_risk_analysis(
    signals: list[dict[str, Any]],
    *,
    side: str,
    window_days: int,
    provenance: str,
) -> dict[str, Any]:
    total = len(signals)
    stop = DEFAULT_STOP_POINTS
    by_rr: dict[str, Any] = {}
    rr_ratios: list[float] = []

    for multiple in RR_MULTIPLES:
        target_pts = stop * multiple
        hits = 0
        for signal in signals:
            mfe = float(signal.get("mfe_points") or 0.0)
            mae = float(signal.get("mae_points") or 0.0)
            # 1:N before stop — MFE covers N×stop while MAE stays inside stop
            if mfe >= target_pts and mae < stop:
                hits += 1
            # Also track realized RR = mfe / max(mae, stop) for distribution
        by_rr[f"1:{multiple}"] = {
            "target_points": target_pts,
            "stop_points": stop,
            "count": hits,
            "probability_pct": round(100.0 * hits / max(total, 1), 2),
            "label": f"P(reach {multiple}R = {target_pts}pts before stop | signal)",
        }

    for signal in signals:
        mfe = float(signal.get("mfe_points") or 0.0)
        mae = float(signal.get("mae_points") or 0.0)
        denom = max(mae, stop, 1.0)
        rr_ratios.append(round(mfe / denom, 4))

    return {
        "side": side,
        "sample_size": total,
        "window_days": window_days,
        "stop_context": DEFAULT_STOP_VARIANT,
        "stop_points": stop,
        "provenance": provenance,
        "methodology": (
            "RR outcomes derived_from_path: hit when mfe >= N*fixed_10 and mae < fixed_10. "
            "Average/median RR = mfe / max(mae, stop)."
        ),
        "by_rr": by_rr,
        "distribution": {
            "avg_rr": round(mean(rr_ratios), 3) if rr_ratios else None,
            "median_rr": round(median(rr_ratios), 3) if rr_ratios else None,
            "max_rr": round(max(rr_ratios), 3) if rr_ratios else None,
        },
    }


def _structure_pnl(
    signal: dict[str, Any],
    structure: dict[str, Any],
    *,
    stop_pts: float,
) -> float:
    if structure.get("kind") == "fixed":
        target = int(structure["target"])
        win, pnl = _fixed_target_pnl(signal, target)
        if win:
            return float(target)
        return round(-min(float(signal.get("mae_points") or 0.0), stop_pts), 2)
    pnl, _ = _tiered_structure_pnl(signal, structure, stop_pts=stop_pts)
    return pnl


def _target_structure_comparison(
    signals: list[dict[str, Any]],
    *,
    side: str,
    window_days: int,
    provenance: str,
) -> dict[str, Any]:
    mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
    by_structure: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for label, structure in MONETIZATION_STRUCTURES.items():
        pnls: list[float] = []
        for signal in signals:
            stop_pts = _resolve_stop_extended(
                signal, DEFAULT_STOP_VARIANT, cohort_mae_median=mae_median,
            )
            pnls.append(_structure_pnl(signal, structure, stop_pts=stop_pts))

        metrics = _extended_metrics(
            pnls, signals=signals, sample_size=len(signals), window_days=window_days,
        )
        row = {
            "structure": label,
            "definition": {
                "kind": structure.get("kind"),
                "t1": structure.get("t1"),
                "t2": structure.get("t2"),
                "t3": structure.get("t3"),
                "runner": structure.get("runner"),
                "leg_weights": list(LEG_WEIGHTS) if structure.get("kind") == "tiered" else None,
            },
            "expected_wr_pct": metrics["win_rate_pct"],
            "expected_pf": metrics["profit_factor"],
            "expected_expectancy": metrics["expectancy"],
            "expected_capture_pct": metrics["capture_efficiency_pct"],
            "monthly_points_estimate": metrics["monthly_points"],
            "max_drawdown_points": metrics["max_drawdown_points"],
            "provenance_label": "derived_from_path",
            "sample_size": len(signals),
        }
        by_structure[label] = row
        ranking.append(
            {
                **row,
                "optimization_score": round(
                    (metrics["expectancy"] or 0.0) * (metrics["profit_factor"] or 0.0),
                    2,
                ),
            },
        )

    best = max(
        ranking,
        key=lambda item: (
            item["expected_expectancy"] or 0.0,
            item["expected_pf"] or 0.0,
            item["expected_capture_pct"] or 0.0,
        ),
    )
    return {
        "side": side,
        "window_days": window_days,
        "provenance": provenance,
        "stop_variant": DEFAULT_STOP_VARIANT,
        "methodology": (
            "Structures simulated from per-signal MFE/MAE paths (derived_from_path), "
            "not a new replay. Prefer measured core metrics from extended_trade_level when cited."
        ),
        "by_structure": by_structure,
        "ranking": sorted(ranking, key=lambda item: item["optimization_score"], reverse=True),
        "best_structure": best["structure"],
        "best_structure_evidence": best,
    }


def _playbook_for_engine(
    *,
    engine: str,
    best_structure_label: str,
    structure_cmp: dict[str, Any],
    path: dict[str, Any],
    rr: dict[str, Any],
    core_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    structure = MONETIZATION_STRUCTURES[best_structure_label]
    evidence = structure_cmp["best_structure_evidence"]
    t1 = int(structure.get("t1") or structure.get("target") or 60)
    t2 = int(structure.get("t2") or t1)
    runner = bool(structure.get("runner"))

    if runner:
        runner_until = (
            f"trail remaining 1/3 after T2 ({t2} pts) until MFE exhaustion / session end; "
            "do not hard-exit at a fixed T3"
        )
        move_sl = (
            f"after T1 ({t1} pts) fill -> move SL to entry (breakeven); "
            f"after T2 -> trail under/over swing"
        )
    else:
        runner_until = "N/A (fixed target — flat all size at target)"
        move_sl = "keep initial fixed_10 until target or stop; no scale"

    one_liner = (
        f"{engine}: Take T1 at {t1} | Take T2 at {t2} | "
        f"Keep Runner until {'MFE trail after T2' if runner else 'N/A'} | "
        f"Move SL when {'T1 hits -> BE' if runner else 'never (fixed)'}"
    )

    return {
        "engine": engine,
        "if_signal_fires": {
            "take_t1_at_points": t1,
            "take_t2_at_points": t2 if runner or t2 != t1 else t1,
            "keep_runner_until": runner_until,
            "move_sl_when": move_sl,
            "stop_variant": DEFAULT_STOP_VARIANT,
            "stop_points": DEFAULT_STOP_POINTS,
            "leg_weights": list(LEG_WEIGHTS) if runner else [1.0],
        },
        "best_structure": best_structure_label,
        "structure_metrics": {
            "expected_wr_pct": evidence.get("expected_wr_pct"),
            "expected_pf": evidence.get("expected_pf"),
            "expected_expectancy": evidence.get("expected_expectancy"),
            "expected_capture_pct": evidence.get("expected_capture_pct"),
        },
        "path_anchors": {
            "p_60_before_stop": _nested(path, "nodes", "60", "probability_pct"),
            "p_100_before_stop": _nested(path, "nodes", "100", "probability_pct"),
            "p_1_to_3_rr": _nested(rr, "by_rr", "1:3", "probability_pct"),
        },
        "measured_core_metrics": core_metrics or {},
        "one_liner": one_liner,
        "provenance": "derived_from_path structure sim + measured extended core metrics where available",
    }


def _confidence_evidence_scores(
    *,
    buy_n: int,
    sell_n: int,
    primary_window: int,
    structures_agree: bool,
    has_extended: bool,
    has_evidence: bool,
) -> dict[str, Any]:
    # Sample adequacy (n>=100 buy, n>=300 sell preferred)
    buy_sample = min(100.0, buy_n / 100.0 * 100.0)
    sell_sample = min(100.0, sell_n / 300.0 * 100.0)
    window_score = min(100.0, primary_window / 240.0 * 100.0)
    agreement = 90.0 if structures_agree else 65.0
    source_score = 40.0 + (30.0 if has_extended else 0.0) + (30.0 if has_evidence else 0.0)

    confidence = round(
        0.25 * buy_sample + 0.25 * sell_sample + 0.20 * window_score + 0.15 * agreement + 0.15 * source_score,
        1,
    )
    evidence = round(
        0.35 * (100.0 if has_extended else 40.0)
        + 0.25 * (100.0 if has_evidence else 40.0)
        + 0.20 * window_score
        + 0.20 * ((buy_sample + sell_sample) / 2.0),
        1,
    )
    return {
        "confidence_score": confidence,
        "evidence_score": evidence,
        "components": {
            "buy_sample_score": round(buy_sample, 1),
            "sell_sample_score": round(sell_sample, 1),
            "window_score": round(window_score, 1),
            "structure_agreement_score": agreement,
            "source_coverage_score": round(source_score, 1),
        },
    }


def _monthly_return_estimate(
    *,
    buy_expectancy: float,
    sell_expectancy: float,
    buy_spm: float,
    sell_spm: float,
    buy_dd: float | None,
    sell_dd: float | None,
) -> dict[str, Any]:
    buy_monthly = round(buy_expectancy * buy_spm, 2)
    sell_monthly = round(sell_expectancy * sell_spm, 2)
    combined = round(buy_monthly + sell_monthly, 2)
    return {
        "label": "estimate",
        "formula": "expectancy × signals_per_month (≈ expectancy × signals/day × 22)",
        "buy_v3_monthly_points": buy_monthly,
        "sell_v6_monthly_points": sell_monthly,
        "combined_monthly_points": combined,
        "inputs": {
            "buy_expectancy": buy_expectancy,
            "sell_expectancy": sell_expectancy,
            "buy_signals_per_month": buy_spm,
            "sell_signals_per_month": sell_spm,
        },
        "expected_drawdown": {
            "buy_v3_max_drawdown_points": buy_dd,
            "sell_v6_max_drawdown_points": sell_dd,
            "note": "Measured from extended core metrics when available; else derived path DD",
        },
    }


class SignalMonetizationTargetProbabilityAuditResearch:
    """Synthesize signal monetization / target probability audit from exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in REQUIRED_EXPORTS.items():
            exists = path.exists()
            loaded[name] = {
                "path": str(path),
                "status": "loaded" if exists else "missing",
                "data": _load_json(path, required=True),
            }
        self.sources = loaded
        return loaded

    def run(self) -> SignalMonetizationTargetProbabilityAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        etl = sources["extended_trade_level_truth_audit"]["data"]
        tla = sources["trade_level_truth_audit"]["data"]
        buy_export = sources["buy_v3_candidate_validation"]["data"]
        sell_export = sources["sell_v6_replay_validation"]["data"]
        evidence = sources["extended_evidence_validation_real_deployment_audit"]["data"]

        primary_window = int(etl.get("max_replay_window") or (etl.get("replay_windows") or [240])[0] or 240)
        supplement_window = int(
            tla.get("trading_days_replayed")
            or buy_export.get("trading_days_replayed")
            or 120,
        )

        # Primary: 240d extended per_signal_details
        buy_primary = _extract_signals(etl, "buy_v3")
        sell_primary = _extract_signals(etl, "sell_v6")
        primary_provenance = "extended_trade_level_truth_audit.per_signal_details (240d measured replay)"

        if not buy_primary:
            buy_primary = _extract_signals(buy_export, "buy_v3")
            primary_provenance = "buy_v3_candidate_validation.per_signal_details (120d fallback)"
        if not sell_primary:
            sell_primary = _extract_signals(sell_export, "sell_v6")
            primary_provenance = (
                primary_provenance
                + " | sell_v6_replay_validation.per_signal_details (120d fallback)"
            )

        if not buy_primary:
            raise SignalMonetizationTargetProbabilityAuditError("No BUY_V3 per-signal rows in exports.")
        if not sell_primary:
            raise SignalMonetizationTargetProbabilityAuditError("No SELL_V6 per-signal rows in exports.")

        # Supplement 120d
        buy_supp = _extract_signals(buy_export, "buy_v3") or _extract_signals(tla, "buy_v3")
        sell_supp = _extract_signals(sell_export, "sell_v6") or _extract_signals(tla, "sell_v6")

        buy_prob = _target_probability_matrix(
            buy_primary, side="BUY", window_days=primary_window, provenance=primary_provenance,
        )
        sell_prob = _target_probability_matrix(
            sell_primary, side="SELL", window_days=primary_window, provenance=primary_provenance,
        )

        buy_path = _target_path_analysis(
            buy_primary, side="BUY", window_days=primary_window, provenance=primary_provenance,
        )
        sell_path = _target_path_analysis(
            sell_primary, side="SELL", window_days=primary_window, provenance=primary_provenance,
        )

        buy_rr = _reward_risk_analysis(
            buy_primary, side="BUY", window_days=primary_window, provenance=primary_provenance,
        )
        sell_rr = _reward_risk_analysis(
            sell_primary, side="SELL", window_days=primary_window, provenance=primary_provenance,
        )

        buy_structures = _target_structure_comparison(
            buy_primary, side="BUY", window_days=primary_window, provenance=primary_provenance,
        )
        sell_structures = _target_structure_comparison(
            sell_primary, side="SELL", window_days=primary_window, provenance=primary_provenance,
        )

        # Measured core metrics from extended (prefer over derived)
        core_240 = _nested(etl, "core_metrics_by_window", str(primary_window)) or {}
        buy_core = core_240.get("buy_v3") or {}
        sell_core = core_240.get("sell_v6") or {}
        combined_core = core_240.get("combined") or {}
        throttled_core = core_240.get("combined_regime_throttle") or {}

        buy_best = buy_structures["best_structure"]
        sell_best = sell_structures["best_structure"]
        shared_stack = buy_best == sell_best
        preferred = "60/100/Runner"

        def _expectancy(cmp: dict[str, Any], label: str) -> float:
            return float((cmp["by_structure"].get(label) or {}).get("expected_expectancy") or 0.0)

        def _near_best(cmp: dict[str, Any], label: str, *, tol_pct: float = 15.0) -> bool:
            best_exp = _expectancy(cmp, cmp["best_structure"])
            cand = _expectancy(cmp, label)
            if best_exp <= 0:
                return label == cmp["best_structure"]
            return cand >= best_exp * (1.0 - tol_pct / 100.0)

        # Unify on 60/100/Runner when it is best or within 15% expectancy on both sides
        if preferred in MONETIZATION_STRUCTURES and _near_best(buy_structures, preferred) and _near_best(
            sell_structures, preferred,
        ):
            playbook_structure = preferred
            playbook_note = (
                "60/100/Runner selected as shared production stack "
                "(best or within 15% expectancy for both engines)."
            )
        elif shared_stack:
            playbook_structure = buy_best
            playbook_note = f"Shared stack {playbook_structure} is path-best for both engines."
        else:
            runner_labels = [k for k, v in MONETIZATION_STRUCTURES.items() if v.get("runner")]
            buy_runner = max(runner_labels, key=lambda lbl: _expectancy(buy_structures, lbl))
            sell_runner = max(runner_labels, key=lambda lbl: _expectancy(sell_structures, lbl))
            playbook_structure = buy_runner if buy_runner == sell_runner else buy_best
            playbook_note = (
                f"BUY path-best={buy_best}, SELL path-best={sell_best}; "
                f"production stack={playbook_structure}."
            )

        buy_playbook = _playbook_for_engine(
            engine="BUY_V3",
            best_structure_label=playbook_structure,
            structure_cmp={
                **buy_structures,
                "best_structure": playbook_structure,
                "best_structure_evidence": buy_structures["by_structure"][playbook_structure],
            },
            path=buy_path,
            rr=buy_rr,
            core_metrics=buy_core,
        )
        sell_playbook = _playbook_for_engine(
            engine="SELL_V6",
            best_structure_label=playbook_structure,
            structure_cmp={
                **sell_structures,
                "best_structure": playbook_structure,
                "best_structure_evidence": sell_structures["by_structure"][playbook_structure],
            },
            path=sell_path,
            rr=sell_rr,
            core_metrics=sell_core,
        )

        # Monthly return: prefer measured expectancy × spm from extended
        buy_exp = float(
            buy_core.get("expectancy")
            or buy_structures["by_structure"][playbook_structure]["expected_expectancy"]
            or 0.0,
        )
        sell_exp = float(
            sell_core.get("expectancy")
            or sell_structures["by_structure"][playbook_structure]["expected_expectancy"]
            or 0.0,
        )
        buy_spm = float(buy_core.get("signals_per_month") or (len(buy_primary) / max(primary_window / TRADING_DAYS_PER_MONTH, 1.0)))
        sell_spm = float(sell_core.get("signals_per_month") or (len(sell_primary) / max(primary_window / TRADING_DAYS_PER_MONTH, 1.0)))
        # Structure-path expectancy for playbook stack (derived) for comparison
        buy_struct_exp = float(buy_structures["by_structure"][playbook_structure]["expected_expectancy"] or 0.0)
        sell_struct_exp = float(sell_structures["by_structure"][playbook_structure]["expected_expectancy"] or 0.0)

        monthly = _monthly_return_estimate(
            buy_expectancy=buy_struct_exp,
            sell_expectancy=sell_struct_exp,
            buy_spm=buy_spm,
            sell_spm=sell_spm,
            buy_dd=buy_core.get("max_drawdown_points")
            or buy_structures["by_structure"][playbook_structure].get("max_drawdown_points"),
            sell_dd=sell_core.get("max_drawdown_points")
            or sell_structures["by_structure"][playbook_structure].get("max_drawdown_points"),
        )
        monthly["measured_core_monthly_proxy"] = {
            "buy_v3": round(buy_exp * buy_spm, 2),
            "sell_v6": round(sell_exp * sell_spm, 2),
            "combined": round(buy_exp * buy_spm + sell_exp * sell_spm, 2),
            "note": "Measured core expectancy × spm (replay realized path; may differ from structure sim)",
            "provenance": "Measured",
        }
        if throttled_core.get("expectancy") and throttled_core.get("effective_signals_per_month"):
            monthly["regime_throttled_monthly_proxy"] = {
                "value": round(
                    float(throttled_core["expectancy"])
                    * float(throttled_core["effective_signals_per_month"]),
                    2,
                ),
                "provenance": "Measured",
                "source": "extended_trade_level.core_metrics_by_window.combined_regime_throttle",
            }

        scores = _confidence_evidence_scores(
            buy_n=len(buy_primary),
            sell_n=len(sell_primary),
            primary_window=primary_window,
            structures_agree=shared_stack or playbook_structure == preferred,
            has_extended=sources["extended_trade_level_truth_audit"]["status"] == "loaded",
            has_evidence=sources["extended_evidence_validation_real_deployment_audit"]["status"] == "loaded"
            and bool(evidence),
        )

        # Best RR structure: highest probability meaningful RR that still clears 1:2
        def _best_rr(rr_block: dict[str, Any]) -> str:
            by = rr_block.get("by_rr") or {}
            # Prefer 1:3 if prob >= 40%, else 1:2, else 1:1
            p3 = _nested(by, "1:3", "probability_pct") or 0.0
            p2 = _nested(by, "1:2", "probability_pct") or 0.0
            if p3 >= 40.0:
                return "1:3 (fixed_10 -> 30pts)"
            if p2 >= 50.0:
                return "1:2 (fixed_10 -> 20pts)"
            return "1:1 (fixed_10 -> 10pts)"

        best_rr_combined = _best_rr(sell_rr)  # SELL dominates frequency; cite both
        buy_best_rr = _best_rr(buy_rr)
        sell_best_rr = _best_rr(sell_rr)

        final_answer = {
            "best_monetization_model": {
                "value": f"Partial scale-out + runner ({playbook_structure})",
                "rationale": playbook_note,
            },
            "best_target_structure": {
                "value": playbook_structure,
                "buy_v3_path_best": buy_best,
                "sell_v6_path_best": sell_best,
                "shared_production_stack": playbook_structure,
            },
            "best_exit_structure": {
                "value": playbook_structure,
                "t1": MONETIZATION_STRUCTURES[playbook_structure]["t1"],
                "t2": MONETIZATION_STRUCTURES[playbook_structure]["t2"],
                "runner": MONETIZATION_STRUCTURES[playbook_structure].get("runner"),
                "stop": DEFAULT_STOP_VARIANT,
            },
            "best_rr_structure": {
                "value": sell_best_rr if sell_best_rr == buy_best_rr else f"BUY {buy_best_rr} | SELL {sell_best_rr}",
                "buy_v3": buy_best_rr,
                "sell_v6": sell_best_rr,
                "combined_emphasis": best_rr_combined,
                "stop_context": DEFAULT_STOP_VARIANT,
            },
            "expected_monthly_return": monthly,
            "expected_drawdown": monthly["expected_drawdown"],
            "confidence_score": scores["confidence_score"],
            "evidence_score": scores["evidence_score"],
            "score_components": scores["components"],
            "production_playbook_one_liners": {
                "buy_v3": buy_playbook["one_liner"],
                "sell_v6": sell_playbook["one_liner"],
                "shared": (
                    f"SHARED {playbook_structure}: Take T1 at "
                    f"{MONETIZATION_STRUCTURES[playbook_structure]['t1']} | "
                    f"Take T2 at {MONETIZATION_STRUCTURES[playbook_structure]['t2']} | "
                    + (
                        "Keep Runner until MFE trail after T2 | Move SL when T1 hits -> BE"
                        if MONETIZATION_STRUCTURES[playbook_structure].get("runner")
                        else "Keep Runner until N/A (flat at fixed target) | "
                        "Move SL when never (fixed_10 until target/stop)"
                    )
                ),
            },
        }

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "no_new_replay": True,
            "no_new_engines": True,
            "engines": [BUY_V3_MODEL_ID, SELL_V6_MODEL_ID],
            "primary_window": f"{primary_window}d extended_trade_level_truth_audit",
            "supplement_window": f"{supplement_window}d trade_level / candidate validation",
            "stop_context": DEFAULT_STOP_VARIANT,
            "target_tiers": list(PATH_LEVELS),
            "structures_evaluated": list(MONETIZATION_STRUCTURES.keys()),
            "rr_multiples": [f"1:{m}" for m in RR_MULTIPLES],
            "provenance_policy": (
                "Prefer Measured figures from extended_trade_level / evidence exports; "
                "structure/path sims labeled derived_from_path; missing timing → null/UNAVAILABLE."
            ),
            "excluded": "BUY_V4, SELL_V7, new indicators/models/discovery/replay",
        }

        limitations = [
            "Synthesis-only: no new replay; MFE/MAE path proxies do not model intrabar stop/target order.",
            "Reach-before-stop is conservative (mae < fixed_10); optimistic mfe-only rates are higher.",
            "Time-to-tier usually derived_from_path (duration×tier/mfe) unless explicit fields exist.",
            "Fixed/tiered structure metrics are derived_from_path — not broker fills.",
            "Monthly return is an estimate (expectancy × signals/month).",
        ]
        if not buy_supp:
            limitations.append("120d BUY supplement signals unavailable.")
        if not sell_supp:
            limitations.append("120d SELL supplement signals unavailable.")

        supplement_block = {
            "window_days": supplement_window,
            "buy_v3": {
                "sample_size": len(buy_supp),
                "target_probability_before_stop": _target_probability_matrix(
                    buy_supp, side="BUY", window_days=supplement_window,
                    provenance="120d supplement",
                ) if buy_supp else None,
                "best_structure": _target_structure_comparison(
                    buy_supp, side="BUY", window_days=supplement_window,
                    provenance="120d supplement",
                )["best_structure"] if buy_supp else None,
            },
            "sell_v6": {
                "sample_size": len(sell_supp),
                "target_probability_before_stop": _target_probability_matrix(
                    sell_supp, side="SELL", window_days=supplement_window,
                    provenance="120d supplement",
                ) if sell_supp else None,
                "best_structure": _target_structure_comparison(
                    sell_supp, side="SELL", window_days=supplement_window,
                    provenance="120d supplement",
                )["best_structure"] if sell_supp else None,
            },
        }

        conclusions = [
            (
                f"Primary window {primary_window}d: BUY n={len(buy_primary)} | "
                f"SELL n={len(sell_primary)}."
            ),
            (
                f"P(60+ before stop): BUY {buy_prob['by_tier']['60']['probability_pct']}% | "
                f"SELL {sell_prob['by_tier']['60']['probability_pct']}%."
            ),
            (
                f"P(100+ before stop): BUY {buy_prob['by_tier']['100']['probability_pct']}% | "
                f"SELL {sell_prob['by_tier']['100']['probability_pct']}%."
            ),
            (
                f"Best structures (path sim): BUY {buy_best} | SELL {sell_best} | "
                f"production stack {playbook_structure}."
            ),
            buy_playbook["one_liner"],
            sell_playbook["one_liner"],
            (
                f"Estimated monthly points (structure sim): "
                f"{monthly['combined_monthly_points']} | "
                f"confidence={scores['confidence_score']} evidence={scores['evidence_score']}."
            ),
        ]

        source_meta = {
            name: {"path": meta["path"], "status": meta["status"]}
            for name, meta in sources.items()
        }

        return SignalMonetizationTargetProbabilityAuditReport(
            report_type="Signal Monetization & Target Probability Audit",
            engines=["BUY_V3", "SELL_V6"],
            symbol=str(etl.get("symbol") or buy_export.get("symbol") or "NIFTY50"),
            timeframe=str(etl.get("timeframe") or buy_export.get("timeframe") or "5M"),
            primary_window_days=primary_window,
            supplement_window_days=supplement_window,
            replay_start_date=str(etl.get("replay_start_date") or buy_export.get("replay_start_date") or ""),
            replay_end_date=str(etl.get("replay_end_date") or buy_export.get("replay_end_date") or ""),
            methodology=methodology,
            source_exports=source_meta,
            limitations=limitations,
            target_probability_before_stop={
                "primary": {"buy_v3": buy_prob, "sell_v6": sell_prob},
                "supplement_120d": supplement_block,
                "measured_conditional_from_extended": _nested(
                    etl, "conditional_probability", str(primary_window),
                )
                or _nested(core_240, "conditional_probability"),
            },
            target_path_analysis={"buy_v3": buy_path, "sell_v6": sell_path},
            reward_risk_analysis={"buy_v3": buy_rr, "sell_v6": sell_rr},
            target_structure_comparison={
                "buy_v3": buy_structures,
                "sell_v6": sell_structures,
                "playbook_stack": playbook_structure,
                "playbook_note": playbook_note,
                "measured_core_metrics_240d": {
                    "buy_v3": buy_core,
                    "sell_v6": sell_core,
                    "combined": combined_core,
                    "combined_regime_throttle": throttled_core,
                    "provenance": "Measured",
                },
            },
            production_playbook={
                "buy_v3": buy_playbook,
                "sell_v6": sell_playbook,
                "shared_stack": playbook_structure,
                "shared_one_liner": final_answer["production_playbook_one_liners"]["shared"],
            },
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(
        self,
        report: SignalMonetizationTargetProbabilityAuditReport | None = None,
    ) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Signal monetization target probability audit exported to %s", self.report_path)
        return self.report_path


def generate_signal_monetization_target_probability_audit_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export signal monetization target probability audit JSON."""
    return SignalMonetizationTargetProbabilityAuditResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_signal_monetization_target_probability_audit_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    print(f"Exported: {path}")
    print(f"Best target structure: {final['best_target_structure']['value']}")
    print(f"Playbook: {final['production_playbook_one_liners']['shared']}")
    print(
        f"Confidence={final['confidence_score']} | Evidence={final['evidence_score']} | "
        f"Monthly est={final['expected_monthly_return']['combined_monthly_points']}"
    )
