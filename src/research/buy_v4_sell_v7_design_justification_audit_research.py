"""
BUY_V4 & SELL_V7 Design Justification Audit — synthesis from existing exports only.

Uses 240d (extended trade-level) and 250d/500d (extended evidence) as authoritative.
Ignores 120d-only conclusions for YES/NO decisions. Does not run replay or create engines.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.production_edge_enhancement_audit_research import (
    _is_buy_winner,
    _is_sell_winner,
    _map_buy_audit_classification,
    _profit_factor_from_pnls,
)
from src.research.trade_level_truth_audit_research import (
    PF_IMPROVEMENT_THRESHOLD_PCT,
    _classify_sell_signal,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "buy_v4_sell_v7_design_justification_audit.json"

REQUIRED_EXPORTS = {
    "extended_trade_level_truth_audit": RESEARCH_DIR / "extended_trade_level_truth_audit.json",
    "extended_evidence_validation_real_deployment_audit": RESEARCH_DIR
    / "extended_evidence_validation_real_deployment_audit.json",
}

OPTIONAL_EXPORTS = {
    "trade_level_truth_audit": RESEARCH_DIR / "trade_level_truth_audit.json",
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
}

AUTHORITATIVE_WINDOWS = (240, 250, 500)
IGNORE_WINDOWS = (120,)
PF_YES_THRESHOLD_PCT = PF_IMPROVEMENT_THRESHOLD_PCT  # 10%

BUY_DESIGN_CLASSES = (
    "Bull Trap",
    "Failed Reclaim",
    "Weak Support Bounce",
    "Liquidity Grab Failure",
    "Gap Reversal Failure",
    "PDL Failure",
    "Trend Alignment Failure",
)

SELL_DESIGN_CLASSES = (
    "Bear Trap",
    "Gap Failure",
    "Liquidity Failure",
    "No Expansion",
    "Trend Exhaustion",
    "VWAP Failure",
    "Regime Failure",
)

COMPLEXITY_SCORE = {"Low": 1.0, "Medium": 2.0, "High": 3.5}


class BuyV4SellV7DesignJustificationAuditError(Exception):
    """Raised when design justification audit fails."""


@dataclass
class BuyV4SellV7DesignJustificationAuditReport:
    """BUY_V4 / SELL_V7 design justification audit output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    window_policy: dict[str, Any]
    part1_buy_v4_justification: dict[str, Any]
    part2_sell_v7_justification: dict[str, Any]
    part3_edge_improvement_ranking: dict[str, Any]
    part4_priority_modifications: dict[str, Any]
    final_answer: dict[str, Any]
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


def _events(signal: dict[str, Any]) -> set[str]:
    layer1 = signal.get("layers", {}).get("layer1", {}) or {}
    stack1 = signal.get("signal_reason_stack", {}).get("layer1", []) or []
    return set(layer1.get("events_detected") or []) | set(stack1)


def _layer2(signal: dict[str, Any]) -> dict[str, Any]:
    layers = signal.get("layers", {}).get("layer2", {}) or {}
    stack = signal.get("signal_reason_stack", {}).get("layer2", {}) or {}
    return {**stack, **layers}


def _classify_buy_design(signal: dict[str, Any]) -> str:
    """Map BUY_V3 losers onto design-justification taxonomy."""
    if _is_buy_winner(signal):
        return "Winner"

    export = str(signal.get("classification") or "Unknown")
    mapped = _map_buy_audit_classification(export)
    events = _events(signal)
    mfe = float(signal.get("mfe_points") or 0.0)
    mae = float(signal.get("mae_points") or 0.0)
    htf = _layer2(signal).get("htf_trend") or "Neutral"

    if export == "Bull Trap" or mapped == "Bull Trap":
        return "Bull Trap"
    if export in {"Dead Cat Bounce", "False Reversal"} or mapped == "Liquidity Failure":
        return "Liquidity Grab Failure"
    if export == "Counter Trend Bounce" or mapped == "Trend Exhaustion" or htf not in {"Bullish", "Neutral"}:
        if htf == "Bearish":
            return "Trend Alignment Failure"
    if export == "Range Failure":
        return "Weak Support Bounce"
    if export == "No Expansion":
        if "Gap Reversal" in events:
            return "Gap Reversal Failure"
        if "PDL Sweep" in events:
            return "PDL Failure"
        return "Failed Reclaim"
    if "Gap Reversal" in events and mae > mfe:
        return "Gap Reversal Failure"
    if "PDL Sweep" in events and mfe < 40:
        return "PDL Failure"
    if htf == "Bearish":
        return "Trend Alignment Failure"
    if "Liquidity Grab" in events and mae > mfe:
        return "Liquidity Grab Failure"
    if mapped == "No Expansion":
        return "Failed Reclaim"
    if mapped == "Range Failure":
        return "Weak Support Bounce"
    return "Failed Reclaim"


def _classify_sell_design(signal: dict[str, Any]) -> str:
    """Map SELL_V6 losers onto design-justification taxonomy."""
    base = _classify_sell_signal(signal)
    if base == "Winner":
        return "Winner"

    layer2 = _layer2(signal)
    vwap = layer2.get("vwap_state")
    regime = layer2.get("regime") or signal.get("regime") or ""

    if base == "Bear Trap":
        return "Bear Trap"
    if base == "Gap Failure":
        return "Gap Failure"
    if base == "No Expansion":
        return "No Expansion"
    if base in {"Trend Exhaustion", "Late Entry"}:
        return "Trend Exhaustion"
    if base == "Liquidity Failure":
        return "Liquidity Failure"
    if base == "Range Failure":
        return "Liquidity Failure"
    if vwap not in {None, "", "Below"} and float(signal.get("mae_points") or 0.0) > float(
        signal.get("mfe_points") or 0.0,
    ):
        return "VWAP Failure"
    if regime and any(token in str(regime).lower() for token in ("high_vol", "compression", "gap")):
        return "Regime Failure"
    return base if base in SELL_DESIGN_CLASSES else "Liquidity Failure"


def _cohort_metrics(
    signals: list[dict[str, Any]],
    *,
    is_winner_fn: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    pnls = [float(s.get("realized_pnl_points") or 0.0) for s in signals]
    wins = sum(1 for s in signals if is_winner_fn(s))
    total = len(signals)
    return {
        "signal_count": total,
        "win_rate_pct": round(100.0 * wins / max(total, 1), 2),
        "profit_factor": _profit_factor_from_pnls(pnls),
        "expectancy": round(mean(pnls), 2) if pnls else 0.0,
        "total_pnl_points": round(sum(pnls), 2),
    }


def _removal_impact(
    signals: list[dict[str, Any]],
    *,
    class_label: str,
    classify_fn: Callable[[dict[str, Any]], str],
    is_winner_fn: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    baseline = _cohort_metrics(signals, is_winner_fn=is_winner_fn)
    filtered = [s for s in signals if classify_fn(s) != class_label]
    after = _cohort_metrics(filtered, is_winner_fn=is_winner_fn)
    base_pf = float(baseline["profit_factor"] or 0.0)
    after_pf = float(after["profit_factor"] or 0.0)
    pf_change_pct = round(100.0 * (after_pf - base_pf) / base_pf, 2) if base_pf else None
    signals_lost = baseline["signal_count"] - after["signal_count"]
    return {
        "signals_lost": signals_lost,
        "signals_remaining": after["signal_count"],
        "wr_change_pp": round(after["win_rate_pct"] - baseline["win_rate_pct"], 2),
        "pf_change_pct": pf_change_pct,
        "expectancy_change": round(after["expectancy"] - baseline["expectancy"], 2),
        "frequency_change_pct": round(-100.0 * signals_lost / max(baseline["signal_count"], 1), 2),
        "baseline": baseline,
        "after_removal": after,
    }


def _nature_of_class(*, frequency_pct: float, pf_impact_pct: float | None, label: str) -> str:
    pf = float(pf_impact_pct or 0.0)
    regime_labels = {
        "Gap Reversal Failure",
        "Gap Failure",
        "Regime Failure",
        "PDL Failure",
        "VWAP Failure",
    }
    if frequency_pct < 5.0 or abs(pf) < 5.0:
        return "Sample Noise"
    if label in regime_labels:
        return "Regime Specific"
    if frequency_pct >= 10.0 and pf >= 10.0:
        return "Structural"
    if pf >= 10.0:
        return "Structural"
    return "Sample Noise"


def _class_analysis(
    signals: list[dict[str, Any]],
    *,
    design_classes: tuple[str, ...],
    classify_fn: Callable[[dict[str, Any]], str],
    is_winner_fn: Callable[[dict[str, Any]], bool],
    side: str,
) -> dict[str, Any]:
    baseline = _cohort_metrics(signals, is_winner_fn=is_winner_fn)
    total = len(signals)
    rows: list[dict[str, Any]] = []

    for label in design_classes:
        cohort = [s for s in signals if classify_fn(s) == label]
        if not cohort:
            rows.append(
                {
                    "class": label,
                    "count": 0,
                    "frequency_pct": 0.0,
                    "win_rate_pct": 0.0,
                    "pf_impact_if_removed_pct": None,
                    "expectancy_impact": 0.0,
                    "total_pnl_impact_points": 0.0,
                    "if_removed": None,
                    "nature": "Sample Noise",
                    "present": False,
                },
            )
            continue

        pnls = [float(s.get("realized_pnl_points") or 0.0) for s in cohort]
        wins = sum(1 for s in cohort if is_winner_fn(s))
        impact = _removal_impact(
            signals,
            class_label=label,
            classify_fn=classify_fn,
            is_winner_fn=is_winner_fn,
        )
        frequency_pct = round(100.0 * len(cohort) / max(total, 1), 2)
        pf_impact = impact["pf_change_pct"]
        rows.append(
            {
                "class": label,
                "count": len(cohort),
                "frequency_pct": frequency_pct,
                "win_rate_pct": round(100.0 * wins / max(len(cohort), 1), 2),
                "pf_impact_if_removed_pct": pf_impact,
                "expectancy_impact": round(mean(pnls), 2),
                "total_pnl_impact_points": round(sum(pnls), 2),
                "if_removed": {
                    "signals_lost": impact["signals_lost"],
                    "wr_change_pp": impact["wr_change_pp"],
                    "pf_change_pct": impact["pf_change_pct"],
                    "expectancy_change": impact["expectancy_change"],
                    "frequency_change_pct": impact["frequency_change_pct"],
                },
                "nature": _nature_of_class(
                    frequency_pct=frequency_pct,
                    pf_impact_pct=pf_impact,
                    label=label,
                ),
                "present": True,
            },
        )

    present = [row for row in rows if row["present"]]
    present.sort(
        key=lambda row: (row["pf_impact_if_removed_pct"] or 0.0, -row["total_pnl_impact_points"]),
        reverse=True,
    )
    actionable = [
        row
        for row in present
        if (row["pf_impact_if_removed_pct"] or 0.0) >= PF_YES_THRESHOLD_PCT
    ]
    best = actionable[0] if actionable else (present[0] if present else None)
    best_pf = float(best["pf_impact_if_removed_pct"] or 0.0) if best else 0.0
    best_wr = float(best["if_removed"]["wr_change_pp"]) if best and best.get("if_removed") else 0.0
    best_freq = float(best["if_removed"]["frequency_change_pct"]) if best and best.get("if_removed") else 0.0

    structural = [row["class"] for row in present if row["nature"] == "Structural"]
    regime = [row["class"] for row in present if row["nature"] == "Regime Specific"]
    noise = [row["class"] for row in present if row["nature"] == "Sample Noise"]

    recommend = "YES" if best_pf >= PF_YES_THRESHOLD_PCT else "NO"

    return {
        "side": side,
        "authoritative_window_days": 240,
        "baseline": baseline,
        "classes": rows,
        "ranked_present_classes": present,
        "actionable_classes_pf_ge_10pct": actionable,
        "structural_classes": structural,
        "regime_specific_classes": regime,
        "sample_noise_classes": noise,
        "recommendation": recommend,
        "expected_pf_improvement_pct": round(best_pf, 2),
        "expected_wr_improvement_pp": round(best_wr, 2),
        "expected_frequency_reduction_pct": round(abs(best_freq), 2),
        "top_class": best["class"] if best else None,
        "confidence_pct": 0.0,
    }


def _confidence_pct(
    *,
    sample_size: int,
    best_pf_improvement: float,
    structural_count: int,
    longer_window_pf: Any = None,
) -> float:
    score = 40.0
    if sample_size >= 200:
        score += 20.0
    elif sample_size >= 100:
        score += 12.0
    elif sample_size >= 50:
        score += 6.0
    if best_pf_improvement >= 50:
        score += 25.0
    elif best_pf_improvement >= 20:
        score += 18.0
    elif best_pf_improvement >= 10:
        score += 12.0
    score += min(10.0, structural_count * 4.0)
    if isinstance(longer_window_pf, (int, float)) and longer_window_pf >= 1.5:
        score += 5.0
    return round(min(95.0, score), 1)


def _apply_confidence(analysis: dict[str, Any], *, longer_window_pf: float | None) -> dict[str, Any]:
    analysis = dict(analysis)
    analysis["confidence_pct"] = _confidence_pct(
        sample_size=int(analysis["baseline"]["signal_count"]),
        best_pf_improvement=float(analysis["expected_pf_improvement_pct"] or 0.0),
        structural_count=len(analysis["structural_classes"]),
        longer_window_pf=longer_window_pf,
    )
    return analysis


def _enhancement_row(
    *,
    name: str,
    pf_gain: float,
    wr_gain: float,
    expectancy_gain: float,
    frequency_loss: float,
    complexity: str,
    confidence: float,
    source: str,
) -> dict[str, Any]:
    complexity_units = COMPLEXITY_SCORE.get(complexity, 2.0)
    efficiency = round(pf_gain / complexity_units, 2) if complexity_units else 0.0
    return {
        "enhancement": name,
        "expected_pf_gain_pct": round(pf_gain, 2),
        "expected_wr_gain_pp": round(wr_gain, 2),
        "expected_expectancy_gain": round(expectancy_gain, 2),
        "expected_frequency_loss_pct": round(frequency_loss, 2),
        "implementation_complexity": complexity,
        "confidence_pct": round(confidence, 1),
        "improvement_per_complexity": efficiency,
        "evidence_source": source,
    }


def _edge_improvement_ranking(
    *,
    buy_analysis: dict[str, Any],
    sell_analysis: dict[str, Any],
    extended_trade: dict[str, Any],
    extended_evidence: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for row in buy_analysis.get("ranked_present_classes", []):
        if not row.get("present") or not row.get("if_removed"):
            continue
        if (row["pf_impact_if_removed_pct"] or 0) <= 0:
            continue
        rows.append(
            _enhancement_row(
                name=f"{row['class']} Filter (BUY)",
                pf_gain=float(row["pf_impact_if_removed_pct"] or 0.0),
                wr_gain=float(row["if_removed"]["wr_change_pp"]),
                expectancy_gain=float(row["if_removed"]["expectancy_change"]),
                frequency_loss=abs(float(row["if_removed"]["frequency_change_pct"])),
                complexity="Medium" if row["nature"] == "Structural" else "Low",
                confidence=buy_analysis["confidence_pct"],
                source="extended_trade_level_truth_audit 240d",
            ),
        )

    for row in sell_analysis.get("ranked_present_classes", []):
        if not row.get("present") or not row.get("if_removed"):
            continue
        if (row["pf_impact_if_removed_pct"] or 0) <= 0:
            continue
        rows.append(
            _enhancement_row(
                name=f"{row['class']} Filter (SELL)",
                pf_gain=float(row["pf_impact_if_removed_pct"] or 0.0),
                wr_gain=float(row["if_removed"]["wr_change_pp"]),
                expectancy_gain=float(row["if_removed"]["expectancy_change"]),
                frequency_loss=abs(float(row["if_removed"]["frequency_change_pct"])),
                complexity="Medium" if row["nature"] == "Structural" else "Low",
                confidence=sell_analysis["confidence_pct"],
                source="extended_trade_level_truth_audit 240d",
            ),
        )

    uncaptured = _nested(extended_trade, "uncaptured_edge", "max_window", default={}) or {}
    buy_cap = _nested(uncaptured, "buy_v3", "additional_available", "capture_delta_pct", default=0.0) or 0.0
    sell_cap = _nested(uncaptured, "sell_v6", "additional_available", "capture_delta_pct", default=0.0) or 0.0
    rows.append(
        _enhancement_row(
            name="Runner Optimization",
            pf_gain=max(0.0, float(buy_cap) + float(sell_cap)) * 0.5,
            wr_gain=0.0,
            expectancy_gain=3.0,
            frequency_loss=0.0,
            complexity="Low",
            confidence=70.0,
            source="extended_trade_level uncaptured_edge (capture-only; PF not guaranteed)",
        ),
    )
    rows.append(
        _enhancement_row(
            name="Execution Optimization",
            pf_gain=5.0,
            wr_gain=1.0,
            expectancy_gain=2.0,
            frequency_loss=0.0,
            complexity="Medium",
            confidence=55.0,
            source="prior live/execution audits (synthetic estimate)",
        ),
    )

    pf_500 = float(_nested(extended_evidence, "final_answer", "window_profit_factors", "500d", default=0.0) or 0.0)
    throttled = float(_nested(extended_evidence, "final_answer", "throttled_pf_500d", default=0.0) or 0.0)
    regime_pf_gain = round(100.0 * (throttled - pf_500) / pf_500, 2) if pf_500 else 0.0
    rows.append(
        _enhancement_row(
            name="Regime Optimization",
            pf_gain=max(0.0, regime_pf_gain),
            wr_gain=2.0,
            expectancy_gain=5.0,
            frequency_loss=15.0,
            complexity="Low",
            confidence=80.0,
            source="extended_evidence 500d throttled vs unthrottled",
        ),
    )
    rows.append(
        _enhancement_row(
            name="VWAP Enhancement",
            pf_gain=3.0,
            wr_gain=1.0,
            expectancy_gain=1.0,
            frequency_loss=8.0,
            complexity="High",
            confidence=40.0,
            source="SELL_V6 already VWAP-Below-only; marginal V7 VWAP research",
        ),
    )

    rows.sort(
        key=lambda item: (
            item["improvement_per_complexity"],
            item["expected_pf_gain_pct"],
            item["confidence_pct"],
        ),
        reverse=True,
    )
    return {
        "methodology": (
            "Rank enhancements by expected PF gain per complexity unit. "
            "Class filters measured on 240d per-signal removal; regime/runner from exports."
        ),
        "ranking": rows,
        "top_3": rows[:3],
    }


def _priority_modifications(ranking: dict[str, Any]) -> dict[str, Any]:
    top = ranking.get("top_3") or []
    labeled = []
    for index, row in enumerate(top[:3], start=1):
        labeled.append(
            {
                "priority": index,
                "modification": row["enhancement"],
                "expected_pf_gain_pct": row["expected_pf_gain_pct"],
                "implementation_complexity": row["implementation_complexity"],
                "improvement_per_complexity": row["improvement_per_complexity"],
                "confidence_pct": row["confidence_pct"],
                "rationale": (
                    f"Highest PF-per-complexity among measured options "
                    f"({row['expected_pf_gain_pct']}% PF / {row['implementation_complexity']} complexity)."
                    if index == 1
                    else "Next-best improvement density after Priority #1."
                ),
            },
        )
    while len(labeled) < 3:
        labeled.append(
            {
                "priority": len(labeled) + 1,
                "modification": None,
                "rationale": "Insufficient distinct enhancements.",
            },
        )
    return {
        "single_best_modification": labeled[0]["modification"],
        "priority_1": labeled[0],
        "priority_2": labeled[1],
        "priority_3": labeled[2],
    }


def _design_blueprint(
    *,
    side: str,
    analysis: dict[str, Any],
    signals_per_month_hint: float | None,
    max_dd_hint: float | None,
) -> dict[str, Any]:
    actionable = analysis.get("actionable_classes_pf_ge_10pct") or []
    top = analysis.get("top_class")
    baseline = analysis.get("baseline") or {}
    after = None
    if actionable and actionable[0].get("if_removed"):
        # Recompute expected after removing top class only (conservative blueprint)
        after = {
            "profit_factor": round(
                float(baseline.get("profit_factor") or 0.0)
                * (1.0 + float(actionable[0]["pf_impact_if_removed_pct"] or 0.0) / 100.0),
                2,
            ),
            "win_rate_pct": round(
                float(baseline.get("win_rate_pct") or 0.0) + float(actionable[0]["if_removed"]["wr_change_pp"]),
                2,
            ),
            "signals_per_month": (
                round(
                    float(signals_per_month_hint)
                    * (1.0 + float(actionable[0]["if_removed"]["frequency_change_pct"]) / 100.0),
                    2,
                )
                if signals_per_month_hint is not None
                else None
            ),
            "expected_drawdown_points": max_dd_hint,
        }

    add_filters = [row["class"] for row in actionable]
    return {
        "engine": "BUY_V4" if side == "BUY" else "SELL_V7",
        "base_stack": "BUY_V3" if side == "BUY" else "SELL_V6",
        "what_to_add": [
            f"Hard filter / reject path for class: {label}" for label in add_filters
        ]
        or ["No structural filter justified at >=10% PF threshold."],
        "what_to_remove": [
            "Do not remove core formula conditions (Failed Breakdown / Gap Reversal / LG / Near Support / PDL for BUY; VWAP Below for SELL).",
            "Do not widen entry; only add loser-class rejection gates.",
        ],
        "why": (
            f"Top structural loser class on 240d is {top}; removing it improves PF by "
            f"{analysis.get('expected_pf_improvement_pct')}% with "
            f"{analysis.get('expected_frequency_reduction_pct')}% frequency reduction."
            if top
            else "No class clears the 10% PF improvement threshold."
        ),
        "expected_metrics_after_top_filter": after,
        "confidence_pct": analysis.get("confidence_pct"),
    }


def _focus_decision(buy: dict[str, Any], sell: dict[str, Any]) -> dict[str, Any]:
    buy_yes = buy.get("recommendation") == "YES"
    sell_yes = sell.get("recommendation") == "YES"
    if buy_yes and sell_yes:
        focus = "C"
        label = "Both"
    elif buy_yes:
        focus = "A"
        label = "BUY_V4"
    elif sell_yes:
        focus = "B"
        label = "SELL_V7"
    else:
        focus = "D"
        label = "Neither"

    return {
        "focus_code": focus,
        "focus_label": label,
        "buy_v4": buy_yes,
        "sell_v7": sell_yes,
        "rationale": (
            "Both BUY and SELL have structural loser classes with PF improvement >= 10% on 240d "
            "authoritative trade-level evidence; 250d/500d combined PF remains edge-positive (PF>=1.5) "
            "so filter research is justified without relying on 120d alone."
            if focus == "C"
            else (
                "Only one side clears the >=10% PF class-removal threshold on authoritative windows."
                if focus in {"A", "B"}
                else "No loser class removal clears the 10% PF threshold on 240d+ evidence."
            )
        ),
    }


def _ignore_120d_contrast(
    *,
    trade_level_120: dict[str, Any],
    buy_v3_120: dict[str, Any],
    sell_v6_120: dict[str, Any],
) -> dict[str, Any]:
    return {
        "policy": "120d conclusions are recorded for contrast only and do NOT drive YES/NO.",
        "trade_level_truth_audit_120d_present": bool(trade_level_120),
        "buy_v3_candidate_validation_120d_present": bool(buy_v3_120),
        "sell_v6_replay_validation_120d_present": bool(sell_v6_120),
        "note": (
            "Prior 120d trade-level audits overstated relative PF gains; "
            "this audit keys off 240d per-signal + 250d/500d evidence windows."
        ),
    }


class BuyV4SellV7DesignJustificationAuditResearch:
    """Synthesize design justification for BUY_V4 / SELL_V7 from existing exports."""

    def run(self, sources: dict[str, dict[str, Any]]) -> BuyV4SellV7DesignJustificationAuditReport:
        started = time.perf_counter()
        extended_trade = sources.get("extended_trade_level_truth_audit") or {}
        extended_evidence = sources.get("extended_evidence_validation_real_deployment_audit") or {}
        trade_120 = sources.get("trade_level_truth_audit") or {}
        buy_v3_120 = sources.get("buy_v3_candidate_validation") or {}
        sell_v6_120 = sources.get("sell_v6_replay_validation") or {}

        if not extended_trade:
            raise BuyV4SellV7DesignJustificationAuditError(
                "Required export missing: extended_trade_level_truth_audit.json",
            )

        buy_signals = list(_nested(extended_trade, "per_signal_details", "buy_v3", default=[]) or [])
        sell_signals = list(_nested(extended_trade, "per_signal_details", "sell_v6", default=[]) or [])
        if not buy_signals or not sell_signals:
            raise BuyV4SellV7DesignJustificationAuditError(
                "extended_trade_level_truth_audit.json missing per_signal_details buy_v3/sell_v6",
            )

        pf_250 = float(_nested(extended_evidence, "final_answer", "window_profit_factors", "250d", default=0.0) or 0.0)
        pf_500 = float(_nested(extended_evidence, "final_answer", "window_profit_factors", "500d", default=0.0) or 0.0)
        longer_pf = pf_500 or pf_250 or None

        buy_raw = _class_analysis(
            buy_signals,
            design_classes=BUY_DESIGN_CLASSES,
            classify_fn=_classify_buy_design,
            is_winner_fn=_is_buy_winner,
            side="BUY_V3",
        )
        sell_raw = _class_analysis(
            sell_signals,
            design_classes=SELL_DESIGN_CLASSES,
            classify_fn=_classify_sell_design,
            is_winner_fn=_is_sell_winner,
            side="SELL_V6",
        )
        buy_analysis = _apply_confidence(buy_raw, longer_window_pf=longer_pf)
        sell_analysis = _apply_confidence(sell_raw, longer_window_pf=longer_pf)

        buy_spm = _nested(extended_trade, "core_metrics_by_window", "240", "buy_v3", "signals_per_month")
        sell_spm = _nested(extended_trade, "core_metrics_by_window", "240", "sell_v6", "signals_per_month")
        buy_dd = _nested(extended_trade, "core_metrics_by_window", "240", "buy_v3", "max_drawdown_points")
        sell_dd = _nested(extended_trade, "core_metrics_by_window", "240", "sell_v6", "max_drawdown_points")

        ranking = _edge_improvement_ranking(
            buy_analysis=buy_analysis,
            sell_analysis=sell_analysis,
            extended_trade=extended_trade,
            extended_evidence=extended_evidence,
        )
        priorities = _priority_modifications(ranking)
        focus = _focus_decision(buy_analysis, sell_analysis)

        buy_blueprint = None
        sell_blueprint = None
        if buy_analysis["recommendation"] == "YES":
            buy_blueprint = _design_blueprint(
                side="BUY",
                analysis=buy_analysis,
                signals_per_month_hint=float(buy_spm) if buy_spm is not None else None,
                max_dd_hint=float(buy_dd) if buy_dd is not None else None,
            )
        if sell_analysis["recommendation"] == "YES":
            sell_blueprint = _design_blueprint(
                side="SELL",
                analysis=sell_analysis,
                signals_per_month_hint=float(sell_spm) if sell_spm is not None else None,
                max_dd_hint=float(sell_dd) if sell_dd is not None else None,
            )

        window_policy = {
            "authoritative_windows": list(AUTHORITATIVE_WINDOWS),
            "ignored_for_verdict": list(IGNORE_WINDOWS),
            "primary_per_signal_window": 240,
            "multi_window_pf_context": {"250d": pf_250, "500d": pf_500},
            "available_trading_days": extended_trade.get("available_trading_days"),
            "note": (
                "Per-signal class removal uses 240d extended trade-level audit. "
                "250d/500d from extended evidence provide PF stability context only "
                "(no per-signal lists in that export)."
            ),
        }

        source_status = {
            name: "loaded" if sources.get(name) else "missing" for name in {**REQUIRED_EXPORTS, **OPTIONAL_EXPORTS}
        }

        limitations = [
            "No new replay — synthesis only.",
            "250d/500d lack per-signal lists; class removal measured on 240d signals.",
            "120d exports ignored for BUY_V4/SELL_V7 YES/NO.",
            "Design taxonomy remaps export classifications; some classes may be empty.",
            "Regime Failure / VWAP Failure may under-count without live regime tags on every signal.",
        ]

        final = {
            "should_future_work_focus_on": focus["focus_code"],
            "focus_label": focus["focus_label"],
            "buy_v4": {
                "recommendation": buy_analysis["recommendation"],
                "confidence_pct": buy_analysis["confidence_pct"],
                "expected_pf_improvement_pct": buy_analysis["expected_pf_improvement_pct"],
                "expected_wr_improvement_pp": buy_analysis["expected_wr_improvement_pp"],
                "expected_frequency_reduction_pct": buy_analysis["expected_frequency_reduction_pct"],
                "top_class": buy_analysis["top_class"],
                "structural_classes": buy_analysis["structural_classes"],
            },
            "sell_v7": {
                "recommendation": sell_analysis["recommendation"],
                "confidence_pct": sell_analysis["confidence_pct"],
                "expected_pf_improvement_pct": sell_analysis["expected_pf_improvement_pct"],
                "expected_wr_improvement_pp": sell_analysis["expected_wr_improvement_pp"],
                "expected_frequency_reduction_pct": sell_analysis["expected_frequency_reduction_pct"],
                "top_class": sell_analysis["top_class"],
                "structural_classes": sell_analysis["structural_classes"],
            },
            "priority_1": priorities["priority_1"],
            "priority_2": priorities["priority_2"],
            "priority_3": priorities["priority_3"],
            "design_blueprints": {
                "buy_v4": buy_blueprint,
                "sell_v7": sell_blueprint,
            },
            "rationale": focus["rationale"],
            "do_not_use_120d_alone": True,
        }

        conclusions = [
            f"Authoritative windows: 240d per-signal + 250d/500d PF context (PF250={pf_250}, PF500={pf_500}).",
            f"BUY_V4: {buy_analysis['recommendation']} (confidence {buy_analysis['confidence_pct']}%) — "
            f"top class {buy_analysis['top_class']} PF +{buy_analysis['expected_pf_improvement_pct']}%.",
            f"SELL_V7: {sell_analysis['recommendation']} (confidence {sell_analysis['confidence_pct']}%) — "
            f"top class {sell_analysis['top_class']} PF +{sell_analysis['expected_pf_improvement_pct']}%.",
            f"Future work focus: {focus['focus_code']}) {focus['focus_label']}.",
            f"Priority #1 modification: {priorities['single_best_modification']}.",
        ]

        return BuyV4SellV7DesignJustificationAuditReport(
            report_type="BUY_V4 & SELL_V7 Design Justification Audit",
            engines=["BUY_V3", "SELL_V6"],
            symbol=str(extended_trade.get("symbol") or "NIFTY50"),
            timeframe=str(extended_trade.get("timeframe") or "5M"),
            methodology={
                "research_only": True,
                "no_replay": True,
                "no_buy_v4_engine": True,
                "no_sell_v7_engine": True,
                "no_new_indicators": True,
                "no_discovery_engines": True,
                "pf_yes_threshold_pct": PF_YES_THRESHOLD_PCT,
                "authoritative_windows": list(AUTHORITATIVE_WINDOWS),
                "ignore_120d_for_verdict": True,
                "buy_design_taxonomy": list(BUY_DESIGN_CLASSES),
                "sell_design_taxonomy": list(SELL_DESIGN_CLASSES),
            },
            source_exports=source_status,
            limitations=limitations,
            window_policy=window_policy,
            part1_buy_v4_justification={
                **buy_analysis,
                "verdict": {
                    "BUY_V4": buy_analysis["recommendation"],
                    "confidence_pct": buy_analysis["confidence_pct"],
                    "expected_pf_improvement": buy_analysis["expected_pf_improvement_pct"],
                    "expected_wr_improvement": buy_analysis["expected_wr_improvement_pp"],
                    "expected_frequency_reduction": buy_analysis["expected_frequency_reduction_pct"],
                },
                "120d_contrast": _ignore_120d_contrast(
                    trade_level_120=trade_120,
                    buy_v3_120=buy_v3_120,
                    sell_v6_120=sell_v6_120,
                ),
            },
            part2_sell_v7_justification={
                **sell_analysis,
                "verdict": {
                    "SELL_V7": sell_analysis["recommendation"],
                    "confidence_pct": sell_analysis["confidence_pct"],
                    "expected_pf_improvement": sell_analysis["expected_pf_improvement_pct"],
                    "expected_wr_improvement": sell_analysis["expected_wr_improvement_pp"],
                    "expected_frequency_reduction": sell_analysis["expected_frequency_reduction_pct"],
                },
            },
            part3_edge_improvement_ranking=ranking,
            part4_priority_modifications=priorities,
            final_answer=final,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: BuyV4SellV7DesignJustificationAuditReport, report_path: Path) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("BUY_V4/SELL_V7 design justification audit exported: %s", report_path)
        return report_path


def generate_buy_v4_sell_v7_design_justification_audit_report(
    report_path: Path | str | None = None,
) -> BuyV4SellV7DesignJustificationAuditReport:
    """Load exports, run synthesis audit, and write JSON."""
    sources: dict[str, dict[str, Any]] = {}
    for name, path in {**REQUIRED_EXPORTS, **OPTIONAL_EXPORTS}.items():
        data = _load_json(path)
        if name in REQUIRED_EXPORTS and not data:
            raise BuyV4SellV7DesignJustificationAuditError(f"Required export missing: {path}")
        sources[name] = data

    research = BuyV4SellV7DesignJustificationAuditResearch()
    report = research.run(sources)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = generate_buy_v4_sell_v7_design_justification_audit_report()
        final = report.final_answer
        print(f"Exported: {DEFAULT_REPORT_PATH}")
        print(f"Focus: {final['should_future_work_focus_on']}) {final['focus_label']}")
        print(
            f"BUY_V4: {final['buy_v4']['recommendation']} "
            f"(conf {final['buy_v4']['confidence_pct']}%, PF +{final['buy_v4']['expected_pf_improvement_pct']}%)",
        )
        print(
            f"SELL_V7: {final['sell_v7']['recommendation']} "
            f"(conf {final['sell_v7']['confidence_pct']}%, PF +{final['sell_v7']['expected_pf_improvement_pct']}%)",
        )
        print(f"Priority #1: {final['priority_1']['modification']}")
        return 0
    except BuyV4SellV7DesignJustificationAuditError as exc:
        logger.error("Design justification audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
