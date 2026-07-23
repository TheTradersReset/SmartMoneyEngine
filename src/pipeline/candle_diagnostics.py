"""
Structured per-candle diagnostics for the realtime BUY_V3 / SELL_V6 pipeline.

Observability only — reads evaluation dicts produced by the signal engines;
does not modify strategy logic, thresholds, or scoring.
"""

from __future__ import annotations

import json
from typing import Any

from src.data.candle_builder import Candle
from src.signals.regime_throttle import ThrottleDecision

RULE_DESCRIPTIONS: dict[str, str] = {
    "FORMULA_INCOMPLETE": "BUY formula events incomplete in lookback window",
    "NO_EARLY_WARNING": "SELL layer1 early-warning events absent",
    "NO_FAILED_BREAKOUT": "Failed Breakout not detected in lookback",
    "HTF_CONFLICT": "Higher-timeframe trend conflicts with signal direction",
    "LOCATION_MISMATCH": "Price location does not match required zone (Near Support for BUY)",
    "DIRECTION_NOT_ALIGNED": "Directional alignment failed (HTF/VWAP/EMA/location gate)",
    "VOLUME_FAILED": "Volume bucket or confirmation candle check failed",
    "CONFIRMATION_FAILED": "Confirmation candle / volume check failed",
    "LOCATION_MID_RANGE": "Price in Mid Range (avoid zone)",
    "DUPLICATE_BAR": "Signal already emitted on this bar",
    "VWAP_MISMATCH": "VWAP gate failed (SELL_V6 requires Below)",
    "EMA_MISMATCH": "EMA structure conflicts with signal direction",
    "LAYER4_EXECUTION_FAILED": "Layer5 passed but structural execution (stop/risk) failed",
    "SAME_BAR_CONFLICT": "Both BUY and SELL fired on the same bar",
    "REGIME_BLOCK": "Regime throttle blocked this composite regime",
    "INSUFFICIENT_BARS": "Not enough history bars for signal evaluation",
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _fmt(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _describe_rule(code: str) -> str:
    if code.startswith("MISSING_"):
        event = code.removeprefix("MISSING_").replace("_", " ").title()
        return f"Required formula event missing: {event}"
    if code.startswith("REGIME_BLOCK:"):
        return f"{RULE_DESCRIPTIONS['REGIME_BLOCK']} ({code.removeprefix('REGIME_BLOCK:')})"
    return RULE_DESCRIPTIONS.get(code, code)


def _event_flag(events: set[str] | list[str], *names: str) -> str:
    normalized = set(events)
    for name in names:
        if name in normalized:
            return "Present"
    return "Absent"


def _buy_formula_score(layer1: dict[str, Any]) -> float:
    matched = layer1.get("formula_events_matched") or []
    missing = layer1.get("formula_events_missing") or []
    total = len(matched) + len(missing)
    if total == 0:
        return 0.0
    return round(100.0 * len(matched) / total, 1)


def _sell_gate_score(evaluation: dict[str, Any]) -> float:
    layer1 = evaluation.get("layer1") or {}
    layer2 = evaluation.get("layer2") or {}
    layer3 = evaluation.get("layer3") or {}
    gates = [
        bool(layer1.get("active")),
        bool(layer1.get("failed_breakout_present")),
        layer2.get("htf_trend") == "Bearish",
        bool(layer2.get("vwap_gate_passes")),
        bool(layer2.get("v4_ema_bearish") or layer2.get("ema_structure") != "Bull Stack"),
        bool(layer2.get("aligned")),
        bool(layer3.get("confirmed")),
    ]
    return round(100.0 * sum(gates) / len(gates), 1)


def _buy_condition_checks(evaluation: dict[str, Any], *, bar: int) -> list[dict[str, Any]]:
    layer1 = evaluation.get("layer1") or {}
    layer2 = evaluation.get("layer2") or {}
    layer3 = evaluation.get("layer3") or {}
    layer5 = evaluation.get("layer5") or {}
    context = evaluation.get("context") or {}
    checks: list[dict[str, Any]] = []

    for event in layer1.get("formula_events_matched") or []:
        checks.append({"rule": f"FORMULA_EVENT:{event}", "passed": True, "detail": f"{event} present in lookback"})
    for event in layer1.get("formula_events_missing") or []:
        checks.append(
            {
                "rule": f"MISSING_{event.upper().replace(' ', '_')}",
                "passed": False,
                "detail": _describe_rule(f"MISSING_{event.upper().replace(' ', '_')}"),
            }
        )

    checks.extend(
        [
            {
                "rule": "FORMULA_COMPLETE",
                "passed": bool(layer1.get("active")),
                "detail": "All BUY_V3 formula events present in lookback",
            },
            {
                "rule": "HTF_NOT_BEARISH",
                "passed": layer2.get("htf_trend") != "Bearish",
                "detail": f"HTF trend={layer2.get('htf_trend')}",
            },
            {
                "rule": "VWAP_BULLISH_GATE",
                "passed": layer2.get("vwap_state") in {"Above", "Reclaimed", "Rejected"},
                "detail": f"VWAP={layer2.get('vwap_state')}",
            },
            {
                "rule": "EMA_NOT_BEAR_STACK",
                "passed": layer2.get("ema_structure") != "Bear Stack",
                "detail": f"EMA structure={layer2.get('ema_structure')}",
            },
            {
                "rule": "LOCATION_NEAR_SUPPORT",
                "passed": bool(layer2.get("location_ok")),
                "detail": f"Location={layer2.get('location')} (required={layer2.get('location_required')})",
            },
            {
                "rule": "DIRECTION_ALIGNED",
                "passed": bool(layer2.get("aligned")),
                "detail": "BUY directional alignment",
            },
            {
                "rule": "VOLUME_CONFIRMED",
                "passed": bool(layer3.get("confirmed")),
                "detail": f"Volume={layer3.get('volume_bucket')} candle={layer3.get('confirmation_candle')}",
            },
            {
                "rule": "NOT_MID_RANGE",
                "passed": context.get("location") != "Mid Range",
                "detail": f"Location={context.get('location')}",
            },
            {
                "rule": "NOT_DUPLICATE_BAR",
                "passed": "DUPLICATE_BAR" not in (layer5.get("reason_codes") or []),
                "detail": "No prior BUY emission on this bar",
            },
            {
                "rule": "LAYER5_PASS",
                "passed": bool(layer5.get("pass")),
                "detail": "All hard filters passed",
            },
        ]
    )

    if layer5.get("pass") and evaluation.get("verdict") != "BUY":
        checks.append(
            {
                "rule": "LAYER4_EXECUTION",
                "passed": False,
                "detail": _describe_rule("LAYER4_EXECUTION_FAILED"),
            }
        )
    elif evaluation.get("verdict") == "BUY":
        checks.append(
            {
                "rule": "LAYER4_EXECUTION",
                "passed": True,
                "detail": "Structural stop/risk execution succeeded",
            }
        )

    return checks


def _sell_condition_checks(evaluation: dict[str, Any], *, bar: int) -> list[dict[str, Any]]:
    layer1 = evaluation.get("layer1") or {}
    layer2 = evaluation.get("layer2") or {}
    layer3 = evaluation.get("layer3") or {}
    layer5 = evaluation.get("layer5") or {}
    context = evaluation.get("context") or {}

    checks: list[dict[str, Any]] = [
        {
            "rule": "EARLY_WARNING",
            "passed": bool(layer1.get("active")),
            "detail": f"Events={layer1.get('events_detected')}",
        },
        {
            "rule": "FAILED_BREAKOUT",
            "passed": bool(layer1.get("failed_breakout_present")),
            "detail": "Failed Breakout in lookback",
        },
        {
            "rule": "HTF_BEARISH",
            "passed": layer2.get("htf_trend") == "Bearish",
            "detail": f"HTF trend={layer2.get('htf_trend')}",
        },
        {
            "rule": "VWAP_BELOW_GATE",
            "passed": bool(layer2.get("vwap_gate_passes")),
            "detail": f"VWAP={layer2.get('vwap_state')} rule={layer2.get('vwap_gate_rule')}",
        },
        {
            "rule": "EMA_BEARISH",
            "passed": bool(layer2.get("v4_ema_bearish")) or layer2.get("ema_structure") != "Bull Stack",
            "detail": f"EMA={layer2.get('ema_structure')} v4_bearish={layer2.get('v4_ema_bearish')}",
        },
        {
            "rule": "DIRECTION_ALIGNED",
            "passed": bool(layer2.get("aligned")),
            "detail": "SELL directional alignment",
        },
        {
            "rule": "VOLUME_CONFIRMED",
            "passed": bool(layer3.get("confirmed")),
            "detail": f"Volume={layer3.get('volume_bucket')} candle={layer3.get('confirmation_candle')}",
        },
        {
            "rule": "NOT_MID_RANGE",
            "passed": context.get("location") != "Mid Range",
            "detail": f"Location={context.get('location')}",
        },
        {
            "rule": "NOT_DUPLICATE_BAR",
            "passed": "DUPLICATE_BAR" not in (layer5.get("reason_codes") or []),
            "detail": "No prior SELL emission on this bar",
        },
        {
            "rule": "LAYER5_PASS",
            "passed": bool(layer5.get("pass")),
            "detail": "All hard filters passed",
        },
    ]

    if layer5.get("pass") and evaluation.get("verdict") != "SELL":
        checks.append(
            {
                "rule": "LAYER4_EXECUTION",
                "passed": False,
                "detail": _describe_rule("LAYER4_EXECUTION_FAILED"),
            }
        )
    elif evaluation.get("verdict") == "SELL":
        checks.append(
            {
                "rule": "LAYER4_EXECUTION",
                "passed": True,
                "detail": "Structural stop/risk execution succeeded",
            }
        )

    return checks


def _decision_summary(
    *,
    engine_verdict: str,
    accepted: bool | None,
    throttle: ThrottleDecision | None,
    same_bar_conflict: bool,
    direction: str,
) -> tuple[str, str | None, str | None]:
    if same_bar_conflict and engine_verdict == direction:
        return "REJECTED", "SAME_BAR_CONFLICT", _describe_rule("SAME_BAR_CONFLICT")
    if engine_verdict != direction:
        if engine_verdict == "NO_TRADE":
            return "NO_TRADE", None, None
        return "NO_TRADE", None, None
    if throttle is not None and not throttle.accepted:
        code = throttle.rejection_reason or "REGIME_BLOCK"
        return "REJECTED", code, _describe_rule(code)
    if accepted is True:
        return "ACCEPTED", None, None
    if accepted is False:
        reason = throttle.rejection_reason if throttle else "REJECTED"
        return "REJECTED", reason, _describe_rule(str(reason).split(":")[0])
    return "NO_TRADE", None, None


def build_candle_report(
    *,
    candle: Candle,
    bar: int | None,
    buy_eval: dict[str, Any] | None,
    sell_eval: dict[str, Any] | None,
    eval_ms: float,
    context_snapshot: dict[str, Any],
    same_bar_conflict: bool = False,
    buy_throttle: ThrottleDecision | None = None,
    sell_throttle: ThrottleDecision | None = None,
    buy_accepted: bool | None = None,
    sell_accepted: bool | None = None,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    """Build a structured diagnostics report for one closed candle."""
    buy_context = (buy_eval or {}).get("context") or context_snapshot.get("buy_context") or {}
    sell_context = (sell_eval or {}).get("context") or context_snapshot.get("sell_context") or {}
    bar_events = set(context_snapshot.get("bar_events") or [])
    lookback_events = set(context_snapshot.get("lookback_events") or [])

    buy_layer1 = (buy_eval or {}).get("layer1") or {}
    sell_layer1 = (sell_eval or {}).get("layer1") or {}
    buy_layer5 = (buy_eval or {}).get("layer5") or {}
    sell_layer5 = (sell_eval or {}).get("layer5") or {}

    buy_checks = _buy_condition_checks(buy_eval or {}, bar=bar or -1) if buy_eval else []
    sell_checks = _sell_condition_checks(sell_eval or {}, bar=bar or -1) if sell_eval else []

    buy_decision, buy_reject_code, buy_reject_detail = _decision_summary(
        engine_verdict=str((buy_eval or {}).get("verdict", "NO_TRADE")),
        accepted=buy_accepted,
        throttle=buy_throttle,
        same_bar_conflict=same_bar_conflict,
        direction="BUY",
    )
    sell_decision, sell_reject_code, sell_reject_detail = _decision_summary(
        engine_verdict=str((sell_eval or {}).get("verdict", "NO_TRADE")),
        accepted=sell_accepted,
        throttle=sell_throttle,
        same_bar_conflict=same_bar_conflict,
        direction="SELL",
    )

    final_signal = "NO_TRADE"
    if skipped_reason:
        final_signal = "SKIPPED"
    elif same_bar_conflict and (buy_eval or {}).get("verdict") == "BUY" and (sell_eval or {}).get("verdict") == "SELL":
        final_signal = "CONFLICT"
    elif buy_decision == "ACCEPTED":
        final_signal = "BUY"
    elif sell_decision == "ACCEPTED":
        final_signal = "SELL"

    regime = (
        (buy_eval or {}).get("regime", {}).get("composite")
        or (sell_eval or {}).get("regime", {}).get("composite")
        or context_snapshot.get("regime_composite")
        or "N/A"
    )

    report: dict[str, Any] = {
        "timestamp": candle.timestamp.isoformat(),
        "ohlc": {
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
        },
        "volume": candle.volume,
        "trend": buy_context.get("htf_trend") or context_snapshot.get("trend_state") or "N/A",
        "market_regime": regime,
        "liquidity_sweep": _event_flag(
            bar_events | lookback_events,
            "Liquidity Grab",
            "Stop Hunt",
            "PDL Sweep",
            "PDH Sweep",
            "PWH Sweep",
            "PWL Sweep",
            "Equal High Sweep",
            "Equal Low Sweep",
        ),
        "bos": buy_context.get("bos") or _event_flag(bar_events, "BOS"),
        "choch": buy_context.get("choch") or _event_flag(bar_events, "CHOCH"),
        "order_block": _event_flag(bar_events | lookback_events, "Order Block"),
        "fair_value_gap": _event_flag(bar_events | lookback_events, "FVG"),
        "vwap_position": buy_context.get("vwap") or sell_context.get("vwap") or "N/A",
        "rsi": context_snapshot.get("rsi"),
        "rsi_bucket": buy_context.get("rsi") or sell_context.get("rsi"),
        "atr": context_snapshot.get("atr"),
        "ema20": context_snapshot.get("ema20"),
        "ema50": context_snapshot.get("ema50"),
        "ema200": context_snapshot.get("ema200"),
        "support_zone": context_snapshot.get("support_zone"),
        "resistance_zone": context_snapshot.get("resistance_zone"),
        "buy_score": _buy_formula_score(buy_layer1) if buy_eval else 0.0,
        "sell_score": _sell_gate_score(sell_eval) if sell_eval else 0.0,
        "buy_decision": buy_decision if buy_eval else "SKIPPED",
        "sell_decision": sell_decision if sell_eval else "SKIPPED",
        "buy_rejection_reason": buy_reject_detail,
        "buy_rejection_rule": buy_reject_code,
        "sell_rejection_reason": sell_reject_detail,
        "sell_rejection_rule": sell_reject_code,
        "buy_failed_conditions": [c for c in buy_checks if not c["passed"]],
        "buy_passed_conditions": [c for c in buy_checks if c["passed"]],
        "sell_failed_conditions": [c for c in sell_checks if not c["passed"]],
        "sell_passed_conditions": [c for c in sell_checks if c["passed"]],
        "buy_engine_reason_codes": buy_layer5.get("reason_codes") or [],
        "sell_engine_reason_codes": sell_layer5.get("reason_codes") or [],
        "final_signal": final_signal,
        "decision": final_signal,
        "reason_codes": collect_reason_codes(
            final_signal=final_signal,
            buy_engine_reason_codes=buy_layer5.get("reason_codes") or [],
            sell_engine_reason_codes=sell_layer5.get("reason_codes") or [],
            buy_rejection_rule=buy_reject_code,
            sell_rejection_rule=sell_reject_code,
            skipped_reason=skipped_reason,
            same_bar_conflict=same_bar_conflict,
        ),
        "eval_ms": round(eval_ms, 2),
        "skipped_reason": skipped_reason,
        "same_bar_conflict": same_bar_conflict,
        "bar_index": bar,
        "events_at_bar": sorted(bar_events),
        "lookback_events": sorted(lookback_events),
    }
    return report


def format_candle_report(report: dict[str, Any]) -> str:
    """Render a human-readable multi-section candle report."""
    ohlc = report["ohlc"]
    lines = [
        "",
        "=" * 72,
        f"CANDLE REPORT | {report['timestamp']}",
        "=" * 72,
        f"Timestamp          : {report['timestamp']}",
        f"OHLC               : O={_fmt(ohlc['open'])} H={_fmt(ohlc['high'])} L={_fmt(ohlc['low'])} C={_fmt(ohlc['close'])}",
        f"Volume             : {report['volume']}",
        f"Trend              : {report['trend']}",
        f"Market Regime      : {report['market_regime']}",
        f"Liquidity Sweep    : {report['liquidity_sweep']}",
        f"BOS                : {report['bos']}",
        f"CHOCH              : {report['choch']}",
        f"Order Block        : {report['order_block']}",
        f"Fair Value Gap     : {report['fair_value_gap']}",
        f"VWAP Position      : {report['vwap_position']}",
        f"RSI                : {_fmt(report['rsi'])} ({report['rsi_bucket']})",
        f"ATR                : {_fmt(report['atr'])}",
        f"EMA20              : {_fmt(report['ema20'])}",
        f"EMA50              : {_fmt(report['ema50'])}",
        f"EMA200             : {_fmt(report['ema200'])}",
        f"Support Zone       : {_fmt(report['support_zone'])}",
        f"Resistance Zone    : {_fmt(report['resistance_zone'])}",
        "-" * 72,
        f"Buy Score          : {report['buy_score']}",
        f"Sell Score         : {report['sell_score']}",
        f"Buy Decision       : {report['buy_decision']}",
        f"Sell Decision      : {report['sell_decision']}",
        f"Final Signal       : {report['final_signal']}",
        f"Eval Time (ms)     : {report['eval_ms']}",
    ]

    if report.get("skipped_reason"):
        lines.append(f"Skipped Reason     : {report['skipped_reason']}")

    lines.extend(["-" * 72, "BUY CONDITIONS"])
    if report["buy_decision"] == "SKIPPED":
        lines.append("  (evaluation skipped)")
    elif report["buy_decision"] == "ACCEPTED":
        for cond in report["buy_passed_conditions"]:
            lines.append(f"  [PASS] {cond['rule']}: {cond['detail']}")
    elif report["buy_decision"] == "NO_TRADE":
        for cond in report["buy_failed_conditions"]:
            lines.append(f"  [FAIL] {cond['rule']}: {cond['detail']}")
        if report["buy_engine_reason_codes"]:
            lines.append(f"  Engine reason codes: {', '.join(report['buy_engine_reason_codes'])}")
    else:
        for cond in report["buy_failed_conditions"]:
            lines.append(f"  [FAIL] {cond['rule']}: {cond['detail']}")
        if report["buy_engine_reason_codes"]:
            lines.append(f"  Engine reason codes: {', '.join(report['buy_engine_reason_codes'])}")
        if report["buy_rejection_rule"]:
            lines.append(f"  Rejection rule     : {report['buy_rejection_rule']}")
        if report["buy_rejection_reason"]:
            lines.append(f"  Rejection reason   : {report['buy_rejection_reason']}")

    lines.extend(["-" * 72, "SELL CONDITIONS"])
    if report["sell_decision"] == "SKIPPED":
        lines.append("  (evaluation skipped)")
    elif report["sell_decision"] == "ACCEPTED":
        for cond in report["sell_passed_conditions"]:
            lines.append(f"  [PASS] {cond['rule']}: {cond['detail']}")
    elif report["sell_decision"] == "NO_TRADE":
        for cond in report["sell_failed_conditions"]:
            lines.append(f"  [FAIL] {cond['rule']}: {cond['detail']}")
        if report["sell_engine_reason_codes"]:
            lines.append(f"  Engine reason codes: {', '.join(report['sell_engine_reason_codes'])}")
    else:
        for cond in report["sell_failed_conditions"]:
            lines.append(f"  [FAIL] {cond['rule']}: {cond['detail']}")
        if report["sell_engine_reason_codes"]:
            lines.append(f"  Engine reason codes: {', '.join(report['sell_engine_reason_codes'])}")
        if report["sell_rejection_rule"]:
            lines.append(f"  Rejection rule     : {report['sell_rejection_rule']}")
        if report["sell_rejection_reason"]:
            lines.append(f"  Rejection reason   : {report['sell_rejection_reason']}")

    lines.append("=" * 72)
    return "\n".join(lines)


def collect_reason_codes(
    *,
    final_signal: str,
    buy_engine_reason_codes: list[str],
    sell_engine_reason_codes: list[str],
    buy_rejection_rule: str | None = None,
    sell_rejection_rule: str | None = None,
    skipped_reason: str | None = None,
    same_bar_conflict: bool = False,
) -> list[str]:
    """Aggregate all rejection / skip reason codes for persistence."""
    codes: list[str] = []
    codes.extend(str(code) for code in buy_engine_reason_codes if code)
    codes.extend(str(code) for code in sell_engine_reason_codes if code)

    for rule in (buy_rejection_rule, sell_rejection_rule):
        if not rule:
            continue
        if rule.startswith("REGIME_BLOCK"):
            codes.append(rule if ":" in rule else "REGIME_BLOCK")
        elif rule not in codes:
            codes.append(rule)

    if skipped_reason:
        codes.append("INSUFFICIENT_BARS")
    if same_bar_conflict:
        codes.append("CONFLICT")
    if final_signal == "NO_TRADE" and not codes:
        codes.append("NO_SIGNAL")

    return sorted(dict.fromkeys(codes))


def decision_record_from_report(*, candle: Candle, symbol: str, report: dict[str, Any]) -> dict[str, Any]:
    """Map a candle diagnostics report to a signal_decisions insert payload."""
    ohlc = report["ohlc"]
    return {
        "timestamp": report["timestamp"],
        "symbol": symbol,
        "open": ohlc["open"],
        "high": ohlc["high"],
        "low": ohlc["low"],
        "close": ohlc["close"],
        "volume": report["volume"],
        "trend": report.get("trend"),
        "market_regime": report.get("market_regime"),
        "buy_score": report.get("buy_score"),
        "sell_score": report.get("sell_score"),
        "final_signal": report.get("final_signal"),
        "decision": report.get("decision") or report.get("final_signal"),
        "reason_codes": report.get("reason_codes") or [],
        "evaluation_time_ms": report.get("eval_ms", 0.0),
    }


def emit_decision_saved_log(*, decision: str, logger: Any) -> None:
    """Print the post-persistence decision summary block."""
    print("[DECISION SAVED]", flush=True)
    print(f"Decision = {decision}", flush=True)
    print("SQLite Insert = SUCCESS", flush=True)
    logger.info("[DECISION SAVED] Decision=%s SQLite Insert=SUCCESS", decision)


def emit_candle_report(report: dict[str, Any], *, logger: Any) -> None:
    """Print and log a structured candle diagnostics report."""
    text = format_candle_report(report)
    print(text, flush=True)
    logger.info("Candle diagnostics report\n%s", text)
    print(json.dumps({"candle_report": report}, default=str), flush=True)
