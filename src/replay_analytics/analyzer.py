"""Pure analytics transforms over replay decision rows."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Any

# Gate / rule codes commonly emitted into signal_decisions.reason_codes.
KNOWN_RULE_CODES: tuple[str, ...] = (
    "FORMULA_INCOMPLETE",
    "MISSING_PDL_SWEEP",
    "MISSING_LIQUIDITY_GRAB",
    "LOCATION_MISMATCH",
    "LOCATION_MID_RANGE",
    "DIRECTION_NOT_ALIGNED",
    "VOLUME_FAILED",
    "VWAP_MISMATCH",
    "HTF_CONFLICT",
    "EMA_MISMATCH",
    "NO_EARLY_WARNING",
    "NO_FAILED_BREAKOUT",
    "CONFIRMATION_FAILED",
    "DUPLICATE_BAR",
    "SAME_BAR_CONFLICT",
    "REGIME_BLOCK",
    "INSUFFICIENT_BARS",
    "NO_SIGNAL",
    "LAYER4_EXECUTION_FAILED",
)


def _parse_ts(value: str) -> datetime | None:
    text = (value or "").strip().replace("T", " ")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 2)


def _safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 2) if values else None


def _safe_max(values: list[float]) -> float | None:
    return round(max(values), 2) if values else None


def _infer_vwap_state(reason_codes: list[str]) -> str:
    if "VWAP_MISMATCH" in reason_codes:
        return "Fail"
    if "INSUFFICIENT_BARS" in reason_codes:
        return "Unknown"
    return "Pass"


def _distribution(counter: Counter[str]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {"label": key, "count": count, "pct": _pct(count, total)}
        for key, count in counter.most_common()
    ]


def analyze_replay(
    *,
    decisions: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    candle_count: int | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """
    Build the full analytics payload from persisted replay rows.

    Does not call BUY_V3 / SELL_V6 or the Replay Engine.
    """
    total_decisions = len(decisions)
    decision_counts = Counter(str(row.get("decision") or row.get("final_signal") or "UNKNOWN") for row in decisions)
    buy_n = decision_counts.get("BUY", 0)
    sell_n = decision_counts.get("SELL", 0)
    no_trade_n = decision_counts.get("NO_TRADE", 0)
    conflict_n = decision_counts.get("CONFLICT", 0)
    skipped_n = decision_counts.get("SKIPPED", 0)

    buy_scores = [float(row["buy_score"]) for row in decisions if row.get("buy_score") is not None]
    sell_scores = [float(row["sell_score"]) for row in decisions if row.get("sell_score") is not None]

    rejection_counter: Counter[str] = Counter()
    passed_counter: Counter[str] = Counter()
    rule_counter: Counter[str] = Counter()

    trend_counter: Counter[str] = Counter()
    regime_counter: Counter[str] = Counter()
    vwap_counter: Counter[str] = Counter()
    htf_counter: Counter[str] = Counter()

    daily: dict[str, dict[str, Any]] = {}
    monthly: dict[str, dict[str, Any]] = {}

    timestamps: list[datetime] = []

    for row in decisions:
        codes = [str(code) for code in (row.get("reason_codes") or [])]
        for code in codes:
            rejection_counter[code] += 1
            rule_counter[code] += 1

        code_set = set(codes)
        for rule in KNOWN_RULE_CODES:
            if rule not in code_set and not any(c.startswith(f"{rule}:") for c in code_set):
                # Treat absence of a known rejection code as a pass observation.
                if rule.startswith("MISSING_") or rule in {
                    "FORMULA_INCOMPLETE",
                    "LOCATION_MISMATCH",
                    "LOCATION_MID_RANGE",
                    "DIRECTION_NOT_ALIGNED",
                    "VOLUME_FAILED",
                    "VWAP_MISMATCH",
                    "HTF_CONFLICT",
                    "EMA_MISMATCH",
                    "NO_EARLY_WARNING",
                    "NO_FAILED_BREAKOUT",
                    "CONFIRMATION_FAILED",
                    "DUPLICATE_BAR",
                    "SAME_BAR_CONFLICT",
                    "REGIME_BLOCK",
                    "INSUFFICIENT_BARS",
                    "NO_SIGNAL",
                    "LAYER4_EXECUTION_FAILED",
                }:
                    passed_counter[rule] += 1

        trend = str(row.get("trend") or "Unknown")
        regime = str(row.get("market_regime") or "Unknown")
        vwap = _infer_vwap_state(codes)
        trend_counter[trend] += 1
        regime_counter[regime] += 1
        vwap_counter[vwap] += 1
        htf_counter[trend] += 1  # persisted trend column is HTF trend snapshot

        ts = _parse_ts(str(row.get("timestamp") or ""))
        if ts is not None:
            timestamps.append(ts)
            day_key = ts.date().isoformat()
            month_key = f"{ts.year:04d}-{ts.month:02d}"
            for bucket_key, bucket in ((day_key, daily), (month_key, monthly)):
                entry = bucket.setdefault(
                    bucket_key,
                    {
                        "period": bucket_key,
                        "candles": 0,
                        "decisions": 0,
                        "BUY": 0,
                        "SELL": 0,
                        "NO_TRADE": 0,
                        "buy_scores": [],
                        "sell_scores": [],
                    },
                )
                entry["decisions"] += 1
                entry["candles"] += 1
                decision = str(row.get("decision") or "UNKNOWN")
                if decision in ("BUY", "SELL", "NO_TRADE"):
                    entry[decision] += 1
                if row.get("buy_score") is not None:
                    entry["buy_scores"].append(float(row["buy_score"]))
                if row.get("sell_score") is not None:
                    entry["sell_scores"].append(float(row["sell_score"]))

    def _finalize_period(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in sorted(entries):
            entry = entries[key]
            buy_scores_p = entry.pop("buy_scores")
            sell_scores_p = entry.pop("sell_scores")
            decisions_n = entry["decisions"]
            out.append(
                {
                    **entry,
                    "BUY_pct": _pct(entry["BUY"], decisions_n),
                    "SELL_pct": _pct(entry["SELL"], decisions_n),
                    "NO_TRADE_pct": _pct(entry["NO_TRADE"], decisions_n),
                    "avg_buy_score": _safe_mean(buy_scores_p),
                    "avg_sell_score": _safe_mean(sell_scores_p),
                    "max_buy_score": _safe_max(buy_scores_p),
                    "max_sell_score": _safe_max(sell_scores_p),
                },
            )
        return out

    signal_buy = sum(1 for s in signals if str(s.get("direction")) == "BUY")
    signal_sell = sum(1 for s in signals if str(s.get("direction")) == "SELL")
    signal_accepted = sum(1 for s in signals if int(s.get("accepted") or 0) == 1)

    window_start = min(timestamps).isoformat() if timestamps else None
    window_end = max(timestamps).isoformat() if timestamps else None

    candles = candle_count if candle_count is not None else total_decisions

    report: dict[str, Any] = {
        "meta": {
            "db_path": db_path,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_tables": ["signal_decisions", "signals", "candles"],
        },
        "replay_summary": {
            "replay_window_start": window_start,
            "replay_window_end": window_end,
            "total_candles": candles,
            "total_decisions": total_decisions,
            "buy_signals": buy_n,
            "sell_signals": sell_n,
            "no_trade_count": no_trade_n,
            "conflict_count": conflict_n,
            "skipped_count": skipped_n,
            "persisted_signal_rows": len(signals),
            "persisted_buy_signal_rows": signal_buy,
            "persisted_sell_signal_rows": signal_sell,
            "persisted_accepted_signals": signal_accepted,
        },
        "decision_statistics": {
            "BUY": buy_n,
            "SELL": sell_n,
            "NO_TRADE": no_trade_n,
            "BUY_pct": _pct(buy_n, total_decisions),
            "SELL_pct": _pct(sell_n, total_decisions),
            "NO_TRADE_pct": _pct(no_trade_n, total_decisions),
            "other": {
                "CONFLICT": conflict_n,
                "SKIPPED": skipped_n,
            },
        },
        "score_statistics": {
            "average_buy_score": _safe_mean(buy_scores),
            "average_sell_score": _safe_mean(sell_scores),
            "maximum_buy_score": _safe_max(buy_scores),
            "maximum_sell_score": _safe_max(sell_scores),
            "buy_score_samples": len(buy_scores),
            "sell_score_samples": len(sell_scores),
        },
        "rule_statistics": {
            "top_rejection_reasons": [
                {"code": code, "count": count, "pct": _pct(count, total_decisions)}
                for code, count in rejection_counter.most_common(25)
            ],
            "top_passed_rules": [
                {"code": code, "count": count, "pct": _pct(count, total_decisions)}
                for code, count in passed_counter.most_common(25)
            ],
            "rule_frequency": [
                {"code": code, "rejection_count": count, "pct": _pct(count, total_decisions)}
                for code, count in rule_counter.most_common()
            ],
        },
        "market_statistics": {
            "trend_distribution": _distribution(trend_counter),
            "regime_distribution": _distribution(regime_counter),
            "vwap_distribution": _distribution(vwap_counter),
            "htf_distribution": _distribution(htf_counter),
            "vwap_note": "VWAP Pass/Fail inferred from absence/presence of VWAP_MISMATCH in reason_codes.",
        },
        "daily_summary": _finalize_period(daily),
        "monthly_summary": _finalize_period(monthly),
    }
    return report
