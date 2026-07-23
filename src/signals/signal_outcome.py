"""
Post-signal forward outcome evaluation.

Separates signal *decision* (realtime Layer1–5 + Layer4 plan) from
*outcome validation* (requires FORWARD_BARS of future candles).

``_trade_outcome`` must never block BUY/SELL emission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from src.research.liquidity_move_reconstruction_research import FORWARD_BARS

IST = ZoneInfo("Asia/Kolkata")


class OutcomeCapable(Protocol):
    """Engine exposing the shared forward-outcome calculator."""

    def _trade_outcome(self, frame: Any, bar: int, direction: str) -> dict[str, Any]: ...


def normalize_timestamp_key(value: Any) -> str | None:
    """
    Canonical timezone-aware key for signal/bar identity.

    Accepts DB strings (``+0530``), ISO strings (``+05:30``), datetime, and
    pandas Timestamps. Returns a stable IST second-resolution key, or None
    if the value cannot be parsed.
    """
    if value is None:
        return None
    text: str | None = None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        ts = None
    if ts is None or pd.isna(ts):
        text = str(value).strip()
        if not text:
            return None
        # fromisoformat / Timestamp may reject +0530; normalize offset colon.
        if (
            len(text) >= 5
            and text[-5] in "+-"
            and text[-3] != ":"
            and text[-4:].replace(":", "").isdigit()
        ):
            text = f"{text[:-2]}:{text[-2:]}"
        try:
            ts = pd.Timestamp(text)
        except (TypeError, ValueError):
            return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST)
    else:
        ts = ts.tz_convert(IST)
    ts = ts.floor("s")
    return ts.isoformat()


def timestamps_equivalent(left: Any, right: Any) -> bool:
    """True when both values denote the same IST instant."""
    a = normalize_timestamp_key(left)
    b = normalize_timestamp_key(right)
    return a is not None and a == b


@dataclass(frozen=True)
class SignalOutcomeUpdate:
    """Fields written back onto a previously stored signal."""

    decision_timestamp: str
    direction: str
    outcome_timestamp: str
    entry: float
    stop: float
    target: float | None
    risk: float
    reward: float
    outcome: str
    holding_bars: int
    forward_outcome: dict[str, Any]


def classify_trade_outcome(outcome: dict[str, Any]) -> str:
    """Map ``_trade_outcome`` realized PnL to WIN / LOSS / BREAKEVEN."""
    if not outcome:
        return "INCOMPLETE"
    realized = float(outcome.get("realized_pnl_points") or 0.0)
    if realized > 0:
        return "WIN"
    if realized < 0:
        return "LOSS"
    return "BREAKEVEN"


def evaluate_post_signal_outcome(
    engine: OutcomeCapable,
    *,
    frame: Any,
    signal_bar: int,
    direction: str,
    decision_timestamp: str,
    outcome_timestamp: str,
    forward_bars: int = FORWARD_BARS,
) -> SignalOutcomeUpdate | None:
    """
    Run ``_trade_outcome`` after the forward window has elapsed.

    Returns None when forward data is still insufficient (should not happen
    once ``holding_bars >= forward_bars`` is guaranteed by the caller).
    """
    side = "bullish" if direction.upper() == "BUY" else "bearish"
    raw = engine._trade_outcome(frame, signal_bar, side)
    if not raw:
        return None
    risk = float(raw.get("risk_points") or 0.0)
    realized = float(raw.get("realized_pnl_points") or 0.0)
    return SignalOutcomeUpdate(
        decision_timestamp=decision_timestamp,
        direction=direction.upper(),
        outcome_timestamp=outcome_timestamp,
        entry=float(raw["entry"]),
        stop=float(raw["stop_loss"]),
        target=raw.get("target"),
        risk=risk,
        reward=round(realized, 2),
        outcome=classify_trade_outcome(raw),
        holding_bars=int(forward_bars),
        forward_outcome=dict(raw),
    )


def build_realtime_layer4_plan(
    *,
    model_id: str,
    direction: str,
    entry: float,
    stop_loss: float,
    risk_points: float,
    liquidity_target: float | None,
    signal_reason_stack: dict[str, Any],
    forward_outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct Layer4 execution plan from realtime structure only."""
    entry_f = float(entry)
    risk = float(risk_points)
    if direction.upper() == "BUY":
        t1 = round(entry_f + risk, 2)
        t2 = round(entry_f + 2 * risk, 2)
        t3 = round(entry_f + 3 * risk, 2)
    else:
        t1 = round(entry_f - risk, 2)
        t2 = round(entry_f - 2 * risk, 2)
        t3 = round(entry_f - 3 * risk, 2)
    return {
        "model_id": model_id,
        "direction": direction.upper(),
        "entry": entry_f,
        "stop_loss": float(stop_loss),
        "target_1": t1,
        "target_2": t2,
        "target_3": t3,
        "liquidity_target": liquidity_target,
        "risk_points": round(risk, 2),
        "signal_reason_stack": signal_reason_stack,
        "forward_outcome": forward_outcome,
        "outcome_pending": forward_outcome is None,
    }
