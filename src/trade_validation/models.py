"""Data models for trade validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Outcome = Literal["WIN", "LOSS", "OPEN", "EXPIRED"]
ExitReason = Literal["TARGET_HIT", "STOP_HIT", "WINDOW_EXPIRED", "OPEN"]


@dataclass(frozen=True)
class CandleBar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class SignalRecord:
    """Read-only view of a persisted BUY/SELL signal."""

    id: int
    timestamp: str
    direction: str
    entry: float
    engine_version: str
    accepted: bool
    symbol: str
    signal_score: float | None
    reason_codes: tuple[str, ...]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class TradeValidationResult:
    source_signal_id: int
    signal_timestamp: str
    symbol: str
    direction: str
    entry_price: float
    signal_score: float | None
    reason_codes: tuple[str, ...]
    next_candle_close: float | None
    next_3_candle_close: float | None
    next_5_candle_close: float | None
    window_high: float | None
    window_low: float | None
    mfe: float
    mae: float
    target_price: float
    stop_price: float
    target_pct: float
    stop_pct: float
    target_hit: bool
    stop_hit: bool
    pnl: float | None
    outcome: Outcome
    holding_bars: int | None
    exit_reason: ExitReason
    evaluation_window_bars: int
    exit_timestamp: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_signal_id": self.source_signal_id,
            "signal_timestamp": self.signal_timestamp,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "signal_score": self.signal_score,
            "reason_codes": list(self.reason_codes),
            "next_candle_close": self.next_candle_close,
            "next_3_candle_close": self.next_3_candle_close,
            "next_5_candle_close": self.next_5_candle_close,
            "window_high": self.window_high,
            "window_low": self.window_low,
            "mfe": self.mfe,
            "mae": self.mae,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "target_pct": self.target_pct,
            "stop_pct": self.stop_pct,
            "target_hit": self.target_hit,
            "stop_hit": self.stop_hit,
            "pnl": self.pnl,
            "outcome": self.outcome,
            "holding_bars": self.holding_bars,
            "exit_reason": self.exit_reason,
            "evaluation_window_bars": self.evaluation_window_bars,
            "exit_timestamp": self.exit_timestamp,
        }
