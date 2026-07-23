"""Forward-looking evaluation of BUY/SELL signals against subsequent candles."""

from __future__ import annotations

from src.trade_validation.config import TradeValidationConfig
from src.trade_validation.models import CandleBar, ExitReason, Outcome, SignalRecord, TradeValidationResult


def _target_stop_prices(
    *,
    direction: str,
    entry: float,
    target_pct: float,
    stop_pct: float,
) -> tuple[float, float]:
    if direction == "BUY":
        return entry * (1.0 + target_pct / 100.0), entry * (1.0 - stop_pct / 100.0)
    if direction == "SELL":
        return entry * (1.0 - target_pct / 100.0), entry * (1.0 + stop_pct / 100.0)
    raise ValueError(f"Unsupported direction: {direction}")


def _bar_hit(
    *,
    direction: str,
    bar: CandleBar,
    target_price: float,
    stop_price: float,
) -> ExitReason | None:
    """Return exit reason if target or stop touched; stop checked before target (conservative)."""
    if direction == "BUY":
        if bar.low <= stop_price:
            return "STOP_HIT"
        if bar.high >= target_price:
            return "TARGET_HIT"
        return None
    if direction == "SELL":
        if bar.high >= stop_price:
            return "STOP_HIT"
        if bar.low <= target_price:
            return "TARGET_HIT"
        return None
    raise ValueError(f"Unsupported direction: {direction}")


def _mfe_mae(*, direction: str, entry: float, bars: list[CandleBar]) -> tuple[float, float]:
    if not bars:
        return 0.0, 0.0
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    if direction == "BUY":
        return max(high - entry for high in highs), max(entry - low for low in lows)
    return max(entry - low for low in lows), max(high - entry for high in highs)


def _pnl_for_exit(*, direction: str, entry: float, exit_reason: ExitReason, target_price: float, stop_price: float) -> float | None:
    if exit_reason == "TARGET_HIT":
        return (target_price - entry) if direction == "BUY" else (entry - target_price)
    if exit_reason == "STOP_HIT":
        return (stop_price - entry) if direction == "BUY" else (entry - stop_price)
    return None


def evaluate_signal(
    signal: SignalRecord,
    forward_candles: list[CandleBar],
    *,
    config: TradeValidationConfig | None = None,
) -> TradeValidationResult:
    """
    Evaluate one signal against candles strictly after the signal bar.

    Does not read or mutate signal-engine state.
    """
    cfg = config or TradeValidationConfig()
    target_price, stop_price = _target_stop_prices(
        direction=signal.direction,
        entry=signal.entry,
        target_pct=cfg.target_pct,
        stop_pct=cfg.stop_pct,
    )

    window = forward_candles[: cfg.evaluation_window_bars]
    mfe, mae = _mfe_mae(direction=signal.direction, entry=signal.entry, bars=window)

    next_candle_close = window[0].close if len(window) >= 1 else None
    next_3_candle_close = window[2].close if len(window) >= 3 else None
    next_5_candle_close = window[4].close if len(window) >= 5 else None
    window_high = max((bar.high for bar in window), default=None)
    window_low = min((bar.low for bar in window), default=None)

    exit_reason: ExitReason = "OPEN"
    exit_timestamp: str | None = None
    holding_bars: int | None = None
    target_hit = False
    stop_hit = False

    for index, bar in enumerate(window, start=1):
        hit = _bar_hit(
            direction=signal.direction,
            bar=bar,
            target_price=target_price,
            stop_price=stop_price,
        )
        if hit is not None:
            exit_reason = hit
            exit_timestamp = bar.timestamp
            holding_bars = index
            target_hit = hit == "TARGET_HIT"
            stop_hit = hit == "STOP_HIT"
            break

    available = len(window)
    if exit_reason == "OPEN":
        if available >= cfg.evaluation_window_bars:
            exit_reason = "WINDOW_EXPIRED"
            exit_timestamp = window[-1].timestamp
            holding_bars = cfg.evaluation_window_bars
        else:
            exit_reason = "OPEN"

    outcome: Outcome
    if exit_reason == "TARGET_HIT":
        outcome = "WIN"
    elif exit_reason == "STOP_HIT":
        outcome = "LOSS"
    elif exit_reason == "WINDOW_EXPIRED":
        outcome = "EXPIRED"
    else:
        outcome = "OPEN"

    pnl = _pnl_for_exit(
        direction=signal.direction,
        entry=signal.entry,
        exit_reason=exit_reason if exit_reason in ("TARGET_HIT", "STOP_HIT") else "OPEN",
        target_price=target_price,
        stop_price=stop_price,
    )

    return TradeValidationResult(
        source_signal_id=signal.id,
        signal_timestamp=signal.timestamp,
        symbol=signal.symbol,
        direction=signal.direction,
        entry_price=signal.entry,
        signal_score=signal.signal_score,
        reason_codes=signal.reason_codes,
        next_candle_close=next_candle_close,
        next_3_candle_close=next_3_candle_close,
        next_5_candle_close=next_5_candle_close,
        window_high=window_high,
        window_low=window_low,
        mfe=round(mfe, 2),
        mae=round(mae, 2),
        target_price=round(target_price, 2),
        stop_price=round(stop_price, 2),
        target_pct=cfg.target_pct,
        stop_pct=cfg.stop_pct,
        target_hit=target_hit,
        stop_hit=stop_hit,
        pnl=round(pnl, 2) if pnl is not None else None,
        outcome=outcome,
        holding_bars=holding_bars,
        exit_reason=exit_reason,
        evaluation_window_bars=cfg.evaluation_window_bars,
        exit_timestamp=exit_timestamp,
    )
