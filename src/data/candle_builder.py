"""
Real-time 5-minute OHLCV candle builder from FYERS websocket ticks.

Buckets ticks into IST-aligned 5-minute bars (NSE cash session) and emits
closed candles when a bucket rolls forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timedelta
import time as time_module
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.core.logger import logger

IST = ZoneInfo("Asia/Kolkata")
BAR_MINUTES = 5
SESSION_OPEN = dt_time(9, 15)
SESSION_CLOSE = dt_time(15, 30)


@dataclass(frozen=True)
class Tick:
    """Normalized live tick."""

    symbol: str
    price: float
    timestamp: datetime
    volume: float = 0.0


@dataclass(frozen=True)
class Candle:
    """Closed 5-minute OHLCV candle."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int

    def as_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class CandleBuilderError(Exception):
    """Raised when tick parsing or candle construction fails."""


def _floor_to_bar(ts: datetime) -> datetime:
    localized = ts.astimezone(IST)
    minute_bucket = (localized.minute // BAR_MINUTES) * BAR_MINUTES
    return localized.replace(minute=minute_bucket, second=0, microsecond=0)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)
    if isinstance(value, (int, float)):
        # FYERS may send epoch seconds.
        return datetime.fromtimestamp(float(value), tz=IST)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.astimezone(IST) if parsed.tzinfo else parsed.replace(tzinfo=IST)
    except ValueError:
        return None


def parse_fyers_tick(message: Any, *, default_symbol: str = "NSE:NIFTY50-INDEX") -> Tick | None:
    """Parse a FYERS websocket message into a normalized tick."""
    if not isinstance(message, dict):
        return None

    symbol = str(message.get("symbol") or message.get("sym") or default_symbol).strip().upper()
    price_raw = message.get("ltp") or message.get("last_price") or message.get("lp")
    if price_raw is None:
        return None
    try:
        price = float(price_raw)
    except (TypeError, ValueError):
        return None

    ts = _parse_timestamp(
        message.get("exch_feed_time")
        or message.get("timestamp")
        or message.get("tt")
        or message.get("last_traded_time")
    )
    if ts is None:
        ts = datetime.now(tz=IST)

    volume_raw = message.get("volume") or message.get("vol") or 0
    try:
        volume = float(volume_raw)
    except (TypeError, ValueError):
        volume = 0.0

    return Tick(symbol=symbol, price=price, timestamp=ts, volume=volume)


def is_trading_session(ts: datetime) -> bool:
    """Return True when timestamp is inside the NSE cash session window."""
    local = ts.astimezone(IST)
    if local.weekday() >= 5:
        return False
    clock = local.time()
    return SESSION_OPEN <= clock <= SESSION_CLOSE


class FiveMinuteCandleBuilder:
    """
    Aggregate ticks into IST 5-minute candles.

  Parameters
    ----------
    on_candle_close : Callable[[Candle], None] | None
        Callback invoked when a candle bucket closes.
    """

    def __init__(
        self,
        *,
        symbol: str = "NSE:NIFTY50-INDEX",
        on_candle_close: Callable[[Candle], None] | None = None,
    ) -> None:
        self.symbol = symbol.strip().upper()
        self.on_candle_close = on_candle_close
        self._active_bucket: datetime | None = None
        self._open: float | None = None
        self._high: float | None = None
        self._low: float | None = None
        self._close: float | None = None
        self._volume: float = 0.0
        self._tick_count: int = 0
        self.closed_candles: list[Candle] = []
        self.last_candle_close_ms: float = 0.0
        # Last closed-bar timestamp known committed (watermark); buckets at/before this
        # are discarded and never emitted via on_candle_close.
        self._committed_through: datetime | None = None

    def sync_after_committed(self, committed_ts: datetime | None) -> None:
        """
        Align builder state with the last committed closed-candle timestamp.

        Called after recovery (or equivalent) so an active bucket for an already
        recovered bar is discarded without emitting ``on_candle_close``.
        """
        if committed_ts is None:
            return
        committed = committed_ts.astimezone(IST) if committed_ts.tzinfo else committed_ts.replace(tzinfo=IST)
        committed = committed.replace(second=0, microsecond=0)
        self._committed_through = committed
        if self._active_bucket is not None and self._active_bucket <= committed:
            logger.info(
                "Discarding active bucket %s after sync to committed %s",
                self._active_bucket.isoformat(),
                committed.isoformat(),
            )
            self._reset_active()

    def _is_at_or_before_committed(self, bucket: datetime) -> bool:
        if self._committed_through is None:
            return False
        local = bucket.astimezone(IST) if bucket.tzinfo else bucket.replace(tzinfo=IST)
        return local.replace(second=0, microsecond=0) <= self._committed_through

    def ingest_tick(self, tick: Tick) -> Candle | None:
        """Ingest a tick; return a closed candle when the bucket rolls."""
        if tick.symbol != self.symbol:
            logger.debug("Ignoring tick for symbol=%s (expected %s)", tick.symbol, self.symbol)
            return None

        bucket = _floor_to_bar(tick.timestamp)
        closed: Candle | None = None

        if self._active_bucket is None:
            if self._is_at_or_before_committed(bucket):
                logger.debug(
                    "Ignoring tick in already-committed bucket %s",
                    bucket.isoformat(),
                )
                return None
            self._start_bucket(bucket, tick)
            return None

        if bucket != self._active_bucket:
            close_started = time_module.perf_counter()
            closed = self._close_active_bucket()
            self.last_candle_close_ms = (time_module.perf_counter() - close_started) * 1000.0
            if not self._is_at_or_before_committed(bucket):
                self._start_bucket(bucket, tick)
            else:
                logger.debug(
                    "Not starting already-committed bucket %s",
                    bucket.isoformat(),
                )
        else:
            self._update_bucket(tick)

        closed = self._emit_closed_if_allowed(closed)
        return closed

    def ingest_message(self, message: Any) -> Candle | None:
        tick = parse_fyers_tick(message, default_symbol=self.symbol)
        if tick is None:
            return None
        return self.ingest_tick(tick)

    def flush(self) -> Candle | None:
        """Force-close the active bucket (session end / shutdown)."""
        if self._active_bucket is None:
            return None
        closed = self._close_active_bucket()
        return self._emit_closed_if_allowed(closed)

    def _emit_closed_if_allowed(self, closed: Candle | None) -> Candle | None:
        """Invoke on_candle_close only for candles not already committed."""
        if closed is None:
            return None
        if self._is_at_or_before_committed(closed.timestamp):
            logger.info(
                "Suppressing emit of already-committed candle %s",
                closed.timestamp.isoformat(),
            )
            if self.closed_candles and self.closed_candles[-1].timestamp == closed.timestamp:
                self.closed_candles.pop()
            return None
        if self.on_candle_close is not None:
            self.on_candle_close(closed)
        return closed

    def _start_bucket(self, bucket: datetime, tick: Tick) -> None:
        self._active_bucket = bucket
        self._open = tick.price
        self._high = tick.price
        self._low = tick.price
        self._close = tick.price
        self._volume = tick.volume
        self._tick_count = 1

    def _update_bucket(self, tick: Tick) -> None:
        assert self._high is not None and self._low is not None and self._close is not None
        self._high = max(self._high, tick.price)
        self._low = min(self._low, tick.price)
        self._close = tick.price
        self._volume += tick.volume
        self._tick_count += 1

    def _close_active_bucket(self) -> Candle | None:
        if self._active_bucket is None or self._open is None:
            self._reset_active()
            return None

        candle = Candle(
            symbol=self.symbol,
            timestamp=self._active_bucket,
            open=round(self._open, 2),
            high=round(self._high or self._open, 2),
            low=round(self._low or self._open, 2),
            close=round(self._close or self._open, 2),
            volume=round(self._volume, 2),
            tick_count=self._tick_count,
        )
        self.closed_candles.append(candle)
        logger.info(
            "Candle closed %s O=%.2f H=%.2f L=%.2f C=%.2f ticks=%s",
            candle.timestamp.isoformat(),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.tick_count,
        )
        self._reset_active()
        return candle

    def _reset_active(self) -> None:
        self._active_bucket = None
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._volume = 0.0
        self._tick_count = 0


def candles_to_frame_rows(candles: list[Candle]) -> list[dict[str, Any]]:
    """Convert candles to pipeline-compatible OHLCV rows."""
    rows: list[dict[str, Any]] = []
    for candle in candles:
        rows.append(
            {
                "Date": candle.timestamp.strftime("%Y-%m-%d %H:%M:%S%z"),
                "Open": candle.open,
                "High": candle.high,
                "Low": candle.low,
                "Close": candle.close,
                "Volume": candle.volume,
            }
        )
    return rows
