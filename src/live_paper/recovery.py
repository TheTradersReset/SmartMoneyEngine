"""
Missed-candle recovery via FYERS historical REST (gap windows only).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from src.data.candle_builder import Candle, is_trading_session
from src.live_paper.health import BAR_MINUTES, detect_missed_candles
from src.live_paper.logging_setup import get_logger
from src.live_paper.metrics import LiveMetrics

IST = ZoneInfo("Asia/Kolkata")


class _PipelineLike(Protocol):
    symbol: str
    db: Any

    def ingest_closed_candle(self, candle: Candle) -> None: ...


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(IST) if parsed.tzinfo else parsed.replace(tzinfo=IST)


def _existing_candle_timestamps(db_path: Any, symbol: str) -> set[str]:
    path = str(db_path)
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT timestamp FROM candles WHERE symbol = ?",
            (symbol,),
        ).fetchall()
        return {str(row[0]) for row in rows}
    finally:
        conn.close()


def _existing_accepted_signal_keys(db_path: Any) -> set[tuple[str, str]]:
    path = str(db_path)
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT timestamp, direction FROM signals WHERE accepted = 1"
        ).fetchall()
        return {(str(row[0]), str(row[1]).upper()) for row in rows}
    finally:
        conn.close()


def _history_to_candles(response: dict[str, Any], symbol: str) -> list[Candle]:
    raw = response.get("candles") or []
    candles: list[Candle] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        ts = datetime.fromtimestamp(float(row[0]), tz=IST)
        candles.append(
            Candle(
                symbol=symbol,
                timestamp=ts,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                tick_count=0,
            )
        )
    candles.sort(key=lambda c: c.timestamp)
    return candles


class MissedCandleRecovery:
    """Fetch and feed missed 5-minute bars after reconnect / stale detection."""

    def __init__(
        self,
        pipeline: _PipelineLike,
        metrics: LiveMetrics,
        *,
        client_factory: Any | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.metrics = metrics
        self._client_factory = client_factory
        self._log = get_logger("candle")
        self._reconnect_log = get_logger("reconnect")
        # Single-flight: heartbeat (stale) and WS reconnect share this gate.
        self._lock = threading.Lock()

    def _make_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        from src.brokers.fyers.client import FyersClient

        return FyersClient.from_token_file()

    def maybe_recover(self, *, last_candle_ts: datetime | str | None, now: datetime | None = None) -> int:
        """
        If the gap exceeds one 5-minute bar, fetch FYERS history for the window and ingest.

        Skips candles already present in SQLite and bars that already have accepted signals.
        Returns the number of candles fed into the pipeline.

        Only one recovery runs at a time. A concurrent trigger is skipped and logged.
        """
        if not self._lock.acquire(blocking=False):
            self._reconnect_log.info(
                "Recovery skipped: already in progress (concurrent trigger)"
            )
            return 0

        try:
            end = (now or datetime.now(tz=IST)).astimezone(IST)
            start = _parse_ts(last_candle_ts)
            if start is None:
                self._log.info("Recovery skipped: no last candle timestamp")
                return 0

            missed = detect_missed_candles(start, end)
            self.metrics.set_system(missed_candles_count=missed)
            if missed < 1:
                self._log.info("No missed candles to recover (last=%s)", start.isoformat())
                return 0

            self._reconnect_log.info(
                "Missed candle recovery starting last=%s now=%s missed=%s",
                start.isoformat(),
                end.isoformat(),
                missed,
            )
            return self.recover_window(start + timedelta(minutes=BAR_MINUTES), end)
        finally:
            self._lock.release()

    def recover_window(self, range_from: datetime, range_to: datetime) -> int:
        symbol = self.pipeline.symbol
        db_path = self.pipeline.db.db_path
        existing_ts = _existing_candle_timestamps(db_path, symbol)
        accepted = _existing_accepted_signal_keys(db_path)

        try:
            client = self._make_client()
            response = client.get_history(
                symbol=symbol,
                resolution="5",
                date_from=range_from.astimezone(IST).strftime("%Y-%m-%d"),
                date_to=range_to.astimezone(IST).strftime("%Y-%m-%d"),
                date_format=1,
            )
        except Exception as exc:  # noqa: BLE001
            self._log.error("History fetch failed during recovery: %s", exc)
            self.metrics.record_error(f"recovery_fetch_failed: {exc}")
            return 0

        candles = _history_to_candles(response if isinstance(response, dict) else {}, symbol)
        fed = 0
        for candle in candles:
            if candle.timestamp < range_from.astimezone(IST) or candle.timestamp > range_to.astimezone(IST):
                continue
            if not is_trading_session(candle.timestamp):
                continue
            iso = candle.timestamp.isoformat()
            # Match common DB formats
            candidates = {
                iso,
                candle.timestamp.strftime("%Y-%m-%d %H:%M:%S%z"),
                candle.timestamp.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            if candidates & existing_ts:
                self._log.info("Skip duplicate candle %s", iso)
                continue
            # Skip bars that already produced accepted signals
            signal_hit = False
            for ts, _direction in accepted:
                parsed = _parse_ts(ts)
                if ts in candidates or (parsed is not None and parsed == candle.timestamp):
                    signal_hit = True
                    break
            if signal_hit:
                self._log.info("Skip candle with existing accepted signal %s", iso)
                continue

            try:
                self.pipeline.ingest_closed_candle(candle)
                fed += 1
                existing_ts.add(iso)
                self._log.info("Recovered candle %s O=%.2f H=%.2f L=%.2f C=%.2f", iso, candle.open, candle.high, candle.low, candle.close)
            except Exception as exc:  # noqa: BLE001
                self._log.error("Failed to ingest recovered candle %s: %s", iso, exc)
                self.metrics.record_error(f"recovery_ingest_failed: {exc}")

        self.metrics.set_system(missed_candles_count=max(0, self.metrics.missed_candles_count - fed))
        self._reconnect_log.info("Missed candle recovery complete fed=%s", fed)
        return fed
