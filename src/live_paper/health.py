"""
Connection health monitoring for the live paper websocket feed.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, time as dtime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from src.live_paper.logging_setup import get_logger
from src.live_paper.metrics import LiveMetrics

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
BAR_MINUTES = 5


def is_ist_session(now: datetime | None = None) -> bool:
    """Return True during NSE cash session (weekdays 09:15-15:30 IST)."""
    local = (now or datetime.now(tz=IST)).astimezone(IST)
    if local.weekday() >= 5:
        return False
    clock = local.time()
    return SESSION_OPEN <= clock <= SESSION_CLOSE


def market_status_label(now: datetime | None = None) -> str:
    local = (now or datetime.now(tz=IST)).astimezone(IST)
    if local.weekday() >= 5:
        return "weekend"
    clock = local.time()
    if clock < SESSION_OPEN:
        return "pre_open"
    if clock > SESSION_CLOSE:
        return "closed"
    return "open"


def detect_missed_candles(last_ts: datetime | None, now: datetime | None = None) -> int:
    """
    Count expected closed 5-minute bars between ``last_ts`` and ``now`` inside session.

    Returns 0 when ``last_ts`` is missing or the gap is at most one bar.
    """
    if last_ts is None:
        return 0
    end = (now or datetime.now(tz=IST)).astimezone(IST)
    start = last_ts.astimezone(IST)
    if end <= start:
        return 0

    expected = 0
    cursor = start + timedelta(minutes=BAR_MINUTES)
    # Align cursor to bar floor
    minute_bucket = (cursor.minute // BAR_MINUTES) * BAR_MINUTES
    cursor = cursor.replace(minute=minute_bucket, second=0, microsecond=0)
    while cursor <= end:
        if is_ist_session(cursor):
            expected += 1
        cursor += timedelta(minutes=BAR_MINUTES)
    # One bar is the current/expected next close — only report extras as missed
    return max(expected - 1, 0)


class ConnectionHealthMonitor:
    """Track websocket freshness, reconnects, and heartbeat metrics."""

    def __init__(
        self,
        metrics: LiveMetrics,
        *,
        stale_tick_seconds: float = 30.0,
        heartbeat_seconds: float = 10.0,
        on_stale: Callable[[], None] | None = None,
    ) -> None:
        self.metrics = metrics
        self.stale_tick_seconds = float(stale_tick_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.on_stale = on_stale
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log = get_logger("websocket")
        self._reconnect_log = get_logger("reconnect")
        self._errors = get_logger("errors")
        self._stale_notified = False

    def record_tick(self) -> None:
        self.metrics.record_tick()
        self._stale_notified = False

    def record_reconnect(self, reason: str) -> None:
        self.metrics.record_reconnect(reason)
        self._reconnect_log.warning("Reconnect: %s (count=%s)", reason, self.metrics.reconnect_count)

    def record_open(self) -> None:
        self.metrics.set_ws_status("connected")
        self._log.info("Websocket open")

    def record_close(self, reason: str = "") -> None:
        self.metrics.set_ws_status("closed")
        self._log.warning("Websocket closed: %s", reason)
        self.record_reconnect(reason or "ws_close")

    def record_error(self, message: str) -> None:
        self.metrics.record_error(message)
        self.metrics.set_ws_status("error")
        self._errors.error("%s", message)
        self._log.error("%s", message)

    def is_stale(self, *, now: float | None = None) -> bool:
        """True when inside IST session and no tick arrived within stale window."""
        if not is_ist_session():
            return False
        last = self.metrics.last_tick_at
        if last is None:
            return True
        clock = now if now is not None else time.time()
        return (clock - last) >= self.stale_tick_seconds

    def start_heartbeat(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="live-paper-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.metrics.record_heartbeat()
            status = market_status_label()
            self.metrics.set_system(market_status=status)
            if self.is_stale():
                msg = f"Stale ticks during session (>{self.stale_tick_seconds}s)"
                self.metrics.record_error(msg)
                self._log.warning("%s", msg)
                if not self._stale_notified:
                    self._stale_notified = True
                    if self.on_stale is not None:
                        try:
                            self.on_stale()
                        except Exception as exc:  # noqa: BLE001
                            self.record_error(f"on_stale failed: {exc}")
            else:
                if status == "open":
                    self.metrics.set_ws_status("connected" if self.metrics.last_tick_at else self.metrics.ws_status)
