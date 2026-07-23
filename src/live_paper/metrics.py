"""
Thread-safe live metrics for the paper trading service and dashboard.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LiveMetrics:
    """Mutable snapshot of runtime health and trading stats."""

    ws_status: str = "disconnected"
    last_tick_at: float | None = None
    last_heartbeat_at: float | None = None
    reconnect_count: int = 0
    reconnect_events: list[dict[str, Any]] = field(default_factory=list)
    current_candle: dict[str, Any] | None = None
    market_status: str = "unknown"
    avg_signal_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)
    today_signals: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    win_rate: float = 0.0
    running_pnl: float = 0.0
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    recent_errors: list[str] = field(default_factory=list)
    db_ok: bool = True
    cpu_pct: float = 0.0
    mem_pct: float = 0.0
    last_candle_ts: str | None = None
    missed_candles_count: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._max_latencies = 200
        self._max_errors = 50
        self._max_reconnect_events = 50
        self._max_equity = 500

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of current metrics."""
        with self._lock:
            payload = asdict(self)
            payload.pop("_lock", None)
            return payload

    def set_ws_status(self, status: str) -> None:
        with self._lock:
            self.ws_status = status

    def record_tick(self, *, at: float | None = None) -> None:
        with self._lock:
            self.last_tick_at = at if at is not None else time.time()
            self.ws_status = "connected"

    def record_heartbeat(self, *, at: float | None = None) -> None:
        with self._lock:
            self.last_heartbeat_at = at if at is not None else time.time()

    def record_reconnect(self, reason: str) -> None:
        with self._lock:
            self.reconnect_count += 1
            event = {"at": time.time(), "reason": reason, "count": self.reconnect_count}
            self.reconnect_events.append(event)
            if len(self.reconnect_events) > self._max_reconnect_events:
                self.reconnect_events = self.reconnect_events[-self._max_reconnect_events :]
            self.ws_status = "reconnecting"

    def record_latency(self, latency_ms: float) -> None:
        with self._lock:
            self.latencies.append(float(latency_ms))
            if len(self.latencies) > self._max_latencies:
                self.latencies = self.latencies[-self._max_latencies :]
            self.avg_signal_latency_ms = sum(self.latencies) / len(self.latencies)

    def record_error(self, message: str) -> None:
        with self._lock:
            self.recent_errors.append(f"{time.strftime('%H:%M:%S')} {message}")
            if len(self.recent_errors) > self._max_errors:
                self.recent_errors = self.recent_errors[-self._max_errors :]

    def set_current_candle(self, candle: dict[str, Any] | None) -> None:
        with self._lock:
            self.current_candle = candle
            if candle and candle.get("timestamp"):
                self.last_candle_ts = str(candle["timestamp"])

    def update_trading_stats(
        self,
        *,
        today_signals: int | None = None,
        open_trades: int | None = None,
        closed_trades: int | None = None,
        win_rate: float | None = None,
        running_pnl: float | None = None,
        equity_curve: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            if today_signals is not None:
                self.today_signals = today_signals
            if open_trades is not None:
                self.open_trades = open_trades
            if closed_trades is not None:
                self.closed_trades = closed_trades
            if win_rate is not None:
                self.win_rate = win_rate
            if running_pnl is not None:
                self.running_pnl = running_pnl
            if equity_curve is not None:
                self.equity_curve = equity_curve[-self._max_equity :]

    def set_system(
        self,
        *,
        db_ok: bool | None = None,
        cpu_pct: float | None = None,
        mem_pct: float | None = None,
        market_status: str | None = None,
        missed_candles_count: int | None = None,
    ) -> None:
        with self._lock:
            if db_ok is not None:
                self.db_ok = db_ok
            if cpu_pct is not None:
                self.cpu_pct = cpu_pct
            if mem_pct is not None:
                self.mem_pct = mem_pct
            if market_status is not None:
                self.market_status = market_status
            if missed_candles_count is not None:
                self.missed_candles_count = missed_candles_count
