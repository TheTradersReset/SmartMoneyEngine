"""
In-memory paper trade tracker over the existing ``signals`` SQLite table.

No schema changes. Does not place broker orders.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.storage.sqlite import PaperSignalDatabase

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class PaperTrade:
    """Open or closed paper trade mirrored from an accepted signal."""

    signal_id: int | None
    timestamp: str
    direction: str
    entry: float
    stop: float
    target1: float
    target2: float
    risk: float
    engine_version: str
    status: str = "OPEN"
    outcome: str | None = None
    reward: float | None = None
    holding_bars: int | None = None
    outcome_timestamp: str | None = None
    opened_at: float = field(default_factory=time.time)


class PaperTradeManager:
    """Track open/closed paper trades and compute simple session stats."""

    def __init__(self, db: PaperSignalDatabase) -> None:
        self.db = db
        self._lock = threading.RLock()
        self._open: dict[str, PaperTrade] = {}
        self._closed: list[PaperTrade] = []
        self._seen: set[str] = set()

    @staticmethod
    def dedupe_key(timestamp: str, direction: str) -> str:
        return f"{timestamp}|{str(direction).upper()}"

    def get_signal_id(
        self,
        timestamp: str,
        direction: str,
        *,
        retries: int = 8,
        delay_seconds: float = 0.05,
    ) -> int | None:
        """Resolve ``signals.id`` after async writer flush with brief retries."""
        direction_u = str(direction).upper()
        for _ in range(max(1, retries)):
            rows = self.db.recent_signals(limit=200)
            for row in rows:
                if str(row.get("direction", "")).upper() != direction_u:
                    continue
                if str(row.get("timestamp")) == str(timestamp):
                    try:
                        return int(row["id"])
                    except (TypeError, ValueError, KeyError):
                        return None
            time.sleep(delay_seconds)
        return None

    def on_signal(self, record: dict[str, Any]) -> PaperTrade | None:
        """Register an accepted signal as an open paper trade."""
        if not record.get("accepted", True):
            return None
        timestamp = str(record["timestamp"])
        direction = str(record["direction"]).upper()
        key = self.dedupe_key(timestamp, direction)
        with self._lock:
            if key in self._seen:
                return self._open.get(key)
            self._seen.add(key)

        signal_id = record.get("id") or record.get("signal_id")
        if signal_id is None:
            signal_id = self.get_signal_id(timestamp, direction)

        entry = float(record["entry"])
        stop = float(record["stop"])
        risk = float(record.get("risk") or abs(entry - stop))
        trade = PaperTrade(
            signal_id=int(signal_id) if signal_id is not None else None,
            timestamp=timestamp,
            direction=direction,
            entry=entry,
            stop=stop,
            target1=float(record.get("target1") or 0.0),
            target2=float(record.get("target2") or 0.0),
            risk=risk,
            engine_version=str(record.get("engine_version") or ""),
        )
        with self._lock:
            self._open[key] = trade
        return trade

    def on_outcome(
        self,
        *,
        timestamp: str,
        direction: str,
        outcome: str,
        reward: float | None = None,
        holding_bars: int | None = None,
        outcome_timestamp: str | None = None,
        signal_id: int | None = None,
    ) -> PaperTrade | None:
        """Move an open trade to closed with outcome fields."""
        key = self.dedupe_key(timestamp, direction)
        with self._lock:
            trade = self._open.pop(key, None)
            if trade is None:
                trade = PaperTrade(
                    signal_id=signal_id,
                    timestamp=timestamp,
                    direction=str(direction).upper(),
                    entry=0.0,
                    stop=0.0,
                    target1=0.0,
                    target2=0.0,
                    risk=0.0,
                    engine_version="",
                )
            trade.status = "CLOSED"
            trade.outcome = str(outcome).upper()
            trade.reward = reward
            trade.holding_bars = holding_bars
            trade.outcome_timestamp = outcome_timestamp
            if signal_id is not None:
                trade.signal_id = signal_id
            self._closed.append(trade)
            return trade

    def list_open(self) -> list[PaperTrade]:
        with self._lock:
            return list(self._open.values())

    def list_closed_today(self) -> list[PaperTrade]:
        today = datetime.now(tz=IST).date().isoformat()
        with self._lock:
            results: list[PaperTrade] = []
            for trade in self._closed:
                ts = trade.outcome_timestamp or trade.timestamp
                if str(ts).startswith(today):
                    results.append(trade)
            return results

    def stats(self) -> dict[str, Any]:
        """Win rate, running PnL, and equity curve from accepted closed trades."""
        with self._lock:
            closed = list(self._closed)
            open_count = len(self._open)

        # Prefer DB-backed equity for durability across restarts
        rows = [
            row
            for row in self.db.recent_signals(limit=5_000)
            if int(row.get("accepted") or 0) == 1
        ]
        rows_sorted = sorted(rows, key=lambda r: (str(r.get("timestamp")), int(r.get("id") or 0)))

        equity = 0.0
        curve: list[dict[str, Any]] = []
        wins = 0
        losses = 0
        decided = 0
        today = datetime.now(tz=IST).date().isoformat()
        today_signals = 0

        for row in rows_sorted:
            ts = str(row.get("timestamp") or "")
            if ts.startswith(today):
                today_signals += 1
            outcome = str(row.get("outcome") or "").upper()
            if outcome in {"", "PENDING", "REJECTED"}:
                continue
            reward = row.get("reward")
            try:
                pnl = float(reward) if reward is not None else 0.0
            except (TypeError, ValueError):
                pnl = 0.0
            equity += pnl
            decided += 1
            if outcome == "WIN":
                wins += 1
            elif outcome == "LOSS":
                losses += 1
            curve.append({"timestamp": ts, "equity": round(equity, 4), "outcome": outcome})

        win_rate = (wins / decided) if decided else 0.0
        if not closed and decided:
            closed_count = decided
        else:
            closed_count = len(self.list_closed_today()) if closed else decided

        return {
            "open_trades": open_count,
            "closed_trades": closed_count,
            "today_signals": today_signals,
            "win_rate": win_rate,
            "running_pnl": equity,
            "equity_curve": curve,
            "wins": wins,
            "losses": losses,
        }
