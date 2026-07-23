"""Read-only access to replay / paper signal SQLite databases."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ReplayAnalyticsReader:
    """Load signal_decisions and signals without modifying the source DB."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Replay database not found: {self.db_path}")
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def fetch_decisions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, timestamp, symbol, open, high, low, close, volume,
                   trend, market_regime, buy_score, sell_score,
                   final_signal, decision, reason_codes, evaluation_time_ms, created_at
            FROM signal_decisions
            ORDER BY timestamp ASC, id ASC
            """,
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            try:
                payload["reason_codes"] = json.loads(payload.get("reason_codes") or "[]")
            except json.JSONDecodeError:
                payload["reason_codes"] = []
            if not isinstance(payload["reason_codes"], list):
                payload["reason_codes"] = []
            results.append(payload)
        return results

    def fetch_signals(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, timestamp, direction, engine_version, entry, accepted,
                   rejection_reason, throttle_level, regime
            FROM signals
            ORDER BY timestamp ASC, id ASC
            """,
        ).fetchall()
        return [dict(row) for row in rows]

    def fetch_candle_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM candles").fetchone()
        return int(row["c"]) if row else 0
