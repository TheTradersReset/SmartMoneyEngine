"""
SQLite persistence for live paper signal pipeline.

Stores closed 5-minute candles and generated signals (paper mode only).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "paper" / "realtime_signals.db"


@dataclass(frozen=True)
class StoredSignal:
    """Persisted signal row."""

    id: int | None
    timestamp: str
    direction: str
    engine_version: str
    entry: float
    stop: float
    target1: float
    target2: float
    target_structure: str
    confidence: float
    regime: str
    throttle_level: str
    accepted: bool
    rejection_reason: str | None
    raw_payload: dict[str, Any]


class PaperSignalDatabase:
    """SQLite store for candles and paper signals."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Websocket callbacks run on a non-main thread; allow cross-thread use.
        # Writers are serialized by RealtimeSignalPipeline._lock.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                tick_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(symbol, timestamp)
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                direction TEXT NOT NULL,
                engine_version TEXT NOT NULL,
                entry REAL NOT NULL,
                stop REAL NOT NULL,
                target1 REAL NOT NULL,
                target2 REAL NOT NULL,
                target_structure TEXT NOT NULL,
                confidence REAL NOT NULL,
                regime TEXT NOT NULL,
                throttle_level TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                rejection_reason TEXT,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                risk REAL,
                reward REAL,
                target REAL,
                outcome TEXT,
                holding_bars INTEGER,
                outcome_timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS signal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                event_type TEXT NOT NULL,
                event_time TEXT NOT NULL,
                details_json TEXT NOT NULL,
                FOREIGN KEY(signal_id) REFERENCES signals(id)
            );

            CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON candles(symbol, timestamp);
            CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);
            CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction);

            CREATE TABLE IF NOT EXISTS signal_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                trend TEXT,
                market_regime TEXT,
                buy_score REAL,
                sell_score REAL,
                final_signal TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason_codes TEXT NOT NULL,
                evaluation_time_ms REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(symbol, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_signal_decisions_ts ON signal_decisions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_signal_decisions_symbol_ts ON signal_decisions(symbol, timestamp);
            """
        )
        self._migrate_signals_outcome_columns(cursor)
        self._conn.commit()
        logger.info("Paper signal database ready: %s", self.db_path)

    def _migrate_signals_outcome_columns(self, cursor: sqlite3.Cursor) -> None:
        """Add post-signal outcome columns on existing databases."""
        existing = {row[1] for row in cursor.execute("PRAGMA table_info(signals)").fetchall()}
        alterations = (
            ("risk", "REAL"),
            ("reward", "REAL"),
            ("target", "REAL"),
            ("outcome", "TEXT"),
            ("holding_bars", "INTEGER"),
            ("outcome_timestamp", "TEXT"),
        )
        for name, typedef in alterations:
            if name not in existing:
                cursor.execute(f"ALTER TABLE signals ADD COLUMN {name} {typedef}")

    def insert_candle(
        self,
        *,
        symbol: str,
        timestamp: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        tick_count: int,
    ) -> int:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO candles
            (symbol, timestamp, open, high, low, close, volume, tick_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                timestamp,
                open_,
                high,
                low,
                close,
                volume,
                tick_count,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def insert_signal_decision(self, decision: dict[str, Any]) -> int:
        """Persist one engine decision row for a closed candle."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO signal_decisions (
                timestamp, symbol, open, high, low, close, volume,
                trend, market_regime, buy_score, sell_score,
                final_signal, decision, reason_codes, evaluation_time_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision["timestamp"],
                decision["symbol"],
                decision["open"],
                decision["high"],
                decision["low"],
                decision["close"],
                decision["volume"],
                decision.get("trend"),
                decision.get("market_regime"),
                decision.get("buy_score"),
                decision.get("sell_score"),
                decision["final_signal"],
                decision["decision"],
                json.dumps(decision.get("reason_codes", []), default=str),
                decision["evaluation_time_ms"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def insert_signal(self, signal: dict[str, Any]) -> int:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO signals (
                timestamp, direction, engine_version, entry, stop, target1, target2,
                target_structure, confidence, regime, throttle_level, accepted,
                rejection_reason, raw_json, created_at,
                risk, reward, target, outcome, holding_bars, outcome_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal["timestamp"],
                signal["direction"],
                signal["engine_version"],
                signal["entry"],
                signal["stop"],
                signal["target1"],
                signal["target2"],
                signal["target_structure"],
                signal["confidence"],
                signal["regime"],
                signal["throttle_level"],
                1 if signal.get("accepted") else 0,
                signal.get("rejection_reason"),
                json.dumps(signal, default=str),
                datetime.now(timezone.utc).isoformat(),
                signal.get("risk"),
                signal.get("reward"),
                signal.get("target"),
                signal.get("outcome", "PENDING"),
                signal.get("holding_bars"),
                signal.get("outcome_timestamp"),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def update_signal_outcome(
        self,
        *,
        timestamp: str,
        direction: str,
        entry: float,
        stop: float,
        target: float | None,
        risk: float,
        reward: float,
        outcome: str,
        holding_bars: int,
        outcome_timestamp: str,
        forward_outcome: dict[str, Any] | None = None,
    ) -> int:
        """
        Update a previously stored signal with post-forward outcome fields.

        Decision ``timestamp`` is never changed. Lookup tries exact match first,
        then timezone-normalized equality (``+0530`` vs ``+05:30``).
        """
        from src.signals.signal_outcome import normalize_timestamp_key, timestamps_equivalent

        cursor = self._conn.cursor()
        match = cursor.execute(
            """
            SELECT id, raw_json, timestamp FROM signals
            WHERE timestamp = ? AND direction = ?
            ORDER BY id DESC LIMIT 1
            """,
            (timestamp, direction),
        ).fetchone()
        if match is None:
            target_key = normalize_timestamp_key(timestamp)
            candidates = cursor.execute(
                """
                SELECT id, raw_json, timestamp FROM signals
                WHERE direction = ?
                ORDER BY id DESC
                LIMIT 5000
                """,
                (direction,),
            ).fetchall()
            for candidate in candidates:
                if timestamps_equivalent(candidate["timestamp"], timestamp) or (
                    target_key is not None
                    and normalize_timestamp_key(candidate["timestamp"]) == target_key
                ):
                    match = candidate
                    break
        if match is None:
            return 0
        raw = {}
        try:
            raw = json.loads(match["raw_json"] or "{}")
        except json.JSONDecodeError:
            raw = {}
        if forward_outcome is not None:
            raw["forward_outcome"] = forward_outcome
            raw["outcome"] = outcome
            raw["outcome_timestamp"] = outcome_timestamp
            raw["holding_bars"] = holding_bars
            raw["risk"] = risk
            raw["reward"] = reward
            raw["target"] = target
        cursor.execute(
            """
            UPDATE signals SET
                entry = ?,
                stop = ?,
                target = ?,
                risk = ?,
                reward = ?,
                outcome = ?,
                holding_bars = ?,
                outcome_timestamp = ?,
                raw_json = ?
            WHERE id = ?
            """,
            (
                entry,
                stop,
                target,
                risk,
                reward,
                outcome,
                holding_bars,
                outcome_timestamp,
                json.dumps(raw, default=str),
                int(match["id"]),
            ),
        )
        self._conn.commit()
        return int(cursor.rowcount)

    def log_event(
        self,
        *,
        signal_id: int | None,
        event_type: str,
        details: dict[str, Any],
    ) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO signal_events (signal_id, event_type, event_time, details_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                signal_id,
                event_type,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(details, default=str),
            ),
        )
        self._conn.commit()

    def recent_candles(self, *, symbol: str, limit: int = 500) -> list[dict[str, Any]]:
        cursor = self._conn.cursor()
        rows = cursor.execute(
            """
            SELECT symbol, timestamp, open, high, low, close, volume, tick_count
            FROM candles
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def recent_signals(self, *, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self._conn.cursor()
        rows = cursor.execute(
            """
            SELECT * FROM signals ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def pending_signal_outcomes(self, *, limit: int = 5_000) -> list[dict[str, Any]]:
        """Accepted signals still waiting for forward outcome evaluation."""
        cursor = self._conn.cursor()
        rows = cursor.execute(
            """
            SELECT id, timestamp, direction, entry, stop, outcome
            FROM signals
            WHERE accepted = 1 AND (outcome IS NULL OR outcome = 'PENDING')
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_decisions(self, *, symbol: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self._conn.cursor()
        if symbol:
            rows = cursor.execute(
                """
                SELECT * FROM signal_decisions
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        else:
            rows = cursor.execute(
                """
                SELECT * FROM signal_decisions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            try:
                payload["reason_codes"] = json.loads(payload.get("reason_codes") or "[]")
            except json.JSONDecodeError:
                payload["reason_codes"] = []
            results.append(payload)
        return results

    def close(self) -> None:
        self._conn.close()
