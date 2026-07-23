"""Persistence for trade validation results (separate from signal-engine DB)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.logger import logger
from src.trade_validation.config import DEFAULT_VALIDATION_DB
from src.trade_validation.models import TradeValidationResult


class TradeValidationDatabase:
    """SQLite store for post-signal validation analytics."""

    def __init__(self, db_path: Path | str = DEFAULT_VALIDATION_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trade_validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_signal_id INTEGER NOT NULL UNIQUE,
                signal_timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                signal_score REAL,
                reason_codes TEXT NOT NULL,
                next_candle_close REAL,
                next_3_candle_close REAL,
                next_5_candle_close REAL,
                window_high REAL,
                window_low REAL,
                mfe REAL NOT NULL,
                mae REAL NOT NULL,
                target_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_pct REAL NOT NULL,
                stop_pct REAL NOT NULL,
                target_hit INTEGER NOT NULL,
                stop_hit INTEGER NOT NULL,
                pnl REAL,
                outcome TEXT NOT NULL,
                holding_bars INTEGER,
                exit_reason TEXT NOT NULL,
                evaluation_window_bars INTEGER NOT NULL,
                exit_timestamp TEXT,
                evaluated_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_trade_validations_outcome
                ON trade_validations(outcome);
            CREATE INDEX IF NOT EXISTS idx_trade_validations_ts
                ON trade_validations(signal_timestamp);
            CREATE INDEX IF NOT EXISTS idx_trade_validations_direction
                ON trade_validations(direction);

            CREATE TABLE IF NOT EXISTS validation_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """,
        )
        self._conn.commit()
        logger.info("Trade validation database ready: %s", self.db_path)

    def get_last_processed_signal_id(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM validation_state WHERE key = 'last_processed_signal_id'",
        ).fetchone()
        return int(row["value"]) if row else 0

    def set_last_processed_signal_id(self, signal_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO validation_state (key, value) VALUES ('last_processed_signal_id', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(signal_id),),
        )
        self._conn.commit()

    def upsert_validation(self, result: TradeValidationResult) -> int:
        now = datetime.now(timezone.utc).isoformat()
        payload = result.as_dict()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO trade_validations (
                source_signal_id, signal_timestamp, symbol, direction, entry_price,
                signal_score, reason_codes, next_candle_close, next_3_candle_close,
                next_5_candle_close, window_high, window_low, mfe, mae,
                target_price, stop_price, target_pct, stop_pct, target_hit, stop_hit,
                pnl, outcome, holding_bars, exit_reason, evaluation_window_bars,
                exit_timestamp, evaluated_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(source_signal_id) DO UPDATE SET
                next_candle_close = excluded.next_candle_close,
                next_3_candle_close = excluded.next_3_candle_close,
                next_5_candle_close = excluded.next_5_candle_close,
                window_high = excluded.window_high,
                window_low = excluded.window_low,
                mfe = excluded.mfe,
                mae = excluded.mae,
                target_hit = excluded.target_hit,
                stop_hit = excluded.stop_hit,
                pnl = excluded.pnl,
                outcome = excluded.outcome,
                holding_bars = excluded.holding_bars,
                exit_reason = excluded.exit_reason,
                exit_timestamp = excluded.exit_timestamp,
                updated_at = excluded.updated_at
            """,
            (
                payload["source_signal_id"],
                payload["signal_timestamp"],
                payload["symbol"],
                payload["direction"],
                payload["entry_price"],
                payload["signal_score"],
                json.dumps(payload["reason_codes"], default=str),
                payload["next_candle_close"],
                payload["next_3_candle_close"],
                payload["next_5_candle_close"],
                payload["window_high"],
                payload["window_low"],
                payload["mfe"],
                payload["mae"],
                payload["target_price"],
                payload["stop_price"],
                payload["target_pct"],
                payload["stop_pct"],
                1 if payload["target_hit"] else 0,
                1 if payload["stop_hit"] else 0,
                payload["pnl"],
                payload["outcome"],
                payload["holding_bars"],
                payload["exit_reason"],
                payload["evaluation_window_bars"],
                payload["exit_timestamp"],
                now,
                now,
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def recent_validations(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM trade_validations ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            try:
                payload["reason_codes"] = json.loads(payload.get("reason_codes") or "[]")
            except json.JSONDecodeError:
                payload["reason_codes"] = []
            payload["target_hit"] = bool(payload.get("target_hit"))
            payload["stop_hit"] = bool(payload.get("stop_hit"))
            results.append(payload)
        return results

    def open_validations(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM trade_validations WHERE outcome = 'OPEN' ORDER BY signal_timestamp ASC",
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()
