"""Write historical OHLCV bars into research_dataset.db."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.dataset_builder.schema import DEFAULT_RESEARCH_DATASET_DB, connect_research_dataset


class ResearchDatasetWriter:
    """Insert / upsert bars into the research dataset database."""

    def __init__(self, db_path: Path | str = DEFAULT_RESEARCH_DATASET_DB) -> None:
        self.db_path = Path(db_path)
        self._conn = connect_research_dataset(self.db_path, read_only=False)

    def close(self) -> None:
        self._conn.close()

    def upsert_bar(
        self,
        *,
        symbol: str,
        resolution: str,
        timestamp: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> int:
        """Insert or replace one bar. Returns row id."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO bars (
                symbol, resolution, timestamp, open, high, low, close, volume, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, resolution, timestamp) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                created_at = excluded.created_at
            """,
            (
                symbol,
                resolution,
                timestamp,
                float(open_),
                float(high),
                float(low),
                float(close),
                float(volume),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def upsert_bars(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """
        Bulk upsert bars.

        Each mapping must include:
        symbol, resolution, timestamp, open, high, low, close, volume
        (open may also be provided as open_).
        """
        if not rows:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        payload = [
            (
                str(row["symbol"]),
                str(row["resolution"]),
                str(row["timestamp"]),
                float(row["open"] if "open" in row else row["open_"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                now,
            )
            for row in rows
        ]
        self._conn.executemany(
            """
            INSERT INTO bars (
                symbol, resolution, timestamp, open, high, low, close, volume, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, resolution, timestamp) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                created_at = excluded.created_at
            """,
            payload,
        )
        self._conn.commit()
        return len(payload)

    def delete_range(
        self,
        *,
        symbol: str,
        resolution: str,
        start_timestamp: str,
        end_timestamp: str,
    ) -> int:
        """Delete bars in [start_timestamp, end_timestamp] inclusive. Returns deleted count."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            DELETE FROM bars
            WHERE symbol = ?
              AND resolution = ?
              AND timestamp >= ?
              AND timestamp <= ?
            """,
            (symbol, resolution, start_timestamp, end_timestamp),
        )
        self._conn.commit()
        return int(cursor.rowcount)
