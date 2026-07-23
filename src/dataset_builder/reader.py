"""Read historical OHLCV bars from research_dataset.db."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.dataset_builder.schema import DEFAULT_RESEARCH_DATASET_DB, connect_research_dataset


class ResearchDatasetReader:
    """Read-only queries against the research dataset database."""

    def __init__(self, db_path: Path | str = DEFAULT_RESEARCH_DATASET_DB) -> None:
        self.db_path = Path(db_path)
        self._conn = connect_research_dataset(self.db_path, read_only=True)

    def close(self) -> None:
        self._conn.close()

    def fetch_bars(
        self,
        *,
        symbol: str,
        resolution: str,
        start_timestamp: str | None = None,
        end_timestamp: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return bars ordered by timestamp ascending."""
        clauses = ["symbol = ?", "resolution = ?"]
        params: list[Any] = [symbol, resolution]
        if start_timestamp is not None:
            clauses.append("timestamp >= ?")
            params.append(start_timestamp)
        if end_timestamp is not None:
            clauses.append("timestamp <= ?")
            params.append(end_timestamp)

        sql = f"""
            SELECT id, symbol, resolution, timestamp, open, high, low, close, volume, created_at
            FROM bars
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp ASC
        """
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def count_bars(
        self,
        *,
        symbol: str,
        resolution: str,
        start_timestamp: str | None = None,
        end_timestamp: str | None = None,
    ) -> int:
        clauses = ["symbol = ?", "resolution = ?"]
        params: list[Any] = [symbol, resolution]
        if start_timestamp is not None:
            clauses.append("timestamp >= ?")
            params.append(start_timestamp)
        if end_timestamp is not None:
            clauses.append("timestamp <= ?")
            params.append(end_timestamp)
        row = self._conn.execute(
            f"SELECT COUNT(*) AS c FROM bars WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        return int(row["c"]) if row else 0

    def min_max_timestamp(
        self,
        *,
        symbol: str,
        resolution: str,
    ) -> tuple[str | None, str | None]:
        row = self._conn.execute(
            """
            SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts
            FROM bars
            WHERE symbol = ? AND resolution = ?
            """,
            (symbol, resolution),
        ).fetchone()
        if row is None:
            return None, None
        return row["min_ts"], row["max_ts"]

    def list_symbols(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT symbol FROM bars ORDER BY symbol ASC",
        ).fetchall()
        return [str(row["symbol"]) for row in rows]

    def list_resolutions(self, *, symbol: str | None = None) -> list[str]:
        if symbol is None:
            rows = self._conn.execute(
                "SELECT DISTINCT resolution FROM bars ORDER BY resolution ASC",
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT DISTINCT resolution FROM bars
                WHERE symbol = ?
                ORDER BY resolution ASC
                """,
                (symbol,),
            ).fetchall()
        return [str(row["resolution"]) for row in rows]
