"""
Research dataset database — schema and connection helpers.

Stores historical OHLCV bars for research / replay consumption.
Does not implement CLI, resume, or replay orchestration.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_DATASET_DB = PROJECT_ROOT / "data" / "datasets" / "research_dataset.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    resolution TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(symbol, resolution, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_resolution_ts
    ON bars(symbol, resolution, timestamp);

CREATE INDEX IF NOT EXISTS idx_bars_timestamp
    ON bars(timestamp);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts
    ON bars(symbol, timestamp);
"""


def connect_research_dataset(
    db_path: Path | str = DEFAULT_RESEARCH_DATASET_DB,
    *,
    read_only: bool = False,
) -> sqlite3.Connection:
    """
    Open a connection to research_dataset.db and ensure schema exists (unless read-only).
    """
    path = Path(db_path)
    if read_only:
        if not path.exists():
            raise FileNotFoundError(f"Research dataset database not found: {path}")
        uri = f"file:{path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Research dataset database ready: %s", path)
    conn.row_factory = sqlite3.Row
    return conn
