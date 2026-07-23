"""Read-only loaders for dataset verification (DB or CSV). Never writes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset_builder.schema import DEFAULT_RESEARCH_DATASET_DB
from src.dataset_verification.calendar import IST


def load_bars_from_db(
    db_path: Path | str = DEFAULT_RESEARCH_DATASET_DB,
    *,
    symbol: str,
    resolution: str = "5",
    start_timestamp: str | None = None,
    end_timestamp: str | None = None,
) -> pd.DataFrame:
    """Load bars from research_dataset.db in read-only mode."""
    from src.dataset_builder.reader import ResearchDatasetReader

    reader = ResearchDatasetReader(db_path)
    try:
        rows = reader.fetch_bars(
            symbol=symbol,
            resolution=resolution,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
    finally:
        reader.close()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows)
    return _normalize_frame(frame)


def load_bars_from_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load OHLCV from a Replay-compatible CSV (Date/Open/High/Low/Close/Volume)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    raw = pd.read_csv(path)
    rename = {}
    if "Date" in raw.columns and "timestamp" not in raw.columns:
        rename["Date"] = "timestamp"
    for src, dst in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")):
        if src in raw.columns and dst not in raw.columns:
            rename[src] = dst
    frame = raw.rename(columns=rename)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return _normalize_frame(frame[list(required)].copy())


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=False)
    if working["timestamp"].dt.tz is None:
        working["timestamp"] = working["timestamp"].dt.tz_localize(IST)
    else:
        working["timestamp"] = working["timestamp"].dt.tz_convert(IST)
    for col in ("open", "high", "low", "close", "volume"):
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.sort_values("timestamp").reset_index(drop=True)
    return working
