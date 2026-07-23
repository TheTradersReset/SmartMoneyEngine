"""
In-memory OHLCV cache for live signal evaluation.

The signal engine reads exclusively from RAM; SQLite is persistence only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.candle_builder import Candle

DEFAULT_MAX_BARS = 2500


@dataclass
class MarketMemoryCache:
    """
    Rolling in-memory store for closed candles and the active bucket.

    Parameters
    ----------
    max_bars : int
        Maximum closed bars retained in RAM.
    """

    max_bars: int = DEFAULT_MAX_BARS
    closed_rows: deque[dict[str, Any]] = field(default_factory=deque)
    current_candle: Candle | None = None
    _initialized: bool = False

    def __post_init__(self) -> None:
        self.closed_rows = deque(maxlen=self.max_bars)

    @property
    def bar_count(self) -> int:
        return len(self.closed_rows)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def load_history_rows(self, rows: list[dict[str, Any]]) -> None:
        """Seed cache from historical OHLCV rows."""
        self.closed_rows.clear()
        for row in rows[-self.max_bars :]:
            self.closed_rows.append(dict(row))
        self._initialized = True

    def append_closed_row(self, row: dict[str, Any]) -> int:
        """Append one closed bar; return its zero-based index."""
        self.closed_rows.append(dict(row))
        self._initialized = True
        return len(self.closed_rows) - 1

    def set_current_candle(self, candle: Candle | None) -> None:
        self.current_candle = candle

    def as_dataframe(self) -> pd.DataFrame:
        if not self.closed_rows:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        return pd.DataFrame(list(self.closed_rows)).reset_index(drop=True)

    def latest_row(self) -> dict[str, Any] | None:
        if not self.closed_rows:
            return None
        return dict(self.closed_rows[-1])

    def recent_rows(self, n: int) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        return [dict(row) for row in list(self.closed_rows)[-n:]]
