"""Historical OHLCV feed for replay — loads candles, never generates signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import pandas as pd

from src.brokers.websocket_client import NIFTY50_SYMBOL
from src.data.candle_builder import Candle

IST = ZoneInfo("Asia/Kolkata")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
WARM_START_BARS = 2000


@dataclass(frozen=True)
class ReplayWindow:
    """Inclusive calendar date window for replay candles."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Replay window end {self.end} is before start {self.start}")


class HistoricalDataFeed:
    """
    Load historical OHLCV and yield closed Candle objects for the replay range.

    Bars before the replay window are exposed for warm-start only.
    """

    def __init__(
        self,
        *,
        csv_path: Path | str | None = DEFAULT_HISTORY_CSV,
        symbol: str = NIFTY50_SYMBOL,
        warm_start_bars: int = WARM_START_BARS,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_HISTORY_CSV
        self.symbol = symbol
        self.warm_start_bars = warm_start_bars
        self._frame: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Historical CSV not found: {self.csv_path}")
        frame = pd.read_csv(self.csv_path)
        if "Date" not in frame.columns:
            raise ValueError(f"Historical CSV missing Date column: {self.csv_path}")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in frame.columns:
                raise ValueError(f"Historical CSV missing {col} column: {self.csv_path}")
        working = frame.copy()
        working["_dt"] = pd.to_datetime(working["Date"], utc=False)
        if working["_dt"].dt.tz is None:
            working["_dt"] = working["_dt"].dt.tz_localize(IST)
        else:
            working["_dt"] = working["_dt"].dt.tz_convert(IST)
        working = working.sort_values("_dt").reset_index(drop=True)
        self._frame = working
        return working

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            return self.load()
        return self._frame

    def warm_start_frame(self, window: ReplayWindow) -> pd.DataFrame:
        """Bars strictly before the replay window (tail-capped for context)."""
        frame = self.frame
        start_ts = pd.Timestamp(datetime.combine(window.start, datetime.min.time()), tz=IST)
        prior = frame[frame["_dt"] < start_ts]
        if prior.empty:
            return prior.drop(columns=["_dt"], errors="ignore")
        return prior.tail(self.warm_start_bars).drop(columns=["_dt"]).reset_index(drop=True)

    def replay_candles(self, window: ReplayWindow) -> list[Candle]:
        """Closed candles whose session date falls inside [start, end]."""
        frame = self.frame
        start_ts = pd.Timestamp(datetime.combine(window.start, datetime.min.time()), tz=IST)
        end_ts = pd.Timestamp(datetime.combine(window.end, datetime.max.time()), tz=IST)
        slice_ = frame[(frame["_dt"] >= start_ts) & (frame["_dt"] <= end_ts)]
        candles: list[Candle] = []
        for _, row in slice_.iterrows():
            candles.append(
                Candle(
                    symbol=self.symbol,
                    timestamp=row["_dt"].to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    tick_count=0,
                ),
            )
        return candles

    def iter_candles(self, window: ReplayWindow) -> Iterator[Candle]:
        yield from self.replay_candles(window)


def window_for_day(day: date) -> ReplayWindow:
    return ReplayWindow(start=day, end=day)


def window_for_week(year: int, week: int) -> ReplayWindow:
    """ISO week (Monday–Sunday)."""
    start = date.fromisocalendar(year, week, 1)
    end = start + timedelta(days=6)
    return ReplayWindow(start=start, end=end)


def window_for_month(year: int, month: int) -> ReplayWindow:
    start = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return ReplayWindow(start=start, end=end)


def window_for_range(start: date, end: date) -> ReplayWindow:
    return ReplayWindow(start=start, end=end)


def parse_iso_week(text: str) -> tuple[int, int]:
    """Parse 'YYYY-Www' or 'YYYY-WW' into (year, week)."""
    cleaned = text.strip().upper().replace("W", "")
    parts = cleaned.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid ISO week '{text}'. Expected YYYY-Www (e.g. 2026-W11).")
    return int(parts[0]), int(parts[1])


def parse_year_month(text: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' into (year, month)."""
    parts = text.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid month '{text}'. Expected YYYY-MM.")
    year, month = int(parts[0]), int(parts[1])
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month number in '{text}'.")
    return year, month
