"""
Incremental indicator updates for live 5-minute bars.

Produces the same columns as ``FilterContextBuilder.enrich`` for EMA / RSI /
ATR / VWAP / volume-spike without recomputing the full history each bar.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from src.research.filter_research_engine import (
    ATR_PERIOD,
    EMA_PERIODS,
    FilterContextBuilder,
    RSI_PERIOD,
    VOLUME_LOOKBACK,
    VOLUME_SPIKE_MULTIPLIER,
)

EMA22_PERIOD = 22
ALL_EMA_PERIODS: tuple[int, ...] = tuple(sorted(set(EMA_PERIODS + (EMA22_PERIOD,))))


@dataclass
class IncrementalIndicatorState:
    """Mutable rolling indicator state for one live symbol stream."""

    ema_values: dict[int, float | None] = field(default_factory=dict)
    prev_close: float | None = None
    rsi_gains: deque[float] = field(default_factory=lambda: deque(maxlen=RSI_PERIOD))
    rsi_losses: deque[float] = field(default_factory=lambda: deque(maxlen=RSI_PERIOD))
    tr_window: deque[float] = field(default_factory=lambda: deque(maxlen=ATR_PERIOD))
    volume_window: deque[float] = field(default_factory=lambda: deque(maxlen=VOLUME_LOOKBACK))
    vwap_session_day: date | None = None
    vwap_cum_tpv: float = 0.0
    vwap_cum_volume: float = 0.0
    enriched_columns: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.ema_values.clear()
        self.prev_close = None
        self.rsi_gains.clear()
        self.rsi_losses.clear()
        self.tr_window.clear()
        self.volume_window.clear()
        self.vwap_session_day = None
        self.vwap_cum_tpv = 0.0
        self.vwap_cum_volume = 0.0
        self.enriched_columns.clear()


class IncrementalIndicatorEngine:
    """
    Maintain enriched indicator columns incrementally for live evaluation.

    Warm-start via full ``FilterContextBuilder.enrich``, then append one bar
    at a time during live processing.
    """

    def __init__(self) -> None:
        self._builder = FilterContextBuilder()
        self._state = IncrementalIndicatorState()
        self.enriched: pd.DataFrame | None = None

    @staticmethod
    def _ema_alpha(period: int) -> float:
        return 2.0 / (period + 1.0)

    @staticmethod
    def _session_day(timestamp: pd.Timestamp) -> date:
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Kolkata")
        else:
            ts = ts.tz_convert("Asia/Kolkata")
        return ts.date()

    def seed_from_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Full enrich for warm-start; capture state from the last bar."""
        enriched = self._builder.enrich(frame)
        self.enriched = enriched.copy()
        self._state.reset()
        if enriched.empty:
            return enriched

        self._state.enriched_columns = list(enriched.columns)
        last_idx = len(enriched) - 1
        row = enriched.iloc[last_idx]
        close = float(row["Close"])
        self._state.prev_close = close

        for period in ALL_EMA_PERIODS:
            col = f"_ema_{period}"
            value = row.get(col)
            self._state.ema_values[period] = None if pd.isna(value) else float(value)

        if last_idx > 0:
            close_series = enriched["Close"].astype(float)
            deltas = close_series.diff().iloc[1 : last_idx + 1]
            for delta in deltas:
                gain = max(float(delta), 0.0)
                loss = max(-float(delta), 0.0)
                self._state.rsi_gains.append(gain)
                self._state.rsi_losses.append(loss)

        high = enriched["High"].astype(float)
        low = enriched["Low"].astype(float)
        close_series = enriched["Close"].astype(float)
        prev_close = close_series.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        start = max(0, last_idx - ATR_PERIOD + 1)
        for value in tr.iloc[start : last_idx + 1]:
            if pd.notna(value):
                self._state.tr_window.append(float(value))

        vol_series = enriched["Volume"].astype(float)
        vol_start = max(0, last_idx - VOLUME_LOOKBACK + 1)
        for value in vol_series.iloc[vol_start : last_idx + 1]:
            self._state.volume_window.append(float(value))

        ts = row["_timestamp"]
        if pd.notna(ts):
            session_day = self._session_day(pd.Timestamp(ts))
            day_mask = enriched["_timestamp"].apply(
                lambda value: pd.notna(value) and self._session_day(pd.Timestamp(value)) == session_day
            )
            day_slice = enriched.loc[day_mask]
            typical = (
                day_slice["High"].astype(float)
                + day_slice["Low"].astype(float)
                + day_slice["Close"].astype(float)
            ) / 3.0
            volume = day_slice["Volume"].astype(float).fillna(0.0)
            self._state.vwap_session_day = session_day
            self._state.vwap_cum_tpv = float((typical * volume).sum())
            self._state.vwap_cum_volume = float(volume.sum())

        return enriched

    def append_bar(self, frame: pd.DataFrame, bar: int) -> pd.DataFrame:
        """Incrementally extend enriched columns for one new bar."""
        if self.enriched is None or bar == 0:
            return self.seed_from_frame(frame)

        row = frame.iloc[bar]
        open_ = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        volume = float(row.get("Volume") or 0.0)
        timestamp = self._builder._ensure_timestamp(frame).iloc[bar]
        ts = pd.Timestamp(timestamp)

        indicator_row: dict[str, Any] = dict(row)
        indicator_row["_timestamp"] = timestamp

        for period in ALL_EMA_PERIODS:
            alpha = self._ema_alpha(period)
            prev = self._state.ema_values.get(period)
            ema = close if prev is None else alpha * close + (1.0 - alpha) * prev
            self._state.ema_values[period] = ema
            indicator_row[f"_ema_{period}"] = ema

        if self._state.prev_close is not None:
            delta = close - self._state.prev_close
            self._state.rsi_gains.append(max(delta, 0.0))
            self._state.rsi_losses.append(max(-delta, 0.0))
        avg_gain = sum(self._state.rsi_gains) / len(self._state.rsi_gains) if self._state.rsi_gains else 0.0
        avg_loss = sum(self._state.rsi_losses) / len(self._state.rsi_losses) if self._state.rsi_losses else 0.0
        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 0.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        indicator_row["_rsi"] = rsi

        prev_close = self._state.prev_close if self._state.prev_close is not None else close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self._state.tr_window.append(tr)
        atr = sum(self._state.tr_window) / len(self._state.tr_window)
        indicator_row["_atr"] = atr

        session_day = self._session_day(ts)
        if self._state.vwap_session_day != session_day:
            self._state.vwap_session_day = session_day
            self._state.vwap_cum_tpv = 0.0
            self._state.vwap_cum_volume = 0.0
        typical = (high + low + close) / 3.0
        self._state.vwap_cum_tpv += typical * volume
        self._state.vwap_cum_volume += volume
        vwap = (
            self._state.vwap_cum_tpv / self._state.vwap_cum_volume
            if self._state.vwap_cum_volume > 0
            else typical
        )
        indicator_row["_vwap"] = vwap

        self._state.volume_window.append(volume)
        volume_mean = sum(self._state.volume_window) / len(self._state.volume_window)
        indicator_row["_volume_spike"] = volume >= (volume_mean * VOLUME_SPIKE_MULTIPLIER)

        self._state.prev_close = close

        new_row = pd.DataFrame([indicator_row])
        if self.enriched is None or self.enriched.empty:
            self.enriched = new_row
        else:
            self.enriched = pd.concat([self.enriched, new_row], ignore_index=True)
        return self.enriched
