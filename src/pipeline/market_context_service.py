"""
Market context builder for live bar-by-bar signal evaluation.

Prepares the same enriched frames / intel maps used by production replay audits.
Uses in-memory cache and incremental updates during live append for performance.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.core.logger import logger
from src.pipeline.incremental_indicators import IncrementalIndicatorEngine
from src.pipeline.market_memory_cache import MarketMemoryCache
from src.research.buy_v3_candidate_validation_research import (
    _events_in_lookback_cached,
    _precompute_bar_events,
)
from src.research.nifty50_trap_to_momentum_validation_research import DEFAULT_SYMBOL
from src.research.smartmoneyengine_v4_candidate_validation_research import _attach_ema22
from src.signals.buy_v3 import BuyV3Engine
from src.signals.sell_v6 import SellV6Engine


def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    working = frame.copy()
    working["_dt"] = pd.to_datetime(working["Date"])
    grouped = (
        working.set_index("_dt")
        .resample(rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
        .reset_index()
    )
    grouped["Date"] = grouped["_dt"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return grouped[["Date", "Open", "High", "Low", "Close", "Volume"]]


class MarketContextService:
    """Maintain rolling market frames and per-bar caches for live evaluation."""

    def __init__(self) -> None:
        self.buy_engine = BuyV3Engine()
        self.sell_engine = SellV6Engine()
        self._buy_research = self.buy_engine._engine
        self._sell_research = self.sell_engine._engine
        self.memory = MarketMemoryCache()
        self.indicators = IncrementalIndicatorEngine()
        self.frame: pd.DataFrame | None = None
        self.calendar: pd.DataFrame | None = None
        self.enriched_buy: pd.DataFrame | None = None
        self.enriched_sell: pd.DataFrame | None = None
        self.intel_frames: dict[str, pd.DataFrame] = {}
        self.bar_events_cache: dict[int, set[str]] = {}
        self.lookback_cache: dict[int, set[str]] = {}
        self.buy_context_cache: dict[int, dict[str, str]] = {}
        self._htf_lengths: dict[str, int] = {}
        self._last_incremental_ms: float = 0.0

    @property
    def last_incremental_ms(self) -> float:
        return self._last_incremental_ms

    def load_history(self, frame: pd.DataFrame) -> None:
        """Load historical/backfill candles and rebuild context caches."""
        history = frame.reset_index(drop=True)
        rows = history.to_dict(orient="records")
        self.memory.load_history_rows(rows)
        self.frame = self.memory.as_dataframe()
        self._rebuild_context()

    def append_candle_row(self, row: dict[str, Any]) -> int:
        """Append one closed candle and update context; return latest bar index."""
        bar = self.memory.append_closed_row(row)
        self.frame = self.memory.as_dataframe()
        if bar == 0:
            self._rebuild_context()
            return bar
        self._incremental_context_update(bar)
        return bar

    def _rebuild_context(self) -> None:
        assert self.frame is not None
        started = time.perf_counter()
        frame = self.frame
        self._buy_research.clear_frame_caches()
        self.calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        self.enriched_buy = self.indicators.seed_from_frame(frame)
        self.enriched_sell = _attach_ema22(self.enriched_buy)

        intel_5m = self._buy_research.intelligence.enrich(frame)
        self.intel_frames = {"5M": intel_5m}
        frame_15m = _resample_ohlcv(frame, "15min")
        frame_1h = _resample_ohlcv(frame, "1h")
        self.intel_frames["15M"] = self._buy_research.intelligence.enrich(frame_15m)
        self.intel_frames["1H"] = self._buy_research.intelligence.enrich(frame_1h)
        self.intel_frames["1D"] = self._buy_research.intelligence.enrich(
            self._buy_research._resample_daily(self.intel_frames["1H"]),
        )
        self._htf_lengths = {
            "15M": len(frame_15m),
            "1H": len(frame_1h),
            "1D": len(self.intel_frames["1D"]),
        }
        self._warm_htf_datetime_indexes(frame)

        latest_bar = len(frame) - 1
        warmup_bars = [latest_bar] if latest_bar >= 0 else []
        self.bar_events_cache = {}
        self.lookback_cache = {}
        self.buy_context_cache = {}
        if warmup_bars:
            self.bar_events_cache, self.lookback_cache = _precompute_bar_events(
                self._buy_research,
                frame=frame,
                calendar=self.calendar,
                replay_bars=warmup_bars,
            )
            self.buy_context_cache[latest_bar] = self._buy_research._context_at_bar(
                frame=frame,
                enriched=self.enriched_buy,
                calendar=self.calendar,
                intel_frames=self.intel_frames,
                bar=latest_bar,
            )
        self._last_incremental_ms = (time.perf_counter() - started) * 1000.0
        logger.info("Full context rebuild completed in %.0f ms for %s bars.", self._last_incremental_ms, len(frame))

    def _warm_htf_datetime_indexes(self, frame: pd.DataFrame) -> None:
        """Pre-parse Date columns once for HTF bar mapping during context build."""
        self._buy_research._parsed_frame_dates(frame)
        for intel_frame in self.intel_frames.values():
            self._buy_research._parsed_frame_dates(intel_frame)

    def _incremental_context_update(self, bar: int) -> None:
        assert self.frame is not None
        started = time.perf_counter()
        frame = self.frame
        self._buy_research.clear_frame_caches()

        self.calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        self.enriched_buy = self.indicators.append_bar(frame, bar)
        self.enriched_sell = _attach_ema22(self.enriched_buy)

        self.intel_frames["5M"] = self._buy_research.intelligence.enrich(frame)
        self._refresh_htf_intel_if_needed(frame)
        self._warm_htf_datetime_indexes(frame)

        if bar not in self.bar_events_cache:
            self.bar_events_cache[bar] = set(
                self._buy_research._detect_events_at_bar(frame, self.calendar, bar),
            )
        self.lookback_cache[bar] = _events_in_lookback_cached(
            self._buy_research,
            frame=frame,
            calendar=self.calendar,
            bar=bar,
            bar_events_cache=self.bar_events_cache,
        )
        self.buy_context_cache[bar] = self._buy_research._context_at_bar(
            frame=frame,
            enriched=self.enriched_buy,
            calendar=self.calendar,
            intel_frames=self.intel_frames,
            bar=bar,
        )
        self._last_incremental_ms = (time.perf_counter() - started) * 1000.0

    def _refresh_htf_intel_if_needed(self, frame: pd.DataFrame) -> None:
        frame_15m = _resample_ohlcv(frame, "15min")
        if len(frame_15m) != self._htf_lengths.get("15M", -1):
            self.intel_frames["15M"] = self._buy_research.intelligence.enrich(frame_15m)
            self._htf_lengths["15M"] = len(frame_15m)

        frame_1h = _resample_ohlcv(frame, "1h")
        if len(frame_1h) != self._htf_lengths.get("1H", -1):
            self.intel_frames["1H"] = self._buy_research.intelligence.enrich(frame_1h)
            self._htf_lengths["1H"] = len(frame_1h)
            daily = self._buy_research._resample_daily(self.intel_frames["1H"])
            if len(daily) != self._htf_lengths.get("1D", -1):
                self.intel_frames["1D"] = self._buy_research.intelligence.enrich(daily)
                self._htf_lengths["1D"] = len(daily)

    def evaluate_latest(self) -> tuple[dict[str, Any], dict[str, Any], int]:
        """Evaluate BUY_V3 and SELL_V6 on the latest closed bar."""
        if self.frame is None or self.calendar is None or self.enriched_buy is None:
            raise RuntimeError("Market context not initialized.")
        bar = len(self.frame) - 1
        buy_eval = self.buy_engine.evaluate_bar(
            frame=self.frame,
            bar=bar,
            context=self.buy_context_cache[bar],
            lookback_events=self.lookback_cache[bar],
            bar_events=self.bar_events_cache[bar],
        )
        sell_eval = self.sell_engine.evaluate_bar(
            frame=self.frame,
            enriched=self.enriched_sell,
            calendar=self.calendar,
            intel_frames=self.intel_frames,
            bar=bar,
        )
        return buy_eval, sell_eval, bar
