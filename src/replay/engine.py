"""
Historical Replay Engine.

Feeds historical closed candles into the EXISTING RealtimeSignalPipeline.
Does not call BUY_V3 / SELL_V6 directly. Does not duplicate signal generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from src.brokers.websocket_client import NIFTY50_SYMBOL
from src.core.logger import logger
from src.pipeline.realtime_signal_pipeline import RealtimeSignalPipeline
from src.replay.controller import ReplayController, ReplayProgress, ReplayState, parse_speed
from src.replay.data_feed import (
    DEFAULT_HISTORY_CSV,
    HistoricalDataFeed,
    ReplayWindow,
    window_for_day,
    window_for_month,
    window_for_range,
    window_for_week,
)
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase
from src.trade_validation.config import TradeValidationConfig
from src.trade_validation.engine import TradeValidationEngine


@dataclass
class ReplayResult:
    """Summary of a completed (or stopped) replay run."""

    window: ReplayWindow
    warm_start_bars: int
    candles_fed: int
    state: ReplayState
    decisions: int = 0
    signals: int = 0
    validations: int = 0
    progress: ReplayProgress = field(default_factory=ReplayProgress)

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window.start.isoformat(),
            "window_end": self.window.end.isoformat(),
            "warm_start_bars": self.warm_start_bars,
            "candles_fed": self.candles_fed,
            "state": self.state.value,
            "decisions": self.decisions,
            "signals": self.signals,
            "validations": self.validations,
            "last_candle_timestamp": self.progress.last_candle_timestamp,
        }


class ReplayEngine:
    """
    Drive historical candles through RealtimeSignalPipeline.ingest_closed_candle.

    Flow:
      Historical OHLCV → Candle Feed → Existing Realtime Pipeline
        → BUY_V3 / SELL_V6 → Decision Persistence → Signal Persistence
        → Trade Validation Engine
    """

    def __init__(
        self,
        *,
        window: ReplayWindow,
        speed: float | str | int = float("inf"),
        csv_path: Path | str | None = DEFAULT_HISTORY_CSV,
        signal_db_path: Path | str | None = None,
        validation_db_path: Path | str | None = None,
        symbol: str = NIFTY50_SYMBOL,
        run_trade_validation: bool = True,
        validate_every_n_candles: int = 1,
    ) -> None:
        self.window = window
        self.symbol = symbol
        self.run_trade_validation = run_trade_validation
        self.validate_every_n_candles = max(1, validate_every_n_candles)
        self.controller = ReplayController(speed=parse_speed(speed))

        self.feed = HistoricalDataFeed(csv_path=csv_path, symbol=symbol)
        db_path = Path(signal_db_path) if signal_db_path else (
            Path(__file__).resolve().parents[2] / "data" / "paper" / "replay_signals.db"
        )
        self.db = PaperSignalDatabase(db_path)
        self.async_db = AsyncDbWriter(db_path)
        self.pipeline = RealtimeSignalPipeline(
            db=self.db,
            async_db=self.async_db,
            history_csv=None,
            symbol=symbol,
        )

        self.validator: TradeValidationEngine | None = None
        if run_trade_validation:
            val_path = Path(validation_db_path) if validation_db_path else (
                Path(__file__).resolve().parents[2] / "data" / "paper" / "replay_trade_validation.db"
            )
            self.validator = TradeValidationEngine(
                TradeValidationConfig(
                    signal_db_path=db_path,
                    validation_db_path=val_path,
                    default_symbol=symbol,
                ),
            )

        self._validations_total = 0

    def run(self) -> ReplayResult:
        """Execute the replay until complete or stopped."""
        self.feed.load()
        warm_frame = self.feed.warm_start_frame(self.window)
        candles = self.feed.replay_candles(self.window)

        logger.info(
            "Replay start | window=%s..%s | warm=%s | candles=%s | speed=%s",
            self.window.start,
            self.window.end,
            len(warm_frame),
            len(candles),
            self.controller.speed if self.controller.speed != float("inf") else "unlimited",
        )
        print(
            f"[REPLAY] window={self.window.start}..{self.window.end} "
            f"warm={len(warm_frame)} candles={len(candles)} "
            f"speed={'unlimited' if self.controller.speed == float('inf') else f'{self.controller.speed:g}x'}",
            flush=True,
        )

        warm_bars = self.pipeline.warm_start_from_frame(warm_frame) if not warm_frame.empty else 0
        self.controller.start(total_candles=len(candles))

        fed = 0
        for candle in candles:
            self.controller.wait_if_paused()
            if self.controller.should_stop():
                break

            # Same path as live: closed candle → pipeline handler → engines → persistence.
            self.pipeline.ingest_closed_candle(candle)
            fed += 1
            self.controller.record_candle(timestamp=candle.timestamp.isoformat())

            if self.validator is not None and fed % self.validate_every_n_candles == 0:
                self.async_db.flush(timeout_seconds=60.0)
                results = self.validator.run_once()
                self._validations_total += len(results)

            if fed % 50 == 0 or fed == len(candles):
                print(
                    f"[REPLAY] progress={fed}/{len(candles)} "
                    f"({self.controller.progress.pct:.1f}%) "
                    f"last={candle.timestamp.isoformat()} "
                    f"state={self.controller.state.value}",
                    flush=True,
                )

            self.controller.pace()
            if self.controller.should_stop():
                break

        if not self.controller.should_stop():
            self.controller.mark_completed()

        self.async_db.flush(timeout_seconds=60.0)
        if self.validator is not None:
            results = self.validator.run_once()
            self._validations_total += len(results)

        decisions = len(self.db.recent_decisions(symbol=self.symbol, limit=10_000))
        signals = len(self.db.recent_signals(limit=10_000))

        result = ReplayResult(
            window=self.window,
            warm_start_bars=warm_bars,
            candles_fed=fed,
            state=self.controller.state,
            decisions=decisions,
            signals=signals,
            validations=self._validations_total,
            progress=self.controller.progress,
        )
        print(f"[REPLAY COMPLETE] {result.as_dict()}", flush=True)
        logger.info("Replay finished: %s", result.as_dict())
        return result

    def pause(self) -> None:
        self.controller.pause()

    def resume(self) -> None:
        self.controller.resume()

    def stop(self) -> None:
        self.controller.stop()

    def close(self) -> None:
        try:
            self.async_db.flush(timeout_seconds=30.0)
        except Exception:
            logger.exception("Failed to flush async DB during replay close.")
        self.async_db.close()
        self.db.close()
        if self.validator is not None:
            self.validator.close()


def build_engine_for_day(
    day: date,
    *,
    speed: float | str | int = float("inf"),
    **kwargs: Any,
) -> ReplayEngine:
    return ReplayEngine(window=window_for_day(day), speed=speed, **kwargs)


def build_engine_for_week(
    year: int,
    week: int,
    *,
    speed: float | str | int = float("inf"),
    **kwargs: Any,
) -> ReplayEngine:
    return ReplayEngine(window=window_for_week(year, week), speed=speed, **kwargs)


def build_engine_for_month(
    year: int,
    month: int,
    *,
    speed: float | str | int = float("inf"),
    **kwargs: Any,
) -> ReplayEngine:
    return ReplayEngine(window=window_for_month(year, month), speed=speed, **kwargs)


def build_engine_for_range(
    start: date,
    end: date,
    *,
    speed: float | str | int = float("inf"),
    **kwargs: Any,
) -> ReplayEngine:
    return ReplayEngine(window=window_for_range(start, end), speed=speed, **kwargs)
