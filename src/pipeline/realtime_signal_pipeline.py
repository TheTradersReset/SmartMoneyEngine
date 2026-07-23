"""
Real-time FYERS tick → 5M candle → BUY_V3 / SELL_V6 paper signal pipeline.

Paper signal mode only: no orders, no Telegram, no live capital.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.brokers.websocket_client import FyersWebsocketClient, NIFTY50_SYMBOL
from src.core.logger import logger
from src.data.candle_builder import Candle, FiveMinuteCandleBuilder, candles_to_frame_rows, is_trading_session
from src.live_paper.runtime.watermark import WatermarkStore, normalize_timestamp
from src.pipeline.candle_diagnostics import (
    build_candle_report,
    decision_record_from_report,
    emit_candle_report,
    emit_decision_saved_log,
)
from src.pipeline.market_context_service import MarketContextService
from src.pipeline.performance_profiler import CandleProfiler
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS
from src.signals.regime_throttle import RegimeThrottle, STACK_FINGERPRINT, ThrottleDecision
from src.signals.signal_outcome import evaluate_post_signal_outcome, normalize_timestamp_key
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
MIN_BARS_FOR_EVAL = 120


class RealtimeSignalPipelineError(Exception):
    """Raised when the realtime signal pipeline cannot run."""


@dataclass
class _PendingSignalOutcome:
    """Accepted signal awaiting FORWARD_BARS before outcome evaluation."""

    bar: int
    timestamp: str
    direction: str


class RealtimeSignalPipeline:
    """
    Consume FYERS ticks, build 5M candles, evaluate frozen BUY_V3 / SELL_V6 stack.

    Parameters
    ----------
    db : PaperSignalDatabase
        SQLite persistence layer.
    history_csv : Path | None
        Optional historical OHLCV CSV for warm-start context.
    """

    def __init__(
        self,
        *,
        db: PaperSignalDatabase | None = None,
        async_db: AsyncDbWriter | None = None,
        history_csv: Path | str | None = DEFAULT_HISTORY_CSV,
        symbol: str = NIFTY50_SYMBOL,
    ) -> None:
        self.db = db or PaperSignalDatabase()
        self.async_db = async_db or AsyncDbWriter(self.db.db_path)
        self.symbol = symbol
        self.history_csv = Path(history_csv) if history_csv else None
        self.context = MarketContextService()
        self.throttle = RegimeThrottle()
        self.buy_engine = self.context.buy_engine
        self.sell_engine = self.context.sell_engine
        self._same_bar_conflict = False
        self._lock = threading.Lock()
        self._profiler = CandleProfiler()
        self._last_tick_ms: float = 0.0
        self._pending_outcomes: list[_PendingSignalOutcome] = []
        self._candle_builder = FiveMinuteCandleBuilder(
            symbol=symbol,
            on_candle_close=self._on_candle_close,
        )
        self._ws_client: FyersWebsocketClient | None = None
        # Phase 2: optional live-eval queue (unused unless enable_pipeline_v2).
        self.enable_pipeline_v2: bool = False
        self._live_close_queue = None  # LiveCloseQueue | None
        self._live_eval_worker_tls = threading.local()
        # Closed-candle commit watermark (source of truth for applied timestamps).
        self._watermark = WatermarkStore()

    def warm_start_from_history(self) -> int:
        """Load historical candles for context warm-start."""
        if self.history_csv is None or not self.history_csv.exists():
            logger.warning("History CSV missing (%s); starting with empty context.", self.history_csv)
            return 0
        frame = pd.read_csv(self.history_csv)
        if "Date" not in frame.columns:
            raise RealtimeSignalPipelineError(f"History CSV missing Date column: {self.history_csv}")
        tail = frame.tail(2000).copy()
        return self.warm_start_from_frame(tail)

    def warm_start_from_frame(self, frame: pd.DataFrame) -> int:
        """Warm-start market context from an OHLCV DataFrame (identical to CSV warm-start)."""
        if frame.empty:
            return 0
        if "Date" not in frame.columns:
            raise RealtimeSignalPipelineError("History frame missing Date column.")
        working = frame.reset_index(drop=True)
        self.context.load_history(working)
        last_date = working.iloc[-1].get("Date")
        seeded = self._watermark.initialize(last_date)
        if seeded is not None:
            logger.info("Watermark seeded from warm-start: %s", seeded.isoformat())
        self._hydrate_pending_outcomes_from_db()
        self._resolve_pending_outcomes_on_frame()
        logger.info("Warm-started market context with %s historical bars.", len(working))
        return len(working)

    def _hydrate_pending_outcomes_from_db(self) -> None:
        """
        Reload PENDING accepted signals whose decision bar is already in the frame.

        Enables day-by-day replay to resolve outcomes once FORWARD_BARS become
        available in a later session without changing the Replay Engine.

        Identity uses timezone-normalized timestamps (not formatted strings),
        so ``+0530`` and ``+05:30`` map to the same bar.
        """
        frame = self.context.frame
        if frame is None or frame.empty:
            return
        try:
            rows = self.db.pending_signal_outcomes(limit=5_000)
        except Exception:  # noqa: BLE001 — older DBs / tests without helper
            return
        key_to_bar: dict[str, int] = {}
        for i in range(len(frame)):
            key = normalize_timestamp_key(frame.iloc[i].get("Date"))
            if key is not None:
                key_to_bar[key] = i
        existing = {
            (normalize_timestamp_key(p.timestamp) or p.timestamp, p.direction)
            for p in self._pending_outcomes
        }
        hydrated = 0
        for row in rows:
            ts = str(row.get("timestamp", ""))
            direction = str(row.get("direction", "")).upper()
            key = normalize_timestamp_key(ts)
            if key is None:
                continue
            if (key, direction) in existing:
                continue
            bar = key_to_bar.get(key)
            if bar is None:
                continue
            self._pending_outcomes.append(
                _PendingSignalOutcome(bar=bar, timestamp=ts, direction=direction)
            )
            existing.add((key, direction))
            hydrated += 1
        if hydrated:
            logger.info(
                "Hydrated %s pending signal outcome(s) from database (queue=%s).",
                hydrated,
                len(self._pending_outcomes),
            )

    def ingest_closed_candle(self, candle: Candle) -> None:
        """
        Feed a closed candle through the identical live candle-close path.

        Used by Historical Replay Engine. Does not alter BUY_V3 / SELL_V6 logic.
        """
        self._on_candle_close(candle)

    def _on_candle_close(self, candle: Candle) -> None:
        with self._lock:
            self._handle_closed_candle(candle)

    def _is_closed_candle_already_committed(self, candle: Candle) -> bool:
        """
        True when ``candle.timestamp`` must not be appended (invalid or <= watermark).

        WatermarkStore is the runtime source of truth for committed closed bars.
        """
        dt = normalize_timestamp(candle.timestamp)
        if dt is None:
            logger.info(
                "Skipping candle: unparseable timestamp %r",
                candle.timestamp,
            )
            return True
        current = self._watermark.get()
        if current is not None and dt <= current:
            logger.info(
                "Skipping candle: timestamp %s already committed (watermark=%s)",
                dt.isoformat(),
                current.isoformat(),
            )
            return True
        return False

    def _advance_watermark_after_apply(self, candle: Candle) -> None:
        """Advance watermark only after append + evaluate + persist completed."""
        if self._watermark.try_advance(candle.timestamp):
            logger.info("Watermark advanced to %s", self._watermark.as_iso())
        else:
            logger.warning(
                "Watermark did not advance after apply for %s (current=%s)",
                candle.timestamp.isoformat(),
                self._watermark.as_iso(),
            )

    def _handle_closed_candle(self, candle: Candle) -> None:
        if self._should_enqueue_for_live_eval():
            self._enqueue_closed_candle_for_live_eval(candle)
            return

        if not is_trading_session(candle.timestamp):
            logger.info("Skipping candle outside session: %s", candle.timestamp)
            return

        if self._is_closed_candle_already_committed(candle):
            return

        total_started = time.perf_counter()
        self._profiler.reset()
        self._profiler.tick_processing_ms = self._last_tick_ms
        self._profiler.candle_creation_ms = self._candle_builder.last_candle_close_ms
        self.context.memory.set_current_candle(candle)

        print("[CANDLE CLOSED]", flush=True)
        print(f"  timestamp={candle.timestamp.isoformat()}", flush=True)

        sqlite_started = time.perf_counter()
        self.async_db.enqueue(
            "candle",
            {
                "symbol": candle.symbol,
                "timestamp": candle.timestamp.isoformat(),
                "open_": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "tick_count": candle.tick_count,
            },
        )
        self._profiler.sqlite_write_ms += (time.perf_counter() - sqlite_started) * 1000.0

        indicator_started = time.perf_counter()
        row = candles_to_frame_rows([candle])[0]
        bar = self.context.append_candle_row(row)
        self._profiler.indicator_calculation_ms = (time.perf_counter() - indicator_started) * 1000.0
        self._profiler.extra["context_update_ms"] = self.context.last_incremental_ms

        if self.context.frame is None or len(self.context.frame) < MIN_BARS_FOR_EVAL:
            frame_len = 0 if self.context.frame is None else len(self.context.frame)
            logger.info(
                "Insufficient bars for signal eval (%s/%s).",
                frame_len,
                MIN_BARS_FOR_EVAL,
            )
            snapshot = self._build_context_snapshot(bar)
            decision_started = time.perf_counter()
            report = build_candle_report(
                candle=candle,
                bar=bar,
                buy_eval=None,
                sell_eval=None,
                eval_ms=0.0,
                context_snapshot=snapshot,
                skipped_reason=f"INSUFFICIENT_BARS ({frame_len}/{MIN_BARS_FOR_EVAL})",
            )
            self._profiler.decision_persistence_ms = (time.perf_counter() - decision_started) * 1000.0
            self._persist_decision(candle, report)
            emit_candle_report(report, logger=logger)
            self._finalize_profiler(candle, total_started=total_started)
            self._advance_watermark_after_apply(candle)
            return

        signal_started = time.perf_counter()
        buy_eval, sell_eval, bar = self.context.evaluate_latest()
        self._profiler.signal_engine_ms = (time.perf_counter() - signal_started) * 1000.0
        snapshot = self._build_context_snapshot(bar)

        buy_signal = self.buy_engine.to_signal(buy_eval)
        sell_signal = self.sell_engine.to_signal(sell_eval)
        self._same_bar_conflict = buy_signal is not None and sell_signal is not None

        candidates: list[tuple[str, Any, dict[str, Any]]] = []
        if buy_signal is not None:
            candidates.append(("BUY", buy_signal, buy_eval))
        if sell_signal is not None:
            candidates.append(("SELL", sell_signal, sell_eval))

        buy_throttle: ThrottleDecision | None = None
        sell_throttle: ThrottleDecision | None = None
        buy_accepted: bool | None = None
        sell_accepted: bool | None = None

        if self._same_bar_conflict:
            logger.warning("Same-bar BUY+SELL conflict at %s — rejecting both.", candle.timestamp)
            sqlite_started = time.perf_counter()
            self.async_db.enqueue(
                "event",
                {
                    "signal_id": None,
                    "event_type": "SAME_BAR_CONFLICT",
                    "details": {
                        "timestamp": candle.timestamp.isoformat(),
                        "buy_verdict": buy_eval.get("verdict"),
                        "sell_verdict": sell_eval.get("verdict"),
                    },
                },
            )
            self._profiler.sqlite_write_ms += (time.perf_counter() - sqlite_started) * 1000.0
            for direction, signal_obj, evaluation in candidates:
                throttle = self.throttle.apply(direction=direction, evaluation=evaluation)
                if direction == "BUY":
                    buy_throttle = throttle
                    buy_accepted = False
                else:
                    sell_throttle = throttle
                    sell_accepted = False
                self._persist_signal(
                    signal_obj,
                    evaluation=evaluation,
                    accepted=False,
                    rejection_reason="SAME_BAR_CONFLICT",
                    throttle_level="BLOCK",
                    regime=self.throttle.composite_key(evaluation),
                )
        else:
            for direction, signal_obj, evaluation in candidates:
                decision = self.throttle.apply(direction=direction, evaluation=evaluation)
                accepted = decision.accepted
                rejection = decision.rejection_reason
                if direction == "BUY":
                    buy_throttle = decision
                    buy_accepted = accepted
                else:
                    sell_throttle = decision
                    sell_accepted = accepted
                if accepted:
                    if direction == "BUY":
                        self.buy_engine.mark_emitted(bar)
                    else:
                        self.sell_engine.mark_emitted(bar)
                    self._pending_outcomes.append(
                        _PendingSignalOutcome(
                            bar=bar,
                            timestamp=str(signal_obj.timestamp),
                            direction=direction,
                        )
                    )
                self._persist_signal(
                    signal_obj,
                    evaluation=evaluation,
                    accepted=accepted,
                    rejection_reason=rejection,
                    throttle_level=decision.throttle_level,
                    regime=decision.composite_regime,
                )
                event_type = "SIGNAL_ACCEPTED" if accepted else "SIGNAL_REJECTED"
                logger.info(
                    "%s %s %s throttle=%s regime=%s",
                    event_type,
                    direction,
                    signal_obj.timestamp,
                    decision.throttle_level,
                    decision.composite_regime,
                )

        self._resolve_pending_outcomes(candle)

        decision_started = time.perf_counter()
        report = build_candle_report(
            candle=candle,
            bar=bar,
            buy_eval=buy_eval,
            sell_eval=sell_eval,
            eval_ms=self._profiler.signal_engine_ms,
            context_snapshot=snapshot,
            same_bar_conflict=self._same_bar_conflict,
            buy_throttle=buy_throttle,
            sell_throttle=sell_throttle,
            buy_accepted=buy_accepted,
            sell_accepted=sell_accepted,
        )
        self._profiler.decision_persistence_ms = (time.perf_counter() - decision_started) * 1000.0
        self._persist_decision(candle, report)
        emit_candle_report(report, logger=logger)
        self._finalize_profiler(candle, total_started=total_started)
        self._advance_watermark_after_apply(candle)

    def _finalize_profiler(self, candle: Candle, *, total_started: float) -> None:
        self._profiler.total_processing_ms = (time.perf_counter() - total_started) * 1000.0
        self._profiler.print_report(timestamp=candle.timestamp.isoformat(), logger=logger)

    def _build_context_snapshot(self, bar: int) -> dict[str, Any]:
        """Collect read-only market context fields for candle diagnostics."""
        frame = self.context.frame
        enriched = self.context.enriched_buy
        assert frame is not None and enriched is not None

        row = enriched.iloc[bar]
        discovery = self.context.buy_engine._engine.discovery
        levels = discovery._market_levels(frame, bar)
        bar_events = self.context.bar_events_cache.get(bar, set())
        lookback_events = self.context.lookback_cache.get(bar, set())
        buy_context = self.context.buy_context_cache.get(bar, {})

        intel_5m = self.context.intel_frames.get("5M")
        trend_state = "N/A"
        if intel_5m is not None and bar < len(intel_5m):
            trend_state = str(intel_5m.iloc[bar].get("trend_state", "N/A"))

        regime_composite = self.throttle.composite_key({"layer2": {"htf_trend": buy_context.get("htf_trend")}})

        def _sf(col: str) -> float | None:
            value = row.get(col)
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            if parsed != parsed:
                return None
            return parsed

        return {
            "buy_context": buy_context,
            "sell_context": buy_context,
            "bar_events": bar_events,
            "lookback_events": lookback_events,
            "trend_state": trend_state,
            "regime_composite": regime_composite,
            "rsi": _sf("_rsi"),
            "atr": _sf("_atr"),
            "ema20": _sf("_ema_20"),
            "ema50": _sf("_ema_50"),
            "ema200": _sf("_ema_200"),
            "support_zone": levels.get("major_support"),
            "resistance_zone": levels.get("major_resistance"),
        }

    def _persist_decision(self, candle: Candle, report: dict[str, Any]) -> None:
        """Queue exactly one signal_decisions row for a closed candle."""
        record = decision_record_from_report(candle=candle, symbol=self.symbol, report=report)
        sqlite_started = time.perf_counter()
        self.async_db.enqueue("signal_decision", record)
        self._profiler.sqlite_write_ms += (time.perf_counter() - sqlite_started) * 1000.0
        emit_decision_saved_log(decision=str(record["decision"]), logger=logger)

    def _persist_signal(
        self,
        signal_obj: Any,
        *,
        evaluation: dict[str, Any],
        accepted: bool,
        rejection_reason: str | None,
        throttle_level: str,
        regime: str,
    ) -> None:
        payload = signal_obj.as_dict()
        layer4 = (evaluation or {}).get("layer4") or {}
        record = {
            "timestamp": payload["timestamp"],
            "direction": payload["direction"],
            "engine_version": payload["engine_version"],
            "entry": payload["entry"],
            "stop": payload["stop"],
            "target1": payload["target1"],
            "target2": payload["target2"],
            "target_structure": payload["target_structure"],
            "confidence": payload["confidence"],
            "regime": regime,
            "throttle_level": throttle_level,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "stack_fingerprint": STACK_FINGERPRINT,
            "evaluation": evaluation,
            "risk": layer4.get("risk_points"),
            "target": layer4.get("target_1"),
            "outcome": "PENDING" if accepted else "REJECTED",
            "holding_bars": None,
            "outcome_timestamp": None,
            "reward": None,
        }
        sqlite_started = time.perf_counter()
        self.async_db.enqueue("signal", record)
        self.async_db.enqueue(
            "event",
            {
                "signal_id": None,
                "event_type": "SIGNAL_GENERATED",
                "details": {"accepted": accepted, "rejection_reason": rejection_reason},
            },
        )
        self._profiler.sqlite_write_ms += (time.perf_counter() - sqlite_started) * 1000.0
        print(json.dumps({"signal": record}, default=str), flush=True)

    def _resolve_pending_outcomes(self, candle: Candle) -> None:
        """Resolve after FORWARD_BARS; outcome_timestamp = completing candle."""
        self._resolve_pending_outcomes_at(outcome_timestamp=candle.timestamp.isoformat())

    def _resolve_pending_outcomes_on_frame(self) -> None:
        """Resolve any hydrated pendings already past FORWARD_BARS in the frame."""
        frame = self.context.frame
        if frame is None or frame.empty:
            return
        outcome_ts = str(frame.iloc[-1].get("Date", ""))
        self._resolve_pending_outcomes_at(outcome_timestamp=outcome_ts)

    def _resolve_pending_outcomes_at(self, *, outcome_timestamp: str) -> None:
        """
        After FORWARD_BARS have elapsed, evaluate stored signals via ``_trade_outcome``.

        Decision timestamp is unchanged; outcome_timestamp is the completing bar time.
        """
        frame = self.context.frame
        if frame is None or not self._pending_outcomes:
            return
        current_bar = len(frame) - 1
        remaining: list[_PendingSignalOutcome] = []
        for pending in self._pending_outcomes:
            held = current_bar - pending.bar
            if held < FORWARD_BARS:
                remaining.append(pending)
                continue
            engine = (
                self.buy_engine._engine
                if pending.direction == "BUY"
                else self.sell_engine._engine
            )
            update = evaluate_post_signal_outcome(
                engine,
                frame=frame,
                signal_bar=pending.bar,
                direction=pending.direction,
                decision_timestamp=pending.timestamp,
                outcome_timestamp=outcome_timestamp,
                forward_bars=FORWARD_BARS,
            )
            if update is None:
                remaining.append(pending)
                continue
            sqlite_started = time.perf_counter()
            self.async_db.enqueue(
                "signal_outcome",
                {
                    "timestamp": update.decision_timestamp,
                    "direction": update.direction,
                    "entry": update.entry,
                    "stop": update.stop,
                    "target": update.target,
                    "risk": update.risk,
                    "reward": update.reward,
                    "outcome": update.outcome,
                    "holding_bars": update.holding_bars,
                    "outcome_timestamp": update.outcome_timestamp,
                    "forward_outcome": update.forward_outcome,
                },
            )
            self.async_db.enqueue(
                "event",
                {
                    "signal_id": None,
                    "event_type": "SIGNAL_OUTCOME",
                    "details": {
                        "timestamp": update.decision_timestamp,
                        "direction": update.direction,
                        "outcome": update.outcome,
                        "outcome_timestamp": update.outcome_timestamp,
                        "holding_bars": update.holding_bars,
                        "reward": update.reward,
                        "risk": update.risk,
                    },
                },
            )
            self._profiler.sqlite_write_ms += (time.perf_counter() - sqlite_started) * 1000.0
            logger.info(
                "SIGNAL_OUTCOME %s %s outcome=%s reward=%s holding_bars=%s",
                update.direction,
                update.decision_timestamp,
                update.outcome,
                update.reward,
                update.holding_bars,
            )
        self._pending_outcomes = remaining

    def _is_live_eval_worker_thread(self) -> bool:
        """True only on the thread currently executing LiveEvalWorker evaluate."""
        return bool(getattr(self._live_eval_worker_tls, "active", False))

    def _should_enqueue_for_live_eval(self) -> bool:
        """True when pipeline v2 must defer evaluation to LiveEvalWorker."""
        return (
            bool(self.enable_pipeline_v2)
            and self._live_close_queue is not None
            and not self._is_live_eval_worker_thread()
        )

    def _enqueue_closed_candle_for_live_eval(self, candle: Candle) -> None:
        """Producer-side enqueue; never evaluates on the calling thread."""
        from src.live_paper.runtime.live_close_queue import ClosedCandleEvent

        event = ClosedCandleEvent(
            symbol=str(candle.symbol),
            timestamp=candle.timestamp,
            candle={
                "symbol": candle.symbol,
                "timestamp": candle.timestamp.isoformat(),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "tick_count": candle.tick_count,
            },
        )
        self._live_close_queue.put(event)

    def _candle_from_closed_event(self, event: Any) -> Candle:
        """Rebuild a Candle from a ClosedCandleEvent (or compatible object)."""
        payload = getattr(event, "candle", None) or {}
        ts = getattr(event, "timestamp", None)
        return Candle(
            symbol=str(payload.get("symbol") or getattr(event, "symbol", self.symbol)),
            timestamp=ts,
            open=float(payload.get("open", 0.0)),
            high=float(payload.get("high", 0.0)),
            low=float(payload.get("low", 0.0)),
            close=float(payload.get("close", 0.0)),
            volume=float(payload.get("volume", 0.0)),
            tick_count=int(payload.get("tick_count", 0) or 0),
        )

    def _evaluate_queued_closed_candle(self, event: Any) -> None:
        """
        LiveEvalWorker callback: apply closed candle on the worker thread.

        Marks this thread via thread-local so ``_handle_closed_candle`` runs the
        existing evaluate path (including ``context.evaluate_latest()``) instead
        of re-enqueueing. Worker never calls BUY_V3 / SELL_V6 directly.
        """
        candle = self._candle_from_closed_event(event)
        self._live_eval_worker_tls.active = True
        try:
            with self._lock:
                self._handle_closed_candle(candle)
        finally:
            self._live_eval_worker_tls.active = False

    def on_ws_message(self, message: Any) -> None:
        """FYERS websocket tick handler."""
        tick_started = time.perf_counter()
        self._candle_builder.ingest_message(message)
        self._last_tick_ms = (time.perf_counter() - tick_started) * 1000.0

    def run(self) -> None:
        """Start websocket stream and process ticks until stopped."""
        warmed = self.warm_start_from_history()
        logger.info(
            "Starting realtime signal pipeline | symbol=%s | warmed_bars=%s | mode=paper_signal_only",
            self.symbol,
            warmed,
        )
        print(f"Realtime signal pipeline (paper only) | stack={STACK_FINGERPRINT}", flush=True)

        client = FyersWebsocketClient.from_env(symbols=[self.symbol])
        original_on_message = client.on_message

        def _wrapped(message: Any) -> None:
            original_on_message(message)
            self.on_ws_message(message)

        client.on_message = _wrapped
        self._ws_client = client
        client.run()

    def stop(self) -> None:
        if self._ws_client is not None:
            self._ws_client.request_stop()
        self._candle_builder.flush()
        self.async_db.close()
        self.db.close()


def main() -> int:
    pipeline = RealtimeSignalPipeline()
    try:
        pipeline.run()
        return 0
    except KeyboardInterrupt:
        pipeline.stop()
        return 0
    except Exception as exc:
        logger.exception("Realtime signal pipeline failed.")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
