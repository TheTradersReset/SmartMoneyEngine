"""
Live paper extensions of ``RealtimeSignalPipeline`` (subclass only).

Does not copy strategy logic from BUY_V3 / SELL_V6. Adds health hooks,
latency measurement, email alerts, trade-manager updates, OHLC checks,
reconnect logging, and missed-candle recovery.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.brokers.websocket_client import FyersWebsocketClient
from src.core.logger import logger
from src.data.candle_builder import Candle
from src.live_paper.config import LivePaperConfig
from src.live_paper.health import ConnectionHealthMonitor, market_status_label
from src.live_paper.logging_setup import get_logger
from src.live_paper.metrics import LiveMetrics
from src.live_paper.recovery import MissedCandleRecovery
from src.live_paper.runtime.live_close_queue import LiveCloseQueue
from src.live_paper.runtime.live_eval_worker import LiveEvalWorker
from src.live_paper.runtime.phases import RuntimePhase, RuntimePhaseController
from src.notifications.email import EmailNotifier
from src.notifications.telegram import TelegramNotifier
from src.paper_trading.trade_manager import PaperTradeManager
from src.pipeline.realtime_signal_pipeline import RealtimeSignalPipeline
from src.signals.regime_throttle import STACK_FINGERPRINT
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase

IST = ZoneInfo("Asia/Kolkata")


def validate_ohlc(candle: Candle) -> list[str]:
    """Return human-readable OHLC integrity issues (empty if valid)."""
    issues: list[str] = []
    if candle.high < candle.low:
        issues.append("high < low")
    if candle.high < candle.open:
        issues.append("high < open")
    if candle.high < candle.close:
        issues.append("high < close")
    if candle.low > candle.open:
        issues.append("low > open")
    if candle.low > candle.close:
        issues.append("low > close")
    return issues


class LivePaperPipeline(RealtimeSignalPipeline):
    """RealtimeSignalPipeline subclass with live-paper operational hooks."""

    def __init__(
        self,
        *,
        config: LivePaperConfig,
        metrics: LiveMetrics,
        health: ConnectionHealthMonitor,
        email: EmailNotifier,
        trade_manager: PaperTradeManager,
        telegram: TelegramNotifier | None = None,
        db: PaperSignalDatabase | None = None,
        async_db: AsyncDbWriter | None = None,
        history_csv: Path | str | None = None,
        symbol: str | None = None,
        dashboard_starter: Any | None = None,
    ) -> None:
        super().__init__(
            db=db,
            async_db=async_db,
            history_csv=history_csv if history_csv is not None else config.history_csv,
            symbol=symbol or config.symbol,
        )
        self.config = config
        self.metrics = metrics
        self.health = health
        self.email = email
        self.telegram = telegram
        self.trade_manager = trade_manager
        self.dashboard_starter = dashboard_starter
        self.recovery = MissedCandleRecovery(self, metrics)
        self._candle_close_perf: float | None = None
        self._signal_log = get_logger("signal")
        self._candle_log = get_logger("candle")
        self._db_log = get_logger("database")
        self._ws_log = get_logger("websocket")
        self._reconnect_log = get_logger("reconnect")
        self._latency_warn_ms = float(config.latency_warn_ms)
        self._phase_controller: RuntimePhaseController | None = None
        self._live_close_queue: LiveCloseQueue | None = None
        self._live_eval_worker: LiveEvalWorker | None = None

    def _on_candle_close(self, candle: Candle) -> None:
        self._candle_close_perf = time.perf_counter()
        issues = validate_ohlc(candle)
        if issues:
            msg = f"Invalid OHLC at {candle.timestamp.isoformat()}: {', '.join(issues)}"
            self._candle_log.error("%s", msg)
            self.metrics.record_error(msg)
            # Still process — minor integrity issues should not drop the bar.
        self.metrics.set_current_candle(
            {
                "timestamp": candle.timestamp.isoformat(),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
        )
        self._candle_log.info(
            "Candle close %s O=%.2f H=%.2f L=%.2f C=%.2f",
            candle.timestamp.isoformat(),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
        )
        super()._on_candle_close(candle)

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
        super()._persist_signal(
            signal_obj,
            evaluation=evaluation,
            accepted=accepted,
            rejection_reason=rejection_reason,
            throttle_level=throttle_level,
            regime=regime,
        )
        if not accepted:
            return

        latency_ms = None
        if self._candle_close_perf is not None:
            latency_ms = (time.perf_counter() - self._candle_close_perf) * 1000.0
            self.metrics.record_latency(latency_ms)
            if latency_ms > self._latency_warn_ms:
                self._signal_log.warning(
                    "Signal latency %.1f ms exceeds target %.1f ms",
                    latency_ms,
                    self._latency_warn_ms,
                )
            else:
                self._signal_log.info("Signal latency %.1f ms", latency_ms)

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
            "risk": layer4.get("risk_points") or abs(float(payload["entry"]) - float(payload["stop"])),
            "accepted": True,
        }
        try:
            self.async_db.flush(timeout_seconds=2.0)
        except Exception:  # noqa: BLE001
            pass
        signal_id = self.trade_manager.get_signal_id(record["timestamp"], record["direction"])
        record["id"] = signal_id
        trade = self.trade_manager.on_signal(record)
        self._refresh_trade_metrics()
        self._signal_log.info(
            "Accepted %s %s id=%s latency_ms=%s",
            record["direction"],
            record["timestamp"],
            signal_id,
            f"{latency_ms:.1f}" if latency_ms is not None else "n/a",
        )
        self._db_log.info("Persisted accepted signal id=%s", signal_id)
        if self.config.enable_email:
            self.email.notify_signal(
                symbol=self.symbol,
                direction=str(record["direction"]),
                entry=float(record["entry"]),
                stop=float(record["stop"]),
                target1=float(record["target1"]),
                target2=float(record["target2"]),
                risk=float(record["risk"]),
                timestamp=str(record["timestamp"]),
                strategy_version=str(record["engine_version"]),
                signal_id=signal_id if signal_id is not None else (trade.signal_id if trade else None),
                latency_ms=latency_ms,
                target3="Runner",
            )

    def _resolve_pending_outcomes_at(self, *, outcome_timestamp: str) -> None:
        before = {
            (p.timestamp, p.direction): p for p in list(self._pending_outcomes)
        }
        super()._resolve_pending_outcomes_at(outcome_timestamp=outcome_timestamp)
        after_keys = {(p.timestamp, p.direction) for p in self._pending_outcomes}
        resolved_keys = [key for key in before if key not in after_keys]
        if not resolved_keys:
            return
        try:
            self.async_db.flush(timeout_seconds=2.0)
        except Exception:  # noqa: BLE001
            pass
        recent = self.db.recent_signals(limit=500)
        by_key = {
            (str(row.get("timestamp")), str(row.get("direction", "")).upper()): row
            for row in recent
        }
        for timestamp, direction in resolved_keys:
            row = by_key.get((timestamp, direction.upper()))
            if row is None:
                # timezone-normalized fallback
                for key, candidate in by_key.items():
                    if key[1] == direction.upper() and str(candidate.get("outcome", "")).upper() not in {
                        "",
                        "PENDING",
                        "REJECTED",
                    }:
                        if str(candidate.get("timestamp")) == timestamp:
                            row = candidate
                            break
            if row is None:
                continue
            outcome = str(row.get("outcome") or "")
            reward = row.get("reward")
            risk = row.get("risk")
            try:
                reward_f = float(reward) if reward is not None else None
            except (TypeError, ValueError):
                reward_f = None
            try:
                risk_f = float(risk) if risk not in (None, 0, 0.0) else None
            except (TypeError, ValueError):
                risk_f = None
            r_multiple = None
            if reward_f is not None and risk_f:
                r_multiple = reward_f / risk_f
            signal_id = row.get("id")
            self.trade_manager.on_outcome(
                timestamp=timestamp,
                direction=direction,
                outcome=outcome,
                reward=reward_f,
                holding_bars=row.get("holding_bars"),
                outcome_timestamp=row.get("outcome_timestamp") or outcome_timestamp,
                signal_id=int(signal_id) if signal_id is not None else None,
            )
            self._refresh_trade_metrics()
            self._signal_log.info(
                "Outcome %s %s outcome=%s reward=%s",
                direction,
                timestamp,
                outcome,
                reward_f,
            )
            if self.config.enable_email:
                self.email.notify_outcome(
                    direction=direction,
                    outcome=outcome,
                    pnl_points=reward_f,
                    r_multiple=r_multiple,
                    holding_bars=row.get("holding_bars"),
                    exit_timestamp=str(row.get("outcome_timestamp") or outcome_timestamp),
                    signal_id=signal_id,
                )

    def _refresh_trade_metrics(self) -> None:
        stats = self.trade_manager.stats()
        self.metrics.update_trading_stats(
            today_signals=int(stats["today_signals"]),
            open_trades=int(stats["open_trades"]),
            closed_trades=int(stats["closed_trades"]),
            win_rate=float(stats["win_rate"]),
            running_pnl=float(stats["running_pnl"]),
            equity_curve=list(stats["equity_curve"]),
        )

    def on_ws_message(self, message: Any) -> None:
        self.health.record_tick()
        super().on_ws_message(message)

    def _ping_db(self) -> bool:
        try:
            self.db.recent_signals(limit=1)
            self.metrics.set_system(db_ok=True)
            return True
        except Exception as exc:  # noqa: BLE001
            self.metrics.set_system(db_ok=False)
            self.metrics.record_error(f"db_ping_failed: {exc}")
            self._db_log.error("DB ping failed: %s", exc)
            return False

    def _trigger_recovery(self, reason: str) -> None:
        self._reconnect_log.info("Recovery trigger: %s", reason)
        last = self.metrics.last_candle_ts
        if last is None and self.context.frame is not None and not self.context.frame.empty:
            last = str(self.context.frame.iloc[-1].get("Date"))
        try:
            fed = self.recovery.maybe_recover(last_candle_ts=last)
            self._reconnect_log.info("Recovery fed=%s reason=%s", fed, reason)
        except Exception as exc:  # noqa: BLE001
            self.health.record_error(f"recovery_failed: {exc}")

    def run(self) -> None:
        """Warm-start, start helpers, connect websocket with reconnect hooks."""
        warmed = self.warm_start_from_history()
        logger.info(
            "Starting live paper pipeline | symbol=%s | warmed_bars=%s | stack=%s | mode=paper",
            self.symbol,
            warmed,
            STACK_FINGERPRINT,
        )
        print(f"Live paper trading (paper only) | stack={STACK_FINGERPRINT}", flush=True)

        self.metrics.set_system(market_status=market_status_label())
        self._ping_db()
        self._refresh_trade_metrics()

        if self.dashboard_starter is not None:
            try:
                self.dashboard_starter()
            except Exception as exc:  # noqa: BLE001
                self.health.record_error(f"dashboard_start_failed: {exc}")

        self.health.on_stale = lambda: self._trigger_recovery("stale_ticks")
        self.health.start_heartbeat()

        if self.config.enable_email:
            self.email.send_message(
                "SmartMoneyEngine live paper started",
                f"SmartMoneyEngine live paper started\n"
                f"Symbol: {self.symbol}\n"
                f"Mode: paper\n"
                f"Stack: {STACK_FINGERPRINT}",
            )

        self._start_pipeline_v2_runtime()

        client = FyersWebsocketClient.from_env(symbols=[self.symbol])
        original_on_message = client.on_message
        original_on_open = client.on_open
        original_on_close = client.on_close
        original_on_error = client.on_error

        def _on_message(message: Any) -> None:
            original_on_message(message)
            self.on_ws_message(message)

        def _on_open() -> None:
            was_reconnect = client.reconnect_attempts > 1
            self.health.record_open()
            self._ws_log.info("WS open (attempt=%s)", client.reconnect_attempts)
            original_on_open()
            if was_reconnect:
                self.health.record_reconnect("ws_reconnected")
                self._trigger_recovery("reconnect")

        def _on_close(message: Any) -> None:
            self.health.record_close(str(message))
            original_on_close(message)

        def _on_error(message: Any) -> None:
            self.health.record_error(str(message))
            original_on_error(message)

        client.on_message = _on_message
        client.on_open = _on_open
        client.on_close = _on_close
        client.on_error = _on_error
        self._ws_client = client
        client.run()

    def _start_pipeline_v2_runtime(self) -> None:
        """Create phase controller, queue, and LiveEvalWorker when v2 is enabled."""
        if not self.config.enable_pipeline_v2:
            self.enable_pipeline_v2 = False
            self._live_close_queue = None
            return

        self._phase_controller = RuntimePhaseController()
        self._phase_controller.transition_to(RuntimePhase.WARM_START)
        self._phase_controller.transition_to(RuntimePhase.WATERMARK_SET)
        self._phase_controller.transition_to(RuntimePhase.LIVE)

        self._live_close_queue = LiveCloseQueue(maxsize=int(self.config.live_close_queue_max))
        self.enable_pipeline_v2 = True

        self._live_eval_worker = LiveEvalWorker(
            self._live_close_queue,
            evaluate_fn=self._evaluate_queued_closed_candle,
        )
        self._live_eval_worker.start()
        logger.info(
            "Pipeline v2 live-eval worker started (queue_max=%s)",
            self.config.live_close_queue_max,
        )

    def _shutdown_pipeline_v2_runtime(self) -> None:
        """Close queue, drain via worker, then join worker."""
        if self._phase_controller is not None:
            self._phase_controller.request_shutdown()

        if self._live_close_queue is not None:
            self._live_close_queue.close()

        if self._live_eval_worker is not None:
            self._live_eval_worker.request_stop()
            drain_sec = float(self.config.shutdown_drain_sec)
            self._live_eval_worker.join(timeout=drain_sec)
            if self._live_eval_worker.is_alive:
                logger.warning("LiveEvalWorker did not exit within %.1fs", drain_sec)

        self.enable_pipeline_v2 = False

    def stop(self) -> None:
        self.health.stop()
        # Shutdown order: stop websocket first, then phase/queue/worker drain.
        if self._ws_client is not None:
            try:
                self._ws_client.request_stop()
            except Exception:  # noqa: BLE001
                pass
        self._shutdown_pipeline_v2_runtime()
        try:
            self._candle_builder.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.async_db.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.db.close()
        except Exception:  # noqa: BLE001
            pass
