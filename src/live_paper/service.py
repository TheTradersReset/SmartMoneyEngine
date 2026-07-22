"""
Live paper trading service orchestration.

Paper mode only — never calls broker order placement APIs.
"""

from __future__ import annotations

import signal
import sys
import threading
from typing import Any

from src.core.logger import logger
from src.live_paper.config import LivePaperConfig, load_config
from src.live_paper.health import ConnectionHealthMonitor
from src.live_paper.logging_setup import setup_live_paper_logging
from src.live_paper.metrics import LiveMetrics
from src.live_paper.pipeline_ext import LivePaperPipeline
from src.notifications.email import EmailNotifier
from src.notifications.telegram import TelegramNotifier
from src.paper_trading.trade_manager import PaperTradeManager
from src.storage.async_db_writer import AsyncDbWriter
from src.storage.sqlite import PaperSignalDatabase


class LivePaperTradingService:
    """Wire config, metrics, pipeline, email, and optional dashboard."""

    def __init__(self, config: LivePaperConfig | None = None) -> None:
        self.config = config or load_config()
        self.config.assert_paper_mode()
        setup_live_paper_logging()
        self.metrics = LiveMetrics()
        self.health = ConnectionHealthMonitor(
            self.metrics,
            stale_tick_seconds=self.config.stale_tick_seconds,
            heartbeat_seconds=self.config.heartbeat_seconds,
        )
        self.db = PaperSignalDatabase(self.config.db_path)
        self.async_db = AsyncDbWriter(self.config.db_path)
        self.trade_manager = PaperTradeManager(self.db)
        self.email = EmailNotifier(
            smtp_host=self.config.smtp_host,
            smtp_port=self.config.smtp_port,
            smtp_user=self.config.smtp_user,
            smtp_password=self.config.smtp_password,
            smtp_from=self.config.smtp_from,
            smtp_to=self.config.smtp_to,
            use_tls=self.config.smtp_use_tls,
            use_ssl=self.config.smtp_use_ssl,
            enabled=self.config.enable_email,
        )
        # Telegram kept for optional legacy; default disabled (corporate networks).
        self.telegram = TelegramNotifier(
            bot_token=self.config.telegram_bot_token,
            chat_id=self.config.telegram_chat_id,
            enabled=self.config.enable_telegram,
        )
        self.pipeline = LivePaperPipeline(
            config=self.config,
            metrics=self.metrics,
            health=self.health,
            email=self.email,
            trade_manager=self.trade_manager,
            telegram=self.telegram,
            db=self.db,
            async_db=self.async_db,
            history_csv=self.config.history_csv,
            symbol=self.config.symbol,
            dashboard_starter=self._start_dashboard_thread if self.config.enable_dashboard else None,
        )
        self._dashboard_thread: threading.Thread | None = None
        self._uvicorn_server: Any | None = None

    def _start_dashboard_thread(self) -> None:
        if self._dashboard_thread is not None and self._dashboard_thread.is_alive():
            return

        def _run() -> None:
            try:
                import uvicorn

                from src.live_paper.dashboard.app import create_app

                app = create_app(
                    metrics=self.metrics,
                    trade_manager=self.trade_manager,
                    db=self.db,
                )
                config = uvicorn.Config(
                    app,
                    host=self.config.dashboard_host,
                    port=self.config.dashboard_port,
                    log_level="info",
                )
                server = uvicorn.Server(config)
                self._uvicorn_server = server
                logger.info(
                    "Dashboard listening on http://%s:%s",
                    self.config.dashboard_host,
                    self.config.dashboard_port,
                )
                server.run()
            except Exception:
                logger.exception("Dashboard failed to start.")
                self.metrics.record_error("dashboard_failed")

        self._dashboard_thread = threading.Thread(target=_run, name="live-paper-dashboard", daemon=True)
        self._dashboard_thread.start()

    def run(self) -> int:
        logger.info(
            "Live paper trading service starting (symbol=%s db=%s capital_mode=%s pipeline_v2=%s)",
            self.config.symbol,
            self.config.db_path,
            self.config.capital_mode,
            self.config.enable_pipeline_v2,
        )

        def _shutdown(signum: int, _frame: Any) -> None:
            print(f"\nReceived signal {signum}; shutting down live paper...", flush=True)
            self.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _shutdown)
            except (ValueError, OSError):
                pass

        try:
            self.pipeline.run()
            return 0
        except KeyboardInterrupt:
            self.stop()
            return 0
        except Exception as exc:
            logger.exception("Live paper trading service failed.")
            print(f"Error: {exc}", file=sys.stderr)
            self.stop()
            return 1

    def stop(self) -> None:
        try:
            if self._uvicorn_server is not None:
                self._uvicorn_server.should_exit = True
        except Exception:  # noqa: BLE001
            pass
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error while stopping pipeline.")


def main() -> int:
    """Entry point used by ``python -m src.live_paper``."""
    service = LivePaperTradingService()
    return service.run()


if __name__ == "__main__":
    sys.exit(main())
