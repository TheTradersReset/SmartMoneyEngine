"""
Component loggers for the live paper trading service.

Writes structured lines to ``logs/live_paper/*.log`` while keeping the main
engine logger (``src.core.logger``) intact for ``logs/engine.log``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.logger import logger as engine_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs" / "live_paper"

COMPONENT_FILES = {
    "websocket": "websocket.log",
    "candle": "candle.log",
    "signal": "signal.log",
    "database": "database.log",
    "email": "email.log",
    "telegram": "telegram.log",
    "errors": "errors.log",
    "reconnect": "reconnect.log",
}

_FORMAT = "%(asctime)s | %(levelname)s | %(component)s | %(message)s"
_loggers: dict[str, logging.Logger] = {}


class _ComponentFilter(logging.Filter):
    def __init__(self, component: str) -> None:
        super().__init__()
        self.component = component

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "component"):
            record.component = self.component  # type: ignore[attr-defined]
        return True


def setup_live_paper_logging(*, log_dir: Path | None = None) -> Path:
    """Create component log files and return the log directory."""
    destination = log_dir or LOG_DIR
    destination.mkdir(parents=True, exist_ok=True)

    for component, filename in COMPONENT_FILES.items():
        logger_name = f"live_paper.{component}"
        component_logger = logging.getLogger(logger_name)
        component_logger.handlers.clear()
        component_logger.setLevel(logging.INFO)
        component_logger.propagate = False

        handler = logging.FileHandler(destination / filename, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler.addFilter(_ComponentFilter(component))
        component_logger.addHandler(handler)
        _loggers[component] = component_logger

    engine_logger.info("Live paper component logging ready under %s", destination)
    return destination


def get_logger(component: str) -> logging.Logger:
    """Return a named component logger (call ``setup_live_paper_logging`` first)."""
    if component in _loggers:
        return _loggers[component]
    if not _loggers:
        setup_live_paper_logging()
    if component in _loggers:
        return _loggers[component]
    fallback = logging.getLogger(f"live_paper.{component}")
    fallback.addFilter(_ComponentFilter(component))
    return fallback
