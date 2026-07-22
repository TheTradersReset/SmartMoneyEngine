"""
Configuration loader for the live paper trading service.

Loads defaults from ``config/live_paper/live_paper.yaml`` (if present),
then overlays environment variables. Capital mode is hard-required to be
``paper`` — live order routing is never enabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = PROJECT_ROOT / "config" / "live_paper" / "live_paper.yaml"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "paper" / "realtime_signals.db"
DEFAULT_HISTORY_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(str(raw).strip())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(str(raw).strip())


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    path = Path(str(raw).strip())
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


@dataclass
class LivePaperConfig:
    """Resolved runtime configuration for live paper trading."""

    symbol: str = "NSE:NIFTY50-INDEX"
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)
    history_csv: Path = field(default_factory=lambda: DEFAULT_HISTORY_CSV)
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080
    stale_tick_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    enable_email: bool = True
    enable_telegram: bool = False
    enable_dashboard: bool = True
    capital_mode: str = "paper"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    latency_warn_ms: float = 500.0
    # Phase-1 isolated pipeline foundations (unused until later phases).
    enable_pipeline_v2: bool = False
    live_close_queue_max: int = 32
    shutdown_drain_sec: float = 5.0
    yaml_path: Path = field(default_factory=lambda: DEFAULT_YAML)

    def assert_paper_mode(self) -> None:
        """Raise if capital mode is anything other than paper."""
        mode = (self.capital_mode or "").strip().lower()
        if mode != "paper":
            raise RuntimeError(
                f"LIVE_PAPER_CAPITAL_MODE must be 'paper' (got {self.capital_mode!r}). "
                "Live order APIs are disabled."
            )


def load_config(*, yaml_path: Path | None = None, env_path: Path | None = None) -> LivePaperConfig:
    """
    Load live-paper configuration from optional YAML + environment.

    Environment variables always win over YAML defaults.
    """
    try:
        from dotenv import load_dotenv

        dotenv_path = env_path or (PROJECT_ROOT / ".env")
        if dotenv_path.exists():
            load_dotenv(dotenv_path, override=False)
    except ImportError:
        pass

    cfg_path = yaml_path or DEFAULT_YAML
    yaml_data = _load_yaml(cfg_path)

    symbol = str(yaml_data.get("symbol", "NSE:NIFTY50-INDEX"))
    db_path = Path(yaml_data.get("db_path", str(DEFAULT_DB_PATH)))
    history_csv = Path(yaml_data.get("history_csv", str(DEFAULT_HISTORY_CSV)))
    dashboard_host = str(yaml_data.get("dashboard_host", "0.0.0.0"))
    dashboard_port = int(yaml_data.get("dashboard_port", 8080))
    stale_tick_seconds = float(yaml_data.get("stale_tick_seconds", 30))
    heartbeat_seconds = float(yaml_data.get("heartbeat_seconds", 10))
    enable_email = bool(yaml_data.get("enable_email", True))
    enable_telegram = bool(yaml_data.get("enable_telegram", False))
    enable_dashboard = bool(yaml_data.get("enable_dashboard", True))
    capital_mode = str(yaml_data.get("capital_mode", "paper"))
    latency_warn_ms = float(yaml_data.get("latency_warn_ms", 500))
    enable_pipeline_v2 = bool(yaml_data.get("enable_pipeline_v2", False))
    live_close_queue_max = int(yaml_data.get("live_close_queue_max", 32))
    shutdown_drain_sec = float(yaml_data.get("shutdown_drain_sec", 5.0))
    smtp_host = str(yaml_data.get("smtp_host", "") or "")
    smtp_port = int(yaml_data.get("smtp_port", 587) or 587)
    smtp_user = str(yaml_data.get("smtp_user", "") or "")
    smtp_password = str(yaml_data.get("smtp_password", "") or "")
    smtp_from = str(yaml_data.get("smtp_from", "") or "")
    smtp_to = str(yaml_data.get("smtp_to", "") or "")
    smtp_use_tls = bool(yaml_data.get("smtp_use_tls", True))
    smtp_use_ssl = bool(yaml_data.get("smtp_use_ssl", False))

    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    if not history_csv.is_absolute():
        history_csv = PROJECT_ROOT / history_csv

    config = LivePaperConfig(
        symbol=os.getenv("LIVE_PAPER_SYMBOL", symbol).strip() or symbol,
        db_path=_env_path("LIVE_PAPER_DB_PATH", db_path),
        history_csv=_env_path("LIVE_PAPER_HISTORY_CSV", history_csv),
        dashboard_host=(os.getenv("LIVE_PAPER_DASHBOARD_HOST") or dashboard_host).strip(),
        dashboard_port=_env_int("LIVE_PAPER_DASHBOARD_PORT", dashboard_port),
        stale_tick_seconds=_env_float("LIVE_PAPER_STALE_TICK_SECONDS", stale_tick_seconds),
        heartbeat_seconds=_env_float("LIVE_PAPER_HEARTBEAT_SECONDS", heartbeat_seconds),
        enable_email=_env_bool("LIVE_PAPER_ENABLE_EMAIL", enable_email),
        enable_telegram=_env_bool("LIVE_PAPER_ENABLE_TELEGRAM", enable_telegram),
        enable_dashboard=_env_bool("LIVE_PAPER_ENABLE_DASHBOARD", enable_dashboard),
        capital_mode=(os.getenv("LIVE_PAPER_CAPITAL_MODE") or capital_mode).strip().lower() or "paper",
        smtp_host=_env_str("SMTP_HOST", smtp_host),
        smtp_port=_env_int("SMTP_PORT", smtp_port),
        smtp_user=_env_str("SMTP_USER", smtp_user),
        smtp_password=_env_str("SMTP_PASSWORD", smtp_password),
        smtp_from=_env_str("SMTP_FROM", smtp_from),
        smtp_to=_env_str("SMTP_TO", smtp_to),
        smtp_use_tls=_env_bool("SMTP_USE_TLS", smtp_use_tls),
        smtp_use_ssl=_env_bool("SMTP_USE_SSL", smtp_use_ssl),
        telegram_bot_token=(os.getenv("TELEGRAM_BOT_TOKEN") or "").strip(),
        telegram_chat_id=(os.getenv("TELEGRAM_CHAT_ID") or "").strip(),
        latency_warn_ms=latency_warn_ms,
        enable_pipeline_v2=_env_bool("LIVE_PAPER_ENABLE_PIPELINE_V2", enable_pipeline_v2),
        live_close_queue_max=_env_int("LIVE_PAPER_LIVE_CLOSE_QUEUE_MAX", live_close_queue_max),
        shutdown_drain_sec=_env_float("LIVE_PAPER_SHUTDOWN_DRAIN_SEC", shutdown_drain_sec),
        yaml_path=cfg_path,
    )
    config.assert_paper_mode()
    return config
