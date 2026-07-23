"""Live paper environment readiness check (no secret values printed)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "data" / "tokens" / "fyers_token.json"


def _status(set_: bool) -> str:
    return "SET" if set_ else "MISSING"


def _truthy_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ImportError:
        pass


def _token_file_info(token_path: Path) -> tuple[bool, bool | None, bool]:
    """
    Returns (file_present, access_token_valid_or_None, refresh_token_present).
    access_token_valid is None when validity cannot be checked.
    """
    if not token_path.exists():
        return False, False, False

    try:
        data = json.loads(token_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True, False, False

    access = data.get("access_token")
    refresh = data.get("refresh_token")
    refresh_present = isinstance(refresh, str) and bool(refresh.strip())

    access_valid: bool | None = False
    if isinstance(access, str) and access.strip():
        try:
            from src.brokers.fyers.auth import is_access_token_valid

            access_valid = bool(is_access_token_valid(access.strip()))
        except Exception:
            access_valid = None
    else:
        access_valid = False

    return True, access_valid, refresh_present


def main() -> int:
    _load_dotenv()

    fyers_app = (os.getenv("FYERS_APP_ID") or os.getenv("FYERS_CLIENT_ID") or "").strip()
    fyers_secret = (os.getenv("FYERS_SECRET_KEY") or "").strip()
    fyers_redirect = (os.getenv("FYERS_REDIRECT_URI") or "").strip()
    fyers_pin = (os.getenv("FYERS_PIN") or "").strip()
    smtp_host = (os.getenv("SMTP_HOST") or "").strip()
    smtp_from = (os.getenv("SMTP_FROM") or "").strip()
    smtp_to = (os.getenv("SMTP_TO") or "").strip()
    tg_token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    tg_chat = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    capital_mode = (os.getenv("LIVE_PAPER_CAPITAL_MODE") or "").strip()
    enable_email_env = os.getenv("LIVE_PAPER_ENABLE_EMAIL")
    enable_tg_env = os.getenv("LIVE_PAPER_ENABLE_TELEGRAM")
    enable_dash = (os.getenv("LIVE_PAPER_ENABLE_DASHBOARD") or "").strip()
    dash_host = (os.getenv("LIVE_PAPER_DASHBOARD_HOST") or "").strip()
    dash_port = (os.getenv("LIVE_PAPER_DASHBOARD_PORT") or "").strip()

    token_present, access_valid, refresh_present = _token_file_info(DEFAULT_TOKEN_PATH)

    enable_email = True
    enable_telegram = False
    try:
        from src.live_paper.config import load_config

        cfg = load_config()
        enable_email = bool(cfg.enable_email)
        enable_telegram = bool(cfg.enable_telegram)
        if not smtp_host:
            smtp_host = (cfg.smtp_host or "").strip()
        if not smtp_from:
            smtp_from = (cfg.smtp_from or "").strip()
        if not smtp_to:
            smtp_to = (cfg.smtp_to or "").strip()
        if not tg_token:
            tg_token = (cfg.telegram_bot_token or "").strip()
        if not tg_chat:
            tg_chat = (cfg.telegram_chat_id or "").strip()
    except Exception:
        enable_email = _truthy_env("LIVE_PAPER_ENABLE_EMAIL", True)
        enable_telegram = _truthy_env("LIVE_PAPER_ENABLE_TELEGRAM", False)

    if access_valid is True:
        access_label = "YES"
    elif access_valid is False:
        access_label = "NO"
    else:
        access_label = "UNKNOWN"

    lines = [
        "LIVE PAPER ENV CHECK",
        "--------------------",
        f"{'FYERS_APP_ID':<28}{_status(bool(fyers_app))}",
        f"{'FYERS_SECRET_KEY':<28}{_status(bool(fyers_secret))}",
        f"{'FYERS_REDIRECT_URI':<28}{_status(bool(fyers_redirect))}",
        f"{'FYERS_PIN':<28}{_status(bool(fyers_pin))}",
        f"{'SMTP_HOST':<28}{_status(bool(smtp_host))}",
        f"{'SMTP_FROM':<28}{_status(bool(smtp_from))}",
        f"{'SMTP_TO':<28}{_status(bool(smtp_to))}",
        f"{'TELEGRAM_BOT_TOKEN':<28}{_status(bool(tg_token))}",
        f"{'TELEGRAM_CHAT_ID':<28}{_status(bool(tg_chat))}",
        f"{'LIVE_PAPER_CAPITAL_MODE':<28}{_status(bool(capital_mode))}",
        f"{'LIVE_PAPER_ENABLE_EMAIL':<28}{_status(enable_email_env is not None and bool(str(enable_email_env).strip()))}",
        f"{'LIVE_PAPER_ENABLE_TELEGRAM':<28}{_status(enable_tg_env is not None and bool(str(enable_tg_env).strip()))}",
        f"{'LIVE_PAPER_ENABLE_DASHBOARD':<28}{_status(bool(enable_dash))}",
        f"{'LIVE_PAPER_DASHBOARD_HOST':<28}{_status(bool(dash_host))}",
        f"{'LIVE_PAPER_DASHBOARD_PORT':<28}{_status(bool(dash_port))}",
        f"{'TOKEN_FILE':<28}{'PRESENT' if token_present else 'MISSING'}",
        f"{'ACCESS_TOKEN_VALID':<28}{access_label}",
        f"{'REFRESH_TOKEN':<28}{'PRESENT' if refresh_present else 'MISSING'}",
    ]

    failures: list[str] = []
    if not fyers_app:
        failures.append("FYERS_APP_ID/FYERS_CLIENT_ID")
    if not fyers_secret:
        failures.append("FYERS_SECRET_KEY")
    if not fyers_redirect:
        failures.append("FYERS_REDIRECT_URI")

    has_valid_access = access_valid is True
    has_refresh_path = refresh_present and bool(fyers_pin)
    if not has_valid_access and not has_refresh_path:
        failures.append("ACCESS_TOKEN_OR_REFRESH+PIN")

    if enable_email:
        if not smtp_host:
            failures.append("SMTP_HOST")
        if not smtp_from:
            failures.append("SMTP_FROM")
        if not smtp_to:
            failures.append("SMTP_TO")

    if enable_telegram:
        if not tg_token:
            failures.append("TELEGRAM_BOT_TOKEN")
        if not tg_chat:
            failures.append("TELEGRAM_CHAT_ID")

    ok = not failures
    lines.append(f"{'RESULT':<28}{'PASS' if ok else 'FAIL'}")
    if failures:
        lines.append("FAILURES: " + ", ".join(failures))

    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
