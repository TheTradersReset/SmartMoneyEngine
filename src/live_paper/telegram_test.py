"""Send a one-shot Telegram TEST message for live paper."""

from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        from src.live_paper.config import load_config
        from src.notifications.telegram import TelegramNotifier
    except Exception as exc:
        print(f"FAIL reason=import_error detail={exc}")
        return 1

    try:
        cfg = load_config()
    except Exception as exc:
        print(f"FAIL reason=config_error detail={exc}")
        return 1

    if not cfg.enable_telegram:
        print("FAIL reason=telegram_disabled")
        return 1
    if not (cfg.telegram_bot_token or "").strip() or not (cfg.telegram_chat_id or "").strip():
        print("FAIL reason=missing_telegram_credentials")
        return 1

    notifier = TelegramNotifier(
        bot_token=cfg.telegram_bot_token,
        chat_id=cfg.telegram_chat_id,
        enabled=cfg.enable_telegram,
    )
    if not notifier.enabled:
        print("FAIL reason=notifier_disabled")
        return 1

    started = time.perf_counter()
    ok = notifier.send_message("SmartMoneyEngine live_paper Telegram TEST")
    latency_ms = (time.perf_counter() - started) * 1000.0

    if ok:
        print(f"PASS latency_ms={latency_ms:.1f}")
        return 0

    print(f"FAIL reason=send_failed latency_ms={latency_ms:.1f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())