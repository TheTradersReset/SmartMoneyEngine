"""Send a one-shot SMTP TEST email for live paper."""

from __future__ import annotations

import sys
import time
from datetime import datetime


def main() -> int:
    try:
        from src.live_paper.config import load_config
        from src.notifications.email import EmailNotifier
    except Exception as exc:
        print(f"FAIL reason=import_error detail={exc}")
        return 1

    try:
        cfg = load_config()
    except Exception as exc:
        print(f"FAIL reason=config_error detail={exc}")
        return 1

    if not cfg.enable_email:
        print("FAIL reason=email_disabled")
        return 1
    if not (cfg.smtp_host or "").strip() or not (cfg.smtp_from or "").strip() or not (cfg.smtp_to or "").strip():
        print("FAIL reason=missing_smtp_credentials")
        return 1

    notifier = EmailNotifier(
        smtp_host=cfg.smtp_host,
        smtp_port=cfg.smtp_port,
        smtp_user=cfg.smtp_user,
        smtp_password=cfg.smtp_password,
        smtp_from=cfg.smtp_from,
        smtp_to=cfg.smtp_to,
        use_tls=cfg.smtp_use_tls,
        use_ssl=cfg.smtp_use_ssl,
        enabled=cfg.enable_email,
    )
    if not notifier.enabled:
        print("FAIL reason=notifier_disabled")
        return 1

    hhmm = datetime.now().strftime("%H:%M")
    subject = f"TEST | SmartMoneyEngine | {hhmm}"
    body = (
        "SmartMoneyEngine live_paper SMTP TEST\n"
        f"Time: {datetime.now().isoformat(timespec='seconds')}\n"
        "If you received this, email notifications are working."
    )

    started = time.perf_counter()
    ok = notifier.send_message(subject, body)
    latency_ms = (time.perf_counter() - started) * 1000.0

    if ok:
        print(f"PASS latency_ms={latency_ms:.1f}")
        return 0

    print(f"FAIL reason=send_failed latency_ms={latency_ms:.1f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
