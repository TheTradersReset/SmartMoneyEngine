"""
Telegram notifier for live paper trading alerts.

Disabled safely when bot token / chat id are missing. Never places orders.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger("live_paper.telegram")


class TelegramNotifier:
    """Thin Telegram Bot API wrapper for signal / outcome alerts."""

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        *,
        enabled: bool = True,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.timeout_seconds = float(timeout_seconds)
        self.enabled = bool(enabled) and bool(self.bot_token) and bool(self.chat_id)
        if enabled and not self.enabled:
            logger.warning(
                "Telegram disabled: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing."
            )

    def send_message(self, text: str) -> bool:
        """Send a plain-text message. Retries once on transient failure."""
        if not self.enabled:
            logger.info("Telegram skipped (disabled): %s", text[:120])
            return False
        ok = self._post(text)
        if not ok:
            logger.warning("Telegram send failed; retrying once.")
            ok = self._post(text)
        return ok

    def _post(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = urllib.parse.urlencode(
            {
                "chat_id": self.chat_id,
                "text": text,
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
            data = json.loads(body) if body else {}
            if not data.get("ok", False):
                logger.error("Telegram API error: %s", body[:300])
                return False
            return True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    @staticmethod
    def format_signal_alert(
        *,
        symbol: str,
        direction: str,
        entry: float,
        stop: float,
        target1: float,
        target2: float,
        risk: float | None = None,
        timestamp: str,
        strategy_version: str,
        signal_id: int | str | None,
        latency_ms: float | None = None,
        target3: str = "Runner",
    ) -> str:
        """Format an accepted signal alert (Target3 = Runner)."""
        risk_pts = risk if risk is not None else abs(float(entry) - float(stop))
        latency = f"{latency_ms:.1f}" if latency_ms is not None else "n/a"
        return (
            f"PAPER SIGNAL\n"
            f"Symbol: {symbol}\n"
            f"Side: {direction}\n"
            f"Entry: {entry}\n"
            f"SL: {stop}\n"
            f"T1: {target1}\n"
            f"T2: {target2}\n"
            f"T3: {target3}\n"
            f"Risk: {risk_pts}\n"
            f"Timestamp: {timestamp}\n"
            f"Strategy: {strategy_version}\n"
            f"Signal ID: {signal_id}\n"
            f"Latency ms: {latency}"
        )

    @staticmethod
    def format_outcome_alert(
        *,
        direction: str,
        outcome: str,
        pnl_points: float | None,
        r_multiple: float | None,
        holding_bars: int | None,
        exit_timestamp: str | None,
        signal_id: int | str | None = None,
    ) -> str:
        """Format WIN / LOSS / BREAKEVEN outcome alert."""
        label = str(outcome or "").upper()
        pnl = f"{pnl_points:.2f}" if pnl_points is not None else "n/a"
        r_mult = f"{r_multiple:.2f}" if r_multiple is not None else "n/a"
        held = str(holding_bars) if holding_bars is not None else "n/a"
        return (
            f"PAPER OUTCOME\n"
            f"Side: {direction}\n"
            f"Result: {label}\n"
            f"PnL points: {pnl}\n"
            f"R multiple: {r_mult}\n"
            f"Holding bars: {held}\n"
            f"Exit timestamp: {exit_timestamp or 'n/a'}\n"
            f"Signal ID: {signal_id}"
        )

    def notify_signal(self, **kwargs: Any) -> bool:
        return self.send_message(self.format_signal_alert(**kwargs))

    def notify_outcome(self, **kwargs: Any) -> bool:
        return self.send_message(self.format_outcome_alert(**kwargs))
