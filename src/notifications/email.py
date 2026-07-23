"""
SMTP email notifier for live paper trading alerts.

Disabled safely when host / from / to are missing. Never places orders.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from typing import Any

logger = logging.getLogger("live_paper.email")


def short_symbol(symbol: str) -> str:
    """NSE:NIFTY50-INDEX -> NIFTY50 (strip exchange prefix and -INDEX)."""
    s = (symbol or "").strip()
    if ":" in s:
        s = s.split(":", 1)[1]
    upper = s.upper()
    if upper.endswith("-INDEX"):
        s = s[: -len("-INDEX")]
    return s


def _hhmm_from_timestamp(timestamp: Any) -> str:
    """Extract HH:MM from a timestamp string/datetime; fall back to now."""
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%H:%M")
    raw = str(timestamp or "").strip()
    if raw:
        # Prefer explicit time portion HH:MM
        for sep in ("T", " "):
            if sep in raw:
                time_part = raw.split(sep, 1)[1]
                # strip timezone offset like +05:30 or Z
                for cut in ("+", "-", "Z", "z"):
                    idx = time_part.find(cut)
                    if cut == "-" and idx == 0:
                        continue
                    if idx > 0:
                        time_part = time_part[:idx]
                        break
                time_part = time_part.strip()
                if len(time_part) >= 5 and time_part[2] == ":":
                    return time_part[:5]
        try:
            cleaned = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned).strftime("%H:%M")
        except ValueError:
            pass
    return datetime.now().strftime("%H:%M")


def format_signal_subject(direction: str, symbol: str, timestamp: Any = None) -> str:
    """Build subject like: BUY | NIFTY50 | 10:15."""
    side = str(direction or "").upper()
    hhmm = _hhmm_from_timestamp(timestamp)
    return f"{side} | {short_symbol(symbol)} | {hhmm}"


def format_outcome_subject(outcome: str, direction: str) -> str:
    """Build subject like: OUTCOME | WIN | BUY."""
    return f"OUTCOME | {str(outcome or '').upper()} | {str(direction or '').upper()}"


class EmailNotifier:
    """Thin SMTP wrapper for signal / outcome alerts."""

    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        smtp_from: str = "",
        smtp_to: str = "",
        *,
        use_tls: bool = True,
        use_ssl: bool = False,
        enabled: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.smtp_host = (smtp_host or "").strip()
        self.smtp_port = int(smtp_port) if smtp_port else 587
        self.smtp_user = (smtp_user or "").strip()
        self.smtp_password = smtp_password or ""
        self.smtp_from = (smtp_from or "").strip()
        self.smtp_to = (smtp_to or "").strip()
        self.use_tls = bool(use_tls)
        self.use_ssl = bool(use_ssl)
        self.timeout_seconds = float(timeout_seconds)
        self.enabled = (
            bool(enabled)
            and bool(self.smtp_host)
            and bool(self.smtp_from)
            and bool(self.smtp_to)
        )
        if enabled and not self.enabled:
            logger.warning(
                "Email disabled: SMTP_HOST / SMTP_FROM / SMTP_TO missing."
            )

    def send_message(self, subject: str, body: str) -> bool:
        """Send a plain-text email. Retries once on failure. Never raises."""
        if not self.enabled:
            logger.info("Email skipped (disabled): %s", (subject or "")[:120])
            return False
        try:
            ok = self._send(subject, body)
            if not ok:
                logger.warning("Email send failed; retrying once.")
                ok = self._send(subject, body)
            return ok
        except Exception as exc:  # noqa: BLE001
            logger.error("Email send unexpected error: %s", exc)
            return False

    def _send(self, subject: str, body: str) -> bool:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.smtp_from
        msg["To"] = self.smtp_to
        msg.set_content(body or "")

        try:
            if self.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    self.smtp_host,
                    self.smtp_port,
                    timeout=self.timeout_seconds,
                    context=context,
                ) as server:
                    if self.smtp_user:
                        server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(
                    self.smtp_host,
                    self.smtp_port,
                    timeout=self.timeout_seconds,
                ) as server:
                    server.ehlo()
                    if self.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                        server.ehlo()
                    if self.smtp_user:
                        server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
            return True
        except (smtplib.SMTPException, OSError, TimeoutError) as exc:
            logger.error("Email send failed: %s", exc)
            return False

    @staticmethod
    def format_signal_body(
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
        """Format an accepted signal email body."""
        risk_pts = risk if risk is not None else abs(float(entry) - float(stop))
        latency = f"{latency_ms:.1f} ms" if latency_ms is not None else "n/a"
        side = str(direction or "").upper()
        return (
            f"Symbol: {symbol}\n"
            f"Side: {side}\n"
            f"Entry: {entry}\n"
            f"Stop Loss: {stop}\n"
            f"Targets:\n"
            f"  T1: {target1}\n"
            f"  T2: {target2}\n"
            f"  T3: {target3}\n"
            f"Risk: {risk_pts}\n"
            f"Timestamp: {timestamp}\n"
            f"Strategy Version: {strategy_version}\n"
            f"Signal ID: {signal_id}\n"
            f"Processing Latency: {latency}"
        )

    @staticmethod
    def format_outcome_body(
        *,
        direction: str,
        outcome: str,
        pnl_points: float | None,
        r_multiple: float | None,
        holding_bars: int | None,
        exit_timestamp: str | None,
        signal_id: int | str | None = None,
    ) -> str:
        """Format WIN / LOSS / BREAKEVEN outcome email body."""
        label = str(outcome or "").upper()
        pnl = f"{pnl_points:.2f}" if pnl_points is not None else "n/a"
        r_mult = f"{r_multiple:.2f}" if r_multiple is not None else "n/a"
        held = str(holding_bars) if holding_bars is not None else "n/a"
        return (
            f"Result: {label}\n"
            f"Side: {str(direction or '').upper()}\n"
            f"Exit Time: {exit_timestamp or 'n/a'}\n"
            f"Holding Bars: {held}\n"
            f"R Multiple: {r_mult}\n"
            f"PnL: {pnl}\n"
            f"Signal ID: {signal_id}"
        )

    def notify_signal(self, **kwargs: Any) -> bool:
        subject = format_signal_subject(
            kwargs.get("direction", ""),
            kwargs.get("symbol", ""),
            kwargs.get("timestamp"),
        )
        body = self.format_signal_body(**kwargs)
        return self.send_message(subject, body)

    def notify_outcome(self, **kwargs: Any) -> bool:
        subject = format_outcome_subject(
            kwargs.get("outcome", ""),
            kwargs.get("direction", ""),
        )
        body = self.format_outcome_body(**kwargs)
        return self.send_message(subject, body)
