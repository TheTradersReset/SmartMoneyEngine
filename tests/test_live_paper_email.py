"""Unit tests for EmailNotifier (no live network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.notifications.email import (
    EmailNotifier,
    format_outcome_subject,
    format_signal_subject,
    short_symbol,
)


def test_short_symbol_strips_exchange_and_index() -> None:
    assert short_symbol("NSE:NIFTY50-INDEX") == "NIFTY50"
    assert short_symbol("NIFTY50-INDEX") == "NIFTY50"


def test_format_signal_subject() -> None:
    subject = format_signal_subject("BUY", "NSE:NIFTY50-INDEX", "2026-07-21 10:15:00+05:30")
    assert subject == "BUY | NIFTY50 | 10:15"


def test_format_outcome_subject() -> None:
    assert format_outcome_subject("WIN", "SELL") == "OUTCOME | WIN | SELL"


def test_format_signal_body_fields() -> None:
    text = EmailNotifier.format_signal_body(
        symbol="NSE:NIFTY50-INDEX",
        direction="BUY",
        entry=25000.0,
        stop=24990.0,
        target1=25060.0,
        target2=25100.0,
        risk=10.0,
        timestamp="2026-07-21 10:00:00+05:30",
        strategy_version="BUY_V3",
        signal_id=42,
        latency_ms=12.5,
    )
    assert "Stop Loss: 24990.0" in text
    assert "T3: Runner" in text
    assert "Strategy Version: BUY_V3" in text
    assert "Signal ID: 42" in text
    assert "Processing Latency: 12.5 ms" in text
    assert "Risk: 10.0" in text


def test_format_outcome_body() -> None:
    text = EmailNotifier.format_outcome_body(
        direction="SELL",
        outcome="WIN",
        pnl_points=60.0,
        r_multiple=6.0,
        holding_bars=12,
        exit_timestamp="2026-07-21 11:00:00+05:30",
        signal_id=7,
    )
    assert "Result: WIN" in text
    assert "Exit Time: 2026-07-21 11:00:00+05:30" in text
    assert "Holding Bars: 12" in text
    assert "R Multiple: 6.00" in text
    assert "PnL: 60.00" in text


def test_disabled_skips_send() -> None:
    notifier = EmailNotifier(
        smtp_host="smtp.example.com",
        smtp_from="a@example.com",
        smtp_to="b@example.com",
        enabled=False,
    )
    assert notifier.enabled is False
    assert notifier.send_message("subj", "body") is False


def test_disabled_without_required_fields() -> None:
    notifier = EmailNotifier(smtp_host="", smtp_from="", smtp_to="", enabled=True)
    assert notifier.enabled is False


def test_notify_signal_calls_smtp_with_buy_subject() -> None:
    notifier = EmailNotifier(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="user",
        smtp_password="pass",
        smtp_from="from@example.com",
        smtp_to="to@example.com",
        use_tls=True,
        use_ssl=False,
        enabled=True,
    )
    mock_smtp = MagicMock()
    mock_smtp.__enter__.return_value = mock_smtp
    mock_smtp.__exit__.return_value = False

    with patch("smtplib.SMTP", return_value=mock_smtp) as smtp_ctor:
        ok = notifier.notify_signal(
            symbol="NSE:NIFTY50-INDEX",
            direction="BUY",
            entry=25000.0,
            stop=24990.0,
            target1=25060.0,
            target2=25100.0,
            risk=10.0,
            timestamp="2026-07-21 10:15:00+05:30",
            strategy_version="BUY_V3",
            signal_id=42,
            latency_ms=12.5,
        )
    assert ok is True
    smtp_ctor.assert_called()
    mock_smtp.starttls.assert_called()
    mock_smtp.login.assert_called_once_with("user", "pass")
    mock_smtp.send_message.assert_called()
    sent_msg = mock_smtp.send_message.call_args[0][0]
    assert "BUY |" in sent_msg["Subject"]
    assert "NIFTY50" in sent_msg["Subject"]
