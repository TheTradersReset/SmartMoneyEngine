"""Unit tests for TelegramNotifier (no live network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.notifications.telegram import TelegramNotifier


def test_disabled_without_credentials() -> None:
    notifier = TelegramNotifier(bot_token="", chat_id="", enabled=True)
    assert notifier.enabled is False
    assert notifier.send_message("hello") is False


def test_format_signal_alert_uses_runner_t3() -> None:
    text = TelegramNotifier.format_signal_alert(
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
    assert "T3: Runner" in text
    assert "Signal ID: 42" in text
    assert "Latency ms: 12.5" in text
    assert "Risk: 10.0" in text


def test_format_outcome_alert() -> None:
    text = TelegramNotifier.format_outcome_alert(
        direction="SELL",
        outcome="WIN",
        pnl_points=60.0,
        r_multiple=6.0,
        holding_bars=12,
        exit_timestamp="2026-07-21 11:00:00+05:30",
        signal_id=7,
    )
    assert "Result: WIN" in text
    assert "PnL points: 60.00" in text
    assert "R multiple: 6.00" in text


def test_send_message_retries_once() -> None:
    notifier = TelegramNotifier(bot_token="token", chat_id="123", enabled=True)
    with patch.object(notifier, "_post", side_effect=[False, True]) as mock_post:
        assert notifier.send_message("hi") is True
        assert mock_post.call_count == 2


def test_send_message_success_first_try() -> None:
    notifier = TelegramNotifier(bot_token="token", chat_id="123", enabled=True)
    fake_resp = MagicMock()
    fake_resp.read.return_value = b'{"ok": true}'
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = False
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
        assert notifier.send_message("hi") is True
        assert mock_open.call_count == 1
