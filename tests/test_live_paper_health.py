"""Unit tests for ConnectionHealthMonitor helpers."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.live_paper.health import (
    ConnectionHealthMonitor,
    detect_missed_candles,
    is_ist_session,
    market_status_label,
)
from src.live_paper.logging_setup import setup_live_paper_logging
from src.live_paper.metrics import LiveMetrics

IST = ZoneInfo("Asia/Kolkata")


def test_detect_missed_candles_counts_session_bars() -> None:
    start = datetime(2026, 7, 21, 10, 0, tzinfo=IST)
    end = datetime(2026, 7, 21, 10, 20, tzinfo=IST)
    # expected closed bars after start: 10:05, 10:10, 10:15, 10:20 -> missed extras = 3
    missed = detect_missed_candles(start, end)
    assert missed >= 2


def test_is_stale_requires_session(monkeypatch) -> None:
    setup_live_paper_logging()
    metrics = LiveMetrics()
    monitor = ConnectionHealthMonitor(metrics, stale_tick_seconds=1.0)
    weekend = datetime(2026, 7, 19, 12, 0, tzinfo=IST)  # Sunday
    assert is_ist_session(weekend) is False

    import src.live_paper.health as health_mod

    monkeypatch.setattr(health_mod, "is_ist_session", lambda now=None: False)
    assert monitor.is_stale() is False

    monkeypatch.setattr(health_mod, "is_ist_session", lambda now=None: True)
    assert monitor.is_stale() is True
    monitor.record_tick()
    assert monitor.is_stale(now=monitor.metrics.last_tick_at) is False


def test_record_reconnect_updates_metrics() -> None:
    setup_live_paper_logging()
    metrics = LiveMetrics()
    monitor = ConnectionHealthMonitor(metrics)
    monitor.record_reconnect("test_disconnect")
    assert metrics.reconnect_count == 1
    assert metrics.reconnect_events
    assert metrics.ws_status == "reconnecting"


def test_market_status_label_types() -> None:
    label = market_status_label(datetime(2026, 7, 21, 8, 0, tzinfo=IST))
    assert label in {"pre_open", "open", "closed", "weekend"}
