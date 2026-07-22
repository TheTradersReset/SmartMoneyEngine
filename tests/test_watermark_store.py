"""Unit tests for WatermarkStore (Phase 1)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.live_paper.runtime.watermark import WatermarkStore, normalize_timestamp

IST = ZoneInfo("Asia/Kolkata")


def test_normalize_iso_and_compact_offset() -> None:
    a = normalize_timestamp("2026-07-22T10:15:00+05:30")
    b = normalize_timestamp("2026-07-22T10:15:00+0530")
    assert a is not None and b is not None
    assert a == b
    assert a.tzinfo is not None
    assert a.utcoffset() == timedelta(hours=5, minutes=30)


def test_normalize_naive_assumes_ist() -> None:
    dt = normalize_timestamp("2026-07-22T10:15:00")
    assert dt is not None
    assert dt.tzinfo == IST


def test_normalize_corrupt_returns_none() -> None:
    assert normalize_timestamp("not-a-timestamp") is None
    assert normalize_timestamp("") is None
    assert normalize_timestamp(None) is None


def test_initialize_takes_max_of_candidates() -> None:
    store = WatermarkStore()
    now = datetime(2026, 7, 22, 15, 0, tzinfo=IST)
    result = store.initialize(
        "2026-07-22T10:00:00+05:30",
        "2026-07-22T10:15:00+0530",
        None,
        "bad",
        now=now,
    )
    assert result == datetime(2026, 7, 22, 10, 15, tzinfo=IST)
    assert store.get() == result


def test_initialize_empty_leaves_none() -> None:
    store = WatermarkStore()
    assert store.initialize(None, "bad") is None
    assert store.get() is None


def test_initialize_rejects_future_maximum() -> None:
    store = WatermarkStore()
    now = datetime(2026, 7, 22, 10, 0, tzinfo=IST)
    future = now + timedelta(hours=2)
    assert store.initialize(future, now=now) is None
    assert store.get() is None


def test_try_advance_monotonic_and_duplicate() -> None:
    store = WatermarkStore()
    now = datetime(2026, 7, 22, 15, 0, tzinfo=IST)
    assert store.try_advance("2026-07-22T10:00:00+05:30", now=now) is True
    assert store.try_advance("2026-07-22T10:00:00+0530", now=now) is False
    assert store.try_advance("2026-07-22T09:55:00+05:30", now=now) is False
    assert store.try_advance("2026-07-22T10:05:00+05:30", now=now) is True
    assert store.as_iso() == "2026-07-22T10:05:00+05:30"


def test_try_advance_rejects_future_and_corrupt() -> None:
    fixed_now = datetime(2026, 7, 22, 10, 0, tzinfo=IST)
    store = WatermarkStore(clock=lambda: fixed_now)
    assert store.try_advance("2026-07-22T12:00:00+05:30") is False
    assert store.try_advance("garbage") is False
    assert store.get() is None


def test_restart_reinitialize_overwrites_memory() -> None:
    store = WatermarkStore()
    now = datetime(2026, 7, 22, 15, 0, tzinfo=IST)
    store.try_advance("2026-07-22T11:00:00+05:30", now=now)
    store.initialize("2026-07-22T09:30:00+05:30", now=now)
    assert store.get() == datetime(2026, 7, 22, 9, 30, tzinfo=IST)


def test_snapshot() -> None:
    store = WatermarkStore()
    now = datetime(2026, 7, 22, 15, 0, tzinfo=IST)
    store.try_advance("2026-07-22T10:00:00+05:30", now=now)
    snap = store.snapshot()
    assert snap["watermark"] == "2026-07-22T10:00:00+05:30"