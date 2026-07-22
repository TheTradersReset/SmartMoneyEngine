"""
Monotonic candle watermark store (Phase 1).

Tracks the last contiguous closed-bar timestamp that has been applied to
runtime context. In-memory only for Phase 1; restart callers must re-seed via
``initialize``.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Allow slight clock skew / late close before treating a stamp as "future".
_DEFAULT_FUTURE_SKEW = timedelta(minutes=5)


def normalize_timestamp(value: Any, *, tz: ZoneInfo = IST) -> datetime | None:
    """
    Normalize a candle timestamp to a timezone-aware ``datetime`` in ``tz``.

    Accepts ``datetime``, ISO-8601 strings with ``+05:30`` or ``+0530`` offsets,
    and naive values (treated as already in ``tz``). Returns None when unparsable.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=tz)
        return dt.astimezone(tz).replace(microsecond=0)

    text = str(value).strip()
    if not text:
        return None

    # Normalize "+0530" / "-0530" style offsets to "+05:30" for fromisoformat.
    if (
        len(text) >= 5
        and text[-5] in "+-"
        and text[-3] != ":"
        and text[-4:].replace(":", "").isdigit()
    ):
        text = f"{text[:-2]}:{text[-2:]}"

    # Tolerate trailing Z
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt.replace(microsecond=0)


class WatermarkStore:
    """
    Thread-safe monotonic watermark.

    Public API
    ----------
    get() -> datetime | None
    as_iso() -> str | None
    normalize(value) -> datetime | None
        Class/static helper wrapping ``normalize_timestamp``.
    initialize(*candidates, now=None) -> datetime | None
        Restart seeding: set watermark to max(normalized candidates).
        Ignores None/corrupt values. Rejects pure-future maxima.
    try_advance(value, *, now=None) -> bool
        Advance only when normalized value is strictly greater than current
        and not beyond ``now + future_skew``. Duplicate / older / corrupt /
        future stamps leave the watermark unchanged and return False.
    clear() -> None
        Test helper only; production restart should call ``initialize``.
    snapshot() -> dict
    """

    def __init__(
        self,
        *,
        tz: ZoneInfo = IST,
        future_skew: timedelta = _DEFAULT_FUTURE_SKEW,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._tz = tz
        self._future_skew = future_skew
        self._clock = clock or (lambda: datetime.now(tz=self._tz))
        self._lock = threading.RLock()
        self._watermark: datetime | None = None

    @staticmethod
    def normalize(value: Any, *, tz: ZoneInfo = IST) -> datetime | None:
        return normalize_timestamp(value, tz=tz)

    def get(self) -> datetime | None:
        with self._lock:
            return self._watermark

    def as_iso(self) -> str | None:
        with self._lock:
            return None if self._watermark is None else self._watermark.isoformat()

    def clear(self) -> None:
        """Reset to empty (intended for unit tests)."""
        with self._lock:
            self._watermark = None

    def initialize(self, *candidates: Any, now: datetime | None = None) -> datetime | None:
        """
        Seed watermark from restart sources (context last bar, DB max, …).

        Uses the maximum of successfully normalized candidates. If that maximum
        is in the future beyond skew, the store stays empty and returns None.
        """
        parsed: list[datetime] = []
        for raw in candidates:
            dt = normalize_timestamp(raw, tz=self._tz)
            if dt is not None:
                parsed.append(dt)

        with self._lock:
            if not parsed:
                self._watermark = None
                return None
            chosen = max(parsed)
            if self._is_future(chosen, now=now):
                self._watermark = None
                return None
            self._watermark = chosen
            return self._watermark

    def try_advance(self, value: Any, *, now: datetime | None = None) -> bool:
        """Monotonic advance. Returns True only when the watermark moved forward."""
        dt = normalize_timestamp(value, tz=self._tz)
        if dt is None:
            return False
        if self._is_future(dt, now=now):
            return False
        with self._lock:
            current = self._watermark
            if current is not None and dt <= current:
                return False
            self._watermark = dt
            return True

    def _is_future(self, dt: datetime, *, now: datetime | None) -> bool:
        ref = now if now is not None else self._clock()
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=self._tz)
        else:
            ref = ref.astimezone(self._tz)
        return dt > (ref + self._future_skew)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "watermark": None if self._watermark is None else self._watermark.isoformat(),
                "tz": str(self._tz),
            }
