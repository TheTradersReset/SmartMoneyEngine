"""Replay pacing and lifecycle control (pause / resume / stop / speed)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal

# 5-minute bar duration in seconds — 1x paces one live bar interval.
BAR_DURATION_SECONDS = 5 * 60

SpeedValue = Literal[1, 5, 10, 100] | float


class ReplayState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


SPEED_ALIASES: dict[str, float] = {
    "1": 1.0,
    "1x": 1.0,
    "5": 5.0,
    "5x": 5.0,
    "10": 10.0,
    "10x": 10.0,
    "100": 100.0,
    "100x": 100.0,
    "unlimited": float("inf"),
    "inf": float("inf"),
    "max": float("inf"),
}


def parse_speed(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        speed = float(value)
    else:
        key = str(value).strip().lower()
        if key not in SPEED_ALIASES:
            raise ValueError(
                f"Unsupported speed '{value}'. "
                "Use 1x, 5x, 10x, 100x, or unlimited.",
            )
        speed = SPEED_ALIASES[key]
    if speed != float("inf") and speed <= 0:
        raise ValueError("Speed must be positive or unlimited.")
    return speed


@dataclass
class ReplayProgress:
    total_candles: int = 0
    processed_candles: int = 0
    state: ReplayState = ReplayState.IDLE
    last_candle_timestamp: str | None = None
    signals_seen: int = 0

    @property
    def pct(self) -> float:
        if self.total_candles <= 0:
            return 0.0
        return 100.0 * self.processed_candles / self.total_candles


class ReplayController:
    """
    Thread-safe pause / resume / stop and speed-based pacing.

    Delay between candles = BAR_DURATION_SECONDS / speed.
    Unlimited speed skips sleeping.
    """

    def __init__(self, *, speed: float = float("inf")) -> None:
        self._speed = parse_speed(speed)
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.progress = ReplayProgress()

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def state(self) -> ReplayState:
        return self.progress.state

    def set_speed(self, speed: str | float | int) -> None:
        with self._lock:
            self._speed = parse_speed(speed)

    def start(self, *, total_candles: int) -> None:
        with self._lock:
            self.progress = ReplayProgress(total_candles=total_candles, state=ReplayState.RUNNING)
            self._stop_event.clear()
            self._pause_event.set()

    def pause(self) -> None:
        with self._lock:
            if self.progress.state == ReplayState.RUNNING:
                self.progress.state = ReplayState.PAUSED
                self._pause_event.clear()

    def resume(self) -> None:
        with self._lock:
            if self.progress.state == ReplayState.PAUSED:
                self.progress.state = ReplayState.RUNNING
                self._pause_event.set()

    def stop(self) -> None:
        with self._lock:
            self.progress.state = ReplayState.STOPPED
            self._stop_event.set()
            self._pause_event.set()  # unblock waiters

    def mark_completed(self) -> None:
        with self._lock:
            if self.progress.state != ReplayState.STOPPED:
                self.progress.state = ReplayState.COMPLETED

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def wait_if_paused(self) -> None:
        self._pause_event.wait()

    def pace(self) -> None:
        """Sleep according to current speed; respects pause and stop."""
        self.wait_if_paused()
        if self.should_stop():
            return
        with self._lock:
            speed = self._speed
        if speed == float("inf"):
            return
        delay = BAR_DURATION_SECONDS / speed
        # Interruptible sleep in small slices so stop/pause stay responsive.
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if self.should_stop():
                return
            self.wait_if_paused()
            if self.should_stop():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))

    def record_candle(self, *, timestamp: str) -> None:
        with self._lock:
            self.progress.processed_candles += 1
            self.progress.last_candle_timestamp = timestamp

    def record_signal(self) -> None:
        with self._lock:
            self.progress.signals_seen += 1
