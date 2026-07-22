"""
Runtime phase state machine for the isolated live pipeline (Phase 1).

Owns a single monotonic phase value. Not wired into the live service until
later phases enable ``enable_pipeline_v2``.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Iterable


class RuntimePhase(str, Enum):
    """Ordered runtime phases for the live paper pipeline."""

    BOOT = "BOOT"
    WARM_START = "WARM_START"
    WATERMARK_SET = "WATERMARK_SET"
    PRELIVE_CATCHUP = "PRELIVE_CATCHUP"
    LIVE = "LIVE"
    BACKFILL_QUEUED = "BACKFILL_QUEUED"
    SHUTDOWN = "SHUTDOWN"


# Explicit allowed edges. SHUTDOWN is reachable from every non-terminal phase.
_ALLOWED: dict[RuntimePhase, frozenset[RuntimePhase]] = {
    RuntimePhase.BOOT: frozenset({RuntimePhase.WARM_START, RuntimePhase.SHUTDOWN}),
    RuntimePhase.WARM_START: frozenset({RuntimePhase.WATERMARK_SET, RuntimePhase.SHUTDOWN}),
    RuntimePhase.WATERMARK_SET: frozenset(
        {RuntimePhase.PRELIVE_CATCHUP, RuntimePhase.LIVE, RuntimePhase.SHUTDOWN}
    ),
    RuntimePhase.PRELIVE_CATCHUP: frozenset({RuntimePhase.LIVE, RuntimePhase.SHUTDOWN}),
    RuntimePhase.LIVE: frozenset({RuntimePhase.BACKFILL_QUEUED, RuntimePhase.SHUTDOWN}),
    RuntimePhase.BACKFILL_QUEUED: frozenset({RuntimePhase.LIVE, RuntimePhase.SHUTDOWN}),
    RuntimePhase.SHUTDOWN: frozenset(),
}


class InvalidPhaseTransition(ValueError):
    """Raised when a phase transition is not permitted."""


class RuntimePhaseController:
    """
    Thread-safe owner of the process runtime phase.

    Public API
    ----------
    phase:
        Current ``RuntimePhase``.
    transition_to(target):
        Perform a validated transition; raise ``InvalidPhaseTransition`` if illegal.
    try_transition_to(target) -> bool:
        Same as ``transition_to`` but returns False instead of raising.
    request_shutdown() -> bool:
        Transition to ``SHUTDOWN`` from any non-SHUTDOWN phase.
    allows_signal_emit() -> bool:
        True only in ``LIVE`` (new live closes may emit paper signals).
    is_live() / is_shutdown() / in_phases(...):
        Convenience predicates.
    snapshot() -> dict:
        Serialisable status for tests/diagnostics.
    """

    def __init__(self, initial: RuntimePhase = RuntimePhase.BOOT) -> None:
        if not isinstance(initial, RuntimePhase):
            raise TypeError(f"initial must be RuntimePhase, got {type(initial)!r}")
        self._lock = threading.RLock()
        self._phase = initial
        self._history: list[str] = [initial.value]

    @property
    def phase(self) -> RuntimePhase:
        with self._lock:
            return self._phase

    def transition_to(self, target: RuntimePhase) -> RuntimePhase:
        """Move to ``target`` if the edge is allowed."""
        if not isinstance(target, RuntimePhase):
            raise TypeError(f"target must be RuntimePhase, got {type(target)!r}")
        with self._lock:
            current = self._phase
            if target == current:
                return current
            allowed = _ALLOWED[current]
            if target not in allowed:
                raise InvalidPhaseTransition(
                    f"Illegal phase transition {current.value} -> {target.value}"
                )
            self._phase = target
            self._history.append(target.value)
            return self._phase

    def try_transition_to(self, target: RuntimePhase) -> bool:
        """Attempt ``transition_to``; return False on illegal transition."""
        try:
            self.transition_to(target)
            return True
        except InvalidPhaseTransition:
            return False

    def request_shutdown(self) -> bool:
        """
        Enter ``SHUTDOWN``.

        Returns True when the transition occurred, False if already shut down.
        """
        with self._lock:
            if self._phase is RuntimePhase.SHUTDOWN:
                return False
            self._phase = RuntimePhase.SHUTDOWN
            self._history.append(RuntimePhase.SHUTDOWN.value)
            return True

    def allows_signal_emit(self) -> bool:
        """New paper signals / outcome emails for live closes are live-only."""
        with self._lock:
            return self._phase is RuntimePhase.LIVE

    def is_live(self) -> bool:
        with self._lock:
            return self._phase is RuntimePhase.LIVE

    def is_shutdown(self) -> bool:
        with self._lock:
            return self._phase is RuntimePhase.SHUTDOWN

    def in_phases(self, phases: Iterable[RuntimePhase]) -> bool:
        with self._lock:
            return self._phase in frozenset(phases)

    def history(self) -> list[str]:
        """Copy of phase value history (including initial)."""
        with self._lock:
            return list(self._history)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "phase": self._phase.value,
                "allows_signal_emit": self._phase is RuntimePhase.LIVE,
                "is_shutdown": self._phase is RuntimePhase.SHUTDOWN,
                "history": list(self._history),
            }
