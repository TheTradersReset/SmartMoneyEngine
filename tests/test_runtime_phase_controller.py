"""Unit tests for RuntimePhaseController (Phase 1)."""

from __future__ import annotations

import threading

import pytest

from src.live_paper.runtime.phases import (
    InvalidPhaseTransition,
    RuntimePhase,
    RuntimePhaseController,
)


def test_starts_in_boot() -> None:
    ctrl = RuntimePhaseController()
    assert ctrl.phase is RuntimePhase.BOOT
    assert ctrl.allows_signal_emit() is False
    assert ctrl.is_shutdown() is False


def test_happy_path_to_live_via_prelive() -> None:
    ctrl = RuntimePhaseController()
    ctrl.transition_to(RuntimePhase.WARM_START)
    ctrl.transition_to(RuntimePhase.WATERMARK_SET)
    ctrl.transition_to(RuntimePhase.PRELIVE_CATCHUP)
    ctrl.transition_to(RuntimePhase.LIVE)
    assert ctrl.is_live()
    assert ctrl.allows_signal_emit() is True


def test_can_skip_prelive_to_live() -> None:
    ctrl = RuntimePhaseController()
    for phase in (
        RuntimePhase.WARM_START,
        RuntimePhase.WATERMARK_SET,
        RuntimePhase.LIVE,
    ):
        ctrl.transition_to(phase)
    assert ctrl.phase is RuntimePhase.LIVE


def test_backfill_queued_roundtrip() -> None:
    ctrl = RuntimePhaseController()
    for phase in (
        RuntimePhase.WARM_START,
        RuntimePhase.WATERMARK_SET,
        RuntimePhase.LIVE,
        RuntimePhase.BACKFILL_QUEUED,
        RuntimePhase.LIVE,
    ):
        ctrl.transition_to(phase)
    assert ctrl.allows_signal_emit() is True


def test_illegal_transition_raises() -> None:
    ctrl = RuntimePhaseController()
    with pytest.raises(InvalidPhaseTransition):
        ctrl.transition_to(RuntimePhase.LIVE)
    assert ctrl.try_transition_to(RuntimePhase.LIVE) is False
    assert ctrl.phase is RuntimePhase.BOOT


def test_idempotent_same_phase() -> None:
    ctrl = RuntimePhaseController()
    assert ctrl.transition_to(RuntimePhase.BOOT) is RuntimePhase.BOOT
    assert ctrl.history() == ["BOOT"]


def test_shutdown_from_any_phase() -> None:
    ctrl = RuntimePhaseController()
    ctrl.transition_to(RuntimePhase.WARM_START)
    assert ctrl.request_shutdown() is True
    assert ctrl.is_shutdown()
    assert ctrl.request_shutdown() is False
    with pytest.raises(InvalidPhaseTransition):
        ctrl.transition_to(RuntimePhase.LIVE)


def test_shutdown_from_boot() -> None:
    ctrl = RuntimePhaseController()
    ctrl.transition_to(RuntimePhase.SHUTDOWN)
    assert ctrl.is_shutdown()


def test_snapshot_and_in_phases() -> None:
    ctrl = RuntimePhaseController()
    ctrl.transition_to(RuntimePhase.WARM_START)
    snap = ctrl.snapshot()
    assert snap["phase"] == "WARM_START"
    assert snap["allows_signal_emit"] is False
    assert ctrl.in_phases({RuntimePhase.WARM_START, RuntimePhase.BOOT})


def test_thread_safe_transitions() -> None:
    ctrl = RuntimePhaseController()
    ctrl.transition_to(RuntimePhase.WARM_START)
    ctrl.transition_to(RuntimePhase.WATERMARK_SET)
    ctrl.transition_to(RuntimePhase.LIVE)
    errors: list[BaseException] = []

    def toggle() -> None:
        try:
            for _ in range(50):
                ctrl.try_transition_to(RuntimePhase.BACKFILL_QUEUED)
                ctrl.try_transition_to(RuntimePhase.LIVE)
        except BaseException as exc:  # noqa: BLE001 — collect for assert
            errors.append(exc)

    threads = [threading.Thread(target=toggle) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert ctrl.phase in {RuntimePhase.LIVE, RuntimePhase.BACKFILL_QUEUED}