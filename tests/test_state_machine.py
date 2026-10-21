"""State machine timing uses injected frame timestamps."""

import pytest

from engine.state_machine import PiracyState, PiracyStateMachine


def test_threshold_uses_injected_timestamps() -> None:
    machine = PiracyStateMachine(piracy_threshold_sec=10.0, soft_decay_sec=3.0)
    assert machine.update(True, True, now=0.0) == PiracyState.IDLE
    assert machine.update(True, True, now=5.0) == PiracyState.SUSPICIOUS
    assert machine.active_seconds == pytest.approx(5.0)
    assert machine.update(True, True, now=8.0) == PiracyState.ALERT_PENDING
    assert machine.update(True, True, now=10.0) == PiracyState.PIRACY_ALERT
    assert machine.active_seconds == pytest.approx(10.0)


def test_soft_decay_then_reset() -> None:
    machine = PiracyStateMachine(piracy_threshold_sec=10.0, soft_decay_sec=3.0)
    machine.update(True, True, now=0.0)
    machine.update(True, True, now=4.0)
    machine.update(False, False, now=6.0)
    assert machine.state == PiracyState.SUSPICIOUS
    assert machine.active_seconds == pytest.approx(3.5)
    machine.update(False, False, now=10.0)
    assert machine.state == PiracyState.IDLE
    assert machine.active_seconds == 0.0


def test_reset_clears_progress() -> None:
    machine = PiracyStateMachine(piracy_threshold_sec=10.0)
    machine.update(True, True, now=0.0)
    machine.update(True, True, now=2.0)
    machine.reset(now=2.0)
    assert machine.state == PiracyState.IDLE
    assert machine.active_seconds == 0.0
    assert machine.lowered_seconds == 0.0


def test_finalize_drops_lost_track() -> None:
    machine = PiracyStateMachine(piracy_threshold_sec=10.0)
    machine.update(True, True, now=0.0)
    machine.update(True, True, now=3.0)
    assert machine.finalize(now=3.0) == PiracyState.IDLE
    assert machine.active_seconds == 0.0


def test_missing_timestamp_requires_clock() -> None:
    machine = PiracyStateMachine(piracy_threshold_sec=10.0)
    with pytest.raises(ValueError, match="frame timestamp"):
        machine.update(True, True)


def test_seek_gap_does_not_add_minutes() -> None:
    machine = PiracyStateMachine(piracy_threshold_sec=30.0, max_step_sec=0.5)
    machine.update(True, True, now=0.0)
    machine.update(True, True, now=120.0)
    assert machine.active_seconds == pytest.approx(0.5)
    assert machine.state == PiracyState.SUSPICIOUS


def test_threshold_change_promotes_on_the_next_frame() -> None:
    machine = PiracyStateMachine(piracy_threshold_sec=10.0)
    machine.update(True, True, now=0.0)
    machine.update(True, True, now=9.0)
    assert machine.state == PiracyState.ALERT_PENDING
    machine.piracy_threshold_sec = 5.0
    assert machine.update(True, True, now=9.2) == PiracyState.PIRACY_ALERT


def test_two_tracks_are_independent_and_grace_holds_an_alert() -> None:
    from engine.state_machine import TrackStateManager

    machines: dict[int, PiracyStateMachine] = {}
    manager = TrackStateManager(machines, piracy_threshold_sec=1.0, max_step_sec=1.0, grace_sec=2.0)
    first = manager.machine_for(1)
    second = manager.machine_for(2)
    first.update(True, True, now=0.0)
    second.update(True, True, now=0.0)
    assert first.update(True, True, now=1.0) == PiracyState.PIRACY_ALERT
    assert second.update(False, False, now=1.0) != PiracyState.PIRACY_ALERT
    assert manager.retire({1}, now=1.2) == [2]
    assert 1 in machines
    assert manager.retire(set(), now=1.5) == []
    assert 1 in machines
    assert manager.retire(set(), now=4.0) == [1]
    assert 1 not in machines


def test_injectable_clock_for_tests() -> None:
    ticks = iter([0.0, 5.0, 10.0])
    machine = PiracyStateMachine(piracy_threshold_sec=10.0, clock=lambda: next(ticks))
    machine.update(True, True)
    machine.update(True, True)
    assert machine.update(True, True) == PiracyState.PIRACY_ALERT
