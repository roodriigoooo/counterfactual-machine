from dataclasses import replace

import numpy as np
import pytest

from counterfactual.relay_encounter import gap, trace
from counterfactual.relay_forecast import forecast, layout
from counterfactual.relay_launch import handoff
from counterfactual.relay_probability import band_radius, probability
from counterfactual.relay_world import release


def test_pass_through_and_reference_contact_labels_are_independent():
    world = layout(123002, True)
    for edit in (0.0375, -0.105):
        baseline, expected, _ = release(world, edit, 1920)
        callbacks = []
        observed, events, _ = release(world, edit, 1920, static_begins=callbacks)
        np.testing.assert_array_equal(observed, baseline)
        assert events == expected and len(callbacks) == len(set(callbacks))
        assert (events["deflection"], 0, "bumper") in callbacks
        free, disabled, _ = release(world, edit, 1920, pair_enabled=False)
        assert "transfer" not in disabled
        minimum = gap(free, world.scene.ball.radius + world.receiver.radius).min()
        assert (minimum <= 0) == ("transfer" in events)


def test_forecast_restart_preserves_the_clock_and_memoryless_suffix():
    world = layout(123002, True)
    edits = [0.0375, 0.0]
    state, _, tick = trace(world, edits, 480, enabled=True)
    path, actual, *_ = forecast(world, edits, world.receiver.mass)
    np.testing.assert_array_equal(tick, actual)
    np.testing.assert_array_equal(path, state[::8, ..., :2].swapaxes(0, 1))
    start = 240
    path2, later, *_ = forecast(
        world, edits, world.receiver.mass, start_tick=start, initial_state=state[start]
    )
    np.testing.assert_array_equal(later, tick)
    np.testing.assert_array_equal(path2, path[:, start // 8 :])
    with pytest.raises(ValueError, match="restart"):
        forecast(world, edits, world.receiver.mass, start_tick=start)


@pytest.mark.parametrize("seed", [123001, 123002])
def test_bumper_handoff_reconstructs_fine_contact_state(seed):
    world = layout(seed, True)
    edits = np.linspace(-0.12, 0.12, 7)
    events = handoff(world, edits, 480, 1920)
    assert sum(map(len, events.values())) >= 6
    baseline, mask, _ = trace(world, edits, 1920)
    for destination, entries in events.items():
        for e in entries:
            onset = np.flatnonzero(mask[:, e.row] & 1)[0]
            assert onset == e.detection_tick
            assert 1 <= 4 * destination - onset + 1 <= 4
            np.testing.assert_allclose(
                e.outgoing_velocity, baseline[onset, e.row, 0, 2:4], atol=2e-10
            )
            np.testing.assert_allclose(
                e.state, baseline[4 * destination, e.row, 0], atol=2e-10
            )
            assert abs(np.linalg.norm(e.normal) - 1) < 1e-12
    moving = replace(
        world,
        scene=world.scene.model_copy(
            update={
                "ball": world.scene.ball.model_copy(update={"velocity": (0.1, 0.0)})
            }
        ),
    )
    assert not handoff(moving, edits, 480, 1920)
    plain = forecast(moving, edits, 1.2)
    with_handoff = forecast(moving, edits, 1.2, first_bumper_hz=1920)
    for a, b in zip(plain, with_handoff, strict=True):
        np.testing.assert_array_equal(a, b)


def test_group_calibration_uses_all_edits_and_both_rates():
    center = np.zeros((9, 3))
    reference = np.zeros((9, 3, 2))
    reference[:, 2, 1] = np.arange(1, 10)
    assert band_radius(reference, center, np.ones_like(center), 0.1) == (9.0, 9)
    with pytest.raises(ValueError, match="Too few"):
        band_radius(reference[:3], center[:3], np.ones_like(center[:3]), 0.1)
    p = probability(np.array([-1e5, -2.0, 0.0, 2.0, 1e5]), 1.0)
    assert np.all(np.diff(p) < 0) and p[2] == 0.5 and p[-1] > 0
    np.testing.assert_allclose(p, 1 - p[::-1], atol=1e-15)
