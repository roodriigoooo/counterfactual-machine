import numpy as np
import pytest
import torch

from counterfactual.continuation_solver import bounded, fly
from counterfactual.persistent_support import support_step
from counterfactual.recurrent import DT, pack
from counterfactual.relay_mass import measure_mass
from counterfactual.relay_world import make_relay, release
from counterfactual.support_resets import restart_curve
from counterfactual.support_study import first_run, fixture_scenes


def test_only_previous_isolated_face_can_hold_a_disk():
    scene = fixture_scenes()[0]
    state, ctx = [torch.tensor(v[None]) for v in pack(scene)]
    previous = torch.zeros(1, 5, dtype=torch.bool)
    following, _, held = support_step(state, ctx, previous)
    assert not held.any() and torch.equal(following, bounded(state, ctx)[0])
    previous[0, 1] = True  # bumper sorts before floor
    following, _, held = support_step(state, ctx, previous)
    assert held[0, 1] and torch.allclose(following, state, atol=1e-6)
    c = ctx.clone()
    c[0, 0, 1:5] = torch.tensor([1.45, 0.34, 1.45, 0.34])
    result, _, held = support_step(state, c, previous)
    assert not held.any() and torch.equal(result, bounded(state, c)[0])
    for change in [(0, 5.899), (3, 1.0)]:  # near end, or strongly separating
        s = state.clone()
        s[0, change[0]] = change[1]
        result, _, held = support_step(s, ctx, previous)
        assert not held.any() and torch.equal(result, bounded(s, ctx)[0])


def test_a_gentle_departure_cannot_be_captured_by_a_tensile_constraint():
    # On a steep incline a sub-slop hop can still be separating at frame end.
    ctx = torch.zeros(1, 5, 10)
    tangent = torch.tensor([0.1, 1.0])
    tangent /= tangent.norm()
    normal = torch.tensor([-tangent[1], tangent[0]])
    ctx[0, 0] = torch.tensor([0.0, 0.0, 0.0, 0.2, 2.0, 0.04, 0.12, 0.2, 0.4, 1.0])
    state = torch.zeros(1, 6)
    state[0, :2] = tangent + 0.16 * normal
    state[0, 2:4] = 0.03 * normal
    previous = torch.zeros(1, 5, dtype=torch.bool)
    previous[0, 0] = True
    following, _, held = support_step(state, ctx, previous)
    assert not held.any() and torch.equal(following, bounded(state, ctx)[0])


def test_exact_restart_uses_the_same_global_observation_window():
    assert first_run([False, True, True, False, True, True, True], 3) == 4
    assert first_run([True, False, True, True], 3) is None
    ctx = torch.zeros(5, 10)
    s = torch.zeros(1, 6)
    states = [s[0].numpy().copy()]
    for i in range(12):
        s = fly(s, torch.tensor([DT]))
        if i == 3:
            s[0, 2] += 3.0
        states.append(s[0].numpy().copy())
    errs = restart_curve(np.array(states), ctx, 8, 4)
    assert np.all(errs[:4] > 0.3)
    assert np.max(errs[4:]) < 1e-6


def test_relay_transfer_is_required_and_the_probe_recovers_mass():
    world = make_relay(100000)
    trace, begins, travel = release(world, 0.105, 1920)
    without, disabled, absent = release(world, 0.105, 1920, pair_enabled=False)
    assert begins["deflection"] < begins["transfer"] and "transfer" not in disabled
    assert absent == 0 and travel > 0.35
    tick = begins["transfer"]
    np.testing.assert_array_equal(trace[:tick, 1], without[:tick, 1])
    assert measure_mass(world.receiver, 1920)["estimated_kg"] == pytest.approx(
        world.receiver.mass
    )
    with pytest.raises(ValueError, match="exceeds"):
        release(world, 0.5, 1920)
