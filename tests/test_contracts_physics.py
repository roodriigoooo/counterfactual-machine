import numpy as np
import pytest
from counterfactual.schemas import Ball, Bumper, EditSpec, GoalSpec, Region, SceneSpec, Segment
from counterfactual.geometry import apply_edit, candidates
from counterfactual.goals import goal_result
from counterfactual.scenes import generate_scene
from counterfactual.simulator import simulate


def flight(**kwargs):
    return SceneSpec(id="fixture", parent_id="fixture", ball=Ball(position=(0.8, 3.5), velocity=(1, 0)),
                     bumper=Bumper(position=(4.7, 2.0)), **kwargs)


def test_roundtrip_edit_permissions_and_immutability():
    scene = generate_scene(0)
    assert SceneSpec.model_validate_json(scene.model_dump_json()) == scene
    assert apply_edit(scene, EditSpec()).bumper.position == scene.bumper.position
    edited = apply_edit(scene, EditSpec(displacement=scene.ball.radius))
    assert edited.ball == scene.ball and edited.segments == scene.segments
    assert edited.bumper.position[0] != scene.bumper.position[0]
    for edit in [EditSpec(displacement=0.121), EditSpec(object_id="ball")]:
        with pytest.raises(ValueError):
            apply_edit(scene, edit)
    with pytest.raises(ValueError):
        EditSpec(displacement=float("nan"))
    grid, valid, _ = candidates(scene)
    assert len(grid) == 257 and grid[128] == 0 and valid.any()


def test_overlap_rejected_without_clipping():
    scene = flight().model_copy(update={"bumper": Bumper(position=(1.25, 3.5))})
    with pytest.raises(ValueError, match="overlap"):
        apply_edit(scene, EditSpec(displacement=-0.12))


def test_closed_center_goals_full_interval_and_brief_crossing():
    t = [0, 1, 2]
    p = [[-1, 0], [1, 0], [2, 0]]
    goal = GoalSpec(region=Region(center=(0, 0), radius=0.01))
    assert goal_result(t, p, goal)["success"]  # endpoints alone miss this
    assert not goal_result(t, p, goal.model_copy(update={"kind": "avoid"}))["success"]
    assert not goal_result(t, p, goal.model_copy(update={"start": 1.0}))["success"]
    touch = GoalSpec(region=Region(center=(-1, 1), radius=1), end=0)
    assert goal_result(t, p, touch)["success"]
    assert not goal_result(t, p, touch.model_copy(update={"kind": "avoid"}))["success"]
    half = GoalSpec(region=Region(kind="halfplane", normal=(1, 0), offset=1.5))
    assert goal_result(t, p, half)["success"]


def test_free_flight_and_resolution():
    scene = flight()
    a = simulate(scene)
    t = a.times
    np.testing.assert_allclose(a.states[:, 0], 0.8+t, atol=1e-10)
    # Pymunk integrates positions using the previous velocity (explicit Euler).
    np.testing.assert_allclose(a.states[:, 1], 3.5-9.81*t*(t-1/1920)/2, atol=1e-10)
    assert not a.contacts.any()
    b = simulate(scene, physics_hz=3840)
    assert np.max(np.abs(a.states[:, :2]-b.states[:, :2])) < 0.021


def test_determinism_canonical_order_and_short_contact_capture():
    scene = generate_scene(1)
    a, b = simulate(scene), simulate(scene)
    np.testing.assert_array_equal(a.states, b.states)
    assert a.events == b.events
    reverse = scene.model_copy(update={"segments": tuple(reversed(scene.segments))})
    np.testing.assert_array_equal(a.states, simulate(reverse).states)
    assert any(e["phase"] == "begin" for e in a.events)
    assert a.contacts.any()
    for event in a.events:
        if event["phase"] == "begin":
            row = max(0, int(round(event["time"]*1920)-1)//32)
            assert a.contacts[row, a.object_ids.index(event["objects"][1])] == 1


def test_bounce_and_initial_touch():
    floor = Segment(id="floor", a=(0.1, 0.2), b=(5.9, 0.2))
    s = flight(segments=(floor,))
    tr = simulate(s)
    assert tr.states[:, 3].max() > 1
    touch = s.model_copy(update={"ball": Ball(position=(0.8, 0.36))})
    assert simulate(touch).contacts[0].any()


def test_ramp_and_near_miss_fixtures():
    tr = simulate(generate_scene(1))
    assert tr.contacts[:, tr.object_ids.index("ramp")].any()
    scene = flight()
    assert not simulate(scene).contacts[:, 0].any()
