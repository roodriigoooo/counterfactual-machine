import json
import inspect
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from counterfactual.data import features
from counterfactual.models import TrajectoryBaseline, Predictor
from counterfactual.planning import plan, select_index
from counterfactual.schemas import GoalSpec, Region
from counterfactual.scenes import generate_scene


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(4)
    model = TrajectoryBaseline()
    path = tmp_path/"model.pt"
    torch.save({"model": model.state_dict(), "name": "fixture", "seed": 4,
                "epoch": 0, "dataset_sha256": "fixture"}, path)
    return path


def test_model_permutation_invariance_and_exact_zero_identity(checkpoint):
    torch.set_num_threads(1)
    predictor = Predictor(checkpoint)
    scene = generate_scene(1)
    g, r, m = [torch.tensor(a[None]) for a in features(scene)]
    model = predictor.model
    permutation = [4, 2, 0, 3, 1]
    a, c = model(g, r, m)
    b, d = model(g, r[:, permutation], m[:, permutation])
    torch.testing.assert_close(a, b, rtol=0, atol=1e-5)
    torch.testing.assert_close(c[:, :, permutation], d, rtol=0, atol=1e-5)
    predictions, _ = predictor.forecast(scene, [0, 0])
    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_goal_independence_and_no_simulator_in_planner(checkpoint):
    import counterfactual.planning as module
    assert "from .simulator" not in inspect.getsource(module)
    predictor = Predictor(checkpoint)
    scene = generate_scene(0)
    before, _ = predictor.forecast(scene, [0])
    task = GoalSpec(region=Region(center=(1, 1), radius=.1))
    result = plan(scene, task, predictor, count=9)
    assert result["simulator_calls"] == 0
    plan(scene, task.model_copy(update={"kind": "avoid"}), predictor, count=9)
    after, _ = predictor.forecast(scene, [0])
    np.testing.assert_array_equal(before, after)


def test_goal_constraint_precedes_edit_cost_and_no_solution_is_not_infeasible():
    grid = np.array([-.12, 0, .12])
    assert select_index(grid, [-.1, -.01, .1], [False, False, True]) == 2
    assert select_index(grid, [-.1, -.01, -.1], [False]*3) is None
    assert select_index(grid, [.1, -.01, .2], [True, False, True]) == 2


def test_saved_authored_scene_end_to_end_one_verification(checkpoint, tmp_path, monkeypatch):
    from counterfactual import api as module
    calls = []
    simulator = module.simulate
    def counted(*args, **kwargs):
        calls.append(1)
        return simulator(*args, **kwargs)
    monkeypatch.setattr(module, "simulate", counted)
    scene = generate_scene(0).model_copy(update={"family": "authored"})
    goal = GoalSpec(region=Region(center=(0, 0), radius=99))
    path = tmp_path/"scene.json"
    path.write_text(json.dumps({"scene": scene.model_dump(), "goal": goal.model_dump()}))
    task = json.loads(path.read_text())
    client = TestClient(module.create_app(checkpoint, tmp_path))
    assert client.post("/api/simulate", json=task).status_code == 200
    assert len(calls) == 1
    response = client.post("/api/plan", json=task)
    assert response.status_code == 200
    result = response.json()
    assert len(calls) == 1 and result["simulator_calls"] == 0 and result["edit_m"] == 0
    response = client.post("/api/verify", json={"plan_id": result["plan_id"]})
    assert response.status_code == 200 and response.json()["status"] == "verified success"
    assert len(calls) == 1  # verified original already establishes minimal zero edit
    assert client.post("/api/verify", json={"plan_id": result["plan_id"]}).json()["new_simulator_calls"] == 0
    assert len(calls) == 1
    invalid = json.loads(path.read_text()); invalid["scene"]["bumper"]["position"] = invalid["scene"]["ball"]["position"]
    assert client.post("/api/plan", json=invalid).status_code == 422


def test_parent_split_isolation_if_pilot_exists():
    from pathlib import Path
    manifest_path = Path("data/pilot/manifest.json")
    if not manifest_path.exists():
        pytest.skip("pilot not generated")
    manifest = json.loads(manifest_path.read_text())
    by_split = {split: {r["scene"]["parent_id"] for r in manifest["records"] if r["split"] == split} for split in ["train", "validation", "test"]}
    assert not by_split["train"] & by_split["validation"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["validation"] & by_split["test"]
    assert sum(map(len, by_split.values())) == manifest["parents"]
