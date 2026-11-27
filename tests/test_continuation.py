import numpy as np
import torch

from counterfactual.continuation_experiment import prediction
from counterfactual.continuation_selection import risk, risk_fit
from counterfactual.continuation_solver import DT, bounded, fly, query
from counterfactual.subinterval import step


def corridor():
    c = torch.zeros(1, 5, 10)
    for edge, x in enumerate([1.0, 1.26]):
        c[0, edge] = torch.tensor([0.0, x, 0.0, x, 2.0, 0.02, 0.10, 0.0, 1.0, 1.0])
    return torch.tensor([[1.125, 1.0, -3.0, 0.0, 0.0, 0.0]]), c


def test_two_sequential_contacts_and_audited_overflow():
    state, c = corridor()
    result, t, posts, overflow = bounded(state, c, audit_overflow=True)
    exact = bounded(state, c, False, True)
    assert all(torch.equal(a, b) for a, b in zip((result, t, posts, overflow), exact))
    assert torch.equal(t[0].isfinite().nonzero(), torch.tensor([[0, 0], [1, 1]]))
    assert t[0, 0, 0] < t[0, 1, 1] < DT
    assert posts[0, 0, 2] > 0 and posts[0, 1, 2] < 0
    assert overflow.item()
    c[:, 1, 9] = 0
    assert torch.allclose(bounded(state, c)[0], step(state, c, DT)[0], atol=1e-6)


def test_free_flight_and_unordered_ties():
    state, c = corridor()
    c[:, :, 9] = 0
    s, t, _, over = bounded(state, c)
    assert torch.equal(s, fly(state, torch.tensor([DT]))) and not t.isfinite().any() and not over.any()
    times, _, mask = query(state, c)
    assert not mask.any() and prediction(times)[0].item() == -1
    first, second, _ = prediction(torch.tensor([[0.01, 0.01, 0.02]]))
    assert first.item() == second.item() == -1


def test_one_class_risk_fit_preserves_ties():
    rows = [{"robust": True, "features": {"a": [v, 0.0, 2.0, v * v]}} for v in (0.0, 1.0, 5.0)]
    model = risk_fit(rows, "a")
    assert model["constant_class"] == 0
    assert np.ptp([risk(row, "a", model) for row in rows]) == 0
