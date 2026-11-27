from dataclasses import replace

import numpy as np
import pytest
import torch

from counterfactual.impact_model import Impact, clock_at_tick, forecast, step
from counterfactual.recurrent import DT, pack
from counterfactual.relay_forecast import forecast as relay_forecast
from counterfactual.relay_forecast import layout
from counterfactual.relay_world import release
from counterfactual.scenes import generate_scene


def test_impact_clocks_use_the_global_tick_and_preserve_rest():
    q = 1 / 1920
    t = torch.tensor([1.3 * q, 0.0, 0.01 * q])
    h = torch.tensor([DT - 0.4 * q, DT - 0.4 * q, 0.01 * q])
    y = clock_at_tick(t, h, q)
    assert torch.allclose(y, torch.tensor([1.6 * q, 0.0, 0.01 * q]), atol=1e-8)
    assert torch.all(y <= h)


def test_zero_learned_residuals_reproduce_the_tick_trajectory():
    s, c = (
        torch.tensor(np.stack(x))
        for x in zip(*(pack(generate_scene(i)) for i in [111000, 111001, 111002]))
    )
    for hz in (1920, 3840):
        target = forecast(s, c, Impact("tick"), hz)
        for name in ("joint", "direct"):
            assert torch.equal(forecast(s, c, Impact(name), hz), target)


def test_restitution_orders_gravity_at_the_detection_tick():
    # Isolated vertical hit. The disk has unit restitution and no tangential motion.
    c = torch.zeros(1, 5, 10, dtype=torch.float64)
    c[0, 0] = torch.tensor([0, 0, 1, 3, 1, 0.04, 0.12, 0, 1, 1], dtype=c.dtype)
    s = torch.tensor([[1.5, 1.166, -0.0, -1.0, 0.0, 0.0]], dtype=c.dtype)
    hz = 1920
    tick, t, _, _ = step(s, c, Impact("tick"), hz)
    out, ot, _, _ = step(s, c, Impact("ordered"), hz)
    assert torch.equal(ot, t) and torch.isfinite(t[0, 0, 0])
    assert abs(float(tick[0, 3] - out[0, 3]) - 9.81 / hz) < 1e-9


def test_masked_mass_and_disabled_relay_preserve_the_information_boundary():
    w = layout(123010, True)
    other = replace(w, receiver=w.receiver.model_copy(update={"mass": 4.5}))
    edits = [-0.09, 0.0, 0.09]
    p = relay_forecast(w, edits, [1.2] * 3)
    q = relay_forecast(other, edits, [1.2] * 3)
    for x, y in zip(p, q):
        np.testing.assert_array_equal(x, y)
    disabled = relay_forecast(w, edits, [1.2] * 3, pair_enabled=False)
    assert np.all(disabled[1] == -1) and not disabled[2].any()
    for i, tick in enumerate(p[1]):
        prefix = np.arange(121) / 60 < (tick / 480 if tick >= 0 else 3.0)
        np.testing.assert_array_equal(p[0][i, prefix, 1], disabled[0][i, prefix, 1])
    assert p[3] < 1e-12
    with pytest.raises(ValueError):
        relay_forecast(w, [0.5], [1.0])
    assert release(w, 0.0, 1920)[0].shape == (3841, 2, 2)


def test_rerun_checks_frozen_weights(tmp_path):
    from counterfactual.impact_data import save_weights

    path = tmp_path / "model.pt"
    model = Impact("joint")
    save_weights(path, model)
    previous = path.read_bytes()
    save_weights(path, model)
    with torch.no_grad():
        model.net[-1].bias.add_(1)
    with pytest.raises(ValueError, match="frozen checkpoint"):
        save_weights(path, model)
    assert path.read_bytes() == previous
