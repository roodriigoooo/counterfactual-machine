"""Retrospective exact-state restarts at every observation preceding a lasting divergence."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .architecture_data import digest
from .continuation_experiment import read_split, write
from .continuation_solver import bounded
from .schemas import SceneSpec
from .simulator import simulate
from .support_study import PROTOCOL, RATES, ROOT, err, observations


def restart_curve(ref, context, first, n):
    """Each row starts at its own exact state. Evaluate every start on the identical window."""
    states = torch.as_tensor(ref[: first + 1], dtype=context.dtype).clone()
    peak = np.zeros(len(states))
    for tick in range(first + n):
        if tick >= first:
            peak = np.maximum(
                peak, np.linalg.norm(states[:, :2].numpy() - ref[tick, :2], axis=-1)
            )
        if tick < first + n - 1:
            count = min(tick + 1, len(states))
            states[:count] = bounded(states[:count], context.expand(count, -1, -1))[0]
    return peak


def component_restarts(ref, context, tick, first, n):
    """Counterfactual resets on the same exact-start transition, at a common target."""
    actual = torch.as_tensor(ref[tick : tick + 2], dtype=context.dtype)
    pred = bounded(actual[:1], context[None])[0]
    labels = [
        "uncorrected",
        "position",
        "velocity",
        "position+velocity",
        "spin",
        "position+spin",
        "velocity+spin",
        "position+velocity+spin",
    ]
    states = pred.expand(8, -1).clone()
    for i in range(8):
        for flag, cols in ((1, slice(0, 2)), (2, slice(2, 4)), (4, slice(5, 6))):
            if i & flag:
                states[i, cols] = actual[1, cols]
    peak = np.zeros(8)
    for sample in range(tick + 1, first + n):
        if sample >= first:
            peak = np.maximum(
                peak, np.linalg.norm(states[:, :2].numpy() - ref[sample, :2], axis=-1)
            )
        if sample < first + n - 1:
            states = bounded(states, context.expand(len(states), -1, -1))[0]
    return dict(zip(labels, peak.tolist()))


def study(output=ROOT):
    torch.set_num_threads(1)
    cfg = json.loads(PROTOCOL.read_text())
    n, limit = cfg["consecutive_samples"], cfg["threshold_m"]
    records = []
    for split in cfg["splits"]:
        cohort = json.loads((output / f"{split}.json").read_text())
        rows, data, manifest = read_split(split)
        scenes = {
            p["scene"]["seed"]: SceneSpec.model_validate(p["scene"])
            for p in manifest["records"]
        }
        for i, (r, prev) in enumerate(zip(rows, cohort["rows"], strict=True)):
            if (r["parent"], r["edit"]) != (prev["parent"], prev["edit"]):
                raise ValueError(
                    "The archived forecast and reference branch orders differ"
                )
            first = prev["joint_divergence_observation"]
            if first is None:
                continue
            rates = {}
            tr_by_rate = {}
            for hz in RATES:
                tr = simulate(scenes[r["parent"]], r["edit"], hz, record_responses=True)
                curve = restart_curve(tr.states, data["contexts"][i], first, n)
                rates[str(hz)] = {"peak_error_by_restart_m": curve.tolist()}
                tr_by_rate[str(hz)] = tr
            curves = np.array([v["peak_error_by_restart_m"] for v in rates.values()])
            ok = np.all(curves <= limit, axis=0)
            if ok[0]:
                raise ValueError("Branch no longer reproduces the release disagreement")
            # Use the start of the terminal passing run. Earlier rescues can reverse.
            last_fail = int(np.flatnonzero(~ok)[-1])
            resolved = last_fail < first
            for hz, tr in tr_by_rate.items():
                if resolved:
                    t = last_fail
                    context = data["contexts"][i : i + 1]
                    state = torch.as_tensor(tr.states[t : t + 1], dtype=context.dtype)
                    local = bounded(state, context)[0].numpy()
                    frame = observations(tr)[0]
                    components = component_restarts(tr.states, context[0], t, first, n)
                    # Orientation is dynamically irrelevant for an isotropic disk.
                    np.testing.assert_allclose(
                        [
                            components["uncorrected"],
                            components["position+velocity+spin"],
                        ],
                        rates[hz]["peak_error_by_restart_m"][t : t + 2],
                        rtol=2e-4,
                        atol=5e-6,
                    )
                    rates[hz].update(
                        component_peak_m=components,
                        contact=frame.get(t, {}),
                        prior_contact=frame.get(t - 1, {}),
                        local_velocity_error_m_s=float(
                            err(local[0], tr.states[t + 1], 2, 4)
                        ),
                        local_position_error_m=float(err(local[0], tr.states[t + 1])),
                    )
            records.append(
                {
                    "split": split,
                    "parent": r["parent"],
                    "edit": r["edit"],
                    "joint_crossing_observation": first,
                    "rate_sensitive": prev["rate_divergence_observation"] is not None,
                    "localization_interval": last_fail if resolved else None,
                    "rescue_reversals": int(np.sum(ok[:-1] & ~ok[1:])),
                    "rates": rates,
                }
            )
    write(
        output / "restarts.json",
        {
            "protocol": digest(PROTOCOL),
            "observation_window": n,
            "limit_m": limit,
            "rows": records,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    with torch.inference_mode():
        study(parser.parse_args().output)
