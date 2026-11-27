"""Measure localized encounters beside matched contacts from numerically stable releases."""

import json
from collections import defaultdict

import numpy as np
import torch

from .continuation_experiment import read_split, write
from .geometry import apply_edit
from .impact_data import RATES, ROOT
from .impact_decisions import models
from .impact_model import step
from .schemas import EditSpec, SceneSpec
from .simulator import simulate
from .support_study import ROOT as SUPPORT


def intervals(tr, scene, ctx):
    ratio = tr.provenance["physics_hz"] // 60
    hz = 60 * ratio
    summaries = defaultdict(dict)
    for resp in tr.responses:
        tick = round(resp["time"] * hz)
        k = (tick - 1) // ratio
        obj = resp["object"]
        if obj in summaries[k]:
            continue
        i = tr.object_ids.index(obj)
        edge = ctx[i]
        normal = -np.array(resp["normal_ball_to_static"])
        tang = np.array([-normal[1], normal[0]])
        vel = np.array(resp["before"][2:4])
        ab = edge[3:5] - edge[1:3]
        if obj == scene.bumper.id:
            shape = "round"
        else:
            f = (tr.physics_positions[tick] - edge[1:3]) @ ab / (ab @ ab)
            shape = "face" if 0 < f < 1 else "endpoint"
        summaries[k][obj] = {
            "object": obj,
            "shape": shape,
            "edge": i,
            "clock_ms": 1000 * (tick / hz - k / 60),
            "begin": bool(resp["first_contact"]),
            "speed": float(np.linalg.norm(vel)),
            "incidence_deg": float(np.rad2deg(np.arctan2(abs(vel @ tang), max(0.0, -vel @ normal)))),
            "normal": normal.tolist(),
            "tangent": tang.tolist(),
        }
    return summaries


def run():
    torch.set_num_threads(1)
    old = json.loads((SUPPORT / "evaluation.json").read_text())["rows"]
    restarts = json.loads((SUPPORT / "restarts.json").read_text())["rows"]
    found = {
        (r["parent"], r["edit"]): r["localization_interval"]
        for r in restarts
        if r["split"] == "evaluation" and not r["rate_sensitive"]
    }
    rows, data, manifest = read_split("evaluation")
    scenes = {p["scene"]["seed"]: SceneSpec.model_validate(p["scene"]) for p in manifest["records"]}
    cases = []
    controls = []
    cache = {}
    for i, (r, orig) in enumerate(zip(rows, old, strict=True)):
        key = (r["parent"], r["edit"])
        assert key == (orig["parent"], orig["edit"])
        clean = orig["rate_divergence_observation"] is None and all(
            v["first_observation"] is None for v in orig["modes"]["bounded"].values()
        )
        if key not in found and not clean:
            continue
        scene = scenes[r["parent"]]
        geo = apply_edit(scene, EditSpec(displacement=r["edit"]))
        tr = simulate(scene, r["edit"], 1920, True)
        obs = intervals(tr, geo, data["contexts"][i].numpy())
        cache[(i, 1920)] = (tr.states, obs)
        for tick, objects in obs.items():
            if not clean and tick != found[key]:
                continue
            first = min(objects.values(), key=lambda v: (v["clock_ms"], v["object"]))
            item = {
                "branch": i,
                "parent": key[0],
                "edit": key[1],
                "tick": tick,
                "surface_count": len(objects),
                **first,
            }
            (controls if clean else cases).append(item)
    matches = []
    used = set()
    for case in cases:
        allowed = [
            v
            for v in controls
            if v["parent"] != case["parent"]
            and v["shape"] == case["shape"]
            and (v["branch"], v["tick"]) not in used
        ]

        def distance(v, case=case):
            return (
                v["begin"] != case["begin"],
                v["surface_count"] != case["surface_count"],
                abs(v["incidence_deg"] - case["incidence_deg"]) / 30
                + abs(v["speed"] - case["speed"]) / 2
                + abs(v["tick"] - case["tick"]) / 120,
            )

        control = min(allowed, key=distance)
        used.add((control["branch"], control["tick"]))
        matches.append((case, control))
    mdl = models()
    for pair in matches:
        for rec in pair:
            i, k = rec["branch"], rec["tick"]
            rec["rates"] = {}
            ctx = data["contexts"][i : i + 1]
            for hz in RATES:
                if (i, hz) not in cache:
                    scene = scenes[rec["parent"]]
                    geo = apply_edit(scene, EditSpec(displacement=rec["edit"]))
                    tr = simulate(scene, rec["edit"], hz, True)
                    cache[(i, hz)] = (tr.states, intervals(tr, geo, ctx[0].numpy()))
                states, obs = cache[(i, hz)]
                measurement = obs.get(k, {}).get(rec["object"])
                cell = {"observed": measurement, "models": {}}
                rec["rates"][str(hz)] = cell
                if measurement is None:
                    continue
                n = np.array(measurement["normal"])
                tang = np.array(measurement["tangent"])
                actual = states[k + 1]
                for name, m in mdl.items():
                    state, clocks, _, _ = step(
                        torch.tensor(states[k : k + 1], dtype=ctx.dtype), ctx, m, hz
                    )
                    pred = state[0].numpy()
                    delta = pred - actual
                    clock = float(clocks[0, :, rec["edge"]].min())
                    cell["models"][name] = {
                        "clock_ms": clock * 1000 if np.isfinite(clock) else None,
                        "normal_displacement_error_mm": float(1000 * delta[:2] @ n),
                        "tangent_velocity_error_mm_s": float(1000 * delta[2:4] @ tang),
                        "velocity_error_mm_s": float(1000 * np.linalg.norm(delta[2:4])),
                        "outgoing_tangent_m_s": float(pred[2:4] @ tang),
                        "actual_outgoing_tangent_m_s": float(actual[2:4] @ tang),
                    }
    write(
        ROOT / "matched-contacts.json", {"pairs": [{"localized": a, "control": b} for a, b in matches]}
    )
    print(len(matches), "pairs", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        run()
