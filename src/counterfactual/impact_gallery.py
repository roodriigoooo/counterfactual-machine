"""Measured margins around the first competing repair gained by the two-rate tick rule."""

import json

import numpy as np
import torch

from .architecture_evaluation import forecast_table
from .competition_scenes import propose
from .continuation_experiment import write
from .geometry import apply_edit, candidates
from .goals import goal_result
from .impact_data import RATES, ROOT
from .impact_decisions import PackedForecast
from .impact_model import Impact
from .recurrent import pack
from .scenes import generate_goal
from .schemas import EditSpec
from .simulator import simulate


def run():
    torch.set_num_threads(1)
    source = json.loads((ROOT / "decisions-competing.json").read_text())
    for p in source["parents"]:
        by = {(r["mode"], r["goal"]): r for r in p["rows"]}
        for r in p["rows"]:
            if (
                r["mode"] != "tick-both"
                or not r["checks"]
                or not all(all(v) for v in r["checks"].values())
            ):
                continue
            other = by["continuous-fixed", r["goal"]]
            if other["checks"] and not all(all(v) for v in other["checks"].values()):
                break
        else:
            continue
        break
    else:
        raise ValueError("No gained repair with proposals under both rules")
    scene = propose(p["parent"])
    goal = generate_goal(8 * scene.seed + r["goal"])
    grid, valid, _ = candidates(scene, 257)
    s, c = [
        torch.from_numpy(np.stack(x))
        for x in zip(*(pack(apply_edit(scene, EditSpec(displacement=float(u)))) for u in grid[valid]))
    ]
    records = {}
    for label, model, hz in [
        ("continuous-1920", None, 1920),
        ("tick-1920", Impact("tick"), 1920),
        ("tick-3840", Impact("tick"), 3840),
    ]:
        margins = forecast_table(PackedForecast(s, c, model, hz), scene, grid, valid, [goal])[
            "margins"
        ][0]
        records[label] = [float(v) if np.isfinite(v) else None for v in margins]

    def actual(edit, hz):
        tr = simulate(scene, edit, hz)
        return tr, goal_result(tr.physics_times, tr.physics_positions, goal)["margin"]

    checks = {}
    traces = {}
    for hz in RATES:
        records[f"reference-{hz}"] = [
            float(actual(float(u), hz)[1]) if legal else None for u, legal in zip(grid, valid)
        ]
        for tag, sel in [("continuous", other), ("tick", r)]:
            vals = []
            for off in [-0.005, -0.0025, 0, 0.0025, 0.005]:
                tr, margin = actual(round(sel["edit_m"] + off, 12), hz)
                vals.append(float(margin))
                if hz == 1920 and off == 0:
                    traces[tag] = tr.states[:, :2].tolist()
            checks[f"{tag}-{hz}"] = vals
    write(
        ROOT / "gallery.json",
        {
            "parent": scene.seed,
            "goal_id": r["goal"],
            "scene": scene.model_dump(mode="json"),
            "goal": goal.model_dump(mode="json"),
            "edits": {"tick": r["edit_m"], "continuous": other["edit_m"]},
            "grid_m": grid.tolist(),
            "margins": records,
            "checks": checks,
            "paths": traces,
        },
    )


if __name__ == "__main__":
    run()
