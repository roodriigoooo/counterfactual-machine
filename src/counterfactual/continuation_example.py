"""Select a verified illustration of a tolerance boundary on the enriched cohort."""

import json

import numpy as np
import torch

from .architecture_evaluation import forecast_table
from .competition_scenes import disable_surface
from .continuation_data import DATA, PROTOCOL, REPORT, window_label
from .continuation_experiment import write
from .continuation_selection import GridForecast
from .geometry import apply_edit, candidates
from .goals import goal_result
from .scenes import generate_goal
from .schemas import EditSpec, SceneSpec
from .simulator import simulate
from .tolerance import successful_components
from .tolerance_experiment import choose


def result(tr, g):
    return goal_result(tr.physics_times, tr.physics_positions, g)


def select():
    torch.set_num_threads(1)
    cfg = json.loads(PROTOCOL.read_text())["selection"]
    records = json.loads((DATA / "evaluation/manifest.json").read_text())["records"]
    for parent in records:
        example = search_parent(parent, cfg)
        if example:
            return example
    return {"found": False}


def component(grid, margins, goal, index):
    ok = [v is not None and (v >= -1e-9 if goal.kind == "reach" else v > 1e-9) for v in margins]
    return next(c for c in successful_components(grid, ok) if c["first"] <= index <= c["last"])


def search_parent(parent, cfg):
    scene = SceneSpec.model_validate(parent["scene"])
    grid, valid, _ = candidates(scene, 257)
    goals = [generate_goal(scene.seed * 8 + j) for j in range(8)]
    table = forecast_table(GridForecast("bounded", {}), scene, grid, valid, goals)
    zero = simulate(scene)
    cache = {}

    def traj(i, hz):
        if (i, hz) not in cache:
            cache[i, hz] = simulate(scene, float(grid[i]), hz)
        return cache[i, hz]

    for j, g in enumerate(goals):
        base = result(zero, g)
        if base["success"]:
            continue
        i = choose(scene, grid, valid, table, j, base, "tolerant", cfg["gamma"], cfg["epsilon"])
        if i is None:
            continue
        checks = {
            str(hz): [result(simulate(scene, float(grid[i] + off), hz), g) for off in cfg["offsets"]]
            for hz in (1920, 3840)
        }
        if not all(r["success"] for rates in checks.values() for r in rates):
            continue
        coarse = [result(traj(k, 1920), g)["margin"] if valid[k] else None for k in range(257)]

        comp = component(grid, coarse, g, i)
        for inside, outside in [(comp["first"], comp["first"] - 1), (comp["last"], comp["last"] + 1)]:
            if not (0 <= outside < len(grid) and valid[outside]):
                continue
            if any(
                not result(traj(inside, hz), g)["success"] or result(traj(outside, hz), g)["success"]
                for hz in (1920, 3840)
            ):
                continue
            branches = []
            for k in (inside, outside):
                tr = traj(k, 1920)
                edited = apply_edit(scene, EditSpec(displacement=float(grid[k])))
                audits = {str(hz): window_label(traj(k, hz), edited, parent["tick"]) for hz in (1920, 3840)}
                if (
                    any(a["tie"] for a in audits.values())
                    or audits["1920"]["order"] != audits["3840"]["order"]
                ):
                    break
                branches.append(
                    {"index": k, "edit": float(grid[k]), "path": tr.states[:, :2].tolist(), "audits": audits}
                )
            if (
                len(branches) != 2
                or branches[0]["audits"]["1920"]["order"] == branches[1]["audits"]["1920"]["order"]
            ):
                continue
            fine = [result(traj(k, 3840), g)["margin"] if valid[k] else None for k in range(257)]
            pred = table["margins"][j]
            return {
                "parent": scene.seed,
                "tick": parent["tick"],
                "goal": g.model_dump(),
                "scene": scene.model_dump(),
                "grid": grid.tolist(),
                "coarse": coarse,
                "fine": fine,
                "predicted": [float(v) if np.isfinite(v) else None for v in pred],
                "components": [comp, component(grid, fine, g, i)],
                "selected": float(grid[i]),
                "index": i,
                "checks": checks,
                "branches": branches,
            }
    return None


def export_fields(demo):
    if demo.get("found") is False:
        return
    from .schemas import GoalSpec

    scene = SceneSpec.model_validate(demo["scene"])
    goal = GoalSpec.model_validate(demo["goal"])
    for label, shift in [
        ("proposed", demo["selected"]),
        ("inside", demo["branches"][0]["edit"]),
        ("outside", demo["branches"][1]["edit"]),
    ]:
        edited = apply_edit(scene, EditSpec(displacement=shift))
        field = {"scene": edited.model_dump(), "goal": demo["goal"], "original_bumper_shift_m": shift}
        replay = simulate(edited)
        if result(replay, goal)["success"] != (label != "outside"):
            raise ValueError("Imported field changed the demonstrated outcome")
        write(REPORT / "fields" / f"{label}.json", field)
    parked = disable_surface(scene, scene.bumper.id)
    demo["parked_bumper_control"] = {
        str(hz): result(simulate(parked, physics_hz=hz), goal) for hz in (1920, 3840)
    }


if __name__ == "__main__":
    demo = select()
    export_fields(demo)
    write(REPORT / "example.json", demo)
    print(demo.get("parent"), demo.get("components"))
