"""Simulator-only reference, kept separate from the learned planner."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from .geometry import candidates
from .goals import goal_result
from .scenes import generate_scene, generate_goal
from .simulator import simulate, PHYSICS_HZ


def sweep(scene, goal, count=257, physics_hz=PHYSICS_HZ):
    started = perf_counter()
    grid, valid, reasons = candidates(scene, count)
    margins = np.full(count, np.nan)
    success = np.zeros(count, dtype=bool)
    bumper_contacts = np.zeros(count, dtype=bool)
    paths, trajectories = [], []
    for i, u in enumerate(grid):
        if not valid[i]:
            paths.append(None)
            trajectories.append(None)
            continue
        tr = simulate(scene, float(u), physics_hz)
        result = goal_result(tr.physics_times, tr.physics_positions, goal)
        margins[i], success[i] = result["margin"], result["success"]
        bumper_contacts[i] = tr.contacts[:, tr.object_ids.index(scene.bumper.id)].any()
        paths.append(tr.states[:, :2])
        trajectories.append(tr)
    passing = np.flatnonzero(success)
    selected = min(passing, key=lambda i: (abs(grid[i]), -margins[i], grid[i])) if len(passing) else None
    return {"grid": grid, "valid": valid, "reasons": reasons, "margins": margins,
            "success": success, "bumper_contacts": bumper_contacts, "paths": paths, "trajectories": trajectories,
            "selected_index": None if selected is None else int(selected),
            "simulator_calls": int(valid.sum()), "seconds": perf_counter()-started}


def numerical_audit(output: Path, parents=32, physics_hz=PHYSICS_HZ):
    started = perf_counter()
    rows = []
    for seed in range(parents):
        scene = generate_scene(seed)
        grid, valid, _ = candidates(scene, 9)
        for u in grid[valid]:
            a, b = simulate(scene, float(u), physics_hz), simulate(scene, float(u), physics_hz*2)
            repeat = simulate(scene, float(u), physics_hz)
            goals = [generate_goal(20_000+seed*4+j) for j in range(4)]
            labels = [goal_result(a.physics_times, a.physics_positions, g)["success"] for g in goals]
            fine = [goal_result(b.physics_times, b.physics_positions, g)["success"] for g in goals]
            sampled = [goal_result(a.times, a.states[:, :2], g)["success"] for g in goals]
            rows.append({"parent_id": scene.parent_id, "family": scene.family, "edit": float(u),
                         "repeat_max_m": float(np.max(np.abs(a.states-repeat.states))),
                         "repeat_events_equal": a.events == repeat.events,
                         "resolution_rmse_m": float(np.sqrt(np.mean((a.states[:, :2]-b.states[:, :2])**2))),
                         "resolution_max_m": float(np.linalg.norm(a.states[:, :2]-b.states[:, :2], axis=1).max()),
                         "contact_label_disagreements": int(np.sum(a.contacts != b.contacts)),
                         "goal_count": len(goals),
                         "resolution_goal_disagreements": sum(x != y for x, y in zip(labels, fine)),
                         "sampled_goal_disagreements": sum(x != y for x, y in zip(labels, sampled))})
    goals_n = sum(r["goal_count"] for r in rows)
    resolution = sum(r["resolution_goal_disagreements"] for r in rows)/goals_n
    sampling = sum(r["sampled_goal_disagreements"] for r in rows)/goals_n
    summary = {"physics_hz": physics_hz, "comparison_hz": physics_hz*2,
               "parents": parents, "branches": len(rows), "goal_checks": goals_n,
               "repeat_max": max(r["repeat_max_m"] for r in rows),
               "repeat_events_equal": all(r["repeat_events_equal"] for r in rows),
               "resolution_goal_disagreement": resolution, "sampling_goal_disagreement": sampling,
               "resolution_rmse_median_m": float(np.median([r["resolution_rmse_m"] for r in rows])),
               "resolution_rmse_max_m": max(r["resolution_rmse_m"] for r in rows),
               "gate_passed": resolution < .01 and sampling < .005 and all(r["repeat_max_m"] == 0 and r["repeat_events_equal"] for r in rows),
               "seconds": perf_counter()-started,
               "scope": "ordinary pilot parents; independent natural goals; not a boundary robustness guarantee"}
    output.mkdir(parents=True, exist_ok=True)
    (output/"audit.json").write_text(json.dumps({"summary": summary, "branches": rows}, indent=2))
    return summary
