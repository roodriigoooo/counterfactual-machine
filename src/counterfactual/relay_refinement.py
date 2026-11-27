"""Gate fresh relay decisions on complete-release improvement with a single finer clock."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from .relay_encounter import ROOT, load, sha, write
from .relay_forecast import forecast, layout
from .relay_probability import estimate
from .relay_world import release
from .tolerance import select_with_tolerance

PROTOCOL = Path("configs/relay-encounter-decisions.json")


PHASE_PROTOCOL = Path("configs/relay-bumper-phase.json")


def measure(parent, stage, cfg):
    world = layout(parent, True)
    grid = np.linspace(-world.scene.ball.radius, world.scene.ball.radius, cfg["grid"])
    masses = np.full(len(grid), world.receiver.mass)
    arms, paths = {}, {}
    for name in cfg["methods"]:
        hz = 480 if name == "handoff" else int(name)
        kwargs = {"first_bumper_hz": 1920} if name == "handoff" else {}
        start = perf_counter()
        path, tick, travel, _ = forecast(world, grid, masses, hz, **kwargs)
        arms[name] = {
            "grid_s": perf_counter() - start,
            "pair_ticks": tick.tolist(),
            "travel_m": travel.tolist(),
            "rates": {},
        }
        paths[name] = path
    row = {"parent": parent, "arms": arms}
    if "handoff" in arms:
        from .relay_launch import handoff

        row["handoff_cells"] = sum(map(len, handoff(world, grid, 480, 1920).values()))
    if stage == "confirmation":
        for hz in cfg["references_hz"]:
            sims = [release(world, float(u), hz) for u in grid]
            ref = np.stack([s[0][:: hz // 60] for s in sims])
            truth = np.array(["transfer" in s[1] for s in sims])
            for k, p in paths.items():
                error = np.sqrt(((p - ref) ** 2).sum(-1).mean(1))
                predicted = np.array(arms[k]["pair_ticks"]) >= 0
                arms[k]["rates"][str(hz)] = {
                    "branch_rmse_mm": (1000 * error).tolist(),
                    "invented": int((predicted & ~truth).sum()),
                    "missed": int((truth & ~predicted).sum()),
                }
        return row
    target = cfg["selection"]["target_m"]
    original = release(world, 0.0, 1920)[2]
    row.update(original_travel_m=original, original_failed=original < target)
    cache = {(1920, 0.0): original}
    for arm in arms.values():
        j = None
        if original < target:
            margins = np.array(arm["travel_m"]) - target
            margins[len(grid) // 2] = original - target
            j = select_with_tolerance(
                grid,
                margins,
                margins >= 0,
                np.ones(len(grid), bool),
                cfg["selection"]["epsilon_m"],
                cfg["selection"]["margin_m"],
            )["index"]
        arm["edit_m"] = None if j is None else float(grid[j])
        if j is not None:
            for hz in cfg["references_hz"]:
                values = []
                for off in cfg["selection"]["offsets_m"]:
                    key = (hz, round(float(grid[j] + off), 12))
                    if key not in cache:
                        cache[key] = release(world, key[1], hz)[2]
                    values.append(cache[key])
                arm["rates"][str(hz)] = values
    row["distinct_simulator_calls"] = len(cache)
    return row


def stage_rows(cfg, stage, workers):
    first, n = cfg[stage]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(partial(measure, stage=stage, cfg=cfg), range(first, first + n))
        )


def run(root, workers, protocol=PROTOCOL):
    cfg = load(protocol)
    prefix = "handoff-" if protocol == PHASE_PROTOCOL else ""
    names = cfg["methods"]
    trial = names[1]
    rows = stage_rows(cfg, "confirmation", workers)
    comparisons = {}
    for hz in cfg["references_hz"]:
        err = np.array(
            [
                [
                    np.mean(
                        r["arms"][str(k)]["rates"][str(hz)]["branch_rmse_mm"], axis=0
                    )
                    for k in names
                ]
                for r in rows
            ]
        )
        comparisons[str(hz)] = {
            "mean_striker_receiver_mm": err.mean(0).tolist(),
            "receiver_difference_mm": estimate(err[:, 1, 1] - err[:, 0, 1]),
        }
    passed = all(
        v["receiver_difference_mm"]["ci95"][1] < 0 for v in comparisons.values()
    )
    write(
        root / (prefix + "confirmation.json"),
        {
            "protocol": sha(protocol),
            "methods": names,
            "passed": passed,
            "comparisons": comparisons,
            "rows": rows,
        },
    )
    print("Full-release gate", passed, comparisons, flush=True)
    if not passed:
        return
    rows = stage_rows(cfg, "decisions", workers)
    failing = [r for r in rows if r["original_failed"]]
    counts = {}
    outcomes = {}
    for hz in names:
        a = [r["arms"][str(hz)] for r in failing]
        proposed = [v for v in a if v["edit_m"] is not None]
        ok = np.array(
            [
                v["edit_m"] is not None
                and all(
                    np.min(d) >= cfg["selection"]["target_m"]
                    for d in v["rates"].values()
                )
                for v in a
            ]
        )
        outcomes[hz] = ok.astype(float)
        counts[str(hz)] = {
            "failing": len(a),
            "proposed": len(proposed),
            "centers": sum(
                v["rates"]["1920"][2] >= cfg["selection"]["target_m"] for v in proposed
            ),
            "robust": int(ok.sum()),
            "median_grid_ms": 1000
            * float(np.median([r["arms"][str(hz)]["grid_s"] for r in rows])),
        }
    delta = outcomes[trial] - outcomes["480"]
    write(
        root / (prefix + "decisions.json"),
        {
            "protocol": sha(protocol),
            "comparison": [trial, "480"],
            "confirmation_sha256": sha(root / (prefix + "confirmation.json")),
            "counts": counts,
            "robust_difference": estimate(delta),
            "gained": int((delta > 0).sum()),
            "lost": int((delta < 0).sum()),
            "rows": rows,
        },
    )
    print(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--handoff", action="store_true")
    args = parser.parse_args()
    run(args.output, args.workers, PHASE_PROTOCOL if args.handoff else PROTOCOL)
