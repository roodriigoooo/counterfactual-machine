"""Compare first-impact refinement and whole-release refinement on every archived edit."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from .relay_encounter import CONFIG, ROOT, load, write
from .relay_forecast import forecast, layout
from .relay_launch import handoff
from .relay_refinement import PHASE_PROTOCOL

ARMS = {"480": (480, None), "handoff": (480, 1920), "1920": (1920, None)}


def audit(row, root):
    world = layout(row["parent"], True)
    grid = row["grid_m"]
    ref = load(root / "parents" / f"{row['parent']}.json")["rows"]
    outcomes = {}
    for name, (hz, clock) in ARMS.items():
        first = forecast(
            world,
            grid,
            np.full(len(grid), world.receiver.mass),
            hz,
            first_bumper_hz=clock,
        )[1]
        outcomes[name] = first.tolist()
    assert outcomes["480"] == row["modes"]["known"]["first_transfer_tick"]
    return {
        "parent": row["parent"],
        "predicted_ticks": outcomes,
        "actual_ticks": {
            h: [r["reference"][h][2] for r in ref] for h in ("1920", "3840")
        },
    }


def run(root, workers):
    source = Path(load(CONFIG)["diagnostic"]["archived"])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(audit, root=root), load(source)["rows"]))
    timing = []
    cfg = load(PHASE_PROTOCOL)
    start = cfg["confirmation"][0]
    for j in range(8):
        world = layout(start + j, True)
        grid = np.linspace(
            -world.scene.ball.radius, world.scene.ball.radius, cfg["grid"]
        )
        mass = np.full(len(grid), world.receiver.mass)
        times = {}
        # Rotate order across parents. One unrecorded warm grid per method.
        names = list(ARMS)
        for name in names[j % 3 :] + names[: j % 3]:
            hz, clock = ARMS[name]
            if j == 0:
                forecast(world, grid, mass, hz, first_bumper_hz=clock)
            tic = perf_counter()
            forecast(world, grid, mass, hz, first_bumper_hz=clock)
            times[name] = 1000 * (perf_counter() - tic)
        events = handoff(world, grid, 480, 1920)
        ticks = [
            4 * t - e.detection_tick + 1
            for t, entries in events.items()
            for e in entries
        ]
        timing.append({"parent": start + j, "ms": times, "local_fine_ticks": ticks})
    write(root / "clocks.json", {"archived": rows, "sequential_profile": timing})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.workers)
