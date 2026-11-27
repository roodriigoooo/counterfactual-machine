"""Signed closest approach of two pass-through disks, with real callbacks as a separate label."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from .architecture_data import digest as sha
from .continuation_experiment import write
from .relay_forecast import forecast, layout
from .relay_world import release

CONFIG = Path("configs/relay-encounter.json")
ROOT = Path("artifacts/relay-encounter")
SURFACES = ("bumper", "floor", "shelf")
FEATURES = (
    "clearance_m",
    "closest_s",
    "relative_speed_m_s",
    "since_impact_s",
    "impact_count",
    "last_shelf",
    "last_floor",
    "shelf_edge_distance_m",
)


def load(path):
    return json.loads(path.read_text())


def gap(xy, radius):
    return np.linalg.norm(xy[..., 0, :] - xy[..., 1, :], axis=-1) - radius


def trace(world, edits, hz, *, enabled=False):
    n = len(edits)
    state = np.empty((2 * hz + 1, n, 2, 5))
    mask = np.zeros((len(state), n), np.uint8)

    def observe(tick, xy, v, w, contacts):
        state[tick, ..., :2] = xy
        state[tick, ..., 2:4] = v
        state[tick, ..., 4] = w
        for a, name, row in contacts:
            if a == 0 and name in SURFACES:
                mask[tick, row] |= 1 << SURFACES.index(name)

    times = forecast(
        world, edits, np.full(n, world.receiver.mass), hz, enabled, observe=observe
    )[1]
    return state, mask, times


def features(world, state, mask, hz):
    clearance = gap(state[..., :2], world.receiver.radius + world.scene.ball.radius)
    tick = int(clearance.argmin())
    surface_begins = mask[1 : tick + 1] & ~mask[:tick]
    beginnings = np.flatnonzero(surface_begins) + 1
    last = int(beginnings[-1]) if len(beginnings) else 0
    last_mask = int(surface_begins[last - 1]) if last else 0
    velocity = state[tick, 0, 2:4] - state[tick, 1, 2:4]
    shelf = next(s for s in world.scene.segments if s.id == "shelf")
    edge_distance = min(
        np.linalg.norm(state[last, 0, :2] - edge) for edge in (shelf.a, shelf.b)
    )
    return [
        float(clearance[tick]),
        tick / hz,
        float(np.linalg.norm(velocity)),
        (tick - last) / hz,
        len(beginnings),
        int(bool(last_mask & 4)),
        int(bool(last_mask & 2)),
        float(edge_distance),
    ]


def measure(item):
    stage, parent, edits, rates, clocks = item
    world = layout(parent, True)
    grid = np.linspace(-world.scene.ball.radius, world.scene.ball.radius, edits)
    radius = world.scene.ball.radius + world.receiver.radius
    rows = [{"edit_m": float(u), "forecast": {}, "reference": {}} for u in grid]
    for hz in clocks:
        state, mask, _ = trace(world, grid, hz)
        for i, row in enumerate(rows):
            row["forecast"][str(hz)] = features(world, state[:, i], mask[:, i], hz)
    for hz in rates:
        for row in rows:
            disabled = release(world, row["edit_m"], hz, pair_enabled=False)[0]
            clearance = gap(disabled, radius)
            tick = int(clearance.argmin())
            events = release(world, row["edit_m"], hz)[1]
            row["reference"][str(hz)] = [
                float(clearance[tick]),
                tick / hz,
                events.get("transfer", -1),
            ]
    return {"stage": stage, "parent": parent, "protocol": sha(CONFIG), "rows": rows}


def save_parent(item, output):
    rec = measure(item)
    write(output / "parents" / f"{rec['parent']}.json", rec)
    return rec["parent"]


def run(output, workers, stages):
    cfg = load(CONFIG)
    rep, diag = cfg["representation"], cfg["diagnostic"]
    tasks = []
    for stage in stages:
        if stage == "diagnosis":
            tasks.extend(
                (stage, row["parent"], len(row["grid_m"]), diag["rates"], [diag["hz"]])
                for row in load(Path(diag["archived"]))["rows"]
            )
        else:
            first, n = rep[stage]
            tasks.extend(
                (stage, p, rep["edits"], rep["references_hz"], rep["forecast_hz"])
                for p in range(first, first + n)
            )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(partial(save_parent, output=output), tasks), 1):
            if n % 10 == 0 or n == len(tasks):
                print(f"Measured {n}/{len(tasks)} parents", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["diagnosis", "train", "calibration", "test"],
        default=["diagnosis", "train", "calibration", "test"],
    )
    args = parser.parse_args()
    run(args.output, args.workers, args.stages)
