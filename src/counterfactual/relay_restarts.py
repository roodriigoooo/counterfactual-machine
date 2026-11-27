"""Localize archived relay errors by substituting the state of one disk before the encounter."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .relay_encounter import CONFIG, ROOT, SURFACES, gap, load, sha, trace, write
from .relay_forecast import forecast, layout
from .relay_world import release


def records(root):
    source = Path(load(CONFIG)["diagnostic"]["archived"])
    population = []
    for archive in load(source)["rows"]:
        parent = archive["parent"]
        frozen = archive["modes"]["known"]
        rows = load(root / "parents" / f"{parent}.json")["rows"]
        for i, r in enumerate(rows):
            t = frozen["first_transfer_tick"][i]
            coarse = r["reference"]["1920"]
            hit, predicted = coarse[2] >= 0, t >= 0
            outcome = (
                ("hit" if predicted else "miss")
                if hit == predicted
                else ("invented" if predicted else "missed")
            )
            times = [v for v in (t / 480, coarse[2] / 1920) if v >= 0]
            population.append(
                dict(
                    parent=parent,
                    index=i,
                    forecast_tick=t,
                    outcome=outcome,
                    encounter_s=min(times, default=coarse[1]),
                    **r,
                )
            )
        for rate in ("1920", "3840"):
            hit = np.array([r["reference"][rate][2] >= 0 for r in rows])
            pred = np.array(frozen["first_transfer_tick"]) >= 0
            report = frozen["rates"][rate]
            assert (sum(hit & ~pred), sum(pred & ~hit)) == (
                report["missed_transfer"],
                report["invented_transfer"],
            )
    return population


def match(case, population, outcome):
    options = [
        r
        for r in population
        if r["outcome"] == outcome and r["parent"] != case["parent"]
    ]

    def key(r):
        dt = abs(r["encounter_s"] - case["encounter_s"])
        if outcome == "miss":
            return (
                max(dt - 0.1, 0),
                abs(r["reference"]["1920"][0]),
                dt,
                r["parent"],
                r["index"],
            )
        return dt, r["parent"], r["index"]

    return min(options, key=key)


def kinematics(state, radius, direction):
    d = state[0, :2] - state[1, :2]
    n = d / max(np.linalg.norm(d), 1e-12)
    v = state[0, 2:4] - state[1, 2:4]
    return {
        "clearance_mm": float(1000 * (np.linalg.norm(d) - radius)),
        "closing_m_s": -float(v @ n),
        "tangential_m_s": float(direction * (n[0] * v[1] - n[1] * v[0])),
    }


def audit(record):
    hz = 480
    world = layout(record["parent"], True)
    edit = record["edit_m"]
    mass = world.receiver.mass
    radius = world.scene.ball.radius + world.receiver.radius
    free, masks, predicted = trace(world, [edit], hz, enabled=True)
    free, masks = free[:, 0], masks[:, 0]
    assert predicted[0] == record["forecast_tick"]
    forecast_begins = [
        (int(t + 1), name)
        for t in np.flatnonzero(masks[1:] & ~masks[:-1])
        for k, name in enumerate(SURFACES)
        if (masks[t + 1] & ~masks[t]) & (1 << k)
    ]
    out = {
        "parent": record["parent"],
        "index": record["index"],
        "edit_m": edit,
        "outcome": record["outcome"],
        "forecast_tick": int(predicted[0]),
        "rates": {},
    }
    for rate in (1920, 3840):
        exact = np.empty_like(free)
        stride = rate // hz
        begins = []

        shelf = next(s for s in world.scene.segments if s.id == "shelf")
        shelf_hits = []
        cursor = 0

        def capture(
            t,
            bodies,
            stride=stride,
            exact=exact,
            begins=begins,
            shelf=shelf,
            shelf_hits=shelf_hits,
        ):
            nonlocal cursor
            for _, a, name in begins[cursor:]:
                if a == 0 and name == "shelf":
                    x = bodies[0].position.x
                    # Positive past the nearer endpoint; negative on the planar span.
                    edge = max(
                        min(shelf.a[0], shelf.b[0]) - x, x - max(shelf.a[0], shelf.b[0])
                    )
                    shelf_hits.append([t, edge])
            cursor = len(begins)
            if t % stride == 0:
                exact[t // stride] = [
                    [*b.position, *b.velocity, b.angular_velocity] for b in bodies
                ]

        _, events, _ = release(world, edit, rate, observe=capture, static_begins=begins)
        actual = events.get("transfer", -1)
        ref = record["reference"][str(rate)]
        assert actual == ref[2]
        clocks = [v for v in (actual / rate, predicted[0] / hz) if v >= 0]
        anchor = min(clocks, default=ref[1])
        base = min(int(np.floor(anchor * hz + 1e-9)), len(free) - 1)
        # The last eligible snapshot always precedes the first hit of either path.
        last = max(0, int(np.ceil(anchor * hz - 1e-9)) - 1)
        upstream = [(t, name) for t, a, name in begins if a == 0 and t / rate < anchor]
        jobs = {
            f"lead-{lead}": min(last, max(0, base - lead)) for lead in (1, 4, 16, 64)
        }
        if record["outcome"] in ("invented", "missed"):
            for i, (t, _name) in enumerate(upstream):
                # Bracket the two clocks' static detection, before either / after both
                # when a corresponding forecast onset occurs within a 12.5 ms window.
                ft = [
                    k
                    for k, name in forecast_begins
                    if name == _name and abs(k / hz - t / rate) <= 0.0125
                ]
                near = (
                    min(ft, key=lambda k: abs(k / hz - t / rate)) if ft else t / stride
                )
                jobs[f"surface-{i}-before"] = max(
                    0, int(np.floor(min(t / stride, near))) - 1
                )
                jobs[f"surface-{i}-after"] = min(
                    last, int(np.ceil(max(t / stride, near))) + 1
                )
        restart = {}
        if record["outcome"] in ("invented", "missed"):
            for tick in sorted(set(jobs.values())):
                states = np.repeat(free[tick][None], 6, axis=0)
                states[1, 0] = exact[tick, 0]
                states[2, 1] = exact[tick, 1]
                states[3] = exact[tick]
                states[4, 0, :2] = exact[tick, 0, :2]
                states[5, 0, 2:] = exact[tick, 0, 2:]
                ts = forecast(
                    world,
                    np.full(len(states), edit),
                    np.full(len(states), mass),
                    hz,
                    start_tick=tick,
                    initial_state=states,
                )[1]
                assert ts[0] == predicted[0]
                restart[tick] = dict(
                    zip(
                        (
                            "free",
                            "striker",
                            "receiver",
                            "both",
                            "striker_position",
                            "striker_motion",
                        ),
                        map(int, ts),
                    )
                )
        tick = jobs["lead-4"]
        result = {
            "actual_tick": actual,
            "anchor_s": float(anchor),
            "static_begins": [[t / rate, name] for t, name in upstream],
            "shelf_begins_mm": [
                [t / rate, 1000 * edge] for t, edge in shelf_hits if t / rate < anchor
            ],
            "predicted_begins": [
                [t / hz, name] for t, name in forecast_begins if t / hz < anchor
            ],
            "incoming_s": tick / hz,
            "forecast_incoming": kinematics(free[tick], radius, world.direction),
            "actual_incoming": kinematics(exact[tick], radius, world.direction),
            "disk_position_errors_mm": (
                1000 * np.linalg.norm(free[tick, :, :2] - exact[tick, :, :2], axis=1)
            ).tolist(),
            "forecast_min_mm": 1000 * record["forecast"]["480"][0],
            "actual_min_mm": 1000 * ref[0],
            "restart": {
                k: dict(tick=v, **restart[v]) for k, v in jobs.items() if v in restart
            },
        }
        if record["parent"] == 123002 and record["index"] == 21:
            ghost, _, _ = trace(world, [edit], hz)
            xy = release(world, edit, rate, pair_enabled=False)[0]
            result["example"] = {
                "time_s": (np.arange(len(free)) / hz).tolist(),
                "forecast_gap_mm": (1000 * gap(ghost[:, 0, :, :2], radius)).tolist(),
                "actual_gap_mm": (1000 * gap(xy[::stride], radius)).tolist(),
            }
        out["rates"][str(rate)] = result
    return out


def run(root, workers):
    population = records(root)
    cases = [r for r in population if r["outcome"] in ("missed", "invented")]
    controls, pairs = {}, []
    for row in cases:
        pair = {"case": [row["parent"], row["index"]]}
        for name in ("hit", "miss"):
            c = match(row, population, name)
            controls[c["parent"], c["index"]] = c
            pair[name] = [c["parent"], c["index"]]
        pairs.append(pair)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        audited = list(pool.map(audit, [*cases, *controls.values()]))
    counts = dict(Counter(r["outcome"] for r in population))
    write(
        root / "diagnosis.json",
        {
            "protocol": sha(CONFIG),
            "counts": counts,
            "archived_sha256": sha(Path(load(CONFIG)["diagnostic"]["archived"])),
            "pairs": pairs,
            "audited": audited,
        },
    )
    print(counts, len(controls), "distinct controls", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.workers)
