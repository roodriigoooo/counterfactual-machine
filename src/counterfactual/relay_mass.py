"""Does the receiver's mass change decisions when observed release geometry is identical?"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pymunk

from .architecture_data import digest
from .continuation_experiment import write
from .relay_world import CONFIG, ROOT, make_relay, release

PROTOCOL = Path("configs/relay-mass-audit.json")


def probe_velocities(receiver, hz, speed=1.0):
    """A separate one-dimensional probe supplies two observed velocity changes."""
    space = pymunk.Space()
    bodies = []
    for x, radius, mass, vx in [
        (0, 0.12, 1.0, speed),
        (0.6, receiver.radius, receiver.mass, 0.0),
    ]:
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
        body.position, body.velocity = (x, 0), (vx, 0)
        shape = pymunk.Circle(body, radius)
        shape.elasticity = receiver.material.restitution
        space.add(body, shape)
        bodies.append(body)
    before = np.array([b.velocity.x for b in bodies])
    for _ in range(hz):
        space.step(1 / hz)
    after = np.array([b.velocity.x for b in bodies])

    return before, after


def measure_mass(receiver, hz):
    before, after = probe_velocities(receiver, hz)

    def estimate(post):
        change = post - before
        if change[1] <= 0:
            raise ValueError("Probe failed to transfer momentum")
        return -float(change[0] / change[1])

    return {
        "before_m_s": before.tolist(),
        "after_m_s": after.tolist(),
        "estimated_kg": estimate(after),
        "quantized_kg": estimate(after.round(2)),
    }


def smallest(indices, grid):
    return (
        None
        if len(indices) == 0
        else float(grid[min(indices, key=lambda j: (abs(j - len(grid) // 2), j))])
    )


def run(output):
    protocol = json.loads(PROTOCOL.read_text())
    cfg = json.loads(CONFIG.read_text())
    rows = []
    for seed in range(cfg["start_parent"], cfg["start_parent"] + cfg["parents"]):
        base = make_relay(seed)
        grid = np.linspace(-base.scene.ball.radius, base.scene.ball.radius, cfg["grid"])
        arms = []
        selected = []
        for mass in protocol["masses_kg"]:
            world = replace(
                base, receiver=base.receiver.model_copy(update={"mass": mass})
            )
            outcomes, probes = {}, {}
            for hz in cfg["rates"]:
                outcomes[str(hz)] = [
                    release(world, float(edit), hz)[2] for edit in grid
                ]
                probes[str(hz)] = measure_mass(world.receiver, hz)
            success = np.array(list(outcomes.values())) >= cfg["target_travel_m"]
            selected.append(success)
            arms.append(
                {
                    "mass_kg": mass,
                    "travel_m": outcomes,
                    "probe": probes,
                    "smallest_two_rate_edit_m": smallest(
                        np.flatnonzero(success.all(0)), grid
                    ),
                }
            )
        a, b = selected
        stable = (a[0] == a[1]) & (b[0] == b[1])
        common = a.all(0) & b.all(0)
        rows.append(
            {
                "parent": seed,
                "edits_m": grid.tolist(),
                "arms": arms,
                "stable_cells": int(stable.sum()),
                "indistinguishable_input_opposite_labels": int(
                    ((a[0] != b[0]) & stable).sum()
                ),
                "common_two_rate_edit_m": smallest(np.flatnonzero(common), grid),
            }
        )
    write(
        output / "mass.json",
        {"protocol": digest(PROTOCOL), "parent_protocol": digest(CONFIG), "rows": rows},
    )
    print(
        "Stable cells",
        sum(r["stable_cells"] for r in rows),
        "opposite labels",
        sum(r["indistinguishable_input_opposite_labels"] for r in rows),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    run(parser.parse_args().output)
