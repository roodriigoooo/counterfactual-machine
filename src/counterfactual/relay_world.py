"""Research-only two-disk relay, with a single editable bumper and an untouched receiver."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymunk

from .architecture_data import digest
from .continuation_experiment import write
from .geometry import apply_edit, segment_distance
from .schemas import Ball, Bumper, EditSpec, SceneSpec, Segment

CONFIG = Path("configs/relay-exposure.json")
ROOT = Path("artifacts/relay-exposure")


@dataclass(frozen=True)
class Relay:
    scene: SceneSpec
    receiver: Ball
    direction: int


def make_relay(seed):
    rng = np.random.default_rng(seed)
    side = int(rng.choice([-1, 1]))
    anchor = 1.8 if side == 1 else 4.2
    floor = Segment(id="floor", a=(0.1, 0.18), b=(5.9, 0.18))
    radius = rng.uniform(0.10, 0.16)
    recv = Ball(
        id="receiver",
        position=(anchor + side * rng.uniform(1.25, 2.45), 0.22 + radius),
        radius=radius,
        mass=rng.uniform(0.55, 2.2),
        velocity=(0, 0),
    )
    scene = SceneSpec(
        id=f"relay-{seed}",
        parent_id=f"relay-parent-{seed}",
        seed=seed,
        family="authored",
        ball=Ball(
            id="striker", position=(anchor, rng.uniform(2.8, 3.5)), velocity=(0, 0)
        ),
        bumper=Bumper(
            position=(anchor - side * rng.uniform(0.09, 0.27), rng.uniform(1.5, 1.9)),
            radius=rng.uniform(0.24, 0.34),
        ),
        segments=(floor,),
    )
    return Relay(scene, recv, side)


def edited_scene(world, edit):
    scene = apply_edit(world.scene, EditSpec(displacement=edit))
    recv = world.receiver
    if any(
        p - recv.radius < 0 or p + recv.radius > lim
        for p, lim in zip(recv.position, scene.world)
    ):
        raise ValueError("Receiver must fit inside the world at release")
    for obj in (scene.ball, scene.bumper):
        if (
            np.linalg.norm(np.subtract(obj.position, recv.position)) + 1e-9
            < obj.radius + recv.radius
        ):
            raise ValueError("Receiver overlaps another circle")
    if any(
        segment_distance(recv.position, seg.a, seg.b) + 1e-9 < seg.radius + recv.radius
        for seg in scene.segments
    ):
        raise ValueError("Receiver overlaps a segment")
    return scene


def release(world, edit, hz, *, pair_enabled=True, observe=None, static_begins=None):
    if hz < 60 or hz % 60:
        raise ValueError(
            "The reference rate must be a positive multiple of the observation rate"
        )
    scene = edited_scene(world, edit)
    recv = world.receiver
    space = pymunk.Space()
    space.gravity, space.iterations, space.collision_slop = scene.gravity, 20, 0.001
    bodies = []
    for collision_type, spec in enumerate((scene.ball, recv), 1):
        body = pymunk.Body(
            spec.mass, pymunk.moment_for_circle(spec.mass, 0, spec.radius)
        )
        body.position, body.velocity = spec.position, spec.velocity
        body.angle, body.angular_velocity = spec.angle, spec.angular_velocity
        shape = pymunk.Circle(body, spec.radius)
        shape.friction, shape.elasticity, shape.collision_type = (
            spec.material.friction,
            spec.material.restitution,
            collision_type,
        )
        if not pair_enabled:
            shape.filter = pymunk.ShapeFilter(group=1)
        space.add(body, shape)
        bodies.append(body)
    bumper = scene.bumper
    shape = pymunk.Circle(space.static_body, bumper.radius, bumper.position)
    shape.friction, shape.elasticity, shape.collision_type = (
        bumper.material.friction,
        bumper.material.restitution,
        3,
    )
    space.add(shape)
    surfaces = {shape: bumper.id}
    for seg in scene.segments:
        shape = pymunk.Segment(space.static_body, seg.a, seg.b, seg.radius)
        shape.friction, shape.elasticity, shape.collision_type = (
            seg.material.friction,
            seg.material.restitution,
            4,
        )
        space.add(shape)

        surfaces[shape] = seg.id

    if static_begins is not None:

        def surface_begin(arb, _space, _data):
            target = next(s for s in arb.shapes if s in surfaces)
            body = next(s for s in arb.shapes if s != target)
            static_begins.append((tick, body.collision_type - 1, surfaces[target]))

        for a in (1, 2):
            space.on_collision(a, 4, begin=surface_begin)
    first = {}

    def callback(name):
        def mark(_arb, _space, _data):
            if static_begins is not None and name != "transfer":
                surface_begin(_arb, _space, _data)
            first.setdefault(name, tick)

        return mark

    for a, b, name in [
        (1, 2, "transfer"),
        (1, 3, "deflection"),
        (2, 3, "receiver_bumper"),
    ]:
        space.on_collision(a, b, begin=callback(name))
    ticks = int(scene.horizon * hz)
    xy = np.empty((ticks + 1, 2, 2))
    xy[0] = [body.position for body in bodies]
    if observe is not None:
        observe(0, bodies)
    for tick in range(1, ticks + 1):
        space.step(1 / hz)
        xy[tick] = [body.position for body in bodies]
        if observe is not None:
            observe(tick, bodies)
    travel = float(np.max(world.direction * (xy[:, 1, 0] - recv.position[0])))
    return xy, first, travel


def parent_record(seed, config, capture):
    world = make_relay(seed)
    grid = np.linspace(
        -world.scene.ball.radius, world.scene.ball.radius, config["grid"]
    )
    target = config["target_travel_m"]
    rates = config["rates"]
    base = len(grid) // 2
    by_rate = {}
    originals = {}
    for hz in rates:
        outcomes = []
        for edit in grid:
            xy, begins, travel = release(world, float(edit), hz)
            if edit == 0:
                originals[hz] = xy
            outcomes.append(
                {"first_tick": begins, "travel_m": travel, "success": travel >= target}
            )
        by_rate[str(hz)] = outcomes
    records = list(by_rate.values())
    witness = None
    # Stable repair and ordering are cheap filters. The disabled-pair control is
    # then an independent physical intervention on the unchanged selected world.
    if all(not v[base]["success"] for v in records):
        for idx in sorted(range(len(grid)), key=lambda j: (abs(j - base), j)):
            if not all(
                v[idx]["success"]
                and 0
                < v[idx]["first_tick"].get("deflection", 2 * hz + 1)
                < v[idx]["first_tick"].get("transfer", 0)
                for hz, v in zip(rates, records, strict=True)
            ):
                continue
            checks = {}
            for hz in rates:
                xy, events, _ = release(world, grid[idx], hz)
                ctrl, disabled_events, travel = release(
                    world, grid[idx], hz, pair_enabled=False
                )
                if "transfer" in disabled_events:
                    raise ValueError("Disabled pair received a collision")
                tick = events["transfer"]
                delta = np.linalg.norm(xy[:tick, 1] - ctrl[:tick, 1], axis=-1)
                prefix = min(
                    tick, by_rate[str(hz)][base]["first_tick"].get("transfer", len(xy))
                )
                paired = np.linalg.norm(
                    xy[:prefix, 1] - originals[hz][:prefix, 1], axis=-1
                )
                checks[str(hz)] = {
                    "prefix_max_m": float(delta.max()),
                    "paired_prefix_max_m": float(paired.max()),
                    "paired_prefix_ticks": prefix,
                    "disabled_travel_m": travel,
                    "disabled_success": travel >= target,
                    "transfer_tick": tick,
                    "deflection_tick": events["deflection"],
                }
                if capture:
                    checks[str(hz)].update(
                        positions=xy[:: hz // 60].tolist(),
                        original_positions=originals[hz][:: hz // 60].tolist(),
                        disabled_positions=ctrl[:: hz // 60].tolist(),
                    )
            if all(
                c["prefix_max_m"] == 0
                and c["paired_prefix_max_m"] == 0
                and not c["disabled_success"]
                for c in checks.values()
            ):
                witness = {"index": idx, "edit_m": float(grid[idx]), "checks": checks}
                break
    return {
        "parent": seed,
        "scene": world.scene.model_dump(mode="json"),
        "receiver": world.receiver.model_dump(mode="json"),
        "direction": world.direction,
        "target_travel_m": target,
        "edits_m": grid.tolist(),
        "rates": by_rate,
        "witness": witness,
    }


def run(output):
    cfg = json.loads(CONFIG.read_text())
    rows = []
    for seed in range(cfg["start_parent"], cfg["start_parent"] + cfg["parents"]):
        rows.append(
            parent_record(seed, cfg, capture=not any(r["witness"] for r in rows))
        )
    write(
        output / "relay.json",
        {"protocol": digest(CONFIG), "pymunk": pymunk.version, "rows": rows},
    )
    write(output / "report-kind.json", {"kind": "relay-exposure-v1"})
    print(
        f"{sum(r['witness'] is not None for r in rows)}/{len(rows)} two-rate, pair-disabled witnesses"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    run(parser.parse_args().output)
