"""Seeded parents; goals are sampled independently of outcomes."""
import numpy as np
from .schemas import Ball, Bumper, SceneSpec, Segment, GoalSpec, Region
from .geometry import validate_scene


def generate_scene(seed: int) -> SceneSpec:
    rng = np.random.default_rng(seed)
    family = "approach" if seed % 2 == 0 else "ramp"
    for _ in range(100):
        x = rng.uniform(1.1, 1.9)
        y = rng.uniform(3.1, 3.6)
        vx = rng.uniform(1.5, 3.0)
        by = rng.uniform(1.7, 2.1)
        flight = np.sqrt(2*(y-by-0.35)/9.81)
        bx = x+vx*flight+rng.uniform(-0.3, 0.3)
        segments = [Segment(id="floor", a=(0.1, 0.18), b=(5.9, 0.18))]
        if family == "ramp":
            segments.append(Segment(id="ramp", a=(x-0.45, y-0.24), b=(x+0.55, y-0.64)))
            bx += 0.3
        scene = SceneSpec(id=f"scene-{seed:05d}", parent_id=f"parent-{seed:05d}", seed=seed,
                          family=family, ball=Ball(position=(x, y), velocity=(vx, 0)),
                          bumper=Bumper(position=(bx, by)), segments=tuple(segments))
        try:
            return validate_scene(scene)
        except ValueError:
            pass
    raise RuntimeError("generator could not produce valid scene")


def generate_goal(seed: int, ball_id: str = "ball"):
    rng = np.random.default_rng(10_000_000+seed)
    return GoalSpec(ball_id=ball_id, kind="reach" if seed % 2 == 0 else "avoid",
                    region=Region(center=(float(rng.uniform(0.8, 5.2)), float(rng.uniform(0.4, 1.6))),
                                  radius=float(rng.uniform(0.25, 0.6))))
