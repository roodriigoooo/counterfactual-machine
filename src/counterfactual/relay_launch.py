"""A release-to-impact map for an isolated vertical drop onto a circular bumper."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BumperImpact:
    row: int
    detection_tick: int
    normal: tuple[float, float]
    outgoing_velocity: tuple[float, float]
    outgoing_spin: float
    state: np.ndarray


def handoff(world, edits, coarse, fine):
    """Return sparse coarse ticks and striker states for geometrically isolated impacts.

    Before the first interaction p[k] = p[0] + g k(k-1)/(2 h²) for a resting disk.
    After detection we advance at most fine/coarse local ticks, then hand back both
    translational coordinates, velocity, and spin. Other objects keep their own clocks.
    """
    if fine < coarse or fine % coarse:
        raise ValueError("Handoff rate must be an integer multiple of the release rate")
    result = {}
    scene = world.scene
    ball, bumper, receiver = scene.ball, scene.bumper, world.receiver
    if (
        ball.velocity != (0.0, 0.0)
        or receiver.velocity != (0.0, 0.0)
        or receiver.angular_velocity
    ):
        return result
    # This fast path assumes the untouched receiver rests on a horizontal support.
    if not any(
        s.a[1] == s.b[1]
        and min(s.a[0], s.b[0]) <= receiver.position[0] <= max(s.a[0], s.b[0])
        and abs(s.a[1] + s.radius + receiver.radius - receiver.position[1]) < 1e-9
        for s in scene.segments
    ):
        return result
    x0, y0 = ball.position
    gravity = -scene.gravity[1]
    combined = ball.radius + bumper.radius
    obstacle_top = (
        max(
            [receiver.position[1] + receiver.radius]
            + [max(s.a[1], s.b[1]) + s.radius for s in scene.segments]
        )
        + ball.radius
    )
    for i, edit in enumerate(edits):
        center = np.array([bumper.position[0] + edit, bumper.position[1]])
        dx = x0 - center[0]
        if abs(dx) >= combined:
            continue
        height = center[1] + math.sqrt(combined**2 - dx**2)
        if not obstacle_top < height < y0:
            continue
        first = math.ceil(0.5 + math.sqrt(0.25 + 2 * fine**2 * (y0 - height) / gravity))
        dest = (first + fine // coarse - 1) // (fine // coarse)
        if dest > scene.horizon * coarse:
            continue
        k = first - 1
        xy = np.array([x0, y0 - gravity * k * (k - 1) / (2 * fine * fine)])
        v = np.array([0.0, -gravity * k / fine])
        spin = ball.angular_velocity
        # Guard the entire coarse splice, including the unused coarse path.
        # Friction and restitution are at most one; a conservative speed bound
        # covers one coarse step past the lower of the two detection heights.
        if (
            height - obstacle_top
            <= 3
            * (np.linalg.norm(v) + abs(spin) * ball.radius + gravity / coarse)
            / coarse
        ):
            continue
        end = dest * (fine // coarse)
        impact = None
        for tick in range(first, end + 1):
            xy += v / fine
            d = xy - center
            dist = np.linalg.norm(d)
            incoming = v.copy()
            v[1] -= gravity / fine
            if dist > combined or dist < 1e-12:
                if tick == first:
                    break
                continue
            n = d / dist
            tangent = np.array([-n[1], n[0]])
            rebound = (
                max(0.0, -incoming @ n)
                * ball.material.restitution
                * bumper.material.restitution
            )
            dvn = max(0.0, rebound - v @ n)
            bound = ball.material.friction * bumper.material.friction * dvn
            dvt = np.clip((spin * ball.radius - v @ tangent) / 3, -bound, bound)
            v += dvn * n + dvt * tangent
            spin -= 2 * dvt / ball.radius
            xy += max(combined - dist - 0.001, 0) * (1 - 0.9 ** (60 / fine)) * n
            if tick == first:
                impact = (tuple(n), tuple(v), float(spin))
        else:
            result.setdefault(dest, []).append(
                BumperImpact(i, first, *impact, np.r_[xy, v, spin])
            )
    return result
