"""A discrete impulse forecast for two dynamic disks, finite capsules and a fixed bumper."""

from dataclasses import replace

import numpy as np

from .relay_world import edited_scene, make_relay
from .schemas import Segment


def layout(seed, elevated=False):
    world = make_relay(seed)
    if not elevated:
        return world
    # An elevated finite shelf introduces a second support edge and a drop to the floor.
    x = world.receiver.position[0]
    height = np.random.default_rng(seed + 500000).uniform(0.58, 0.86)
    shelf = Segment(id="shelf", a=(x - 0.52, height), b=(x + 0.52, height))
    receiver = world.receiver.model_copy(
        update={"position": (x, height + shelf.radius + world.receiver.radius)}
    )
    return replace(
        world,
        receiver=receiver,
        scene=world.scene.model_copy(
            update={"segments": (*world.scene.segments, shelf)}
        ),
    )


def contact_velocity(v, w, a, b, row, n, t, radii):
    rel = v[row, a].copy()
    spin = w[row, a] * radii[a]
    if b is not None:
        rel -= v[row, b]
        spin = spin + w[row, b] * radii[b]
    return (rel * n).sum(-1), (rel * t).sum(-1) - spin


def forecast(
    world,
    edits,
    masses,
    hz=480,
    pair_enabled=True,
    *,
    start_tick=0,
    initial_state=None,
    observe=None,
    first_bumper_hz=None,
):
    """Only supplied masses enter dynamics. Each row owns its own bodies and constraints."""
    if hz < 60 or hz % 60:
        raise ValueError("Forecast rate must be a multiple of 60")
    edits, masses = np.broadcast_arrays(
        np.asarray(edits, float), np.asarray(masses, float)
    )
    if (
        edits.ndim != 1
        or not len(edits)
        or not np.isfinite(edits).all()
        or not np.isfinite(masses).all()
        or np.any(masses <= 0)
    ):
        raise ValueError("Expected one positive receiver mass per row")
    for edit in np.unique(edits):
        edited_scene(world, float(edit))
    scene = world.scene
    balls = (scene.ball, world.receiver)
    radii = np.array([b.radius for b in balls])
    inv = np.column_stack((np.full(len(edits), 1 / scene.ball.mass), 1 / masses))
    xy = np.broadcast_to([b.position for b in balls], (len(edits), 2, 2)).copy()
    v = np.broadcast_to([b.velocity for b in balls], xy.shape).copy()
    w = np.broadcast_to([b.angular_velocity for b in balls], inv.shape).copy()
    ticks = round(scene.horizon * hz)
    if not 0 <= start_tick < ticks or (start_tick and initial_state is None):
        raise ValueError("A restart needs a state and a tick within the release")
    handoffs = {}
    if first_bumper_hz is not None:
        if initial_state is not None:
            raise ValueError("A release handoff requires the original release state")
        from .relay_launch import handoff

        handoffs = handoff(world, edits, hz, first_bumper_hz)
    if initial_state is not None:
        state = np.asarray(initial_state, dtype=float)
        if state.shape != (*inv.shape, 5) or not np.isfinite(state).all():
            raise ValueError(
                "Restart state must contain x, y, vx, vy and spin per body"
            )
        xy, v, w = state[..., :2].copy(), state[..., 2:4].copy(), state[..., 4].copy()
    center = np.broadcast_to(scene.bumper.position, (len(edits), 2)).copy()
    center[:, 0] += edits
    static = [(center, None, scene.bumper)] + [
        (np.asarray(s.a), np.subtract(s.b, s.a), s) for s in scene.segments
    ]
    path = [xy.copy()]
    first = np.full(len(edits), -1, int)
    maximum = np.zeros(len(edits))
    residual_momentum = 0.0
    if observe is not None:
        observe(start_tick, xy, v, w, [])
    for tick in range(start_tick + 1, ticks + 1):
        xy += v / hz
        constraints = []
        contacts = []

        def add(
            a,
            b,
            delta,
            dist_min,
            material,
            surface,
            *,
            constraints=constraints,
            contacts=contacts,
            tick=tick,
        ):
            norm2 = np.einsum("ij,ij->i", delta, delta)
            row = np.flatnonzero(norm2 <= dist_min**2 + 1e-13)
            if not len(row):
                return
            dist = np.sqrt(norm2[row])
            n = delta[row] / np.maximum(dist[:, None], 1e-12)
            t = np.column_stack((-n[:, 1], n[:, 0]))
            rev, _ = contact_velocity(v, w, a, b, row, n, t, radii)
            denom = inv[row, a] + (inv[row, b] if b is not None else 0.0)
            constraints.append(
                (
                    a,
                    b,
                    row,
                    n,
                    t,
                    denom,
                    (-rev).clip(0)
                    * balls[a].material.restitution
                    * material.restitution,
                    balls[a].material.friction * material.friction,
                    np.zeros(len(row)),
                    np.zeros(len(row)),
                    dist_min - dist,
                )
            )
            if observe is not None:
                contacts.append((a, surface, row))
            if b is not None:
                first[row[first[row] < 0]] = tick

        for a in (0, 1):
            for start, ab, obj in static:
                d = xy[:, a] - start
                if ab is not None:
                    d = d - np.clip((d @ ab) / (ab @ ab), 0, 1)[:, None] * ab
                add(a, None, d, obj.radius + radii[a], obj.material, obj.id)
        if pair_enabled:
            add(0, 1, xy[:, 0] - xy[:, 1], radii.sum(), world.receiver.material, "pair")
        v[:, :, 1] -= 9.81 / hz
        for _ in range(3):
            for a, b, row, n, t, denom, rebound, mu, jn, jt, _depth in constraints:
                vn, vt = contact_velocity(v, w, a, b, row, n, t, radii)
                normal = np.maximum(0, jn + (rebound - vn) / denom)
                tangent = np.clip(jt - vt / (3 * denom), -mu * normal, mu * normal)
                delta_t = tangent - jt
                impulse = (normal - jn)[:, None] * n + delta_t[:, None] * t
                dv = impulse * inv[row, a, None]
                v[row, a] += dv
                w[row, a] -= 2 * delta_t * inv[row, a] / radii[a]
                if b is not None:
                    other = -impulse * inv[row, b, None]
                    v[row, b] += other
                    w[row, b] -= 2 * delta_t * inv[row, b] / radii[b]
                    imbalance = dv / inv[row, a, None] + other / inv[row, b, None]
                    residual_momentum = max(
                        residual_momentum, float(np.abs(imbalance).max())
                    )
                jn[:], jt[:] = normal, tangent
        for a, b, row, n, _t, denom, _rebound, _mu, _jn, _jt, depth in constraints:
            shift = (
                np.maximum(depth - 0.001, 0)[:, None]
                * n
                * (1 - 0.9 ** (60 / hz))
                / denom[:, None]
            )
            xy[row, a] += inv[row, a, None] * shift
            if b is not None:
                xy[row, b] -= inv[row, b, None] * shift
        if tick in handoffs:
            for impact in handoffs[tick]:
                i, state = impact.row, impact.state
                xy[i, 0], v[i, 0], w[i, 0] = state[:2], state[2:4], state[4]
        if observe is not None:
            observe(tick, xy, v, w, contacts)
        maximum = np.maximum(
            maximum, world.direction * (xy[:, 1, 0] - world.receiver.position[0])
        )
        if tick % (hz // 60) == 0:
            path.append(xy.copy())
    return np.stack(path, 1), first, maximum, residual_momentum
