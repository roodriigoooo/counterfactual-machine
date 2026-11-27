"""Bounded impact resolution with a conservative free-flight clearance test."""

import torch

from .contact_mechanics import nearest_normal
from .recurrent import DT
from .subinterval import admission, contact_response, step

PHYSICS_DT = 1 / 1920


def fly(s, h, physics_dt=PHYSICS_DT):
    g = s.new_tensor([0.0, -9.81])
    h = h[:, None]
    return torch.cat(
        [
            s[:, :2] + s[:, 2:4] * h + g * (h * (h - physics_dt) / 2),
            s[:, 2:4] + g * h,
            s[:, 4:5] + s[:, 5:6] * h,
            s[:, 5:6],
        ],
        -1,
    )


def sweep(s, c, h, gated, physics_dt=PHYSICS_DT):
    if not gated:
        return admission(s, c, h, physics_dt)
    _, gap = nearest_normal(s[:, None, :2], c)
    travel = h * (s[:, 2:4].norm(dim=-1) + 4.905 * (h + physics_dt))
    row, edge = torch.where((c[..., 9] > 0) & (gap <= travel[:, None] + 2e-6))
    eligible = torch.zeros_like(gap, dtype=torch.bool)
    time = h[:, None].expand_as(gap).clone()
    n = torch.zeros_like(c[..., :2])
    if len(row):
        hit, t, normal, _ = admission(s[row], c[row, edge, None], h[row], physics_dt)
        eligible[row, edge], time[row, edge], n[row, edge] = hit[:, 0], t[:, 0], normal[:, 0]
    return eligible, time, n, gap


def bounded(s, c, gated=True, audit_overflow=False, *, physics_dt=PHYSICS_DT, transition=None):
    """Return per-slot impact clocks (seconds, inf for absent) and post-impact states."""
    h = torch.full_like(s[:, 0], DT)
    result = fly(s, h, physics_dt)
    clocks = s.new_full((len(s), 2, c.shape[1]), torch.inf)
    posts = s.new_zeros((len(s), 2, 6))
    overflow = torch.zeros_like(h, dtype=torch.bool)
    rows = torch.arange(len(s), device=s.device)
    for slot in range(2):
        eligible, times, normals, gaps = sweep(s, c, h, gated, physics_dt)
        hit = eligible.any(-1)
        if not hit.any():
            break
        rows, s, c, h = rows[hit], s[hit], c[hit], h[hit]
        eligible, times, normals, gaps = [v[hit] for v in (eligible, times, normals, gaps)]
        t = times.masked_fill(~eligible, torch.inf).amin(-1)
        marked = eligible & ((times - t[:, None]).abs() < 1e-7)
        if transition is None:
            impulse, correction, spin = contact_response(s, c, times, normals, gaps, marked, h[:, None])
            post = fly(s, t, physics_dt)
            post[:, :2] += (correction[..., None] * normals).sum(1)
            post[:, 2:4] += impulse.sum(1)
            post[:, 5] += spin.sum(1)
        else:
            t, post = transition(s, c, h, t, marked, gaps, physics_dt)
        clocks[rows, slot] = (DT - h + t)[:, None].expand_as(times).masked_fill(~marked, torch.inf)
        posts[rows, slot] = post
        h = (h - t).clamp_min(0)
        result[rows] = fly(post, h, physics_dt)
        # A zero-time resting constraint gets a support impulse for the remainder.
        # Finish this interval to avoid repeatedly resolving it at unchanged time.
        normal_speed = (s[:, None, 2:4] * normals).sum(-1)
        support = ((times < 1e-7) & (normal_speed.abs() < 1e-5) & marked).any(-1)
        more = (h > 1e-8) & ~support
        rows, s, c, h = rows[more], post[more], c[more], h[more]
        if not len(rows):
            break
        if slot == 1 and audit_overflow:
            overflow[rows] = sweep(s, c, h, gated, physics_dt)[0].any(-1)
    return result, clocks, posts, overflow


def four(s, c):
    clocks = []
    for k in range(4):
        s, hit, times, _ = step(s, c, DT / 4)
        clocks.append((times + k * DT / 4).masked_fill(~hit, torch.inf))
    return s, torch.stack(clocks, 1)


@torch.inference_mode()
def rollout(s, c, mode="bounded", gated=True, audit=False):
    states, events, overflow = [s], [], []
    for _ in range(120):
        if mode == "four":
            s, clocks = four(s, c)
        else:
            s, clocks, _, excess = bounded(s, c, gated, audit)
            overflow.append(excess)
        states.append(s)
        events.append(clocks)
    return torch.stack(states, 1), torch.stack(events, 1), torch.stack(overflow, 1) if overflow else None


def features(post, c, remaining, first):
    ball = c[..., 6].clamp_min(0.08)
    normal, gap = nearest_normal(post[:, None, :2], c)
    u = torch.linspace(0, 1, 17, device=c.device)
    t = remaining[:, None] * u
    path = (
        post[:, None, :2]
        + post[:, None, 2:4] * t[..., None]
        + c.new_tensor([0.0, -9.81]) * (t * (t - PHYSICS_DT) / 2)[..., None]
    )
    _, gaps = nearest_normal(path[:, :, None, :], c[:, None])
    minimum, closest = gaps.min(1)
    vn = (post[:, None, 2:4] * normal).sum(-1)
    tangent = torch.stack([-normal[..., 1], normal[..., 0]], -1)
    vt = (post[:, None, 2:4] * tangent).sum(-1) - post[:, None, 5] * ball
    expand = lambda v: v[:, None].expand_as(gap)
    x = torch.stack(
        [
            gap / ball,
            minimum / ball,
            vn / 4,
            vt / 4,
            closest / 16,
            expand(remaining) / (2 * DT),
            expand(post[:, 5]) / 20,
            normal[..., 0],
            normal[..., 1],
            c[..., 0],
            c[..., 7],
            c[..., 8],
        ],
        -1,
    )
    mask = (c[..., 9] > 0) & ~first & first.any(-1, keepdim=True)
    return x.clamp(-15, 15), mask


@torch.inference_mode()
def query(s, c, gated=True, *, incoming_only=False):
    """Two 60 Hz frames, conditioned solely on the supplied start state and geometry."""
    initial = s
    clocks, posts = [], []
    for k in range(2):
        s, times, post, _ = bounded(s, c, gated)
        clocks.append(times + k * DT)
        posts.append(post)
    times = torch.cat(clocks, 1)
    post = torch.cat(posts, 1)
    earliest, slot = times.amin(-1).min(-1)
    row = torch.arange(len(s))
    first = times[row, slot].isfinite()
    conditional_state = fly(initial, earliest.clamp_max(2 * DT)) if incoming_only else post[row, slot]
    x, mask = features(conditional_state, c, (2 * DT - earliest).clamp_min(0), first)
    return times.amin(1), x, mask
