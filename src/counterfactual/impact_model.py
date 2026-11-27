"""Decode a rounded impact as a clock and a bounded impulse, or direct state residuals."""

import torch
from torch import nn

from .contact_mechanics import nearest_normal
from .continuation_solver import bounded, fly
from .recurrent import DT
from .subinterval import contact_response


def tangent(normal):
    return torch.stack((-normal[..., 1], normal[..., 0]), -1)


def clock_at_tick(time, remaining, quantum):
    # Both slots use the global grid within the observation. Touching at slot start
    # is an existing constraint and consumes no additional flight time.
    elapsed = DT - remaining
    rounded = torch.ceil((elapsed + time) / quantum - 1e-5) * quantum - elapsed
    return torch.where(time > 1e-7, rounded.clamp_min(0).minimum(remaining), time)


class Impact(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("tick", "ordered", "joint", "direct"):
            raise ValueError(mode)
        self.mode = mode
        if mode in ("joint", "direct"):
            # Widths give the two heads essentially the same trainable budget.
            mid, out = (33, 3) if mode == "joint" else (31, 6)
            self.net = nn.Sequential(
                nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, mid), nn.SiLU(), nn.Linear(mid, out)
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, c, h, t, marked, gaps, quantum):
        row = torch.arange(len(s), device=s.device)
        idx = marked.to(torch.int64).argmax(-1)
        edge = c[row, idx]
        raw_time = t
        t = clock_at_tick(t, h, quantum)
        post = fly(s, t, quantum)
        normals, _ = nearest_normal(post[:, None, :2], c)
        unique = (marked.sum(-1) == 1) & (raw_time > 1e-7)
        raw = None
        if self.mode in ("joint", "direct"):
            n = normals[row, idx]
            v = post[:, 2:4]
            to_end = edge[:, 3:5] - edge[:, 1:3]
            length = to_end.norm(dim=-1).clamp_min(1e-8)
            along = ((post[:, :2] - edge[:, 1:3]) * to_end).sum(-1) / length
            others = gaps.masked_fill(marked | (c[..., 9] == 0), 10.0).amin(-1)
            x = torch.stack(
                [
                    (v * n).sum(-1) / 5,
                    (v * tangent(n)).sum(-1) / 5,
                    post[:, 5] * edge[:, 6] / 5,
                    n[:, 0],
                    n[:, 1],
                    edge[:, 0],
                    edge[:, 5],
                    edge[:, 6],
                    edge[:, 7],
                    edge[:, 8],
                    (t - raw_time) / quantum,
                    raw_time / DT,
                    h / DT,
                    torch.minimum(along, length - along) / edge[:, 6],
                    others / edge[:, 6],
                    gaps[row, idx] / edge[:, 6],
                ],
                -1,
            )
            raw = self.net(x.clamp(-8, 8)).tanh() * unique[:, None]
            if self.mode == "joint":
                t = (t + quantum * raw[:, 0]).clamp_min(0).minimum(h)
                post = fly(s, t, quantum)
                normals, _ = nearest_normal(post[:, None, :2], c)
        times = t[:, None].expand_as(gaps)
        impulse, correction, spin = contact_response(s, c, times, normals, gaps, marked, h[:, None])
        n = normals[row, idx]
        tang = tangent(n)
        base_v = post[:, 2:4] + impulse.sum(1)
        base_p = post[:, :2] + (correction[..., None] * normals).sum(1)
        omega = post[:, 5] + spin.sum(1)
        angle = post[:, 4]
        if self.mode == "joint":
            # Bound the outgoing impulse by unilateral restitution and Coulomb friction.
            speed = (-(post[:, 2:4] * n).sum(-1)).clamp_min(0)
            nominal_n = (1 + edge[:, 8]) * speed
            slip = (post[:, 2:4] * tang).sum(-1) - post[:, 5] * edge[:, 6]
            nominal_limit = nominal_n * edge[:, 7]
            nominal_t = (-slip / 3).clamp(-nominal_limit, nominal_limit)
            new_n = ((1 + edge[:, 8] + 0.08 * raw[:, 1]).clamp(1, 2)) * speed
            limit = new_n * edge[:, 7]
            new_t = (nominal_t + 0.08 * new_n * raw[:, 2]).clamp(-limit, limit)
            # Subtract the nominal scalar law before projection. Projecting an
            # already rounded vector back onto n,t produces nonzero sham effects.
            dn = torch.where(unique, new_n - nominal_n, 0.0)
            dtan = torch.where(unique, new_t - nominal_t, 0.0)
            base_v = base_v + dn[:, None] * n + dtan[:, None] * tang
            omega = omega - 2 * dtan / edge[:, 6]
        elif self.mode == "ordered":
            # Restitution uses the normal speed before the detection tick's gravity
            # increment. The constraint impulse acts on the velocity after gravity.
            vn = (post[:, 2:4] * n).sum(-1)
            prior_vn = vn + 9.81 * n[:, 1] * t.clamp_max(quantum)
            jn = (-vn - edge[:, 8] * prior_vn).clamp_min(0)
            slip = (post[:, 2:4] * tang).sum(-1) - post[:, 5] * edge[:, 6]
            jt = (-slip / 3).clamp(-edge[:, 7] * jn, edge[:, 7] * jn)
            j = impulse[row, idx]
            dn = torch.where(unique, jn - (j * n).sum(-1), 0.0)
            dtan = torch.where(unique, jt - (j * tang).sum(-1), 0.0)
            base_v = base_v + dn[:, None] * n + dtan[:, None] * tang
            omega = omega - 2 * dtan / edge[:, 6]
        elif self.mode == "direct":
            base_p = base_p + 0.003 * (raw[:, 0, None] * n + raw[:, 1, None] * tang)
            base_v = base_v + 0.12 * (raw[:, 2, None] * n + raw[:, 3, None] * tang)
            angle = angle + 0.01 * raw[:, 4]
            omega = omega + raw[:, 5]
        return t, torch.cat([base_p, base_v, angle[:, None], omega[:, None]], -1)


def step(s, c, mode, hz):
    return bounded(s, c, physics_dt=1 / hz, transition=mode)


@torch.inference_mode()
def forecast(s, c, model, hz=1920):
    states = [s]
    for _ in range(120):
        s = step(s, c, model, hz)[0]
        states.append(s)
    return torch.stack(states, 1)
