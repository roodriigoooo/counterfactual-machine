"""One analytic ablation: carry a single planar support across observation steps."""

import torch

from .contact_mechanics import nearest_normal
from .continuation_solver import PHYSICS_DT, bounded
from .recurrent import DT

SLOP = 0.001


def support_step(state, context, history):
    """Advance resting/sliding/rolling contacts, with the bounded solver as fallback."""
    # Short flight, an impact or a lost supporting face clears its history through
    # the fallback's actual admissions. Newly encountered surfaces must first collide.
    active = torch.zeros_like(history)
    following = torch.empty_like(state)
    clocks = state.new_full((len(state), 2, context.shape[1]), torch.inf)
    prior = history.any(-1)
    if prior.any():
        rows = torch.where(prior)[0]
        s, c = state[rows], context[rows]
        n, gap = nearest_normal(s[:, None, :2], c)
        vn = (s[:, None, 2:4] * n).sum(-1)
        gn = -9.81 * n[..., 1]
        ab = c[..., 3:5] - c[..., 1:3]
        length = ab.norm(dim=-1)
        along = ((s[:, None, :2] - c[..., 1:3]) * ab).sum(-1) / length.clamp_min(1e-9)
        # Upper bound includes tangential acceleration due to friction and spin.
        travel = DT * (s[:, 2:4].norm(dim=-1) + 3 * 9.81 * DT) + 3 * SLOP
        close = (gap <= travel[:, None]) & (c[..., 9] > 0)
        held = (
            history[rows]
            & (c[..., 0] == 0)
            & (c[..., 9] > 0)
            & (gn < 0)
            & (gap.abs() <= SLOP + 1e-6)
            & (vn.square() <= -2 * gn * SLOP)
            & (vn + gn * DT <= 0)  # a unilateral constraint cannot pull the disk back
            & (along > travel[:, None])
            & (length - along > travel[:, None])
            & (close.sum(-1, keepdim=True) == 1)
        )
        local, edge = torch.where(held)
        if len(local):
            ix = rows[local]
            active[ix, edge] = True
            normal = n[local, edge]
            tangent = torch.stack([-normal[:, 1], normal[:, 0]], -1)
            vel = state[ix, 2:4]
            vt = (vel * tangent).sum(-1)
            r = c[local, edge, 6]
            omega = state[ix, 5]
            gt = -9.81 * tangent[:, 1]
            # Project end-of-step slip onto the Coulomb cone for a uniform disk.
            # The M^-1 terms yield 1 + m*r^2/I = 3 in tangential coordinates.
            normal_j = (-gn[local, edge] * DT - vn[local, edge]).clamp_min(0)
            limit = c[local, edge, 7] * normal_j
            jt = ((r * omega - vt - gt * DT) / 3).clamp(-limit, limit)
            vt_next = vt + gt * DT + jt
            omega_next = omega - 2 * jt / r
            travel_t = vt * DT + (vt_next - vt) * (DT - PHYSICS_DT) / 2
            following[ix] = torch.cat(
                [
                    state[ix, :2]
                    - gap[local, edge, None] * normal
                    + travel_t[:, None] * tangent,
                    vt_next[:, None] * tangent,
                    (state[ix, 4] + (omega + omega_next) * DT / 2)[:, None],
                    omega_next[:, None],
                ],
                dim=-1,
            )
            clocks[ix, 0, edge] = 0
    fallback = ~active.any(-1)
    if fallback.any():
        following[fallback], clocks[fallback], _, _ = bounded(
            state[fallback], context[fallback]
        )
    return following, clocks, active


@torch.inference_mode()
def support_rollout(s, c):
    history = torch.zeros_like(c[..., 0], dtype=torch.bool)
    states, events, held = [s], [], []
    for _ in range(120):
        s, clocks, active = support_step(s, c, history)
        history = clocks.isfinite().any(1)
        states.append(s)
        events.append(clocks)
        held.append(active)
    return torch.stack(states, 1), torch.stack(events, 1), torch.stack(held, 1)
