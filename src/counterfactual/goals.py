"""Exact distances on the piecewise-linear path, including clipped interval ends.

Simulator verification supplies physics-rate paths. This is a numerical monitor,
not a guarantee about a continuous physical trajectory between physics steps.
"""
import numpy as np
from .schemas import GoalSpec
from .geometry import TOLERANCE


def goal_result(times, positions, goal: GoalSpec):
    t, p = np.asarray(times), np.asarray(positions)
    if t[0] > goal.start+TOLERANCE or t[-1] < goal.end-TOLERANCE:
        raise ValueError("path must cover the goal interval")
    if not np.isfinite(p).all():
        return {"success": False, "margin": None, "valid": False}
    interior = (t > goal.start) & (t < goal.end)
    endpoints = np.array([[np.interp(v, t, p[:, j]) for j in range(2)] for v in [goal.start, goal.end]])
    q = np.concatenate([endpoints[:1], p[interior], endpoints[1:]])
    r = goal.region
    if r.kind == "halfplane":
        distance = float(np.min(r.offset-q@np.array(r.normal)))
    else:
        a, b = q[:-1]-r.center, q[1:]-r.center
        d = b-a
        u = np.clip(-np.sum(a*d, axis=1)/np.maximum(np.sum(d*d, axis=1), 1e-30), 0, 1)
        distance = float(np.min(np.linalg.norm(a+u[:, None]*d, axis=1))-r.radius)
    margin = -distance if goal.kind == "reach" else distance
    success = margin >= -TOLERANCE if goal.kind == "reach" else margin > TOLERANCE
    return {"success": bool(success), "margin": margin, "valid": True}
