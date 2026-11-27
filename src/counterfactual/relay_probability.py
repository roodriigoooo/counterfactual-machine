"""Fit clearance-error distributions and calibrate a simultaneous band over one shelf's edits."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .relay_encounter import CONFIG, ROOT, load, sha, write

MODES = ("constant", "clock", "context")
CONTEXT = (
    "log_clock_disagreement",
    "closest_s",
    "relative_speed_m_s",
    "since_impact_s",
    "log_impact_count",
    "last_shelf",
    "last_floor",
    "shelf_edge_distance_m",
)


def dataset(root, start, count):
    parents = [
        load(root / "parents" / f"{p}.json") for p in range(start, start + count)
    ]
    if any(p["protocol"] != sha(CONFIG) for p in parents):
        raise ValueError("Encounter data belongs to a different protocol")
    x = np.array(
        [
            [[r["forecast"][str(h)] for h in (480, 960)] for r in p["rows"]]
            for p in parents
        ]
    )
    y = np.array(
        [
            [[r["reference"][str(h)] for h in (1920, 3840)] for r in p["rows"]]
            for p in parents
        ]
    )
    return x, y[..., 0], y[..., 2] >= 0


def design(x, mode):
    fast, fine = x[..., 0, 0], x[..., 1, 0]
    delta = fine - fast
    cols = []
    if mode in ("clock", "context"):
        cols.append(np.log1p(abs(delta) / 0.001))
    if mode == "context":
        f = x[..., 1, :]
        cols.extend(
            [
                f[..., 1],
                f[..., 2],
                f[..., 3],
                np.log1p(f[..., 4]),
                f[..., 5],
                f[..., 6],
                f[..., 7],
            ]
        )
    return fine, delta, np.stack(cols, -1) if cols else np.empty((*fine.shape, 0))


def fit(x, y, mode):
    fine, delta, features = design(x, mode)
    dims = features.shape[-1]
    mean = features.mean(axis=(0, 1))
    sd = features.std(axis=(0, 1)).clip(0.001)
    z = torch.as_tensor((features - mean) / sd)
    fine, delta, y = map(torch.as_tensor, (fine, delta, y))
    coeff = torch.zeros(dims + 1, dtype=torch.float64, requires_grad=True)
    gain = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [coeff, gain],
        max_iter=140,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def objective():
        opt.zero_grad()
        location = fine + (2 * gain.tanh() * delta if mode != "constant" else 0)
        log_scale = (-4.6 + coeff[0] + z @ coeff[1:]).clamp(-12.0, 1.0)
        residual = (y - location[..., None]) / log_scale.exp()[..., None]
        loss = (
            log_scale[..., None] + 1.5 * torch.log1p(0.5 * residual.square())
        ).mean()
        loss = loss + 0.002 * coeff[1:].square().sum()
        loss.backward()
        return loss

    opt.step(objective)
    return {
        "mode": mode,
        "features": list(CONTEXT[:dims]),
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "coefficients": coeff.detach().tolist(),
        "gain": float(2 * gain.detach().tanh()),
    }


def decode(x, model):
    fine, delta, f = design(x, model["mode"])
    c = np.asarray(model["coefficients"])
    z = (f - model["mean"]) / model["sd"]
    return fine + model["gain"] * delta, np.exp(
        np.clip(-4.6 + c[0] + z @ c[1:], -12.0, 1.0)
    )


def probability(location, scale):
    # Exact t_2 CDF evaluated at zero, with a stable small-tail expression.
    z = location / scale
    h = np.hypot(z, np.sqrt(2.0))
    tail = 1 / (2 + np.abs(z) * (np.abs(z) + h))
    return np.where(z >= 0, tail, 1 - tail)


def band_radius(y, location, scale, alpha):
    scores = (abs(y - location[..., None]) / scale[..., None]).max(axis=(1, 2))
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if rank > len(scores):
        raise ValueError("Too few calibration parents for a finite band")
    return float(np.sort(scores)[rank - 1]), rank


def estimate(values, seed=5721, draws=2000):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(len(v), size=(draws, len(v)))].mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "parents": len(v),
    }


def worst_edits(root, first_parent, target, location, scale, q):
    """Record the most extreme standardized error in each complete test grid."""
    standardized = abs(target - location[..., None]) / scale[..., None]
    result = []
    for i, grid in enumerate(standardized):
        edit, clock = np.unravel_index(grid.argmax(), grid.shape)
        parent = first_parent + i
        row = load(root / "parents" / f"{parent}.json")["rows"][edit]
        hz = (1920, 3840)[clock]
        result.append(
            {
                "parent": parent,
                "edit_m": row["edit_m"],
                "reference_hz": hz,
                "covered_grid": bool(grid[edit, clock] <= q),
                "standardized_error": float(grid[edit, clock]),
                "forecast_closest_s": row["forecast"]["960"][1],
                "actual_closest_s": row["reference"][str(hz)][1],
            }
        )
    return result


def rates(hit, p):
    brier = (hit - p[..., None]) ** 2
    logscore = -np.where(
        hit,
        np.log(np.maximum(p[..., None], 1e-15)),
        np.log(np.maximum(1 - p[..., None], 1e-15)),
    )
    return brier, logscore


def run(root):
    cfg = load(CONFIG)["representation"]
    x, y, _ = dataset(root, *cfg["train"])
    cx, cy, _ = dataset(root, *cfg["calibration"])
    tx, ty, hit = dataset(root, *cfg["test"])
    near = abs(tx[..., 1, 0]) <= 0.05
    report = {}
    losses = {}
    raw = {}
    extremes = []
    for mode in MODES:
        m = fit(x, y, mode)
        loc, scale = decode(tx, m)
        q, rank = band_radius(cy, *decode(cx, m), cfg["alpha"])
        lo, hi = loc - q * scale, loc + q * scale
        if mode == "context":
            extremes = worst_edits(root, cfg["test"][0], ty, loc, scale, q)
        p = probability(loc, scale)
        brier, logscore = rates(hit, p)
        losses[mode] = brier.mean(axis=1)
        certain = (hi < 0) | (lo > 0)
        covered = (lo[..., None] <= ty) & (hi[..., None] >= ty)
        wrong = certain[..., None] & ((hi < 0)[..., None] != hit)
        report[mode] = {
            "model": m,
            "calibration_q": q,
            "rank": rank,
            "brier": brier.mean(axis=(0, 1)).tolist(),
            "log_score": logscore.mean(axis=(0, 1)).tolist(),
            "near_brier": brier[near].mean(axis=0).tolist(),
            "sign_errors": np.sum((p >= 0.5)[..., None] != hit, axis=(0, 1)).tolist(),
            "band": {
                "parents_covered": int(covered.all(axis=(1, 2)).sum()),
                "parents": len(tx),
                "resolved": int(certain.sum()),
                "cells": certain.size,
                "wrong_resolved": wrong.sum(axis=(0, 1)).tolist(),
                "mean_width_mm": float(1000 * (hi - lo).mean()),
                "median_width_mm": float(1000 * np.median(hi - lo)),
            },
        }
        raw[mode] = {
            "location_m": loc.tolist(),
            "scale_m": scale.tolist(),
            "probability": p.tolist(),
        }
    for name, loc in zip(
        ("480", "960", "extrapolated"),
        (tx[..., 0, 0], tx[..., 1, 0], 2 * tx[..., 1, 0] - tx[..., 0, 0]),
    ):
        brier, _ = rates(hit, (loc <= 0).astype(float))
        report[name] = {
            "brier": brier.mean(axis=(0, 1)).tolist(),
            "errors": brier.sum(axis=(0, 1)).astype(int).tolist(),
        }
        losses[name] = brier.mean(1)
    contrasts = {}
    for a, b in (
        ("context", "clock"),
        ("clock", "constant"),
        ("context", "constant"),
        ("context", "960"),
    ):
        contrasts[f"{a}-{b}"] = [
            estimate(v, cfg["seed"], cfg["bootstrap"])
            for v in (losses[a] - losses[b]).T
        ]
    sets = {}
    for split in ("train", "calibration", "test"):
        _, clear, events = dataset(root, *cfg[split])
        sets[split] = {
            "parents": len(events),
            "cells": int(events[..., 0].size),
            "contacts": events.sum(axis=(0, 1)).tolist(),
            "sign_callback_disagreements": ((clear <= 0) != events)
            .sum(axis=(0, 1))
            .tolist(),
            "rate_disagreements": int((events[..., 0] != events[..., 1]).sum()),
        }
    write(
        root / "probability.json",
        {
            "protocol": sha(CONFIG),
            "models": report,
            "contrasts": contrasts,
            "near_cells": int(near.sum()),
            "sets": sets,
            "test_parent_start": cfg["test"][0],
            "predictions": raw,
            "context_worst_edits": extremes,
        },
    )
    write(root / "report-kind.json", {"kind": "relay-encounter-v1"})
    print("Context minus clock Brier", contrasts["context-clock"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    torch.set_num_threads(1)
    run(parser.parse_args().output)
