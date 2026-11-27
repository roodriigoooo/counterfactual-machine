"""Known, hidden and probed mass in identical two-body forecasts on two layout families."""

import json
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .architecture_data import digest
from .continuation_experiment import write
from .impact_data import CONFIG, ROOT, save_weights
from .relay_forecast import forecast, layout
from .relay_mass import probe_velocities
from .relay_world import release
from .tolerance import select_with_tolerance

MODES = ("known", "prior", "probe-ratio", "probe-learned")


def reading(world, seed):
    before, after = probe_velocities(world.receiver, 1920, speed=0.0 if seed % 4 == 0 else 1.0)
    return (after.round(2) - before).astype(np.float32)


class Probe(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU(), nn.Linear(24, 1))


def fit(cfg):
    sets = []
    for stage in ("train", "validation"):
        start, n = cfg[stage]
        worlds = [layout(i) for i in range(start, start + n)]
        x = torch.tensor(np.stack([reading(w, w.scene.seed) for w in worlds]))
        y = torch.tensor([w.receiver.mass for w in worlds])
        sets.append((x[x[:, 1] > 0], y[x[:, 1] > 0]))
    (x, y), (vx, vy) = sets
    models = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        net = Probe()
        opt = torch.optim.Adam(net.parameters(), lr=0.005)
        history = []
        states = []
        for k in range(1, 601):
            loss = (net(x)[:, 0] - y).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if k in (200, 400, 600):
                with torch.no_grad():
                    history.append(float((net(vx)[:, 0] - vy).abs().mean()))
                    states.append({name: p.clone() for name, p in net.state_dict().items()})
        best = int(np.argmin(history))
        net.load_state_dict(states[best])
        net.eval()
        cp = ROOT / f"probe-{seed}.pt"
        save_weights(cp, net)
        models.append((history[best], net))
        write(
            ROOT / f"probe-{seed}.json",
            {"validation_mae_kg": history, "chosen_step": (best + 1) * 200, "protocol": digest(CONFIG)},
        )
    # Freeze the median validation run as a single deployable encoder.
    rank = np.argsort([v for v, _ in models])
    return models[rank[1]][1], int(rank[1])


def one(world, net, cfg):
    seed = world.scene.seed
    probe = reading(world, seed)
    prior = cfg["prior_mass"]
    if probe[1] > 0:
        ratio = float(np.clip(-probe[0] / probe[1], 0.55, 2.2))
        learned = float(net(torch.from_numpy(probe)).detach().clamp(0.55, 2.2).item())
    else:
        ratio = learned = prior
    masses = [world.receiver.mass, prior, ratio, learned]
    grid = np.linspace(-world.scene.ball.radius, world.scene.ball.radius, cfg["grid"])
    size = len(grid)
    hz = cfg["forecast_hz"]
    start = perf_counter()
    paths, first, travel, imbalance = forecast(
        world, np.tile(grid, len(masses)), np.repeat(masses, size), hz
    )
    batch_s = perf_counter() - start
    disabled = forecast(world, grid, np.full(size, masses[0]), hz, False)
    if disabled[2].max() > 1e-9:
        raise ValueError("Disabled pair moved receiver horizontally")
    ref = {}
    verification = {}
    for rate in cfg["references_hz"]:
        rows = [release(world, float(u), rate) for u in grid]
        ref[rate] = (
            np.stack([xy[:: rate // 60] for xy, _, _ in rows]),
            np.array([evt.get("transfer", -1) for _, evt, _ in rows]),
            np.array([d for _, _, d in rows]),
        )
        verification[rate] = {round(float(u), 12): d for u, (_, _, d) in zip(grid, rows)}
    out = {}
    offsets = np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
    for j, (mode, mass) in enumerate(zip(MODES, masses)):
        sl = slice(j * size, (j + 1) * size)
        path, t, m = paths[sl], first[sl], travel[sl] - cfg["target_m"]
        valid = np.ones(size, bool)
        passed = m >= 0
        orig = ref[1920][2][size // 2] - cfg["target_m"]
        margins = m.copy()
        margins[size // 2] = orig
        passed[size // 2] = orig >= 0
        idx = (
            select_with_tolerance(grid, margins, passed, valid, 0.005, 0.03)["index"]
            if orig < 0
            else None
        )
        rec = {
            "mass_kg": mass,
            "predicted_travel_m": travel[sl].tolist(),
            "first_transfer_tick": t.tolist(),
            "edit": None if idx is None else float(grid[idx]),
            "rates": {},
        }
        for rate, (xy, actual, dist) in ref.items():
            exists = actual >= 0
            pred = t >= 0
            matched = exists & pred
            obs = np.arange(121)[None, :] / 60
            pre = obs < np.where(exists, actual / rate, 3)[:, None]
            ownpre = obs < np.where(pred, t / hz, 3)[:, None]
            delta = np.linalg.norm(path[:, :, 1] - disabled[0][:, :, 1], axis=-1)
            checks = []
            if idx is not None:
                for off in offsets:
                    key = round(float(grid[idx] + off), 12)
                    if key not in verification[rate]:
                        verification[rate][key] = release(world, key, rate)[2]
                    checks.append(verification[rate][key] >= cfg["target_m"])
            rec["rates"][str(rate)] = {
                "path_rmse_mm": (
                    1000 * np.sqrt(np.mean(np.sum((path - xy) ** 2, -1), axis=1)).mean(0)
                ).tolist(),
                "scoring_correct": int(np.sum((m >= 0) == (dist >= cfg["target_m"]))),
                "invented_transfer": int((pred & ~exists).sum()),
                "missed_transfer": int((exists & ~pred).sum()),
                "matched": int(matched.sum()),
                "timing_abs_ms": (1000 * np.abs(t[matched] / hz - actual[matched] / rate)).tolist(),
                "prefix_before_true_transfer_mm": float(1000 * delta[pre].max()),
                "prefix_before_predicted_transfer_mm": float(1000 * delta[ownpre].max()),
                "checks": list(map(bool, checks)),
            }
        out[mode] = rec
    return {
        "parent": seed,
        "direction": world.direction,
        "probe_delta_m_s": probe.tolist(),
        "mass_kg": masses[0],
        "original_failed": bool(ref[1920][2][size // 2] < cfg["target_m"]),
        "grid_m": grid.tolist(),
        "reference_travel": {str(k): v[2].tolist() for k, v in ref.items()},
        "modes": out,
        "forecast_batch_s": batch_s,
        "disabled_max_travel_m": float(disabled[2].max()),
        "pair_impulse_imbalance_kg_m_s": imbalance,
    }


def run():
    torch.set_num_threads(1)
    cfg = json.loads(CONFIG.read_text())["relay"]
    ROOT.mkdir(parents=True, exist_ok=True)
    net, seed = fit(cfg)
    for family in ("ordinary", "elevated"):
        rows = []
        start, n = cfg[family]
        for i in range(start, start + n):
            rows.append(one(layout(i, family == "elevated"), net, cfg))
            if len(rows) % 8 == 0:
                print(family, len(rows), flush=True)
        write(
            ROOT / f"relay-{family}.json",
            {"protocol": digest(CONFIG), "encoder_seed": seed, "rows": rows},
        )


if __name__ == "__main__":
    run()
