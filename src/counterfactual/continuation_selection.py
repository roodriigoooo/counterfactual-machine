"""Fresh selected-proposal tolerance audit with the four-step and bounded solvers."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .architecture_data import digest
from .architecture_evaluation import forecast_table
from .competition_scenes import propose
from .continuation_data import PROTOCOL, REPORT
from .continuation_experiment import ConditionalHead, learned, prediction, write
from .continuation_solver import DT, query, rollout
from .geometry import apply_edit, candidates
from .goals import goal_result
from .recurrent import pack
from .scenes import generate_goal, generate_scene
from .schemas import EditSpec
from .simulator import simulate
from .tolerance import select_with_tolerance


class GridForecast:
    def __init__(self, mode, heads):
        self.mode = mode
        self.heads = heads
        self.key = None

    @torch.inference_mode()
    def forecast(self, scene, edits):
        key = (scene.content_hash(), tuple(edits))
        if key != self.key:
            items = [pack(apply_edit(scene, EditSpec(displacement=float(u)))) for u in edits]
            s, c = [torch.tensor(np.stack(arr)) for arr in zip(*items)]
            states, events, _ = rollout(s, c, self.mode)
            self.cost = {}
            if self.mode == "bounded":
                start = perf_counter()
                # Released-state clock of first predicted bumper or ramp encounter.
                obj = sorted([scene.bumper.id, *(seg.id for seg in scene.segments)])
                edges = [i for i, o in enumerate(obj) if o != "floor"]
                any_hit = events[:, :, :, edges].isfinite().any(dim=(2, 3))
                tick = any_hit.int().argmax(1).clamp_max(118)
                times, x, mask = query(states[torch.arange(len(s)), tick], c)
                _, second, t = prediction(times)
                present = any_hit.any(-1)
                gap = x[..., 1].masked_fill(~mask, 15.0).min(-1).values
                self.aux = {
                    "analytic": np.column_stack([gap, (second >= 0).float(), t.clamp_max(2 * DT) / (2 * DT)])
                }
                self.cost["query_s"] = perf_counter() - start
                for seed, head in self.heads.items():
                    start = perf_counter()
                    _, p = learned(head, times, x, mask)
                    mass = p[:, :-1].reshape(len(p), -1, 16).sum(1)
                    elapsed = (mass * (torch.arange(16) + 0.5) / 16).sum(-1) / mass.sum(-1).clamp_min(1e-9)
                    first = times.amin(-1).clamp_max(2 * DT) / (2 * DT)
                    self.aux[f"head-{seed}"] = np.column_stack(
                        [gap, 1 - p[:, -1], first + (1 - first) * elapsed]
                    )
                    self.cost[f"head-{seed}_s"] = perf_counter() - start
                for features in self.aux.values():
                    features[~present.numpy()] = [15.0, 0.0, 1.0]
            self.states = states.numpy()
            self.scores = events.isfinite().any(2).float().numpy()
            self.key = key
        return self.states, self.scores


def one_parent(scene, heads, cfg):
    seed = scene.seed
    goals = [generate_goal(seed * 8 + j) for j in range(cfg["goals"])]
    grid, valid, _ = candidates(scene, cfg["grid"])
    original = simulate(scene)
    bases = [goal_result(original.physics_times, original.physics_positions, g) for g in goals]
    simcache = {}
    timings = {}
    rows = []
    for mode in ("bounded", "four"):
        model = GridForecast(mode, heads)
        table = forecast_table(model, scene, grid, valid, goals)
        cached = forecast_table(model, scene, grid, valid, goals)
        timings[mode] = dict(
            cold_table_s=table["table_s"],
            cached_table_s=cached["table_s"],
            forecast_s=table["cold_s"],
            **model.cost,
        )
        for j, g in enumerate(goals):
            if bases[j]["success"]:
                continue
            margins, passed = table["margins"][j].copy(), table["passed"][j].copy()
            zero = len(grid) // 2
            margins[zero], passed[zero] = bases[j]["margin"], False
            chosen = select_with_tolerance(grid, margins, passed, valid, cfg["epsilon"], cfg["gamma"])
            i = chosen["index"]
            r = {"parent": seed, "j": j, "model": mode, "selected": i is not None}
            if i is not None:
                u = float(grid[i])
                steps = round(chosen["effective_epsilon_m"] / (grid[1] - grid[0]))
                window = np.arange(i - steps, i + steps + 1)
                limiting = int(window[np.argmin(margins[window])])
                checks = []
                for off in cfg["offsets"]:
                    delta = u + off
                    key = round(delta, 12)
                    if key not in simcache:
                        try:
                            simcache[key] = simulate(scene, delta)
                        except ValueError:
                            simcache[key] = None
                    tr = simcache[key]
                    checks.append(
                        bool(tr and goal_result(tr.physics_times, tr.physics_positions, g)["success"])
                    )
                r.update(
                    edit=u, checks=checks, robust=all(checks), center=checks[2], index=i, limiting=limiting
                )
                if mode == "bounded":
                    ix = int(np.searchsorted(np.flatnonzero(valid), limiting))
                    r["features"] = {
                        name: [float(margins[limiting]), *v[ix].tolist()] for name, v in model.aux.items()
                    }
            rows.append(r)
    return {
        "seed": seed,
        "rows": rows,
        "timings": timings,
        "unique_simulations": 1 + sum(x is not None for x in simcache.values()),
    }


def risk_fit(rows, method):
    x = torch.tensor([r["features"][method] for r in rows], dtype=torch.float64)
    y = torch.tensor([not r["robust"] for r in rows], dtype=torch.float64)
    mu, sd = x.mean(0), x.std(0, unbiased=False).clamp_min(0.01)
    if y.unique().numel() == 1:
        # A constant outcome supplies no information to rank failures. Preserve ties exactly.
        return {
            "mean": mu.tolist(),
            "sd": sd.tolist(),
            "weights": [0.0] * (x.shape[1] + 1),
            "constant_class": int(y[0]),
        }
    x = (x - mu) / sd
    w = torch.zeros(x.shape[1] + 1, dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w[:-1] + w[-1], y) + 0.02 * w.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"mean": mu.tolist(), "sd": sd.tolist(), "weights": w.detach().tolist()}


def risk(r, m, fit):
    x = (np.array(r["features"][m]) - fit["mean"]) / fit["sd"]
    return float(x @ fit["weights"][:-1] + fit["weights"][-1])


def counts(rows):
    return {
        "proposed": len(rows),
        "all_five": sum(r["robust"] for r in rows),
        "failures": sum(not r["robust"] for r in rows),
        "center_success": sum(r["center"] for r in rows),
        "mean_edit_mm": float(np.mean([abs(r["edit"]) * 1000 for r in rows])) if rows else None,
    }


def run(enriched=False):
    torch.set_num_threads(1)
    cfg = json.loads(PROTOCOL.read_text())["selection"]
    output = REPORT
    if enriched:
        extra = Path("configs/continuation-risk-followup.json")
        cfg = json.loads(extra.read_text())
        output = REPORT / "enriched-risk"
    heads = {}
    binding = {"protocol": digest(PROTOCOL)}
    for seed in range(3):
        cp = REPORT / f"head-{seed}.pt"
        binding[str(seed)] = digest(cp)
        mdl = ConditionalHead()
        mdl.load_state_dict(torch.load(cp, weights_only=True)["model"])
        heads[seed] = mdl.eval()
    if enriched:
        binding["followup"] = digest(extra)
    models = ["analytic", *(f"head-{s}" for s in heads)]
    fits = {}
    out = {}
    for stage in ("calibration", "evaluation"):
        start, n = cfg[stage]
        parents = []
        for seed in range(start, start + cfg.get("limit", n)):
            try:
                scene = propose(seed) if enriched else generate_scene(seed)
            except ValueError:
                if enriched:
                    continue
                raise
            dest = output / "selection" / stage / f"{seed}.json"
            if dest.exists():
                rec = json.loads(dest.read_text())
                if rec["binding"] != binding:
                    raise ValueError("Saved proposal inputs changed")
            else:
                rec = one_parent(scene, heads, cfg)
                rec["binding"] = binding
                write(dest, rec)
            parents.append(rec)
            if len(parents) % 16 == 0:
                print(stage, len(parents), flush=True)
            if len(parents) == n:
                break
        if len(parents) != n:
            raise ValueError("Independent decision pool exhausted")
        rows = [r for p in parents for r in p["rows"]]
        proposed = [r for r in rows if r["model"] == "bounded" and r["selected"]]
        if stage == "calibration":
            training = [r for r in proposed if r["parent"] < parents[n // 2]["seed"]]
            threshold = [r for r in proposed if r["parent"] >= parents[n // 2]["seed"]]
            for m in models:
                f = risk_fit(training, m)
                scores = [risk(r, m, f) for r in threshold]
                f["cutoff"] = float(np.quantile(scores, cfg["coverage_fraction"], method="higher"))
                fits[m] = f
            write(
                output / "selection-calibration.json",
                {"fits": fits, "train": counts(training), "threshold": counts(threshold)},
            )
        else:
            out["tasks"] = len([r for r in rows if r["model"] == "bounded"])
            out["bounded"] = counts(proposed)
            out["four"] = counts([r for r in rows if r["model"] == "four" and r["selected"]])
            for m, f in fits.items():
                accepted = [r for r in proposed if risk(r, m, f) <= f["cutoff"]]
                out[m] = dict(
                    **counts(accepted),
                    curve=[
                        counts([r for r in proposed if risk(r, m, f) <= cut])
                        for cut in sorted({risk(r, m, f) for r in proposed})
                    ],
                )
            out["timings"] = {
                m: {
                    k: np.quantile([p["timings"][m][k] for p in parents], [0.5, 0.95]).tolist()
                    for k in parents[0]["timings"][m]
                }
                for m in ("bounded", "four")
            }
            write(output / "selection.json", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enriched", action="store_true")
    run(parser.parse_args().enriched)
