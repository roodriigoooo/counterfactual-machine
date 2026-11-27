"""Check broad-phase equivalence, locality, sustained contacts and frame boundaries."""

import json

import numpy as np
import torch

from .architecture_data import digest
from .contact_events import event_labels
from .contact_suite import fixtures, match_events
from .continuation_data import DATA, PROTOCOL, REPORT
from .continuation_experiment import read_split, write
from .continuation_solver import DT, rollout
from .geometry import apply_edit, candidates
from .recurrent import pack
from .schemas import EditSpec, SceneSpec
from .simulator import simulate


def onsets(clocks, mode):
    # Match the four-step study's admission-onset convention. For event slots,
    # an empty optional slot carries no elapsed time. Suppress consecutive
    # frame-start support impulses on the same surface.
    if mode == "four":
        active = np.isfinite(clocks).reshape(-1, clocks.shape[-1])
        began = active & ~np.vstack([np.zeros_like(active[0]), active[:-1]])
        frame, slot, edge = np.nonzero(began.reshape(clocks.shape))
    else:
        frame, slot, edge = np.nonzero(np.isfinite(clocks))
        previous = np.concatenate([np.zeros((1, clocks.shape[-1]), bool), np.isfinite(clocks).any(1)[:-1]])
        keep = (clocks[frame, slot, edge] > 1e-7) | ~previous[frame, edge]
        frame, slot, edge = frame[keep], slot[keep], edge[keep]
    return [(int(e), float(k * DT + clocks[k, h, e])) for k, h, e in zip(frame, slot, edge)]


def full_release(name):
    rows, d, _m = read_split(name)
    out = {}
    for mode in ("bounded", "four"):
        states, events, overflow = rollout(d["states"][:, 0], d["contexts"], mode, audit=True)
        records = []
        for i, r in enumerate(rows):
            preds = onsets(events[i].numpy(), mode)
            stats = {}
            for e, obj in enumerate(r["ids"]):
                actual = [v["time_s"] for v in r["callbacks"] if v["phase"] == "begin" and v["object"] == obj]
                pred = [t for edge, t in preds if edge == e]
                stats[obj] = match_events(actual, pred)
            item = {"parent": r["parent"], "edit": r["edit"], "surface": stats}
            if r["edit"] == 0:
                base = i
            else:
                earliest = min(
                    (
                        v["time_s"]
                        for k in (base, i)
                        for v in rows[k]["callbacks"]
                        if v["phase"] == "begin" and v["object"] == "bumper"
                    ),
                    default=3.0,
                )
                delta = states[i, :, :2] - states[base, :, :2]
                prefix = torch.arange(121, dtype=torch.float64) * DT < earliest - 1e-9
                item.update(
                    pre_max_m=float(delta[prefix].norm(dim=-1).max()),
                    no_bumper=earliest > 2,
                    effect_rmse_m=float(
                        (delta - (d["states"][i, :, :2] - d["states"][base, :, :2]))
                        .square()
                        .sum(-1)
                        .mean()
                        .sqrt()
                    ),
                )
            records.append(item)
        err = states - d["states"]
        out[mode] = {
            "rows": records,
            "position_rmse_m": float(err[:, :, :2].square().sum(-1).mean().sqrt()),
            "velocity_p95": float(torch.quantile(err[:, :, 2:4].norm(dim=-1), 0.95)),
            "overflow": int(overflow.sum()) if overflow is not None else None,
        }
    return out


def gate_grid():
    pilot = json.loads((DATA / "pilot/manifest.json").read_text())
    records = []
    for parent in pilot["records"]:
        scene = SceneSpec.model_validate(parent["scene"])
        grid, valid, _ = candidates(scene, 257)
        pairs = [pack(apply_edit(scene, EditSpec(displacement=float(u)))) for u in grid[valid]]
        s, c = [torch.tensor(np.stack(arr)) for arr in zip(*pairs)]
        a, ac, ao = rollout(s, c, gated=True, audit=True)
        b, bc, bo = rollout(s, c, gated=False, audit=True)
        records.append(
            {
                "parent": scene.seed,
                "valid": len(s),
                "state_max": float((a - b).abs().max()),
                "clock_max": float((ac.nan_to_num(posinf=0) - bc.nan_to_num(posinf=0)).abs().max()),
                "slot_disagreements": int((ac.isfinite() != bc.isfinite()).sum()),
                "overflow_disagreements": int((ao != bo).sum()),
            }
        )
    return records


def run():
    torch.set_num_threads(1)
    fixture = []
    for scene in fixtures():
        for hz in (1920, 3840):
            tr = simulate(scene, physics_hz=hz, record_responses=True)
            labels = event_labels(tr)["event_labels"]
            s, c = pack(scene)
            for mode in ("bounded", "four"):
                _states, clocks, ov = rollout(torch.tensor(s[None]), torch.tensor(c[None]), mode, audit=True)
                pred = onsets(clocks[0].numpy(), mode)
                compare = []
                for e, obj in enumerate(tr.object_ids):
                    truth = [v["time"] for v in tr.events if v["phase"] == "begin" and v["objects"][1] == obj]
                    compare.append(match_events(truth, [t for edge, t in pred if edge == e]))
                fixture.append(
                    {
                        "scene": scene.id,
                        "hz": hz,
                        "solver": mode,
                        "support_intervals": int(labels[..., 5].any(-1).sum()),
                        "truth_begins": sum(v["matched"] + v["missed"] for v in compare),
                        "invented": sum(v["invented"] for v in compare),
                        "missed": sum(v["missed"] for v in compare),
                        "overflow": int(ov.sum()) if ov is not None else None,
                    }
                )
    write(
        REPORT / "audit.json",
        {
            "protocol": digest(PROTOCOL),
            "fixture": fixture,
            "releases": {s: full_release(s) for s in ("evaluation", "ordinary_evaluation")},
            "pilot_grids": gate_grid(),
        },
    )


if __name__ == "__main__":
    run()
