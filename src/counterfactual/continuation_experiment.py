"""Fit and evaluate a conditional surface/time readout on paired fresh releases."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .architecture_data import digest
from .continuation_data import DATA, PROTOCOL, REPORT
from .continuation_solver import DT, four, query, rollout

WINDOW = 2 * DT
BINS = 16
NONE = -1


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging:
        tmp = Path(staging) / path.name
        tmp.write_text(json.dumps(obj, allow_nan=False))
        tmp.replace(path)


def read_split(name):
    folder = DATA / name
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol"] != digest(PROTOCOL) or manifest["archive"] != digest(folder / "branches.npz"):
        raise ValueError(f"Changed inputs in {name}")
    with np.load(folder / "branches.npz") as archive:
        d = {k: torch.from_numpy(archive[k]) for k in archive.files}
    rows = []
    for p in manifest["records"]:
        scene = p["scene"]
        ids = sorted([scene["bumper"]["id"], *(s["id"] for s in scene["segments"])])
        for b in p["branches"]:
            if b["row"] != len(rows):
                raise ValueError("Branch order mismatch")
            rows.append(dict(parent=scene["seed"], tick=p["tick"], ids=ids, **b))
    return rows, d, manifest


def gold(row):
    a, f = [row["audit"][hz] for hz in ("1920", "3840")]
    seq = a["order"]
    second = seq[1] if len(seq) > 1 else None
    time = next((t / 1000 for s, t in a["events"] if s == second), None)
    if a["tie"] or f["tie"]:
        stratum = "tie"
    elif a["order"] != f["order"]:
        stratum = "rate disagreement"
    elif max(len(x["events"]) for x in (a, f)) > 2:
        stratum = "overflow"
    elif not seq:
        stratum = "no first"
    else:
        stratum = "primary"
    return {
        "stratum": stratum,
        "first": row["ids"].index(seq[0]) if seq else NONE,
        "second": row["ids"].index(second) if second else NONE,
        "time": time,
    }


class ConditionalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, BINS)
        )

    def forward(self, x, mask):
        scores = self.edge(x) - np.log(BINS)
        logits = scores.masked_fill(~mask[..., None], -1e4).flatten(1)
        return torch.cat([logits, torch.zeros_like(logits[:, :1])], dim=1)


def prediction(times):
    values, ids = times.sort(-1)
    # No arbitrary cross-surface tie-break in the first slot.
    unique = values[:, 0].isfinite() & ((values[:, 1] - values[:, 0]) > 1e-7)
    second = unique & values[:, 1].isfinite()
    return (
        ids[:, 0].masked_fill(~unique, NONE),
        ids[:, 1].masked_fill(~second, NONE),
        values[:, 1].masked_fill(~second, torch.inf),
    )


def learned(head, times, x, mask):
    first = prediction(times)[0]
    with torch.no_grad():
        p = head(x, mask).softmax(-1)
    mass = p[:, :-1].reshape(len(x), -1, BINS)
    objprob = mass.sum(-1)
    choice = torch.cat([objprob, p[:, -1:]], -1).argmax(-1)
    present = (first != NONE) & (choice < objprob.shape[1])
    edge = choice.clamp_max(objprob.shape[1] - 1)
    dist = mass[torch.arange(len(x)), edge]
    bins = (torch.arange(BINS) + 0.5) / BINS
    elapsed = (dist * bins).sum(-1) / dist.sum(-1).clamp_min(1e-12)
    start = times.amin(-1).clamp_max(WINDOW)
    t = start + (WINDOW - start) * elapsed
    return (first, choice.masked_fill(~present, NONE), t.masked_fill(~present, torch.inf)), p


def fit(rows, d, config, seed, *, incoming_only=False):
    torch.manual_seed(seed)
    model = ConditionalHead()
    idx = torch.arange(len(rows))
    t, x, mask = query(
        d["states"][idx, [r["tick"] for r in rows]], d["contexts"], incoming_only=incoming_only
    )
    x, mask = x.clone(), mask.clone()
    first = prediction(t)[0]
    g = [gold(r) for r in rows]
    usable = torch.tensor(
        [v["stratum"] == "primary" and int(first[i]) == v["first"] for i, v in enumerate(g)]
    )
    labels = []
    for i, v in enumerate(g):
        if v["time"] is None:
            labels.append(t.shape[1] * BINS)
        else:
            start = float(t[i].min().clamp_max(WINDOW))
            frac = (v["time"] - start) / max(1e-8, WINDOW - start)
            labels.append(v["second"] * BINS + int(np.clip(frac * BINS, 0, BINS - 1)))
    kept = torch.where(usable)[0]
    labels = torch.tensor(labels)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    for _ in range(config["updates"]):
        ids = kept[torch.randint(len(kept), (config["batch"],))]
        loss = F.cross_entropy(model(x[ids], mask[ids]), labels[ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval(), len(kept)


def record(pred, i):
    first, second, t = pred
    return {
        "first": int(first[i]),
        "second": int(second[i]),
        "time": float(t[i]) if t[i].isfinite() else None,
    }


def metrics(rows, name):
    if not rows:
        return {"branches": 0}
    y = np.array([r["gold"]["second"] for r in rows])
    p = np.array([r[name]["second"] for r in rows])
    hits = (y == p) & (y != NONE)
    errors = [abs(r["gold"]["time"] - r[name]["time"]) * 1000 for i, r in enumerate(rows) if hits[i]]
    parents = {}
    for i, r in enumerate(rows):
        parents.setdefault(r["parent"], []).append(bool(y[i] == p[i]))
    return {
        "branches": len(y),
        "parents": len(parents),
        "parent_accuracy": float(np.mean([np.mean(v) for v in parents.values()])),
        "first_correct": sum(r["gold"]["first"] == r[name]["first"] for r in rows),
        "tp": int(hits.sum()),
        "missed": int(((y != NONE) & ~hits).sum()),
        "invented": int(((p != NONE) & ~hits).sum()),
        "precision": float(hits.sum() / max(1, (p != NONE).sum())),
        "recall": float(hits.sum() / max(1, (y != NONE).sum())),
        "timing_mae_ms": float(np.mean(errors)) if errors else None,
        "timing_p95_ms": float(np.quantile(errors, 0.95)) if errors else None,
    }


def pairs(rows, method):
    by_parent = {}
    for r in rows:
        by_parent.setdefault(r["parent"], {})[r["edit"]] = r
    out = {k: [0, 0] for k in ("created", "removed", "unchanged", "false_flip")}
    for group in by_parent.values():
        if 0.0 not in group:
            continue
        base = group[0.0]
        for edit, r in group.items():
            if not edit:
                continue
            y = [s["gold"]["second"] != NONE for s in (base, r)]
            p = [s[method]["second"] != NONE for s in (base, r)]
            kind = "unchanged" if y[0] == y[1] else "created" if y[1] else "removed"
            out[kind][0] += int(y == p)
            out[kind][1] += 1
            if kind == "unchanged":
                out["false_flip"][0] += int(p[0] != p[1])
                out["false_flip"][1] += 1
    return out


def evaluate(rows, d, heads):
    ids = torch.arange(len(rows))
    ticks = torch.tensor([r["tick"] for r in rows])
    c = d["contexts"]
    paths, timings, physical = {}, {}, {}
    for mode, gate in [("four", True), ("bounded", True), ("ungated", False)]:
        start = perf_counter()
        states, events, over = rollout(
            d["states"][:, 0], c, "four" if mode == "four" else "bounded", gate, True
        )
        timings[mode] = perf_counter() - start
        pos = (states[..., :2] - d["states"][..., :2]).norm(dim=-1)
        vel = (states[..., 2:4] - d["states"][..., 2:4]).norm(dim=-1)
        physical[mode] = {
            "position_rmse_m": float(pos.square().mean().sqrt()),
            "velocity_p95": float(torch.quantile(vel, 0.95)),
            "overflow_intervals": int(over.sum()) if over is not None else None,
        }
        if mode == "ungated":
            ref, clocks = paths["bounded"]
            physical["gate_comparison"] = {
                "max_state_difference": float((states - ref).abs().max()),
                "different_surface_slots": int((events.isfinite() != clocks.isfinite()).sum()),
                "max_clock_difference_s": float(
                    (events.nan_to_num(posinf=0) - clocks.nan_to_num(posinf=0)).abs().max()
                ),
            }
        else:
            paths[mode] = states, events
    exact = d["states"][ids, ticks]
    modes = {}
    for exposure in ("exact", "release"):
        state = exact if exposure == "exact" else paths["bounded"][0][ids, ticks]
        t, x, mask = query(state, c)
        p = {"bounded": prediction(t)}
        sf = exact if exposure == "exact" else paths["four"][0][ids, ticks]
        sf, a = four(sf, c)
        _, b = four(sf, c)
        p["four"] = prediction(torch.minimum(a.amin(1), DT + b.amin(1)))
        for seed, head in heads.items():
            p[f"head-{seed}"] = learned(head, t, x, mask)[0]
        records = [
            dict(
                parent=r["parent"],
                edit=r["edit"],
                tick=r["tick"],
                ids=r["ids"],
                gold=gold(r),
                **{k: record(v, i) for k, v in p.items()},
            )
            for i, r in enumerate(rows)
        ]
        strata = {}
        for row in records:
            strata.setdefault(row["gold"]["stratum"], []).append(row)
        group = strata["primary"]
        diffs = {}
        for r in group:
            y = r["gold"]["second"]
            delta = np.mean([r[f"head-{s}"]["second"] == y for s in heads]) - int(r["bounded"]["second"] == y)
            diffs.setdefault(r["parent"], []).append(delta)
        delta = np.array([np.mean(v) for v in diffs.values()])
        boots = np.random.default_rng(948).choice(delta, (2000, len(delta))).mean(1)
        modes[exposure] = {
            "records": records,
            "strata": {k: {m: metrics(v, m) for m in p} for k, v in strata.items()},
            "paired": {m: pairs(group, m) for m in p},
            "head_minus_bounded": {
                "mean": float(delta.mean()),
                "ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
            },
        }
    return {"modes": modes, "physical": physical, "batch_seconds": timings}


def run():
    torch.set_num_threads(1)
    config = json.loads(PROTOCOL.read_text())
    train, d, m = read_split("train")
    test, dt, mtest = read_split("evaluation")
    for manifest in (m, mtest):
        if any(
            manifest["exposure"].get(k, 0) < n
            for k, n in config["minimum_train_evaluation_parent_support"].items()
        ):
            raise ValueError("Independent parent support gate failed")
    heads = {}
    hashes = {}
    REPORT.mkdir(parents=True, exist_ok=True)
    for seed in config["training"]["seeds"]:
        cp = REPORT / f"head-{seed}.pt"
        binding = {"seed": seed, "protocol": digest(PROTOCOL), "train": digest(DATA / "train/manifest.json")}
        if cp.exists():
            saved = torch.load(cp, weights_only=True)
            if saved["binding"] != binding:
                raise ValueError("Head binding changed")
            model = ConditionalHead()
            model.load_state_dict(saved["model"])
        else:
            model, n = fit(train, d, config["training"], seed)
            tmp = cp.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "binding": binding, "training_rows": n}, tmp)
            tmp.replace(cp)
        heads[seed] = model.eval()
        hashes[str(cp)] = digest(cp)
    report = evaluate(test, dt, heads)
    report.update(
        protocol=digest(PROTOCOL), checkpoints=hashes, eval_manifest=digest(DATA / "evaluation/manifest.json")
    )
    write(REPORT / "experiment.json", report)
    print(json.dumps(report["modes"]["release"]["strata"]["primary"], indent=1))


if __name__ == "__main__":
    run()
