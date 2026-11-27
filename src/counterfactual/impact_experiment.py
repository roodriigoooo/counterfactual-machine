"""Fit matched supervised impact decoders and score complete, free-running edited releases."""

import json
from collections import defaultdict
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .architecture_data import digest
from .continuation_experiment import write
from .impact_data import CONFIG, DATA, RATES, ROOT, read, save_weights
from .impact_model import Impact, forecast, step


def branch_metrics(rows, data, model, hz):
    ctx = torch.from_numpy(data["contexts"])
    before = perf_counter()
    paths = forecast(torch.from_numpy(data["states"][:, 0, 0]), ctx, model, hz).numpy()
    elapsed = perf_counter() - before
    result = []
    for i, row in enumerate(rows):
        if row["edit"] == 0:
            base = i
        dif = paths[i, :, :2] - data["states"][i, :, :, :2]
        sibling = paths[i, :, :2] - paths[base, :, :2]
        prefix = (
            np.arange(121) / 60 < min(row["first_bumper_s"][0], rows[base]["first_bumper_s"][0]) - 1e-9
        )
        effect = sibling - (data["states"][i, :, :, :2] - data["states"][base, :, :, :2])
        result.append(
            {
                **{k: row[k] for k in ("parent", "family", "edit")},
                "rmse_mm": (1000 * np.sqrt((dif**2).sum(-1).mean(-1))).tolist(),
                "sibling_rmse_mm": (1000 * np.sqrt((effect**2).sum(-1).mean(-1))).tolist(),
                "prefix_max_mm": float(1000 * np.max(np.linalg.norm(sibling[prefix], axis=-1)))
                if prefix.any()
                else None,
            }
        )
    return {"rows": result, "seconds": elapsed}


def group(records, key="rmse_mm", edited=False):
    grouped = defaultdict(list)
    for r in records:
        if not edited or r["edit"] != 0:
            grouped[(r["family"], r["parent"])].append(r[key])
    return {
        fam: np.stack([np.mean(val, 0) for (f, _), val in grouped.items() if f == fam])
        for fam in ("competing", "ordinary")
    }


def score(records):
    return np.mean([v[:, 0].mean() for v in group(records).values()]).item()


def train(mode, seed, train_data, valid):
    cfg = json.loads(CONFIG.read_text())["training"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(317 + seed)
    model = Impact(mode)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    s = torch.from_numpy(train_data["states"][:, 0])
    c = torch.from_numpy(train_data["contexts"])
    occupied = train_data["contact"][:, 0].any(-1)
    # Half the samples are the first occupied frame after a clear frame. This
    # retains repeated support, impact-free negatives and complete sibling releases.
    entry = occupied & ~np.pad(occupied[:, :-1], ((0, 0), (1, 0)))
    pools = [np.argwhere(entry), np.argwhere(occupied), np.argwhere(~occupied)]
    scale = s.new_tensor([0.001, 0.001, 0.04, 0.04, 0.005, 0.4])
    hist = []
    configs = []
    batch = cfg["batch"]
    counts = [batch // 2, batch - batch // 2 - batch // 8, batch // 8]
    for k in range(1, cfg["updates"] + 1):
        selected = np.vstack([p[rng.integers(len(p), size=n)] for p, n in zip(pools, counts)])
        i, t = selected.T
        pred = step(s[i, t], c[i], model, 1920)[0]
        loss = F.smooth_l1_loss((pred - s[i, t + 1]) / scale, torch.zeros_like(pred))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if k in cfg["checkpoints"]:
            v = branch_metrics(*valid, model, 1920)
            hist.append({"step": k, "loss": loss.item(), "validation_mm": score(v["rows"])})
            configs.append({key: w.detach().clone() for key, w in model.state_dict().items()})
    win = int(np.argmin([r["validation_mm"] for r in hist]))
    model.load_state_dict(configs[win])
    folder = ROOT / f"{mode}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    save_weights(folder / "model.pt", model)
    write(
        folder / "training.json",
        {
            "validation": hist,
            "chosen": hist[win],
            "parameters": sum(p.numel() for p in model.parameters()),
            "data": digest(DATA / "train/branches.npz"),
            "protocol": digest(CONFIG),
        },
    )
    print(mode, seed, hist[win], flush=True)
    return model


def paired(a, b):
    d = a - b
    picks = np.random.default_rng(840).integers(len(d), size=(2000, len(d)))
    return {
        "mean_mm": d.mean(0).tolist(),
        "ci95_mm": np.quantile(d[picks].mean(1), [0.025, 0.975], axis=0).T.tolist(),
    }


def gate(stats):
    return any(all(ci[1] < 0 for ci in x["ci95_mm"]) for x in stats.values()) and all(
        max(x["mean_mm"]) < 1 for x in stats.values()
    )


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = json.loads(CONFIG.read_text())
    _, train_data = read("train")
    valid, v = read("validation")
    valid = (valid["rows"], v)
    models = {"continuous": None, "tick": Impact("tick")}
    for mode in ("joint", "direct"):
        for seed in cfg["training"]["seeds"]:
            models[f"{mode}-{seed}"] = train(mode, seed, train_data, valid)
    val = {k: score(branch_metrics(*valid, m, 1920)["rows"]) for k, m in models.items()}
    best = min(("continuous", "tick"), key=val.get)
    representatives = {
        k: sorted([f"{k}-{i}" for i in cfg["training"]["seeds"]], key=val.get)[1]
        for k in ("joint", "direct")
    }
    write(
        ROOT / "selection-lock.json",
        {"validation_mm": val, "analytic": best, "representatives": representatives},
    )
    meta, data = read("evaluation")
    comparisons = {}
    for hz in RATES:
        groups = {}
        for name, m in models.items():
            record = branch_metrics(meta["rows"], data, m, hz)
            groups[name] = group(record["rows"])
            write(ROOT / f"release-{name}-{hz}.json", record)
        out = {}
        for name in ["continuous", "tick", "joint", "direct"]:
            keys = (
                [name]
                if name in ("continuous", "tick")
                else [f"{name}-{i}" for i in cfg["training"]["seeds"]]
            )
            values = {fam: np.mean([groups[k][fam] for k in keys], axis=0) for fam in groups[best]}
            out[name] = {
                "mean_mm": {fam: v.mean(0).tolist() for fam, v in values.items()},
                "p95_parent_mm": {
                    fam: np.quantile(v, 0.95, axis=0).tolist() for fam, v in values.items()
                },
            }
            for ref in ("continuous", best):
                out[name][f"vs_{ref}"] = {fam: paired(v, groups[ref][fam]) for fam, v in values.items()}
            out[name]["gate"] = gate(out[name]["vs_continuous"]) and (
                name == "tick" or gate(out[name][f"vs_{best}"])
            )
        comparisons[str(hz)] = out
    write(
        ROOT / "results.json",
        {
            "protocol": digest(CONFIG),
            "groups": comparisons,
            "analytic": best,
            "representatives": representatives,
        },
    )
    print("fixed forecast gates", {k: v["gate"] for k, v in comparisons["1920"].items()}, flush=True)


if __name__ == "__main__":
    run()
