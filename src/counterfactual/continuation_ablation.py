"""Exploratory matched readout with the impulse omitted from its features."""

import json

import numpy as np
import torch

from .architecture_data import digest
from .continuation_data import DATA, PROTOCOL, REPORT
from .continuation_experiment import (
    ConditionalHead,
    fit,
    learned,
    metrics,
    read_split,
    record,
    write,
)
from .continuation_solver import query, rollout


def run():
    torch.set_num_threads(1)
    cfg = json.loads(PROTOCOL.read_text())["training"]
    train, d, _ = read_split("train")
    rows, test, _ = read_split("evaluation")
    prior = json.loads((REPORT / "experiment.json").read_text())
    out = prior["modes"]["release"]["records"]
    c = test["contexts"]
    states, _, _ = rollout(test["states"][:, 0], c)
    times, x, mask = query(states[torch.arange(len(rows)), [r["tick"] for r in rows]], c, incoming_only=True)
    binding = {"protocol": digest(PROTOCOL), "training": digest(DATA / "train/manifest.json")}
    for seed in cfg["seeds"]:
        folder = REPORT / "incoming-control"
        folder.mkdir(exist_ok=True)
        cp = folder / f"seed-{seed}.pt"
        if cp.exists():
            saved = torch.load(cp, weights_only=True)
            if saved["binding"] != binding or saved["seed"] != seed:
                raise ValueError("Incoming control binding changed")
            model = ConditionalHead()
            model.load_state_dict(saved["model"])
        else:
            model, n = fit(train, d, cfg, seed, incoming_only=True)
            temp = cp.with_suffix(".tmp")
            torch.save(
                {"model": model.state_dict(), "binding": binding, "seed": seed, "training_rows": n}, temp
            )
            temp.replace(cp)
        pred, _ = learned(model.eval(), times, x, mask)
        for i, r in enumerate(out):
            r[f"incoming-{seed}"] = record(pred, i)
    primary = [r for r in out if r["gold"]["stratum"] == "primary"]
    differences = {}
    for r in primary:
        y = r["gold"]["second"]
        diff = np.mean(
            [
                int(r[f"head-{s}"]["second"] == y) - int(r[f"incoming-{s}"]["second"] == y)
                for s in cfg["seeds"]
            ]
        )
        differences.setdefault(r["parent"], []).append(diff)
    v = np.array([np.mean(x) for x in differences.values()])
    boots = np.random.default_rng(948).choice(v, (2000, len(v))).mean(1)
    write(
        REPORT / "ablation.json",
        {
            "binding": binding,
            "records": out,
            "primary": {f"incoming-{s}": metrics(primary, f"incoming-{s}") for s in cfg["seeds"]},
            "outgoing_minus_incoming": float(v.mean()),
            "ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
        },
    )


if __name__ == "__main__":
    run()
