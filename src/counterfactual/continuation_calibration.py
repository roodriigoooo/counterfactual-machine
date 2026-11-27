"""Natural-prevalence and enriched calibration controls for the frozen E2 readout."""

import numpy as np
import torch
from torch.nn import functional as F

from .architecture import conditions, integrate
from .architecture_data import digest
from .competition_scenes import REPORT as FROZEN
from .continuation_data import PROTOCOL, REPORT
from .continuation_experiment import read_split, write
from .episode_model import EpisodeModel


@torch.inference_mode()
def observations(model, rows, d, ordinary):
    parts, targets = [], []
    c = d["contexts"]
    stages = range(120) if ordinary else range(max(r["tick"] for r in rows) + 1)
    s = d["states"][:, 0]
    active = torch.zeros_like(c[..., 9])
    coefficients = torch.stack([c[..., 8], torch.ones_like(active)], -1)
    order = []
    for tick in stages:
        mask = c[..., 9].bool().clone()
        if not ordinary:
            mask.zero_()
            for i, r in enumerate(rows):
                if tick == r["tick"]:
                    mask[i, r["ids"].index("ramp")] = True
                    order.append(i)
        if mask.any():
            parts.append(model.readout(s, c, active)[0][..., 1][mask])
            targets.append(d["begin"][:, tick][mask])
        _, geo = conditions(s, c)
        s, _ = integrate(s, c, geo, coefficients)
        active = geo[1].float()
    logits, truth = torch.cat(parts), torch.cat(targets)
    if not ordinary:
        perm = np.argsort(order)
        logits, truth = logits[perm], truth[perm]
    return logits, truth


def affine(logits, truth):
    raw = torch.tensor([0.0, 0.0], requires_grad=True)
    opt = torch.optim.LBFGS([raw], max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        pred = raw[0].exp() * logits + raw[1]
        loss = F.binary_cross_entropy_with_logits(pred, truth) + 1e-6 * raw.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(raw[0].detach().exp()), float(raw[1].detach())


def scores(logits, y, rows, ordinary):
    p = logits.sigmoid()
    out = {
        "positive": int(y.sum()),
        "n": len(y),
        "bce": float(F.binary_cross_entropy_with_logits(logits, y)),
        "brier": float((p - y).square().mean()),
    }
    if not ordinary:
        pairs = []
        for i, r in enumerate(rows):
            if r["edit"] == 0:
                base = i
            else:
                pairs.append(
                    {
                        "parent": r["parent"],
                        "edit": r["edit"],
                        "flip": bool(y[base] != y[i]),
                        "predicted_flip": bool((p[base] >= 0.5) != (p[i] >= 0.5)),
                        "correct": bool(((p[[base, i]] >= 0.5) == y[[base, i]].bool()).all()),
                    }
                )
        out["pairs"] = pairs
    return out


def run():
    torch.set_num_threads(1)
    results = {}
    for domain in ("enriched", "ordinary"):
        ordinary = domain == "ordinary"
        pre = "ordinary_" if ordinary else ""
        cr, cd, _ = read_split(pre + "calibration")
        er, ed, _ = read_split(pre + "evaluation")
        runs = []
        for seed in range(3):
            cp = FROZEN / "E2" / f"seed-{seed}" / "model.pt"
            model = EpisodeModel("E2")
            model.load_state_dict(torch.load(cp, weights_only=True)["model"])
            model.eval()
            logits, y = observations(model, cr, cd, ordinary)
            slope, offset = affine(logits.clone(), y.clone())
            logits, y = observations(model, er, ed, ordinary)
            runs.append(
                {
                    "seed": seed,
                    "checkpoint": digest(cp),
                    "slope": slope,
                    "intercept": offset,
                    "raw": scores(logits, y, er, ordinary),
                    "calibrated": scores(logits * slope + offset, y, er, ordinary),
                }
            )
        results[domain] = runs
    write(REPORT / "e2-calibration.json", {"protocol": digest(PROTOCOL), "domains": results})


if __name__ == "__main__":
    run()
