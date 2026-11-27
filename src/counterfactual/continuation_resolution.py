"""Two-rate path reference and exact-state overflow for the enriched evaluation."""

import torch

from .continuation_data import REPORT
from .continuation_experiment import read_split, write
from .continuation_solver import bounded, rollout
from .schemas import SceneSpec
from .simulator import simulate


def run():
    torch.set_num_threads(1)
    rows, d, m = read_split("evaluation")
    scenes = {p["scene"]["seed"]: SceneSpec.model_validate(p["scene"]) for p in m["records"]}
    s, c = d["states"], d["contexts"]
    release, _, excess = rollout(s[:, 0], c, audit=True)
    # Teacher-forced check separates overflow on true geometry from accumulated drift.
    exact = torch.stack([bounded(s[:, t], c, audit_overflow=True)[3] for t in range(120)], 1)
    records = []
    for i, r in enumerate(rows):
        fine = simulate(scenes[r["parent"]], r["edit"], 3840)
        xf = torch.tensor(fine.states[:, :2], dtype=s.dtype)
        truth = d["begin"][i]
        n = torch.zeros(120, dtype=torch.int)
        for event in r["callbacks"]:
            if event["phase"] == "begin":
                n[event["interval"]] += 1

        def error(a, b):
            return float((a - b).square().sum(-1).mean().sqrt())

        records.append(
            {
                "parent": r["parent"],
                "edit": r["edit"],
                "solver_coarse_m": error(release[i, :, :2], s[i, :, :2]),
                "solver_fine_m": error(release[i, :, :2], xf),
                "coarse_fine_m": error(s[i, :, :2], xf),
                "exact_overflow": int(exact[i].sum()),
                "release_overflow": int(excess[i].sum()),
                "true_over_two": int((n > 2).sum()),
                "exact_overflow_at_true_over_two": int((exact[i] & (n > 2)).sum()),
                "true_multi_surface": int((truth.sum(-1) > 1).sum()),
            }
        )
    write(REPORT / "resolution.json", records)


if __name__ == "__main__":
    run()
