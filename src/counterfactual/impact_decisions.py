"""Independent confirmation of rate-conditioned path gains, then paired-rate edit selection."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .architecture_data import digest
from .architecture_evaluation import forecast_table
from .continuation_experiment import write
from .geometry import apply_edit, candidates
from .goals import goal_result
from .impact_data import RATES, ROOT, scenes
from .impact_experiment import paired
from .impact_model import Impact, forecast
from .recurrent import pack
from .scenes import generate_goal
from .schemas import EditSpec
from .simulator import simulate
from .tolerance import select_with_tolerance

PROTOCOL = Path("configs/impact-two-rate-selection.json")


def models():
    lock = json.loads((ROOT / "selection-lock.json").read_text())
    result = {"continuous": None, "tick": Impact("tick")}
    for name, rep in lock["representatives"].items():
        m = Impact(name)
        m.load_state_dict(torch.load(ROOT / rep / "model.pt", weights_only=True))
        result[name] = m.eval()
    return result


def confirm(cfg):
    result = {}
    for family in ("competing", "ordinary"):
        states = []
        contexts = []
        refs = []
        parents = []
        for scene in scenes(*cfg[family], family):
            for edit in cfg["edit_m"]:
                try:
                    s, c = pack(apply_edit(scene, EditSpec(displacement=edit)))
                except ValueError:
                    continue
                states.append(s)
                contexts.append(c)
                parents.append(scene.seed)
                refs.append([simulate(scene, edit, hz).states[:, :2] for hz in RATES])
        s, c = [torch.tensor(np.stack(x)) for x in (states, contexts)]
        rows = []
        for j, hz in enumerate(RATES):
            per = []
            for m in (None, Impact("tick")):
                error = forecast(s, c, m, hz).numpy()[:, :, :2] - np.array(refs)[:, j]
                branch = 1000 * np.sqrt((error**2).sum(-1).mean(-1))
                per.append(
                    np.array([branch[np.array(parents) == p].mean() for p in sorted(set(parents))])
                )
            rows.append(
                {
                    "hz": hz,
                    "continuous_mm": per[0].tolist(),
                    "tick_mm": per[1].tolist(),
                    "paired": paired(per[1][:, None], per[0][:, None]),
                }
            )
        result[family] = {"parents": sorted(set(parents)), "results": rows}
    return result


class PackedForecast:
    def __init__(self, s, c, model, hz):
        self.s, self.c, self.model, self.hz = s, c, model, hz

    def forecast(self, scene, edits):
        return forecast(self.s, self.c, self.model, self.hz).numpy(), None


def decide(scene, mdl, cfg):
    grid, valid, _ = candidates(scene, cfg["grid"])
    goals = [generate_goal(8 * scene.seed + j) for j in range(cfg["goals"])]
    s, c = [
        torch.from_numpy(np.stack(x))
        for x in zip(*(pack(apply_edit(scene, EditSpec(displacement=float(e)))) for e in grid[valid]))
    ]
    actual = {rate: {0.0: simulate(scene, 0.0, rate)} for rate in RATES}
    original = {
        rate: [goal_result(tr[0.0].physics_times, tr[0.0].physics_positions, g) for g in goals]
        for rate, tr in actual.items()
    }
    rows = []
    timings = {}
    for name, m in mdl.items():
        start = perf_counter()
        tables = [
            forecast_table(PackedForecast(s, c, m, rate), scene, grid, valid, goals) for rate in RATES
        ]
        timings[name] = perf_counter() - start
        for rule in ["fixed", "both"] if name == "continuous" else ["both"]:
            tables_used = tables if rule == "both" else tables[:1]
            for j, g in enumerate(goals):
                if original[1920][j]["success"]:
                    continue
                # Only the known coarse original is allowed to bind the zero cell.
                margins = np.minimum.reduce([t["margins"][j] for t in tables_used])
                ok = np.logical_and.reduce([t["passed"][j] for t in tables_used])
                z = len(grid) // 2
                margins[z] = original[1920][j]["margin"]
                ok[z] = False
                idx = select_with_tolerance(grid, margins, ok, valid, cfg["epsilon_m"], cfg["gamma_m"])[
                    "index"
                ]
                checks = {}
                if idx is not None:
                    for rate in RATES:
                        passed = []
                        for offset in [-0.005, -0.0025, 0, 0.0025, 0.005]:
                            delta = round(float(grid[idx]) + offset, 12)
                            if delta not in actual[rate]:
                                actual[rate][delta] = simulate(scene, delta, rate)
                            tr = actual[rate][delta]
                            passed.append(
                                bool(goal_result(tr.physics_times, tr.physics_positions, g)["success"])
                            )
                        checks[str(rate)] = passed
                rows.append(
                    {
                        "goal": j,
                        "mode": name + "-" + rule,
                        "edit_m": None if idx is None else float(grid[idx]),
                        "checks": checks,
                        "fine_original_pass": bool(original[3840][j]["success"]),
                    }
                )
    return {
        "parent": scene.seed,
        "rows": rows,
        "paired_table_seconds": timings,
        "unique_simulations": {str(hz): len(v) for hz, v in actual.items()},
    }


def run():
    torch.set_num_threads(1)
    cfg = json.loads(PROTOCOL.read_text())
    record = confirm(cfg["confirmation"])
    passed = all(x["paired"]["ci95_mm"][0][1] < 0 for v in record.values() for x in v["results"])
    write(
        ROOT / "two-rate-confirmation.json",
        {"protocol": digest(PROTOCOL), "pass": passed, "families": record},
    )
    print("Confirmation", passed, flush=True)
    if not passed:
        return
    mdl = models()
    cfg = cfg["decisions"]
    for family in ("competing", "ordinary"):
        rows = []
        for scene in scenes(*cfg[family], family):
            rows.append(decide(scene, mdl, cfg))
            if len(rows) % 8 == 0:
                print(family, len(rows), flush=True)
        write(ROOT / f"decisions-{family}.json", {"protocol": digest(PROTOCOL), "parents": rows})


if __name__ == "__main__":
    run()
