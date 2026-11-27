"""Alternate solver order on complete single-goal plans with a known original."""

import json
from time import perf_counter

import numpy as np
import torch

from .architecture_evaluation import forecast_table
from .competition_scenes import propose
from .continuation_data import REPORT
from .continuation_experiment import write
from .continuation_selection import GridForecast
from .geometry import candidates
from .goals import goal_result
from .scenes import generate_goal, generate_scene
from .simulator import simulate
from .tolerance_experiment import choose


def run():
    torch.set_num_threads(1)
    records = []
    for cohort, root, make in [
        ("ordinary", REPORT, generate_scene),
        ("enriched", REPORT / "enriched-risk", propose),
    ]:
        files = sorted((root / "selection/evaluation").glob("*.json"))[:16]
        for number, path in enumerate(files):
            seed = json.loads(path.read_text())["seed"]
            scene = make(seed)
            original = simulate(scene)
            for j in range(8):
                g = generate_goal(seed * 8 + j)
                base = goal_result(original.physics_times, original.physics_positions, g)
                if not base["success"]:
                    break
            if base["success"]:
                continue
            order = ["bounded", "four"] if number % 2 == 0 else ["four", "bounded"]
            for mode in order:
                predictor = GridForecast(mode, {})
                times = []
                indices = []
                for _ in range(2):
                    start = perf_counter()
                    grid, valid, _ = candidates(scene, 257)
                    table = forecast_table(predictor, scene, grid, valid, [g])
                    indices.append(choose(scene, grid, valid, table, 0, base, "tolerant", 0.03, 0.005))
                    times.append(perf_counter() - start)
                if indices[0] != indices[1]:
                    raise ValueError("Cached planning changed the selected edit")
                records.append(
                    {
                        "seed": seed,
                        "cohort": cohort,
                        "mode": mode,
                        "goal": j,
                        "valid": int(valid.sum()),
                        "selected": indices[0],
                        "cold_s": times[0],
                        "cached_s": times[1],
                    }
                )
    write(REPORT / "latency.json", records)
    for cohort in ("ordinary", "enriched"):
        for mode in ("bounded", "four"):
            rows = [r for r in records if r["cohort"] == cohort and r["mode"] == mode]
            print(
                cohort,
                mode,
                len(rows),
                [np.quantile([r[k] for r in rows], [0.5, 0.95]).tolist() for k in ("cold_s", "cached_s")],
            )


if __name__ == "__main__":
    run()
