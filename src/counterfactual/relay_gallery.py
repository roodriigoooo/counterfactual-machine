"""The first elevated parent/edit with at least 50 mm of premature receiver travel."""

import json

import numpy as np

from .continuation_experiment import write
from .impact_data import ROOT
from .relay_forecast import forecast, layout
from .relay_world import release


def run():
    rows = json.loads((ROOT / "relay-elevated.json").read_text())["rows"]
    for row in rows:
        known = row["modes"]["known"]["rates"]["1920"]
        if known["prefix_before_true_transfer_mm"] < 50:
            continue
        world = layout(row["parent"], True)
        grid = row["grid_m"]
        path, times, _, _ = forecast(world, grid, np.full(len(grid), world.receiver.mass))
        disabled = forecast(world, grid, np.full(len(grid), world.receiver.mass), pair_enabled=False)[0]
        for i, u in enumerate(grid):
            xy, event, _ = release(world, u, 1920)
            clock = event.get("transfer")
            prefix = np.arange(121) / 60 < (clock / 1920 if clock else 3.0)
            if np.linalg.norm(path[i, prefix, 1] - disabled[i, prefix, 1], axis=-1).max() < 0.05:
                continue
            fine, fine_events, _ = release(world, u, 3840)
            write(
                ROOT / "relay-example.json",
                {
                    "parent": row["parent"],
                    "edit_m": u,
                    "direction": world.direction,
                    "receiver": world.receiver.model_dump(mode="json"),
                    "scene": world.scene.model_dump(mode="json"),
                    "times_s": {
                        "forecast": times[i] / 480 if times[i] >= 0 else None,
                        "1920": clock / 1920 if clock else None,
                        "3840": fine_events["transfer"] / 3840 if "transfer" in fine_events else None,
                    },
                    "paths": {
                        "forecast": path[i].tolist(),
                        "disabled": disabled[i].tolist(),
                        "1920": xy[::32].tolist(),
                        "3840": fine[::64].tolist(),
                    },
                },
            )
            return
    raise ValueError("No trajectory met the gallery criterion")


if __name__ == "__main__":
    run()
