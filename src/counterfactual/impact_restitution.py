"""Can the reference engine's order of gravity and restitution explain the learned gain?"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .architecture_data import digest
from .continuation_experiment import write
from .geometry import apply_edit
from .impact_data import RATES, ROOT, scenes
from .impact_decisions import decide, models
from .impact_experiment import branch_metrics, group, paired
from .impact_model import Impact
from .recurrent import pack
from .schemas import EditSpec
from .simulator import simulate

PROTOCOL = Path("configs/impact-restitution-audit.json")


def run():
    torch.set_num_threads(1)
    cfg = json.loads(PROTOCOL.read_text())
    mdl = models()
    mdl["ordered"] = Impact("ordered")
    rows = []
    contexts = []
    states = []
    for fam in ("competing", "ordinary"):
        for scene in scenes(*cfg[fam], fam):
            for u in cfg["edits"]:
                try:
                    geo = apply_edit(scene, EditSpec(displacement=u))
                except ValueError:
                    continue
                refs = [simulate(scene, u, hz) for hz in RATES]
                states.append([tr.states for tr in refs])
                contexts.append(pack(geo)[1])
                rows.append(
                    {
                        "parent": scene.seed,
                        "family": fam,
                        "edit": u,
                        "first_bumper_s": [
                            min(
                                (
                                    e["time"]
                                    for e in tr.events
                                    if e["phase"] == "begin" and e["objects"][1] == "bumper"
                                ),
                                default=3.0,
                            )
                            for tr in refs
                        ],
                    }
                )
    data = {"states": np.array(states, np.float32), "contexts": np.array(contexts)}
    record = {}
    for hz in RATES:
        result = {k: branch_metrics(rows, data, m, hz)["rows"] for k, m in mdl.items()}
        ref = group(result["ordered"])
        record[str(hz)] = {
            "branches": result,
            "versus_ordered": {
                name: {f: paired(v, ref[f]) for f, v in group(a).items()} for name, a in result.items()
            },
        }
    write(ROOT / "restitution.json", {"protocol": digest(PROTOCOL), "rates": record})


def decisions():
    torch.set_num_threads(1)
    gate = json.loads((ROOT / "restitution.json").read_text())["rates"]
    if not all(
        gate[str(hz)]["versus_ordered"][k][fam]["ci95_mm"][i][0] > 0
        for i, hz in enumerate(RATES)
        for k in ("joint", "tick")
        for fam in ("competing", "ordinary")
    ):
        return
    control = Path("configs/impact-ordered-decisions.json")
    cfg = json.loads(control.read_text())
    fitted = models()
    arms = {"tick": fitted["tick"], "joint": fitted["joint"], "ordered": Impact("ordered")}
    for fam in ("competing", "ordinary"):
        rows = [decide(scene, arms, cfg) for scene in scenes(*cfg[fam], fam)]
        write(ROOT / f"ordered-decisions-{fam}.json", {"protocol": digest(control), "parents": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", action="store_true")
    if parser.parse_args().decisions:
        decisions()
    else:
        run()
