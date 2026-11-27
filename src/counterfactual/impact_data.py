"""Fresh, parent-disjoint paired two-rate releases for the impact-decoder screen."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .architecture_data import digest
from .competition_scenes import propose
from .continuation_experiment import write
from .geometry import apply_edit
from .recurrent import pack
from .scenes import generate_scene
from .schemas import EditSpec
from .simulator import simulate

CONFIG = Path("configs/impact-decoder.json")
ROOT = Path("artifacts/impact-decoder")
DATA = Path("data/impact-decoder")
RATES = (1920, 3840)


def save_weights(path, model):
    """Publish a complete new checkpoint, or verify the deterministic rerun."""
    weights = model.state_dict()
    with TemporaryDirectory(dir=path.parent) as staging:
        fresh = Path(staging) / path.name
        torch.save(weights, fresh)
        try:
            os.link(fresh, path)
        except FileExistsError:
            old = torch.load(path, weights_only=True)
            if old.keys() != weights.keys() or any(
                not torch.equal(old[k], v) for k, v in weights.items()
            ):
                raise ValueError(f"Rerun differs from frozen checkpoint {path}")


def scenes(start, count, family):
    accepted = 0
    for seed in range(start, start + 1000):
        try:
            scene = (propose if family == "competing" else generate_scene)(seed)
        except ValueError:
            continue
        yield scene
        accepted += 1
        if accepted == count:
            return
    raise ValueError("Parent range exhausted")


def build(split):
    start, comp, ordinary = json.loads(CONFIG.read_text())["splits"][split]
    folder = DATA / split
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "branches.npz"
    if (folder / "manifest.json").exists():
        return read(split)[0]
    rows, contexts, states, contact = [], [], [], []
    for family, offset, n in [("competing", 0, comp), ("ordinary", 1000, ordinary)]:
        for scene in scenes(start + offset, n, family):
            for edit in (0.0, -0.015, 0.015):
                try:
                    geo = apply_edit(scene, EditSpec(displacement=edit))
                except ValueError:
                    continue
                refs = [simulate(scene, edit, hz) for hz in RATES]
                ctx = pack(geo)[1]
                occupied = np.zeros((2, 120, 5), bool)
                for j, tr in enumerate(refs):
                    occupied[j, :, : len(tr.object_ids)] = tr.contacts
                first_bumper = [
                    min(
                        (
                            e["time"]
                            for e in tr.events
                            if e["phase"] == "begin" and e["objects"][1] == "bumper"
                        ),
                        default=3.0,
                    )
                    for tr in refs
                ]
                rows.append(
                    {
                        "parent": scene.seed,
                        "family": family,
                        "edit": edit,
                        "scene_hash": scene.content_hash(),
                        "first_bumper_s": first_bumper,
                    }
                )
                contexts.append(ctx)
                states.append([tr.states for tr in refs])
                contact.append(occupied)
    np.savez_compressed(
        archive,
        contexts=np.array(contexts),
        states=np.array(states, np.float32),
        contact=np.array(contact),
    )
    record = {"protocol": digest(CONFIG), "archive": digest(archive), "rows": rows}
    write(folder / "manifest.json", record)
    return record


def read(split):
    p = DATA / split
    meta = json.loads((p / "manifest.json").read_text())
    if meta["protocol"] != digest(CONFIG) or meta["archive"] != digest(p / "branches.npz"):
        raise ValueError("Impact data binding mismatch")
    with np.load(p / "branches.npz") as f:
        return meta, {k: f[k] for k in f.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation", "evaluation"])
    args = parser.parse_args()
    for k in [args.split] if args.split else ["train", "validation", "evaluation"]:
        print(k, len(build(k)["rows"]), flush=True)
