"""Paired first-encounter windows and two-rate labels for the continuation study."""

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .architecture_data import digest
from .competition_scenes import propose
from .contact_events import event_labels
from .geometry import apply_edit, segment_distance
from .recurrent import DT, pack
from .scenes import generate_scene
from .schemas import EditSpec
from .simulator import simulate

PROTOCOL = Path("configs/contact-continuation.json")
DATA = Path("data/contact-continuation")
REPORT = Path("artifacts/contact-continuation")


def window_label(tr, scene, tick):
    hz = tr.provenance["physics_hz"]
    events = sorted(
        (round(e["time"] * hz), e["objects"][1])
        for e in tr.events
        if e["phase"] == "begin" and tick < e["time"] * 60 - 1e-9 <= tick + 2
    )
    if not events:
        return {"events": [], "order": [], "tie": False, "gap_m": None}
    first_step, first = events[0]
    times = [t for t, _ in events]
    other = "ramp" if first == "bumper" else "bumper"
    xy = tr.physics_positions[first_step : (tick + 2) * (hz // 60) + 1]
    if other == "bumper":
        gap = np.linalg.norm(xy - scene.bumper.position, axis=1) - scene.bumper.radius
    else:
        ramp = next(s for s in scene.segments if s.id == "ramp")
        gap = segment_distance(xy, ramp.a, ramp.b) - ramp.radius
    return {
        "events": [[s, (t / hz - tick * DT) * 1000] for t, s in events],
        "order": list(dict.fromkeys(s for _, s in events)),
        "tie": len(times) != len(set(times)),
        "gap_m": float(gap.min() - scene.ball.radius),
    }


def parent(scene, ordinary):
    seed = scene.seed
    # Same scheduled interval as the authored construction, available before release.
    tick = 0 if ordinary else int(np.random.default_rng(seed).integers(7, 22))
    branches, arrays = [], []
    for edit in [0.0] if ordinary else [0.0, -0.015, 0.015]:
        try:
            geo = apply_edit(scene, EditSpec(displacement=edit))
        except ValueError:
            continue
        tr = simulate(scene, edit)
        labels = event_labels(tr)
        audit = {}
        if not ordinary:
            fine = simulate(scene, edit, 3840)
            if any(x.contacts[:tick].any() for x in (tr, fine)):
                continue
            if edit == 0.0:
                prefix = [
                    x.physics_positions[: tick * (x.provenance["physics_hz"] // 60) + 1] for x in (tr, fine)
                ]
            elif not branches or not all(
                np.array_equal(p, x.physics_positions[: len(p)]) for p, x in zip(prefix, (tr, fine))
            ):
                continue
            audit = {str(x.provenance["physics_hz"]): window_label(x, geo, tick) for x in (tr, fine)}
        branches.append({"edit": edit, "audit": audit, "callbacks": labels["ordered_events"]})
        arrays.append((tr.states, pack(geo)[1], labels["event_labels"][..., 1], labels["active_start"]))
    if not ordinary and (len(branches) < 2 or branches[0]["edit"] != 0.0):
        return None
    return {"scene": scene.model_dump(mode="json"), "tick": tick, "branches": branches}, arrays


def exposure(records):
    counts = Counter()
    for parent in records:
        flags = set()
        original = None
        for b in parent["branches"]:
            if not b["audit"]:
                continue
            a, f = (b["audit"][k] for k in ("1920", "3840"))
            stable = a["order"] == f["order"] and not (a["tie"] or f["tie"])
            double = len(a["order"]) >= 2
            if a["tie"] or f["tie"]:
                flags.add("tie")
            if len(a["events"]) > 2:
                flags.add("overflow")
            if not stable:
                flags.add("sensitive")
            elif double:
                flags.add(" then ".join(a["order"][:2]))
            elif len(a["order"]) == 1 and all(
                x["gap_m"] is not None and 0 < x["gap_m"] <= 0.010 for x in (a, f)
            ):
                flags.add("near miss after " + a["order"][0])
            if b["edit"] == 0:
                original = (stable, double)
            elif stable and original[0]:
                flags.add("unchanged" if original[1] == double else "created" if double else "removed")
        counts.update(flags)
    return dict(sorted(counts.items()))


def build(split, start, count, limit, root, binding):
    target = root / split
    if target.exists():
        m = json.loads((target / "manifest.json").read_text())
        if m["protocol"] != binding or digest(target / "branches.npz") != m["archive"]:
            raise ValueError(f"Changed inputs in {target}")
        return m
    records, arrays, rejected = [], [], Counter()
    for seed in range(start, start + limit):
        try:
            scene = generate_scene(seed) if split.startswith("ordinary") else propose(seed)
        except ValueError:
            if split.startswith("ordinary"):
                raise
            rejected["invalid_geometry"] += 1
            continue
        item = parent(scene, split.startswith("ordinary"))
        if item is None:
            rejected["no_aligned_pair"] += 1
            continue
        record, rows = item
        for b, row in zip(record["branches"], rows, strict=True):
            b["row"] = len(arrays)
            arrays.append(row)
        records.append(record)
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError(f"Pool exhausted in {split}")
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as tmp:
        folder = Path(tmp) / split
        folder.mkdir()
        archive = folder / "branches.npz"
        np.savez_compressed(
            archive,
            **{
                k: np.asarray(v, np.float32)
                for k, v in zip(["states", "contexts", "begin", "active_start"], zip(*arrays))
            },
        )
        manifest = {
            "protocol": binding,
            "archive": digest(archive),
            "split": split,
            "rejected": dict(rejected),
            "exposure": exposure(records),
            "records": records,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest))
        folder.rename(target)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--root", type=Path, default=DATA)
    args = parser.parse_args()
    splits = {"pilot": [70000, 48, 400]} if args.pilot else json.loads(PROTOCOL.read_text())["splits"]
    for name, spec in splits.items():
        m = build(name, *spec, args.root, "construction-pilot" if args.pilot else digest(PROTOCOL))
        print(name, len(m["records"]), m["exposure"], flush=True)
