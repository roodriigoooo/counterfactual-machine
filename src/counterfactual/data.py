import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from .geometry import candidates, apply_edit
from .schemas import EditSpec
from .scenes import generate_scene
from .simulator import simulate

MAX_STATIC = 5


def features(scene):
    """IDs select objects; they are never numeric inputs. Pooling is order invariant."""
    b = scene.ball
    global_features = np.array([b.position[0]/6, b.position[1]/4, b.velocity[0]/5,
                               b.velocity[1]/5, b.radius/.12, b.angle/10, b.angular_velocity/20,
                               b.material.friction, b.material.restitution, b.mass], dtype=np.float32)
    objects = sorted([scene.bumper, *scene.segments], key=lambda o: o.id)
    relations = np.zeros((MAX_STATIC, 10), dtype=np.float32)
    mask = np.zeros(MAX_STATIC, dtype=np.float32)
    for i, o in enumerate(objects):
        circle = o.id == scene.bumper.id
        a, c = (o.position, o.position) if circle else (o.a, o.b)
        mid = (np.asarray(a)+c)/2
        relations[i] = [float(circle), (mid[0]-b.position[0])/6, (mid[1]-b.position[1])/4,
                        (a[0]-b.position[0])/6, (a[1]-b.position[1])/4,
                        (c[0]-b.position[0])/6, (c[1]-b.position[1])/4,
                        o.radius, o.material.friction, o.material.restitution]
        mask[i] = 1
    return global_features, relations, mask


def generate_dataset(output: Path, parents=256):
    if parents < 16:
        raise ValueError("need at least 16 parents for disjoint train/validation/test splits")
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    permutation = np.random.default_rng(20261108).permutation(parents)
    split = np.full(parents, 2, dtype=np.int8)
    split[permutation[:int(parents*.75)]] = 0
    split[permutation[int(parents*.75):int(parents*.875)]] = 1
    arrays = {k: [] for k in ["global_features", "relations", "mask", "states", "contacts", "edits"]}
    records = []
    for seed in range(parents):
        scene = generate_scene(seed)
        grid, valid, _ = candidates(scene, 9)
        # Retain 9 paired slots; invalid endpoints use the valid zero branch, explicitly recorded.
        edits = np.where(valid, grid, 0)
        row = {k: [] for k in arrays}
        for u in edits:
            tr = simulate(scene, float(u))
            edited = apply_edit(scene, EditSpec(object_id=scene.bumper.id, displacement=float(u)))
            g, r, m = features(edited)
            c = np.zeros((120, MAX_STATIC), dtype=np.float32)
            c[:, :len(tr.object_ids)] = tr.contacts
            for key, value in zip(row, [g, r, m, tr.states, c, u]):
                row[key].append(value)
        for key in arrays:
            arrays[key].append(row[key])
        records.append({"scene": scene.model_dump(), "scene_hash": scene.content_hash(),
                        "split": ["train", "validation", "test"][split[seed]],
                        "edits_m": edits.tolist(), "requested_valid": valid.tolist()})
    packed = {k: np.array(v, dtype=np.float32) for k, v in arrays.items()}
    packed["split"] = split
    np.savez_compressed(output/"paired.npz", **packed)
    digest = hashlib.sha256((output/"paired.npz").read_bytes()).hexdigest()
    manifest = {"generator": "pilot-v1", "schema_version": 1, "parents": parents,
                "branches": parents*9, "split_parents": {name: int(np.sum(split == i)) for i, name in enumerate(["train", "validation", "test"])},
                "split_seed": 20261108, "sha256": digest, "physics": tr.provenance,
                "normalization": "fixed SI scales only; no fitted statistics", "seconds": perf_counter()-started,
                "records": records}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return {k: v for k, v in manifest.items() if k != "records"}
