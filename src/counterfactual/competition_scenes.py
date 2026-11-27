"""Generate and verify single-interval contact preemption in the one-ball world."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .architecture import conditions
from .architecture_data import digest
from .contact_events import event_labels
from .geometry import apply_edit, validate_scene
from .recurrent import pack
from .scenes import generate_scene
from .schemas import Ball, Bumper, EditSpec, Material, SceneSpec, Segment
from .simulator import simulate

PROTOCOL = Path("configs/competition-scenes.json")
DATA = Path("data/competition-scenes")
REPORT = Path("artifacts/competition-scenes")


def propose(seed):
    """Place two rounded surfaces tangent to one free flight at different times."""
    rng = np.random.default_rng(seed)
    tick = int(rng.integers(7, 22))
    times = (tick + np.sort(rng.uniform(.12, .88, 2))) / 60
    origin = np.array([rng.uniform(2.5, 3.5), rng.uniform(2.8, 3.55)])
    velocity = rng.uniform([-1.8, -1.1], [1.8, .2])
    gravity = np.array([0, -9.81])
    # Pymunk advances position before velocity. This is its discrete free flight.
    points = [origin + velocity*t + .5*gravity*t*(t-1/1920) for t in times]
    incoming = velocity + gravity*times.mean()
    up = -incoming / np.linalg.norm(incoming)
    side = np.array([-up[1], up[0]]) * rng.choice([-1, 1])
    spread = rng.uniform(23, 45)
    tilt = rng.uniform(-29, 29)
    angles = np.deg2rad(rng.permutation([tilt-spread, tilt+spread]))
    normals = [up*np.cos(a) + side*np.sin(a) for a in angles]
    ball_r = rng.uniform(.10, .16)
    bumper_r = rng.uniform(.18, .34)
    ramp_r = rng.uniform(.025, .065)
    b = int(rng.integers(2))
    center = points[b] - (bumper_r+ball_r)*normals[b]
    end = points[1-b] - (ramp_r+ball_r)*normals[1-b]
    # Extend away from both the incoming ball and the other collider.
    direction = end-center
    direction /= np.linalg.norm(direction)
    if direction @ normals[1-b] > -.05:
        direction = -normals[1-b]
    other = end + rng.uniform(.35, .85)*direction
    mat = Material(restitution=rng.uniform(.68, .97), friction=rng.uniform(.15, .65))
    return validate_scene(SceneSpec(
        id=f"competition-{seed}", parent_id=f"competition-parent-{seed}", seed=seed,
        family="authored", ball=Ball(position=origin, velocity=velocity, radius=ball_r,
                                    angular_velocity=rng.uniform(-5, 5), material=mat),
        bumper=Bumper(position=center, radius=bumper_r, material=mat),
        segments=(Segment(id="ramp", a=end, b=other, radius=ramp_r, material=mat),
                  Segment(id="floor", a=(.1,.18), b=(5.9,.18)))))


def disable_surface(scene, winner):
    # The schema requires a bumper. Park it outside the pre-encounter path
    # and verify the control's entire history through interval start below.
    if winner == scene.bumper.id:
        return validate_scene(scene.model_copy(update={
            "bumper": scene.bumper.model_copy(update={"position": (.5, 3.48)})}))
    return scene.model_copy(update={"segments": tuple(s for s in scene.segments if s.id != winner)})


@torch.inference_mode()
def opportunities(scene, tr):
    _, context = pack(scene)
    states = torch.from_numpy(tr.states[:-1].astype(np.float32))
    _, geometry = conditions(states, torch.from_numpy(context).expand(len(states), -1, -1))
    eligible, _, times, _, _, gaps, *_ = geometry
    records = []
    for tick in torch.where(eligible.sum(-1) >= 2)[0].tolist():
        candidates = torch.where(eligible[tick])[0].tolist()
        records.append(dict(tick=tick, candidate_ids=[tr.object_ids[i] for i in candidates],
                            times_s=times[tick,candidates].tolist(),
                            gaps_m=gaps[tick,candidates].tolist()))
    return records


def begins(tr, tick):
    return [e for e in tr.events if e['phase']=='begin' and
            tick+1e-9 < e['time']*60 <= tick+1+1e-9]


def witness(scene, tr, races):
    for race in races:
        tick = race["tick"]
        if tr.contacts[:tick].any() or set(race["candidate_ids"]) != {"bumper", "ramp"}:
            continue
        actual = np.flatnonzero(tr.contacts[tick])
        if len(actual) != 1:
            continue
        winner = tr.object_ids[actual[0]]
        if winner not in race["candidate_ids"]:
            continue
        events = begins(tr, tick)
        if not any(winner in e["objects"] for e in events):
            continue
        loser = next(o for o in race["candidate_ids"] if o != winner)
        isolated = simulate(disable_surface(scene, winner), physics_hz=tr.provenance["physics_hz"])
        recovered = [e for e in begins(isolated, tick) if loser in e["objects"]]
        if not recovered or isolated.contacts[:tick].any() or not np.array_equal(
                tr.states[:tick+1], isolated.states[:tick+1]):
            continue
        gaps = race["gaps_m"]
        nearest = race["candidate_ids"][np.argmin(gaps)] if abs(gaps[0]-gaps[1]) > 1e-6 else None
        return dict(**race, winner=winner, omitted=loser, nearest=nearest,
                    begin_s=events[0]["time"], isolated_begin_s=recovered[0]["time"],
                    interval_start=tr.states[tick].tolist(),
                    observed_xy=tr.physics_positions[tick*(tr.provenance['physics_hz']//60):(tick+1)*(tr.provenance['physics_hz']//60)+1].tolist(),
                    isolated_xy=isolated.physics_positions[tick*(tr.provenance['physics_hz']//60):(tick+1)*(tr.provenance['physics_hz']//60)+1].tolist())
    return None


def same_encounter(coarse, fine):
    return bool(coarse and fine and all(coarse[k] == fine[k] for k in
                                        ("tick", "winner", "omitted", "candidate_ids")))


def collect_split(split, config, protocol_hash, folder):
    pool = config["splits"][split]
    enriched = split in config["gate"]
    records, rejected, arrays = [], Counter(), []
    for seed in range(pool["start"], pool["start"]+pool.get("limit", pool["parents"])):
        try:
            scene = propose(seed) if enriched else generate_scene(seed)
        except ValueError:
            if not enriched:
                raise
            rejected["invalid_geometry"] += 1
            continue
        original = simulate(scene)
        races = opportunities(scene, original)
        audits = {}
        if enriched:
            coarse = witness(scene, original, races)
            if not coarse:
                rejected["no_coarse_witness"] += 1
                continue
            fine_tr = simulate(scene, physics_hz=3840)
            fine = witness(scene, fine_tr, opportunities(scene, fine_tr))
            if not same_encounter(coarse, fine):
                rejected["rate_disagreement"] += 1
                continue
            audits = {"1920": coarse, "3840": fine}
        branches = []
        for edit in config["branches_m"]:
            try:
                geometry = apply_edit(scene, EditSpec(displacement=edit))
            except ValueError as err:
                branches.append(dict(edit_m=edit, valid=False, reason=str(err)))
                continue
            tr = original if edit == 0 else simulate(scene, edit)
            race_list = races if edit == 0 else opportunities(geometry, tr)
            labels = event_labels(tr)
            branches.append(dict(edit_m=edit, valid=True, row=len(arrays), opportunities=race_list,
                                 callbacks=labels['ordered_events']))
            arrays.append((tr.states, pack(geometry)[1], labels["event_labels"][...,1],
                           labels["active_start"]))
        records.append(dict(scene=scene.model_dump(mode="json"), branches=branches, audits=audits))
        if len(records) == pool["parents"]:
            break
    path = folder/"branches.npz"
    np.savez_compressed(path, **{name: np.array(rows, np.float32) for name, rows in
                                 zip(["states", "contexts", "begin", "active_start"], zip(*arrays))})
    winners = Counter(r["audits"]["1920"]["winner"] for r in records if r["audits"])
    nearest_loses = sum(all(w["nearest"] == w["omitted"] for w in r["audits"].values())
                        for r in records if r["audits"])
    valid_branches = [b for r in records for b in r['branches'] if b['valid']]
    stats = dict(parents=len(records), attempted=len(records)+sum(rejected.values()), rejected=dict(rejected),
                 witness_parents=sum(bool(r['audits']) for r in records), winner_parents=dict(winners),
                 nearest_preempted=nearest_loses, valid_branches=len(valid_branches),
                 competing_intervals=sum(len(b['opportunities']) for b in valid_branches),
                 competing_parents=sum(any(b.get('opportunities') for b in r['branches']) for r in records))
    gate = config["gate"].get(split)
    passed = len(records) == pool["parents"] and (not gate or (
        stats["witness_parents"] >= gate["witness_parents"] and
        min(winners[o] for o in ("ramp","bumper")) >= gate["each_winner"] and
        nearest_loses >= gate["nearest_preempted"]))
    manifest = dict(protocol_sha256=protocol_hash, dataset_sha256=digest(path), split=split,
                    statistics=stats, gate_passed=bool(passed), records=records)
    (folder/"manifest.json").write_text(json.dumps(manifest))
    return manifest


def build(root=DATA, protocol=PROTOCOL):
    config = json.loads(protocol.read_text())
    protocol_hash = digest(protocol)
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    for split in config["splits"]:
        target = root/split
        if target.exists():
            manifest = json.loads((target/"manifest.json").read_text())
            if manifest["protocol_sha256"] != protocol_hash or digest(target/"branches.npz") != manifest["dataset_sha256"]:
                raise ValueError(f"Dataset binding changed in {split}")
        else:
            with tempfile.TemporaryDirectory(dir=root) as staging:
                folder = Path(staging)/split
                folder.mkdir()
                manifest = collect_split(split, config, protocol_hash, folder)
                folder.rename(target)
        print(split, manifest["statistics"], "pass", manifest["gate_passed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    build(args.root, args.protocol)
