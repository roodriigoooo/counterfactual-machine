"""Retrospective on the frozen competition study's within-interval ramp predictions."""
import argparse
import json
from itertools import groupby
from pathlib import Path

import pymunk
import torch

from .architecture_data import digest
from .competition_experiment import load_split, surface_ids
from .competition_scenes import DATA, PROTOCOL, REPORT
from .contact_events import event_labels
from .recurrent import DT
from .schemas import SceneSpec
from .simulator import simulate
from .subinterval import step


def begin_groups(callbacks, tick, intervals=1):
    """Store equal-time cross-surface begins as one mark, independent of callback order."""
    events = sorted((e["time_s"], e["object"]) for e in callbacks
                    if e["phase"] == "begin" and tick <= e["interval"] < tick + intervals)
    return [{"ms": (time - tick * DT) * 1000, "surfaces": [obj for _, obj in group]}
            for time, group in groupby(events, key=lambda e: e[0])]


@torch.inference_mode()
def control_rows(manifest, data):
    rows = []
    for record in manifest["records"]:
        scene = SceneSpec.model_validate(record["scene"])
        tick = record["audits"]["1920"]["tick"]
        original = next(b["row"] for b in record["branches"] if b["edit_m"] == 0)
        ramp = surface_ids(scene).index("ramp")
        for branch in record["branches"]:
            if not branch["valid"]:
                continue
            row = branch["row"]
            if data["begin"][row, :tick].any() or not torch.equal(
                data["states"][row, :tick+1], data["states"][original, :tick+1]
            ):
                continue
            rows.append(dict(parent_id=scene.parent_id, edit_m=branch["edit_m"], row=row,
                             tick=tick, ramp_index=ramp, ramp_begin=bool(data["begin"][row, tick, ramp]),
                             analytic_ramp={}))

    ids = [r["row"] for r in rows]
    ticks = [r["tick"] for r in rows]
    edges = [r["ramp_index"] for r in rows]
    context = data["contexts"][ids]
    for n in (1, 4, 16):
        state = data["states"][ids, ticks]
        touched = torch.zeros_like(context[..., 9], dtype=torch.bool)
        for _ in range(n):
            state, admitted, _, _ = step(state, context, DT / n)
            touched |= admitted
        for row, predicted in zip(rows, touched[range(len(rows)), edges].tolist(), strict=True):
            row["analytic_ramp"][str(n)] = predicted
    return rows


def threshold_counts(pairs, seed):
    scores = [pair["scores"][seed] for pair in pairs]

    def correct(t):
        return sum(p["original"] < t <= p["edited"] for p in scores)

    # Each successful pair allows (original, edited]. A maximum overlap occurs
    # at a right endpoint. This is an in-sample bound, never a fitted threshold.
    return {"parents": len(scores), "fixed_half": correct(.5),
            "upper_bound": max((correct(p["edited"]) for p in scores), default=0)}


def diagnose():
    validation = DATA / "validation"
    audit_path = REPORT / "audit.json"
    audit = json.loads(audit_path.read_text())
    hashes = {str(audit_path): digest(audit_path), str(PROTOCOL): digest(PROTOCOL)}
    manifest, data = load_split(validation, hashes[str(PROTOCOL)])
    for name in ("manifest.json", "branches.npz"):
        path = validation / name
        hashes[str(path)] = digest(path)
        if hashes[str(path)] != audit["dataset_hashes"][f"validation/{name}"]:
            raise ValueError(f"Saved replay audit references different data at {path}")
    scores = {}
    for seed in json.loads(PROTOCOL.read_text())["comparison"]["seeds"]:
        path = REPORT / "E2" / f"seed-{seed}" / "evaluation.json"
        result = json.loads(path.read_text())
        if result["binding"]["manifest_sha256"]["validation"] != hashes[str(validation / "manifest.json")]:
            raise ValueError(f"Different validation parents in {path}")
        scores[str(seed)] = {r["parent_id"]: r for r in result["parents"] if r["legal_edit"]}
        hashes[str(path)] = digest(path)

    aligned = control_rows(manifest, data)
    branches = {(r["parent_id"], r["edit_m"]): r for r in aligned}
    records = {r["scene"]["parent_id"]: r for r in manifest["records"]}
    audited = {p["parent_id"] for p in audit["legal_parent_replays"]}
    if any(set(run) != audited for run in scores.values()):
        raise ValueError("Saved predictors and replay audit selected different legal pairs")
    pairs = []
    for p in audit["legal_parent_replays"]:
        parent, tick, edit = p["parent_id"], p["tick"], p["edit_m"]
        record = records[parent]
        branch = next(b for b in record["branches"] if b["edit_m"] == edit)
        coarse = begin_groups(branch["callbacks"], tick)
        scene = SceneSpec.model_validate(record["scene"])
        fine = event_labels(simulate(scene, edit, physics_hz=3840), scene.ball.id)["ordered_events"]
        if any(run[parent]["legal_edit"]["edit_m"] != edit or run[parent]["winner"] != "bumper"
               for run in scores.values()):
            raise ValueError(f"Saved ramp scores have a different edit or target in {parent}")
        base, edited = branches[parent, 0.], branches[parent, edit]
        fine_hit = any("ramp" in g["surfaces"] for g in begin_groups(fine, tick))
        checks = p["checks"]
        if (base["ramp_begin"] or not edited["ramp_begin"] or
                not all(v["prefix_equal"] and not v["original_ramp"] for v in checks.values()) or
                not checks["1920"]["edited_ramp"] or fine_hit != checks["3840"]["edited_ramp"]):
            raise ValueError(f"Interval labels or shared prefix disagree with the audit in {parent}")
        pairs.append(dict(parent_id=parent, edit_m=edit, tick=tick, stable_label=fine_hit,
                          begins_1920=coarse, begins_3840_two_intervals=begin_groups(fine, tick, 2),
                          scores={s: dict(original=run[parent]["loser_probability"],
                                          edited=run[parent]["legal_edit"]["ramp_probability"])
                                  for s, run in scores.items()},
                          analytic_pair={n: not base["analytic_ramp"][n] and hit
                                         for n, hit in edited["analytic_ramp"].items()}))

    strata = {"all": pairs, "stable_label": [p for p in pairs if p["stable_label"]]}
    for p in pairs:
        sequence = " then ".join(" + ".join(g["surfaces"]) for g in p["begins_1920"])
        strata.setdefault(sequence, []).append(p)
    results = {k: dict(parents=len(v), e2={s: threshold_counts(v, s) for s in scores},
                       analytic={n: sum(p["analytic_pair"][n] for p in v) for n in ("1", "4", "16")})
               for k, v in strata.items()}
    confusion = {}
    for n in ("1", "4", "16"):
        confusion[n] = {name: sum(r["ramp_begin"] == truth and r["analytic_ramp"][n] == pred for r in aligned)
                        for name, truth, pred in [("tp", True, True), ("fn", True, False),
                                                  ("fp", False, True), ("tn", False, False)]}
    return dict(kind="continuation-review-v1", input_sha256=hashes, pymunk=pymunk.version,
                aligned_branches=aligned, exact_state_control=confusion, pairs=pairs, strata=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/continuation-review"))
    output = parser.parse_args().output
    report = diagnose()
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report-kind.json").write_text(json.dumps({"kind": report["kind"]}) + "\n")
    print(json.dumps(report["strata"]["all"], indent=2))
