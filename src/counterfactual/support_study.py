"""Trace the first sustained path error against two references and isolate planar support."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .architecture_data import digest
from .contact_suite import fixtures, match_events
from .continuation_audit import onsets
from .continuation_data import DATA
from .continuation_experiment import read_split, write
from .continuation_solver import bounded, rollout
from .persistent_support import SLOP, support_rollout
from .recurrent import pack
from .schemas import Ball, Bumper, SceneSpec, Segment
from .simulator import simulate

PROTOCOL = Path("configs/persistent-support.json")
ROOT = Path("artifacts/persistent-support")
MODES = ("bounded", "support", "four")
RATES = (1920, 3840)


def first_run(mask, n):
    """Index of the first qualifying observation (or None, including censored tails)."""
    mask = np.asarray(mask, bool)
    if len(mask) < n:
        return None
    starts = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(mask, n).all(-1))
    return int(starts[0]) if len(starts) else None


def fixture_scenes():
    floor = Segment(id="floor", a=(0.1, 0.18), b=(5.9, 0.18))
    slope = Segment(id="ramp", a=(0.3, 1.8), b=(5.7, 0.72))
    tangent = np.array([1, -0.2]) / np.sqrt(1.04)
    normal = np.array([-tangent[1], tangent[0]])
    base = np.array(slope.a) + tangent * 0.8 + 0.16 * normal
    cases = [
        ("flat_rest", (1, 0.34), (0, 0), (floor,), (5, 3)),
        ("incline_roll", base, 0.5 * tangent, (slope,), (5, 3)),
        (
            "leave_platform",
            (1, 1.36),
            (1.8, 0),
            (Segment(id="platform", a=(0.2, 1.2), b=(2.2, 1.2)),),
            (5, 3),
        ),
        ("lift_off", (1, 0.34), (0.5, 1.5), (floor,), (5, 3)),
        ("roll_into_bumper", (1, 0.34), (1.8, 0), (floor,), (2.3, 0.57)),
    ]
    authored = [
        SceneSpec(
            id=k,
            parent_id=k,
            family="authored",
            ball=Ball(position=p, velocity=v),
            bumper=Bumper(position=b),
            segments=segments,
        )
        for k, p, v, segments, b in cases
    ]
    return authored + [next(s for s in fixtures() if s.id == "repeated_contact")]


def observations(tr):
    hz = tr.provenance["physics_hz"]
    ratio = hz // 60
    by_frame = defaultdict(dict)
    prev, age = {}, {}
    times = defaultdict(list)
    for e in tr.events:
        if e["phase"] == "begin":
            times[e["objects"][1]].append(e["time"])
    for resp in tr.responses:
        obj = resp["object"]
        tick = round(resp["time"] * hz)
        age[obj] = (
            age.get(obj, 0) + 1
            if prev.get(obj) == tick - 1 and not resp["first_contact"]
            else 1
        )
        prev[obj] = tick
        f = (tick - 1) // ratio
        item = by_frame[f].setdefault(
            obj,
            {
                "ticks": 0,
                "longest_age_s": 0,
                "support_ticks": 0,
                "begins": 0,
                "max_approach_m_s": 0,
                "normal_impulse_Ns": 0,
            },
        )
        nx, ny = resp["normal_ball_to_static"]
        vx, vy = resp["before"][2:4]
        jx, jy = resp["impulse_on_ball"]
        speed = vx * nx + vy * ny
        impulse = -(jx * nx + jy * ny)
        item["ticks"] += 1
        item["longest_age_s"] = max(item["longest_age_s"], age[obj] / hz)
        item["support_ticks"] += int(speed <= 0.2 and impulse > 1e-8)
        item["begins"] += int(resp["first_contact"])
        item["max_approach_m_s"] = max(item["max_approach_m_s"], speed)
        item["normal_impulse_Ns"] += impulse
    return by_frame, times


def err(states, ref, a=0, b=2):
    return np.linalg.norm(states[..., a:b] - ref[..., a:b], axis=-1)


def forecasts(states, contexts):
    paths, events, times = {}, {}, {}
    for mode in MODES:
        t = perf_counter()
        if mode == "support":
            s, clocks, held = support_rollout(states, contexts)
        else:
            s, clocks, _ = rollout(states, contexts, mode)
        times[mode] = perf_counter() - t
        paths[mode], events[mode] = s.numpy(), clocks.numpy()
    # Compute diagnostic flags separately so timing uses the normal forecast.
    s, _, excess = rollout(states, contexts, audit=True)
    np.testing.assert_array_equal(s.numpy(), paths["bounded"])
    return paths, events, held.numpy(), excess.numpy(), times


def reset_error(reference, context, tick, target):
    """Forecast from two neighboring exact states to the same end observation."""
    if tick is None:
        return None
    xy = []
    for start in (tick, tick + 1):
        state = torch.as_tensor(reference[start : start + 1], dtype=context.dtype)
        for _ in range(start, target):
            state = bounded(state, context[None])[0]
        xy.append(float(np.linalg.norm(state[0, :2].numpy() - reference[target, :2])))
    return {
        "start_interval": tick,
        "target_observation": target,
        "reset_before_m": xy[0],
        "reset_after_m": xy[1],
    }


def summarize_branch(row, scene, x, events, held, overflow, context, cfg, traces=False):
    refs = [simulate(scene, row["edit"], hz, record_responses=True) for hz in RATES]
    references = np.array([tr.states for tr in refs])
    s = torch.as_tensor(references[:, :-1].reshape(-1, 6), dtype=context.dtype)
    exact, _, _, capped = bounded(
        s, context.expand(len(s), -1, -1), audit_overflow=True
    )
    exact = exact.reshape(2, 120, 6).numpy()
    capped = capped.reshape(2, 120).numpy()
    vel = err(exact, references[:, 1:], 2, 4)
    path_err = {k: err(v, references) for k, v in x.items()}
    ref_err = err(*references)
    above = cfg["threshold_m"]
    n = cfg["consecutive_samples"]
    first = first_run((path_err["bounded"] > above).all(0), n)
    unstable = first_run(ref_err > above, n)
    details = [observations(tr) for tr in refs]
    eligible = (vel > cfg["one_step_velocity_threshold_m_s"]).all(0)
    for tr in refs:
        eligible &= tr.contacts.any(-1)
    if first is None:
        eligible[:] = False
    else:
        eligible[first:] = False
    earliest = first_run(eligible, 1)
    landmarks = {
        "joint_crossing_interval": first - 1 if first is not None else None,
        "first_release_overflow": first_run(overflow, 1),
        "first_local_velocity_error": earliest,
        **{
            f"first_exact_overflow_{hz}": first_run(capped[i], 1)
            for i, hz in enumerate(RATES)
        },
    }
    records = {}
    for k, t in landmarks.items():
        if t is None:
            continue
        records[k] = {
            "interval": t,
            "release_overflow": bool(overflow[t]),
            "rates": {
                str(hz): {
                    "objects": details[i][0].get(t, {}),
                    "exact_overflow": bool(capped[i, t]),
                    "velocity_error_m_s": float(vel[i, t]),
                    "position_error_m": float(path_err["bounded"][i, t + 1]),
                    "position_after_6_intervals_m": float(
                        path_err["bounded"][i, min(120, t + 6)]
                    ),
                }
                for i, hz in enumerate(RATES)
            },
        }
    modes = {}
    for k in x:
        on = onsets(events[k], "four" if k == "four" else "bounded")
        metrics = {}
        for i, hz in enumerate(RATES):
            e = path_err[k][i]
            counts = Counter()
            for j, obj in enumerate(refs[i].object_ids):
                match = match_events(
                    details[i][1][obj], [t for edge, t in on if edge == j]
                )
                counts.update(
                    {key: match[key] for key in ("matched", "missed", "invented")}
                )
            metrics[str(hz)] = dict(
                rmse_m=float(np.sqrt(np.mean(e**2))),
                p95_m=float(np.quantile(e, 0.95)),
                max_m=float(e.max()),
                first_observation=first_run(e > above, n),
                **counts,
            )
        modes[k] = metrics
    held_n = held.any(-1)
    item = {
        "parent": row["parent"],
        "edit": row["edit"],
        "scene_id": scene.id,
        "modes": modes,
        "held_frames": int(held_n.sum()),
        "held_edges": {
            obj: int(held[:, j].sum()) for j, obj in enumerate(refs[0].object_ids)
        },
        "overflow_frames": int(overflow.sum()),
        "rate_divergence_observation": unstable,
        "rate_rmse_m": float(np.sqrt(np.mean(ref_err**2))),
        "joint_divergence_observation": first,
        "landmarks": records,
        "support_evidence": {},
    }
    for i, hz in enumerate(RATES):
        truth = details[i][0]
        item["support_evidence"][str(hz)] = {
            "actual_low_speed_frames": sum(
                any(obj["support_ticks"] for obj in objs.values())
                for objs in truth.values()
            ),
            "actual_aged_frames": sum(
                any(obj["longest_age_s"] >= 1 / 60 for obj in objs.values())
                for objs in truth.values()
            ),
            "held_with_true_edge": sum(
                any(
                    held[t, j] and truth.get(t, {}).get(obj, {}).get("support_ticks", 0)
                    for j, obj in enumerate(refs[i].object_ids)
                )
                for t in np.flatnonzero(held_n)
            ),
            "exact_overflow_frames": int(capped[i].sum()),
            "reset": reset_error(
                references[i],
                context,
                earliest,
                min(first + n - 1, earliest + 24) if earliest is not None else 0,
            ),
        }
    if traces:
        item["trace"] = {k: v[:, :2].tolist() for k, v in x.items()}
        item["trace"].update(
            {str(hz): tr.states[:, :2].tolist() for hz, tr in zip(RATES, refs)}
        )
        item["trace"]["held_intervals"] = np.flatnonzero(held_n).tolist()
    return item


def comparison(rows, hz):
    parents = defaultdict(list)
    for r in rows:
        parents[r["parent"]].append([r["modes"][k][str(hz)]["rmse_m"] for k in MODES])
    per_parent = np.array([np.mean(v, axis=0) for v in parents.values()])
    delta = per_parent[:, 1] - per_parent[:, 0]
    bootstrap = (
        np.random.default_rng(9031).choice(delta, size=(2000, len(delta))).mean(-1)
    )
    return {
        "parent_mean_m": dict(zip(MODES, per_parent.mean(0))),
        "support_minus_bounded_m": float(delta.mean()),
        "ci95_m": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "branch_p95_rmse_m": {
            k: float(
                np.quantile([r["modes"][k][str(hz)]["rmse_m"] for r in rows], 0.95)
            )
            for k in MODES
        },
    }


def study(output=ROOT):
    torch.set_num_threads(1)
    cfg = json.loads(PROTOCOL.read_text())
    if cfg["slop_m"] != SLOP or tuple(cfg["references_hz"]) != RATES:
        raise ValueError("Protocol must match the reference rates and simulator slop")
    groups = {}
    for split in ["fixtures", *cfg["splits"]]:
        if split == "fixtures":
            scenes = fixture_scenes()
            rows = [{"parent": s.id, "edit": 0} for s in scenes]
            states, contexts = [
                torch.from_numpy(np.stack(v)) for v in zip(*(pack(s) for s in scenes))
            ]
        else:
            rows, data, manifest = read_split(split)
            by_seed = {
                r["scene"]["seed"]: SceneSpec.model_validate(r["scene"])
                for r in manifest["records"]
            }
            scenes = [by_seed[r["parent"]] for r in rows]
            states, contexts = data["states"][:, 0], data["contexts"]
        paths, events, held, overflow, timings = forecasts(states, contexts)
        items = []
        example_saved = False
        for i, (r, scene) in enumerate(zip(rows, scenes, strict=True)):
            x = {k: v[i] for k, v in paths.items()}
            item = summarize_branch(
                r,
                scene,
                x,
                {k: v[i] for k, v in events.items()},
                held[i],
                overflow[i],
                contexts[i],
                cfg,
                traces=(split == "fixtures" or not example_saved),
            )
            # First joint crossing on a two-rate stable branch, in frozen manifest order.
            if split != "fixtures" and "trace" in item:
                if (
                    item["joint_divergence_observation"] is not None
                    and item["rate_divergence_observation"] is None
                ):
                    example_saved = True
                else:
                    del item["trace"]
            items.append(item)
        groups[split] = {
            "rows": items,
            "batch_seconds": timings,
            "comparison": {str(hz): comparison(items, hz) for hz in RATES},
        }
        write(output / f"{split}.json", groups[split])
        print(split, len(items), groups[split]["comparison"], flush=True)
    cohorts = [groups[s]["comparison"] for s in cfg["splits"]]
    gate = any(all(v["ci95_m"][1] < 0 for v in c.values()) for c in cohorts) and all(
        v["support_minus_bounded_m"] < 0.001 for c in cohorts for v in c.values()
    )
    write(
        output / "summary.json",
        {
            "protocol_sha256": digest(PROTOCOL),
            "decision_gate": gate,
            "cohort_files": {s: digest(output / f"{s}.json") for s in groups},
            "inputs": {s: digest(DATA / s / "manifest.json") for s in cfg["splits"]},
        },
    )
    write(output / "report-kind.json", {"kind": "persistent-support-v1"})
    print("Fresh decision gate:", gate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    with torch.inference_mode():
        study(parser.parse_args().output)
