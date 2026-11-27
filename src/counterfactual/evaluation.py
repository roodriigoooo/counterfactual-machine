"""Pilot evidence only: parent-held-out tasks, fixed thresholds, explicit budgets."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from .models import Predictor
from .planning import plan
from .reference import sweep
from .schemas import SceneSpec
from .scenes import generate_goal
from .goals import goal_result
from .simulator import simulate


def public_plan(p):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()
            if k not in ["paths", "states", "contacts"]}


def interval(rows, numerator, denominator, seed=20261108):
    parents = sorted({r["parent_id"] for r in rows})
    totals = np.array([[sum(numerator(r) for r in rows if r["parent_id"] == p),
                        sum(denominator(r) for r in rows if r["parent_id"] == p)] for p in parents])
    if not len(totals) or not totals[:, 1].sum():
        return {"estimate": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    bootstrap = totals[rng.integers(len(parents), size=(2000, len(parents)))].sum(1)
    ratios = bootstrap[bootstrap[:, 1] > 0, 0]/bootstrap[bootstrap[:, 1] > 0, 1]
    return {"estimate": float(totals[:, 0].sum()/totals[:, 1].sum()),
            "ci95": np.quantile(ratios, [.025, .975]).tolist(), "n": int(totals[:, 1].sum())}


def evaluate(data: Path, checkpoint: Path, output: Path, parents=16, goals_per_parent=8, predictor_override=None, use_baseline=False):
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data/"manifest.json").read_text())
    predictor = predictor_override if predictor_override is not None else Predictor(checkpoint)
    if predictor.identity["dataset_sha256"] != manifest["sha256"]:
        raise ValueError("checkpoint and evaluation manifest do not match")
    test = [r for r in manifest["records"] if r["split"] == "test"][:parents]
    rows, galleries, tasks_manifest = [], [], []
    rng = np.random.default_rng(20261108)
    started = perf_counter()
    # Warm up inference; measured planning latency includes feature preparation + scoring.
    predictor.forecast(SceneSpec.model_validate(test[0]["scene"]), [0.0])
    for record in test:
        scene = SceneSpec.model_validate(record["scene"])
        original = simulate(scene)
        # Dynamics are independent of goals: oracle paths can be shared across intentions.
        base_goal = generate_goal(scene.seed*goals_per_parent)
        oracle = sweep(scene, base_goal)
        # Physics-rate paths are needed for the closed-center monitor, not only observations.
        # Cache the full trajectories once here for alternative goals.
        trajectories = oracle["trajectories"]
        for j in range(goals_per_parent):
            goal = generate_goal(scene.seed*goals_per_parent+j, scene.ball.id)
            tasks_manifest.append({"scene": scene.model_dump(), "scene_hash": scene.content_hash(),
                                   "goal_index": j, "goal": goal.model_dump()})
            original_result = goal_result(original.physics_times, original.physics_positions, goal)
            truth = [goal_result(t.physics_times, t.physics_positions, goal) if t else {"success": False, "margin": None} for t in trajectories]
            success = np.array([v["success"] for v in truth])
            margins = np.array([v["margin"] if v["margin"] is not None else np.nan for v in truth])
            passing = np.flatnonzero(success)
            best = min(passing, key=lambda i: (abs(oracle["grid"][i]), -margins[i], oracle["grid"][i])) if len(passing) else None
            from .outcomes import BaselineOutcome
            baseline = BaselineOutcome.from_verification(scene, goal, original_result) if use_baseline else None
            result = plan(scene, goal, predictor, baseline=baseline)
            chosen = result["selected_index"]
            # Selection is frozen; one new simulator rollout, never an oracle-selected retry.
            verified = simulate(scene, float(result["grid"][chosen])) if chosen is not None else None
            actual = goal_result(verified.physics_times, verified.physics_positions, goal) if verified else None
            valid_indices = np.flatnonzero(oracle["valid"])
            random_i = int(rng.choice(valid_indices))
            search_indices = rng.choice(valid_indices, size=min(16, len(valid_indices)), replace=False)
            search_success = [i for i in search_indices if success[i]]
            random_best = min(search_success, key=lambda i: (abs(oracle["grid"][i]), -margins[i], oracle["grid"][i])) if search_success else None
            row = {"parent_id": scene.parent_id, "scene_seed": scene.seed, "goal_index": j,
                   "initial_success": original_result["success"], "reference_solvable": best is not None,
                   "selected": chosen is not None, "verified_success": bool(actual and actual["success"]),
                   "edit_cost": abs(float(result["grid"][chosen]))/scene.ball.radius if chosen is not None else None,
                   "oracle_cost": abs(float(oracle["grid"][best]))/scene.ball.radius if best is not None else None,
                   "random_success": bool(success[random_i]), "random_search_success": random_best is not None,
                   "expected_random_success": float(success[valid_indices].mean()),
                   "successful_candidates": int(success.sum()),
                   "random_search_cost": abs(float(oracle["grid"][random_best]))/scene.ball.radius if random_best is not None else None,
                   "predicted_margin_m": float(result["margins"][chosen]) if chosen is not None else None,
                   "verified_margin_m": actual["margin"] if actual else None,
                   "planner_seconds": result["seconds"], "learned_verification_calls": int(chosen is not None),
                   "oracle_candidate_calls": int(oracle["valid"].sum()), "random_search_calls": len(search_indices)}
            rows.append(row)
            # Deterministic galleries: first verified intervention, first false selection,
            # first missed solvable task, and first initially failing grid-unsolved task.
            category = ("success" if row["verified_success"] else "false_success" if row["selected"] else "missed_solution" if row["reference_solvable"] else "grid_unsolved")
            if not row["initial_success"] and category not in [g["category"] for g in galleries]:
                display_i = chosen if chosen is not None else best
                u = float(oracle["grid"][display_i]) if display_i is not None else 0.
                predicted, _ = predictor.forecast(scene, [0., u])
                actual_edit = verified if verified is not None else simulate(scene, u)
                galleries.append({"category": category, "scene": scene.model_dump(), "goal": goal.model_dump(), "metrics": row,
                                  "edit_m": u, "display_edit_source": "learned selection" if chosen is not None else "oracle for diagnostic display only",
                                  "times": original.times.tolist(), "original": original.public(), "edited": actual_edit.public(),
                                  "predicted_original": predicted[0].tolist(), "predicted_edited": predicted[1].tolist(),
                                  "grid_m": oracle["grid"].tolist(), "valid": oracle["valid"].tolist(),
                                  "actual_margins_m": [None if not np.isfinite(v) else float(v) for v in margins],
                                  "predicted_margins_m": [None if not np.isfinite(v) else float(v) for v in result["margins"]],
                                  "bumper_contact": oracle["bumper_contacts"].tolist()})
        print(f"evaluated {scene.parent_id}", flush=True)
    intervention = [r for r in rows if not r["initial_success"]]
    solvable = [r for r in intervention if r["reference_solvable"]]
    metrics = {
        "verified_success_all": interval(intervention, lambda r: r["verified_success"], lambda r: 1),
        "verified_success_solvable": interval(solvable, lambda r: r["verified_success"], lambda r: 1),
        "false_predicted_success": interval(intervention, lambda r: r["selected"] and not r["verified_success"], lambda r: r["selected"]),
        "selection_coverage": interval(intervention, lambda r: r["selected"], lambda r: 1),
        "random_success_solvable": interval(solvable, lambda r: r["random_success"], lambda r: 1),
        "random_search_success_solvable": interval(solvable, lambda r: r["random_search_success"], lambda r: 1),
    }
    costs = [r["edit_cost"]-r["oracle_cost"] for r in solvable if r["verified_success"]]
    latencies = [r["planner_seconds"] for r in rows]
    report = {"kind": "exploratory development; not final A/B/C/D evidence", "model": predictor.identity,
              "baseline_guard": use_baseline,
              "split": "first held-out test parents in fixed manifest order", "test_parents": len(test),
              "goals_per_parent": goals_per_parent, "tasks": len(rows), "initially_failing": len(intervention),
              "initially_failing_grid_solvable": len(solvable), "metrics": metrics,
              "median_excess_cost_successes": float(np.median(costs)) if costs else None,
              "planning_latency_seconds": {"median": float(np.median(latencies)), "p95": float(np.quantile(latencies, .95))},
              "confidence_interval": "95% percentile bootstrap, 2000 parent-cluster resamples, one training seed",
              "safety_margin_m": .03, "threshold_policy": "fixed exploratory 0.03 m, not calibrated or tuned on test",
              "oracle_rollouts_total": sum(r["oracle_candidate_calls"] for r in rows[::goals_per_parent]),
              "notes": ["Oracle trajectory cache reused across goals; one grid simulated per parent.",
                        "Random baselines use offline oracle cache; logical budgets are 1 and 16 simulator calls per task.",
                        "Test goals are natural independent samples; no boundary challenge cohort in this pilot.",
                        "Model identity names the architecture; this development run is not a matched factorial comparison.",
                        "Shared-success excess cost is against matched grid only; dense audit is diagnostic."],
              "seconds": perf_counter()-started, "tasks_detail": rows}
    (output/"evaluation.json").write_text(json.dumps(report, indent=2))
    (output/"gallery.json").write_text(json.dumps(galleries, indent=2))
    (output/"tasks.json").write_text(json.dumps(tasks_manifest, indent=2))
    from .diagnostics import physical_diagnostics
    physical_diagnostics(data, checkpoint, output, predictor_override=predictor)
    return {k: v for k, v in report.items() if k not in ["tasks_detail", "notes"]}
