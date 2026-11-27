import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Machine · reproducible local pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="determinism and step-halving audit")
    audit.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    audit.add_argument("--parents", type=int, default=32)
    audit.add_argument("--physics-hz", type=int, default=1920)
    data = sub.add_parser("data", help="generate paired pilot after a passing audit")
    data.add_argument("--output", type=Path, default=Path("data/pilot"))
    data.add_argument("--parents", type=int, default=256)
    data.add_argument("--audit", type=Path, default=Path("artifacts/pilot/audit.json"))
    train = sub.add_parser("train", help="fit the exploratory E0 trajectory baseline")
    train.add_argument("--data", type=Path, default=Path("data/pilot"))
    train.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    evaluation = sub.add_parser("evaluate", help="held-out decisions and a fixed failure gallery")
    evaluation.add_argument("--data", type=Path, default=Path("data/pilot"))
    evaluation.add_argument("--checkpoint", type=Path, default=Path("artifacts/pilot/baseline.pt"))
    evaluation.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    evaluation.add_argument("--parents", type=int, default=16)
    evaluation.add_argument("--baseline-guard", action="store_true", help="use verified original outcome as separate decision evidence")
    figures = sub.add_parser("figures", help="export PNG, SVG and PDF from raw results")
    figures.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    serve = sub.add_parser("serve", help="start the local editor and replay interface")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--checkpoint", type=Path, default=Path("artifacts/pilot/baseline.pt"))
    serve.add_argument("--artifacts", type=Path, default=Path("artifacts/pilot"))
    dense = sub.add_parser("dense", help="2049-point diagnostic reference on a saved gallery case")
    dense.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    reference = sub.add_parser("reference", help="simulator grid search on a saved scene/goal JSON")
    reference.add_argument("--task", type=Path, default=Path("examples/avoidance.json"))
    reference.add_argument("--output", type=Path, default=Path("artifacts/reference"))
    reference.add_argument("--count", type=int, default=257)
    args = parser.parse_args()
    if args.command == "audit":
        from .reference import numerical_audit
        result = numerical_audit(args.output, args.parents, args.physics_hz)
    elif args.command == "data":
        from .data import generate_dataset
        audit_result = json.loads(args.audit.read_text())["summary"]
        if not audit_result["gate_passed"] or audit_result["physics_hz"] != 1920:
            parser.error("a passing audit at the current 1920 Hz physics setting is required")
        result = generate_dataset(args.output, args.parents)
    elif args.command == "train":
        from .training import train
        result = train(args.data, args.output, args.epochs, args.seed, args.device)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        from .models import load_predictor
        result = evaluate(args.data, args.checkpoint, args.output, args.parents,
                          predictor_override=load_predictor(args.checkpoint), use_baseline=args.baseline_guard)
    elif args.command == "figures":
        marker = args.output/"report-kind.json"
        kind = json.loads(marker.read_text()).get("kind") if marker.exists() else None
        if kind == "competition-scenes-v2":
            from .competition_report import draw
            result = draw(args.output)
        elif kind == "relay-encounter-v1":
            from .relay_encounter_figures import draw
            result = draw(args.output)
        elif kind == "impact-decoder-v1":
            from .impact_report import draw
            result = draw(args.output)
        elif kind == "persistent-support-v1":
            from .support_figures import support_figures
            result = support_figures(args.output)
        elif kind == "relay-exposure-v1":
            from .support_figures import relay_figure
            result = relay_figure(args.output)
        elif kind == "continuation-study-v1":
            from .continuation_report import draw
            result = draw(args.output)
        elif kind == "continuation-review-v1":
            from .competition_report import draw_continuation
            result = draw_continuation(args.output)
        elif kind == "contact-followup-v1":
            from .contact_report import build
            result = build(args.output)
        elif (args.output/"frozen-checkpoints.json").exists():
            from .architecture_report import build
            result = build(args.output)
        else:
            from .figures import build_figures
            result = build_figures(args.output)
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.checkpoint, args.artifacts), host="127.0.0.1", port=args.port)
        return
    elif args.command == "reference":
        import numpy as np
        from .reference import sweep
        from .schemas import SceneSpec, GoalSpec
        task = json.loads(args.task.read_text())
        scene, goal = SceneSpec.model_validate(task["scene"]), GoalSpec.model_validate(task["goal"])
        reference = sweep(scene, goal, args.count)
        selected = reference["selected_index"]
        result = {"method": "simulator reference, finite grid", "scene_hash": scene.content_hash(),
                  "status": "successful candidate found" if selected is not None else "no successful candidate found on this grid",
                  "edit_m": float(reference["grid"][selected]) if selected is not None else None,
                  "simulator_calls": reference["simulator_calls"], "seconds": reference["seconds"],
                  "grid_m": reference["grid"].tolist(), "valid": reference["valid"].tolist(),
                  "success": reference["success"].tolist(),
                  "margins_m": [float(v) if np.isfinite(v) else None for v in reference["margins"]]}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"reference.json").write_text(json.dumps(result, indent=2))
        if selected is not None:
            (args.output/"selected-replay.json").write_text(json.dumps(reference["trajectories"][selected].public(), indent=2))
        result = {k: v for k, v in result.items() if k not in ["grid_m", "valid", "success", "margins_m"]}
    else:
        import numpy as np
        from .reference import sweep
        from .schemas import SceneSpec, GoalSpec
        entries = json.loads((args.output/"gallery.json").read_text())
        entry = next((e for e in entries if e["category"] == "success"), entries[0])
        scene, goal = SceneSpec.model_validate(entry["scene"]), GoalSpec.model_validate(entry["goal"])
        reference = sweep(scene, goal, 2049)
        selected = reference["selected_index"]
        dense_cost = abs(float(reference["grid"][selected]))/scene.ball.radius if selected is not None else None
        result = {"parent_id": scene.parent_id, "category": entry["category"], "grid_count": 2049,
                  "valid_count": int(reference["valid"].sum()), "simulator_calls": reference["simulator_calls"],
                  "seconds": reference["seconds"], "coarse_cost": entry["metrics"]["oracle_cost"], "dense_cost": dense_cost,
                  "coarse_minus_dense": entry["metrics"]["oracle_cost"]-dense_cost if dense_cost is not None else None,
                  "scope": "one deterministic gallery case; finite nested grid, no continuous minimality claim; local refinement pending"}
        (args.output/"dense.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
