"""Local API: forecast-only planning, then an explicit single verification."""
import json
from pathlib import Path
from collections import OrderedDict
from uuid import uuid4
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .schemas import Record, SceneSpec, GoalSpec, EditSpec
from .geometry import validate_scene, apply_edit
from .models import load_predictor
from .outcomes import BaselineOutcome, digest
from .planning import plan
from .goals import goal_result
from .scenes import generate_scene, generate_goal
from .simulator import simulate


class Task(Record):
    scene: SceneSpec
    goal: GoalSpec


class VerificationRequest(Record):
    plan_id: str


class ManualTask(Task):
    edit_m: float = 0.0


def create_app(checkpoint=Path("artifacts/pilot/baseline.pt"), artifacts=Path("artifacts/pilot")):
    torch.set_num_threads(4)
    app = FastAPI(title="Counterfactual Machine", version="0.1.0")
    plans = OrderedDict()
    predictor = load_predictor(checkpoint) if checkpoint.exists() else None
    outcomes = OrderedDict()
    ui = Path(__file__).resolve().parents[2]/"ui"/"dist"

    def validate(task):
        try:
            validate_scene(task.scene)
            if task.goal.ball_id != task.scene.ball.id:
                raise ValueError("goal ball ID does not match scene")
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/info")
    def info():
        return {"model": predictor.identity if predictor else None,
                "physics_hz": 1920, "horizon_s": 2, "candidate_count": 257,
                "report_available": (artifacts/"evaluation.json").exists()}

    @app.get("/api/presets")
    def presets():
        gallery_file = artifacts/"gallery.json"
        gallery = json.loads(gallery_file.read_text()) if gallery_file.exists() else []
        if isinstance(gallery, dict):
            # The factorial report stores named diagnostic examples, including
            # non-task contact plots. Only complete scene/goal entries are playable.
            gallery = [{**entry, "category": "success" if key == "success" else "false_success"}
                       for key, entry in gallery.items() if "scene" in entry and "goal" in entry]
        order = {"success": 0, "false_success": 1, "missed_solution": 2, "grid_unsolved": 3}
        gallery.sort(key=lambda e: order[e["category"]])
        result = [{"name": e["category"].replace("_", " ").capitalize()+" · research example", "scene": e["scene"], "goal": e["goal"]} for e in gallery]
        result += [{"name": f"{generate_scene(s).family.capitalize()} · scene {s}", "scene": generate_scene(s).model_dump(), "goal": generate_goal(s).model_dump()} for s in [0, 1, 8, 11]]
        return result

    @app.post("/api/simulate")
    def original(task: Task):
        validate(task)
        tr = simulate(task.scene)
        result = goal_result(tr.physics_times, tr.physics_positions, task.goal)
        key = (task.scene.content_hash(), digest(task.goal.model_dump()))
        outcomes[key] = (BaselineOutcome.from_verification(task.scene, task.goal, result), tr.public(), result)
        while len(outcomes) > 128:
            outcomes.popitem(last=False)
        return {"trajectory": tr.public(), "goal": result, "simulator_calls": 1}

    @app.post("/api/validate")
    def validate_authored(task: ManualTask):
        validate(task)
        try:
            apply_edit(task.scene, EditSpec(object_id=task.scene.bumper.id, displacement=task.edit_m))
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"valid": True, "simulator_calls": 0}

    @app.post("/api/try")
    def try_manual_edit(task: ManualTask):
        # A player's bounded move is distinct from a model-selected verification.
        validate_authored(task)
        tr = simulate(task.scene, task.edit_m)
        result = goal_result(tr.physics_times, tr.physics_positions, task.goal)
        return {"trajectory": tr.public(), "goal": result, "simulator_calls": 1}

    @app.post("/api/plan")
    def forecast(task: Task):
        validate(task)
        if predictor is None:
            raise HTTPException(503, "Train the E0 checkpoint first: counterfactual train")
        known = outcomes.get((task.scene.content_hash(), digest(task.goal.model_dump())))
        result = plan(task.scene, task.goal, predictor, baseline=known[0] if known else None)
        chosen = result["selected_index"]
        original_states, _ = predictor.forecast(task.scene, [0.])
        token = str(uuid4())
        cached = {"status": "verified success", "goal": known[2], "trajectory": known[1], "new_simulator_calls": 0} if known and known[0].success else None
        plans[token] = (task, result, cached)
        while len(plans) > 128:
            plans.popitem(last=False)
        return {"plan_id": token, "status": result["status"], "model": result["model"],
                "edit_m": float(result["grid"][chosen]) if chosen is not None else None,
                "normalized_cost": abs(float(result["grid"][chosen]))/task.scene.ball.radius if chosen is not None else None,
                "predicted_margin_m": float(result["margins"][chosen]) if chosen is not None else None,
                "predicted_original": original_states[0].tolist(),
                "predicted_edited": result["states"][chosen].tolist() if chosen in result["states"] else None,
                "candidate_count": result["evaluated_candidates"], "seconds": result["seconds"], "simulator_calls": 0,
                "grid_m": result["grid"].tolist(),
                "margins_m": [float(v) if np.isfinite(v) else None for v in result["margins"]]}

    @app.post("/api/verify")
    def verify(request: VerificationRequest):
        if request.plan_id not in plans:
            raise HTTPException(404, "Plan expired; request a new prediction")
        task, result, cached = plans[request.plan_id]
        if cached:
            return {**cached, "new_simulator_calls": 0}
        chosen = result["selected_index"]
        if chosen is None:
            raise HTTPException(409, "No selected edit to verify")
        tr = simulate(task.scene, float(result["grid"][chosen]))
        actual = goal_result(tr.physics_times, tr.physics_positions, task.goal)
        response = {"status": "verified success" if actual["success"] else "verification failed",
                    "goal": actual, "trajectory": tr.public(), "new_simulator_calls": 1}
        plans[request.plan_id] = (task, result, response)
        return response

    @app.get("/api/report")
    def report():
        file = artifacts/"evaluation.json"
        if not file.exists():
            raise HTTPException(404, "Run counterfactual evaluate first")
        return json.loads(file.read_text())

    @app.get("/api/research")
    def research():
        root = Path("artifacts/architecture")
        file = root/"summary.json"
        if not file.exists():
            raise HTTPException(404, "Architecture comparison is not available yet")
        return {"summary": json.loads(file.read_text()),
                "figures": [p.stem for p in sorted(root.glob("architecture-*.png"))]}

    @app.get("/research-figures/{name}")
    def research_figure(name: str):
        if Path(name).name != name or not name.startswith("architecture-") or Path(name).suffix not in [".png", ".svg", ".pdf"]:
            raise HTTPException(404)
        file = Path("artifacts/architecture")/name
        if not file.is_file():
            raise HTTPException(404)
        return FileResponse(file)

    @app.get("/api/contact-research")
    def contact_research():
        root = Path("artifacts/contact-followup")
        if not (root/"summary.json").is_file():
            raise HTTPException(404, "Contact research is not available yet")
        summary = json.loads((root/"summary.json").read_text())
        example = summary.get("demonstration", {}).get("examples", {}).get("success")
        # Loading a measured field only supplies geometry and a move. Drop ball
        # still executes the usual manual verification; research is not a hint.
        return {"summary": summary["tolerance"], "example": example,
                "figures": [p.stem for p in sorted(root.glob("*.png"))
                            if p.name.startswith(("contact-", "tolerance-"))]}

    @app.get("/contact-figures/{name}")
    def contact_figure(name: str):
        if Path(name).name != name or not name.startswith(("contact-", "tolerance-")) or Path(name).suffix not in [".png", ".svg", ".pdf"]:
            raise HTTPException(404)
        file = Path("artifacts/contact-followup")/name
        if not file.is_file():
            raise HTTPException(404)
        return FileResponse(file)

    if artifacts.exists():
        # Deliberately expose only report figures, not checkpoint or arbitrary files.
        @app.get("/figures/{name}")
        def figure(name: str):
            if Path(name).name != name or Path(name).suffix not in [".png", ".svg", ".pdf"]:
                raise HTTPException(404)
            file = artifacts/name
            if not file.is_file():
                raise HTTPException(404)
            return FileResponse(file)

    if ui.exists():
        app.mount("/", StaticFiles(directory=ui, html=True), name="ui")
    return app
