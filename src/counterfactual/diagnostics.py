"""Common physical/event diagnostics on held-out paired branches."""
import json
from pathlib import Path
import numpy as np
from .models import Predictor
from .schemas import SceneSpec


def physical_diagnostics(data: Path, checkpoint: Path, output: Path, predictor_override=None):
    manifest = json.loads((data/"manifest.json").read_text())
    arrays = np.load(data/"paired.npz")
    predictor = predictor_override if predictor_override is not None else Predictor(checkpoint)
    rows, errors, differences = [], [], []
    tp = fp = fn = 0
    for i, r in enumerate(manifest["records"]):
        if r["split"] != "test":
            continue
        scene = SceneSpec.model_validate(r["scene"])
        prediction, contacts = predictor.forecast(scene, r["edits_m"])
        truth = arrays["states"][i]
        error = np.linalg.norm(prediction[:, :, :2]-truth[:, :, :2], axis=-1)/scene.ball.radius
        difference = np.linalg.norm((prediction[:, :, :2]-prediction[4:5, :, :2])-(truth[:, :, :2]-truth[4:5, :, :2]), axis=-1)/scene.ball.radius
        mask = arrays["mask"][i, :, None, :].astype(bool)
        predicted = contacts >= .5
        actual = arrays["contacts"][i].astype(bool)
        tp += int(np.sum(predicted & actual & mask))
        fp += int(np.sum(predicted & ~actual & mask))
        fn += int(np.sum(~predicted & actual & mask))
        errors.append(error.mean(0)); differences.append(difference.mean(0))
        rows.append({"parent_id": scene.parent_id, "position_error_mean_radii": float(error.mean()),
                     "endpoint_error_mean_radii": float(error[:, -1].mean()),
                     "difference_error_mean_radii": float(difference.mean()), "nonfinite": int((~np.isfinite(prediction)).sum())})
    result = {"cohort": f"all {len(rows)} held-out test parents, 9 paired branches each", "parents": len(rows),
              "contact_threshold": .5, "contact_true_positive": tp, "contact_false_positive": fp, "contact_false_negative": fn,
              "contact_precision": tp/(tp+fp) if tp+fp else None, "contact_recall": tp/(tp+fn) if tp+fn else None,
              "parent_metrics": rows, "times_s": np.linspace(0,2,121).tolist(),
              "position_error_by_parent_radii": np.array(errors).tolist(),
              "difference_error_by_parent_radii": np.array(differences).tolist()}
    (output/"physical-diagnostics.json").write_text(json.dumps(result, indent=2))
    return result
