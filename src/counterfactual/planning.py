"""Forecast-only selection. This module deliberately does not import the simulator."""
from time import perf_counter
import numpy as np
from .geometry import candidates
from .goals import goal_result


def select_index(grid, margins, success):
    passing = np.flatnonzero(success)
    if not len(passing):
        return None
    return int(min(passing, key=lambda i: (abs(grid[i]), -margins[i], grid[i])))


def plan(scene, goal, predictor, count=257, safety_margin=.03, baseline=None):
    if goal.ball_id != scene.ball.id:
        raise ValueError("goal ball ID does not match scene")
    if safety_margin < 0 or not np.isfinite(safety_margin):
        raise ValueError("safety margin must be finite and nonnegative")
    started = perf_counter()
    if baseline is not None:
        baseline.validate_for(scene, goal)
    grid, valid, reasons = candidates(scene, count)
    indices = np.flatnonzero(valid)
    margins = np.full(count, np.nan)
    passed = np.zeros(count, dtype=bool)
    paths, states, contacts = {}, {}, {}
    if len(indices):
        forecasts, scores = predictor.forecast(scene, grid[indices])
        for j, i in enumerate(indices):
            path = forecasts[j, :, :2]
            # Fixed broad numerical bounds, not simulator oracle knowledge.
            numerical = np.isfinite(forecasts[j]).all() and np.max(np.abs(path)) < 100
            if not numerical:
                continue
            result = goal_result(np.linspace(0, 2, 121), path, goal)
            margins[i] = result["margin"]
            passed[i] = result["success"] and result["margin"] >= safety_margin
            paths[int(i)], states[int(i)], contacts[int(i)] = path, forecasts[j], scores[j]
    if baseline is not None:
        zero = int(np.argmin(abs(grid)))
        passed[zero] = baseline.success
        margins[zero] = baseline.margin
    selected = select_index(grid, margins, passed)
    status = "known baseline success" if baseline is not None and baseline.success else ("predicted success" if selected is not None else ("no predicted solution found" if len(indices) else "no valid edits"))
    return {"status": status,
            "grid": grid, "valid": valid, "reasons": reasons, "margins": margins, "success": passed,
            "selected_index": selected, "paths": paths, "states": states, "contacts": contacts,
            "safety_margin_m": safety_margin, "model": predictor.identity,
            "baseline_evidence_used": baseline is not None,
            "evaluated_candidates": int(len(indices)), "simulator_calls": 0, "seconds": perf_counter()-started}
