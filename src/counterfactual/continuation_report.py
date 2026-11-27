"""Numerical and visual record of the post-impact representation study."""

import json
import platform
from collections import Counter

import numpy as np
import pymunk
import torch

from .architecture_data import digest
from .continuation_data import DATA, PROTOCOL, REPORT
from .continuation_experiment import write
from .figures import BLUE, GREEN, GREY, ORANGE, RUST, Circle, plt, save, style


def read(path):
    return json.loads(path.read_text())


def exposure_detail(m):
    counts = Counter()
    for p in m["records"]:
        flags = set()
        for branch in p["branches"]:
            if not branch["audit"]:
                continue
            a, b = branch["audit"]["1920"], branch["audit"]["3840"]
            if a["tie"] and b["tie"]:
                flags.add("tie_both_rates")
            times = [next((t for obj, t in v["events"] if obj != v["order"][0]), None) for v in (a, b)]
            if (
                a["order"] == b["order"]
                and not a["tie"]
                and not b["tie"]
                and all(t is not None for t in times)
            ) and (times[0] <= 1000 / 60) != (times[1] <= 1000 / 60):
                flags.add("second_changes_frame")
            if branch["edit"] == 0:
                orig = (a, b)
                continue
            if (
                any(v["tie"] for v in (*orig, a, b))
                or orig[0]["order"] != orig[1]["order"]
                or a["order"] != b["order"]
            ):
                continue
            neg, pos = sorted([orig, (a, b)], key=lambda v: len(v[0]["order"]))
            if (
                len(neg[0]["order"]) == 1
                and len(pos[0]["order"]) == 2
                and neg[0]["order"][0] == pos[0]["order"][0]
            ) and all(0 < v["gap_m"] <= 0.010 for v in neg):
                flags.add("matched_near_miss_after_" + neg[0]["order"][0])
        counts.update(flags)
    return dict(**m["exposure"], **counts)


def decision_stats(root):
    selection = read(root / "selection.json")
    fits = read(root / "selection-calibration.json")
    from .continuation_selection import risk

    ps = [read(f) for f in sorted((root / "selection/evaluation").glob("*.json"))]
    diffs = {}
    common = {}
    for p in ps:
        rows = {m: {r["j"]: r for r in p["rows"] if r["model"] == m} for m in ("bounded", "four")}
        ds = []
        for j, b in rows["bounded"].items():
            f = rows["four"][j]
            ds.append(int(b.get("robust", False)) - int(f.get("robust", False)))
            if b["selected"] and f["selected"]:
                common[(p["seed"], j)] = abs(b["edit"]) - abs(f["edit"])
        diffs[p["seed"]] = (sum(ds), len(ds))
    v = np.array(list(diffs.values()))
    draws = np.random.default_rng(321).integers(len(v), size=(2000, len(v)))
    samples = v[draws].sum(1)
    ci = np.quantile(samples[:, 0] / samples[:, 1], [0.025, 0.975])
    rows = [r for p in ps for r in p["rows"] if r["model"] == "bounded" and r["selected"]]
    # Frozen cutoffs preserve deployment coverage in each paired parent draw.
    contrast = dict.fromkeys(diffs, 0.0)
    for row in rows:
        if row["robust"]:
            accepted = {m: int(risk(row, m, f) <= f["cutoff"]) for m, f in fits["fits"].items()}
            contrast[row["parent"]] += (
                np.mean([accepted[f"head-{seed}"] for seed in range(3)]) - accepted["analytic"]
            )
    delta = np.array(list(contrast.values()))
    selection["fixed_cutoff_head_minus_analytic"] = {
        "per_failing_task": float(delta.sum() / v[:, 1].sum()),
        "ci95": np.quantile(delta[draws].sum(1) / samples[:, 1], [0.025, 0.975]).tolist(),
    }
    matched = {}
    for m, f in fits["fits"].items():
        order = sorted(rows, key=lambda r: risk(r, m, f))
        risks = [risk(r, m, f) for r in order]
        cuts = {n for n in range(1, len(order) + 1) if n == len(order) or risks[n - 1] != risks[n]}
        matched[m] = {k: sum(r["robust"] for r in order[:k]) for k in cuts}
    shared = set.intersection(*(set(v) for v in matched.values()))
    k = min(shared, key=lambda k: (abs(k - 0.8 * len(rows)), k))
    selection["matched"] = {"proposals": k, "all_five": {m: v[k] for m, v in matched.items()}}
    selection["calibration"] = {key: fits[key] for key in ("train", "threshold")}
    selection["paired"] = {
        "success_delta": float(v[:, 0].sum() / v[:, 1].sum()),
        "ci95": ci.tolist(),
        "common": len(common),
        "mean_cost_delta_mm": float(np.mean(list(common.values())) * 1000),
    }
    selection["latency_parents"] = {
        m: [p["timings"][m]["cold_table_s"] for p in ps] for m in ("bounded", "four")
    }
    return selection


def summarize(root):
    run = read(root / "experiment.json")
    ab = read(root / "ablation.json")
    audited = read(root / "audit.json")
    cfg = read(PROTOCOL)
    summary = {
        "kind": "continuation-study-v1",
        "protocol": digest(PROTOCOL),
        "environment": {
            "torch": torch.__version__,
            "pymunk": pymunk.version,
            "system": platform.platform(),
            "machine": platform.machine(),
            "threads": 1,
        },
        "exposure": {s: exposure_detail(read(DATA / s / "manifest.json")) for s in cfg["splits"]},
        "release": run["modes"]["release"]["strata"],
        "paired": run["modes"]["release"]["paired"],
        "learned_delta": run["modes"]["release"]["head_minus_bounded"],
        "incoming": ab["primary"],
        "outgoing_delta": [ab["outgoing_minus_incoming"], ab["ci95"]],
        "selection": {
            s: decision_stats(p) for s, p in [("ordinary", root), ("enriched", root / "enriched-risk")]
        },
    }
    raw = read(root / "e2-calibration.json")["domains"]
    summary["e2"] = {
        s: [
            {k: {m: r[k][m] for m in ("n", "positive", "bce", "brier")} for k in ("raw", "calibrated")}
            for r in runs
        ]
        for s, runs in raw.items()
    }
    summary["releases"] = {}
    for split, runs in audited["releases"].items():
        summary["releases"][split] = {}
        for solver, r in runs.items():
            surface = [s for row in r["rows"] for s in row["surface"].values()]
            loc = [row["pre_max_m"] for row in r["rows"] if "pre_max_m" in row]
            summary["releases"][split][solver] = dict(
                **{k: r[k] for k in ("position_rmse_m", "velocity_p95", "overflow")},
                **{k: sum(s[k] for s in surface) for k in ("matched", "invented", "missed")},
                max_pre_m=max(loc, default=None),
            )
    write(root / "summary.json", summary)
    write(root / "report-kind.json", {"kind": summary["kind"]})


def representation(s, root):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.55), layout="constrained")
    primary = s["release"]["primary"]
    for key, color, marker, label in [
        ("incoming", ORANGE, "s", "Incoming flight"),
        ("head", BLUE, "o", "Outgoing flight"),
    ]:
        vals = s["incoming"] if key == "incoming" else primary
        y = [vals[f"{key}-{i}"]["parent_accuracy"] * 100 for i in range(3)]
        axes[0].plot(range(3), y, marker=marker, color=color, label=label)
    axes[0].axhline(
        primary["bounded"]["parent_accuracy"] * 100, color=GREEN, ls="--", label="Analytic solvers"
    )
    axes[0].set(
        xticks=range(3),
        xlabel="Training seed",
        ylabel="Correct second-event outcome (%)",
        ylim=(55, 104),
        title="(a) Before and after the bounce",
    )
    axes[0].legend(fontsize=9, loc="center right")
    for i, (name, label, col) in enumerate(
        [
            ("four", "Four substeps", GREY),
            ("bounded", "Two impacts", GREEN),
            ("head-0", "Head 0", BLUE),
            ("head-1", "Head 1", BLUE),
            ("head-2", "Head 2", BLUE),
        ]
    ):
        r = primary[name]
        axes[1].plot([r["timing_mae_ms"], r["timing_p95_ms"]], [i, i], color=col)
        axes[1].plot(r["timing_mae_ms"], i, "o", color=col)
        axes[1].plot(r["timing_p95_ms"], i, "|", color=col, ms=10)
    axes[1].set(
        yticks=range(5),
        yticklabels=["Four substeps", "Two impacts", "Head 0", "Head 1", "Head 2"],
        xlabel="Matched timing error (ms)",
        title="(b) Mean and 95th percentile",
    )
    axes[1].invert_yaxis()
    for i, (k, title) in enumerate([("enriched", "Ramp begins"), ("ordinary", "Ordinary begins")]):
        for j, r in enumerate(s["e2"][k]):
            axes[2].plot(
                [0, 1],
                [r["raw"]["bce"], r["calibrated"]["bce"]],
                ls=["-", "--", ":"][j],
                color=[BLUE, GREEN][i],
                marker=["o", "s"][i],
                label=title if j == 0 else None,
            )
    axes[2].set(
        xlim=(-0.15, 1.15),
        xticks=[0, 1],
        xticklabels=["Frozen E2", "Calibrated"],
        yscale="log",
        ylabel="Log loss (nats)",
        title="(c) Held-out calibration",
    )
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.32)
    save(fig, root, "continuation-representation")


def decisions(s, root):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), layout="constrained")
    for k, ls in [("ordinary", "-"), ("enriched", "--")]:
        rec = s["selection"][k]
        for solver, c in [("bounded", GREEN), ("four", GREY)]:
            x = np.sort(rec["latency_parents"][solver]) * 1000
            axes[0].step(
                x,
                np.arange(1, len(x) + 1) / len(x),
                where="post",
                color=c,
                ls=ls,
                label=f"{'Two impacts' if solver == 'bounded' else 'Four substeps'}, {k}",
            )
    axes[0].set(
        xscale="log",
        xlabel="Cold eight-goal forecast table (ms)",
        ylabel="Fraction of parents",
        title="(a) Cost across 257 edits",
    )
    axes[0].legend(fontsize=9, loc="center", bbox_to_anchor=(0.64, 0.45))
    ax = axes[1]
    rec = s["selection"]["enriched"]
    for name, col, ls in [
        ("analytic", GREY, "-"),
        ("head-0", BLUE, "-"),
        ("head-1", ORANGE, "--"),
        ("head-2", GREEN, ":"),
    ]:
        curve = rec[name]["curve"]
        ax.step(
            [v["proposed"] for v in curve],
            [v["failures"] for v in curve],
            color=col,
            ls=ls,
            where="post",
            label=name.capitalize().replace("-", " "),
            alpha=0.9,
        )
        ax.plot(rec[name]["proposed"], rec[name]["failures"], marker="o", ms=5, mfc="none", mec=col)
    ax.set(
        xlabel="Accepted proposals",
        ylabel="Proposals failing any of five checks",
        title="(b) Selected edits on competing surfaces",
    )
    ax.legend(fontsize=9, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.3)
    save(fig, root, "continuation-decisions")


def example(d, root):
    if d.get("found") is False:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.65), layout="constrained")
    ax = axes[0]
    grid = np.array(d["grid"]) * 1000
    for k, label, col, ls in [
        ("coarse", "1920 Hz", GREEN, "-"),
        ("fine", "3840 Hz", GREY, ":"),
        ("predicted", "Two-impact forecast", BLUE, "--"),
    ]:
        ax.plot(grid, np.array(d[k], float), color=col, ls=ls, label=label, lw=1.5)
    comp = d["components"][0]
    ax.axvspan(comp["low_m"] * 1000, comp["high_m"] * 1000, color=GREEN, alpha=0.065, lw=0)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(d["selected"] * 1000, color=ORANGE, lw=1)
    ax.plot(d["selected"] * 1000, d["coarse"][d["index"]], "o", color=ORANGE, label="Proposed shift")
    ax.set(xlabel="Bumper shift (mm)", ylabel="Reach margin (m)", title="(a) A measured edit range")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.52))
    scene = d["scene"]
    goal = d["goal"]["region"]
    ax = axes[1]
    ax.add_patch(Circle(goal["center"], goal["radius"], fc=ORANGE, ec=ORANGE, alpha=0.22))
    for seg in scene["segments"]:
        pts = np.array([seg["a"], seg["b"]])
        ax.plot(*pts.T, color=GREY, lw=3)
    for i, b in enumerate(d["branches"]):
        path = np.array(b["path"])
        ax.plot(*path.T, color=[GREEN, RUST][i], lw=1.6, label=f"{b['edit'] * 1000:.2f} mm")
    bumper = scene["bumper"]
    ax.add_patch(
        Circle(
            [bumper["position"][0] + d["branches"][0]["edit"], bumper["position"][1]],
            bumper["radius"],
            fc="#ece9df",
            ec=GREY,
        )
    )
    ax.set(
        xlim=(0, 6),
        ylim=(0, 4),
        aspect="equal",
        xlabel="Horizontal position (m)",
        ylabel="Height (m)",
        title="(b) Neighboring grid edits",
    )
    ax.legend(fontsize=9)
    ax = axes[2]
    for i, b in enumerate(d["branches"]):
        for k, hz in enumerate(["1920", "3840"]):
            y = 3 - 2 * i - k
            for obj, t in b["audits"][hz]["events"]:
                ax.plot(t, y, "o" if obj == "ramp" else "s", color=GREEN if obj == "ramp" else RUST, ms=6)
    ax.plot([], [], "o", color=GREEN, label="Ramp")
    ax.plot([], [], "s", color=RUST, label="Bumper")
    ax.set(
        yticks=[3, 2, 1, 0],
        yticklabels=[
            f"{b['edit'] * 1000:.2f} mm · {hz} Hz" for b in d["branches"] for hz in ["1920", "3840"]
        ],
        xlabel="Begin after interval start (ms)",
        title="(c) The additional contact",
        ylim=(-0.5, 3.9),
    )
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3)
    save(fig, root, "continuation-boundary")


def draw(root):
    style()
    s = read(root / "summary.json")
    representation(s, root)
    decisions(s, root)
    example(read(root / "example.json"), root)
    return {"figures": ["continuation-representation", "continuation-decisions", "continuation-boundary"]}


if __name__ == "__main__":
    summarize(REPORT)
    print(draw(REPORT))
