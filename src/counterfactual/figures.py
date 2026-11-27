"""Publication-oriented static figures, with adjacent machine-readable source data."""
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/.cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("artifacts/.cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BLUE, ORANGE, GREEN, RUST, GREY = "#4d79a7", "#d39845", "#559c88", "#bf6c54", "#7b818b"


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .85, "lines.linewidth": 1.8, "lines.markersize": 4,
                         "grid.color": "#d9d9d9", "grid.alpha": .5, "grid.linewidth": .6,
                         "legend.frameon": False, "savefig.dpi": 200, "svg.fonttype": "none"})


def save(fig, output, name):
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(output/f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scene_axes(ax, entry):
    s, g = entry["scene"], entry["goal"]
    b = s["bumper"]
    ax.add_patch(Circle(b["position"], b["radius"], facecolor="#e3e6e6", edgecolor=GREY, lw=1))
    ax.add_patch(Circle((b["position"][0]+entry["edit_m"], b["position"][1]), b["radius"], fill=False, edgecolor=GREEN, lw=1.5))
    for seg in s["segments"]:
        ax.plot([seg["a"][0], seg["b"][0]], [seg["a"][1], seg["b"][1]], color=GREY, lw=3)
    r = g["region"]
    if r["kind"] == "disk":
        ax.add_patch(Circle(r["center"], r["radius"], facecolor=ORANGE, edgecolor=ORANGE, alpha=.17))
    ax.set(xlabel="Horizontal position (m)", ylabel="Height (m)", xlim=(0, 6), ylim=(0, 4))
    ax.set_aspect("equal")
    ax.grid(True)


def build_figures(output: Path):
    style()
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((output/"evaluation.json").read_text())
    training = json.loads((output/"training.json").read_text())
    gallery = json.loads((output/"gallery.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5), layout="constrained")
    h = training["history"]
    axes[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], color=BLUE, label="Training")
    axes[0].plot([r["epoch"] for r in h], [r["validation_loss"] for r in h], color=GREEN, label="Validation")
    axes[0].set(xlabel="Training epoch", ylabel="Supervised loss", title="(a) Learning the future", yscale="log")
    axes[0].legend()
    keys = ["random_success_solvable", "random_search_success_solvable", "verified_success_solvable"]
    labels = ["Random\n1 call", "Random search\n16 calls", "E0 learned\n≤1 call"]
    for i, (key, color) in enumerate(zip(keys, [GREY, ORANGE, GREEN])):
        m = report["metrics"][key]
        if m["estimate"] is not None:
            lo, hi = m["ci95"]
            axes[1].errorbar(i, m["estimate"], yerr=[[max(0, m["estimate"]-lo)], [max(0, hi-m["estimate"])]], fmt="o", color=color, capsize=3)
    axes[1].set(xticks=range(3), xticklabels=labels, ylabel="Verified intervention success", ylim=(-.03, 1.03), title="(b) Decisions, with different budgets")
    axes[1].axhline(1, color=GREY, ls=":", lw=1, label="Matched-grid oracle")
    axes[1].legend(loc="lower left", fontsize=10)
    tasks = [r for r in report["tasks_detail"] if not r["initial_success"] and r["selected"]]
    for ok, color, label in [(False, RUST, "Verification failed"), (True, GREEN, "Verified success")]:
        rows = [r for r in tasks if r["verified_success"] == ok]
        axes[2].scatter([r["predicted_margin_m"] for r in rows], [r["verified_margin_m"] for r in rows], c=color, s=21, alpha=.7, label=label)
    axes[2].axhline(0, color=GREY, lw=1)
    axes[2].axvline(report["safety_margin_m"], color=GREY, ls=":", lw=1)
    axes[2].set(xlabel="Predicted goal margin (m)", ylabel="Verified goal margin (m)", title="(c) Predicted versus verified margins")
    if tasks:
        axes[2].legend(fontsize=10)
    for ax in axes:
        ax.grid(True)
    save(fig, output, "pilot-summary")
    # A fixed selection rule is used in evaluation; success and failure both get space.
    for entry in gallery:
        fig, axes = plt.subplots(1, 2, figsize=(11.3, 3.6), layout="constrained")
        scene_axes(axes[0], entry)
        for key, color, ls, label in [("original", BLUE, "-", "Original · simulator"), ("edited", GREEN, "-", "Edited · simulator")]:
            p = np.array(entry[key]["states"])
            axes[0].plot(p[:, 0], p[:, 1], color=color, ls=ls, label=label)
            ids = entry[key]["object_ids"]
            for event in entry[key]["events"]:
                if event["phase"] == "begin":
                    i = min(120, round(event["time"]*60))
                    axes[0].plot(p[i, 0], p[i, 1], "o", color=color, ms=3)
        p = np.array(entry["predicted_edited"])
        axes[0].plot(p[:, 0], p[:, 1], "--", color=GREEN, alpha=.65, label="Edited · E0 prediction")
        axes[0].set_title("(a) "+entry["category"].replace("_", " ").capitalize())
        axes[0].legend(fontsize=9, loc="upper right")
        u = np.array(entry["grid_m"])/entry["scene"]["ball"]["radius"]
        actual = np.array([np.nan if v is None else v for v in entry["actual_margins_m"]])
        predicted = np.array([np.nan if v is None else v for v in entry["predicted_margins_m"]])
        axes[1].plot(u, actual, color=GREEN, label="Simulator margin")
        axes[1].plot(u, predicted, "--", color=BLUE, label="E0 margin")
        axes[1].axhline(0, color=GREY, lw=1)
        axes[1].axhline(.03, color=GREY, ls=":", lw=1, label="Selection threshold")
        axes[1].axvline(entry["edit_m"]/entry["scene"]["ball"]["radius"], color=RUST, ls=":", lw=1)
        axes[1].fill_between(u, 0, .035, where=np.array(entry["bumper_contact"]), transform=axes[1].get_xaxis_transform(), color=GREY, alpha=.4, label="Bumper contact")
        axes[1].set(xlabel="Bumper displacement / ball radius", ylabel="Goal margin (m)", title="(b) Small edits, changing outcomes", xlim=(-1, 1))
        axes[1].grid(True)
        axes[1].legend(fontsize=9)
        save(fig, output, "landscape-"+entry["category"])
    # Audit trajectory error and goal labels separately: a label gate is not convergence.
    audits = []
    for file in sorted(output.parent.glob("audit-*/audit.json")):
        audits.append(json.loads(file.read_text()))
    if (output/"audit.json").exists():
        audits.append(json.loads((output/"audit.json").read_text()))
    audits = list({a["summary"]["physics_hz"]: a for a in audits}.values())
    audits.sort(key=lambda a: a["summary"]["physics_hz"])
    if audits:
        fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.4), layout="constrained")
        hz = [a["summary"]["physics_hz"] for a in audits]
        axes[0].plot(hz, [100*a["summary"]["resolution_goal_disagreement"] for a in audits], "o-", color=RUST, label="Step halving")
        axes[0].plot(hz, [100*a["summary"]["sampling_goal_disagreement"] for a in audits], "s-", color=BLUE, label="60 Hz path sampling")
        axes[0].axhline(1, color=GREY, ls=":", lw=1, label="Resolution gate")
        axes[0].set(xlabel="Physics frequency (Hz)", ylabel="Goal-label disagreement (%)", title="(a) Establish the numerical reference", xscale="log", xticks=hz, xticklabels=hz)
        axes[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[0].legend(fontsize=10)
        axes[1].boxplot([[r["resolution_rmse_m"] for r in a["branches"]] for a in audits], tick_labels=hz, showfliers=True, flierprops={"marker": ".", "markersize": 3, "markeredgecolor": GREY})
        axes[1].set(xlabel="Physics frequency (Hz)", ylabel="Step-halving path RMSE (m)", title="(b) A stable label can hide path drift", yscale="log")
        for ax in axes:
            ax.grid(True)
        save(fig, output, "numerical-audit")
    physical_file = output/"physical-diagnostics.json"
    if physical_file.exists():
        diagnostics = json.loads(physical_file.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.4), layout="constrained")
        t = diagnostics["times_s"]
        rng = np.random.default_rng(20261108)
        for key, color, label in [("position_error_by_parent_radii", BLUE, "Future position"), ("difference_error_by_parent_radii", GREEN, "Intervention difference")]:
            values = np.array(diagnostics[key])
            boots = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
            lo, hi = np.quantile(boots, [.025, .975], axis=0)
            axes[0].plot(t, values.mean(0), color=color, label=label)
            axes[0].fill_between(t, lo, hi, color=color, alpha=.12, linewidth=0)
        axes[0].set(xlabel="Forecast horizon (s)", ylabel="Mean error (ball radii)", title="(a) Where predictions drift")
        axes[0].legend(fontsize=10)
        rows = diagnostics["parent_metrics"]
        axes[1].scatter([r["position_error_mean_radii"] for r in rows], [r["difference_error_mean_radii"] for r in rows], s=22, color=GREEN)
        axes[1].set(xlabel="Future position error (ball radii)", ylabel="Intervention-difference error (ball radii)", title="(b) Physical fit versus edit sensitivity")
        for ax in axes:
            ax.grid(True)
        save(fig, output, "physical-diagnostics")
    return [str(p) for p in output.glob("*.png")]
