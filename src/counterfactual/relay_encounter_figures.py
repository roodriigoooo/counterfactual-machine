"""Draw the upstream-state experiment and its probabilistic and computational consequences."""

import numpy as np

from .figures import BLUE, GREEN, GREY, ORANGE, RUST, plt, save, style
from .relay_encounter import load


def panels():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.65), layout="constrained")
    for ax in axes:
        ax.grid(True)
        ax.set_axisbelow(True)
    return fig, axes


def causes(root):
    data = load(root / "diagnosis.json")
    cases = [r for r in data["audited"] if r["outcome"] in ("invented", "missed")]
    example = next(r["rates"] for r in cases if "example" in r["rates"]["1920"])
    fig, (ax, bx, cx) = panels()
    for hz, field, label, color, ls in [
        ("1920", "forecast_gap_mm", "480 Hz forecast", RUST, "--"),
        ("1920", "actual_gap_mm", "1920 Hz reference", BLUE, "-"),
        ("3840", "actual_gap_mm", "3840 Hz reference", GREEN, ":"),
    ]:
        record = example[hz]["example"]
        t = np.array(record["time_s"])
        m = (t >= 1.22) & (t <= 1.37)
        ax.plot(t[m], np.array(record[field])[m], ls=ls, color=color, label=label)
    ax.axhline(0, color=GREY, lw=1)
    ax.set(
        title="(a) An invented collision",
        xlabel="Time since release (s)",
        ylabel="Pass-through clearance (mm)",
    )
    ax.legend(fontsize=9)
    rates = [r["rates"]["1920"] for r in cases]
    leads = np.array([64, 16, 4, 1])
    for label, field, color, marker, ls in [
        ("Whole striker", "striker", GREEN, "o", "-"),
        ("Striker position", "striker_position", BLUE, "s", "--"),
        ("Striker velocity + spin", "striker_motion", ORANGE, "^", "-."),
        ("Whole receiver", "receiver", GREY, "x", ":"),
    ]:
        y = [
            sum(
                (r["restart"][f"lead-{n}"][field] >= 0) == (r["actual_tick"] >= 0)
                for r in rates
            )
            for n in leads
        ]
        bx.plot(range(len(leads)), y, color=color, marker=marker, ls=ls, label=label)
    bx.set(
        title="(b) Restore one state",
        xlabel="Lead before encounter (ms, approx.)",
        ylabel="Corrected cases (of 20)",
        xticks=range(4),
        xticklabels=["133", "33", "8", "2"],
        ylim=(-1, 22),
    )
    bx.legend(fontsize=9, loc="center right")
    for marker, color, stage, offset in [
        ("o", BLUE, "before", -0.07),
        ("s", GREEN, "after", 0.07),
    ]:
        y = []
        for surface in ("bumper", "shelf"):
            hits = 0
            for r in rates:
                j = next(
                    i
                    for i, (_, name) in enumerate(r["static_begins"])
                    if name == surface
                )
                hits += (r["restart"][f"surface-{j}-{stage}"]["striker"] >= 0) == (
                    r["actual_tick"] >= 0
                )
            y.append(hits)
        xx = np.arange(2) + offset
        cx.scatter(xx, y, color=color, marker=marker, s=38, label=f"Restore {stage}")
        for xp, yp in zip(xx, y, strict=True):
            cx.annotate(
                str(yp),
                (xp, yp),
                xytext=(0, 8 if stage == "after" else -15),
                textcoords="offset points",
                ha="center",
                color=color,
            )
    cx.set(
        title="(c) Trace the earlier bounces",
        xticks=[0, 1],
        xticklabels=["First bumper", "First shelf"],
        ylim=(0, 23),
        xlim=(-0.4, 1.4),
        ylabel="Corrected cases (of 20)",
        xlabel="Striker contact bracket",
    )
    cx.legend(loc="lower right", fontsize=9)
    save(fig, root, "relay-causes")


def uncertainty(root):
    data = load(root / "probability.json")
    p = data["test_parent_start"]
    rows = load(root / "parents" / f"{p}.json")["rows"]
    m = data["models"]["context"]
    x = np.array([r["edit_m"] for r in rows]) * 1000
    loc = np.array(data["predictions"]["context"]["location_m"][0]) * 1000
    halfwidth = (
        np.array(data["predictions"]["context"]["scale_m"][0])
        * m["calibration_q"]
        * 1000
    )
    fig, (ax, bx, cx) = panels()
    ax.errorbar(
        x,
        loc,
        yerr=halfwidth,
        color=GREEN,
        ecolor="#a3c8bd",
        ls="--",
        marker=".",
        capsize=2,
        label="Conditional center and band",
    )
    for h, marker, color in [("1920", "o", BLUE), ("3840", "x", RUST)]:
        y = [1000 * r["reference"][h][0] for r in rows]
        ax.scatter(
            x, y, marker=marker, s=14, color=color, label=f"{h} Hz reference", zorder=3
        )
    ax.axhline(0, color=GREY, lw=1)
    ax.set(
        title="(a) One shelf, 17 edits",
        xlabel="Bumper shift (mm)",
        ylabel="Minimum clearance (mm)",
    )
    ax.legend(fontsize=8)
    modes = ["constant", "clock", "context"]
    xx = np.arange(3)
    for h, color, ls, mark, j in [
        ("1920", BLUE, "-", "o", 0),
        ("3840", RUST, "--", "s", 1),
    ]:
        bx.plot(
            xx,
            [1000 * data["models"][k]["brier"][j] for k in modes],
            color=color,
            ls=ls,
            marker=mark,
            label=f"{h} Hz labels",
        )
    bx.set(
        title="(b) Forecasting contact",
        xticks=xx,
        xticklabels=["Constant", "Clock", "Context"],
        xlabel="Clearance-error model",
        ylabel="Brier score (× 1000)",
    )
    bx.legend(fontsize=9)
    for field, label, col, mark, ls in [
        ("parents_covered", "Entire parent covered", BLUE, "o", "-"),
        ("resolved", "Edit signs resolved", GREEN, "s", "--"),
    ]:
        y = []
        for k in modes:
            b = data["models"][k]["band"]
            y.append(
                100 * b[field] / b["parents" if field == "parents_covered" else "cells"]
            )
        cx.plot(xx, y, color=col, marker=mark, ls=ls, label=label)
    cx.axhline(90, color=GREY, ls=":", lw=1, label="Parent target")
    cx.set(
        title="(c) What the band resolves",
        xticks=xx,
        xticklabels=["Constant", "Clock", "Context"],
        xlabel="Clearance-error model",
        ylabel="Held-out proportion (%)",
        ylim=(60, 95),
    )
    cx.legend(fontsize=9)
    save(fig, root, "relay-uncertainty")


def precision(root):
    confirm = load(root / "handoff-confirmation.json")
    decisions = load(root / "handoff-decisions.json")
    profile = load(root / "clocks.json")["sequential_profile"]
    fig, (ax, bx, cx) = panels()
    names = confirm["methods"]
    xticks = ["480 Hz", "First impact", "1920 Hz"]
    for h, col, marker, ls in [("1920", BLUE, "o", "-"), ("3840", GREEN, "s", "--")]:
        y = np.array(confirm["comparisons"][h]["mean_striker_receiver_mm"])[:, 1]
        ax.plot(range(3), y, ls=ls, color=col, marker=marker, label=f"{h} Hz reference")
    ax.set(
        title="(a) Precision at first impact",
        xticks=range(3),
        xticklabels=xticks,
        ylabel="Receiver path RMSE (mm)",
        xlabel="Forecast",
        ylim=(0, None),
    )
    ax.legend(fontsize=9)
    times = np.array([[r["ms"][k] for k in names] for r in profile])
    for y in times:
        bx.plot(range(3), y, color=GREY, alpha=0.28, lw=0.8)
    bx.plot(
        range(3),
        np.median(times, axis=0),
        color=BLUE,
        marker="o",
        label="Median of eight layouts",
    )
    bx.set(
        title="(b) Cost of a 33-edit grid",
        xticks=range(3),
        xticklabels=xticks,
        ylabel="Wall time (ms)",
        xlabel="Forecast",
        ylim=(0, None),
    )
    bx.legend(fontsize=9)

    def robust(arm):
        return bool(arm["rates"]) and all(min(v) >= 0.35 for v in arm["rates"].values())

    row = next(
        r
        for r in decisions["rows"]
        if robust(r["arms"]["handoff"]) and not robust(r["arms"]["480"])
    )
    for key, col, marker in [("480", RUST, "o"), ("handoff", GREEN, "s")]:
        a = row["arms"][key]
        for hz, ls in [("1920", "-"), ("3840", "--")]:
            cx.plot(
                [-5, -2.5, 0, 2.5, 5],
                1000 * np.array(a["rates"][hz]),
                color=col,
                marker=marker,
                ls=ls,
                label=f"{a['edit_m'] * 1000:g} mm shift" if hz == "1920" else None,
            )
    cx.axhline(350, color=GREY, lw=1, ls=":", label="Travel target")
    cx.set(
        title="(c) One recovered repair",
        xlabel="Offset from chosen shift (mm)",
        ylabel="Verified receiver travel (mm)",
    )
    cx.legend(fontsize=9)
    save(fig, root, "relay-precision")


def draw(root):
    style()
    causes(root)
    uncertainty(root)
    precision(root)
    return {"figures": ["relay-causes", "relay-uncertainty", "relay-precision"]}
