"""Paper figures drawn from saved support and relay observations."""

import json

import numpy as np

from .figures import BLUE, GREEN, GREY, ORANGE, RUST, Circle, plt, save, style


def load(folder, name):
    return json.loads((folder / f"{name}.json").read_text())


def path_difference(a, b):
    return np.linalg.norm(np.array(a) - b, axis=-1) * 1000


def support_figures(folder):
    style()
    enriched, ordinary, fixtures = [
        load(folder, k) for k in ("evaluation", "ordinary_evaluation", "fixtures")
    ]
    restarts = load(folder, "restarts")["rows"]
    case = next(r for r in enriched["rows"] if "trace" in r)
    reset = next(
        r
        for r in restarts
        if (r["parent"], r["edit"]) == (case["parent"], case["edit"])
    )
    trace = case["trace"]
    time = np.arange(121) / 60
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6), layout="constrained")
    ax, bx, cx = axes
    for hz, color, ls in [("3840", GREEN, "-"), ("1920", BLUE, "--")]:
        ax.plot(
            time,
            path_difference(trace[hz], trace["bounded"]),
            ls=ls,
            color=color,
            label=f"{hz} Hz reference",
        )
        curve = reset["rates"][hz]["peak_error_by_restart_m"]
        bx.plot(np.arange(len(curve)) / 60, np.array(curve) * 1000, ls=ls, color=color)
    ax.plot(
        time,
        path_difference(trace["1920"], trace["3840"]),
        color=GREY,
        lw=1.2,
        ls=":",
        label="Between references",
    )
    first = reset["joint_crossing_observation"]
    ax.axvspan(
        first / 60,
        (first + 5) / 60,
        color=ORANGE,
        alpha=0.26,
        label="Six-sample window",
    )
    for panel in [ax, bx]:
        panel.axhline(25, color=GREY, ls=":", lw=1)
    t = reset["localization_interval"] / 60
    bx.axvspan(t, t + 1 / 60, color=ORANGE, alpha=0.38)
    bx.annotate(
        "Bumper impact",
        xy=(t + 1 / 120, 39),
        xytext=(0.46, 47),
        arrowprops={"arrowstyle": "-", "color": GREY},
        fontsize=11,
    )
    ax.set(
        title="(a) When the paths diverge",
        xlabel="Time since release (s)",
        ylabel="Position discrepancy (mm)",
        xlim=(0, 2),
        ylim=(0, None),
    )
    ax.legend(fontsize=9, loc="upper left")
    bx.set(
        title="(b) Where a restart helps",
        xlabel="Time of exact-state restart (s)",
        ylabel=f"Peak error at {first / 60:.2f}–{(first + 5) / 60:.2f} s (mm)",
        ylim=(0, None),
    )
    for flag, color, marker, label in [
        (True, GREY, "x", "Rate-sensitive"),
        (False, BLUE, "o", "Rate-stable"),
    ]:
        rows = [
            r
            for r in restarts
            if r["split"] == "evaluation"
            and r["rate_sensitive"] == flag
            and r["localization_interval"] is not None
        ]
        cx.scatter(
            [(r["localization_interval"] + 1) / 60 for r in rows],
            [r["joint_crossing_observation"] / 60 for r in rows],
            s=15,
            alpha=0.66,
            marker=marker,
            color=color,
            label=f"{label} ({len(rows)})",
            linewidths=0.8,
        )
    cx.plot([0, 2], [0, 2], ls=":", color=GREY, lw=1)
    cx.set(
        title="(c) Error can emerge much later",
        xlabel="End of localized interval (s)",
        ylabel="First lasting divergence (s)",
        xlim=(0, 2),
        ylim=(0, 2),
    )
    cx.legend(loc="lower right", fontsize=10)
    for ax in axes:
        ax.grid(True)
        ax.set_axisbelow(True)
    save(fig, folder, "divergence")

    fig, (ax, bx, cx) = plt.subplots(
        1,
        3,
        figsize=(13.4, 3.6),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.12, 1, 0.95]},
    )
    rows = fixtures["rows"]
    names = [
        "Rest on floor",
        "Roll on incline",
        "Leave platform",
        "Lift off",
        "Roll into bumper",
        "Repeated bounce",
    ]
    for k, color, mark, shift, label in [
        ("bounded", BLUE, "o", -0.17, "Bounded"),
        ("support", GREEN, "s", 0, "Support"),
        ("four", ORANGE, "^", 0.17, "Four substeps"),
    ]:
        ax.scatter(
            [1000 * r["modes"][k]["1920"]["rmse_m"] for r in rows],
            np.arange(len(rows)) + shift,
            color=color,
            marker=mark,
            s=24,
            label=label,
            zorder=3,
        )
    ax.set(
        xscale="log",
        yticks=np.arange(len(rows)),
        yticklabels=names,
        xlabel="Two-second position RMSE (mm)",
        title="(a) Isolating the constraint",
        ylim=(len(rows) - 0.5, -1.25),
        xticks=[0.3, 1, 3, 10, 30, 100],
    )
    ax.set_xticklabels(["0.3", "1", "3", "10", "30", "100"])
    ax.legend(
        ncol=3, fontsize=8.5, loc="upper center", columnspacing=0.7, handletextpad=0.2
    )
    trace = next(r for r in rows if r["scene_id"] == "leave_platform")["trace"]
    for k, color, ls, label in [
        ("bounded", BLUE, "--", "Bounded"),
        ("support", GREEN, "-", "Support"),
        ("four", ORANGE, ":", "Four substeps"),
    ]:
        bx.plot(
            time,
            path_difference(trace[k], trace["1920"]),
            color=color,
            ls=ls,
            label=label,
        )
    iv = np.asarray(trace["held_intervals"]) / 60
    bx.vlines(
        iv, 0, 0.032, transform=bx.get_xaxis_transform(), color=GREEN, linewidth=1.5
    )
    bx.text(
        0.07, 0.14, "Support active", transform=bx.transAxes, color=GREEN, fontsize=10
    )
    bx.set(
        title="(b) Leaving a platform",
        xlabel="Time since release (s)",
        ylabel="Position error (mm)",
        xlim=(0, 2),
        ylim=(0, None),
    )
    bx.legend(loc="upper left", fontsize=9)
    cx.axvline(0, color=GREY, lw=1, ls=":")
    yloc = [3, 2, 1, 0]
    for y, (data, hz) in zip(
        yloc,
        [
            (enriched, "1920"),
            (enriched, "3840"),
            (ordinary, "1920"),
            (ordinary, "3840"),
        ],
    ):
        stat = data["comparison"][hz]
        lo, hi = 1000 * np.asarray(stat["ci95_m"])
        val = 1000 * stat["support_minus_bounded_m"]
        cx.errorbar(
            val,
            y,
            xerr=[[val - lo], [hi - val]],
            fmt="o" if hz == "1920" else "s",
            color=RUST if y >= 2 else BLUE,
            capsize=3,
        )
    cx.set(
        yticks=yloc,
        yticklabels=[
            "Competing · 1920",
            "Competing · 3840",
            "Ordinary · 1920",
            "Ordinary · 3840",
        ],
        title="(c) Full-release cost in accuracy",
        xlabel="Support − bounded RMSE (mm)",
        xlim=(-0.55, 4.2),
        ylim=(-0.7, 3.7),
    )
    for panel in [ax, bx, cx]:
        panel.grid(True)
        panel.set_axisbelow(True)
    save(fig, folder, "persistent-support")
    return {"figures": ["divergence", "persistent-support"]}


def relay_figure(folder):
    style()
    rows = load(folder, "relay")["rows"]
    row = next(r for r in rows if r["witness"])
    case = row["witness"]
    check = case["checks"]["1920"]
    path = np.array(check["positions"])
    orig = np.array(check["original_positions"])
    off = np.array(check["disabled_positions"])
    scene = row["scene"]
    receiver = row["receiver"]
    goal_x = receiver["position"][0] + row["direction"] * row["target_travel_m"]
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.4, 3.6), layout="constrained")
    bumper = scene["bumper"]
    p = bumper["position"]
    ax.add_patch(Circle(p, bumper["radius"], fill=False, edgecolor=GREY, ls=":"))
    ax.add_patch(
        Circle(
            [p[0] + case["edit_m"], p[1]],
            bumper["radius"],
            facecolor="#e3e5e5",
            edgecolor=GREY,
        )
    )
    ax.axhline(0.22, color=GREY, lw=1)
    for i, color, label, spec in [
        (0, BLUE, "Striker", scene["ball"]),
        (1, GREEN, "Receiver", receiver),
    ]:
        ax.plot(*path[:, i].T, color=color, label=label)
        ax.add_patch(
            Circle(
                path[0, i], spec["radius"], facecolor="white", edgecolor=color, lw=1.5
            )
        )
    ix = round(check["transfer_tick"] / 1920 * 60)
    ax.scatter(*path[ix, 0], color=ORANGE, s=24, zorder=3, label="First transfer")
    ax.vlines(goal_x, 0.22, 0.85, color=GREY, lw=1, ls="--")
    ax.set(
        xlabel="Horizontal position (m)",
        ylabel="Height (m)",
        title="(a) Edit one body, move another",
        aspect="equal",
        ylim=(0, 3.08),
    )
    ax.legend(fontsize=9, loc="upper left")
    for k, ls, color, label in [
        (path, "-", GREEN, "Edited bumper"),
        (orig, "--", BLUE, "Original bumper"),
        (off, ":", GREY, "Pair collision disabled"),
    ]:
        bx.plot(
            np.arange(len(k)) / 60,
            row["direction"] * (k[:, 1, 0] - receiver["position"][0]),
            color=color,
            ls=ls,
            label=label,
        )
    bx.axhline(row["target_travel_m"], color=ORANGE, lw=1, ls="--")
    bx.axvline(check["transfer_tick"] / 1920, color=GREY, lw=1, ls=":")
    bx.set(
        title="(b) When the receiver responds",
        xlabel="Time since release (s)",
        ylabel="Receiver travel toward goal (m)",
        xlim=(0, 2),
    )
    bx.legend(loc="center left", fontsize=9)
    grid = 1000 * np.asarray(row["edits_m"])
    for hz, ls, color, marker in [("1920", "-", BLUE, "o"), ("3840", "--", GREEN, ".")]:
        cx.plot(
            grid,
            [v["travel_m"] for v in row["rates"][hz]],
            color=color,
            ls=ls,
            marker=marker,
            ms=3,
            label=f"{hz} Hz",
        )
    cx.axhline(
        row["target_travel_m"], color=ORANGE, lw=1, ls="--", label="Reach threshold"
    )
    i = case["index"]
    cx.scatter(
        grid[i],
        row["rates"]["1920"][i]["travel_m"],
        s=58,
        edgecolor=RUST,
        facecolor="none",
        zorder=4,
    )
    cx.set(
        title="(c) Which shifts transfer enough?",
        xlabel="Bumper shift (mm)",
        ylabel="Maximum receiver travel (m)",
    )
    cx.legend(fontsize=9, loc="upper left")
    for panel in [ax, bx, cx]:
        panel.grid(True)
        panel.set_axisbelow(True)
    save(fig, folder, "relay")
    if (folder / "mass.json").exists():
        mass_figure(folder)
        return {"figures": ["relay", "receiver-mass"]}
    return {"figures": ["relay"]}


def mass_figure(folder):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    rows = load(folder, "mass")["rows"]
    cases = load(folder, "relay")["rows"]
    targets = {r["parent"]: r["target_travel_m"] for r in cases}
    row = next(
        r
        for r in rows
        if r["common_two_rate_edit_m"] is not None
        and r["arms"][0]["smallest_two_rate_edit_m"]
        != r["arms"][1]["smallest_two_rate_edit_m"]
    )
    grid = 1000 * np.array(row["edits_m"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.1, 3.8), layout="constrained")
    for arm, color, ls in zip(row["arms"], [BLUE, GREEN], ["-", "--"]):
        y = np.array(arm["travel_m"]["1920"])
        ax.plot(
            grid,
            y,
            ls=ls,
            color=color,
            marker=".",
            ms=4,
            label=f"{arm['mass_kg']:.2f} kg",
        )
        i = row["edits_m"].index(arm["smallest_two_rate_edit_m"])
        ax.scatter(
            grid[i], y[i], s=64, edgecolor=color, facecolor="white", lw=1.5, zorder=3
        )
    ax.axhline(
        targets[row["parent"]], color=ORANGE, ls=":", lw=1.4, label="Reach threshold"
    )
    ax.grid(True)
    ax.legend(fontsize=10, loc="upper left")
    ax.set(
        title="(a) Same release, different receiver mass",
        xlabel="Bumper shift (mm)",
        ylabel="Maximum receiver travel (m)",
    )
    cells = []
    for r in rows:
        a, b = [
            np.array(list(arm["travel_m"].values())) >= targets[r["parent"]]
            for arm in r["arms"]
        ]
        stable = (a[0] == a[1]) & (b[0] == b[1])
        cells.append(np.where(stable, a[0].astype(int) + 2 * b[0].astype(int), 4))
    colors = ["#eeefef", ORANGE, RUST, GREEN, "#676978"]
    spacing = grid[1] - grid[0]
    bx.imshow(
        cells,
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=4,
        aspect="auto",
        interpolation="nearest",
        extent=[grid[0] - spacing / 2, grid[-1] + spacing / 2, len(rows) + 0.5, 0.5],
    )
    bx.set(
        title="(b) Where mass changes the answer",
        xlabel="Bumper shift (mm)",
        ylabel="Pilot parent (generation order)",
        yticks=[1, 6, 12, 18, 24],
    )
    labels = ["Neither", "Light only", "Heavy only", "Both", "Rate-sensitive"]
    present = np.unique(cells)
    bx.legend(
        handles=[Patch(facecolor=colors[i], label=labels[i]) for i in present],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        fontsize=9.5,
        ncol=4,
        handlelength=1,
        columnspacing=0.9,
    )
    save(fig, folder, "receiver-mass")
