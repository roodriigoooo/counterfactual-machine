"""Parent-paired decision summaries and paper figures from the saved impact and relay records."""

import json

import numpy as np

from .continuation_experiment import write
from .figures import BLUE, GREEN, GREY, ORANGE, RUST, plt, save, style
from .impact_data import ROOT


def load(root, name):
    return json.loads((root / (name + ".json")).read_text())


def rate_difference(records, denominators):
    """Bootstrap a difference of unconditional per-task rates, clustering by parent."""
    n = np.asarray(denominators)
    d = np.asarray(records)
    draws = np.random.default_rng(43).integers(len(n), size=(2000, len(n)))
    denom = n[draws].sum(1)
    boot = d[draws].sum(1)[denom > 0] / denom[denom > 0]
    return [100 * d.sum() / n.sum(), *(100 * np.quantile(boot, [0.025, 0.975])).tolist()]


def passed(r):
    return bool(r["checks"]) and all(all(v) for v in r["checks"].values())


def decision_summary(parents, baseline):
    modes = list(dict.fromkeys(r["mode"] for p in parents for r in p["rows"]))
    result = {}
    for mode in modes:
        all_rows = []
        deltas = []
        sizes = []
        cost = []
        different = 0
        for p in parents:
            a = {r["goal"]: r for r in p["rows"] if r["mode"] == mode}
            b = {r["goal"]: r for r in p["rows"] if r["mode"] == baseline}
            assert a.keys() == b.keys()
            deltas.append(sum(int(passed(r)) - int(passed(b[j])) for j, r in a.items()))
            sizes.append(len(a))
            all_rows.extend(a.values())
            different += sum(passed(r) != passed(b[j]) for j, r in a.items())
            cost.extend(
                1000 * (abs(r["edit_m"]) - abs(b[j]["edit_m"]))
                for j, r in a.items()
                if r["checks"] and b[j]["checks"]
            )
        offers = [r for r in all_rows if r["checks"]]
        result[mode] = {
            "failed_original": len(all_rows),
            "offered": len(offers),
            "center": sum(r["checks"]["1920"][2] for r in offers),
            "all_five_coarse": sum(all(r["checks"]["1920"]) for r in offers),
            "all_ten": sum(passed(r) for r in offers),
            "robust_flips_vs_baseline": different,
            "paired_gain_pp_and_ci": rate_difference(deltas, sizes),
            "joint_proposal_delta_cost_mm": float(np.mean(cost)) if cost else None,
        }
    return result


def summary(root):
    totals = {}
    for family in ("ordinary", "competing"):
        for name, ref in [("decisions", "continuous-fixed"), ("ordered-decisions", "tick-both")]:
            totals[f"{name}-{family}"] = decision_summary(
                load(root, f"{name}-{family}")["parents"], ref
            )
    relay = {}
    for fam in ("ordinary", "elevated"):
        rows = load(root, f"relay-{fam}")["rows"]
        record = {}
        for mode in rows[0]["modes"]:
            ms = [r["modes"][mode] for r in rows]
            offered = [v for v in ms if v["edit"] is not None]
            informative = [
                abs(r["modes"][mode]["mass_kg"] - r["mass_kg"])
                for r in rows
                if r["probe_delta_m_s"][1] > 0
            ]
            info = {
                "informative_mass_mae_kg": float(np.mean(informative)),
                "n_informative": len(informative),
                "failing": sum(r["original_failed"] for r in rows),
                "offered": len(offered),
                "center": sum(v["rates"]["1920"]["checks"][2] for v in offered),
                "all_ten": sum(all(all(a["checks"]) for a in v["rates"].values()) for v in offered),
            }
            for hz in ("1920", "3840"):
                z = [m["rates"][hz] for m in ms]
                err = np.array([v["path_rmse_mm"] for v in z])
                dif = err[:, 1] - [
                    r["modes"]["probe-ratio"]["rates"][hz]["path_rmse_mm"][1] for r in rows
                ]
                ci = np.random.default_rng(41).choice(dif, (2000, len(rows))).mean(1)
                info[hz] = {
                    "path_mm": err.mean(0).tolist(),
                    "receiver_minus_ratio_mm": float(dif.mean()),
                    "ci95_mm": np.quantile(ci, [0.025, 0.975]).tolist(),
                    "correct": sum(v["scoring_correct"] for v in z),
                    "invented": sum(v["invented_transfer"] for v in z),
                    "missed": sum(v["missed_transfer"] for v in z),
                    "matched": sum(v["matched"] for v in z),
                    "matched_timing_mae_ms": float(np.mean([t for v in z for t in v["timing_abs_ms"]])),
                    "max_true_prefix_mm": max(v["prefix_before_true_transfer_mm"] for v in z),
                    "max_own_prefix_mm": max(v["prefix_before_predicted_transfer_mm"] for v in z),
                }
            record[mode] = info
        relay[fam] = record
    write(root / "summary.json", {"decisions": totals, "relay": relay})
    write(root / "report-kind.json", {"kind": "impact-decoder-v1"})


def draw(root):
    style()
    primary = load(root, "results")["groups"]
    numerics = load(root, "restitution")["rates"]
    pairs = load(root, "matched-contacts")["pairs"]
    report = load(root, "summary")
    colors = {"continuous": BLUE, "tick": GREEN, "joint": RUST, "direct": ORANGE}
    labels = {"continuous": "Continuous", "tick": "Tick", "joint": "Joint", "direct": "Direct"}
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.5, 3.55), layout="constrained")
    for name in ("continuous", "tick", "joint"):
        v = sorted(
            abs(p["localized"]["rates"]["1920"]["models"][name]["normal_displacement_error_mm"])
            for p in pairs
        )
        ax.step(
            np.maximum(v, 0.0001),
            np.arange(1, len(v) + 1) / len(v),
            color=colors[name],
            where="post",
            label=labels[name],
        )
    ax.set(
        title="(a) Leaving the contact",
        xlabel="Absolute normal position error (mm)",
        ylabel="Fraction of localized contacts",
        xscale="log",
        xlim=(0.0001, 20),
    )
    ax.legend(fontsize=10, loc="lower right")
    names = list(colors)
    for idx, (hz, col, marker) in enumerate([("1920", BLUE, "o"), ("3840", GREEN, "s")]):
        vals = [primary["1920"][name]["mean_mm"]["competing"][idx] for name in names]
        bx.plot(
            range(4),
            vals,
            color=col,
            marker=marker,
            ls="-" if idx == 0 else "--",
            label=f"{hz} Hz reference",
        )
    bx.set(
        title="(b) One forecast, two references",
        ylabel="Parent-mean path RMSE (mm)",
        xticks=range(4),
        xticklabels=[labels[n] for n in names],
        ylim=(0, None),
    )
    bx.legend(fontsize=10)
    for i, hz in enumerate(("1920", "3840")):
        v = numerics[hz]["versus_ordered"]
        for j, k in enumerate(["tick", "joint", "direct"]):
            mean = v[k]["competing"]["mean_mm"][i]
            lo, hi = v[k]["competing"]["ci95_mm"][i]
            cx.errorbar(
                j + (i - 0.5) * 0.13,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt=["o", "s"][i],
                color=[BLUE, GREEN][i],
                capsize=3,
                label=(hz + " Hz") if j == 0 else None,
            )
    cx.axhline(0, lw=1, ls=":", color=GREY)
    cx.set(
        title="(c) Accounting for gravity order",
        ylabel="Excess RMSE over ordered rebound (mm)",
        xticks=range(3),
        xticklabels=["Tick", "Joint", "Direct"],
    )
    cx.legend(fontsize=10, loc="upper center")
    for panel in [ax, bx, cx]:
        panel.grid(True)
        panel.set_axisbelow(True)
    save(fig, root, "impact-clocks")

    example = load(root, "gallery")
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.5, 3.6), layout="constrained")
    ds = report["decisions"]["decisions-competing"]
    order = ["continuous-fixed", "continuous-both", "tick-both", "joint-both", "direct-both"]
    for j, k in enumerate(order):
        ax.plot([ds[k]["all_ten"], ds[k]["offered"]], [j, j], color=GREY, lw=1.8)
        ax.scatter(ds[k]["offered"], j, marker="s", color=ORANGE, label="Proposed" if j == 0 else None)
        ax.scatter(ds[k]["all_ten"], j, color=GREEN, label="Pass all ten" if j == 0 else None)
    n_tasks = ds["tick-both"]["failed_original"]
    ax.set(
        yticks=range(5),
        yticklabels=[
            "Continuous · one rate",
            "Continuous · both",
            "Tick · both",
            "Joint · both",
            "Direct · both",
        ],
        ylim=(5.1, -0.6),
        xlabel=f"Edits on {n_tasks} failing goals",
        title="(a) What reaches verification",
    )
    ax.legend(fontsize=10, loc="lower left", ncol=2)
    gx = 1000 * np.array(example["grid_m"])
    u = 1000 * example["edits"]["tick"]
    other = 1000 * example["edits"]["continuous"]
    for hz, color, ls in [("1920", BLUE, "-"), ("3840", GREEN, "--")]:
        y = np.asarray(example["margins"]["reference-" + hz], float) * 1000
        mask = (gx >= min(u, other) - 9) & (gx <= max(u, other) + 9)
        bx.plot(gx[mask], y[mask], color=color, ls=ls, marker=".", label=hz + " Hz")
        for k, mark in [("continuous", "o"), ("tick", "s")]:
            vals = 1000 * np.array(example["checks"][k + "-" + hz])
            cx.plot(
                np.linspace(-5, 5, 5),
                vals,
                color=BLUE if k == "continuous" else GREEN,
                marker=mark,
                ls=ls,
                label=(("Continuous" if k == "continuous" else "Tick") + " choice")
                if hz == "1920"
                else None,
            )
    for shift, color in ((u, GREEN), (other, BLUE)):
        bx.axvline(shift, color=color, ls=":", lw=1)
    bx.set(
        title="(b) Two nearby bumper shifts",
        xlabel="Bumper shift (mm)",
        ylabel="Verified goal margin (mm)",
    )
    cx.set(
        title="(c) The five execution offsets",
        xlabel="Offset from selected shift (mm)",
        ylabel="Verified goal margin (mm)",
    )
    for panel in (bx, cx):
        panel.axhline(0, lw=1, color=GREY)
        panel.legend(loc="upper left", fontsize=9)
    for panel in (ax, bx, cx):
        panel.grid(True)
        panel.set_axisbelow(True)
    save(fig, root, "impact-edits")

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.5, 3.6), layout="constrained")
    kinds = ["prior", "probe-ratio", "probe-learned", "known"]
    for k, color, marker, ls in [("ordinary", BLUE, "o", "-"), ("elevated", GREEN, "s", "--")]:
        ax.plot(
            range(4),
            [report["relay"][k][m]["1920"]["path_mm"][1] for m in kinds],
            color=color,
            marker=marker,
            ls=ls,
            label="Floor" if k == "ordinary" else "Shelf",
        )
    ax.set(
        xticks=range(4),
        xticklabels=["Prior", "Ratio", "Learned", "Known"],
        ylabel="Receiver path RMSE (mm)",
        title="(a) Which mass is available?",
        ylim=(0, None),
    )
    ax.legend(fontsize=10)
    ex = load(root, "relay-example")
    scene = ex["scene"]
    for seg in scene["segments"]:
        bx.plot([seg["a"][0], seg["b"][0]], [seg["a"][1], seg["b"][1]], lw=3, color=GREY)
    b = scene["bumper"]
    bx.add_patch(
        plt.Circle(
            (b["position"][0] + ex["edit_m"], b["position"][1]), b["radius"], color=GREY, alpha=0.55
        )
    )
    for k, color, ls in [("1920", BLUE, "-"), ("forecast", RUST, "--")]:
        path = np.array(ex["paths"][k])
        bx.plot(*path[:, 0].T, color=color, ls=ls, label="Reference" if k == "1920" else "Forecast")
        cx.plot(
            np.arange(121) / 60,
            ex["direction"] * (path[:, 1, 0] - ex["receiver"]["position"][0]),
            color=color,
            ls=ls,
        )
    bx.scatter(*ex["receiver"]["position"], s=45, color=GREEN, zorder=3, label="Receiver at release")
    bx.set(
        xlim=(0, 6),
        ylim=(0, 4),
        aspect="equal",
        title="(b) A shelf held out of fitting",
        xlabel="Horizontal position (m)",
        ylabel="Height (m)",
    )
    bx.legend(fontsize=9, loc="upper center", ncol=2, columnspacing=0.5)
    cx.axhline(0.35, color=GREY, lw=1, ls=":")
    for k, col in [("forecast", RUST), ("1920", BLUE)]:
        t = ex["times_s"][k]
        if t is not None:
            cx.axvline(t, ls=":", color=col, lw=1)
    cx.set(
        title="(c) An invented transfer",
        xlabel="Time since release (s)",
        ylabel="Receiver travel toward goal (m)",
    )
    for panel in [ax, bx, cx]:
        panel.grid(True)
        panel.set_axisbelow(True)
    save(fig, root, "relay-forecast")
    return {"figures": ["impact-clocks", "impact-edits", "relay-forecast"]}


if __name__ == "__main__":
    summary(ROOT)
    draw(ROOT)
