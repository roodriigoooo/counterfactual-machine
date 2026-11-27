"""Render the README diagrams from the published gallery and held-out relay records."""

import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as Shape

from counterfactual.figures import plt, style
from counterfactual.relay_forecast import forecast, layout

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
INK, SLATE, LINE = "#283a43", "#52656e", "#82949a"
BLUE, TEAL, VIOLET, CORAL = "#486f9a", "#267966", "#73548d", "#b15d42"


def read(relative):
    return json.loads((ROOT / relative).read_text())


def canvas(height):
    fig = plt.figure(figsize=(12, height / 80), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 960), ylim=(height, 0))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def label(ax, x, y, text, size=13, color=INK, align="center", **kwargs):
    ax.text(x, y, text, fontsize=size, color=color, ha=align, va="center", **kwargs)


def arrow(ax, x, y, end_x, end_y, color=LINE, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            (x, y),
            (end_x, end_y),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.1,
            color=color,
            linestyle=(0, (3, 3)) if dashed else "-",
        )
    )


def encoder(ax, x, y, w, h, title, tint, edge):
    # Rounded output face distinguishes a learned map from a tensor or physical field.
    vertices = [
        (x, y),
        (x + w * 0.68, y),
        (x + w, y),
        (x + w, y + h / 2),
        (x + w, y + h),
        (x + w * 0.68, y + h),
        (x, y + h),
        (x, y),
    ]
    codes = [
        Shape.MOVETO,
        Shape.LINETO,
        Shape.CURVE3,
        Shape.CURVE3,
        Shape.CURVE3,
        Shape.CURVE3,
        Shape.LINETO,
        Shape.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Shape(vertices, codes), fc=tint, ec=edge, lw=1.05))
    label(ax, x + w * 0.46, y + h / 2, title, 14)


def tiles(ax, x, y, tint):
    # Diagrammatic feature cells. Layer and color distinguish tensors, not values.
    for dx, dy in [(-10, -7), (0, 0)]:
        for i in range(4):
            for j in range(3):
                left = x + dx + 10 * i
                top = y + dy + 11 * j - 2 * i
                ax.add_patch(
                    Polygon(
                        [
                            (left, top),
                            (left + 10, top - 2),
                            (left + 10, top + 9),
                            (left, top + 11),
                        ],
                        fc=tint if (i + j) % 3 else "white",
                        ec=SLATE,
                        lw=0.6,
                    )
                )


def playfield(ax, scene, x, y, w, h, *, edit=0.0, paths=(), receiver=None):
    ax.add_patch(Rectangle((x, y), w, h, fc="#f7f9f7", ec="#b1bdbb", lw=0.8))
    # One isotropic transform preserves physical geometry in the miniature.
    scale = min((w - 8) / 6, (h - 6) / 4)
    origin = np.array([x + w / 2 - 3 * scale, y + h / 2 + 2 * scale])

    def pixels(data):
        return origin + np.asarray(data) * [scale, -scale]

    for seg in scene["segments"]:
        xy = pixels([seg["a"], seg["b"]])
        ax.plot(*xy.T, color=SLATE, lw=1.8, solid_capstyle="round")
    bumper = scene["bumper"]
    center = pixels(np.array(bumper["position"]) + [edit, 0])
    ax.add_patch(
        Circle(center, bumper["radius"] * scale, fc="#e8b787", ec=CORAL, lw=0.9)
    )
    for path, color, dashes in paths:
        xy = pixels(np.asarray(path)[:, :2])
        (line,) = ax.plot(*xy.T, color=color, lw=1.7, ls=dashes, solid_capstyle="round")
        line.set_clip_path(Rectangle((x, y), w, h, transform=ax.transData))
    ball = scene["ball"]
    ax.add_patch(
        Circle(
            pixels(ball["position"]),
            ball["radius"] * scale,
            fc=BLUE,
            ec="white",
            lw=0.6,
        )
    )
    if receiver:
        ax.add_patch(
            Circle(
                pixels(receiver.position),
                receiver.radius * scale,
                fc="#e1f1e9",
                ec=TEAL,
                lw=1.3,
            )
        )


def save(fig, name, description):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / f"{name}.svg",
        facecolor="white",
        metadata={
            "Title": name.replace("-", " ").title(),
            "Description": description,
            "Date": None,
            "Creator": "Counterfactual Machine",
        },
    )
    plt.close(fig)


def learning():
    example = read("artifacts/architecture/gallery.json")["success"]
    scene, edit = example["scene"], example["edit_m"]
    fig, ax = canvas(307)
    # The upper path is available at inference; targets below are observed during fitting.
    label(ax, 25, 22, "Predicting the effect of a bumper edit", 21, align="left")
    ax.plot([25, 933], [44, 44], color="#dbe3e1", lw=0.8)
    playfield(ax, scene, 26, 71, 96, 64)
    playfield(ax, scene, 61, 87, 96, 64, edit=edit)
    label(ax, 89, 168, "Original + edited scene", 12)
    arrow(ax, 165, 109, 208, 109)
    encoder(ax, 214, 77, 126, 63, "Shared\nencoder", "#e5edf3", BLUE)
    arrow(ax, 344, 109, 389, 109)
    tiles(ax, 405, 98, "#c5d8e6")
    label(ax, 425, 157, "Local geometry", 12)
    label(ax, 425, 172, "and motion", 12)
    arrow(ax, 453, 109, 496, 109)
    encoder(ax, 502, 77, 126, 63, "Rollout\npredictor", "#e4f0eb", TEAL)
    arrow(ax, 634, 109, 727, 109)
    playfield(
        ax,
        scene,
        736,
        65,
        148,
        87,
        edit=edit,
        paths=[(example["predicted"], TEAL, "--")],
    )
    label(ax, 810, 168, "Edited trajectory", 12)
    playfield(
        ax,
        scene,
        26,
        202,
        125,
        76,
        edit=edit,
        paths=[(example["original"], BLUE, "-"), (example["verified"], TEAL, "-")],
    )
    label(ax, 89, 293, "Observed futures", 12)
    arrow(ax, 159, 239, 208, 239)
    encoder(ax, 214, 207, 126, 63, "Target\nencoder", "#efeaf4", VIOLET)
    arrow(ax, 276, 146, 276, 201, VIOLET, True)
    label(ax, 284, 176, "EMA", 10, VIOLET, "left")
    arrow(ax, 344, 239, 389, 239, VIOLET, True)
    tiles(ax, 405, 230, "#d2c2e0")
    label(ax, 425, 283, "Target embeddings", 12)
    arrow(ax, 454, 239, 722, 239, VIOLET, True)
    label(ax, 512, 257, "Stop gradient", 11, VIOLET)
    ax.plot([566, 566, 698], [145, 197, 197], ls=(0, (3, 3)), color=TEAL, lw=1.1)
    arrow(ax, 698, 197, 744, 223, TEAL, True)
    label(ax, 636, 186, "Predicted embeddings", 11, TEAL)
    ax.add_patch(Rectangle((731, 222), 177, 36, fc="#f6f1f7", ec=VIOLET, lw=1))
    label(ax, 820, 240, "Embedding loss", 14)
    label(ax, 820, 279, "Joint training only", 11, VIOLET)
    save(
        fig,
        "learning-from-edits",
        "Shared encoders and physically constrained rollouts. The optional future-embedding loss uses observed simulator targets only during training.",
    )


def contact():
    # First held-out edit in saved order with contact probability between .15 and .85.
    base = "artifacts/relay-encounter"
    report = read(f"{base}/probability.json")
    pred = report["predictions"]["context"]
    probs = np.asarray(pred["probability"])
    i, j = np.argwhere((probs > 0.15) & (probs < 0.85))[0]
    parent = int(report["test_parent_start"] + i)
    row = read(f"{base}/parents/{parent}.json")["rows"][j]
    world = layout(parent, elevated=True)
    edit = row["edit_m"]
    xy = forecast(world, [edit], [world.receiver.mass], 960, pair_enabled=False)[0][
        0, :, 0
    ]
    mu, scale = 1000 * pred["location_m"][i][j], 1000 * pred["scale_m"][i][j]
    fig, ax = canvas(247)
    label(ax, 25, 22, "From a near miss to a probability", 21, align="left")
    ax.plot([25, 933], [44, 44], color="#dbe3e1", lw=0.8)
    playfield(
        ax,
        world.scene.model_dump(),
        25,
        68,
        152,
        99,
        edit=edit,
        paths=[(xy, BLUE, "--")],
        receiver=world.receiver,
    )
    label(ax, 101, 187, f"Bumper shift   {edit * 1000:+.0f} mm", 12)
    arrow(ax, 182, 118, 215, 118)
    label(ax, 290, 84, "Closest clearance", 14)
    for y, hz, color in [(116, "480", BLUE), (144, "960", TEAL)]:
        ax.plot([227, 242], [y, y], lw=2, color=color)
        label(ax, 249, y, f"{hz} Hz", 12, SLATE, "left")
        gap = row["forecast"][hz][0] * 1000
        label(ax, 362, y, f"{gap:+.1f} mm", 13, color, "right")
    label(ax, 292, 185, "Pass-through forecasts", 12)
    arrow(ax, 369, 118, 400, 118)
    encoder(ax, 405, 88, 133, 61, "Conditional\ndistribution", "#e6f0eb", TEAL)
    label(ax, 470, 185, "Location + scale", 12)
    arrow(ax, 544, 118, 576, 118)
    left, right = 588, 764
    mid = (left + right) / 2
    x = np.linspace(-4 * scale, 4 * scale, 161)
    z = (x - mu) / scale
    density = (1 + z * z / 2) ** -1.5
    xx = mid + x / (4 * scale) * (right - left) / 2
    yy = 153 - 74 * density
    ax.fill_between(xx, yy, 153, color="#e8edf0")
    ax.fill_between(xx, yy, 153, where=x <= 0, color="#bba5ca")
    ax.plot(xx, yy, color=VIOLET, lw=1.6)
    ax.plot([left, right], [153, 153], color=SLATE, lw=0.9)
    ax.plot([mid, mid], [76, 157], color=SLATE, ls=":", lw=0.9)
    label(ax, mid, 167, "0", 10, SLATE)
    label(ax, mid, 188, "Signed clearance (mm)", 12)
    for v in (-10, 10):
        pixel = mid + v / (4 * scale) * (right - left) / 2
        ax.plot([pixel, pixel], [153, 157], color=SLATE, lw=0.7)
        label(ax, pixel, 167, f"{v:+d}", 10, SLATE)
    arrow(ax, 772, 118, 802, 118)
    label(ax, 866, 106, f"{probs[i, j]:.0%}", 27, VIOLET)
    label(ax, 866, 139, "contact probability", 12)
    check = row["reference"]["1920"][0] * 1000
    label(
        ax,
        25,
        225,
        f"Shelf {parent}   ·   The 1920 Hz reference measured {check:.1f} mm of clearance.",
        12,
        SLATE,
        "left",
    )
    save(
        fig,
        "contact-probability",
        "A measured held-out edit. The shaded area is the fitted Student-t probability of zero or negative clearance.",
    )


if __name__ == "__main__":
    style()
    plt.rcParams.update(
        {"svg.hashsalt": "counterfactual-machine-overview", "svg.fonttype": "none"}
    )
    learning()
    contact()
