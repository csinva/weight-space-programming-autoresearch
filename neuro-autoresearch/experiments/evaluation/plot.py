"""Publication-style cohort comparison plots for evaluation.json artifacts.

Examples
--------
python experiments/evaluation/plot.py --dataset nnn
python experiments/evaluation/plot.py --input path/to/evaluation.json --output figure.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"

DATASET_LABELS = {
    "nsd": "Natural Scenes Dataset",
    "tvsd": "TVSD",
    "nnn": "Triple-N",
    "huth": "Huth natural speech",
    "ieeg": "Podcast iEEG",
}
METRIC_LABELS = {
    "noise_normalized_r2": "Noise-normalized explained variance",
    "mean_pearson_r": "Mean Pearson correlation",
}

INK = "#17212B"
MUTED = "#66717D"
GRID = "#DDE2E6"
BASELINE = "#7893A5"
CHAMPION = "#D05A47"
RANDOM = "#59636E"
PAPER = "#FAFAF8"


def _result_path(dataset: str) -> Path:
    """Find the canonical evaluation artifact without assuming publication state."""
    candidates = [
        ROOT / "runs" / "best" / dataset / "evaluation.json",
        RESULTS / dataset / "evaluation.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"no evaluation result for {dataset!r}; searched:\n  {searched}")


def _short_label(model_id: str, model: dict) -> str:
    if model.get("role") == "champion":
        return "Champion"
    family = model.get("family")
    labels = {
        "alexnet": "AlexNet",
        "resnet50": "ResNet-50",
        "vit": "ViT-B/16",
        "dino": "DINOv2",
        "clip": "CLIP",
        "gpt2": "GPT-2 XL",
    }
    if family in labels:
        label = labels[family]
        if model.get("weights") in {"pretrained", "untrained"} and family in {
            "alexnet", "resnet50", "vit"
        }:
            state = "trained" if model["weights"] == "pretrained" else "untrained"
            return f"{label}  ·  {state}"
        return label
    label = model.get("label", model_id)
    return label.split(" (")[0]


def _ordered_models(models: dict) -> list[tuple[str, dict]]:
    """Keep JSON order while moving the random projection into a reference line."""
    return [(model_id, model) for model_id, model in models.items()
            if model.get("family") != "random_projection"]


def _row_positions(models: list[tuple[str, dict]]) -> np.ndarray:
    """Keep trained/untrained pairs tight and add air between model families."""
    positions: list[float] = []
    position = 0.0
    previous_family = None
    for model_id, model in models:
        family = model.get("family", model_id)
        if previous_family is not None and family != previous_family:
            position += 0.24
        positions.append(position)
        position += 1.0
        previous_family = family
    return np.asarray(positions)


def _axis_limit(document: dict) -> tuple[float, float]:
    values = [
        float(score)
        for model in document["models"].values()
        for cohort in ("search", "heldout")
        for score in model["scores"][cohort]["subjects"].values()
    ]
    low, high = min(0.0, min(values)), max(values)
    span = high - low or 1.0
    return low - 0.035 * span, high + 0.20 * span


def plot_evaluation(document: dict, output: Path, *, dpi: int = 220) -> Path:
    """Render one evaluation document to PNG, PDF, or SVG."""
    dataset = document["dataset"]
    models = _ordered_models(document["models"])
    random_model = next(
        model for model in document["models"].values()
        if model.get("family") == "random_projection"
    )
    y = _row_positions(models)
    xlim = _axis_limit(document)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": 650,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    figure_height = max(3.8, min(6.4, 2.8 + 0.4 * len(models)))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, figure_height), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    panel_titles = {
        "search": "Search subjects",
        "heldout": "Out-of-distribution subjects",
    }

    for ax, cohort in zip(axes, ("search", "heldout")):
        ax.set_facecolor(PAPER)
        random_mean = float(random_model["scores"][cohort]["mean"])
        ax.axvline(random_mean, color=RANDOM, linewidth=1.35,
                   linestyle=(0, (2.2, 2.4)), zorder=1)

        for row, (model_id, model) in zip(y, models):
            mean = float(model["scores"][cohort]["mean"])
            subjects = list(model["scores"][cohort]["subjects"].values())
            champion = model.get("role") == "champion"
            untrained = model.get("weights") == "untrained"
            color = CHAMPION if champion else BASELINE

            ax.barh(
                row, mean, height=0.57,
                facecolor=PAPER if untrained else color,
                edgecolor=color,
                linewidth=1.35 if untrained else 0,
                hatch="////" if untrained else None,
                zorder=2,
            )
            offsets = np.linspace(-0.035, 0.035, len(subjects)) if len(subjects) > 1 else [0]
            ax.scatter(subjects, row - 0.19 + offsets, s=20, facecolor=PAPER,
                       edgecolor=INK, linewidth=0.75, zorder=4)
            ax.text(mean + 0.012 * (xlim[1] - xlim[0]), row, f"{mean:.3f}",
                    va="center", ha="left", fontsize=8.6, color=INK, zorder=5)

        subject_text = ", ".join(document["cohorts"][cohort])
        ax.set_title(panel_titles[cohort], loc="left", color=INK, pad=16)
        ax.text(0, 1.012, f"Subjects {subject_text}  ·  equal-weight mean",
                transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=8.6)
        ax.text(random_mean, -0.72, f" random  {random_mean:.3f}", color=RANDOM,
                fontsize=8, ha="left", va="center")
        ax.set_xlim(xlim)
        ax.set_ylim(y[-1] + 0.65, -1.05)
        ax.xaxis.grid(True, color=GRID, linewidth=0.75, zorder=0)
        ax.yaxis.grid(False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", colors=MUTED, length=0, pad=7)
        ax.tick_params(axis="y", length=0, pad=10)

    labels = [_short_label(model_id, model) for model_id, model in models]
    axes[0].set_yticks(y, labels=labels, color=INK)
    axes[1].tick_params(axis="y", labelleft=False)
    metric = METRIC_LABELS.get(dataset["metric"], dataset["metric"].replace("_", " ").title())
    fig.supxlabel(metric, x=0.59, y=0.075, color=INK, fontsize=10)

    title = DATASET_LABELS.get(dataset["id"], dataset["id"].upper())
    fig.suptitle(f"{title}: model generalization across subjects", x=0.095, y=0.975,
                 ha="left", color=INK, fontsize=16, fontweight=650)
    fig.text(0.095, 0.925,
             "Bars show cohort means; points show individual subjects. Higher is better.",
             ha="left", color=MUTED, fontsize=9.4)

    legend = [
        Patch(facecolor=CHAMPION, edgecolor="none", label="Searched champion · never trained"),
        Patch(facecolor=BASELINE, edgecolor="none", label="Pretrained baseline"),
    ]
    if any(model.get("weights") == "untrained" for _, model in models):
        legend.append(Patch(facecolor=PAPER, edgecolor=BASELINE, hatch="////",
                            label="Untrained baseline"))
    legend.extend([
        Line2D([0], [0], color=RANDOM, linewidth=1.35, linestyle=(0, (2.2, 2.4)),
               label="Random projection"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER,
               markeredgecolor=INK, markeredgewidth=0.75, markersize=5, label="Subject"),
    ])
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.59, 0.008),
               ncol=len(legend), frameon=False, handlelength=1.7,
               columnspacing=1.6, fontsize=8.5)

    fig.subplots_adjust(left=0.185, right=0.985, top=0.845, bottom=0.16, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight", "facecolor": PAPER}
    if output.suffix.lower() == ".png":
        save_kwargs["dpi"] = dpi
    fig.savefig(output, **save_kwargs)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="dataset ID, e.g. nnn or nsd")
    source.add_argument("--input", type=Path, help="evaluation.json path")
    parser.add_argument("--output", type=Path, help="output .png, .pdf, or .svg path")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution (default: 220)")
    args = parser.parse_args()

    input_path = args.input or _result_path(args.dataset)
    with input_path.open() as f:
        document = json.load(f)
    output = args.output or input_path.with_name("evaluation.png")
    if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("--output must end in .png, .pdf, or .svg")
    print(plot_evaluation(document, output, dpi=args.dpi))


if __name__ == "__main__":
    main()
