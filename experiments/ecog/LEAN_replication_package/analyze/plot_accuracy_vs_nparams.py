"""Plot accuracy vs n_params (log scale) for all experiments in runs/."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
OUT_PATH = Path(__file__).resolve().parent / "accuracy_vs_nparams.png"


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")

    run_dirs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())
    for i, run_dir in enumerate(run_dirs):
        csv_path = run_dir / "results" / "overall_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["n_params"] = pd.to_numeric(df["n_params"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df = df.dropna(subset=["n_params", "accuracy"]).reset_index(drop=True)
        if df.empty:
            continue
        color = cmap(i % 10)
        xs = df["n_params"].to_numpy()
        ys = df["accuracy"].to_numpy()
        ax.scatter(xs, ys, color=color, s=30, zorder=3, alpha=0.6, label=run_dir.name)
        for j in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[j + 1], ys[j + 1]),
                xytext=(xs[j], ys[j]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.2,
                    alpha=0.5,
                    shrinkA=4,
                    shrinkB=4,
                ),
            )

    ax.set_xscale("log")
    ax.set_xlabel("n_params (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs n_params per experiment")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
