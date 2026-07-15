"""Build publication figures from the completed Phase 2 result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("qbench/results/results.csv")
PAIRED = Path("qbench/analysis/plots/paired_statistics.csv")
ABLATIONS = Path("qbench/analysis/plots/ablations.csv")
OUTPUT = Path("articles/assets")
BLUE = "#2563eb"
ORANGE = "#ea580c"
INK = "#172033"
GRID = "#d8dee9"


def setup():
    plt.rcParams.update({
        "figure.figsize": (10, 5.6),
        "figure.dpi": 180,
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)


def finish(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()


def crossover_plot():
    stats = pd.read_csv(PAIRED)
    x = stats.token_budget.to_numpy()
    mean = stats.mean_delta.to_numpy()
    low = stats.ci_low.to_numpy()
    high = stats.ci_high.to_numpy()

    fig, ax = plt.subplots()
    ax.axhspan(ax.get_ylim()[0], 0, color=BLUE, alpha=0.035)
    ax.axhline(0, color=INK, linewidth=1.2)
    ax.plot(x, mean, color=INK, linewidth=1.5, zorder=2)
    colors = [BLUE if value < 0 else ORANGE for value in mean]
    for budget, value, lower, upper, color in zip(x, mean, low, high, colors):
        ax.errorbar(budget, value, yerr=[[value - lower], [upper - value]], fmt="o",
                    color=color, ecolor=color, markersize=7, capsize=4, linewidth=1.8, zorder=3)
    ax.axvspan(5_000_000, 10_000_000, color="#f59e0b", alpha=0.12)
    ax.annotate("measured crossover bracket", xy=(7_100_000, 0.082), ha="center",
                color="#92400e", fontsize=10)
    ax.text(540_000, -0.112, "quaternion lower loss", color=BLUE, fontsize=10)
    ax.text(540_000, 0.100, "real lower loss", color=ORANGE, fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(x, ["0.5M", "1M", "2M", "5M", "10M", "25M", "50M"])
    ax.set_ylim(-0.13, 0.115)
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Validation loss difference, quaternion minus real")
    ax.set_title("Quaternion wins early. Real wins after the crossover.", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    finish(OUTPUT / "crossover.png")


def throughput_plot():
    results = pd.read_csv(RESULTS)
    primary = results[results.model.isin(["real", "quat"])]
    speed = primary.groupby("model").tokens_per_sec.mean()
    values = np.array([speed.real, speed.quat]) / 1000

    fig, ax = plt.subplots()
    bars = ax.bar(["Real matrix", "Quaternion"], values, color=[ORANGE, BLUE], width=0.56)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.55, f"{value:.1f}K",
                ha="center", fontweight="bold", fontsize=13)
    ax.text(0.5, max(values) * 0.64, f"Real trained {values[0] / values[1]:.1f}× faster",
            ha="center", fontsize=12, color=INK)
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("Training tokens per second, thousands")
    ax.set_title("The full quaternion model paid a compute cost", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    finish(OUTPUT / "training-throughput.png")


def ablation_plot():
    stats = pd.read_csv(ABLATIONS)
    stats = stats[stats.token_budget == 50_000_000].copy()
    names = {"quat_attn": "Quaternion attention only",
             "quat_ffn": "Quaternion FFN only",
             "quat_gaussian": "Gaussian initialization",
             "quat_w64": "Quaternion width 64"}
    stats["label"] = stats.model.map(names)
    stats = stats.sort_values("mean_delta_vs_quat")
    y = np.arange(len(stats))

    fig, ax = plt.subplots()
    values = stats.mean_delta_vs_quat.to_numpy()
    low = stats.ci_low.to_numpy()
    high = stats.ci_high.to_numpy()
    colors = [BLUE if value < 0 else ORANGE for value in values]
    for position, value, lower, upper, color in zip(y, values, low, high, colors):
        ax.errorbar(value, position, xerr=[[value - lower], [upper - value]], fmt="o",
                    color=color, ecolor=color, markersize=8, capsize=4, linewidth=2)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_yticks(y, stats.label)
    ax.set_xlabel("Validation loss difference versus full quaternion")
    ax.set_title("Ablations at 50M tokens", loc="left", pad=30)
    ax.text(0.0, 1.02, "Negative is better. Hybrid placement models have more parameters.",
            transform=ax.transAxes, color="#5b6474", fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    finish(OUTPUT / "ablation-effects.png")


def main():
    setup()
    crossover_plot()
    throughput_plot()
    ablation_plot()


if __name__ == "__main__":
    main()
