"""Supplementary figures from saved results; no fitting is performed.
Run: uv run python code/build_supplementary_figures.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_reference_phase import GENOTYPES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"figure/supplementary"
M, E = "#ED8630", "#2024F6"


def table(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fits = json.loads((ROOT/"out/model_comparison/results.json").read_text())["results"]
    model, uncoupled = fits["fixed_activation"], fits["no_coupling"]
    data = table(ROOT/"experimental_phase_vs_daylength.csv")
    curve = [row for row in table(ROOT/"figure/figure5/model_peaks.csv")
             if row["genotype"] == "WT"]
    P = np.array([float(row["daylength_h"]) for row in data])
    grid = np.array([float(row["light_h"]) for row in curve])
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6), layout="constrained")

    ax = axes[0, 0]
    for name, color in (("M", M), ("E", E)):
        ax.errorbar(P, [float(row[f"{name}_bootstrap_mean_zt"]) for row in data],
                    yerr=[float(row[f"{name}_bootstrap_sd_h"]) for row in data],
                    fmt="o", ms=3.5, capsize=2, color=color, label=f"{name} data")
        ax.plot(grid, [float(row[f"{name}_ZT"]) for row in curve], color=color,
                label=f"{name} model")
    ax.set(xlabel="Photoperiod (h)", ylabel="Peak time (ZT, h)",
           xticks=[4, 8, 12, 16, 20], yticks=[8, 12, 16, 20], title="(a) Wild-type peak times")
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[0, 1]
    ax.errorbar(P, [float(row["bootstrap_mean_phase_gap_h"]) for row in data],
                yerr=[float(row["bootstrap_sd_phase_gap_h"]) for row in data],
                fmt="o", ms=3.5, capsize=2, color="black", label="Data")
    ax.plot(grid, [float(row["gap_h"]) for row in curve], color="black", label="Model")
    ax.set(xlabel="Photoperiod (h)", ylabel="E $-$ M gap (h)",
           xticks=[4, 8, 12, 16, 20], title="(b) Wild-type phase gap")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 0]
    for c, (name, color) in enumerate((("M", M), ("E", E))):
        x = np.arange(3)+(-.1 if c == 0 else .1)
        ax.scatter(x, [GENOTYPES[g][c] for g in GENOTYPES], s=30, facecolors="none",
                   edgecolors=color, label=f"{name} target")
        ax.scatter(x, [model["genotypes"][g]["peaks"][c] for g in GENOTYPES], s=24,
                   marker="s", color=color, label=f"{name} model")
    ax.set(xticks=range(3), xlim=(-.5, 2.5), ylim=(12, 18.5), ylabel="Peak time (ZT, h)",
           xticklabels=["PDF loss", "PDFR loss\nin M", "PDFR loss\nin E"],
           title="(c) Mutants at 12L:12D")
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[1, 1]
    days = range(1, 5)
    ax.plot(days, [d["gap"] for d in model["DD_transfer"]], "o-", ms=3.5, color="black",
            label="Model")
    ax.plot(days, [d["gap"] for d in uncoupled["DD_transfer"]], "o:", ms=3.5, color=".5",
            label=r"Refit with $\kappa=0$")
    ax.scatter([1], [3.1], marker="x", s=36, color="#CC3311", zorder=5, label="Day-1 target")
    ax.vlines(4, 0, 1, lw=4, color="#CC3311", alpha=.35, label="Day-4 target")
    ax.set(xlabel="Day in constant darkness", ylabel="E $-$ M gap (h)", xticks=list(days),
           ylim=(0, 4.9), title="(d) Gap in constant darkness")
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)

    fig.savefig(OUT/"model_fit.pdf", bbox_inches="tight")
    fig.savefig(OUT/"model_fit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT/'model_fit.pdf'}")

    # The saved profile figures carry working titles; the SI captions replace them.
    from PIL import Image
    for name, top in (("profiles", 90), ("tradeoffs", 58), ("combination_profiles", 52)):
        image = Image.open(ROOT/"out/identifiability"/f"{name}.png")
        image.crop((0, top, image.width, image.height)).save(OUT/f"{name}.png")


if __name__ == "__main__":
    main()
