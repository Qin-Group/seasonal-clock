"""Build vector comparison figures and a checksum manifest from saved results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import phase_model as compact
import review_phase_fit as review
from fit_reference_phase import GENOTYPES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"figure/report"
DATA = ROOT/"out/model_comparison/results.json"
COLORS = ("#ED8630", "#2024F6")


def read(path):
    return json.loads(path.read_text())


def save(fig, name):
    fig.savefig(OUT/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = read(DATA)
    ref, fits = r["reference"], r["results"]
    seven, four = fits["linear_window"], fits["fixed_activation"]
    raw = read(ROOT/"out/reference/raw_joint/results.json")
    observation = read(ROOT/"out/reference/observation.json")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    candidates = [(ref, "Reference model", "#888888", "--"),
                  (four, "reduced model phase preset (7/4)", "#228833", "-")]
    targets = review.load_targets()
    P = np.array(sorted(targets))
    obs = np.array([targets[p][0] for p in P])
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.1), layout="constrained")
    for c, ax in enumerate(axes[0]):
        ax.scatter(P, obs[:, c], color="black", s=14, label="Stored target", zorder=5)
        for x, label, color, style in candidates:
            y = np.array([x["WT"][str(p)]["peaks"] for p in P])
            ax.plot(P, y[:, c], ls=style, color=color, label=label)
        ax.set(xlabel="Photoperiod (h)", ylabel="Peak ZT (h)",
               title=f"({'ab'[c]}) {'M' if c == 0 else 'E'} timing")
    axes[0, 0].legend(fontsize=7, frameon=False)
    ax = axes[1, 0]
    ax.scatter(P, obs[:, 1]-obs[:, 0], color="black", s=14)
    for x, label, color, style in candidates:
        y = np.array([x["WT"][str(p)]["peaks"] for p in P])
        ax.plot(P, review.model.folded(y[:, 1]-y[:, 0]), ls=style, color=color)
    ax.set(xlabel="Photoperiod (h)", ylabel="E - M gap (h)", title="(c) Seasonal phase gap")
    ax = axes[1, 1]
    for c, marker in enumerate(("o", "^")):
        gx = np.arange(3)+(-.08 if c == 0 else .08)
        ax.scatter(gx, [x[c] for x in GENOTYPES.values()], c="black", s=24,
                   marker=marker, label=f"{'M' if c == 0 else 'E'} target")
        ax.scatter(gx, [four["genotypes"][g]["peaks"][c] for g in GENOTYPES],
                   facecolors="none", edgecolors="#228833", s=24, marker=marker,
                   label=f"{'M' if c == 0 else 'E'} fit")
    ax.set(xticks=range(3), xticklabels=["pdf0", "pdfr-M", "pdfr-E"],
           ylabel="Peak ZT (h)", ylim=(12, 18.5), title="(d) LD12 knockouts")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    save(fig, "seasonal_fit")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), layout="constrained")
    ax = axes[0]
    selected = [ref, seven, four, fits["no_coupling"]]
    labels = ["reference\n10 free", "reduced model\n7 free", "reduced model\n4 free", "No coupling\n4 free"]
    ax.bar(range(4), [x["WT_peak_rmse_h"] for x in selected],
           color=["#888888", "#4477AA", "#228833", "#CC6677"], width=.6)
    ax.scatter(range(3), [x["cross_validation"]["rmse_h"] for x in selected[:3]],
               marker="D", s=23, color="black", label="Conditional CV")
    ax.set(xticks=range(4), xticklabels=labels, ylabel="WT peak RMSE (h)", ylim=(0, .43),
           title="(a) Parameter reduction")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for x, label, color, style in candidates + [
            (fits["no_coupling"], "No coupling", "#CC6677", ":")]:
        ax.plot(range(1, 5), [d["gap"] for d in x["DD_transfer"]],
                marker="o", ms=3, ls=style, color=color, label=label)
    ax.scatter([1], [3.1], marker="x", color="black", s=24)
    ax.vlines(4, 0, 1, lw=3, color=".65")
    ax.set(xlabel="Day in DD", ylabel="E - M gap (h)", xticks=range(1, 5),
           ylim=(0, 3.7), title="(b) Retain synchronizing coupling")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    save(fig, "reduction_and_dd")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), layout="constrained")
    for P0, color in ((9, "#888888"), (12, "#4477AA"), (15, "#228833")):
        t, sol = compact.simulate(P0)
        mask = t >= t[-1]-24
        zt = t[mask]-(t[-1]-24)
        # Choose equivalent phase lifts with the relative winding removed.
        # Shifting one phase by 2*pi also shifts the mean by pi.
        phases = sol[mask].copy()
        winding = np.round(np.median(phases[:, 1]-phases[:, 0])/(2*np.pi))
        phases[:, 1] -= 2*np.pi*winding
        mean = phases.mean(axis=1)
        theta = compact.OMEGA*t[mask]+np.pi-compact.OMEGA*compact.reference_peak(P0)
        u = (mean-theta+np.pi)%(2*np.pi)-np.pi
        if np.max(np.abs(u)) >= .01 or np.max(np.abs(phases[:, 1]-phases[:, 0])) >= np.pi:
            raise RuntimeError("WT phase lifts do not match the plotted entrained branch")
        axes[0].plot(zt, u, color=color, label=f"P = {P0} h")
        axes[1].plot(zt, phases[:, 1]-phases[:, 0], color=color)
    axes[0].set(xlabel="ZT (h)", ylabel="m - reference (rad)", title="(a) Entrained WT mean",
                ylim=(-.01, .01), xticks=[0, 6, 12, 18, 24])
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set(xlabel="ZT (h)", ylabel="Phase difference (rad)", title="(b) Seasonal separation",
                xticks=[0, 6, 12, 18, 24])
    save(fig, "mean_and_gap")

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.3), sharex=True, sharey=True,
                             layout="constrained")
    for ax, P0 in zip(axes.flat, (4, 9, 12, 13, 15, 20)):
        i = next(i for i, x in enumerate(observation["conditions"]) if x["P"] == P0)
        row = observation["conditions"][i]
        for c, color in enumerate(COLORS):
            ax.errorbar(row["ZT"], row["mean"][c], yerr=row["SEM"][c],
                        fmt="o", ms=2.5, capsize=1.5, color=color)
            ax.plot(row["ZT"], raw["cases"]["reference_raw_joint_fit"]["conditions"][i]["predicted"][c],
                    "--", lw=1.2, color=color)
            ax.plot(row["ZT"], r["raw_refits"]["fixed_activation"]["conditions"][i]["predicted"][c],
                    "-", lw=1.2, color=color)
        ax.axvspan(P0, 24, color=".94", zorder=0)
        ax.set(title=f"{P0} h light", xticks=[0, 12, 24], ylim=(-.03, 1.02))
    for ax in axes[-1]:
        ax.set_xlabel("ZT (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Speckle fraction")
    fig.legend(handles=[Line2D([], [], color=COLORS[0], label="M"),
                        Line2D([], [], color=COLORS[1], label="E"),
                        Line2D([], [], color=".4", ls="--", label="reference raw fit"),
                        Line2D([], [], color=".4", label="reduced model four-parameter raw fit")],
               loc="outside upper center", ncol=4, frameon=False, fontsize=7)
    save(fig, "raw_waveforms")

    inputs = [DATA, ROOT/"out/reference/raw_joint/results.json", ROOT/"out/reference/observation.json",
              ROOT/"experimental_phase_vs_daylength.csv",
              ROOT/"out/model_comparison/core_validation.json",
              ROOT/"code/build_report_assets.py", ROOT/"code/phase_model.py"]
    outputs = sorted(p for p in OUT.iterdir() if p.suffix == ".pdf")
    manifest = {"report_version": "1.0.0", "fitting_performed": False,
                "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in inputs+outputs}}
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Built {len(outputs)} vector figures in {OUT}")


if __name__ == "__main__":
    main()
