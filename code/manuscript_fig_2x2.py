"""Manuscript 2x2 subfigure (5.9 x 5.3 in).

Panel (a) is the M/E circuit schematic, drawn as vector; panels (b)-(d) are
the three panels of ``minimal_model_continuous`` restyled for print size.
Panel (d) is ordered WT, pdf0, pdfr-M, pdfr-E. M/E peak times coincide only
for pdf0 (both cells at ZT15), so that column alone is offset vertically by
+/-0.07 h. The symbols still overlap at that offset -- the point is only to
expose both colours, not to show a difference -- so say so in the caption.

Regenerate with:  uv run python code/manuscript_fig_2x2.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure")
from dual_oscillator_minimal import (
    TUNED, peak_zt, folded, EXPERIMENTAL_GAP_SDS, EXPERIMENTAL_PHASE_SDS,
    EXPERIMENTAL_TARGETS, GENOTYPE_TARGETS_12,
)

FIG_W, FIG_H = 4.0, 3.6

# "vector" draws panel (a) natively; a path embeds that image instead.
SCHEMATIC = "vector"

# Sampled from the panel (a) schematic so the plots match it exactly.
M_COLOR = "#ED8630"
E_COLOR = "#2024F6"
WT_COLOR = "#2E7D32"
DATA_GRAY = "#777777"

# Sizes are for the final print size above, so nothing drops below 6 pt.
LABEL_FS = 11.0    # axis labels, matched to Fig. 5H/I
TICK_FS = 11.0     # tick labels, matched to Fig. 5H/I
ANNOT_FS = 8.0     # in-panel M/E series labels
KEY_FS = 6.5       # "line: model / dot: data" encoding keys

# Panel (d): data sits left of each genotype tick, model right of it.
DX_DATA, DX_MODEL = -0.10, +0.10
MS_DATA, MS_MODEL = 13, 10        # open-circle / filled-square marker areas
# M and E coincide exactly for pdf0, so nudge them apart by well under one
# marker height: the symbols still overlap, which keeps the displacement from
# reading as a real difference, but both colours show. Flag this in the caption.
JITTER_H = 0.07
JITTER_GENOTYPES = {"pdf0"}

# Explicit margins rather than tight_layout, so the saved page is exactly
# FIG_W x FIG_H and the four panels align on a fixed grid.
LEFT, RIGHT, BOTTOM, TOP = 0.195, 0.99, 0.165, 0.995
WSPACE, HSPACE = 0.60, 0.85
_AX_W = (RIGHT - LEFT) / (2 + WSPACE)
_AX_H = (TOP - BOTTOM) / (2 + HSPACE)
# Panel (a) has no tick/axis labels, so it starts left of the plot column.
PANEL_A_X0 = 0.055
_PANEL_A_W = (LEFT + _AX_W - PANEL_A_X0) * FIG_W
_PANEL_A_H = (_AX_H + 0.015) * FIG_H
# Schematic data units, matched to the panel aspect so aspect="equal" fills it.
SCHEM_H = 10.0
SCHEM_W = SCHEM_H * _PANEL_A_W / _PANEL_A_H
SCHEM_FS = 11.0


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        # Arial ships as a plain .ttf on macOS; the system Helvetica is a .ttc
        # collection that fontTools warns about while subsetting for the PDF.
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,      # keep all text editable in Illustrator
        "ps.fonttype": 42,
        # dual_oscillator_minimal sets savefig.bbox="tight" on import, which
        # would re-crop the page and lose the requested figure size.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "axes.labelsize": LABEL_FS,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
    })


def _draw_schematic(ax):
    """Panel (a) M/E circuit, drawn as vector so it stays sharp in print.

    Data units are chosen to match the axes aspect ratio, so `aspect="equal"`
    keeps the cell bodies circular while still filling the panel.
    """
    from matplotlib.patches import Circle, FancyArrowPatch

    ax.set_xlim(0, SCHEM_W)
    ax.set_ylim(0, SCHEM_H)
    ax.set_aspect("equal")

    r = 1.15
    cy = 4.6
    # E's x-position is set relative to SCHEM_W (not hardcoded) so the
    # schematic re-centers itself if the panel's aspect ratio changes.
    m_c, e_c = (SCHEM_W * 0.15, cy), (SCHEM_W - 3.0, cy)
    lw = 1.2

    ax.add_patch(Circle(m_c, r, facecolor="none", edgecolor=M_COLOR, lw=lw))
    ax.add_patch(Circle(e_c, r, facecolor="none", edgecolor=E_COLOR, lw=lw))
    ax.text(*m_c, "M", color=M_COLOR, fontsize=SCHEM_FS + 0.5,
            ha="center", va="center")
    ax.text(*e_c, "E", color=E_COLOR, fontsize=SCHEM_FS + 0.5,
            ha="center", va="center")

    # M --| E : blunt-ended inhibition of the E cell by PDF release.
    bar_x = e_c[0] - r - 0.42
    ax.plot([m_c[0] + r, bar_x], [m_c[1], m_c[1]], color="k", lw=lw,
            solid_capstyle="butt")
    ax.plot([bar_x, bar_x], [m_c[1] - 0.78, m_c[1] + 0.78], color="k", lw=lw,
            solid_capstyle="butt")
    # At this font size relative to the panel, "PDF release" is wide enough
    # that centering it on the bar's midpoint lets its left edge reach into
    # the Advance arrow above M; anchor past the midpoint (toward E) instead.
    pdf_x = (m_c[0] + r) + 0.55 * (bar_x - (m_c[0] + r))
    ax.text(pdf_x, m_c[1] + 0.20, "PDF release",
            fontsize=SCHEM_FS, ha="center", va="bottom")

    # Self-loop: light drives M to advance its own phase.
    ax.add_patch(FancyArrowPatch(
        (m_c[0] + 1.00, m_c[1] + 0.86), (m_c[0] - 0.32, m_c[1] + 1.74),
        connectionstyle="arc3,rad=0.62", arrowstyle="-|>",
        mutation_scale=7, lw=lw, color="k", shrinkA=0, shrinkB=0))
    ax.text(m_c[0] - 0.55, m_c[1] + 2.30, "Advance", fontsize=SCHEM_FS,
            ha="center", va="bottom")
    ax.text(e_c[0], e_c[1] + r + 0.45, "Delay", fontsize=SCHEM_FS,
            ha="center", va="bottom")

    # Light --> M, via membrane depolarisation.
    ax.add_patch(FancyArrowPatch(
        (m_c[0], 1.70), (m_c[0], m_c[1] - r - 0.10),
        arrowstyle="-|>", mutation_scale=7, lw=lw, color="k",
        shrinkA=0, shrinkB=0))
    ax.text(m_c[0], 1.55, "Light", fontsize=SCHEM_FS, ha="center", va="top")
    ax.text(m_c[0] + 0.50, 2.70, "depolarization", fontsize=SCHEM_FS,
            ha="left", va="center")


def _trimmed(path, pad=6):
    """Load an image and crop its white border so it fills panel (a)."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = np.where((a < 245).any(axis=2))
    if ink[0].size == 0:
        return a
    y0, y1 = ink[0].min(), ink[0].max()
    x0, x1 = ink[1].min(), ink[1].max()
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(a.shape[0] - 1, y1 + pad), min(a.shape[1] - 1, x1 + pad)
    return a[y0:y1 + 1, x0:x1 + 1]


def build_figure(params=TUNED, out_path=os.path.join(FIGURE_DIR, "manuscript_fig_2x2.pdf"),
                 schematic=SCHEMATIC):
    _style()
    Ps = sorted(EXPERIMENTAL_TARGETS)
    dM = [EXPERIMENTAL_TARGETS[P][0] for P in Ps]
    dE = [EXPERIMENTAL_TARGETS[P][1] for P in Ps]
    sdM = [EXPERIMENTAL_PHASE_SDS[P][0] for P in Ps]
    sdE = [EXPERIMENTAL_PHASE_SDS[P][1] for P in Ps]
    dGap = [EXPERIMENTAL_TARGETS[P][1] - EXPERIMENTAL_TARGETS[P][0] for P in Ps]
    sdGap = [EXPERIMENTAL_GAP_SDS[P] for P in Ps]

    grid = np.arange(Ps[0], Ps[-1] + 1e-9, 0.2)
    wt = [peak_zt(P, params, genotype="WT") for P in grid]
    mS = np.array([m for m, _ in wt])
    eS = np.array([e for _, e in wt])
    gap_wt = np.array([folded(e - m) for m, e in wt])

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ---- (a) M/E circuit schematic ---------------------------------------
    ax_a.set_axis_off()
    if schematic == "vector":
        _draw_schematic(ax_a)
    elif schematic and os.path.exists(schematic):
        ax_a.imshow(_trimmed(schematic), interpolation="lanczos")
    elif schematic:
        print(f"warning: schematic not found, panel (a) left blank: {schematic}")

    # ---- (b) phase gap vs daylength --------------------------------------
    ax_b.plot(grid, gap_wt, "-", color=WT_COLOR, lw=1.4, zorder=3)
    ax_b.errorbar(Ps, dGap, yerr=sdGap, fmt="o", linestyle="none",
                  color=DATA_GRAY, markeredgecolor="white", markeredgewidth=0.4,
                  markersize=3.6, elinewidth=0.55, capsize=1.3, zorder=5)
    ax_b.set_xlabel("Daylength (h)", labelpad=1.0)
    ax_b.set_ylabel("Phase gap (h)", labelpad=1.0)
    ax_b.set_ylim(-0.5, 6.0)
    ax_b.set_xticks([4, 8, 12, 16, 20])
    ax_b.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax_b.grid(alpha=0.25, lw=0.4)
    ax_b.text(0.03, 0.96, "line: model\ndot: data", transform=ax_b.transAxes,
              fontsize=KEY_FS, color="0.35", ha="left", va="top",
              linespacing=0.9)

    # ---- (c) M and E peak times ------------------------------------------
    ax_c.plot(grid, mS, "-", color=M_COLOR, lw=1.4, zorder=3)
    ax_c.plot(grid, eS, "-", color=E_COLOR, lw=1.4, zorder=3)
    ax_c.errorbar(Ps, dM, yerr=sdM, fmt="o", linestyle="none",
                  color=M_COLOR, markeredgecolor="black", markeredgewidth=0.4,
                  markersize=3.6, elinewidth=0.55, capsize=1.3, zorder=5)
    ax_c.errorbar(Ps, dE, yerr=sdE, fmt="o", linestyle="none",
                  color=E_COLOR, markeredgecolor="black", markeredgewidth=0.4,
                  markersize=3.6, elinewidth=0.55, capsize=1.3, zorder=5)
    ax_c.set_ylim(4, 24)
    ax_c.set_yticks([4, 8, 12, 16, 20, 24])
    ax_c.set_xlabel("Daylength (h)", labelpad=1.0)
    ax_c.set_ylabel("M/E peak (ZT)", labelpad=1.0)
    ax_c.set_xticks([4, 8, 12, 16, 20])
    ax_c.grid(alpha=0.25, lw=0.4)
    ax_c.text(17.2, 21.4, "E", color=E_COLOR, fontsize=ANNOT_FS + 1,
              fontweight="bold", ha="center", va="center")
    ax_c.text(17.2, 13.3, "M", color=M_COLOR, fontsize=ANNOT_FS + 1,
              fontweight="bold", ha="center", va="center")
    ax_c.text(0.03, 0.96, "line: model   dot: data", transform=ax_c.transAxes,
              fontsize=KEY_FS, color="0.35", ha="left", va="top")

    # ---- (d) 12:12 genotype collapse -------------------------------------
    genos = ["WT", "pdf0", "pdfrM", "pdfrE"]
    labels = ["WT", "pdf0", "pdfr-M", "pdfr-E"]
    tgt = {"WT": (13.0, 17.0), **GENOTYPE_TARGETS_12}
    xs = np.arange(len(genos), dtype=float)
    mSim = np.array([peak_zt(12, params, genotype=g)[0] for g in genos])
    eSim = np.array([peak_zt(12, params, genotype=g)[1] for g in genos])

    # Display-only nudge, applied to both the data and the model symbols of
    # the genotypes where M and E land on the same value.
    nudge = np.array([JITTER_H if g in JITTER_GENOTYPES else 0.0
                      for g in genos])
    mDat = np.array([tgt[g][0] for g in genos]) - nudge
    eDat = np.array([tgt[g][1] for g in genos]) + nudge

    ax_d.axhline(15, color="gray", ls=":", lw=0.6, zorder=1)
    ax_d.scatter(xs + DX_DATA, mDat, s=MS_DATA,
                 facecolor="none", edgecolor=M_COLOR, lw=0.8, zorder=4)
    ax_d.scatter(xs + DX_DATA, eDat, s=MS_DATA,
                 facecolor="none", edgecolor=E_COLOR, lw=0.8, zorder=4)
    ax_d.scatter(xs + DX_MODEL, mSim - nudge, s=MS_MODEL, marker="s",
                 color=M_COLOR, zorder=4)
    ax_d.scatter(xs + DX_MODEL, eSim + nudge, s=MS_MODEL, marker="s",
                 color=E_COLOR, zorder=4)
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(labels, rotation=35, ha="right",
                         rotation_mode="anchor")
    ax_d.set_xlim(-0.55, len(genos) - 0.45)
    ax_d.set_ylim(11.5, 18.5)
    ax_d.set_yticks([12, 14, 16, 18])
    ax_d.set_ylabel("Peak ZT at 12:12 (h)", labelpad=1.0)
    ax_d.grid(alpha=0.25, lw=0.4, axis="y")
    ax_d.text(0.03, 0.96, "open: data   filled: model",
              transform=ax_d.transAxes, fontsize=KEY_FS, color="0.35",
              ha="left", va="top")

    for a in (ax_b, ax_c, ax_d):
        a.tick_params(pad=1.0)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP,
                        wspace=WSPACE, hspace=HSPACE)
    # Panel (a) carries no axis decorations, so let it claim the margin space
    # the other panels need for their labels. Must follow subplots_adjust.
    ax_a.set_position([PANEL_A_X0, TOP - _AX_H,
                       LEFT + _AX_W - PANEL_A_X0, _AX_H + 0.015])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", dpi=400)
    plt.close(fig)
    print(f"Saved: {out_path} (+ .png)  [{FIG_W} x {FIG_H} in]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FIGURE_DIR, "manuscript_fig_2x2.pdf"))
    ap.add_argument("--schematic", default=SCHEMATIC,
                    help="'vector' draws panel (a); a path embeds that image; "
                         "'' leaves the panel blank")
    args = ap.parse_args()
    build_figure(out_path=args.out, schematic=args.schematic)
