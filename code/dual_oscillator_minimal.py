"""Minimal dual-oscillator model of the M-E transcriptional phase gap.

Rebuilt from scratch around five essential experimental observations, one
model ingredient per observation:

  1. The WT M-E gap grows with daylength (~9-12 h) then plateaus near 4 h.
  2. PDF is required for the gap; without it M and E peaks overlap.
  3. PDF is released by the M cell under light.
  4. PDF ADVANCES M and DELAYS E. (Fig. 5D/E cell-specific pdfr knockouts:
     removing PDFR delays M / advances E, so PDF itself does the opposite.
     The pdf0/DD state collapses both cells onto ~ZT15.)
  5. Daylength is encoded by BOTH the M peak time and the gap width.

Design choices that keep it minimal and tie each obs to one mechanism:

  * One shared intrinsic frequency (omega = 2*pi/24) and one shared light
    input, so M and E are IDENTICAL without PDF -> pdf0 collapse is exact and
    automatic (obs 2), no soft prior needed.
  * Light entrains each cell by attraction to a DUSK-ANCHORED reference phase
    theta_D(t) whose peak sits at dusk + lambda. A phase pulled toward
    theta_D while lights are on, then released at dusk, free-runs to its peak
    a fixed lag after lights-off -> the peak locks to dusk regardless of
    photoperiod (obs 5: M tracks dusk). The RESTORING force limits E's
    PDF-driven delay; together with the observed long-day phase cap, this
    produces the gap plateau (obs 1).
  * Effective phase-separating PDF action is gated by light, M's phase, and a
    saturating photoperiod-onset function that is exactly zero through 9 h of
    light (obs 1, 3); it enters M with +alpha (advance) and E with -beta
    (delay) (obs 4).
  * Phase velocity is clamped non-negative (see rhs): PDF and light may HOLD a
    cell but never reverse it, removing the observed source of repeat TIM-peak
    crossings. The fit also penalises any secondary TIM lobe, which peak-time
    targets alone cannot see (see fit_loss).

Six fitted parameters: K, lambda, q, h_P, alpha, beta. omega is fixed.

Regenerate the figure with:  uv run python code/dual_oscillator_minimal.py
Refit with:                  uv run python code/dual_oscillator_minimal.py --fit
"""
from __future__ import annotations

__version__ = "1.0.0"

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from numbers import Integral

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl-minimal"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution

# Publication styling: vector-friendly, editable text, no chartjunk spines.
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,   # embed TrueType so text stays selectable/editable
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure")
# Reuse only the extracted DATA, so the two models never disagree on targets.
from dual_oscillator_pdf import (
    EXPERIMENTAL_GAP_SDS,
    EXPERIMENTAL_PHASE_SDS,
    EXPERIMENTAL_TARGETS,
    GENOTYPE_TARGETS_12,
)

TWOPI = 2.0 * np.pi
OMEGA = TWOPI / 24.0          # shared intrinsic frequency (period 24 h)
PDF_ONSET = 9.0               # no differential PDF action through 9 h of light
PDF_ONSET_POWER = 2.0         # fixed steepness of the saturating onset
# Phase limit: M and E reach their phase limits at ~15 h daylength (paper),
# so dusk-tracking and the PDF onset both stop advancing beyond this. This is
# the minimal, data-pinned stand-in for the phase-ceiling / amplitude-decline
# regime a pure phase model cannot otherwise reach; it makes M ceiling at
# ~ZT16 and E (= M + plateau gap) at ~ZT20 with a single clamp.
P_CAP = 15.0


def p_eff(P):
    """Effective photoperiod for dusk-tracking and PDF activation."""
    return min(P, P_CAP)


@dataclass(frozen=True)
class MinimalParams:
    K: float = 0.35        # strength of the dusk-anchored light attraction
    lam: float = 3.0       # peak locks to dusk + lam (hours) without PDF
    q: float = 2.0         # sharpness of the M-gated PDF release window
    h_P: float = 1.0       # half-activation scale of PDF onset above 9 h
    alpha: float = 0.15    # PDF ADVANCE of M
    beta: float = 0.15     # PDF DELAY of E


# Best fit (differential evolution, seed 1, 60 generations) against the paired-
# bootstrap phase means for the 11 WT conditions plus the 12:12 genotype
# collapse points, with the non-negative-velocity clamp and secondary-lobe
# penalty active. Replacing baseline-plus-linear PDF gain with the thresholded
# saturating onset lowers the full objective from 27.020 to 19.170 and gap SSE
# from 1.85 to 0.63; M absolute-phase SSE rises from 5.36 to 6.59. Every cell
# remains unimodal. Refit with --fit.
TUNED = MinimalParams(
    K=0.09367016459814559,
    lam=2.7226099615653103,
    q=2.9422816823602695,
    h_P=1.491025227677921,
    alpha=0.5116159746740329,
    beta=0.5463532934933886,
)


def light(t: float, P: float) -> float:
    """Square-wave light: on for the first P hours of each 24 h day."""
    return 1.0 if (t % 24.0) < P else 0.0


def light_transition_times(P: float, t_stop: float) -> np.ndarray:
    """Dawn/dusk times where the square-wave forcing is discontinuous.

    ``odeint`` otherwise has no knowledge of these boundaries and may step
    across them.  Supplying them through ``tcrit`` makes the trajectory
    independent of the adaptive solver's incidental step placement.
    """
    if not 0.0 < P < 24.0:
        return np.empty(0)
    days = np.arange(0.0, t_stop // 24.0 + 1.0) * 24.0
    transitions = np.concatenate((days + P, days[1:]))
    return np.unique(transitions[(transitions > 0.0) & (transitions < t_stop)])


def dusk_reference(t: float, P: float, params: MinimalParams) -> float:
    """Phase of a 24 h reference whose peak (theta = pi) sits at dusk + lam.

    theta_D(t) = omega*t + pi - omega*(p_eff + lam):  equals pi when
    (t mod 24) == p_eff + lam, i.e. lambda hours after dusk (dusk-tracking is
    frozen beyond P_CAP, the observed phase limit).
    """
    return OMEGA * t + np.pi - OMEGA * (p_eff(P) + params.lam)


def pdf_drive(phi_M: float, t: float, P: float, params: MinimalParams) -> float:
    """Effective PDF action with exact short-day off and saturating onset."""
    release = max(0.0, np.cos(phi_M)) ** params.q
    x = max(0.0, p_eff(P) - PDF_ONSET)
    onset = x**PDF_ONSET_POWER / (x**PDF_ONSET_POWER + params.h_P**PDF_ONSET_POWER)
    return light(t, P) * release * onset


def rhs(y, t, P, params: MinimalParams, pdf_M: bool, pdf_E: bool):
    """Coupled phase ODEs. pdf_M / pdf_E gate PDF action per cell (knockouts)."""
    phi_M, phi_E = y
    L = light(t, P)
    theta = dusk_reference(t, P, params)
    S = pdf_drive(phi_M, t, P, params)
    a = params.alpha if pdf_M else 0.0
    b = params.beta if pdf_E else 0.0
    dphi_M = OMEGA + params.K * L * np.sin(theta - phi_M) + a * S   # PDF advances M
    dphi_E = OMEGA + params.K * L * np.sin(theta - phi_E) - b * S   # PDF delays E
    # Phase velocity is non-negative: PDF (and the light restoring force) may
    # HOLD a cell, but a transcriptional oscillator does not run backward. This
    # removes reversal-induced extra lobes. Without the clamp, in long days a
    # strong PDF delay can swamp omega and reverse phi_E, making
    # TIM re-cross its peak -> a spurious second lobe near dawn.
    return [max(0.0, dphi_M), max(0.0, dphi_E)]


def simulate(P, params=TUNED, *, pdf_M=True, pdf_E=True, total_days=30, dt=0.02):
    if not np.isfinite(P) or not 0.0 <= P <= 24.0:
        raise ValueError("P must be a finite photoperiod between 0 and 24 hours")
    if not isinstance(total_days, Integral) or isinstance(total_days, bool) or total_days < 1:
        raise ValueError("total_days must be a positive integer")
    if not np.isfinite(dt) or dt <= 0.0 or dt >= total_days * 24.0:
        raise ValueError("dt must be finite, positive, and shorter than the simulation")
    t = np.arange(0.0, total_days * 24.0, dt)
    sol = odeint(rhs, [np.pi, np.pi], t, args=(P, params, pdf_M, pdf_E),
                 rtol=1e-8, atol=1e-9,
                 tcrit=light_transition_times(P, float(t[-1])))
    return t, sol[:, 0], sol[:, 1]


def tim_proxy(phi):
    """TIM transcription proxy, peaking at phi = pi."""
    return np.maximum(0.0, -np.cos(phi)) ** 4


def folded(x):
    return ((x + 12.0) % 24.0) - 12.0


GENOTYPE_FLAGS = {
    "WT": (True, True), "pdf0": (False, False),
    "pdfrM": (False, True), "pdfrE": (True, False),
}


def genotype_flags(genotype):
    """Return PDF-action flags, with a useful error for unknown genotypes."""
    try:
        return GENOTYPE_FLAGS[genotype]
    except KeyError as exc:
        choices = ", ".join(GENOTYPE_FLAGS)
        raise ValueError(f"unknown genotype {genotype!r}; choose one of: {choices}") from exc


def peak_zt(P, params=TUNED, *, genotype="WT", total_days=30, day_index=25):
    """Settled-day M/E peak ZT for a genotype."""
    if not isinstance(day_index, Integral) or isinstance(day_index, bool):
        raise ValueError("day_index must be an integer")
    if not 0 <= day_index < total_days:
        raise ValueError("day_index must identify a day within the simulation")
    pdf_M, pdf_E = genotype_flags(genotype)
    t, pM, pE = simulate(P, params, pdf_M=pdf_M, pdf_E=pdf_E, total_days=total_days)
    day = (t >= day_index * 24) & (t < (day_index + 1) * 24)
    td = t[day] - day_index * 24
    return (float(td[np.argmax(tim_proxy(pM[day]))]),
            float(td[np.argmax(tim_proxy(pE[day]))]))


def secondary_lobe(wave):
    """Height (0..1) of the tallest non-dominant TIM lobe; 0 if unimodal.

    A cell should fire once per settled day; a nonzero value flags a bimodal
    profile (e.g. from a phase reversal). Used to keep the fit honest about
    waveform shape, which peak-time targets alone cannot see.
    """
    top = wave.max()
    if top <= 0:
        return 0.0
    w = wave / top
    # Treat the daily trace as cyclic. Looking only at interior samples misses
    # a lobe whose maximum falls at (or straddles) ZT 0.
    is_max = (w > np.roll(w, 1)) & (w >= np.roll(w, -1)) & (w > 0.05)
    heights = np.sort(w[is_max])
    return float(heights[-2]) if heights.size >= 2 else 0.0


def peak_and_lobes(P, params, *, genotype="WT", total_days=30, day_index=25):
    """Settled-day M/E peak ZT plus each cell's secondary-lobe height."""
    if not isinstance(day_index, Integral) or isinstance(day_index, bool):
        raise ValueError("day_index must be an integer")
    if not 0 <= day_index < total_days:
        raise ValueError("day_index must identify a day within the simulation")
    pdf_M, pdf_E = genotype_flags(genotype)
    t, pM, pE = simulate(P, params, pdf_M=pdf_M, pdf_E=pdf_E, total_days=total_days)
    day = (t >= day_index * 24) & (t < (day_index + 1) * 24)
    td = t[day] - day_index * 24
    wm, we = tim_proxy(pM[day]), tim_proxy(pE[day])
    return (float(td[np.argmax(wm)]), float(td[np.argmax(we)]),
            secondary_lobe(wm), secondary_lobe(we))


# ------------------------------- fitting ---------------------------------

def params_from_vector(x):
    return MinimalParams(K=float(x[0]), lam=float(x[1]), q=float(x[2]),
                         h_P=float(x[3]), alpha=float(x[4]), beta=float(x[5]))


# Penalty on any secondary TIM lobe: peak-time targets cannot see waveform
# shape, so without this the fit is free to (re)introduce a bimodal cell.
LOBE_WEIGHT = 8.0


def fit_loss(x):
    params = params_from_vector(x)
    loss = 0.0
    gaps = []
    for P, (m_t, e_t) in EXPERIMENTAL_TARGETS.items():
        m_s, e_s, m_lobe, e_lobe = peak_and_lobes(P, params, genotype="WT")
        gap_s, gap_t = folded(e_s - m_s), e_t - m_t
        loss += folded(m_s - m_t) ** 2
        loss += folded(e_s - e_t) ** 2
        loss += 6.0 * (gap_s - gap_t) ** 2
        loss += LOBE_WEIGHT * (m_lobe ** 2 + e_lobe ** 2)
        gaps.append(gap_s)
    # soft monotonicity: gap should not shrink as daylength grows
    for g0, g1 in zip(gaps, gaps[1:]):
        loss += 3.0 * max(0.0, g0 - g1 - 0.3) ** 2
    # genotype collapse at 12:12 (obs 2, 4): pdf0 both ~ZT15, pdfrM->M to
    # baseline while E stays delayed, pdfrE->E to baseline while M advanced
    for genotype, (m_t, e_t) in GENOTYPE_TARGETS_12.items():
        m_s, e_s = peak_zt(12, params, genotype=genotype)
        loss += 4.0 * folded(m_s - m_t) ** 2
        loss += 4.0 * folded(e_s - e_t) ** 2
    return float(loss)


def fit(seed=1, maxiter=60):
    bounds = [
        (0.05, 1.5),   # K
        (0.0, 6.0),    # lam
        (0.5, 6.0),    # q
        (0.05, 6.0),   # h_P: half-activation above the 9 h onset
        (0.0, 0.8),    # alpha (M advance)
        (0.0, 0.8),    # beta  (E delay)
    ]
    res = differential_evolution(fit_loss, bounds, seed=seed, maxiter=maxiter,
                                 popsize=12, tol=0.02, polish=True,
                                 updating="deferred", workers=-1)
    return params_from_vector(res.x)


# ------------------------------- reporting -------------------------------

def scorecard(params):
    print(f"K={params.K:.3f}  lambda={params.lam:.2f} h  q={params.q:.2f}  "
          f"h_P={params.h_P:.3f} h  alpha(M adv)={params.alpha:.3f}  "
          f"beta(E del)={params.beta:.3f}")
    print("\n-- WT peak ZT and gap vs data --")
    print(f'{"P":>4} {"dusk+1":>7} {"M tgt":>6} {"M sim":>6} {"E tgt":>6} {"E sim":>6} {"g tgt":>6} {"g sim":>6}')
    gsse = msse = 0.0
    for P, (m_t, e_t) in EXPERIMENTAL_TARGETS.items():
        m_s, e_s = peak_zt(P, params, genotype="WT")
        gs, gt = folded(e_s - m_s), e_t - m_t
        gsse += (gs - gt) ** 2
        msse += folded(m_s - m_t) ** 2
        print(f"{P:>4} {P+1:>7.0f} {m_t:>6.1f} {m_s:>6.2f} {e_t:>6.1f} {e_s:>6.2f} {gt:>6.1f} {gs:>6.2f}")
    print(f"gap SSE = {gsse:.2f}   M-abs SSE = {msse:.2f}")
    print("\n-- 12:12 genotype collapse --")
    for g, (m_t, e_t) in {"WT": (13.0, 17.0), **GENOTYPE_TARGETS_12}.items():
        m_s, e_s = peak_zt(12, params, genotype=g)
        print(f"{g:>6}  M {m_s:>6.2f} (tgt {m_t:>4.1f})   E {e_s:>6.2f} (tgt {e_t:>4.1f})")
    return dict(gap=gsse, mabs=msse)


def build_figure(params, out_path):
    Ps = sorted(EXPERIMENTAL_TARGETS)
    wt = [peak_zt(P, params, genotype="WT") for P in Ps]
    ko = [peak_zt(P, params, genotype="pdf0") for P in Ps]
    mS, eS = zip(*wt)
    dM = [EXPERIMENTAL_TARGETS[P][0] for P in Ps]
    dE = [EXPERIMENTAL_TARGETS[P][1] for P in Ps]

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    ax[0].plot(Ps, [folded(e - m) for m, e in wt], "o-", color="#2E7D32", lw=2.4, label="WT model")
    ax[0].plot(Ps, [folded(e - m) for m, e in ko], "--", color="#C0392B", lw=1.6, label="pdf0 model")
    ax[0].scatter(Ps, [EXPERIMENTAL_TARGETS[P][1] - EXPERIMENTAL_TARGETS[P][0] for P in Ps],
                  color="black", zorder=6, label="data")
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_xlabel("photoperiod (h)"); ax[0].set_ylabel("M-E gap (h)")
    ax[0].set_title("(a) gap vs daylength (obs 1)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    ax[1].plot(Ps, mS, "o-", color="#E67E22", lw=2.4, label="M model")
    ax[1].plot(Ps, eS, "o-", color="#3498DB", lw=2.4, label="E model")
    ax[1].scatter(Ps, dM, color="#E67E22", edgecolor="black", zorder=6, label="M data")
    ax[1].scatter(Ps, dE, color="#3498DB", edgecolor="black", zorder=6, label="E data")
    ax[1].plot(Ps, [min(16.0, max(8.0, P + 1)) for P in Ps], ":", color="gray", lw=1.6, label="dusk+1")
    ax[1].set_ylim(0, 24); ax[1].set_xlabel("photoperiod (h)"); ax[1].set_ylabel("peak ZT (h)")
    ax[1].set_title("(b) M tracks dusk (obs 5)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7, ncol=2)

    genos = ["WT", "pdfrM", "pdfrE", "pdf0"]
    labels = ["WT", "pdfr-M", "pdfr-E", "pdf0"]
    tgt = {"WT": (13.0, 17.0), **GENOTYPE_TARGETS_12}
    xs = np.arange(len(genos))
    mSim = [peak_zt(12, params, genotype=g)[0] for g in genos]
    eSim = [peak_zt(12, params, genotype=g)[1] for g in genos]
    ax[2].scatter(xs - 0.08, [tgt[g][0] for g in genos], s=120, facecolor="none", edgecolor="#E67E22", lw=2, label="M data")
    ax[2].scatter(xs - 0.08, [tgt[g][1] for g in genos], s=120, facecolor="none", edgecolor="#3498DB", lw=2, label="E data")
    ax[2].scatter(xs + 0.08, mSim, s=90, marker="s", color="#E67E22", label="M model")
    ax[2].scatter(xs + 0.08, eSim, s=90, marker="s", color="#3498DB", label="E model")
    ax[2].axhline(15, color="gray", ls=":", lw=0.9)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(labels)
    ax[2].set_ylim(8, 20); ax[2].set_ylabel("peak ZT at 12:12 (h)")
    ax[2].set_title("(c) PDF collapse (obs 2, 4)"); ax[2].grid(alpha=0.3, axis="y"); ax[2].legend(fontsize=8)

    fig.suptitle("Minimal dual-oscillator model: dusk-anchored M, light-gated repulsive PDF",
                 fontsize=13, fontweight="bold", y=1.02)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _rescaled(params, *, pdf=1.0, light=1.0):
    """Params with the PDF drive and/or light input strength rescaled."""
    from dataclasses import replace
    return replace(params, alpha=pdf * params.alpha, beta=pdf * params.beta,
                   K=light * params.K)


def predict_scan(params, kind, scales, P=12):
    """M/E/gap at fixed photoperiod P while scaling 'pdf' or 'light' input."""
    rows = []
    for c in scales:
        p = _rescaled(params, **{kind: c})
        m, e = peak_zt(P, p, genotype="WT")
        rows.append((c, m, e, folded(e - m)))
    return rows


def build_predictions_figure(params, out_path):
    """Two testable predictions at fixed equinoctial photoperiod (12L:12D)."""
    pdf_scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    light_scales = [0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0]
    pdf_rows = predict_scan(params, "pdf", pdf_scales)
    light_rows = predict_scan(params, "light", light_scales)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

    c, m, e, g = zip(*pdf_rows)
    ax[0].plot(c, g, "o-", color="#2E7D32", lw=2.2)
    ax[0].axvline(1.0, color="gray", ls=":", lw=1.0)
    ax[0].text(1.02, 0.3, "wild type", color="gray", fontsize=8, rotation=90, va="bottom")
    ax[0].set_xlabel("PDF-pathway gain (× wild type)")
    ax[0].set_ylabel("M–E gap at 12L:12D (h)")
    ax[0].set_title("(a) Gap is buffered against PDF level")
    ax[0].set_ylim(0, max(6.5, 1.08 * max(g)))
    ax[0].grid(alpha=0.3)
    ax[0].annotate("threshold:\ngap needs PDF", xy=(0.25, pdf_rows[1][3]),
                   xytext=(0.5, 1.2), fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    c, m, e, g = zip(*light_rows)
    ax[1].plot(c, g, "o-", color="#C0392B", lw=2.2)
    ax[1].axvline(1.0, color="gray", ls=":", lw=1.0)
    ax[1].text(1.02, 0.3, "wild type", color="gray", fontsize=8, rotation=90, va="bottom")
    ax[1].set_xlabel("light-input strength (× wild type)")
    ax[1].set_ylabel("M–E gap at 12L:12D (h)")
    ax[1].set_title("(b) Light intensity tunes gap width")
    # Derive the limit from the data so the first point is not silently clipped
    # when retraining changes the weak-light prediction.
    ax[1].set_ylim(0, max(6.5, 1.08 * max(g)))
    ax[1].grid(alpha=0.3)

    fig.suptitle("Minimal-model predictions at a fixed photoperiod",
                 fontsize=12.5, fontweight="bold", y=1.02)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print("\n-- Prediction (a): PDF-pathway gain at 12:12 --")
    print(f'{"xPDF":>5} {"M":>6} {"E":>6} {"gap":>6}')
    for c, m, e, g in pdf_rows:
        print(f"{c:>5.2f} {m:>6.2f} {e:>6.2f} {g:>6.2f}")
    print("\n-- Prediction (b): light-input strength at 12:12 --")
    print(f'{"xK":>5} {"M":>6} {"E":>6} {"gap":>6}')
    for c, m, e, g in light_rows:
        print(f"{c:>5.2f} {m:>6.2f} {e:>6.2f} {g:>6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--predict", action="store_true", help="Also build the predictions figure.")
    ap.add_argument("--out", default=os.path.join(FIGURE_DIR, "minimal_model.pdf"))
    ap.add_argument("--pred-out", default=os.path.join(FIGURE_DIR, "minimal_predictions.pdf"))
    args = ap.parse_args()
    params = fit(maxiter=args.maxiter) if args.fit else TUNED
    if args.fit:
        print("# fitted:", params)
    scorecard(params)
    build_figure(params, args.out)
    print(f"\nSaved: {args.out}")
    if args.predict:
        build_predictions_figure(params, args.pred_out)
        print(f"Saved: {args.pred_out}")


if __name__ == "__main__":
    main()
