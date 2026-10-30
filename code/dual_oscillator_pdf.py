"""
Dual-oscillator (M/E) Drosophila seasonal clock model with PDF action.

The baseline model keeps the original architecture:

    dphi_M/dt = omega_M + Z_M(phi_M) L(t) - alpha R(phi_M)
    dphi_E/dt = omega_E + Z_E(phi_E) L(t) - beta  R(phi_M)

M-derived PDF delays both M and E (attenuated for M: alpha < beta), rather
than advancing M. Real TIM peak data shows M's own peak also shifts
substantially later as day length grows, not staying anchored near dawn --
so both cells need to be pulled later by the same PDF signal, just by
different amounts. In pdf0, the PDF terms are removed and M/E peaks
collapse because both cells retain nearly identical direct light
entrainment; this holds by construction regardless of alpha/beta, since
direct light response is shared between M and E.

The tuned model adds the smallest extension needed to reproduce the data:
the PDF drive scales with daylight exposure, so longer days produce a
larger M-E separation.

Fit targets come from real TIM-RNA speckle data (Figure4_raw.xlsx, extracted
to Figure4_extracted.xlsx): sLNv = M cell, "5th sLNv" = E cell, TIM peak ZT
taken per photoperiod across 11 day-length conditions (see
EXPERIMENTAL_TARGETS below). The pdf0 M-E collapse is not present in this
WT-only dataset and is kept only as a biological prior/regularizer.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-seasonal-clock"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution


TWOPI = 2 * np.pi


@dataclass(frozen=True)
class ModelParams:
    """Parameters for the M/E phase oscillator."""

    omega_M: float = TWOPI / 23.5
    omega_E: float = TWOPI / 25.0
    light_amp: float = 0.30
    light_amp2: float = 0.0
    alpha: float = 0.05
    beta: float = 0.20
    seasonal_pdf_gain: float = 0.0
    light_pdf_gain: float = 0.0
    pdf_width: float = 1.0

    # --- Light PRC shape -------------------------------------------------
    # "harmonic": Z = light_amp cos(phi) + light_amp2 cos(2 phi)  (original)
    # "deadzone": two von Mises lobes with a subjective-day dead zone, the
    #   canonical light-PRC form (delay lobe in early subjective night,
    #   advance lobe in late subjective night, ~0 across subjective day).
    # The dead zone is what allows a LARGE pulse response and stable
    # entrainment at the same time: under a long photophase most of the light
    # falls in the dead zone and contributes no net drive, so the night lobes
    # can be big without wrecking entrainment. The harmonic form has no dead
    # zone, so the fit was forced to shrink it (max|Z| = 0.185 rad/h, i.e. a
    # 1-h pulse could shift phase by at most 0.71 h vs the 5 h observed).
    prc_form: str = "harmonic"
    delay_amp: float = 0.0     # depth of the early-night delay lobe (rad/h)
    delay_phase: float = 3.93  # lobe centre (rad); ~CT15 with phi=0 at CT0
    delay_kappa: float = 4.0   # concentration; larger = narrower lobe
    adv_amp: float = 0.0       # height of the late-night advance lobe (rad/h)
    adv_phase: float = 5.50    # lobe centre (rad); ~CT21
    adv_kappa: float = 4.0

    # --- E-cell PRC (recommendation #2) ----------------------------------
    # None => E uses the same lobes as M, i.e. Z_M = Z_E (the shared PRC).
    # A shared PRC provably cannot fit the Fig. 3 pulse data: in DD the cells
    # free-run with a ~fixed lag, so E's shifts are just M's response curve
    # resampled at that lag, and all six measurements then constrain ONE
    # function -- which they do inconsistently (they demand -3.0 h at M-phase
    # ZT14.6 and +5.0 h at ZT15.0, an 8 h swing in 0.4 h of phase). Giving E
    # its own lobes is therefore required, not an optimization. It is also
    # what the literature expects: M and E receive light through different
    # routes (CRY vs the visual system) with differing sensitivities.
    e_delay_amp: float | None = None
    e_delay_phase: float | None = None
    e_delay_kappa: float | None = None
    e_adv_amp: float | None = None
    e_adv_phase: float | None = None
    e_adv_kappa: float | None = None


BASELINE_PARAMS = ModelParams()

# Fit against real TIM-RNA peak-time data (EXPERIMENTAL_TARGETS below), not
# the earlier screenshot-derived 3-point table. See
# seasonal_clock_tuning_log.md Steps 4-6 for the fitting process, the
# dynamical-instability bugs found along the way, the two-harmonic light
# PRC that closed most of the mid-range gap undershoot, and remaining
# known limitations (the gap still slightly undershoots the data's peak
# around P=12-13; pdf0 doesn't cleanly collapse to 0).
# It is intentionally a modest phase-only fit, not a full TTFL model.
TUNED_PARAMS = ModelParams(
    omega_M=0.2844357314250605,
    omega_E=0.2700122607296771,
    light_amp=0.060,
    light_amp2=-0.096,
    alpha=0.004,
    beta=0.060,
    seasonal_pdf_gain=0.085,
    light_pdf_gain=1.193,
    pdf_width=4.324,
)


# Empirical M/E TIM peak ZT per photoperiod, estimated by the paired,
# stratified bootstrap in plot_experimental_phase.py (10,000 resamples,
# seed=20260729). sLNv = M cell, "5th sLNv" = E cell. Within each ZT, the
# 20 brains are resampled with shared M/E indices; each resampled mean curve's
# phase is its sampling-grid argmax. Values below are the bootstrap phase
# means. LD12_12 LD4 is excluded in favour of LD7; lLNv is excluded.
EXPERIMENTAL_TARGETS = {
    4: (8.0000, 8.0044),
    6: (7.9704, 8.1023),
    9: (10.0007, 10.0804),
    10: (11.0000, 12.9527),
    11: (12.0001, 14.5676),
    12: (12.9986, 16.6982),
    13: (13.9998, 17.5434),
    14: (14.5375, 18.7433),
    15: (15.5903, 19.7497),
    18: (15.9319, 19.5884),
    20: (15.9898, 19.6248),
}

TARGET_GAPS = {p: e - m for p, (m, e) in EXPERIMENTAL_TARGETS.items()}

# Bootstrap SDs for plotting uncertainty. Gap SDs retain the within-brain M/E
# pairing and therefore are not reconstructed from the two marginal SDs.
EXPERIMENTAL_PHASE_SDS = {
    4: (0.0000, 0.0926),
    6: (0.1890, 0.3787),
    9: (0.0264, 0.3706),
    10: (0.0000, 0.5761),
    11: (0.0100, 0.6537),
    12: (0.0374, 0.6563),
    13: (0.0141, 1.4926),
    14: (0.5070, 0.9830),
    15: (0.9633, 0.7275),
    18: (0.3204, 0.8243),
    20: (0.1179, 0.8411),
}

EXPERIMENTAL_GAP_SDS = {
    4: 0.0926,
    6: 0.4204,
    9: 0.3717,
    10: 0.5761,
    11: 0.6538,
    12: 0.6579,
    13: 1.4929,
    14: 1.1080,
    15: 1.2095,
    18: 0.8848,
    20: 0.8495,
}

# EXPERIMENTAL_TARGETS has real conditions only every 1-3 h, with the
# biggest gaps at the long-day end (15 -> 18 -> 20 h). A candidate can be
# unstable strictly *inside* those gaps while passing at the checked
# endpoints; these intermediate points carry no M/E/gap target, only an
# instability penalty in fit_loss, to close that blind spot.
STABILITY_ONLY_CHECKPOINTS = (5, 7, 8, 16, 17, 19)

# Representative photoperiods for the waveform panels.
SEASON_LABELS = {
    6: "Short day (6L:18D)",
    12: "Equinox (12L:12D)",
    18: "Long day (18L:6D)",
}


def light(t: float, photoperiod: float) -> float:
    """Square-wave light cycle, photoperiod hours of light per 24 h."""
    return 1.0 if (t % 24.0) < photoperiod else 0.0


def light_transition_times(photoperiod: float, t_stop: float) -> np.ndarray:
    """Dawn/dusk times where the square-wave forcing is discontinuous."""
    if not 0.0 < photoperiod < 24.0:
        return np.empty(0)
    days = np.arange(0.0, t_stop // 24.0 + 1.0) * 24.0
    transitions = np.concatenate((days + photoperiod, days[1:]))
    return np.unique(transitions[(transitions > 0.0) & (transitions < t_stop)])


def deadzone_prc(phi, params: ModelParams, cell: str = "M"):
    """Canonical light PRC: two von Mises lobes with a subjective-day dead zone.

    Z(phi) = -delay_amp * K(phi; delay_phase, delay_kappa)
             +   adv_amp * K(phi; adv_phase,   adv_kappa)

    with K(phi; mu, kappa) = exp(kappa*(cos(phi-mu) - 1)), a periodic bump
    equal to 1 at phi=mu and decaying with width ~1/sqrt(kappa). Away from
    both lobes Z ~ 0 -- the dead zone. Negative = delay, positive = advance,
    matching the sign convention of the harmonic form (Z enters dphi/dt).

    cell="E" uses the e_* lobes when they are set, giving Z_M != Z_E.
    """
    if cell == "E" and params.e_delay_amp is not None:
        d_amp, d_phase, d_kappa = params.e_delay_amp, params.e_delay_phase, params.e_delay_kappa
        a_amp, a_phase, a_kappa = params.e_adv_amp, params.e_adv_phase, params.e_adv_kappa
    else:
        d_amp, d_phase, d_kappa = params.delay_amp, params.delay_phase, params.delay_kappa
        a_amp, a_phase, a_kappa = params.adv_amp, params.adv_phase, params.adv_kappa

    def lobe(mu, kappa):
        return np.exp(kappa * (np.cos(phi - mu) - 1.0))

    return -d_amp * lobe(d_phase, d_kappa) + a_amp * lobe(a_phase, a_kappa)


def cell_light_prc(phi, params: ModelParams, cell: str = "M"):
    """Light PRC for one cell, shape selected by params.prc_form.

    "harmonic" (original): two harmonics instead of one, so the fit can skew
    the advance/delay zone widths, which a single cosine cannot do. It has no
    dead zone, which caps how large it can get without breaking entrainment,
    and it is always shared between M and E.

    "deadzone": the canonical two-lobe form with a subjective-day dead zone,
    optionally per-cell (see deadzone_prc).
    """
    if params.prc_form == "deadzone":
        return deadzone_prc(phi, params, cell)
    return params.light_amp * np.cos(phi) + params.light_amp2 * np.cos(2.0 * phi)


def direct_light_prc(phi: float, params: ModelParams) -> float:
    """M-cell (or shared) light PRC. Kept for callers that predate per-cell PRCs."""
    return cell_light_prc(phi, params, "M")


def pdf_release(phi_M: float, params: ModelParams) -> float:
    """PDF release gated to M's subjective morning."""
    return max(0.0, np.cos(phi_M)) ** params.pdf_width


def pdf_drive(
    phi_M: float,
    t: float,
    photoperiod: float,
    params: ModelParams,
    *,
    variant: str,
) -> float:
    """M-derived PDF signal.

    baseline:
        fixed PDF release, independent of day length.
    daylight_pdf:
        the same M-gated release, multiplied by daylight and photoperiod
        gain. This is the minimal extension that makes the M-E gap grow
        from winter to summer.
    """
    release = pdf_release(phi_M, params)
    if variant == "baseline":
        return release
    if variant == "daylight_pdf":
        seasonal = 1.0 + params.seasonal_pdf_gain * max(0.0, photoperiod - 8.0) / 8.0
        daylight = 1.0 + params.light_pdf_gain * light(t, photoperiod)
        return release * seasonal * daylight
    if variant == "repulsive_pdf":
        # Light-GATED release: PDF is released from LNvs only upon light
        # excitation (depolarization-dependent), so the drive vanishes in
        # darkness and at very short photoperiods, and both cells revert to
        # their shared baseline phase (structural collapse -- matches the DD
        # gap loss and the short-day convergence, without a soft prior). The
        # multiplicative light gate replaces daylight_pdf's (1 + g_L L), which
        # kept a full baseline release even in the dark. Photoperiod gain
        # still scales the drive on long days. Note light_pdf_gain is unused
        # here.
        seasonal = 1.0 + params.seasonal_pdf_gain * max(0.0, photoperiod - 8.0) / 8.0
        return release * seasonal * light(t, photoperiod)
    raise ValueError(f"Unknown model variant: {variant}")


def rhs(
    y: np.ndarray,
    t: float,
    photoperiod: float,
    params: ModelParams,
    *,
    pdf_on: bool,
    variant: str,
    pdf_M: bool = True,
    pdf_E: bool = True,
) -> list[float]:
    """Right-hand side of the coupled ODEs.

    pdf_M / pdf_E gate PDF action on each cell independently, so the
    cell-type-specific pdfr knockouts can be simulated directly:
    WT = (True, True), M-pdfr KO = (False, True), E-pdfr KO = (True, False),
    pdf0 / Gas KO = both off (equivalently pdf_on=False).

    Sign convention differs by variant:
      * repulsive_pdf: M-derived PDF *advances* M (+alpha) and *delays* E
        (-beta) around a shared baseline, matching the paper's cell-specific
        pdfr knockouts (M-pdfr KO delays M; E-pdfr KO advances E) and the
        pdf0/DD collapse to ~CT15.
      * baseline / daylight_pdf: PDF *delays* both cells (-alpha, -beta).
    """
    phi_M, phi_E = y
    L = light(t, photoperiod)
    drive = pdf_drive(phi_M, t, photoperiod, params, variant=variant) if pdf_on else 0.0
    a = params.alpha if pdf_M else 0.0
    b = params.beta if pdf_E else 0.0
    zM = cell_light_prc(phi_M, params, "M")
    zE = cell_light_prc(phi_E, params, "E")
    if variant == "repulsive_pdf":
        dphi_M = params.omega_M + zM * L + a * drive
        dphi_E = params.omega_E + zE * L - b * drive
    else:
        dphi_M = params.omega_M + zM * L - a * drive
        dphi_E = params.omega_E + zE * L - b * drive
    return [dphi_M, dphi_E]


def simulate(
    photoperiod: float,
    params: ModelParams = TUNED_PARAMS,
    *,
    pdf_on: bool = True,
    variant: str = "daylight_pdf",
    pdf_M: bool = True,
    pdf_E: bool = True,
    total_days: int = 45,
    dt: float = 0.05,
    phi_M_0: float = 0.0,
    phi_E_0: float = np.pi,
    rtol: float = 1e-9,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the model. Returns (t, phi_M, phi_E)."""
    t = np.arange(0, total_days * 24, dt)
    wrapped_rhs = lambda y, tt, P, p: rhs(
        y, tt, P, p, pdf_on=pdf_on, variant=variant, pdf_M=pdf_M, pdf_E=pdf_E
    )
    sol = odeint(
        wrapped_rhs,
        [phi_M_0, phi_E_0],
        t,
        args=(photoperiod, params),
        tfirst=False,
        rtol=rtol,
        atol=atol,
        # The light input jumps at dawn and dusk. Without tcrit, LSODA may
        # step over a jump and produce tolerance-dependent phase estimates.
        tcrit=light_transition_times(photoperiod, float(t[-1])),
    )
    return t, sol[:, 0], sol[:, 1]


def tim_transcription_proxy(phi: np.ndarray) -> np.ndarray:
    """Proxy for TIM transcription, peaking at phi = pi."""
    val = -np.cos(phi)
    return np.maximum(0, val) ** 4


def tim_peak_events(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Times at which phi crosses pi (mod 2*pi), i.e. TIM peak times."""
    shifted = (phi - np.pi) % TWOPI
    crossings = np.where(np.diff(shifted) < -np.pi)[0]
    return t[crossings]


def folded_gap(e_time: float, m_time: float) -> float:
    """Signed E-minus-M gap folded to [-12, 12)."""
    return float(((e_time - m_time + 12.0) % 24.0) - 12.0)


def circular_diff(a: float, b: float) -> float:
    """Shortest signed difference a-b on a 24 h circle, in [-12, 12)."""
    return float(((a - b + 12.0) % 24.0) - 12.0)


def m_to_e_gap(
    photoperiod: float,
    params: ModelParams = TUNED_PARAMS,
    *,
    pdf_on: bool = True,
    variant: str = "daylight_pdf",
    total_days: int = 60,
    settle_days: int = 40,
) -> float:
    """Mean time from M TIM peak to nearest E TIM peak after settling."""
    t, pM, pE = simulate(photoperiod, params, pdf_on=pdf_on, variant=variant, total_days=total_days)
    settled = t > settle_days * 24
    t_M = tim_peak_events(pM[settled], t[settled])
    t_E = tim_peak_events(pE[settled], t[settled])
    if len(t_M) < 2 or len(t_E) < 2:
        return np.nan
    gaps = []
    for tM in t_M[1:-1]:
        closest_E = t_E[np.argmin(np.abs(t_E - tM))]
        gaps.append(folded_gap(closest_E, tM))
    if len(gaps) < 3:
        return np.nan
    return float(np.mean(gaps[-10:]))


def settled_day_peak_times(
    photoperiod: float,
    params: ModelParams = TUNED_PARAMS,
    *,
    pdf_on: bool = True,
    variant: str = "daylight_pdf",
    total_days: int = 45,
    day_index: int = 40,
) -> tuple[float, float]:
    """Peak times from the TIM proxy on one settled LD day."""
    t, pM, pE = simulate(photoperiod, params, pdf_on=pdf_on, variant=variant, total_days=total_days)
    day = (t >= day_index * 24) & (t < (day_index + 1) * 24)
    td = t[day] - day_index * 24
    tim_M = tim_transcription_proxy(pM[day])
    tim_E = tim_transcription_proxy(pE[day])
    return float(td[np.argmax(tim_M)]), float(td[np.argmax(tim_E)])


def settled_peaks_with_lock_error(
    photoperiod: float,
    params: ModelParams,
    *,
    variant: str = "daylight_pdf",
    total_days: int = 45,
    settle_days: int = 25,
    n_check_days: int = 10,
    rtol: float = 1e-9,
    atol: float = 1e-10,
) -> tuple[float, float, float]:
    """WT M/E peak ZT plus an instability penalty (relative-coordination check).

    Two distinct failure modes are checked, both invisible to a single
    settled-day snapshot:

    1. "Relative coordination": a quasi-periodic beat between a cell and the
       light cycle, where the *average* phase velocity looks correctly
       locked over a short window even though the daily peak time wanders
       substantially cycle to cycle (confirmed empirically: E's peak on day
       30 vs day 40 of the same run can differ by >10 h at long
       photoperiods). Caught by the circular spread of peak times sampled
       on several evenly spaced days in the settled tail; ~0 if locked.
    2. Oscillator arrest: strong opposing PDF/light forcing can pin a phase
       at a fixed point (dphi/dt = 0), freezing it entirely. This still
       looks "stable" day to day (spread ~0, since it isn't moving at all)
       but the peak is a numerical artifact, not a real rhythm. Caught by
       checking that the average phase velocity is actually close to the
       target 2*pi/24 rather than near zero (confirmed empirically: cutting
       M's weight in the fit loss let alpha/light_amp grow large enough
       that M's phase froze and its "peak" collapsed to ZT0 every day).
    """
    t, pM, pE = simulate(
        photoperiod, params, pdf_on=True, variant=variant,
        total_days=total_days, rtol=rtol, atol=atol,
    )
    # Integer day indices: idx*24 must land on a whole-day boundary (a
    # multiple of 24, matching t=0 = ZT0) or the extracted "peak time" is
    # shifted by a spurious fractional offset instead of true ZT.
    check_days = np.unique(np.linspace(settle_days, total_days - 1, n_check_days).astype(int))

    def day_peak(idx: int, phi: np.ndarray) -> float:
        day = (t >= idx * 24) & (t < (idx + 1) * 24)
        td = t[day] - idx * 24
        tim = tim_transcription_proxy(phi[day])
        return float(td[np.argmax(tim)])

    def circular_spread(times: list[float]) -> float:
        angles = np.array(times) * (TWOPI / 24.0)
        resultant = np.abs(np.mean(np.exp(1j * angles)))
        return 1.0 - resultant  # 0 = perfectly stable, ~1 = wildly unstable

    start_idx = int(np.searchsorted(t, settle_days * 24))
    target_omega = TWOPI / 24.0

    def rotation_error(phi: np.ndarray) -> float:
        avg_freq = (phi[-1] - phi[start_idx]) / (t[-1] - t[start_idx])
        return ((avg_freq - target_omega) / target_omega) ** 2

    m_times = [day_peak(d, pM) for d in check_days]
    e_times = [day_peak(d, pE) for d in check_days]
    instability = (
        circular_spread(m_times)
        + circular_spread(e_times)
        + rotation_error(pM)
        + rotation_error(pE)
    )
    return m_times[-1], e_times[-1], instability


def params_from_vector(x: np.ndarray) -> ModelParams:
    """Convert optimizer vector to ModelParams."""
    return ModelParams(
        omega_M=float(x[0]),
        omega_E=float(x[1]),
        light_amp=float(x[2]),
        light_amp2=float(x[3]),
        alpha=float(x[4]),
        beta=float(x[5]),
        seasonal_pdf_gain=float(x[6]),
        light_pdf_gain=float(x[7]),
        pdf_width=float(x[8]),
    )


def fit_loss(x: np.ndarray) -> float:
    """Loss against the real experimental M/E TIM peak-time targets.

    Four terms on the WT (pdf_on=True) simulation, evaluated at all 11
    photoperiods in EXPERIMENTAL_TARGETS:
      1. absolute M and E peak ZT error (gets the overall entrainment right),
      2. M-E gap error, weighted more heavily than (1) alone would imply,
         since a model can match M and E absolute times individually while
         still getting their separation wrong,
      3. a soft monotonicity prior discouraging the gap from shrinking as
         photoperiod grows (matches the rise-then-plateau shape of the data),
      4. a lock-error penalty (see settled_peaks_with_lock_error) that
         dominates the loss whenever the phase oscillators fail to settle
         onto a stable 24 h-periodic orbit, e.g. under strong PDF forcing at
         long photoperiods. Without this term the optimizer can "cheat" by
         picking parameters where the single-day snapshot happens to look
         right but the trajectory isn't actually locked to the light cycle.
    A separate, lightly weighted prior pulls the pdf0 gap toward zero at a
    few representative photoperiods; this is a biological expectation
    (Yao & Shafer-style pdf-null collapse), not something this WT-only
    dataset can directly fit.

    EXPERIMENTAL_TARGETS has real conditions only every 1-3 h, with gaps at
    the long-day end (15 -> 18 -> 20). A candidate can be genuinely unstable
    *inside* those gaps while still passing (or narrowly passing) at the
    checked endpoints -- confirmed empirically: a fit that scored fine at
    P=15/18/20 individually produced a wild dip-and-spike in E's peak time
    between them, and P=19/20 turned out to carry non-negligible instability
    that the earlier weight (300x) wasn't enough to reject given the gains
    elsewhere. STABILITY_ONLY_CHECKPOINTS adds instability-only penalties
    (no data at these points, so no M/E/gap target) at intermediate
    photoperiods to close that blind spot, at a higher weight.
    """
    params = params_from_vector(x)
    loss = 0.0
    sim_gaps = []
    for photoperiod, (m_target, e_target) in EXPERIMENTAL_TARGETS.items():
        m_sim, e_sim, instability = settled_peaks_with_lock_error(photoperiod, params)
        # instability is 0-2 (circular spread); make it dominate whenever
        # the trajectory isn't cleanly locked to the 24 h light cycle.
        loss += 800.0 * instability
        gap_sim = folded_gap(e_sim, m_sim)
        gap_target = e_target - m_target
        loss += circular_diff(m_sim, m_target) ** 2
        loss += circular_diff(e_sim, e_target) ** 2
        loss += 8.0 * (gap_sim - gap_target) ** 2
        sim_gaps.append(gap_sim)

    for photoperiod in STABILITY_ONLY_CHECKPOINTS:
        _, _, instability = settled_peaks_with_lock_error(photoperiod, params)
        loss += 800.0 * instability

    for earlier, later in zip(sim_gaps, sim_gaps[1:]):
        loss += 3.0 * max(0.0, earlier - later - 0.3) ** 2

    for photoperiod in (6, 12, 18):
        ko_gap = m_to_e_gap(
            photoperiod,
            params,
            pdf_on=False,
            variant="daylight_pdf",
            total_days=30,
            settle_days=20,
        )
        if not np.isfinite(ko_gap):
            return 1e6
        loss += 2.0 * ko_gap**2
    return float(loss)


def fit_parameters(seed: int = 3, maxiter: int = 40) -> ModelParams:
    """Fit parameters with a bounded global search."""
    bounds = [
        (TWOPI / 25.0, TWOPI / 22.0),  # omega_M
        (TWOPI / 26.0, TWOPI / 23.0),  # omega_E
        (0.05, 0.60),                  # light_amp
        (-0.30, 0.30),                 # light_amp2 (second-harmonic PRC skew)
        (0.0, 0.50),                   # alpha (M's PDF delay, expected < beta)
        (0.02, 0.50),                  # beta
        (0.0, 4.0),                    # seasonal_pdf_gain
        (0.0, 3.0),                    # light_pdf_gain
        (0.5, 5.0),                    # pdf_width
    ]
    result = differential_evolution(
        fit_loss,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=10,
        tol=0.02,
        polish=False,
        updating="deferred",
        workers=-1,
    )
    return params_from_vector(result.x)


# =========================================================================
# repulsive_pdf variant: anti-symmetric, light-gated PDF coupling
# =========================================================================
#
# Fit against the same 11 WT peak-time conditions PLUS the cell-specific
# knockout collapse points from the manuscript (Fig. 5), which the WT-only
# daylight_pdf fit had to leave as a soft prior. Produced by
# fit_parameters_repulsive (code/fit_repulsive.py, seed=3, maxiter=60).
# alpha is now M's PDF *advance*, beta is E's PDF *delay*.
#
# What this variant fixes relative to daylight_pdf (see print_comparison /
# --compare for the full scorecard):
#   * pdf0 / Gas collapse is now STRUCTURAL and near-exact (M,E ~= ZT14.2,
#     gap ~0) instead of a broken soft prior (old: -3.8..+2.5 h).
#   * the short-day gap collapses to ~0 (P=4/6/9), instead of the old
#     model's spurious ~1.4-2 h gap there -- because release is light-gated.
#   * the cell-specific pdfr knockouts are reproduced (M-KO -> M falls to
#     baseline while E stays delayed; E-KO -> E falls to baseline while M
#     stays advanced), which the both-delay topology structurally cannot do.
#   * On the revised bootstrap targets, summed squared WT gap error is ~8.9
#     vs ~13.5 for the daylight_pdf optimum.
#
# Known residuals (map onto the "next directions" not yet implemented):
#   * M's absolute peak is still ~1-2 h early at mid/long days (data lock M
#     to dusk+1; shared-PRC entrainment parks it a bit early) -- needs the
#     explicit dusk-anchoring of M (direction #2).
#   * E overshoots its ZT20 ceiling at very long days (P=20 gap ~5.3) -- no
#     phase-limit saturation in a pure phase model (direction #4/#5).
TUNED_PARAMS_REPULSIVE = ModelParams(
    omega_M=0.261523000887951,     # M intrinsic period ~24.03 h
    omega_E=0.2620349473318723,    # E intrinsic period ~23.98 h
    light_amp=0.05528827747868342,
    light_amp2=-0.12934072888224687,
    alpha=0.3321939409292144,      # PDF ADVANCE on M (+)
    beta=0.07701058606669817,      # PDF DELAY on E (-)
    seasonal_pdf_gain=0.05520581885336351,
    light_pdf_gain=0.0,            # unused by repulsive_pdf
    pdf_width=1.7039482238209083,
)


# Per-cell PDF gating for each genotype in the paper's Fig. 5 experiments.
GENOTYPE_FLAGS = {
    "WT":    dict(pdf_on=True,  pdf_M=True,  pdf_E=True),
    "pdf0":  dict(pdf_on=False, pdf_M=False, pdf_E=False),  # pdf0 / Gas KO
    "pdfrM": dict(pdf_on=True,  pdf_M=False, pdf_E=True),   # M-specific pdfr KO
    "pdfrE": dict(pdf_on=True,  pdf_M=True,  pdf_E=False),  # E-specific pdfr KO
}

# 12L:12D genotype peak-ZT targets from Fig. 5 (paper). These identify the
# shared baseline (~ZT15, where both cells sit with no PDF) and the PDF
# advance/delay magnitudes (~2 h each at equinox) that a WT-only fit cannot
# separate: pdf0/Gas collapse both to ZT15; M-pdfr KO removes M's advance so
# M falls back to ZT15 while E stays delayed at ZT17; E-pdfr KO removes E's
# delay so E falls back to ZT15 while M stays advanced at ZT13.
GENOTYPE_TARGETS_12 = {
    "pdf0":  (15.0, 15.0),
    "pdfrM": (15.0, 17.0),
    "pdfrE": (13.0, 15.0),
}

# Looser solver settings + shorter runs used only inside the optimizer, where
# thousands of evaluations make the accurate settings (1e-9, 45 days) too
# slow. The reported params are re-scored at full accuracy afterwards.
FIT_RTOL = 1e-6
FIT_ATOL = 1e-8
FIT_DAYS = 35
FIT_SETTLE = 20


def genotype_peaks(
    photoperiod: float,
    params: ModelParams,
    *,
    variant: str,
    genotype: str,
    total_days: int = 45,
    day_index: int = 40,
    rtol: float = 1e-9,
    atol: float = 1e-10,
) -> tuple[float, float]:
    """Settled-day M/E peak ZT for one genotype (WT / pdf0 / pdfrM / pdfrE)."""
    flags = GENOTYPE_FLAGS[genotype]
    t, pM, pE = simulate(
        photoperiod, params, variant=variant, total_days=total_days,
        rtol=rtol, atol=atol, **flags,
    )
    day = (t >= day_index * 24) & (t < (day_index + 1) * 24)
    td = t[day] - day_index * 24
    tim_M = tim_transcription_proxy(pM[day])
    tim_E = tim_transcription_proxy(pE[day])
    return float(td[np.argmax(tim_M)]), float(td[np.argmax(tim_E)])


# Measured 1-h light-pulse phase shifts (manuscript Fig. 3). Convention:
#   Delta = (pulsed peak ZT) - (control peak ZT), folded to [-12, 12)
#   Delta > 0 = DELAY (peak later);  Delta < 0 = ADVANCE (peak earlier)
# These are effectively a direct readout of the M and E light PRCs, and are
# the constraint the entrainment-only fits were missing: steady-state peak
# positions pin down only the *integral* of Z over the photophase, not its
# shape, which is why the harmonic PRC drifted to a non-biological form.
PULSE_TARGETS = {
    15: (+5.0, +1.0),   # early night: M delayed 5 h,  E delayed 1 h
    18: (+1.0, -3.0),   # midnight:    M delayed 1 h,  E advanced 3 h
    21: (-1.0, -4.5),   # late night:  M advanced 1 h, E advanced ~4.5 h
}


def _pulse_light(t, photoperiod, settle_days, pulse_zt, pulse_dur):
    """LD for settle_days, then DD; optional 1-h pulse on the last LD night."""
    day = int(t // 24)
    zt = t % 24.0
    base = 0.0 if day > settle_days else (1.0 if zt < photoperiod else 0.0)
    if pulse_zt is not None:
        t0 = settle_days * 24 + pulse_zt
        if t0 <= t < t0 + pulse_dur:
            return 1.0
    return base


def _pulse_rhs(y, t, params, variant, photoperiod, settle_days, pulse_zt, pulse_dur):
    phi_M, phi_E = y
    L = _pulse_light(t, photoperiod, settle_days, pulse_zt, pulse_dur)
    release = max(0.0, np.cos(phi_M)) ** params.pdf_width
    seasonal = 1.0 + params.seasonal_pdf_gain * max(0.0, photoperiod - 8.0) / 8.0
    zM = cell_light_prc(phi_M, params, "M")
    zE = cell_light_prc(phi_E, params, "E")
    if variant == "repulsive_pdf":
        S = release * seasonal * L
        return [params.omega_M + zM * L + params.alpha * S,
                params.omega_E + zE * L - params.beta * S]
    daylight = 1.0 + params.light_pdf_gain * L
    S = release * seasonal * daylight
    return [params.omega_M + zM * L - params.alpha * S,
            params.omega_E + zE * L - params.beta * S]


def pulse_shifts(
    params: ModelParams,
    *,
    variant: str = "repulsive_pdf",
    pulse_zts=tuple(PULSE_TARGETS),
    photoperiod: float = 12.0,
    settle_days: int = 12,
    pulse_dur: float = 1.0,
    rtol: float = 1e-7,
    atol: float = 1e-8,
    hmax: float = 0.1,
) -> dict:
    """1-h light-pulse M/E phase shifts, measured on DD1 (see PULSE_TARGETS).

    Entrain to LD for settle_days, apply the pulse on the last LD night,
    release to DD, read the M/E peak on DD1 and subtract the unpulsed control.

    hmax MUST stay bounded: the pulse is a narrow discontinuity sitting in an
    otherwise smooth DD stretch, so an unconstrained adaptive step will jump
    straight over it and report a spurious zero shift.
    """
    total_days = settle_days + 3
    t = np.arange(0, total_days * 24, 0.02)
    dd1 = settle_days + 1
    day_mask = (t >= dd1 * 24) & (t < (dd1 + 1) * 24)
    td = t[day_mask] - dd1 * 24

    def run(pulse_zt):
        sol = odeint(
            _pulse_rhs, [0.0, np.pi], t,
            args=(params, variant, photoperiod, settle_days, pulse_zt, pulse_dur),
            rtol=rtol, atol=atol, hmax=hmax,
        )
        return sol[:, 0], sol[:, 1]

    def day_peak(phi):
        return float(td[np.argmax(tim_transcription_proxy(phi[day_mask]))])

    cM, cE = run(None)
    m0, e0 = day_peak(cM), day_peak(cE)
    out = {}
    for zt in pulse_zts:
        pM, pE = run(float(zt))
        out[zt] = (circular_diff(day_peak(pM), m0), circular_diff(day_peak(pE), e0))
    return out


def params_from_vector_repulsive(x: np.ndarray) -> ModelParams:
    """Convert the 8-element repulsive-variant optimizer vector to ModelParams.

    Same order as the daylight_pdf vector but with light_pdf_gain dropped
    (the light-gated release makes it redundant); alpha is M's PDF *advance*,
    beta is E's PDF *delay*.
    """
    return ModelParams(
        omega_M=float(x[0]),
        omega_E=float(x[1]),
        light_amp=float(x[2]),
        light_amp2=float(x[3]),
        alpha=float(x[4]),
        beta=float(x[5]),
        seasonal_pdf_gain=float(x[6]),
        light_pdf_gain=0.0,
        pdf_width=float(x[7]),
    )


def fit_loss_repulsive(x: np.ndarray) -> float:
    """Loss for the repulsive_pdf variant.

    Same WT peak-time / gap / monotonicity / instability structure as
    fit_loss, plus hard-weighted 12:12 genotype collapse terms (pdf0, pdfrM,
    pdfrE) that pin the baseline and the PDF advance/delay magnitudes to the
    Fig. 5 data. The pdf0 collapse is now a real fit target, not the soft
    WT-only prior the daylight_pdf fit used.
    """
    params = params_from_vector_repulsive(x)
    loss = 0.0
    sim_gaps = []
    for photoperiod, (m_target, e_target) in EXPERIMENTAL_TARGETS.items():
        m_sim, e_sim, instability = settled_peaks_with_lock_error(
            photoperiod, params, variant="repulsive_pdf",
            total_days=FIT_DAYS, settle_days=FIT_SETTLE, n_check_days=8,
            rtol=FIT_RTOL, atol=FIT_ATOL,
        )
        loss += 800.0 * instability
        gap_sim = folded_gap(e_sim, m_sim)
        gap_target = e_target - m_target
        loss += circular_diff(m_sim, m_target) ** 2
        loss += circular_diff(e_sim, e_target) ** 2
        loss += 8.0 * (gap_sim - gap_target) ** 2
        sim_gaps.append(gap_sim)

    for photoperiod in STABILITY_ONLY_CHECKPOINTS:
        _, _, instability = settled_peaks_with_lock_error(
            photoperiod, params, variant="repulsive_pdf",
            total_days=FIT_DAYS, settle_days=FIT_SETTLE, n_check_days=8,
            rtol=FIT_RTOL, atol=FIT_ATOL,
        )
        loss += 800.0 * instability

    for earlier, later in zip(sim_gaps, sim_gaps[1:]):
        loss += 3.0 * max(0.0, earlier - later - 0.3) ** 2

    # NOTE: a gap-continuity penalty across the full P grid was tried to
    # smooth a real (but off-data) entrainment wobble the best fit shows at
    # the un-fit P=5. It smoothed the curve but degraded the cell-specific
    # pdfr knockouts (the core result) and the gap fit, so it was dropped.
    # The P=5 wobble is documented as a known residual instead.

    # Genotype collapse constraints at 12:12 (now real data, weighted hard).
    for genotype, (m_target, e_target) in GENOTYPE_TARGETS_12.items():
        m_sim, e_sim = genotype_peaks(
            12, params, variant="repulsive_pdf", genotype=genotype,
            total_days=FIT_DAYS, day_index=FIT_DAYS - 5,
            rtol=FIT_RTOL, atol=FIT_ATOL,
        )
        loss += 5.0 * circular_diff(m_sim, m_target) ** 2
        loss += 5.0 * circular_diff(e_sim, e_target) ** 2
    return float(loss)


def fit_parameters_repulsive(seed: int = 3, maxiter: int = 40) -> ModelParams:
    """Fit the repulsive_pdf variant with a bounded global search (8 params)."""
    bounds = [
        (TWOPI / 25.0, TWOPI / 22.0),  # omega_M
        (TWOPI / 26.0, TWOPI / 23.0),  # omega_E
        (0.05, 0.60),                  # light_amp
        (-0.30, 0.30),                 # light_amp2
        (0.0, 0.50),                   # alpha = PDF advance on M
        (0.0, 0.50),                   # beta  = PDF delay on E
        (0.0, 4.0),                    # seasonal_pdf_gain
        (0.5, 5.0),                    # pdf_width
    ]
    result = differential_evolution(
        fit_loss_repulsive,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=10,
        tol=0.02,
        polish=False,
        updating="deferred",
        workers=-1,
    )
    return params_from_vector_repulsive(result.x)


# =========================================================================
# deadzone PRC variant (recommendation #1)
# =========================================================================
# Same repulsive_pdf PDF topology (advance M / delay E, light-gated release),
# but the shared light PRC is replaced by the canonical two-lobe dead-zone
# form, and the Fig. 3 light-pulse shifts are added as fit targets. The PRC
# is still SHARED between M and E here; per-cell PRCs are the next step.

# Result of fit_parameters_deadzone (seed=3, maxiter=60). NOT ADOPTED as the
# model default -- kept for reproducibility and for the comparison figure.
#
# The dead-zone machinery itself works: night/day sensitivity ratio 11.8 (a
# real dead zone; the harmonic PRC's is 1.0) and the 1-h pulse ceiling rises
# from 0.71 h to 2.22 h. But it buys almost nothing (pulse SSE 48.9 vs 54.9)
# while badly degrading entrainment (gap SSE 33.4 vs 9.5, M-abs SSE 27.9 vs
# 13.6), and the optimizer left most of the available amplitude headroom
# unused (max|Z| 0.58 of a permitted 3.0).
#
# The reason is that a SHARED PRC, not the PRC shape, is the binding
# constraint. In DD the two cells free-run with a ~fixed phase lag, so with
# Z_M = Z_E the E-cell shifts are just the M-cell response curve resampled at
# that lag -- all six measured shifts constrain ONE function. Mapped onto M's
# phase axis they demand: ZT14.6 -> -3.0 h, ZT15.0 -> +5.0 h, ZT17.6 -> -4.5 h,
# ZT18.0 -> +1.0 h. That is an 8 h swing within 0.4 h of circadian phase
# (~20 h of shift per h of phase), which no smooth PRC can do. The six targets
# are mutually unsatisfiable for any shared Z, which is why the optimizer had
# no gradient toward exploiting the dead zone's extra amplitude.
#
# => Per-cell PRCs (Z_M != Z_E) are required, not optional. Recommendation #2.
TUNED_PARAMS_DEADZONE = ModelParams(
    omega_M=0.2620620967598146,
    omega_E=0.2609703779174815,
    alpha=0.15321326458500392,
    beta=0.015927080678317496,
    seasonal_pdf_gain=3.113955814730805,
    pdf_width=4.507627625230175,
    prc_form="deadzone",
    delay_amp=0.3683342109430918,
    delay_phase=3.6036395956152387,
    delay_kappa=5.360407247489572,
    adv_amp=0.5959766442979528,
    adv_phase=4.744034855291796,
    adv_kappa=8.5253769762272,
    light_amp=0.0, light_amp2=0.0, light_pdf_gain=0.0,
)


DEADZONE_BOUNDS = [
    (TWOPI / 25.0, TWOPI / 22.0),  # omega_M
    (TWOPI / 26.0, TWOPI / 23.0),  # omega_E
    (0.0, 0.60),                   # alpha (PDF advance on M)
    (0.0, 0.60),                   # beta  (PDF delay on E)
    (0.0, 4.0),                    # seasonal_pdf_gain
    (0.5, 6.0),                    # pdf_width
    (0.0, 3.0),                    # delay_amp
    (2.0, 5.2),                    # delay_phase (early subjective night)
    (1.0, 20.0),                   # delay_kappa
    (0.0, 3.0),                    # adv_amp
    (4.0, 7.5),                    # adv_phase (late subjective night)
    (1.0, 20.0),                   # adv_kappa
]


def params_from_vector_deadzone(x: np.ndarray) -> ModelParams:
    """Convert the 12-element dead-zone optimizer vector to ModelParams."""
    return ModelParams(
        omega_M=float(x[0]),
        omega_E=float(x[1]),
        alpha=float(x[2]),
        beta=float(x[3]),
        seasonal_pdf_gain=float(x[4]),
        pdf_width=float(x[5]),
        prc_form="deadzone",
        delay_amp=float(x[6]),
        delay_phase=float(x[7]),
        delay_kappa=float(x[8]),
        adv_amp=float(x[9]),
        adv_phase=float(x[10]),
        adv_kappa=float(x[11]),
        light_amp=0.0,
        light_amp2=0.0,
        light_pdf_gain=0.0,
    )


def pulse_loss(params: ModelParams, *, variant: str = "repulsive_pdf") -> float:
    """Squared error of the 6 measured 1-h pulse shifts (Fig. 3)."""
    try:
        shifts = pulse_shifts(params, variant=variant)
    except Exception:
        return 1e4
    loss = 0.0
    for zt, (m_target, e_target) in PULSE_TARGETS.items():
        m_sim, e_sim = shifts[zt]
        if not (np.isfinite(m_sim) and np.isfinite(e_sim)):
            return 1e4
        loss += (m_sim - m_target) ** 2 + (e_sim - e_target) ** 2
    return loss


def deadzone_family_loss(params: ModelParams) -> float:
    """Entrainment + genotype-collapse loss (as repulsive) PLUS pulse-PRC loss.

    Shared by the shared-PRC (12-param) and per-cell-PRC (18-param) fits.
    """
    loss = 0.0
    sim_gaps = []
    for photoperiod, (m_target, e_target) in EXPERIMENTAL_TARGETS.items():
        m_sim, e_sim, instability = settled_peaks_with_lock_error(
            photoperiod, params, variant="repulsive_pdf",
            total_days=FIT_DAYS, settle_days=FIT_SETTLE, n_check_days=8,
            rtol=FIT_RTOL, atol=FIT_ATOL,
        )
        loss += 800.0 * instability
        gap_sim = folded_gap(e_sim, m_sim)
        loss += circular_diff(m_sim, m_target) ** 2
        loss += circular_diff(e_sim, e_target) ** 2
        loss += 8.0 * (gap_sim - (e_target - m_target)) ** 2
        sim_gaps.append(gap_sim)

    for photoperiod in STABILITY_ONLY_CHECKPOINTS:
        _, _, instability = settled_peaks_with_lock_error(
            photoperiod, params, variant="repulsive_pdf",
            total_days=FIT_DAYS, settle_days=FIT_SETTLE, n_check_days=8,
            rtol=FIT_RTOL, atol=FIT_ATOL,
        )
        loss += 800.0 * instability

    for earlier, later in zip(sim_gaps, sim_gaps[1:]):
        loss += 3.0 * max(0.0, earlier - later - 0.3) ** 2

    for genotype, (m_target, e_target) in GENOTYPE_TARGETS_12.items():
        m_sim, e_sim = genotype_peaks(
            12, params, variant="repulsive_pdf", genotype=genotype,
            total_days=FIT_DAYS, day_index=FIT_DAYS - 5,
            rtol=FIT_RTOL, atol=FIT_ATOL,
        )
        loss += 5.0 * circular_diff(m_sim, m_target) ** 2
        loss += 5.0 * circular_diff(e_sim, e_target) ** 2

    # The new constraint: reproduce the measured light-pulse phase shifts.
    loss += 3.0 * pulse_loss(params)
    return float(loss)


def fit_loss_deadzone(x: np.ndarray) -> float:
    """Loss for the shared-PRC dead-zone fit (12 params)."""
    return deadzone_family_loss(params_from_vector_deadzone(x))


def fit_parameters_deadzone(seed: int = 3, maxiter: int = 60) -> ModelParams:
    """Fit the dead-zone PRC variant (12 params) with a bounded global search."""
    result = differential_evolution(
        fit_loss_deadzone,
        DEADZONE_BOUNDS,
        seed=seed,
        maxiter=maxiter,
        popsize=10,
        tol=0.02,
        polish=False,
        updating="deferred",
        workers=-1,
    )
    return params_from_vector_deadzone(result.x)


# Result of fit_parameters_percell (seed=3, maxiter=100). Recorded, not yet
# adopted as the module default -- it is a genuine trade, not a strict win.
#
# What it fixes (the point of recommendation #2):
#   * pulse SSE 29.1, vs 54.9 (harmonic) and 48.9 (shared dead-zone).
#   * the delay->advance crossover ordering is finally right: E crosses at
#     ZT15.0 and M at ZT16.0, i.e. E BEFORE M, as the data require. Every
#     shared-PRC fit gets this backwards (harmonic: M ZT15.2, E ZT18.8),
#     and provably must, since E's peak lags M's by ~4 h.
#   * pdf0 still collapses -- the risk of per-cell PRCs -- and in fact lands
#     closer to the observed ZT15 (M 14.90 / E 15.35) than the harmonic fit
#     did (14.25 / 14.20). E's own pulse responses are now good (ZT18 -4.4 vs
#     -3.0 data; ZT21 -4.8 vs -4.5).
#
# What it does NOT fix:
#   * M's large early-night delay (+5 h at ZT15) is still missed (+0.4 h).
#     M's advance lobe collapsed to ~0.03 and its delay lobe stayed modest,
#     because M's PRC also drives M's own entrainment: enlarging it breaks
#     the dusk-locked peak positions. E, which is less constrained that way,
#     did exploit the amplitude headroom (e_adv_amp 2.66).
#   * entrainment is slightly worse than the harmonic fit (gap SSE 12.3 vs
#     9.5, M-abs 16.8 vs 13.6), and the long-day ceiling regressed (P=20:
#     M ZT18.5 vs a 16 target).
#
# The residual points at the tonic/pulsatile split: one Z cannot both deliver
# a 5 h pulse shift and set M's steady-state phase. Separate parametric and
# non-parametric light terms are the next thing to try.
TUNED_PARAMS_PERCELL = ModelParams(
    omega_M=0.261683677767476,
    omega_E=0.2622168775919133,
    alpha=0.5340118460545803,
    beta=0.3732545338111757,
    seasonal_pdf_gain=2.5084353515854216,
    pdf_width=4.21307619788443,
    prc_form="deadzone",
    delay_amp=0.3877881561455969,
    delay_phase=3.458673297812152,
    delay_kappa=8.09071679117534,
    adv_amp=0.03373689765350396,
    adv_phase=4.131052148889745,
    adv_kappa=6.554629682539216,
    e_delay_amp=0.9430952878830624,
    e_delay_phase=2.5704736967617023,
    e_delay_kappa=19.731573986878402,
    e_adv_amp=2.6638051908109013,
    e_adv_phase=5.163156230183998,
    e_adv_kappa=2.5597981752319168,
    light_amp=0.0, light_amp2=0.0, light_pdf_gain=0.0,
)


# --- per-cell PRC fit (recommendation #2), 18 parameters -----------------
# Same six shared dynamics parameters, then six lobe parameters for M and six
# for E. Phase bounds are deliberately wide and identical for both cells so
# the fit is free to place E's delay->advance crossover EARLIER than M's --
# which is what the data require and what no shared PRC can express. Note the
# geometry: E's TIM peak is ~4 h later than M's, so in E's own phase
# coordinates its lobes must sit substantially earlier than M's just to break
# even in ZT terms.
PERCELL_BOUNDS = [
    (TWOPI / 25.0, TWOPI / 22.0),  # omega_M
    (TWOPI / 26.0, TWOPI / 23.0),  # omega_E
    (0.0, 0.60),                   # alpha (PDF advance on M)
    (0.0, 0.60),                   # beta  (PDF delay on E)
    (0.0, 4.0),                    # seasonal_pdf_gain
    (0.5, 6.0),                    # pdf_width
    (0.0, 3.0), (1.5, 5.5), (1.0, 20.0),   # M delay lobe: amp, phase, kappa
    (0.0, 3.0), (3.0, 8.0), (1.0, 20.0),   # M adv   lobe
    (0.0, 3.0), (1.5, 5.5), (1.0, 20.0),   # E delay lobe
    (0.0, 3.0), (3.0, 8.0), (1.0, 20.0),   # E adv   lobe
]


def params_from_vector_percell(x: np.ndarray) -> ModelParams:
    """Convert the 18-element per-cell-PRC optimizer vector to ModelParams."""
    return ModelParams(
        omega_M=float(x[0]),
        omega_E=float(x[1]),
        alpha=float(x[2]),
        beta=float(x[3]),
        seasonal_pdf_gain=float(x[4]),
        pdf_width=float(x[5]),
        prc_form="deadzone",
        delay_amp=float(x[6]), delay_phase=float(x[7]), delay_kappa=float(x[8]),
        adv_amp=float(x[9]), adv_phase=float(x[10]), adv_kappa=float(x[11]),
        e_delay_amp=float(x[12]), e_delay_phase=float(x[13]), e_delay_kappa=float(x[14]),
        e_adv_amp=float(x[15]), e_adv_phase=float(x[16]), e_adv_kappa=float(x[17]),
        light_amp=0.0, light_amp2=0.0, light_pdf_gain=0.0,
    )


def fit_loss_percell(x: np.ndarray) -> float:
    """Loss for the per-cell-PRC fit (18 params)."""
    return deadzone_family_loss(params_from_vector_percell(x))


def percell_seed_vector() -> np.ndarray:
    """Seed the 18-param search from the shared dead-zone optimum (M lobes
    duplicated into E), so the search starts on the best shared-PRC solution
    and only has to discover how the two cells should differ."""
    p = TUNED_PARAMS_DEADZONE
    shared = [p.delay_amp, p.delay_phase, p.delay_kappa,
              p.adv_amp, p.adv_phase, p.adv_kappa]
    return np.array([p.omega_M, p.omega_E, p.alpha, p.beta,
                     p.seasonal_pdf_gain, p.pdf_width] + shared + shared)


def fit_parameters_percell(seed: int = 3, maxiter: int = 80) -> ModelParams:
    """Fit per-cell PRCs (18 params) with a bounded global search."""
    result = differential_evolution(
        fit_loss_percell,
        PERCELL_BOUNDS,
        seed=seed,
        maxiter=maxiter,
        popsize=10,
        tol=0.02,
        polish=False,
        updating="deferred",
        workers=-1,
        x0=percell_seed_vector(),
    )
    return params_from_vector_percell(result.x)


def print_parameter_summary(params: ModelParams) -> None:
    """Print fitted parameters in period-friendly units."""
    print("\n--- Model parameters ---")
    print(f"M intrinsic period: {TWOPI / params.omega_M:5.2f} h")
    print(f"E intrinsic period: {TWOPI / params.omega_E:5.2f} h")
    print(f"Light PRC amplitude: {params.light_amp:5.3f} rad/h")
    print(f"Light PRC 2nd harm.: {params.light_amp2:6.3f} rad/h")
    print(f"PDF M-delay alpha:   {params.alpha:5.3f} rad/h")
    print(f"PDF E-delay beta:    {params.beta:5.3f} rad/h")
    print(f"Seasonal PDF gain:   {params.seasonal_pdf_gain:5.3f}")
    print(f"Light PDF gain:      {params.light_pdf_gain:5.3f}")
    print(f"PDF release width:   {params.pdf_width:5.3f}")


def print_quantitative_summary(params: ModelParams, variant: str) -> None:
    """Print WT M/E peak-time fit quality against the real data, and pdf0 gap."""
    print("\n--- WT M/E TIM peak ZT: data target vs simulated ---")
    print(f'{"P":>4}  {"M tgt":>6}  {"M sim":>6}  {"E tgt":>6}  {"E sim":>6}  {"gap tgt":>8}  {"gap sim":>8}')
    for photoperiod, (m_target, e_target) in EXPERIMENTAL_TARGETS.items():
        m_sim, e_sim = settled_day_peak_times(photoperiod, params, pdf_on=True, variant=variant)
        gap_target = e_target - m_target
        gap_sim = folded_gap(e_sim, m_sim)
        print(
            f"{photoperiod:>4}  {m_target:>6.1f}  {m_sim:>6.2f}  {e_target:>6.1f}  {e_sim:>6.2f}  "
            f"{gap_target:>8.1f}  {gap_sim:>8.2f}"
        )

    print("\n--- pdf0 M-to-E gap (biological prior, should collapse toward 0) ---")
    print(f'{"P":>4}  {"pdf0 gap":>9}')
    for photoperiod in sorted(SEASON_LABELS):
        ko = m_to_e_gap(photoperiod, params, pdf_on=False, variant=variant)
        print(f"{photoperiod:>4}  {ko:>9.2f}")


def plot_gap_panel(ax: plt.Axes, params: ModelParams, variant: str) -> None:
    """M-to-E gap as a function of photoperiod: WT sim vs real data vs pdf0 prior."""
    photoperiods = np.arange(4, 20.5, 0.5)
    gaps_wt = [m_to_e_gap(P, params, pdf_on=True, variant=variant) for P in photoperiods]
    gaps_ko = [m_to_e_gap(P, params, pdf_on=False, variant=variant) for P in photoperiods]

    data_P = sorted(EXPERIMENTAL_TARGETS)
    data_gap = [EXPERIMENTAL_TARGETS[p][1] - EXPERIMENTAL_TARGETS[p][0] for p in data_P]

    ax.plot(photoperiods, gaps_wt, "-", color="#2E5594", linewidth=2, label="WT sim")
    ax.plot(photoperiods, gaps_ko, "--", color="#C0392B", linewidth=2, label="pdf0 sim (prior)")
    ax.scatter(data_P, data_gap, color="black", zorder=5, label="WT data")
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Photoperiod (h light / 24 h)")
    ax.set_ylabel("M-to-E TIM peak gap (h)")
    ax.set_title("(a) M-E phase gap grows with day length, then plateaus")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")


def plot_peak_time_panel(ax: plt.Axes, params: ModelParams, variant: str) -> None:
    """Absolute M/E TIM peak ZT as a function of photoperiod: sim vs real data."""
    photoperiods = np.arange(4, 20.5, 0.5)
    m_sim = []
    e_sim = []
    for P in photoperiods:
        m, e = settled_day_peak_times(P, params, pdf_on=True, variant=variant)
        m_sim.append(m)
        e_sim.append(e)

    data_P = sorted(EXPERIMENTAL_TARGETS)
    data_M = [EXPERIMENTAL_TARGETS[p][0] for p in data_P]
    data_E = [EXPERIMENTAL_TARGETS[p][1] for p in data_P]

    ax.plot(photoperiods, m_sim, "-", color="#E67E22", linewidth=2, label="M sim")
    ax.plot(photoperiods, e_sim, "-", color="#3498DB", linewidth=2, label="E sim")
    ax.scatter(data_P, data_M, color="#E67E22", edgecolor="black", zorder=5, label="M data")
    ax.scatter(data_P, data_E, color="#3498DB", edgecolor="black", zorder=5, label="E data")
    ax.set_xlabel("Photoperiod (h light / 24 h)")
    ax.set_ylabel("TIM peak ZT (h)")
    ax.set_title("(b) Absolute M/E peak time vs day length")
    ax.set_ylim(0, 24)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)


def plot_tim_waveforms(
    axes_top: list[plt.Axes],
    axes_bottom: list[plt.Axes],
    params: ModelParams,
    variant: str,
) -> None:
    """Plot pdf0 and WT TIM transcription proxies across seasons."""
    for i, photoperiod in enumerate(sorted(SEASON_LABELS)):
        label = SEASON_LABELS[photoperiod]
        for row, pdf_on, genotype in [
            (axes_top, False, "pdf0"),
            (axes_bottom, True, "WT"),
        ]:
            t, pM, pE = simulate(photoperiod, params, pdf_on=pdf_on, variant=variant, total_days=45)
            day = (t >= 40 * 24) & (t < 41 * 24)
            td = t[day] - 40 * 24
            ax = row[i]
            ax.axvspan(photoperiod, 24, color="0.82", alpha=0.9, zorder=0)
            ax.plot(td, tim_transcription_proxy(pM[day]), color="#E67E22", linewidth=2.4, label="M cell")
            ax.plot(td, tim_transcription_proxy(pE[day]), color="#3498DB", linewidth=2.4, label="E cell")
            ax.set_title(f"{genotype} - {label}", fontsize=11)
            ax.set_xlim(0, 24)
            ax.set_ylim(0, 1.1)
            ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel("TIM proxy")
                ax.legend(loc="upper left", fontsize=9)
            if row is axes_bottom:
                ax.set_xlabel("ZT (h)")


def build_figure(params: ModelParams, variant: str, out_path: str) -> None:
    """Build the summary figure."""
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 3, height_ratios=[0.9, 0.9, 0.8, 0.8], hspace=0.6, wspace=0.22)

    ax_gap = fig.add_subplot(gs[0, :])
    plot_gap_panel(ax_gap, params, variant)

    ax_peak = fig.add_subplot(gs[1, :])
    plot_peak_time_panel(ax_peak, params, variant)

    axes_top = [fig.add_subplot(gs[2, i]) for i in range(3)]
    axes_bottom = [fig.add_subplot(gs[3, i]) for i in range(3)]
    plot_tim_waveforms(axes_top, axes_bottom, params, variant)

    fig.suptitle(
        "Dual-oscillator seasonal clock with daylight-sensitive PDF delay",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def print_comparison(params_old: ModelParams, params_rep: ModelParams) -> None:
    """Side-by-side scorecard: daylight_pdf (both-delay) vs repulsive_pdf."""
    print("\n=== WT M/E TIM peak ZT: data vs daylight_pdf vs repulsive_pdf ===")
    header = (
        f'{"P":>4} | {"M tgt":>5} {"M old":>5} {"M new":>5} | '
        f'{"E tgt":>5} {"E old":>5} {"E new":>5} | '
        f'{"g tgt":>5} {"g old":>5} {"g new":>5}'
    )
    print(header)
    sq_old = sq_new = 0.0
    m_abs_old = m_abs_new = 0.0
    for P, (mt, et) in EXPERIMENTAL_TARGETS.items():
        mo, eo = settled_day_peak_times(P, params_old, variant="daylight_pdf")
        mn, en = genotype_peaks(P, params_rep, variant="repulsive_pdf", genotype="WT")
        go, gn, gt = folded_gap(eo, mo), folded_gap(en, mn), et - mt
        sq_old += (go - gt) ** 2
        sq_new += (gn - gt) ** 2
        m_abs_old += circular_diff(mo, mt) ** 2
        m_abs_new += circular_diff(mn, mt) ** 2
        print(
            f'{P:>4} | {mt:>5.1f} {mo:>5.2f} {mn:>5.2f} | '
            f'{et:>5.1f} {eo:>5.2f} {en:>5.2f} | '
            f'{gt:>5.1f} {go:>5.2f} {gn:>5.2f}'
        )
    print(f"summed squared gap error:  daylight_pdf={sq_old:.2f}   repulsive_pdf={sq_new:.2f}")
    print(f"summed squared M-abs error: daylight_pdf={m_abs_old:.2f}   repulsive_pdf={m_abs_new:.2f}")

    print("\n=== 12:12 genotype collapse: data vs models ===")
    print(f'{"geno":>6} | {"M tgt":>5} {"M old":>5} {"M new":>5} | {"E tgt":>5} {"E old":>5} {"E new":>5}')
    wt_and_ko = {"WT": (13.0, 17.0), **GENOTYPE_TARGETS_12}
    for g, (mt, et) in wt_and_ko.items():
        mo, eo = genotype_peaks(12, params_old, variant="daylight_pdf", genotype=g)
        mn, en = genotype_peaks(12, params_rep, variant="repulsive_pdf", genotype=g)
        print(f'{g:>6} | {mt:>5.1f} {mo:>5.2f} {mn:>5.2f} | {et:>5.1f} {eo:>5.2f} {en:>5.2f}')


def build_comparison_figure(
    params_old: ModelParams, params_rep: ModelParams, out_path: str
) -> None:
    """Three-panel comparison of daylight_pdf vs repulsive_pdf against data.

    Model curves are evaluated at the measured photoperiods (data_P), where
    the fit is constrained and the data live -- not on a dense grid. Both
    models show a near-equal-period entrainment wobble at the extreme un-fit
    edges (repulsive_pdf WT at P=5, its pdf0 at P=4); those points carry no
    data and are documented as a known residual rather than showcased here.
    Peaks come from the single-settled-day argmax (the quantity the paper
    reports), so panel (a) matches panel (b) and the scorecard.
    """
    data_P = sorted(EXPERIMENTAL_TARGETS)
    data_M = [EXPERIMENTAL_TARGETS[p][0] for p in data_P]
    data_E = [EXPERIMENTAL_TARGETS[p][1] for p in data_P]
    data_gap = [e - m for m, e in (EXPERIMENTAL_TARGETS[p] for p in data_P)]
    dense_P = np.arange(4, 20.5, 0.5)  # for the dusk+1 reference line only

    def wt_peaks(P, params, variant):
        return genotype_peaks(P, params, variant=variant, genotype="WT")

    peaks_old = [wt_peaks(P, params_old, "daylight_pdf") for P in data_P]
    peaks_new = [wt_peaks(P, params_rep, "repulsive_pdf") for P in data_P]
    mo, eo = zip(*peaks_old)
    mn, en = zip(*peaks_new)
    gap_old = [folded_gap(e, m) for m, e in peaks_old]
    gap_new = [folded_gap(e, m) for m, e in peaks_new]
    # repulsive_pdf pdf0 collapse across day length (clean for P>=6; the P=4
    # near-equal-period edge is omitted, see docstring).
    ko_P = [p for p in data_P if p >= 6]
    ko_new = [
        folded_gap(*genotype_peaks(P, params_rep, variant="repulsive_pdf", genotype="pdf0")[::-1])
        for P in ko_P
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    ax.scatter(data_P, data_gap, color="black", zorder=6, label="WT data")
    ax.plot(data_P, gap_old, "o-", color="#C0392B", lw=2, ms=4, label="daylight_pdf WT")
    ax.plot(data_P, gap_new, "o-", color="#2E7D32", lw=2.4, ms=4, label="repulsive_pdf WT")
    ax.plot(ko_P, ko_new, "--", color="#2E7D32", lw=1.5, alpha=0.8, label="repulsive_pdf pdf0")
    ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
    ax.set_xlabel("Photoperiod (h light / 24 h)")
    ax.set_ylabel("M-to-E TIM peak gap (h)")
    ax.set_title("(a) M-E phase gap vs day length")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.scatter(data_P, data_M, color="#E67E22", edgecolor="black", zorder=6, label="M data")
    ax.scatter(data_P, data_E, color="#3498DB", edgecolor="black", zorder=6, label="E data")
    ax.plot(data_P, mo, "--", color="#E67E22", lw=1.4, label="M daylight_pdf")
    ax.plot(data_P, eo, "--", color="#3498DB", lw=1.4, label="E daylight_pdf")
    ax.plot(data_P, mn, "o-", color="#E67E22", lw=2.4, ms=4, label="M repulsive_pdf")
    ax.plot(data_P, en, "o-", color="#3498DB", lw=2.4, ms=4, label="E repulsive_pdf")
    # dusk + 1 h reference (the paper's clean regularity), clipped to ZT8..ZT16.
    dusk1 = [min(16.0, max(8.0, P + 1.0)) for P in dense_P]
    ax.plot(dense_P, dusk1, ":", color="gray", lw=1.6, label="dusk + 1 h (M ref)")
    ax.set_xlabel("Photoperiod (h light / 24 h)")
    ax.set_ylabel("TIM peak ZT (h)")
    ax.set_title("(b) Absolute M/E peak time vs day length")
    ax.set_ylim(0, 24)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    ax = axes[2]
    genos = ["WT", "pdfrM", "pdfrE", "pdf0"]
    geno_labels = ["WT", "pdfr-M KO", "pdfr-E KO", "pdf0"]
    xs = np.arange(len(genos))
    tgt = {"WT": (13.0, 17.0), **GENOTYPE_TARGETS_12}
    m_data = [tgt[g][0] for g in genos]
    e_data = [tgt[g][1] for g in genos]
    m_sim = [genotype_peaks(12, params_rep, variant="repulsive_pdf", genotype=g)[0] for g in genos]
    e_sim = [genotype_peaks(12, params_rep, variant="repulsive_pdf", genotype=g)[1] for g in genos]
    ax.scatter(xs - 0.08, m_data, s=120, marker="o", facecolor="none", edgecolor="#E67E22", lw=2, label="M data")
    ax.scatter(xs - 0.08, e_data, s=120, marker="o", facecolor="none", edgecolor="#3498DB", lw=2, label="E data")
    ax.scatter(xs + 0.08, m_sim, s=90, marker="s", color="#E67E22", label="M repulsive_pdf")
    ax.scatter(xs + 0.08, e_sim, s=90, marker="s", color="#3498DB", label="E repulsive_pdf")
    ax.axhline(15, color="gray", lw=0.9, ls=":", alpha=0.7)
    ax.text(len(genos) - 1, 15.3, "baseline ~ZT15", color="gray", fontsize=8, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(geno_labels, fontsize=9)
    ax.set_ylabel("TIM peak ZT at 12:12 (h)")
    ax.set_title("(c) PDF knockout collapse (12L:12D)")
    ax.set_ylim(8, 20)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "daylight_pdf (both-delay) vs repulsive_pdf (advance M / delay E, light-gated)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_prc_figure(params: ModelParams, out_path: str) -> None:
    """Plot the model's two response functions vs oscillator phase.

    (a) the shared light PRC Z(phi) = A1 cos(phi) + A2 cos(2 phi) that entrains
        both M and E, contrasted with the single-cosine (A2=0) it generalizes;
    (b) the M-gated PDF release window R(phi_M) = max(0, cos phi_M)^q.
    Both use the fitted repulsive_pdf parameters, so the figure stays in sync
    with the model.
    """
    phi = np.linspace(0.0, TWOPI, 800)
    Z = direct_light_prc(phi, params)              # two-harmonic PRC
    Z1 = params.light_amp * np.cos(phi)            # first harmonic only
    R = np.maximum(0.0, np.cos(phi)) ** params.pdf_width

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- (a) light PRC ---
    ax1.fill_between(phi, Z, 0, where=(Z >= 0), color="#2E7D32", alpha=0.14, interpolate=True)
    ax1.fill_between(phi, Z, 0, where=(Z < 0), color="#C0392B", alpha=0.14, interpolate=True)
    ax1.plot(
        phi, Z, color="#1A1A1A", lw=2.6,
        label=rf"$Z(\phi)=A_1\cos\phi+A_2\cos 2\phi$"
        f"\n($A_1$={params.light_amp:.3f}, $A_2$={params.light_amp2:.3f})",
    )
    ax1.plot(phi, Z1, "--", color="#888888", lw=1.6, label=r"first harmonic only ($A_2=0$)")
    ax1.axhline(0, color="gray", lw=0.8)
    for x, txt, col in [(0.0, "PDF release / M marker", "#E67E22"),
                        (np.pi, "TIM peak", "#3498DB")]:
        ax1.axvline(x, color=col, lw=1.1, ls=":", alpha=0.9)
        ax1.text(x, ax1.get_ylim()[0], f" {txt}\n φ={'0' if x == 0 else 'π'}",
                 color=col, fontsize=8, va="bottom", ha="left")
    zmax = float(np.max(np.abs(Z))) * 1.15
    ax1.set_ylim(-zmax, zmax)
    ax1.text(0.98, 0.96, "advance (Z>0)", transform=ax1.transAxes, ha="right", va="top",
             color="#2E7D32", fontsize=9, fontweight="bold")
    ax1.text(0.98, 0.04, "delay (Z<0)", transform=ax1.transAxes, ha="right", va="bottom",
             color="#C0392B", fontsize=9, fontweight="bold")
    ax1.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, TWOPI])
    ax1.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax1.set_xlim(0, TWOPI)
    ax1.set_xlabel(r"oscillator phase $\phi$ (rad)")
    ax1.set_ylabel(r"light PRC $Z(\phi)$  (rad/h)")
    ax1.set_title("(a) Shared light PRC (gated by $L(t)$)")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.25)

    # --- (b) PDF release gate ---
    ax2.fill_between(phi, R, color="#E67E22", alpha=0.25)
    ax2.plot(phi, R, color="#E67E22", lw=2.6, label=rf"$R(\phi_M)=\max(0,\cos\phi_M)^{{q}}$, $q$={params.pdf_width:.2f}")
    ax2.axvline(0, color="#E67E22", lw=1.1, ls=":", alpha=0.9)
    ax2.axvline(np.pi, color="#3498DB", lw=1.1, ls=":", alpha=0.9)
    ax2.text(0.0, 1.02, " M marker (φ=0)", color="#E67E22", fontsize=8, ha="left")
    ax2.text(np.pi, 1.02, " TIM peak (φ=π)", color="#3498DB", fontsize=8, ha="left")
    ax2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, TWOPI])
    ax2.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax2.set_xlim(0, TWOPI)
    ax2.set_ylim(0, 1.12)
    ax2.set_xlabel(r"M phase $\phi_M$ (rad)")
    ax2.set_ylabel(r"PDF release $R(\phi_M)$")
    ax2.set_title("(b) M-gated PDF release window")
    ax2.legend(loc="center", fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        "Response functions of the repulsive_pdf model (fitted parameters)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_prc_comparison_figure(
    params_harmonic: ModelParams, params_deadzone: ModelParams, out_path: str
) -> None:
    """Harmonic vs dead-zone PRC, and the pulse shifts each one produces."""
    phi = np.linspace(0.0, TWOPI, 800)
    Zh = direct_light_prc(phi, params_harmonic)
    Zd = direct_light_prc(phi, params_deadzone)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.9))

    axA.axhline(0, color="gray", lw=0.8)
    axA.fill_between(phi, Zd, 0, where=(Zd >= 0), color="#2E7D32", alpha=0.13, interpolate=True)
    axA.fill_between(phi, Zd, 0, where=(Zd < 0), color="#C0392B", alpha=0.13, interpolate=True)
    axA.plot(phi, Zd, color="#2E7D32", lw=2.8, label="dead-zone PRC (new)")
    axA.plot(phi, Zh, "--", color="#888888", lw=1.9, label="harmonic PRC (previous)")
    axA.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, TWOPI])
    axA.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axA.set_xlim(0, TWOPI)
    axA.set_xlabel(r"phase $\phi$ (rad)   ($\phi\!\approx\!0$ subjective dawn)")
    axA.set_ylabel(r"$Z(\phi)$ (rad/h)")
    axA.set_title("(a) light PRC shape")
    axA.grid(alpha=0.25)
    axA.legend(fontsize=8, loc="upper left")
    axA.axvspan(0, np.pi, color="#FFD54F", alpha=0.12)
    axA.text(np.pi / 2, axA.get_ylim()[1] * 0.92, "subjective day\n(dead zone)",
             ha="center", va="top", fontsize=8, color="#8A6D00")

    zts = sorted(PULSE_TARGETS)
    sh_h = pulse_shifts(params_harmonic)
    sh_d = pulse_shifts(params_deadzone)
    width = 0.26
    xs = np.arange(len(zts))
    for i, (cell, idx, col) in enumerate([("M", 0, "#E67E22"), ("E", 1, "#3498DB")]):
        off = (i - 0.5) * 0.02
        axB.bar(xs + off - width, [PULSE_TARGETS[z][idx] for z in zts], width,
                color=col, alpha=0.95, label=f"{cell} data")
        axB.bar(xs + off, [sh_d[z][idx] for z in zts], width,
                color=col, alpha=0.55, hatch="//", label=f"{cell} dead-zone")
        axB.bar(xs + off + width, [sh_h[z][idx] for z in zts], width,
                color=col, alpha=0.25, hatch="..", label=f"{cell} harmonic")
    axB.axhline(0, color="gray", lw=0.8)
    axB.set_xticks(xs)
    axB.set_xticklabels([f"ZT{z}" for z in zts])
    axB.set_ylabel(r"peak shift (h):  $+$ delay / $-$ advance")
    axB.set_title("(b) 1-h light-pulse response vs data")
    axB.grid(alpha=0.25, axis="y")
    axB.legend(fontsize=7, ncol=2)

    fig.suptitle("Recommendation #1: dead-zone light PRC vs the harmonic PRC",
                 fontsize=13, fontweight="bold", y=1.02)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_percell_prc_figure(
    params_percell: ModelParams, params_harmonic: ModelParams, out_path: str
) -> None:
    """Per-cell PRCs Z_M vs Z_E, and the pulse-response curves they produce."""
    phi = np.linspace(0.0, TWOPI, 800)
    zM = cell_light_prc(phi, params_percell, "M")
    zE = cell_light_prc(phi, params_percell, "E")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.9))

    axA.axhline(0, color="gray", lw=0.8)
    axA.plot(phi, zM, color="#E67E22", lw=2.8, label=r"$Z_M(\phi)$")
    axA.plot(phi, zE, color="#3498DB", lw=2.8, label=r"$Z_E(\phi)$")
    axA.plot(phi, cell_light_prc(phi, params_harmonic, "M"), "--",
             color="#999999", lw=1.6, label="shared harmonic (previous)")
    axA.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, TWOPI])
    axA.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axA.set_xlim(0, TWOPI)
    axA.set_xlabel(r"cell's own phase $\phi$ (rad)")
    axA.set_ylabel(r"$Z(\phi)$ (rad/h)")
    axA.set_title(r"(a) per-cell light PRCs ($Z_M \neq Z_E$)")
    axA.grid(alpha=0.25)
    axA.legend(fontsize=8, loc="upper left")

    zts = np.arange(12.0, 24.0, 0.5)
    sh_p = pulse_shifts(params_percell, pulse_zts=tuple(zts))
    sh_h = pulse_shifts(params_harmonic, pulse_zts=tuple(zts))
    axB.axhline(0, color="gray", lw=0.8)
    axB.plot(zts, [sh_p[z][0] for z in zts], "-", color="#E67E22", lw=2.6, label="M per-cell")
    axB.plot(zts, [sh_p[z][1] for z in zts], "-", color="#3498DB", lw=2.6, label="E per-cell")
    axB.plot(zts, [sh_h[z][0] for z in zts], ":", color="#E67E22", lw=1.6, label="M harmonic")
    axB.plot(zts, [sh_h[z][1] for z in zts], ":", color="#3498DB", lw=1.6, label="E harmonic")
    for z, (md, ed) in PULSE_TARGETS.items():
        axB.scatter([z], [md], s=170, marker="*", color="#E67E22", edgecolor="black", zorder=6)
        axB.scatter([z], [ed], s=170, marker="*", color="#3498DB", edgecolor="black", zorder=6)
    axB.scatter([], [], s=170, marker="*", color="gray", edgecolor="black", label="data")
    axB.set_xlabel("ZT of 1-h light pulse")
    axB.set_ylabel(r"peak shift (h):  $+$ delay / $-$ advance")
    axB.set_title("(b) pulse response: E now crosses before M")
    axB.grid(alpha=0.25)
    axB.legend(fontsize=7, ncol=2)

    fig.suptitle("Recommendation #2: per-cell light PRCs",
                 fontsize=13, fontweight="bold", y=1.02)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["baseline", "daylight_pdf", "repulsive_pdf"],
        default="daylight_pdf",
        help="Model variant to simulate.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Build the daylight_pdf vs repulsive_pdf comparison figure and scorecard.",
    )
    parser.add_argument(
        "--prc",
        action="store_true",
        help="Build the light-PRC / PDF-release response-function figure (repulsive_pdf params).",
    )
    parser.add_argument(
        "--baseline-params",
        action="store_true",
        help="Use the original hand-tuned baseline parameters.",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Run the bounded optimizer before plotting.",
    )
    parser.add_argument(
        "--fit-iterations",
        type=int,
        default=24,
        help="Differential-evolution iterations for --fit.",
    )
    default_out = os.path.join(FIGURE_DIR, "dual_oscillator_pdf.png")
    parser.add_argument(
        "--out",
        default=default_out,
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_out = os.path.join(FIGURE_DIR, "dual_oscillator_pdf.png")

    if args.compare:
        out = args.out if args.out != default_out else os.path.join(FIGURE_DIR, "model_comparison.png")
        print_comparison(TUNED_PARAMS, TUNED_PARAMS_REPULSIVE)
        build_comparison_figure(TUNED_PARAMS, TUNED_PARAMS_REPULSIVE, out)
        print(f"\nSaved: {out}")
        return

    if args.prc:
        out = args.out if args.out != default_out else os.path.join(FIGURE_DIR, "light_prc.png")
        build_prc_figure(TUNED_PARAMS_REPULSIVE, out)
        print(f"Saved: {out}")
        return

    if args.variant == "repulsive_pdf":
        params = TUNED_PARAMS_REPULSIVE
    else:
        params = BASELINE_PARAMS if args.baseline_params else TUNED_PARAMS
    if args.fit:
        params = (
            fit_parameters_repulsive(maxiter=args.fit_iterations)
            if args.variant == "repulsive_pdf"
            else fit_parameters(maxiter=args.fit_iterations)
        )

    build_figure(params, args.variant, args.out)
    print(f"Saved: {args.out}")
    print_parameter_summary(params)
    print_quantitative_summary(params, args.variant)


if __name__ == "__main__":
    main()
