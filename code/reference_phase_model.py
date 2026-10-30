"""Version 0.2 phase models with synchronization and causal light memory.

The manuscript baseline model remains in dual_oscillator_minimal.py.
Variants are explicit ablations, not automatically selected publication models.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.integrate import odeint

import dual_oscillator_minimal as baseline

__version__ = "1.0.0"
CANONICAL_VARIANT = "separate_release"
OMEGA, TWOPI = baseline.OMEGA, baseline.TWOPI
DARK_TAU = 48.0
DARK_POWER = 4.0
# Phases are unwrapped and grow by 2*pi per day, so a relative tolerance
# loosens with elapsed time. Looser values (the original 2e-7/1e-9) leave
# solver noise of about 1e-2 in the fitting residuals, larger than the
# finite-difference signal; tighter than about 1e-11 fails at the
# release-window kink (cos(source) = 0).
RTOL, ATOL = 1e-10, 1e-10


@dataclass(frozen=True)
class Params:
    K: float = 0.08
    lam: float = 3.5
    q: float = 1.5
    h_P: float = 1.5
    alpha: float = 0.12
    beta: float = 0.15
    kappa: float = 0.012
    P_floor: float = 7.0
    lam_short: float = 1.0
    release_offset: float = 0.0
    tau_light: float = 72.0
    kappa_dark: float = 0.03


TUNED = Params(K=0.0733763997763125, lam=3.04913212958034,
               q=0.8357465541423458, h_P=1.1692609666538485,
               alpha=0.10411418376942086, beta=0.10233371753233458,
               kappa=0.005940955410337736, P_floor=7.04286526749537,
               lam_short=0.9638695603005869, release_offset=-0.04120838447482835)


PARAMETERS = {
    "synchronized": ("K", "lam", "q", "h_P", "alpha", "beta", "kappa"),
    "flexible_reference": ("K", "lam", "q", "h_P", "alpha", "beta", "kappa", "P_floor", "lam_short"),
    "separate_release": ("K", "lam", "q", "h_P", "alpha", "beta", "kappa", "P_floor", "lam_short", "release_offset"),
    "light_history": ("K", "lam", "q", "h_P", "alpha", "beta", "kappa", "P_floor", "lam_short", "release_offset", "tau_light"),
    "dark_adaptation": ("K", "lam", "q", "h_P", "alpha", "beta", "kappa", "P_floor", "lam_short", "release_offset", "kappa_dark"),
}
BOUNDS = {"K": (0.02, 0.5), "lam": (0., 6.), "q": (0.5, 6.),
          "h_P": (0.15, 5.), "alpha": (0., 0.8), "beta": (0., 0.8),
          "kappa": (0.001, 0.08), "P_floor": (4., 9.), "lam_short": (0., 4.),
          "release_offset": (-math.pi, math.pi), "tau_light": (24., 240.), "kappa_dark": (0., .08)}


def vector(params, variant):
    return np.array([getattr(params, n) for n in PARAMETERS[variant]])


def from_vector(x, variant):
    if len(x) != len(PARAMETERS[variant]):
        raise ValueError("Parameter vector length does not match variant")
    return replace(Params(), **dict(zip(PARAMETERS[variant], map(float, x))))


def activation(P, p):
    x = max(0., min(P, 15.)-9.)
    return x*x/(x*x+p.h_P*p.h_P)


def baseline(P, p, variant):
    if variant == "synchronized":
        return min(P, 15.)+p.lam
    return max(p.P_floor, min(P, 15.)) + p.lam_short + (p.lam-p.lam_short)*activation(P, p)


def rhs(y, t, P, p, variant, genotype="WT", light_value=None):
    """P supplies the lamp schedule; history variant never uses it as a cue.

    Its cue is 24*a where da/dt=(L-a)/tau_light. The time reference is elapsed
    time from the experimental dawn origin. No future dusk is supplied to its
    phase reference or PDF gain. Baseline onset/cap remain empirical assumptions.
    """
    L = float(t % 24 < P) if light_value is None else float(light_value)
    cue = 24*max(0., min(1., y[2])) if variant == "light_history" else P
    theta = OMEGA*t+math.pi-OMEGA*baseline(cue, p, variant)
    source = theta+p.release_offset if variant in ("separate_release", "light_history", "dark_adaptation") else y[0]
    release = max(0., math.cos(source))**p.q
    S = L*release*activation(cue, p)
    pdf_M, pdf_E = baseline.genotype_flags(genotype)
    strength = p.kappa + (p.kappa_dark*max(0., min(1., y[2]))**DARK_POWER
                          if variant == "dark_adaptation" else 0.)
    coupling = strength*math.sin(y[1]-y[0])
    dm = OMEGA+p.K*L*math.sin(theta-y[0])+coupling+(p.alpha*S if pdf_M else 0)
    de = OMEGA+p.K*L*math.sin(theta-y[1])-coupling-(p.beta*S if pdf_E else 0)
    out = [max(0., dm), max(0., de)]
    if variant == "light_history":
        out.append((L-y[2])/p.tau_light)
    elif variant == "dark_adaptation":
        out.append((float(L <= 0)-y[2])/DARK_TAU)
    return out


def simulate(P, p=Params(), variant="flexible_reference", genotype="WT", *,
             days=40, dt=0.1, initial=None, tail_days=2):
    if not 0 <= P <= 24 or days <= tail_days or dt <= 0:
        raise ValueError("Invalid photoperiod, duration, or time step")
    stop, start = days*24., (days-tail_days)*24.
    critical = baseline.light_transition_times(P, stop)
    times = np.unique(np.r_[0., critical[critical < start],
                            start+np.arange(round((stop-start)/dt)+1)*dt, stop])
    if initial is None:
        initial = [math.pi, math.pi]+([0.5] if variant in ("light_history", "dark_adaptation") else [])
    sol = odeint(rhs, initial, times, args=(P, p, variant, genotype),
                 rtol=RTOL, atol=ATOL, tcrit=critical, hmax=1.)
    return times, sol


def crossings(t, phi, begin, end):
    mask = (t >= begin-1e-8) & (t <= end+1e-8)
    td, yd = t[mask]-begin, phi[mask]
    rotations = float((yd[-1]-yd[0])/TWOPI)
    ks = range(math.floor((yd[0]-math.pi)/TWOPI)+1,
               math.floor((yd[-1]-math.pi)/TWOPI)+1)
    peaks = [float(np.interp(math.pi+k*TWOPI, yd, td)) for k in ks]
    peaks = [z for z in peaks if z < end-begin-1e-7]
    return (peaks[0] if peaks else float(td[np.argmax(baseline.tim_proxy(yd))])), rotations, len(peaks)


def tail_summary(t, sol):
    end = t[-1]
    peaks, rotations, counts = [], [], []
    for day in (end-48, end-24):
        triples = [crossings(t, sol[:, c], day, day+24) for c in range(2)]
        peaks.append([v[0] for v in triples])
        rotations.append([v[1] for v in triples])
        counts.append([v[2] for v in triples])
    drift = baseline.folded(np.diff(peaks, axis=0)[0])
    locked = bool(np.max(np.abs(drift)) < .05 and
                  np.max(np.abs(np.array(rotations)-1)) < .005 and np.all(np.array(counts)==1))
    return {"peaks": peaks[-1], "tail_peaks": peaks, "rotations": rotations,
            "counts": counts, "drift": drift.tolist(), "locked": locked}


def periodic_summary(P, p, variant, genotype="WT", days=40, initial=None):
    t, sol = simulate(P, p, variant, genotype, days=days, initial=initial)
    return tail_summary(t, sol)


def dd_transfer(p, variant, *, P=12, days=40, dd_days=4):
    """Actually transfer an entrained LD trajectory to DD at the next ZT0."""
    t, sol = simulate(P, p, variant, days=days)
    start = t[-1]
    td = start+np.arange(dd_days*240+1)*0.1
    dd = odeint(rhs, sol[-1], td, args=(0, p, variant, "WT", 0.),
                rtol=RTOL, atol=ATOL, hmax=.5)
    rows = []
    for day in range(dd_days):
        peaks = [crossings(td, dd[:, c], start+day*24, start+(day+1)*24)[0] for c in range(2)]
        rows.append({"DD_day": day+1, "peaks": peaks,
                     "gap": float(baseline.folded(peaks[1]-peaks[0]))})
    return rows


def light_switch(p, variant, *, before=9., after=15., acclimation=40, days=10):
    """Old LD history, then actual new lighting. History state is not reset."""
    t, sol = simulate(before, p, variant, days=acclimation)
    start = t[-1]
    stop = start+days*24
    critical = baseline.light_transition_times(after, stop)
    critical = critical[critical > start]
    td = start+np.arange(days*240+1)*.1
    switched = odeint(rhs, sol[-1], td, args=(after, p, variant, "WT"),
                       rtol=RTOL, atol=ATOL, tcrit=critical, hmax=.5)
    rows = []
    for d in range(days):
        peaks = [crossings(td, switched[:, c], start+d*24, start+(d+1)*24)[0] for c in range(2)]
        row = {"day": d+1, "peaks": peaks, "gap": float(baseline.folded(peaks[1]-peaks[0]))}
        if variant == "light_history":
            row["inferred_daylength_at_dawn"] = float(24*switched[d*240, 2])
        rows.append(row)
    return rows
