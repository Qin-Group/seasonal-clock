"""Experimental four-parameter simplification of the reference model model.

State variables are mean phase m and phase difference delta=phi_E-phi_M.
Rounded phase-reference constants are inherited from the observed data:
they are not independently predicted seasonal thresholds. This model is
valid for binary light and gains with K+gamma+kappa < omega.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import odeint

import dual_oscillator_minimal as old
import reference_phase_model as reference

__version__ = "1.0.0"
OMEGA = 2*math.pi/24
P_ONSET, P_CAP = 9., 15.
SHORT_PEAK, SHORT_LAG, ACTIVATION_HALF = 8., 1., 1.


@dataclass(frozen=True)
class Params:
    K: float = 0.07293379243972402
    lam: float = 3.0089764240871326
    gamma: float = 0.10824960173209562
    kappa: float = 0.006216294327871823


TUNED = Params()


def validate(p):
    values = (p.K, p.lam, p.gamma, p.kappa)
    if not np.isfinite(values).all() or min(p.K, p.gamma, p.kappa) < 0:
        raise ValueError("Finite, nonnegative dynamical gains are required")
    if p.K+p.gamma+p.kappa >= OMEGA:
        raise ValueError("This unclamped binary-light model requires K+gamma+kappa < omega")


def activation(P):
    x = max(0., min(P, P_CAP)-P_ONSET)
    return x*x/(x*x+ACTIVATION_HALF**2)


def reference_peak(P, p=TUNED):
    return max(SHORT_PEAK, min(P, P_CAP)+SHORT_LAG)+(p.lam-SHORT_LAG)*activation(P)


def rhs(y, t, P, p=TUNED, genotype="WT", light_value=None):
    mean, delta = y
    L = float(t % 24 < P) if light_value is None else float(light_value)
    if L not in (0., 1.):
        raise ValueError("The simplified model uses binary light, not pulse-intensity scaling")
    theta = OMEGA*t+math.pi-OMEGA*reference_peak(P, p)
    S = L*max(0., math.cos(theta))*activation(P)
    gm, ge = old.genotype_flags(genotype)
    return [
        OMEGA+p.K*L*math.sin(theta-mean)*math.cos(delta/2)+(gm-ge)*p.gamma*S/2,
        -2*p.K*L*math.cos(theta-mean)*math.sin(delta/2)
        -2*p.kappa*math.sin(delta)-(gm+ge)*p.gamma*S,
    ]


def simulate(P, p=TUNED, genotype="WT", *, days=60, dt=.1, initial=None):
    """Integrate mean/difference, returning the reconstructed M/E phases."""
    validate(p)
    if not np.isfinite(P) or not 0 <= P <= 24 or days <= 2 or dt <= 0:
        raise ValueError("Invalid schedule, duration, or time step")
    stop, start = days*24., (days-2)*24.
    critical = old.light_transition_times(P, stop)
    times = np.unique(np.r_[0., critical[critical < start],
                            start+np.arange(round(48/dt)+1)*dt, stop])
    phases = np.array([math.pi, math.pi] if initial is None else initial, dtype=float)
    if phases.shape != (2,) or not np.isfinite(phases).all():
        raise ValueError("Initial state must contain two finite M/E phases")
    y0 = [float(phases.mean()), float(phases[1]-phases[0])]
    sol = odeint(rhs, y0, times, args=(P, p, genotype),
                 rtol=2e-8, atol=1e-10, tcrit=critical, hmax=.5)
    return times, np.column_stack([sol[:, 0]-sol[:, 1]/2, sol[:, 0]+sol[:, 1]/2])


def periodic_summary(P, p=TUNED, genotype="WT", *, days=60, initial=None):
    t, sol = simulate(P, p, genotype, days=days, initial=initial)
    return reference.tail_summary(t, sol)


def dd_transfer(p=TUNED, *, P=12., days=60, dd_days=4):
    validate(p)
    t, sol = simulate(P, p, days=days)
    start = t[-1]
    td = start+np.arange(dd_days*240+1)*.1
    y0 = [float(sol[-1].mean()), float(sol[-1, 1]-sol[-1, 0])]
    dd = odeint(rhs, y0, td, args=(0., p, "WT", 0.), rtol=2e-9, atol=1e-11, hmax=.5)
    phases = np.column_stack([dd[:, 0]-dd[:, 1]/2, dd[:, 0]+dd[:, 1]/2])
    rows = []
    for day in range(dd_days):
        peaks = [reference.crossings(td, phases[:, c], start+day*24, start+(day+1)*24)[0] for c in range(2)]
        rows.append({"DD_day": day+1, "peaks": peaks, "gap": float(old.folded(peaks[1]-peaks[0]))})
    return rows
