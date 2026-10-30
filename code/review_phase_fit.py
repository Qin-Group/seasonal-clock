"""Reproducible, exploratory fitting audit; does not change publication presets.

Fit paired M/E phases with their covariance, without scoring the derived gap
again. Compare the existing reference to a seven-parameter reference with a
short-day floor. The floor is phenomenological, just like the existing cap.
Uncertainty is from the repository CSV (single-layer bootstrap), not the
newer nested bootstrap in the October manuscript. A resolution variance is
added to avoid treating a sampling-grid argmax SD of zero as exact knowledge.

Run: uv run python code/review_phase_fit.py --cv --out out/fitting_review
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares

import dual_oscillator_minimal as model

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("K", "lam", "q", "h_P", "alpha", "beta")
LOW = np.array([0.05, 0, 0.5, 0.05, 0, 0])
HIGH = np.array([1.5, 6, 6, 6, 0.8, 0.8])


def load_targets(resolution_h=1.0):
    """Recover Cov(M,E) from Var(E-M), then add resolution variance.

    resolution_h / sqrt(12) is the SD of an unknown position within a bin.
    It is a sensitivity assumption, not an experimentally estimated error.
    Both coordinates receive independent resolution variances here.
    """
    if not np.isfinite(resolution_h) or resolution_h <= 0:
        raise ValueError("resolution_h must be finite and positive")
    targets = {}
    with (ROOT / "experimental_phase_vs_daylength.csv").open() as handle:
        for row in csv.DictReader(handle):
            p = float(row["daylength_h"])
            sm, se, sg = (float(row[k]) for k in (
                "M_bootstrap_sd_h", "E_bootstrap_sd_h", "bootstrap_sd_phase_gap_h"))
            cov = (sm * sm + se * se - sg * sg) / 2
            matrix = np.array([[sm * sm, cov], [cov, se * se]])
            if np.linalg.eigvalsh(matrix).min() < -1e-8:
                raise ValueError(f"Inconsistent phase/gap variances at P={p}")
            matrix += np.eye(2) * resolution_h ** 2 / 12
            targets[p] = (np.array([float(row["M_bootstrap_mean_zt"]),
                                    float(row["E_bootstrap_mean_zt"])]),
                          np.linalg.cholesky(matrix))
    return targets


def periodic_peaks(P, x, genotype="WT", *, days=40, dt=0.05, initial=None,
                   return_diagnostics=False, strict=True):
    """Continuous phase crossings, with explicit dawn/dusk integration breaks.

    Sparse burn-in outputs reduce memory. The last two days are sampled to
    check a full rotation and convergence. Crossings are linearly interpolated
    (error checked against dt=0.01); unlike sampled argmax, they give a smooth
    objective for local refinement. No arbitrary M/E peak-offset is introduced.
    """
    params = model.params_from_vector(x[:6])
    floor = float(x[6]) if len(x) == 7 else 0.0
    ref_P = max(floor, min(P, model.P_CAP))
    pdf_M, pdf_E = model.genotype_flags(genotype)
    # In the absence of effective PDF, this is the analytic attracting orbit.
    # Use simulation instead when auditing dependence on initial conditions.
    if initial is None and (P <= model.PDF_ONSET or genotype == "pdf0"):
        peak = np.full(2, (ref_P + params.lam) % 24)
        diag = {"peaks": [peak.tolist()] * 2, "crossings_per_day": [[1, 1]] * 2,
                "rotations": [[1., 1.]] * 2, "day_to_day_drift_h": [0., 0.]}
        return (peak, diag) if return_diagnostics else peak

    def rhs(y, t):
        m, e = y
        L = model.light(t, P)
        theta = model.OMEGA * t + np.pi - model.OMEGA * (ref_P + params.lam)
        S = model.pdf_drive(m, t, P, params)
        return [max(0, model.OMEGA + params.K * L * np.sin(theta - m)
                    + (params.alpha * S if pdf_M else 0)),
                max(0, model.OMEGA + params.K * L * np.sin(theta - e)
                    - (params.beta * S if pdf_E else 0))]

    stop = days * 24.0
    start = stop - 48
    critical = model.light_transition_times(P, stop)
    # Integer-index construction keeps daily endpoints exact.
    times = np.unique(np.r_[0, critical[critical < start],
                            start + np.arange(round(48 / dt) + 1) * dt, stop])
    phases = odeint(rhs, [np.pi, np.pi] if initial is None else initial, times,
                    tcrit=critical, rtol=1e-8, atol=1e-9, hmax=0.5)
    peaks = np.zeros((2, 2))
    counts = np.zeros((2, 2), dtype=int)
    rotations = np.zeros((2, 2))
    for day in range(2):
        begin = start + day * 24
        mask = (times >= begin) & (times <= begin + 24)
        td, yd = times[mask] - begin, phases[mask]
        for cell in range(2):
            phi = yd[:, cell]
            rotations[day, cell] = (phi[-1] - phi[0]) / model.TWOPI
            # Integer cycles enumerated rather than using modulo differences.
            ks = range(int(np.floor((phi[0] - np.pi) / model.TWOPI)) + 1,
                       int(np.floor((phi[-1] - np.pi) / model.TWOPI)) + 1)
            crossings = [float(np.interp(np.pi + k * model.TWOPI, phi, td)) for k in ks]
            crossings = [t for t in crossings if t < 24 - 1e-8]
            counts[day, cell] = len(crossings)
            peaks[day, cell] = crossings[0] if crossings else float(td[np.argmax(model.tim_proxy(phi))])
    diag = {"peaks": peaks.tolist(), "crossings_per_day": counts.tolist(),
            "rotations": rotations.tolist(),
            "day_to_day_drift_h": model.folded(peaks[1] - peaks[0]).tolist()}
    if return_diagnostics:
        return peaks[-1], diag
    if strict and (not np.isfinite(peaks).all() or (counts != 1).any()
            or np.max(np.abs(rotations - 1)) > 0.005
            or np.max(np.abs(model.folded(peaks[1] - peaks[0]))) > 0.05):
        raise ValueError("Candidate is not a settled, one-cycle-per-day orbit")
    return peaks[-1]


def residuals(x, targets, genotype_sigma=0.5):
    values = []
    try:
        for P, (observed, chol) in targets.items():
            predicted, diag = periodic_peaks(P, x, return_diagnostics=True)
            error = model.folded(predicted - observed)
            values.extend(np.linalg.solve(chol, error))
            values.extend(lock_residuals(diag))
        for genotype, observed in model.GENOTYPE_TARGETS_12.items():
            # Genotype uncertainties are unavailable: assumed 0.5 h, exposed
            # as a command-line option. This is NOT a measured likelihood.
            predicted, diag = periodic_peaks(12, x, genotype, return_diagnostics=True)
            error = model.folded(predicted - observed)
            # pdf0 has two copies of one predicted baseline; use one datum.
            values.extend(error[:1] / genotype_sigma if genotype == "pdf0"
                          else error / genotype_sigma)
            values.extend(lock_residuals(diag))
    except (ValueError, FloatingPointError):
        return np.full(6 * len(targets) + 17, 1e4)
    return np.asarray(values)


def lock_residuals(diag):
    """Continuous guard against drift/arrest; followed by a longer audit."""
    rotations = np.asarray(diag["rotations"])
    counts = np.asarray(diag["crossings_per_day"])
    return np.r_[10 * np.asarray(diag["day_to_day_drift_h"]),
                 100 * np.sqrt(np.mean((rotations - 1) ** 2, axis=0))
                 + 100 * np.max(np.abs(counts - 1), axis=0)]


def bounds(with_floor):
    return (np.r_[LOW, 4] if with_floor else LOW,
            np.r_[HIGH, 9] if with_floor else HIGH)


def fit(targets, with_floor, genotype_sigma, starts=3):
    seed = np.array(list(asdict(model.TUNED).values()))
    # The published beta puts pdfrM in a phase-slipping region. Starting there
    # makes a local optimizer chase an arbitrary beat phase rather than the
    # entrained orbit. Begin with weaker PDF and a stronger light attraction.
    seed[0], seed[4], seed[5] = 0.16, 0.3, 0.2
    if with_floor:
        seed = np.r_[seed, 5.5]
    lo, hi = bounds(with_floor)
    rng = np.random.default_rng(20261008)
    results = []
    for i in range(starts):
        x0 = seed if i == 0 else np.clip(seed * rng.uniform(0.7, 1.3, len(seed)), lo + 1e-5, hi - 1e-5)
        opt = least_squares(residuals, x0, args=(targets, genotype_sigma),
                            bounds=(lo, hi), diff_step=1e-3, x_scale="jac",
                            max_nfev=80, ftol=1e-6, xtol=1e-6, gtol=1e-5)
        results.append(opt)
        print(f"floor={with_floor} start={i} score={np.dot(opt.fun, opt.fun):.3f} "
              f"nfev={opt.nfev} success={opt.success}", flush=True)
    return min(results, key=lambda r: np.dot(r.fun, r.fun)), results


def metrics(x, targets, genotype_sigma):
    predictions = {str(P): periodic_peaks(P, x, strict=False).tolist() for P in targets}
    error = np.array([model.folded(np.array(predictions[str(P)]) - y)
                      for P, (y, _) in targets.items()])
    return {"paired_phase_sse_h2": float((error ** 2).sum()),
            "M_sse_h2": float((error[:, 0] ** 2).sum()),
            "E_sse_h2": float((error[:, 1] ** 2).sum()),
            "gap_sse_h2": float((np.diff(error, axis=1) ** 2).sum()),
            "phase_rmse_h": float(np.sqrt(np.mean(error ** 2))),
            "weighted_objective": float(np.sum(residuals(x, targets, genotype_sigma) ** 2)),
            "WT_peaks": predictions,
            "genotype_peaks": {g: periodic_peaks(12, x, g, strict=False).tolist()
                               for g in model.GENOTYPE_FLAGS},
            "genotype_lock_diagnostics": {g: periodic_peaks(12, x, g, return_diagnostics=True)[1]
                                          for g in model.GENOTYPE_FLAGS},
            "predictions_P12": {kind: {str(scale): prediction(rescale(x, kind, scale))
                for scale in ([0.25, 1, 3] if kind == "pdf" else [0.4, 1, 3])}
                for kind in ("pdf", "light")}}


def prediction(x):
    try:
        peaks = periodic_peaks(12, x)
        return {"gap_h": float(model.folded(peaks[1]-peaks[0])), "entrained": True}
    except ValueError:
        _, diag = periodic_peaks(12, x, return_diagnostics=True)
        return {"gap_h": None, "entrained": False, "diagnostics": diag}


def rescale(x, kind, scale):
    scaled = np.array(x).copy()
    if kind == "light":
        scaled[0] *= scale
    else:
        scaled[4:6] *= scale
    return scaled


def cross_validate(x, targets, genotype_sigma):
    rows = []
    for P, (observed, chol) in targets.items():
        train = {p: t for p, t in targets.items() if p != P}
        opt = least_squares(residuals, x, args=(train, genotype_sigma),
                            bounds=bounds(len(x) == 7), diff_step=1e-3,
                            x_scale="jac", max_nfev=80,
                            ftol=1e-5, xtol=1e-5, gtol=1e-4)
        pred = periodic_peaks(P, opt.x)
        error = model.folded(pred - observed)
        rows.append({"P": P, "predicted": pred.tolist(), "error_h": error.tolist(),
                     "weighted_squared_error": float(np.sum(np.linalg.solve(chol, error) ** 2)),
                     "fit_success": bool(opt.success)})
        print(f"CV floor={len(x)==7} P={P:g} error={error.round(3)}", flush=True)
    return {
        "phase_rmse_h": float(np.sqrt(np.mean(np.array([r["error_h"] for r in rows]) ** 2))),
        "weighted_squared_error": sum(r["weighted_squared_error"] for r in rows),
        "folds": rows,
        "caveat": "Warm-start local refits; onset/cap fixed using all data. At 12 h, genotype targets remain in training."}


def plot(results, targets, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    ps = np.array(sorted(targets))
    ys = np.array([targets[p][0] for p in ps])
    for cell, ax in enumerate(axes[:2]):
        sd = np.array([np.sqrt((targets[p][1] @ targets[p][1].T)[cell, cell]) for p in ps])
        ax.errorbar(ps, ys[:, cell], yerr=sd, fmt="o", color="black", ms=4, capsize=2,
                    label="data + resolution allowance")
        ax.set_ylabel(f"{'M' if cell==0 else 'E'} peak ZT (h)")
    axes[2].plot(ps, ys[:, 1] - ys[:, 0], "ko", ms=4, label="data")
    axes[2].set_ylabel("E − M gap (h)")
    grid = np.linspace(4, 20, 65)
    for label, result in results.items():
        curves = np.array([periodic_peaks(p, result["vector"]) for p in grid])
        for cell, ax in enumerate(axes[:2]):
            ax.plot(grid, curves[:, cell], label=label)
        axes[2].plot(grid, model.folded(curves[:, 1]-curves[:, 0]), label=label)
    for ax in axes:
        ax.set_xlabel("Daylength (h)")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def audit_orbits(x):
    """Longer simulations at fitted/intermediate P and distinct initial phases.

    This numerical screen is not a proof of global stability. It detects the
    knockout drift hidden by the published day-25 snapshot and checks that
    independently started trajectories converge to the same daily peaks.
    """
    failures = []
    max_drift = max_rotation_error = max_initial_difference = 0.0
    for P, genotype in [(float(p), "WT") for p in range(4, 21)] + [
            (12., g) for g in ("pdf0", "pdfrM", "pdfrE")]:
        runs = []
        for initial in ([np.pi, np.pi], [0., 1.7], [2.2, 5.4]):
            peaks, diag = periodic_peaks(P, x, genotype, days=80, initial=initial,
                                         return_diagnostics=True)
            drift = float(np.max(np.abs(diag["day_to_day_drift_h"])))
            rotation = float(np.max(np.abs(np.asarray(diag["rotations"]) - 1)))
            max_drift = max(max_drift, drift)
            max_rotation_error = max(max_rotation_error, rotation)
            if drift > 0.05 or rotation > 0.005 or np.any(np.asarray(diag["crossings_per_day"]) != 1):
                failures.append({"P": P, "genotype": genotype, "initial": initial, **diag})
            runs.append(peaks)
        max_initial_difference = max(max_initial_difference,
                                    float(np.max(np.abs(model.folded(np.array(runs) - runs[0])))))
    return {"days": 80, "P_grid_WT": list(range(4, 21)),
            "genotype_P": 12, "initial_conditions_per_condition": 3,
            "max_daily_drift_h": max_drift,
            "max_rotation_error_cycles": max_rotation_error,
            "max_initial_condition_peak_difference_h": max_initial_difference,
            "failures": failures}


def save_results(out, results, resolution, genotype_sigma):
    report = {"assumptions": {"resolution_h": resolution,
                              "genotype_sigma_h": genotype_sigma,
                              "fixed_onset_h": model.PDF_ONSET, "fixed_cap_h": model.P_CAP},
              "results": results}
    (out / "results.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "fitting_review")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--genotype-sigma", type=float, default=0.5)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--evaluate-json", type=Path,
                        help="Audit already fitted results without refitting")
    parser.add_argument("--resume", action="store_true", help="Keep completed fits in the output directory")
    args = parser.parse_args()
    if args.genotype_sigma <= 0 or not np.isfinite(args.genotype_sigma) or args.starts < 1:
        parser.error("genotype-sigma must be positive and finite; starts must be positive")
    targets = load_targets(args.resolution)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_json:
        report = json.loads(args.evaluate_json.read_text())
        for name, result in report["results"].items():
            result["long_orbit_audit"] = audit_orbits(result["vector"])
            print(name, "audit failures:", len(result["long_orbit_audit"]["failures"]), flush=True)
        (args.out / "results.json").write_text(json.dumps(report, indent=2))
        plot(report["results"], targets, args.out / "comparison.png")
        return
    baseline = np.array(list(asdict(model.TUNED).values()))
    results = {"Published preset": {"vector": baseline.tolist(),
                                     **metrics(baseline, targets, args.genotype_sigma)}}
    if args.resume and (args.out / "results.json").exists():
        previous = json.loads((args.out / "results.json").read_text())
        assumptions = previous.get("assumptions", {})
        if assumptions and (assumptions["resolution_h"] != args.resolution or
                            assumptions["genotype_sigma_h"] != args.genotype_sigma):
            parser.error("Resume requires matching resolution and genotype uncertainty")
        results = previous.get("results", previous)
    for floor, label in ((False, "Covariance fit"), (True, "Covariance fit + floor")):
        if label in results and "WT_peaks" in results[label]:
            print("Reusing", label, flush=True)
            if args.cv and "leave_one_photoperiod_out" not in results[label]:
                results[label]["leave_one_photoperiod_out"] = cross_validate(
                    results[label]["vector"], targets, args.genotype_sigma)
                save_results(args.out, results, args.resolution, args.genotype_sigma)
            continue
        opt, starts = fit(targets, floor, args.genotype_sigma, args.starts)
        result = {"vector": opt.x.tolist(), "parameters": dict(zip(NAMES + (("P_floor",) if floor else ()), opt.x.tolist())),
                  "fit_success": bool(opt.success), "fit_message": str(opt.message),
                  "start_objectives": [float(np.sum(r.fun ** 2)) for r in starts]}
        results[label] = result
        save_results(args.out, results, args.resolution, args.genotype_sigma)
        result.update(metrics(opt.x, targets, args.genotype_sigma))
        save_results(args.out, results, args.resolution, args.genotype_sigma)
        if args.cv:
            result["leave_one_photoperiod_out"] = cross_validate(opt.x, targets, args.genotype_sigma)
        save_results(args.out, results, args.resolution, args.genotype_sigma)
    save_results(args.out, results, args.resolution, args.genotype_sigma)
    plot(results, targets, args.out / "comparison.png")
    print(json.dumps({k: {n: v[n] for n in ("phase_rmse_h", "gap_sse_h2", "weighted_objective")}
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
