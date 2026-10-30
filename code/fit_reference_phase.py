"""Fit and compare reference ablations without changing the frozen baseline model model.

Default phase targets are baseline model's stored paired bootstrap means for comparable
scoring. --targets continuous runs a sensitivity fit to raw waveform peaks.
DD constraints are manuscript-derived qualitative calibration assumptions,
not invented measured SDs. Source/protocol/assumptions are saved with results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

import dual_oscillator_minimal as baseline
import reference_phase_model as model
import review_phase_fit as review

ROOT = Path(__file__).resolve().parents[1]
GENOTYPES = {"pdf0": (15., 15.), "pdfrM": (15., 17.), "pdfrE": (13., 14.8)}
PULSE_TARGETS = {15: (4.1, 1.1), 18: (0.1, -2.7), 21: (-0.8, -4.6)}


def load_targets(kind, observation):
    if kind == "stored":
        return review.load_targets(1.)
    rows = json.loads(observation.read_text())["conditions"]
    return {r["P"]: (np.array(r["phase"]), np.linalg.cholesky(
        np.array(r["phase_covariance"])+np.eye(2)*0.25**2)) for r in rows}


def loss_parts(x, variant, targets, *, dd=True):
    p = model.from_vector(x, variant)
    phase, genotype, locking = [], [], []
    summaries = {}
    for P, (observed, chol) in targets.items():
        s = model.periodic_summary(P, p, variant)
        summaries[(P, "WT")] = s
        phase.extend(np.linalg.solve(chol, baseline.folded(np.array(s["peaks"])-observed)))
        locking.extend(lock_error(s))
    for g, observed in GENOTYPES.items():
        s = model.periodic_summary(12, p, variant, g)
        error = baseline.folded(np.array(s["peaks"])-observed)/0.5
        genotype.extend(error[:1] if g == "pdf0" else error)
        locking.extend(lock_error(s))
    dark = []
    if dd:
        rows = model.dd_transfer(p, variant)
        # DD1 manuscript gap=3.1 h; DD4 described as disappeared. Both
        # calibration scales below are explicit assumptions for exploratory
        # fitting until the raw DD observations and bootstrap errors exist.
        dark = [(rows[0]["gap"]-3.1)/0.75,
                max(0., abs(rows[-1]["gap"])-1.0)/0.5]
    return {"phase": phase, "genotype": genotype, "locking": locking, "DD": dark}


def lock_error(s):
    rotations = np.array(s["rotations"])
    counts = np.array(s["counts"])
    return np.r_[10*np.array(s["drift"]),
                 100*np.sqrt(np.mean((rotations-1)**2, axis=0))
                 +100*np.max(abs(counts-1), axis=0)]


def residual(x, variant, targets, dd=True):
    parts = loss_parts(x, variant, targets, dd=dd)
    return np.concatenate([parts[k] for k in ("phase", "genotype", "locking", "DD")])


def fit(variant, targets, starts=2, seed=20261008, initial=None, dd=True, max_nfev=80):
    names = model.PARAMETERS[variant]
    low, high = np.array([model.BOUNDS[n] for n in names]).T
    p0 = initial or model.Params()
    if variant == "synchronized":
        p0 = replace(p0, lam=2.7, K=.07, q=1., alpha=.07, beta=.1, kappa=.008)
    x0 = np.clip(model.vector(p0, variant), low+1e-6, high-1e-6)
    rng = np.random.default_rng(seed)
    attempts = []
    for i in range(starts):
        start = x0 if i == 0 else np.clip(x0+rng.normal(0,.08,len(x0))*(high-low), low+1e-6, high-1e-6)
        fun = lambda x: residual(x, variant, targets, dd)
        steps = (high-low)*3e-4
        def jac(x):
            # Absolute steps are essential for release_offset initialized at
            # zero. Relative differences there fall below the ODE error.
            fx = fun(x)
            columns = []
            for j, step in enumerate(steps):
                if x[j]+step > high[j]:
                    step = -step
                shifted = x.copy()
                shifted[j] += step
                columns.append((fun(shifted)-fx)/step)
            return np.array(columns).T
        opt = least_squares(fun, start, jac=jac,
                            bounds=(low, high), x_scale="jac",
                            max_nfev=max_nfev, ftol=2e-6, xtol=2e-6, gtol=1e-5)
        attempts.append(opt)
        print(f"{variant} start={i} objective={np.sum(opt.fun**2):.3f} "
              f"nfev={opt.nfev} success={opt.success}", flush=True)
    return min(attempts, key=lambda r: np.sum(r.fun**2)), attempts


def evaluate(p, variant, targets):
    wt = {str(P): model.periodic_summary(P, p, variant, days=60) for P in targets}
    errors = np.array([baseline.folded(np.array(wt[str(P)]["peaks"])-obs) for P, (obs, c) in targets.items()])
    g = {name: model.periodic_summary(12, p, variant, name, days=60) for name in GENOTYPES}
    parts = loss_parts(model.vector(p, variant), variant, targets)
    return {"WT_peak_rmse_h": float(np.sqrt(np.mean(errors**2))),
            "WT_gap_sse_h2": float(np.sum(np.diff(errors, axis=1)**2)),
            "WT": wt, "genotypes": g, "DD_transfer": model.dd_transfer(p, variant, days=60),
            "objective_parts": {k: float(np.sum(np.array(v)**2)) for k, v in parts.items()},
            "gain_scans": {kind: {str(scale): model.periodic_summary(12,
                replace(p, **({"K": p.K*scale} if kind=="light" else
                              {"alpha": p.alpha*scale, "beta": p.beta*scale})), variant, days=60)
                for scale in (.4, 1., 3.)} for kind in ("light", "PDF")}}


def audit(p, variant):
    failures, max_drift, max_initial = [], 0., 0.
    for P, g in [(float(P), "WT") for P in range(4, 21)]+[(12., g) for g in GENOTYPES]:
        runs = []
        for pair in ([np.pi, np.pi], [0., 1.7], [2.2, 5.4]):
            initial = pair+([0.5] if variant in ("light_history", "dark_adaptation") else [])
            s = model.periodic_summary(P, p, variant, g, days=80, initial=initial)
            runs.append(s["peaks"])
            max_drift = max(max_drift, float(np.max(np.abs(s["drift"]))))
            if not s["locked"]:
                failures.append({"P": P, "genotype": g, "initial": initial, **s})
        max_initial = max(max_initial, float(np.max(np.abs(baseline.folded(np.array(runs)-runs[0])))))
    return {"days": 80, "initial_conditions": 3, "WT_P_grid": list(range(4, 21)),
            "genotype_P": 12, "failures": failures, "max_drift_h": max_drift,
            "max_initial_peak_difference_h": max_initial}


def cv(p, variant, targets):
    rows = []
    for P, (observed, chol) in targets.items():
        train = {q: v for q, v in targets.items() if q != P}
        opt, _ = fit(variant, train, starts=1, initial=p, max_nfev=60)
        pred = model.periodic_summary(P, model.from_vector(opt.x, variant), variant, days=60)
        rows.append({"P": P, "error_h": baseline.folded(np.array(pred["peaks"])-observed).tolist(),
                     "locked": pred["locked"], "optimizer_success": bool(opt.success)})
    return {"rmse_h": float(np.sqrt(np.mean(np.array([r["error_h"] for r in rows])**2))),
            "folds": rows,
            "limitation": "Local warm starts; data-derived landmarks and genotype/DD information remain in training."}


def plot(results, targets, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    P = np.array(sorted(targets))
    obs = np.array([targets[p][0] for p in P])
    for c, ax in enumerate(axes[0]):
        ax.scatter(P, obs[:, c], color="black", s=20, label="stored phase estimates")
        ax.set(xlabel="Daylength (h)", ylabel=f"{'M' if c==0 else 'E'} peak ZT (h)")
    axes[1, 0].scatter(P, obs[:, 1]-obs[:, 0], color="black", s=20)
    axes[1, 0].set(xlabel="Daylength (h)", ylabel="E − M gap (h)")
    axes[1, 1].set(xlabel="Day in DD", ylabel="E − M peak gap (h)")
    for name, r in results.items():
        ys = np.array([r["WT"][str(p)]["peaks"] for p in P])
        for c, ax in enumerate(axes[0]):
            ax.plot(P, ys[:, c], label=name)
        axes[1, 0].plot(P, baseline.folded(ys[:, 1]-ys[:, 0]), label=name)
        axes[1, 1].plot([row["DD_day"] for row in r["DD_transfer"]],
                        [row["gap"] for row in r["DD_transfer"]], "o-", label=name)
    for ax in axes.flat:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=model.PARAMETERS, default=list(model.PARAMETERS))
    parser.add_argument("--targets", choices=("stored", "continuous"), default="stored")
    parser.add_argument("--observation", type=Path, default=ROOT/"out"/"reference"/"observation.json")
    parser.add_argument("--out", type=Path, default=ROOT/"out"/"reference"/"stored")
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refine", action="store_true", help="Refit existing parameters using absolute numerical derivatives")
    parser.add_argument("--initial-results", type=Path, help="Use matching variants from another target fit as feasible starting points")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.targets, args.observation)
    path = args.out/"results.json"
    metadata = {"model_version": model.__version__, "phase_targets": args.targets,
                "genotype_targets": GENOTYPES, "genotype_scale_h_assumed": .5,
                "DD1_gap_target_h": 3.1, "DD1_calibration_scale_h_assumed": .75,
                "DD4_gap_upper_limit_h_assumed": 1., "DD4_calibration_scale_h_assumed": .5,
                "protocol_source": str(ROOT.parent/"manuscript"/"Manuscirpt 1007.docx"),
                "pulse_targets_updated": PULSE_TARGETS,
                "fixed_landmarks_h": [9, 15], "seed": 20261008}
    metadata["dark_adaptation_fixed"] = {"tau_h": model.DARK_TAU, "power": model.DARK_POWER}
    report = json.loads(path.read_text()) if (args.resume or args.refine) and path.exists() else {"metadata": metadata, "results": {}}
    if report["metadata"]["phase_targets"] != args.targets:
        parser.error("Cannot resume with different phase targets")
    results = report["results"]
    report["metadata"].update(metadata)
    initial_results = json.loads(args.initial_results.read_text())["results"] if args.initial_results else {}
    previous = None
    for variant in args.variants:
        if variant not in results or args.refine:
            start = time.monotonic()
            initial = model.Params(**results[variant]["parameters"]) if args.refine and variant in results else previous
            if variant in initial_results:
                initial = model.Params(**initial_results[variant]["parameters"])
            elif variant == "dark_adaptation" and "separate_release" in initial_results:
                initial = model.Params(**initial_results["separate_release"]["parameters"])
            opt, tries = fit(variant, targets, starts=args.starts, initial=initial)
            p = model.from_vector(opt.x, variant)
            r = {"parameters": asdict(p), "free_parameters": list(model.PARAMETERS[variant]),
                 "optimizer_success": bool(opt.success), "optimizer_message": opt.message,
                 "start_objectives": [float(np.sum(t.fun**2)) for t in tries],
                 "seconds": time.monotonic()-start, **evaluate(p, variant, targets)}
            r["orbit_audit"] = audit(p, variant)
            r["light_switch_9_to_15"] = model.light_switch(p, variant)
            r["light_switch_15_to_9"] = model.light_switch(p, variant, before=15, after=9)
            results[variant] = r
            path.write_text(json.dumps(report, indent=2))
        previous = model.Params(**results[variant]["parameters"])
        if args.cv and "cross_validation" not in results[variant]:
            results[variant]["cross_validation"] = cv(previous, variant, targets)
            path.write_text(json.dumps(report, indent=2))
    plot(results, targets, args.out/"comparison.png")
    print(json.dumps({k: {n: r[n] for n in ("WT_peak_rmse_h", "WT_gap_sse_h2", "objective_parts")}
                      for k, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
