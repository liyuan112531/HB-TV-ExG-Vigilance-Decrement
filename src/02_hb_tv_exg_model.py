"""HB-TV-ExG: hierarchical Bayesian time-varying Ex-Gaussian model.

Model
-----
For operator i = 1..I and trial t = 1..T the response time is modelled as

    tau_i,t  = clip(tau_base_i + theta_i * t_norm, 0.001, inf)
    rt_i,t  ~ ExGaussian(mu_i, sigma_i, tau_i,t)

with t_norm = (t - 1) / (T - 1) in [0, 1]. ``tau_base_i`` is the static initial
tail risk carried into the monitoring block and ``theta_i`` is the individual
fatigue slope (seconds per full session); theta replaces the former discrete
first-half / second-half decomposition with a continuous drift term.

Hierarchy (all individual parameters are non-centred for sampling efficiency)
    mu_pop ~ Normal(mean(rt), 1)              sigma_mu  ~ HalfNormal(0.3)
    log_sigma_pop ~ Normal(log(sd(rt)) - 1, 0.5)          s_sigma ~ HalfNormal(0.4)
    tau_pop ~ HalfNormal(0.5)                  s_tau   ~ HalfNormal(0.5)
    theta_pop ~ Normal(0, 0.3)                 s_theta ~ HalfNormal(0.3)
    mu_i      = mu_pop + sigma_mu * z_mu_i
    sigma_i   = exp(log_sigma_pop + s_sigma * z_sigma_i)
    tau_base_i = tau_pop * exp(s_tau * z_tau_i)          (multiplicative -> positive)
    theta_i   = theta_pop + s_theta * z_theta_i           (sign free)

Outputs (``results/tables``)
    hb_tv_posterior_summary.csv            posterior summary, anonymous labels
    hb_tv_convergence_diagnostics.csv      r-hat, ESS, divergences, timings
    hb_tv_loo_diagnostics.csv              ELPD_LOO, Pareto-k shape diagnostics
    hb_tv_loo_pointwise.csv                per-trial elpd_loo_i and Pareto-k
    hb_tv_tau_base_vs_p99_correlation.csv  tau_base vs empirical P99 agreement
    hb_tv_quadrant_assignment.csv          per-operator tau_base, theta, quadrant

Run from the repository root:
    python src/02_hb_tv_exg_model.py
Smoke test (tune=100, draws=200, chains=1):
    HBM_SMOKE=1 python src/02_hb_tv_exg_model.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
CLEAN_CSV = ROOT / "data" / "cleaned_trials_1329.csv"

MODEL_CHANNEL = "foot_rt_us"
US2S = 1e6
SEED = 2026
TAU_FLOOR = 0.001                 # lower bound of the tail parameter (s)
TARGET_ACCEPT = 0.92

SMOKE = os.environ.get("HBM_SMOKE", "0") == "1"
TUNE, DRAWS, CHAINS = (100, 200, 1) if SMOKE else (1000, 2000, 4)

QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
QUADRANT_LABEL = {
    "Q1": "Extreme high-risk",
    "Q2": "Fatigue deteriorating",
    "Q3": "Robust throughout",
    "Q4": "Static sluggish",
}


def load_trials() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Return (rt in s, subject codes, t_norm, subject labels, trials)."""
    df = pd.read_csv(CLEAN_CSV)
    subjects = sorted(df["subject_id"].astype(str).unique())
    codes = df["subject_id"].astype(str).map({s: i for i, s in enumerate(subjects)})
    y = df[MODEL_CHANNEL].values.astype(np.float64) / US2S
    t_norm = (df["trial_id"].values.astype(np.float64) - 1.0) / 29.0
    assert t_norm.min() >= 0.0 and t_norm.max() <= 1.0, "t_norm outside [0, 1]"
    return y, codes.values.astype(int), t_norm, subjects, df


def build_model(y, codes, t_norm, subjects, ybar, sdy):
    """Hierarchical time-varying Ex-Gaussian likelihood (non-centred)."""
    import pymc as pm

    with pm.Model(coords={"subject": subjects}) as model:
        mu_pop = pm.Normal("mu_pop", mu=ybar, sigma=1.0)
        sigma_mu = pm.HalfNormal("sigma_mu", sigma=0.3)
        log_sigma_pop = pm.Normal("log_sigma_pop",
                                  mu=np.log(max(sdy, 1e-3)) - 1.0, sigma=0.5)
        s_sigma = pm.HalfNormal("s_sigma", sigma=0.4)
        tau_pop = pm.HalfNormal("tau_pop", sigma=0.5)
        s_tau = pm.HalfNormal("s_tau", sigma=0.5)
        theta_pop = pm.Normal("theta_pop", mu=0.0, sigma=0.3)
        s_theta = pm.HalfNormal("s_theta", sigma=0.3)

        z_mu = pm.Normal("z_mu", 0.0, 1.0, dims="subject")
        z_sigma = pm.Normal("z_sigma", 0.0, 1.0, dims="subject")
        z_tau = pm.Normal("z_tau", 0.0, 1.0, dims="subject")
        z_theta = pm.Normal("z_theta", 0.0, 1.0, dims="subject")

        mu_i = pm.Deterministic("mu", mu_pop + sigma_mu * z_mu, dims="subject")
        sigma_i = pm.Deterministic(
            "sigma", pm.math.exp(log_sigma_pop + s_sigma * z_sigma), dims="subject")
        tau_base_i = pm.Deterministic(
            "tau_base", tau_pop * pm.math.exp(s_tau * z_tau), dims="subject")
        theta_i = pm.Deterministic(
            "theta", theta_pop + s_theta * z_theta, dims="subject")

        tau_it = pm.Deterministic(
            "tau_it",
            pm.math.clip(tau_base_i[codes] + theta_i[codes] * t_norm,
                         TAU_FLOOR, np.inf))
        pm.ExGaussian("rt", mu=mu_i[codes], sigma=sigma_i[codes],
                      nu=tau_it, observed=y)
    return model


def sample(model, tag: str):
    """Draw with nutpie (Rust NUTS) and fall back to the PyMC sampler."""
    import pymc as pm

    try:
        import nutpie  # noqa: F401
        sampler = "nutpie"
        extra = {"nuts_sampler": "nutpie"}
    except ImportError:
        sampler = "pymc-NUTS"
        extra = {"cores": min(CHAINS, 2)}

    # pointwise log-likelihood is needed for PSIS-LOO and Pareto-k diagnostics
    extra["idata_kwargs"] = {"log_likelihood": True}

    print(f"[mcmc] {tag}: tune={TUNE} draws={DRAWS} chains={CHAINS} "
          f"sampler={sampler} target_accept={TARGET_ACCEPT}")
    t0 = time.time()
    with model:
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=SEED,
                          progressbar=True, **extra)
    print(f"[mcmc] {tag}: {time.time() - t0:.1f}s")
    return idata


def convergence(idata, core_vars: list[str]) -> dict:
    """Summarise convergence on the population and individual parameters."""
    import arviz as az

    have = [v for v in core_vars if v in idata.posterior]
    try:
        summ = az.summary(idata, var_names=have, ci_prob=0.94)
    except TypeError:                       # ArviZ < 1.0 naming
        summ = az.summary(idata, var_names=have, hdi_prob=0.94)
    sum_col = "r_hat" if "r_hat" in summ.columns else \
        [c for c in summ.columns if "r_hat" in c][0]
    return {
        "summary": summ,
        "max_r_hat": float(np.nanmax(summ[sum_col].values)),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "divergences": int(np.asarray(idata.sample_stats["diverging"]).sum()),
        "n_rhat_gt_1.01": int(np.nansum(summ[sum_col].values > 1.01)),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 72)
    print("1. Trial table")
    print("=" * 72)
    y, codes, t_norm, subjects, trials = load_trials()
    ybar, sdy = float(y.mean()), float(y.std())
    print(f"observations={len(y)} operators={len(subjects)} "
          f"trials/operator={trials.groupby('subject_id').size().min()}-"
          f"{trials.groupby('subject_id').size().max()}")
    print(f"rt (s): mean={ybar:.4f} sd={sdy:.4f} range=[{y.min():.4f}, {y.max():.4f}]")
    print(f"t_norm: [{t_norm.min():.3f}, {t_norm.max():.3f}]")

    p99 = (trials.groupby("subject_id")[MODEL_CHANNEL]
           .apply(lambda v: np.percentile(v / US2S, 99)).loc[subjects].values)

    print("\n" + "=" * 72)
    print("2. Posterior sampling")
    print("=" * 72)
    model = build_model(y, codes, t_norm, subjects, ybar, sdy)
    idata = sample(model, "HB-TV-ExG")

    core_vars = ["mu_pop", "sigma_mu", "log_sigma_pop", "s_sigma",
                 "tau_pop", "s_tau", "theta_pop", "s_theta",
                 "mu", "sigma", "tau_base", "theta"]
    conv = convergence(idata, core_vars)
    conv["summary"].to_csv(TABLE_DIR / "hb_tv_posterior_summary.csv",
                           encoding="utf-8")
    ok = conv["max_r_hat"] < 1.01
    print(f"[diagnostics] max r-hat={conv['max_r_hat']:.4f} "
          f"({'converged' if ok else 'NOT converged'}) | "
          f"min ESS={conv['min_ess_bulk']:.0f} | "
          f"divergences={conv['divergences']} | "
          f"parameters with r-hat>1.01: {conv['n_rhat_gt_1.01']}")

    print("\n" + "=" * 72)
    print("2b. PSIS-LOO and Pareto-k diagnostics")
    print("=" * 72)
    import arviz as az

    loo = az.loo(idata, pointwise=True)
    pk = np.asarray(loo.pareto_k).reshape(-1)
    elpd_i = np.asarray(loo.elpd_i).reshape(-1)
    assert len(pk) == len(y), "Pareto-k length does not match the observation count"

    n_k07 = int((pk > 0.7).sum())
    n_k10 = int((pk > 1.0).sum())
    pd.DataFrame({
        "subject_id": trials["subject_id"].values,
        "trial_id": trials["trial_id"].values,
        "rt_s": y,
        "elpd_loo_i": elpd_i,
        "pareto_k": pk,
    }).to_csv(TABLE_DIR / "hb_tv_loo_pointwise.csv", index=False,
              encoding="utf-8")

    pd.DataFrame([{
        "model": "HB-TV-ExG", "n_obs": int(len(pk)),
        "elpd_loo": round(float(loo.elpd), 3),
        "se_elpd_loo": round(float(loo.se), 3),
        "p_loo": round(float(loo.p), 3),
        "pareto_k_max": round(float(pk.max()), 3),
        "n_pareto_k_gt_0.7": n_k07,
        "pct_pareto_k_gt_0.7": round(100 * n_k07 / len(pk), 2),
        "n_pareto_k_gt_1.0": n_k10,
        "pct_pareto_k_gt_1.0": round(100 * n_k10 / len(pk), 2),
        "n_chains": CHAINS, "n_draws": DRAWS, "seed": SEED,
    }]).to_csv(TABLE_DIR / "hb_tv_loo_diagnostics.csv", index=False,
              encoding="utf-8")

    print(f"ELPD_LOO={float(loo.elpd):.2f} (SE {float(loo.se):.2f}) | "
          f"p_loo={float(loo.p):.1f}")
    print(f"Pareto-k: max={pk.max():.3f} | k>0.7: {n_k07} trials "
          f"({100 * n_k07 / len(pk):.2f}%) | k>1.0: {n_k10} trials "
          f"({100 * n_k10 / len(pk):.2f}%)")

    print("\n" + "=" * 72)
    print("3. Individual feature extraction")
    print("=" * 72)
    tau_base = np.asarray(idata.posterior["tau_base"].mean(dim=("chain", "draw")))
    theta = np.asarray(idata.posterior["theta"].mean(dim=("chain", "draw")))
    print(f"tau_base: mean={tau_base.mean():.4f}s median={np.median(tau_base):.4f}s "
          f"range=[{tau_base.min():.4f}, {tau_base.max():.4f}]s")
    print(f"theta   : mean={theta.mean():+.4f}s median={np.median(theta):+.4f}s "
          f"range=[{theta.min():+.4f}, {theta.max():+.4f}]s "
          f"| share of positive slopes {(theta > 0).mean():.0%}")

    r_p, p_p = pearsonr(tau_base, p99)
    r_s, p_s = spearmanr(tau_base, p99)
    print(f"[check] tau_base vs empirical P99: Pearson r={r_p:.4f} (p={p_p:.2e}) | "
          f"Spearman rho={r_s:.4f} (p={p_s:.2e})")
    pd.DataFrame({"method": ["Pearson", "Spearman"], "r": [r_p, r_s],
                  "p_value": [p_p, p_s]}).to_csv(
        TABLE_DIR / "hb_tv_tau_base_vs_p99_correlation.csv",
        index=False, encoding="utf-8")

    print("\n" + "=" * 72)
    print("4. Two-axis risk quadrant assignment")
    print("=" * 72)
    quad = pd.DataFrame({"subject_id": subjects, "tau_base": tau_base,
                         "theta": theta, "p99_s": p99})
    tau_med, theta_med = quad["tau_base"].median(), quad["theta"].median()
    hi_base, hi_theta = quad["tau_base"] > tau_med, quad["theta"] > theta_med
    quad["quadrant"] = "Q3"
    quad.loc[hi_base & hi_theta, "quadrant"] = "Q1"
    quad.loc[~hi_base & hi_theta, "quadrant"] = "Q2"
    quad.loc[hi_base & ~hi_theta, "quadrant"] = "Q4"
    quad["quadrant_label"] = quad["quadrant"].map(QUADRANT_LABEL)
    quad.to_csv(TABLE_DIR / "hb_tv_quadrant_assignment.csv",
                index=False, encoding="utf-8")

    print(f"cut-offs: tau_base={tau_med:.4f}s | theta={theta_med:+.4f}s")
    for q in QUADRANTS:
        sub = quad[quad["quadrant"] == q]
        print(f"  {q} {QUADRANT_LABEL[q]:<22s} n={len(sub):2d} "
              f"({len(sub) / len(quad):.0%})  tau_base={sub['tau_base'].mean():.4f}s "
              f"theta={sub['theta'].mean():+.4f}s")

    pd.DataFrame([{
        "model": "HB-TV-ExG", "smoke": SMOKE,
        "tune": TUNE, "draws": DRAWS, "chains": CHAINS,
        "target_accept": TARGET_ACCEPT, "seed": SEED,
        "n_observations": int(len(y)), "n_subjects": int(len(subjects)),
        "max_r_hat": conv["max_r_hat"], "min_ess_bulk": conv["min_ess_bulk"],
        "divergences": conv["divergences"],
        "n_params_rhat_gt_1.01": conv["n_rhat_gt_1.01"],
        "theta_pop_mean": float(idata.posterior["theta_pop"].mean()),
        "elapsed_s": round(time.time() - t_start, 1),
    }]).to_csv(TABLE_DIR / "hb_tv_convergence_diagnostics.csv",
               index=False, encoding="utf-8")

    print(f"\n[done] tables -> {TABLE_DIR.relative_to(ROOT)}  "
          f"({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    main()
