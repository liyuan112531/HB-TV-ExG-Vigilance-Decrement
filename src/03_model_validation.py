"""Validation suite for the HB-TV-ExG model.

Five independent checks are run against the longitudinal trial table:

1. Predictive model comparison
   The time-varying model is refitted together with a static-exposure variant
   (``tau_i,t = tau_base_i``) on the identical 1,329 trials. Predictive accuracy
   is compared with leave-one-out cross-validation (PSIS-LOO) and WAIC, both
   computed from the pointwise log likelihood of the posterior draws.

2. Time-varying location ablation
   Two further specifications move the drift from the tail to the location
   (``HB-TV-Mu-ExG``) or to both components (``HB-TV-Both-ExG``). All four
   specifications share the same prior family and are compared on LOO and WAIC,
   so the comparison isolates where the time dependence lives.

3. Distributional adequacy
   Ex-Gaussian, lognormal, inverse Gaussian, gamma and normal densities are
   fitted by maximum likelihood on the pooled sample; AIC/BIC differences are
   bootstrapped with the operator as the resampling cluster (B = 1,000).

4. Sensitivity to quality control
   The model is refitted under four cleaning strategies:
   A  physical bounds only (0.15 s < RT < 10 s, no replacement, no trimming)
   A2 analyst ceiling of 5.0 s on single trials, set deliberately below the 6.0 s
      acquisition truncation
   B  single-pass 3 x IQR trimming
   C  main pipeline (3-sigma median replacement + 3 x IQR), N = 1,329
   Strategy A2 retains an untrimmed right tail, so its population scale prior is
   widened to HalfNormal(1.0) to avoid prior-data conflict.

5. Out-of-sample stability, posterior geometry and predictive checks
   Subject-stratified 10-fold cross-validation of the time-varying against the
   static-exposure model, computed from the analytic Ex-Gaussian density and so
   independent of the PSIS approximation; per-operator posterior quadrant
   classification probabilities; the posterior distribution of the session time
   at which the rising quadrant overtakes the stable one; quantile coverage of
   the posterior predictive distribution; and the time evolution of the
   threshold-exceedance probability P(RT > T_crit).

Outputs (``results/tables``)
    model_comparison.csv                 LOO / WAIC, time-varying vs static
    ablation_time_varying_location.csv   four specifications, LOO and WAIC
    ablation_population_effects.csv      delta_pop and theta_pop per ablation fit
    kfold_cv.csv / kfold_cv_summary.csv  subject-stratified 10-fold CV
    distribution_fit_full_sample.csv     five-density AIC/BIC on the pooled sample
    distribution_fit_cluster_bootstrap.csv  bootstrap CIs for dAIC / dBIC
    cleaning_sensitivity.csv             four-strategy refit summary
    posterior_quadrant_probability.csv   per-operator quadrant probabilities
    crossover_time.csv                   posterior Q2/Q4 crossover time
    ppc_coverage.csv / ppc_density_band.csv     posterior predictive checks
    risk_curves.csv / risk_escalation_summary.csv  P(RT > T_crit) curves

Run from the repository root (after ``02_hb_tv_exg_model.py``):
    python src/03_model_validation.py
    python src/03_model_validation.py --only-cleaning-sensitivity
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
RAW_CSV = ROOT / "data" / "pseudonymized_raw_trials.csv"
CLEAN_CSV = ROOT / "data" / "cleaned_trials_1329.csv"
for _d in (TABLE_DIR,):
    _d.mkdir(parents=True, exist_ok=True)

CHANNELS = ["foot_rt_us", "hand_rt_us"]
MODEL_CHANNEL = "foot_rt_us"
US2S = 1e6
SEED = 2026
TAU_FLOOR = 0.001
SMOKE = os.environ.get("HBM_SMOKE", "0") == "1"
TUNE, DRAWS, CHAINS, TARGET_ACCEPT = (100, 200, 1, 0.92) if SMOKE \
    else (1000, 2000, 4, 0.92)
N_BOOT = 20 if SMOKE else 1000
T_GRID = np.linspace(0.0, 1.0, 101)      # fraction of the 20 min monitoring block
THRESHOLDS = (2.0, 2.5, 3.0)             # critical takeover budgets (s)
EXPECTED_THETA_POP = 0.4794              # reference run, seed 2026
EXPECTED_N_CLEAN = 1329
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
QUADRANT_LABEL = {"Q1": "Extreme high-risk", "Q2": "Fatigue deteriorating",
                  "Q3": "Robust throughout", "Q4": "Static sluggish"}


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Model definition and sampling (identical settings to 02)
# ---------------------------------------------------------------------------
def prepare_arrays(df: pd.DataFrame) -> dict:
    """DataFrame(subject_id, trial_id, rt) -> arrays for the likelihood."""
    subjects = sorted(df["subject_id"].astype(str).unique())
    codes = df["subject_id"].astype(str).map({s: i for i, s in enumerate(subjects)})
    y = df[MODEL_CHANNEL].values.astype(np.float64) / US2S
    t_norm = (df["trial_id"].values.astype(np.float64) - 1.0) / 29.0
    return {"y": y, "codes": codes.values.astype(int), "t_norm": t_norm,
            "subjects": subjects, "ybar": float(y.mean()), "sdy": float(y.std())}


def build_tv_model(arr: dict, tau_prior_sd: float = 0.5):
    """HB-TV-ExG with tau_i,t = clip(tau_base_i + theta_i * t_norm, 0.001, inf).

    ``tau_prior_sd`` sets tau_pop ~ HalfNormal(tau_prior_sd); the main pipeline
    and the model comparison use 0.5, the untrimmed-tail strategies use 1.0.
    """
    import pymc as pm

    subjects, codes, t_norm = arr["subjects"], arr["codes"], arr["t_norm"]
    with pm.Model(coords={"subject": subjects}) as model:
        mu_pop = pm.Normal("mu_pop", mu=arr["ybar"], sigma=1.0)
        sigma_mu = pm.HalfNormal("sigma_mu", sigma=0.3)
        log_sigma_pop = pm.Normal("log_sigma_pop",
                                  mu=np.log(max(arr["sdy"], 1e-3)) - 1.0, sigma=0.5)
        s_sigma = pm.HalfNormal("s_sigma", sigma=0.4)
        tau_pop = pm.HalfNormal("tau_pop", sigma=tau_prior_sd)
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
                      nu=tau_it, observed=arr["y"])
    return model


def build_static_model(arr: dict):
    """HB-Static-ExG: tau_i,t = tau_base_i, no time slope, priors otherwise equal."""
    import pymc as pm

    subjects, codes = arr["subjects"], arr["codes"]
    with pm.Model(coords={"subject": subjects}) as model:
        mu_pop = pm.Normal("mu_pop", mu=arr["ybar"], sigma=1.0)
        sigma_mu = pm.HalfNormal("sigma_mu", sigma=0.3)
        log_sigma_pop = pm.Normal("log_sigma_pop",
                                  mu=np.log(max(arr["sdy"], 1e-3)) - 1.0, sigma=0.5)
        s_sigma = pm.HalfNormal("s_sigma", sigma=0.4)
        tau_pop = pm.HalfNormal("tau_pop", sigma=0.5)
        s_tau = pm.HalfNormal("s_tau", sigma=0.5)

        z_mu = pm.Normal("z_mu", 0.0, 1.0, dims="subject")
        z_sigma = pm.Normal("z_sigma", 0.0, 1.0, dims="subject")
        z_tau = pm.Normal("z_tau", 0.0, 1.0, dims="subject")

        mu_i = pm.Deterministic("mu", mu_pop + sigma_mu * z_mu, dims="subject")
        sigma_i = pm.Deterministic(
            "sigma", pm.math.exp(log_sigma_pop + s_sigma * z_sigma), dims="subject")
        tau_base_i = pm.Deterministic(
            "tau_base", tau_pop * pm.math.exp(s_tau * z_tau), dims="subject")
        pm.ExGaussian("rt", mu=mu_i[codes], sigma=sigma_i[codes],
                      nu=tau_base_i[codes], observed=arr["y"])
    return model


def sample_model(model, tag: str, with_loglik: bool = True):
    import pymc as pm

    kw = dict(draws=DRAWS, tune=TUNE, chains=CHAINS, nuts_sampler="nutpie",
              target_accept=TARGET_ACCEPT, random_seed=SEED, progressbar=False)
    if with_loglik:
        kw["idata_kwargs"] = {"log_likelihood": True}
    t0 = time.time()
    with model:
        idata = pm.sample(**kw)
    elapsed = time.time() - t0
    print(f"  [{tag}] sampled in {elapsed:.1f}s")
    return idata, elapsed


def draw_matrix(idata, name: str) -> np.ndarray:
    """(chain, draw, subject) -> (n_draws, subject)."""
    a = np.asarray(idata.posterior[name])
    return a.reshape(-1, a.shape[-1]).astype(np.float64)


def core_summary(idata) -> dict:
    """Convergence and hyper-parameter summary for the core parameters."""
    import arviz as az

    core = ["mu_pop", "sigma_mu", "log_sigma_pop", "s_sigma",
            "tau_pop", "s_tau", "theta_pop", "s_theta",
            "mu", "sigma", "tau_base", "theta"]
    have = [v for v in core if v in idata.posterior]
    try:
        summ = az.summary(idata, var_names=have, ci_prob=0.94)
    except TypeError:
        summ = az.summary(idata, var_names=have, hdi_prob=0.94)

    def mean_of(name):
        return float(np.asarray(idata.posterior[name]).mean())

    def eti(name):
        v = np.asarray(idata.posterior[name]).reshape(-1)
        return float(np.percentile(v, 3)), float(np.percentile(v, 97))

    out = {
        "max_r_hat": float(np.nanmax(summ["r_hat"].values)),
        "min_ess_bulk": float(np.nanmin(summ["ess_bulk"].values)),
        "divergences": int(np.asarray(idata.sample_stats["diverging"]).sum()),
        "theta_pop_mean": mean_of("theta_pop") if "theta_pop" in idata.posterior else np.nan,
        "tau_pop_mean": mean_of("tau_pop"),
    }
    if "theta_pop" in idata.posterior:
        lo, hi = eti("theta_pop")
        out["theta_pop_eti94_lb"], out["theta_pop_eti94_ub"] = lo, hi
    else:
        out["theta_pop_eti94_lb"] = out["theta_pop_eti94_ub"] = np.nan
    return out


# ---------------------------------------------------------------------------
# 1. Predictive model comparison
# ---------------------------------------------------------------------------
def waic_pointwise(idata, lik: str = "rt") -> dict:
    """WAIC by definition (lppd - p_WAIC), ArviZ offers no direct helper."""
    ll = np.asarray(idata.log_likelihood[lik])
    ll = ll.reshape(-1, ll.shape[-1]).astype(np.float64)
    n_draws, n_obs = ll.shape
    lppd_i = logsumexp(ll, axis=0) - np.log(n_draws)
    p_i = ll.var(axis=0, ddof=1)
    elpd_i = lppd_i - p_i
    return {"elpd_i": elpd_i, "elpd": float(elpd_i.sum()), "p": float(p_i.sum()),
            "se": float(np.sqrt(n_obs * elpd_i.var(ddof=1))), "n_obs": n_obs}


def loo_pointwise(idata) -> dict:
    """PSIS-LOO pointwise expected log predictive density."""
    import arviz as az

    loo = az.loo(idata, pointwise=True)
    pk = getattr(loo, "pareto_k", None)
    return {
        "elpd_i": np.asarray(loo.elpd_i).reshape(-1),
        "elpd": float(loo.elpd), "p": float(loo.p), "se": float(loo.se),
        "pareto_k_max": float(np.nanmax(np.asarray(pk))) if pk is not None else np.nan,
        "n_obs": int(loo.n_data_points),
    }


def run_model_comparison():
    section("1. Predictive comparison: HB-Static-ExG vs HB-TV-ExG (N = 1,329)")
    arr = prepare_arrays(pd.read_csv(CLEAN_CSV))
    n = len(arr["y"])
    assert n == EXPECTED_N_CLEAN, f"unexpected analysis sample size: {n}"
    print(f"analysis set: N={n} operators={len(arr['subjects'])} "
          f"rt mean={arr['ybar']:.4f}s sd={arr['sdy']:.4f}s")

    print("  [1/2] refitting HB-TV-ExG with pointwise log likelihood ...")
    idata_tv, t_tv = sample_model(build_tv_model(arr), "HB-TV-ExG")
    s_tv = core_summary(idata_tv)
    drift = s_tv["theta_pop_mean"] - EXPECTED_THETA_POP
    print(f"    theta_pop = {s_tv['theta_pop_mean']:.4f} "
          f"(reference {EXPECTED_THETA_POP:.4f}, deviation {drift:+.2e})")
    if not SMOKE:
        assert abs(drift) < 5e-3, (
            "theta_pop deviates from the reference run; check the trial table and "
            "the sampling settings before interpreting downstream checks")

    print("  [2/2] fitting HB-Static-ExG (tau_i,t = tau_base_i) ...")
    idata_st, t_st = sample_model(build_static_model(arr), "HB-Static-ExG")
    s_st = core_summary(idata_st)

    loo_tv, loo_st = loo_pointwise(idata_tv), loo_pointwise(idata_st)
    waic_tv, waic_st = waic_pointwise(idata_tv), waic_pointwise(idata_st)
    d_loo_i = loo_tv["elpd_i"] - loo_st["elpd_i"]
    d_waic_i = waic_tv["elpd_i"] - waic_st["elpd_i"]
    dse_loo = float(np.sqrt(n * d_loo_i.var(ddof=1)))
    dse_waic = float(np.sqrt(n * d_waic_i.var(ddof=1)))

    rows = []
    for name, s, loo, wa, elapsed in (
            ("HB-TV-ExG", s_tv, loo_tv, waic_tv, t_tv),
            ("HB-Static-ExG", s_st, loo_st, waic_st, t_st)):
        rows.append({
            "model": name,
            "description": "time-varying fatigue slope theta_i" if name == "HB-TV-ExG"
            else "static tail risk, no time slope",
            "n_obs": n, "n_subjects": len(arr["subjects"]),
            "n_draws": DRAWS * CHAINS,
            "elpd_loo": round(loo["elpd"], 3), "se_elpd_loo": round(loo["se"], 3),
            "p_loo": round(loo["p"], 3),
            "elpd_waic": round(wa["elpd"], 3), "se_elpd_waic": round(wa["se"], 3),
            "p_waic": round(wa["p"], 3),
            "pareto_k_max": round(loo["pareto_k_max"], 3),
            "max_r_hat": round(s["max_r_hat"], 4),
            "min_ess_bulk": round(s["min_ess_bulk"], 1),
            "divergences": s["divergences"],
            "theta_pop_mean": round(s["theta_pop_mean"], 4),
            "theta_pop_eti94_lb": round(s["theta_pop_eti94_lb"], 4),
            "theta_pop_eti94_ub": round(s["theta_pop_eti94_ub"], 4),
            "tau_pop_mean": round(s["tau_pop_mean"], 4),
            "elapsed_s": round(elapsed, 1), "note": "",
        })
    rows.append({
        "model": "HB-TV-ExG vs HB-Static-ExG",
        "description": "paired difference (time-varying minus static); "
                       "positive values favour the time-varying model",
        "n_obs": n, "n_subjects": len(arr["subjects"]), "n_draws": DRAWS * CHAINS,
        "elpd_loo": np.nan, "se_elpd_loo": np.nan, "p_loo": np.nan,
        "elpd_waic": np.nan, "se_elpd_waic": np.nan, "p_waic": np.nan,
        "pareto_k_max": np.nan, "max_r_hat": np.nan, "min_ess_bulk": np.nan,
        "divergences": np.nan, "theta_pop_mean": np.nan,
        "theta_pop_eti94_lb": np.nan, "theta_pop_eti94_ub": np.nan,
        "tau_pop_mean": np.nan, "elapsed_s": np.nan,
        "note": (f"d_elpd_loo={loo_tv['elpd'] - loo_st['elpd']:+.2f}, "
                 f"dSE_loo={dse_loo:.2f}; "
                 f"d_elpd_waic={waic_tv['elpd'] - waic_st['elpd']:+.2f}, "
                 f"dSE_waic={dse_waic:.2f}"),
    })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "model_comparison.csv", index=False, encoding="utf-8")

    print(f"\n  LOO : time-varying {loo_tv['elpd']:.2f} (p_loo {loo_tv['p']:.1f}) | "
          f"static {loo_st['elpd']:.2f} (p_loo {loo_st['p']:.1f}) | "
          f"dELPD {loo_tv['elpd'] - loo_st['elpd']:+.2f} +/- {dse_loo:.2f}")
    print(f"  WAIC: time-varying {waic_tv['elpd']:.2f} | static {waic_st['elpd']:.2f} | "
          f"dELPD {waic_tv['elpd'] - waic_st['elpd']:+.2f} +/- {dse_waic:.2f}")
    print(f"  -> {TABLE_DIR / 'model_comparison.csv'}")
    return idata_tv, arr


# ---------------------------------------------------------------------------
# 2. Distributional adequacy
# ---------------------------------------------------------------------------
DIST_NAMES = ["Ex-Gaussian", "Lognormal", "Inverse Gaussian", "Gamma", "Normal"]
_EXG_INIT = np.array([0.85, 0.14, 0.50])


def _exg_fit(x: np.ndarray, init=None):
    """Ex-Gaussian MLE in (mu, sigma, tau), L-BFGS-B with two starts."""
    init = _EXG_INIT if init is None else np.asarray(init, float)

    def nll(p):
        mu, sg, ta = p
        if sg <= 1e-5 or ta <= 1e-5:
            return 1e12
        return -float(np.sum(stats.exponnorm.logpdf(x, ta / sg, loc=mu, scale=sg)))

    starts = [init, np.array([x.mean() - 0.5, max(x.std() * 0.5, 1e-3), 0.4])]
    best = None
    for s0 in starts:
        r = minimize(nll, s0, method="L-BFGS-B",
                     bounds=[(-5.0, 8.0), (1e-4, 5.0), (1e-4, 12.0)])
        if best is None or r.fun < best.fun:
            best = r
    return 3, -float(best.fun), tuple(float(v) for v in best.x)


def fit_all_dists(x: np.ndarray, exg_init=None) -> dict:
    """Return {density: {k, loglik, params, aic, bic}} for the five candidates."""
    n = len(x)
    mu, sd = float(x.mean()), float(x.std(ddof=0))
    lg = np.log(x)
    s_ln, scale_ln = float(lg.std(ddof=0)), float(np.exp(lg.mean()))
    g = stats.gamma.fit(x, floc=0)
    ig = stats.invgauss.fit(x, floc=0)
    fits = {
        "Normal": (2, float(np.sum(stats.norm.logpdf(x, mu, sd))), (mu, sd)),
        "Lognormal": (2, float(np.sum(stats.lognorm.logpdf(x, s_ln, 0, scale_ln))),
                      (s_ln, scale_ln)),
        "Gamma": (2, float(np.sum(stats.gamma.logpdf(x, g[0], 0, g[2]))), tuple(g)),
        "Inverse Gaussian": (2, float(np.sum(stats.invgauss.logpdf(x, ig[0], 0, ig[2]))),
                             tuple(ig)),
        "Ex-Gaussian": _exg_fit(x, exg_init),
    }
    return {name: {"k": k, "loglik": ll, "params": p,
                   "aic": 2 * k - 2 * ll, "bic": k * np.log(n) - 2 * ll}
            for name, (k, ll, p) in fits.items()}


def run_cluster_bootstrap():
    section(f"2. Five-density comparison, cluster bootstrap over operators "
            f"(B = {N_BOOT})")
    clean = pd.read_csv(CLEAN_CSV)
    clean["rt_s"] = clean[MODEL_CHANNEL] / US2S
    subjects = sorted(clean["subject_id"].astype(str).unique())
    groups = {s: clean.loc[clean["subject_id"].astype(str) == s, "rt_s"].values
              for s in subjects}
    x_all = np.concatenate([groups[s] for s in subjects])
    print(f"pooled sample: N={len(x_all)} operators={len(subjects)} "
          f"mean={x_all.mean():.4f}s sd={x_all.std():.4f}s")

    full = fit_all_dists(x_all)
    full_df = pd.DataFrame([{
        "distribution": name, "k_params": v["k"],
        "loglik": round(v["loglik"], 3),
        "AIC": round(v["aic"], 3), "BIC": round(v["bic"], 3),
        "params": ", ".join(f"{p:.5g}" for p in v["params"]),
    } for name, v in full.items()]).sort_values("AIC").reset_index(drop=True)
    full_df.to_csv(TABLE_DIR / "distribution_fit_full_sample.csv",
                   index=False, encoding="utf-8")
    print(full_df.to_string(index=False))

    aic_order = sorted(full, key=lambda k: full[k]["aic"])
    second = aic_order[1]
    d_aic_full = full[second]["aic"] - full["Ex-Gaussian"]["aic"]
    d_bic_full = full[second]["bic"] - full["Ex-Gaussian"]["bic"]
    print(f"runner-up density = {second} | dAIC(2nd - ExG)={d_aic_full:+.2f} | "
          f"dBIC(2nd - ExG)={d_bic_full:+.2f}")

    rng = np.random.default_rng(SEED)
    exg_init = np.array(full["Ex-Gaussian"]["params"], dtype=float)
    d_aic, d_bic, second_names, exg_best = [], [], [], []
    t0 = time.time()
    for b in range(N_BOOT):
        pick = rng.integers(0, len(subjects), len(subjects))
        xb = np.concatenate([groups[subjects[j]] for j in pick])
        try:
            fb = fit_all_dists(xb, exg_init)
        except Exception:
            continue
        order = sorted(fb, key=lambda k: fb[k]["aic"])
        sec = order[1]
        d_aic.append(fb[sec]["aic"] - fb["Ex-Gaussian"]["aic"])
        d_bic.append(fb[sec]["bic"] - fb["Ex-Gaussian"]["bic"])
        second_names.append(sec)
        exg_best.append(order[0] == "Ex-Gaussian")
        if (b + 1) % 200 == 0:
            print(f"    bootstrap {b + 1}/{N_BOOT} ({time.time() - t0:.0f}s)")

    d_aic, d_bic = np.array(d_aic), np.array(d_bic)
    rows = [{
        "statistic": "delta_AIC_runner_up_minus_ExGaussian",
        "point_estimate_full_sample": round(d_aic_full, 3),
        "boot_mean": round(float(d_aic.mean()), 3),
        "boot_sd": round(float(d_aic.std(ddof=1)), 3),
        "ci95_low": round(float(np.percentile(d_aic, 2.5)), 3),
        "ci95_high": round(float(np.percentile(d_aic, 97.5)), 3),
        "n_boot_valid": len(d_aic),
        "frac_exgaussian_best_aic": round(float(np.mean(exg_best)), 4),
        "modal_runner_up": pd.Series(second_names).mode().iloc[0],
        "note": "positive values favour Ex-Gaussian; cluster unit = operator",
    }, {
        "statistic": "delta_BIC_runner_up_minus_ExGaussian",
        "point_estimate_full_sample": round(d_bic_full, 3),
        "boot_mean": round(float(d_bic.mean()), 3),
        "boot_sd": round(float(d_bic.std(ddof=1)), 3),
        "ci95_low": round(float(np.percentile(d_bic, 2.5)), 3),
        "ci95_high": round(float(np.percentile(d_bic, 97.5)), 3),
        "n_boot_valid": len(d_bic),
        "frac_exgaussian_best_aic": round(float(np.mean(exg_best)), 4),
        "modal_runner_up": pd.Series(second_names).mode().iloc[0],
        "note": "BIC penalty uses the bootstrap sample size; cluster unit = operator",
    }]
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "distribution_fit_cluster_bootstrap.csv",
               index=False, encoding="utf-8")
    print(f"\n  dAIC(2nd-ExG) = {d_aic_full:+.2f} | bootstrap 95% CI "
          f"[{np.percentile(d_aic, 2.5):.2f}, {np.percentile(d_aic, 97.5):.2f}] | "
          f"Ex-Gaussian best in {np.mean(exg_best):.1%} of resamples")
    print(f"  dBIC(2nd-ExG) = {d_bic_full:+.2f} | bootstrap 95% CI "
          f"[{np.percentile(d_bic, 2.5):.2f}, {np.percentile(d_bic, 97.5):.2f}]")
    return out


# ---------------------------------------------------------------------------
# 3. Sensitivity to quality control
# ---------------------------------------------------------------------------
def build_strategy_frames() -> dict:
    """Construct the four cleaning variants from the de-identified trial table."""
    raw = pd.read_csv(RAW_CSV)
    frames = {}

    lo, hi = 0.15 * US2S, 10.0 * US2S
    a = raw[raw[CHANNELS[0]].between(lo, hi) & raw[CHANNELS[1]].between(lo, hi)]
    frames["A"] = (a.reset_index(drop=True),
                   "Physical bounds only: drop RT < 0.15 s or > 10 s on both "
                   "channels; no 3-sigma replacement, no IQR trimming")

    lo2, hi2 = 0.15 * US2S, 5.0 * US2S
    a2 = raw[raw[CHANNELS[0]].between(lo2, hi2) & raw[CHANNELS[1]].between(lo2, hi2)]
    frames["A2"] = (a2.reset_index(drop=True),
                    "Single-trial window: drop RT < 0.15 s (anticipation) or "
                    "> 5.0 s (cross-trial omission beyond the 5 s ITI); no "
                    "replacement, no trimming; tau_pop prior widened to "
                    "HalfNormal(1.0)")

    b = raw.copy()
    for c in CHANNELS:
        q1, q3 = b[c].quantile(0.25), b[c].quantile(0.75)
        iqr = q3 - q1
        b = b[b[c].between(q1 - 3 * iqr, q3 + 3 * iqr)].reset_index(drop=True)
    frames["B"] = (b, "Single-pass 3 x IQR trimming per channel "
                      "(no 3-sigma replacement)")

    c = raw.copy()
    for col in CHANNELS:
        c[col] = c[col].astype(float)
        mu, sd = c[col].mean(), c[col].std()
        lo3, hi3 = max(mu - 3 * sd, 0), mu + 3 * sd
        c.loc[(c[col] < lo3) | (c[col] > hi3), col] = c[col].median()
        q1, q3 = c[col].quantile(0.25), c[col].quantile(0.75)
        iqr = q3 - q1
        c = c[c[col].between(q1 - 3 * iqr, q3 + 3 * iqr)].reset_index(drop=True)
    frames["C"] = (c, "Main pipeline: 3-sigma median replacement + 3 x IQR "
                      "row removal")
    return frames


def classify_quadrants(tau, theta, tau_med, theta_med) -> np.ndarray:
    q = np.full(len(tau), "Q3", dtype=object)
    hi_b, hi_t = tau > tau_med, theta > theta_med
    q[hi_b & hi_t] = "Q1"
    q[~hi_b & hi_t] = "Q2"
    q[hi_b & ~hi_t] = "Q4"
    return q


def fit_strategy(df: pd.DataFrame, tag: str, tau_prior_sd: float = 0.5) -> dict:
    """Refit HB-TV-ExG on one cleaning variant and summarise the outcome."""
    arr = prepare_arrays(df)
    idata, elapsed = sample_model(build_tv_model(arr, tau_prior_sd), tag,
                                  with_loglik=False)
    s = core_summary(idata)
    tau_base = draw_matrix(idata, "tau_base").mean(axis=0)
    theta = draw_matrix(idata, "theta").mean(axis=0)
    quad = classify_quadrants(tau_base, theta, np.median(tau_base), np.median(theta))
    counts = {q: int((quad == q).sum()) for q in QUADRANTS}
    return {
        "n_rows": len(df), "n_subjects": len(arr["subjects"]),
        "rt_min_s": float(arr["y"].min()), "rt_max_s": float(arr["y"].max()),
        "rt_mean_s": float(arr["y"].mean()), "rt_sd_s": float(arr["y"].std()),
        "theta_pop_mean": s["theta_pop_mean"],
        "theta_pop_eti94_lb": s["theta_pop_eti94_lb"],
        "theta_pop_eti94_ub": s["theta_pop_eti94_ub"],
        "pct_theta_gt0": float((theta > 0).mean() * 100),
        "tau_pop_mean": s["tau_pop_mean"], "tau_prior_sd": tau_prior_sd,
        "tau_base_mean": float(tau_base.mean()), "theta_mean": float(theta.mean()),
        "max_r_hat": s["max_r_hat"], "min_ess_bulk": s["min_ess_bulk"],
        "divergences": s["divergences"], "elapsed_s": round(elapsed, 1),
        "quad_counts": counts,
        "pct": {q: 100.0 * counts[q] / len(quad) for q in counts},
    }


def run_cleaning_sensitivity():
    section("3. Sensitivity to quality control (A / A2 / B / C)")
    frames = build_strategy_frames()
    n_raw = len(pd.read_csv(RAW_CSV))
    c_df = frames["C"][0]
    clean_ref = pd.read_csv(CLEAN_CSV)
    same_n = len(c_df) == len(clean_ref) == EXPECTED_N_CLEAN
    same_vals = np.allclose(np.sort(c_df[MODEL_CHANNEL].values),
                            np.sort(clean_ref[MODEL_CHANNEL].values))
    print(f"raw trials = {n_raw}")
    print(f"strategy C reproduces the analysis set: N={len(c_df)} "
          f"(expected {EXPECTED_N_CLEAN}), identical RT multiset = {same_vals}")
    assert same_n and same_vals, "strategy C does not reproduce 01_data_cleaning.py"

    tau_prior = {"A": 0.5, "A2": 1.0, "B": 0.5, "C": 0.5}
    rows = []
    for key in ("A", "A2", "B", "C"):
        df, desc = frames[key]
        retention = 100.0 * len(df) / n_raw
        print(f"\n  strategy {key}: N={len(df)} ({retention:.1f}% retained) | "
              f"tau_pop ~ HalfNormal({tau_prior[key]})")
        r = fit_strategy(df, f"strategy-{key}", tau_prior_sd=tau_prior[key])
        r.update({"strategy": key, "description": desc, "retention_pct": retention})
        rows.append(r)

    out = pd.DataFrame([{
        "strategy": r["strategy"], "description": r["description"],
        "n_observations": r["n_rows"], "retention_pct": round(r["retention_pct"], 1),
        "n_subjects": r["n_subjects"],
        "rt_min_s": round(r["rt_min_s"], 4), "rt_max_s": round(r["rt_max_s"], 4),
        "rt_mean_s": round(r["rt_mean_s"], 4), "rt_sd_s": round(r["rt_sd_s"], 4),
        "theta_pop_mean": round(r["theta_pop_mean"], 4),
        "theta_pop_eti94_lb": round(r["theta_pop_eti94_lb"], 4),
        "theta_pop_eti94_ub": round(r["theta_pop_eti94_ub"], 4),
        "pct_subjects_theta_gt0": round(r["pct_theta_gt0"], 1),
        "tau_pop_mean": round(r["tau_pop_mean"], 4),
        "tau_prior_sd": r["tau_prior_sd"],
        "tau_base_mean": round(r["tau_base_mean"], 4),
        "theta_mean": round(r["theta_mean"], 4),
        "Q1_n": r["quad_counts"]["Q1"], "Q1_pct": round(r["pct"]["Q1"], 1),
        "Q2_n": r["quad_counts"]["Q2"], "Q2_pct": round(r["pct"]["Q2"], 1),
        "Q3_n": r["quad_counts"]["Q3"], "Q3_pct": round(r["pct"]["Q3"], 1),
        "Q4_n": r["quad_counts"]["Q4"], "Q4_pct": round(r["pct"]["Q4"], 1),
        "max_r_hat": round(r["max_r_hat"], 4),
        "min_ess_bulk": round(r["min_ess_bulk"], 1),
        "divergences": r["divergences"], "elapsed_s": r["elapsed_s"],
    } for r in rows])
    out.to_csv(TABLE_DIR / "cleaning_sensitivity.csv", index=False, encoding="utf-8")
    cols = ["strategy", "n_observations", "retention_pct", "theta_pop_mean",
            "theta_pop_eti94_lb", "theta_pop_eti94_ub", "pct_subjects_theta_gt0",
            "tau_pop_mean", "tau_prior_sd", "Q1_pct", "Q2_pct", "Q3_pct", "Q4_pct",
            "max_r_hat", "divergences"]
    print("\n" + out[cols].to_string(index=False))
    print(f"  -> {TABLE_DIR / 'cleaning_sensitivity.csv'}")
    return out


# ---------------------------------------------------------------------------
# 4. Out-of-sample stability, posterior predictive checks, risk evolution
# ---------------------------------------------------------------------------
def build_tv_mu_model(arr: dict):
    """HB-TV-Mu-ExG: location drifts, tail scale static (ablation)."""
    import pymc as pm

    subjects, codes, t_norm = arr["subjects"], arr["codes"], arr["t_norm"]
    with pm.Model(coords={"subject": subjects}) as model:
        mu_pop = pm.Normal("mu_pop", mu=arr["ybar"], sigma=1.0)
        sigma_mu = pm.HalfNormal("sigma_mu", sigma=0.3)
        log_sigma_pop = pm.Normal("log_sigma_pop",
                                  mu=np.log(max(arr["sdy"], 1e-3)) - 1.0, sigma=0.5)
        s_sigma = pm.HalfNormal("s_sigma", sigma=0.4)
        tau_pop = pm.HalfNormal("tau_pop", sigma=0.5)
        s_tau = pm.HalfNormal("s_tau", sigma=0.5)
        delta_pop = pm.Normal("delta_pop", mu=0.0, sigma=0.3)
        s_delta = pm.HalfNormal("s_delta", sigma=0.3)

        z_mu = pm.Normal("z_mu", 0.0, 1.0, dims="subject")
        z_sigma = pm.Normal("z_sigma", 0.0, 1.0, dims="subject")
        z_tau = pm.Normal("z_tau", 0.0, 1.0, dims="subject")
        z_delta = pm.Normal("z_delta", 0.0, 1.0, dims="subject")

        mu_base_i = pm.Deterministic("mu_base", mu_pop + sigma_mu * z_mu,
                                     dims="subject")
        sigma_i = pm.Deterministic(
            "sigma", pm.math.exp(log_sigma_pop + s_sigma * z_sigma), dims="subject")
        tau_base_i = pm.Deterministic(
            "tau_base", tau_pop * pm.math.exp(s_tau * z_tau), dims="subject")
        delta_i = pm.Deterministic("delta", delta_pop + s_delta * z_delta,
                                   dims="subject")
        mu_it = pm.Deterministic("mu_it",
                                 mu_base_i[codes] + delta_i[codes] * t_norm)
        pm.ExGaussian("rt", mu=mu_it, sigma=sigma_i[codes],
                      nu=tau_base_i[codes], observed=arr["y"])
    return model


def build_tv_both_model(arr: dict):
    """HB-TV-Both-ExG: both location and tail scale drift (ablation)."""
    import pymc as pm

    subjects, codes, t_norm = arr["subjects"], arr["codes"], arr["t_norm"]
    with pm.Model(coords={"subject": subjects}) as model:
        mu_pop = pm.Normal("mu_pop", mu=arr["ybar"], sigma=1.0)
        sigma_mu = pm.HalfNormal("sigma_mu", sigma=0.3)
        log_sigma_pop = pm.Normal("log_sigma_pop",
                                  mu=np.log(max(arr["sdy"], 1e-3)) - 1.0, sigma=0.5)
        s_sigma = pm.HalfNormal("s_sigma", sigma=0.4)
        tau_pop = pm.HalfNormal("tau_pop", sigma=0.5)
        s_tau = pm.HalfNormal("s_tau", sigma=0.5)
        theta_pop = pm.Normal("theta_pop", mu=0.0, sigma=0.3)
        s_theta = pm.HalfNormal("s_theta", sigma=0.3)
        delta_pop = pm.Normal("delta_pop", mu=0.0, sigma=0.3)
        s_delta = pm.HalfNormal("s_delta", sigma=0.3)

        z_mu = pm.Normal("z_mu", 0.0, 1.0, dims="subject")
        z_sigma = pm.Normal("z_sigma", 0.0, 1.0, dims="subject")
        z_tau = pm.Normal("z_tau", 0.0, 1.0, dims="subject")
        z_theta = pm.Normal("z_theta", 0.0, 1.0, dims="subject")
        z_delta = pm.Normal("z_delta", 0.0, 1.0, dims="subject")

        mu_base_i = pm.Deterministic("mu_base", mu_pop + sigma_mu * z_mu,
                                     dims="subject")
        sigma_i = pm.Deterministic(
            "sigma", pm.math.exp(log_sigma_pop + s_sigma * z_sigma), dims="subject")
        tau_base_i = pm.Deterministic(
            "tau_base", tau_pop * pm.math.exp(s_tau * z_tau), dims="subject")
        theta_i = pm.Deterministic(
            "theta", theta_pop + s_theta * z_theta, dims="subject")
        delta_i = pm.Deterministic("delta", delta_pop + s_delta * z_delta,
                                   dims="subject")
        mu_it = pm.Deterministic("mu_it",
                                 mu_base_i[codes] + delta_i[codes] * t_norm)
        tau_it = pm.Deterministic(
            "tau_it",
            pm.math.clip(tau_base_i[codes] + theta_i[codes] * t_norm,
                         TAU_FLOOR, np.inf))
        pm.ExGaussian("rt", mu=mu_it, sigma=sigma_i[codes],
                      nu=tau_it, observed=arr["y"])
    return model


def run_ablation(arr: dict) -> pd.DataFrame:
    """Location ablation: static / location-only / tail-only / both."""
    section("Step 3: time-varying location ablation (LOO / WAIC)")
    n = len(arr["y"])
    specs = [
        ("HB-Static-ExG", "neither mu nor tau varies with time", build_static_model),
        ("HB-TV-Mu-ExG", "mu_it = mu_base_i + delta_i t_norm; tau static",
         build_tv_mu_model),
        ("HB-TV-ExG", "tau_it = tau_base_i + theta_i t_norm; mu static (main)",
         build_tv_model),
        ("HB-TV-Both-ExG", "both mu_it and tau_it drift linearly",
         build_tv_both_model),
    ]
    fits = {}
    for name, desc, builder in specs:
        idata, _ = sample_model(builder(arr), name, with_loglik=True)
        fits[name] = dict(desc=desc, loo=loo_pointwise(idata),
                          wa=waic_pointwise(idata), core=core_summary(idata),
                          idata=idata)

    base = fits["HB-TV-ExG"]["loo"]["elpd_i"]
    rows = []
    for name, desc, _ in specs:
        f = fits[name]
        d_i = f["loo"]["elpd_i"] - base
        rows.append({
            "model": name, "description": desc,
            "n_obs": n, "n_subjects": len(arr["subjects"]),
            "n_draws": DRAWS * CHAINS,
            "elpd_loo": round(f["loo"]["elpd"], 3),
            "se_elpd_loo": round(f["loo"]["se"], 3),
            "p_loo": round(f["loo"]["p"], 3),
            "elpd_waic": round(f["wa"]["elpd"], 3),
            "p_waic": round(f["wa"]["p"], 3),
            "pareto_k_max": round(f["loo"]["pareto_k_max"], 3),
            "d_elpd_loo_vs_main": round(float(f["loo"]["elpd"]
                                             - fits["HB-TV-ExG"]["loo"]["elpd"]), 3),
            "dse_vs_main": round(float(np.sqrt(n * d_i.var(ddof=1))), 3),
            "max_r_hat": round(f["core"]["max_r_hat"], 4),
            "min_ess_bulk": round(f["core"]["min_ess_bulk"], 1),
            "divergences": f["core"]["divergences"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "ablation_time_varying_location.csv",
               index=False, encoding="utf-8-sig")

    eff = []
    for name in ("HB-TV-Mu-ExG", "HB-TV-Both-ExG"):
        for pname in ("delta_pop", "theta_pop", "mu_pop", "tau_pop"):
            if pname in fits[name]["idata"].posterior:
                v = np.asarray(fits[name]["idata"].posterior[pname]).reshape(-1)
                eff.append({"model": name, "parameter": pname,
                            "posterior_mean": round(float(v.mean()), 4),
                            "eti94_lb": round(float(np.percentile(v, 3)), 4),
                            "eti94_ub": round(float(np.percentile(v, 97)), 4),
                            "prob_gt_zero": round(float((v > 0).mean()), 4)})
    pd.DataFrame(eff).to_csv(TABLE_DIR / "ablation_population_effects.csv",
                             index=False, encoding="utf-8-sig")
    print(out[["model", "elpd_loo", "p_loo", "d_elpd_loo_vs_main",
               "dse_vs_main", "divergences"]].to_string(index=False))
    return out


def _heldout_loglik(idata, arr, idx, tv=True, n_draws=1000):
    """Pointwise log predictive density on held-out trials."""
    mu = draw_matrix(idata, "mu")
    sg = draw_matrix(idata, "sigma")
    tb = draw_matrix(idata, "tau_base")
    pick = np.linspace(0, mu.shape[0] - 1, min(n_draws, mu.shape[0])).astype(int)
    c = arr["codes"][idx]
    mu_s, sg_s, tb_s = mu[pick][:, c], sg[pick][:, c], tb[pick][:, c]
    if tv:
        th = draw_matrix(idata, "theta")[pick][:, c]
        tau = np.maximum(tb_s + th * arr["t_norm"][idx], TAU_FLOOR)
    else:
        tau = tb_s
    return stats.exponnorm.logpdf(arr["y"][idx][None, :], tau / sg_s,
                                  loc=mu_s, scale=sg_s)


def run_kfold(arr: dict, k=10) -> pd.DataFrame:
    """Subject-stratified k-fold cross-validation of the TV advantage."""
    section(f"Step 3b: {k}-fold cross-validation (TV vs Static)")
    y, codes, t_norm = arr["y"], arr["codes"], arr["t_norm"]
    n = len(y)
    rng = np.random.default_rng(SEED)
    fold_id = np.empty(n, dtype=int)
    for subj in np.unique(codes):
        rows = np.where(codes == subj)[0]
        fold_id[rows] = rng.permutation(len(rows)) % k

    elpd_tv = np.full(n, np.nan)
    elpd_st = np.full(n, np.nan)
    fold_rows = []
    for f in range(k):
        test = np.where(fold_id == f)[0]
        train = np.where(fold_id != f)[0]
        sub = {"y": y[train], "codes": codes[train], "t_norm": t_norm[train],
               "subjects": arr["subjects"], "ybar": arr["ybar"], "sdy": arr["sdy"]}
        idata_tv, _ = sample_model(build_tv_model(sub), f"TV fold {f + 1}",
                                   with_loglik=False)
        idata_st, _ = sample_model(build_static_model(sub), f"Static fold {f + 1}",
                                   with_loglik=False)
        for idata_, store, tv_flag in ((idata_tv, elpd_tv, True),
                                       (idata_st, elpd_st, False)):
            ll = _heldout_loglik(idata_, arr, test, tv=tv_flag)
            store[test] = logsumexp(ll, axis=0) - np.log(ll.shape[0])
        fold_rows.append({"fold": f + 1, "n_test": len(test),
                          "elpd_test_tv": round(float(elpd_tv[test].sum()), 3),
                          "elpd_test_static": round(float(elpd_st[test].sum()), 3)})
        print(f"  fold {f + 1}/{k}: dELPD {elpd_tv[test].sum() - elpd_st[test].sum():+.2f}")

    d_i = elpd_tv - elpd_st
    d_elpd = float(np.nansum(d_i))
    pd.DataFrame(fold_rows).to_csv(TABLE_DIR / "kfold_cv.csv", index=False,
                                   encoding="utf-8-sig")
    out = pd.DataFrame([{
        "scheme": f"{k}-fold CV (subject-stratified)", "n_folds": k, "n_obs": n,
        "n_draws": DRAWS * CHAINS,
        "elpd_kfold_tv": round(float(np.nansum(elpd_tv)), 3),
        "elpd_kfold_static": round(float(np.nansum(elpd_st)), 3),
        "d_elpd_kfold": round(d_elpd, 3),
        "dse_kfold": round(float(np.sqrt(n * np.nanvar(d_i, ddof=1))), 3),
        "pct_folds_favoring_tv": round(100 * np.mean(
            [r["elpd_test_tv"] > r["elpd_test_static"] for r in fold_rows]), 1),
    }])
    out.to_csv(TABLE_DIR / "kfold_cv_summary.csv", index=False, encoding="utf-8-sig")
    print(f"  k-fold dELPD = {d_elpd:+.2f}")
    return out


def run_posterior_quadrants(idata, arr, quad_full) -> pd.DataFrame:
    """Posterior quadrant probabilities, replacing the LOSO agreement statistic."""
    section("Step 4: posterior quadrant classification probabilities")
    tau_d = draw_matrix(idata, "tau_base")
    th_d = draw_matrix(idata, "theta")
    n_subj = tau_d.shape[1]
    idx = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    probs = np.zeros((n_subj, 4))
    for s_i in range(tau_d.shape[0]):
        tm, thm = np.median(tau_d[s_i]), np.median(th_d[s_i])
        hi_t, hi_th = tau_d[s_i] > tm, th_d[s_i] > thm
        lab = np.where(hi_t & hi_th, "Q1", np.where(~hi_t & hi_th, "Q2",
                         np.where(hi_t & ~hi_th, "Q4", "Q3")))
        for k, q in enumerate(lab):
            probs[k, idx[q]] += 1
    probs /= tau_d.shape[0]

    modal = np.array([["Q1", "Q2", "Q3", "Q4"][i] for i in probs.argmax(axis=1)])
    out = pd.DataFrame({
        "subject": arr["subjects"],
        "quadrant_point": quad_full,
        "quadrant_modal": modal,
        "posterior_confidence": probs.max(axis=1),
        "posterior_top2_mass": np.sort(probs, axis=1)[:, -2:].sum(axis=1),
        "p_Q1": probs[:, 0], "p_Q2": probs[:, 1],
        "p_Q3": probs[:, 2], "p_Q4": probs[:, 3],
    })
    out.to_csv(TABLE_DIR / "posterior_quadrant_probability.csv", index=False,
               encoding="utf-8-sig")
    print(f"  mean posterior confidence = {out['posterior_confidence'].mean():.3f}")
    return out


def run_crossover(idata, arr, quad_full, thr=2.5, n_draws=1000):
    """Time at which Q2 overtakes Q4 in mean exceedance probability."""
    section(f"Step 5: Q2 vs Q4 crossover time at T_crit = {thr} s")
    mu = draw_matrix(idata, "mu")
    sg = draw_matrix(idata, "sigma")
    tb = draw_matrix(idata, "tau_base")
    th = draw_matrix(idata, "theta")
    pick = np.linspace(0, mu.shape[0] - 1, min(n_draws, mu.shape[0])).astype(int)
    m2 = np.where(quad_full == "Q2")[0]
    m4 = np.where(quad_full == "Q4")[0]
    if len(m2) == 0 or len(m4) == 0:
        print("  [skip] Q2 or Q4 empty")
        return None

    t_fine = np.linspace(0.0, 1.0, 201)
    stars, n_behind = [], 0
    for d in pick:
        tau = np.maximum(tb[d][None, :] + th[d][None, :] * t_fine[:, None], TAU_FLOOR)
        p = stats.exponnorm.sf(thr, tau / sg[d][None, :],
                               loc=mu[d][None, :], scale=sg[d][None, :])
        dlt = p[:, m2].mean(axis=1) - p[:, m4].mean(axis=1)
        if dlt[0] >= 0:
            continue
        n_behind += 1
        hit = np.where(dlt > 0)[0]
        if not hit.size:
            continue
        j = hit[0]
        if j == 0:
            stars.append(0.0)
        else:
            stars.append(float(t_fine[j - 1] + (t_fine[j] - t_fine[j - 1])
                               * (-dlt[j - 1]) / (dlt[j] - dlt[j - 1])))

    stars = np.asarray(stars, dtype=float)
    out = pd.DataFrame([{
        "comparison": "Q2 overtaking Q4, mean exceedance probability",
        "threshold_s": thr, "n_posterior_draws": int(len(pick)),
        "pct_draws_q2_starts_below": round(100.0 * n_behind / len(pick), 1),
        "pct_crossover_given_starts_below": round(100.0 * stars.size / max(n_behind, 1), 1),
        "median_crossover_min": round(float(np.median(stars) * 20.0), 2) if stars.size else np.nan,
        "eti94_lb_min": round(float(np.percentile(stars, 3) * 20.0), 2) if stars.size else np.nan,
        "eti94_ub_min": round(float(np.percentile(stars, 97) * 20.0), 2) if stars.size else np.nan,
    }])
    out.to_csv(TABLE_DIR / "crossover_time.csv", index=False, encoding="utf-8-sig")
    print(f"  median crossover = {out.loc[0, 'median_crossover_min']} min "
          f"(94% ETI {out.loc[0, 'eti94_lb_min']}–{out.loc[0, 'eti94_ub_min']})")
    return out


def run_ppc(idata, arr, n_draws: int = 800) -> pd.DataFrame:
    section("4b. Posterior predictive checks and quantile coverage")
    y, codes = arr["y"], arr["codes"]
    rng = np.random.default_rng(SEED)

    mu_d, sg_d = draw_matrix(idata, "mu"), draw_matrix(idata, "sigma")
    tb_d, th_d = draw_matrix(idata, "tau_base"), draw_matrix(idata, "theta")
    idx = np.linspace(0, mu_d.shape[0] - 1, n_draws).astype(int)
    mu_s, sg_s = mu_d[idx][:, codes], sg_d[idx][:, codes]
    tau_it = np.maximum(tb_d[idx][:, codes] + th_d[idx][:, codes] * arr["t_norm"],
                        TAU_FLOOR)
    y_rep = rng.normal(loc=mu_s, scale=sg_s) + rng.exponential(scale=tau_it)

    qs = [50, 95, 99]
    obs = np.percentile(y, qs)
    rep = np.percentile(y_rep, qs, axis=1)
    rows = []
    for k, q in enumerate(qs):
        lo, hi = np.percentile(rep[k], [3, 97])
        p_two = 2 * min((rep[k] <= obs[k]).mean(), (rep[k] >= obs[k]).mean())
        rows.append({
            "quantile": f"P{q}",
            "observed_s": round(float(obs[k]), 4),
            "ppc_median_s": round(float(np.median(rep[k])), 4),
            "ppc_ci94_lb": round(float(lo), 4),
            "ppc_ci94_ub": round(float(hi), 4),
            "covered_by_94pct_band": bool(lo <= obs[k] <= hi),
            "bayesian_p_two_sided": round(float(min(1.0, p_two)), 4),
            "n_post_draws": int(len(idx)),
        })
    ppc_df = pd.DataFrame(rows)
    ppc_df.to_csv(TABLE_DIR / "ppc_coverage.csv", index=False, encoding="utf-8")
    print(ppc_df.to_string(index=False))

    # Analytic posterior predictive density: mean over draws of the conditional
    # density, which avoids the sampling noise of a histogram in the tail.
    grid = np.linspace(0.0, 4.5, 181)
    n_band = min(300, mu_s.shape[0])
    bidx = np.linspace(0, mu_s.shape[0] - 1, n_band).astype(int)
    dens = np.empty((n_band, grid.size))
    for j, d in enumerate(bidx):
        k_d = (tau_it[d] / sg_s[d])[:, None]
        dens[j] = stats.exponnorm.pdf(grid[None, :], k_d,
                                      loc=mu_s[d][:, None],
                                      scale=sg_s[d][:, None]).mean(axis=0)
    pd.DataFrame({
        "rt_s": grid,
        "ppc_p3": np.percentile(dens, 3, axis=0),
        "ppc_p50": np.percentile(dens, 50, axis=0),
        "ppc_p97": np.percentile(dens, 97, axis=0),
    }).to_csv(TABLE_DIR / "ppc_density_band.csv", index=False, encoding="utf-8")
    print(f"  -> ppc_coverage.csv | ppc_density_band.csv")
    return ppc_df


def run_risk_curves(idata, arr, quad_full, n_draws: int = 1000):
    section("4c. Threshold-exceedance probability P(RT > T_crit) over the session")
    mu_d, sg_d = draw_matrix(idata, "mu"), draw_matrix(idata, "sigma")
    tb_d, th_d = draw_matrix(idata, "tau_base"), draw_matrix(idata, "theta")
    idx = np.linspace(0, mu_d.shape[0] - 1, n_draws).astype(int)
    mu_d, sg_d, tb_d, th_d = mu_d[idx], sg_d[idx], tb_d[idx], th_d[idx]
    n_subj = mu_d.shape[1]
    members = {q: np.where(quad_full == q)[0] for q in QUADRANTS}

    curve_rows, summary_rows = [], []
    for thr in THRESHOLDS:
        p_t0 = p_t1 = None
        for t in T_GRID:
            tau_it = np.maximum(tb_d + th_d * t, TAU_FLOOR)
            p = stats.exponnorm.sf(thr, tau_it / sg_d, loc=mu_d, scale=sg_d)
            if t == 0.0:
                p_t0 = p
            if t == 1.0:
                p_t1 = p
            for q in QUADRANTS:
                pq = p[:, members[q]].mean(axis=1)
                curve_rows.append({
                    "threshold_s": thr, "t_norm": round(float(t), 3),
                    "time_min": round(float(t) * 20.0, 2), "quadrant": q,
                    "n_subjects": int(len(members[q])),
                    "p_mean": float(pq.mean()),
                    "p_lo94": float(np.percentile(pq, 3)),
                    "p_hi94": float(np.percentile(pq, 97)),
                })
        for q in QUADRANTS:
            m = members[q]
            a0, a1 = p_t0[:, m].mean(axis=1), p_t1[:, m].mean(axis=1)
            summary_rows.append({
                "quadrant": q, "quadrant_label": QUADRANT_LABEL[q],
                "n_subjects": int(len(m)),
                "pct_subjects": round(100.0 * len(m) / n_subj, 1),
                "threshold_s": thr,
                "p_t0_mean": float(a0.mean()),
                "p_t0_lo94": float(np.percentile(a0, 3)),
                "p_t0_hi94": float(np.percentile(a0, 97)),
                "p_t1_mean": float(a1.mean()),
                "p_t1_lo94": float(np.percentile(a1, 3)),
                "p_t1_hi94": float(np.percentile(a1, 97)),
                "risk_amplification": float(a1.mean() / max(a0.mean(), 1e-12)),
                "delta_pp_t1_minus_t0": float(100 * (a1.mean() - a0.mean())),
            })

    pd.DataFrame(curve_rows).to_csv(TABLE_DIR / "risk_curves.csv",
                                    index=False, encoding="utf-8")
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(TABLE_DIR / "risk_escalation_summary.csv",
                index=False, encoding="utf-8")
    show = summ[summ["threshold_s"] == 2.5]
    print("  T_crit = 2.5 s, start (t=0) vs end (t=1) of the monitoring block:")
    print(show[["quadrant", "n_subjects", "p_t0_mean", "p_t1_mean",
                "risk_amplification"]].to_string(index=False))
    print("  -> risk_curves.csv | risk_escalation_summary.csv")
    return summ


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(only_sensitivity: bool = False) -> None:
    t_global = time.time()
    print("#" * 78)
    print("# HB-TV-ExG validation suite")
    print("#" * 78)

    if only_sensitivity:
        print("[mode] --only-cleaning-sensitivity: steps 1, 2 and 4 are skipped")
        run_cleaning_sensitivity()
    else:
        idata_tv, arr = run_model_comparison()
        run_ablation(arr)
        run_kfold(arr, k=10)
        run_cluster_bootstrap()
        run_cleaning_sensitivity()

        tau_base = draw_matrix(idata_tv, "tau_base").mean(axis=0)
        theta = draw_matrix(idata_tv, "theta").mean(axis=0)
        quad_full = classify_quadrants(tau_base, theta,
                                       float(np.median(tau_base)),
                                       float(np.median(theta)))
        run_posterior_quadrants(idata_tv, arr, quad_full)
        run_ppc(idata_tv, arr)
        run_risk_curves(idata_tv, arr, quad_full)
        run_crossover(idata_tv, arr, quad_full)

    print("\n" + "=" * 78)
    print(f"validation complete in {(time.time() - t_global) / 60:.1f} min; "
          f"tables -> {TABLE_DIR.relative_to(ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="HB-TV-ExG validation suite")
    ap.add_argument("--only-cleaning-sensitivity", action="store_true",
                    help="run only the four-strategy quality-control sensitivity")
    main(only_sensitivity=ap.parse_args().only_cleaning_sensitivity)
