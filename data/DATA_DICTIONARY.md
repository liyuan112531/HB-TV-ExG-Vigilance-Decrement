# Data dictionary

Two tables are released. Both use anonymous operator identifiers `S01`-`S50`; the
identity mapping is held separately by the data controller and is not part of this
repository. Response times are stored in microseconds exactly as acquired, so every
statistic can be recomputed without a rescaling step.

## `data/pseudonymized_raw_trials.csv`

1,500 rows = 50 operators x 30 trials, one row per takeover trial of the automated
passive-monitoring block. Row order follows acquisition order; trials are grouped by
operator and ordered 1-30 within an operator.

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `subject_id` | string | - | Anonymous operator identifier, `S01`-`S50` |
| `trial_id` | integer | - | Trial position within the session, 1-30 |
| `foot_rt_us` | float | microseconds | Foot-lift takeover latency (primary modelling channel) |
| `hand_rt_us` | float | microseconds | Hand-response latency to the takeover request |

No missing values are present. The two channels are simultaneous recordings of the same
takeover event; `foot_rt_us` is the channel modelled in all reported results, while
`hand_rt_us` is used as a secondary channel in the quality-control step.

## `data/cleaned_trials_1329.csv`

1,329 rows retained after quality control (88.6% of the raw table). Same columns and
units as above. Derived by `src/01_data_cleaning.py`:

1. 3-sigma screening per channel: values outside `mean +/- 3 sd` are replaced by the
   channel median (the lower threshold is floored at 0 s).
2. 3 x IQR trimming per channel: trials outside `[Q1 - 3 IQR, Q3 + 3 IQR]` are removed.

The stricter 3 x IQR rule (rather than the conventional 1.5 x IQR) is deliberate: mild
positive skew is a genuine property of the Ex-Gaussian tail, not an artefact, and trimming
at 1.5 x IQR would remove the signal the model is designed to estimate. Attrition per
step, including the replaced and removed counts, is written to
`results/tables/data_cleaning_funnel.csv`.

## Acquisition ceiling

The event logger closes a takeover response at 6.0 s. In the released raw table 20 of the
1,500 records sit exactly at 6.000 s (six on the foot channel) and 11 exceed it, up to
8.252 s on the foot channel. The ceiling is a property of the acquisition software, not of
the task: the protocol spaces the 30 requests of the monitored block about 40 s apart, so a
6 s latency is a genuine slow response rather than the concatenation of consecutive trials.

A 5.0 s window is used in the cleaning-sensitivity analysis (strategy A2) as an analyst
ceiling deliberately below the acquisition truncation. It is not a protocol timeout, and it
is not applied to the main analysis set, whose maximum retained foot latency is 3.961 s.

## Sampling and prior settings

| Item | Value |
| --- | --- |
| Sampler | `nutpie` NUTS (PyMC fallback if unavailable) |
| Chains / tune / draws | 4 / 1,000 / 2,000 (8,000 posterior draws) |
| `target_accept` | 0.92 |
| Seed | 2026 |
| `tau_i,t` lower bound | 0.001 s |
| `mu_pop` | Normal(mean(rt), 1) |
| `sigma_mu` | HalfNormal(0.3) |
| `log_sigma_pop` | Normal(log(sd(rt)) - 1, 0.5) |
| `s_sigma` | HalfNormal(0.4) |
| `tau_pop` | HalfNormal(0.5); HalfNormal(1.0) for the untrimmed-tail strategy A2 |
| `s_tau` | HalfNormal(0.5) |
| `theta_pop` | Normal(0, 0.3) |
| `s_theta` | HalfNormal(0.3) |
| `delta_pop` (ablation only) | Normal(0, 0.3) |
| `s_delta` (ablation only) | HalfNormal(0.3) |

Interval estimates are equal-tailed 94% intervals, matching the credible-interval width
used for the reported figures.

## Result tables

| File | Content |
| --- | --- |
| `data_cleaning_funnel.csv` | attrition per quality-control step and channel |
| `hb_tv_posterior_summary.csv` | posterior summary of population and operator parameters |
| `hb_tv_convergence_diagnostics.csv` | r-hat, ESS, divergences, runtime |
| `hb_tv_tau_base_vs_p99_correlation.csv` | agreement between `tau_base` and empirical P99 |
| `hb_tv_quadrant_assignment.csv` | per-operator `tau_base`, `theta`, P99 and quadrant |
| `model_comparison.csv` | PSIS-LOO and WAIC, time-varying vs static model |
| `distribution_fit_full_sample.csv` | five response-time densities, AIC and BIC |
| `distribution_fit_cluster_bootstrap.csv` | operator-cluster bootstrap of dAIC and dBIC |
| `cleaning_sensitivity.csv` | refit under strategies A / A2 / B / C |
| `hb_tv_loo_diagnostics.csv` | ELPD_LOO, p_loo, Pareto-k shape diagnostics |
| `hb_tv_loo_pointwise.csv` | per-trial `elpd_loo_i` and `pareto_k` |
| `ablation_time_varying_location.csv` | four time-variation specifications, LOO and WAIC |
| `ablation_population_effects.csv` | `delta_pop` and `theta_pop` under each ablation fit |
| `kfold_cv.csv`, `kfold_cv_summary.csv` | subject-stratified 10-fold cross-validation |
| `posterior_quadrant_probability.csv` | per-operator quadrant probabilities |
| `crossover_time.csv` | posterior distribution of the Q2/Q4 crossover time |
| `ppc_coverage.csv`, `ppc_density_band.csv` | posterior predictive coverage and density band |
| `risk_curves.csv`, `risk_escalation_summary.csv` | P(RT > T_crit) over the monitoring block |
| `reproduction_summary.csv` | headline quantities collected by `run_all.py` |
