# HB-TV-ExG: Hierarchical Bayesian Time-Varying Ex-Gaussian Modelling of Vigilance Decrement

Reference implementation and de-identified data for a longitudinal model of takeover
reaction time under automated driving. Response times in sustained monitoring tasks are
right-skewed with a heavy upper tail; the tail is the behaviourally relevant part, because
it determines whether an operator can resume control within the available budget. The
Ex-Gaussian density captures that tail with a single positive parameter `nu` (the mean and
standard deviation of the exponential component). HB-TV-ExG lets that parameter drift
within the session instead of assuming a stationary operator.

The repository contains the full analysis: quality control, a hand-written hierarchical
model sampled with NUTS, five validation layers (information criteria, a time-varying
location ablation, subject-stratified k-fold cross-validation, posterior quadrant
classification, and posterior predictive checks), and the scripts that regenerate every
table and figure.

## Model

For operator *i* and trial *t* (`t_norm = (t - 1) / 29` rescales trials 1-30 to the unit
interval):

$$
\tau_{i,t} = \operatorname{clip}\left(\tau^{base}_i + \theta_i \, t_{norm},\; 0.001,\; \infty\right)
$$

$$
rt_{i,t} \sim \mathrm{ExGaussian}\left(\mu_i,\; \sigma_i,\; \tau_{i,t}\right)
$$

`tau_base_i` is the static initial tail risk and `theta_i` is the individual fatigue slope
in seconds per session, so the two axes separate *how slow an operator starts* from *how
fast that operator deteriorates*. Both are drawn from a population distribution:

$$
\mu_i = \mu_{pop} + \sigma_\mu z^\mu_i, \qquad
\sigma_i = \exp\left(\log\sigma_{pop} + s_\sigma z^\sigma_i\right)
$$

$$
\tau^{base}_i = \tau_{pop}\exp\left(s_\tau z^\tau_i\right), \qquad
\theta_i = \theta_{pop} + s_\theta z^\theta_i
$$

All individual parameters are non-centred. `tau_base` is multiplicative, which keeps it
strictly positive; `theta` is left free in sign, so an operator may improve over the
session. Weakly informative priors are used throughout, and the model block in
`src/02_hb_tv_exg_model.py` is the single source of truth for their values.

Operators are split into four risk quadrants at the sample medians of `tau_base` and
`theta`: Q1 extreme high-risk, Q2 fatigue deteriorating, Q3 robust throughout, Q4 static
sluggish.

Three variants are used only for the location ablation in
`results/tables/ablation_time_varying_location.csv` and never for inference:
`HB-Static-ExG` (neither `mu` nor `tau` varies with time), `HB-TV-Mu-ExG`
(`mu_{i,t} = mu_base_i + delta_i * t_norm`, static `tau`) and `HB-TV-Both-ExG` (both
position and tail drift). They share the prior family of the main model, so the
comparison isolates the functional form rather than the regularisation.

## Repository layout

```text
.
├── run_all.py                     end-to-end driver (01 -> 02 -> 03 -> 04)
├── data/
│   ├── DATA_DICTIONARY.md         field definitions, units, sampling settings
│   ├── pseudonymized_raw_trials.csv   1,500 trials, 50 operators x 30
│   └── cleaned_trials_1329.csv        analysis set after quality control
├── src/
│   ├── __init__.py                 package marker and version
│   ├── 01_data_cleaning.py        3-sigma replacement + 3 x IQR removal, funnel
│   ├── 02_hb_tv_exg_model.py      model definition, NUTS sampling, PSIS-LOO, quadrants
│   ├── 03_model_validation.py     ablation, k-fold CV, bootstrap, sensitivity, PPC
│   ├── 04_generate_figures.py     Figure_1 - Figure_6 at 300 DPI, PNG and PDF
│   └── viz_utils.py               shared style, 300 DPI export, font-embedded PDF
└── results/
    ├── tables/                    pre-computed CSV tables (all identifiers S01-S50)
    └── figures/                   pre-computed Figure_1.png/.pdf - Figure_6.png/.pdf
```

## Environment

Python 3.12 with the pinned versions in `requirements.txt`; the sampling backend is
`nutpie` (Rust NUTS), so no C++ compiler is needed.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduction

```bash
python run_all.py                  # full pipeline, four stages, ~30 min on 8 cores
python run_all.py --steps 01,02    # subset
python run_all.py --list           # stage identifiers
```

Stages can also be run individually from the repository root, in order:

```bash
python src/01_data_cleaning.py         # data/cleaned_trials_1329.csv, attrition funnel
python src/02_hb_tv_exg_model.py       # posterior, PSIS-LOO, quadrant assignment
python src/03_model_validation.py      # ablation, k-fold CV, bootstrap, sensitivity, PPC
python src/04_generate_figures.py      # results/figures/Figure_1.png ... Figure_6.pdf
```

Every script writes only into `results/`, and `run_all.py` finishes by collecting the
headline quantities into `results/tables/reproduction_summary.csv`.

Sampling is seeded throughout (`seed = 2026`, `tune = 1000`, `draws = 2000`, **`chains = 4`**,
`target_accept = 0.92`), giving 8,000 posterior draws per fit. Four chains rather than two
make the split r-hat and the bulk ESS meaningful for the population scale parameters,
which is where a two-chain run can mask a poorly mixed tail. The pointwise log-likelihood
is retained (`idata_kwargs = {"log_likelihood": True}`) so that PSIS-LOO, its Pareto-k
shape diagnostic and the ablation all derive from the same draws. A rerun on the same data
reproduces the values below.

The pipeline is deterministic apart from floating-point reduction order. Reproducibility was
checked cell by cell against the primary analysis run: 17 of the 22 released tables
reproduce to the last stored digit, including the posterior summary, the per-trial
log-likelihood and Pareto-k table, the pointwise LOO comparison against the static model,
the quadrant assignment, the cluster bootstrap, the cleaning sensitivity, the posterior
quadrant probabilities, the posterior predictive band, the risk curves and the crossover
distribution. The tables that move are the ones that refit rather than reuse a posterior:
the two ablation variants and the 20 k-fold fits, which shift in the third significant
figure (Delta ELPD within 0.8 units, against standard errors of 6 to 11). NUTS trajectories
are chaotic, so a different thread schedule can flip a boundary divergence and nudge such a
fit; the movement is an order of magnitude below the uncertainty the comparison is reported
with. The k-fold partition is drawn from the same seed, so the fold assignment is fixed.

## Model comparison and ablations

Leave-one-out comparison against the static model is the inferential claim; the ablation
asks whether the *tail* is the right place to put the time variation. All four
specifications are sampled with the same 4 x 2,000 schedule and scored on identical
observations (`results/tables/ablation_time_varying_location.csv`).

| Specification | Time-varying component | ELPD_LOO | Delta vs main | dSE |
| --- | --- | --- | --- | --- |
| `HB-Static-ExG` | none | -785.0 | -45.1 | 10.5 |
| `HB-TV-Mu-ExG` | location only | -755.3 | -15.4 | 11.2 |
| `HB-TV-ExG` | tail only (main) | -739.9 | 0.0 | - |
| `HB-TV-Both-ExG` | location and tail | -722.0 | +17.9 | 6.1 |

Two readings follow. First, moving the time variation from the tail to the location costs
15 ELPD relative to the tail-only model, so the deterioration is not a shift of the bulk
of the distribution. Second, allowing both components to drift buys a further 18 ELPD,
with a positive location drift of `delta_pop` = 0.146 s per session (94% ETI 0.097-0.202)
and a reduced tail slope of `theta_pop` = 0.359 s (0.240-0.481). The main model is kept
for inference because it isolates the tail, which is the quantity a takeover budget is
written against; `HB-TV-Both-ExG` is reported as the better-fitting description rather
than hidden.

Because PSIS-LOO can be optimistic when the importance ratios are heavy-tailed, the
advantage is re-derived without it. A subject-stratified 10-fold cross-validation refits
both models on 90% of each operator's trials and scores the held-out 10% with the analytic
Ex-Gaussian density (`results/tables/kfold_cv_summary.csv`): Delta ELPD = +44.3 (SE 10.4),
with 9 of 10 folds favoring the time-varying model. The two estimates agree to within
0.8 ELPD, which is the intended check, since the k-fold figure does not depend on the
PSIS approximation at all.

## Pareto-k diagnostics

The per-trial shape parameter is written to
`results/tables/hb_tv_loo_pointwise.csv` (1,329 rows) and summarised in
`results/tables/hb_tv_loo_diagnostics.csv`. The maximum is k = 1.007; four trials (0.3%)
exceed the conventional 0.7 threshold and one (0.08%) exceeds 1.0. Those four are the
longest reaction times in the sample, where a single observation can dominate its own
leave-one-out posterior. Reloo on so few points would shift ELPD by less than its own
standard error, so the raw PSIS estimate is reported and the k-fold result stands as the
robustness check.

## Posterior quadrant classification

The point-estimate quadrant of an operator is a summary of two posterior means and
carries no uncertainty. Recomputing both medians inside every posterior draw
(`results/tables/posterior_quadrant_probability.csv`) gives, for each operator, the
probability of falling in each quadrant; the modal quadrant agrees with the point-estimate
label for 48 of 50 operators. The probabilities are deliberately reported as they are:
mean confidence in the modal quadrant is 0.489 and only 20 of 50 operators exceed 0.5, so
the quadrant is a coarse descriptive device at this sample size, not a diagnosis.

## Crossover time

The escalation question is when the fatigue-deteriorating group (Q2) overtakes the static
sluggish group (Q4) in the probability of exceeding a 2.5 s takeover budget. Evaluating
the model on a 201-point grid inside each of 1,000 posterior draws gives a distribution
rather than a point (`results/tables/crossover_time.csv`): 75.3% of draws start with Q2
below Q4, and 99.1% of those cross within the block, at a median of 3.36 min (94% ETI
0.17-11.06). The ratio of mean exceedance probability moves from 0.774 at the start of the
block to 1.513 at the end.

## Reproduced quantities

| Quantity | Value | Source table |
| --- | --- | --- |
| Analysed trials | 1,329 of 1,500 (88.6%) | `data_cleaning_funnel.csv` |
| Operators | 50 | `hb_tv_quadrant_assignment.csv` |
| Population fatigue slope `theta_pop` (94% ETI) | 0.478 s [0.369, 0.592] | `model_comparison.csv` |
| Quadrant counts Q1/Q2/Q3/Q4 | 13 / 12 / 13 / 12 | `hb_tv_quadrant_assignment.csv` |
| `tau_base` vs empirical P99 | Pearson r = 0.597, Spearman rho = 0.613 | `hb_tv_tau_base_vs_p99_correlation.csv` |
| Predictive gain over static model | dELPD_LOO = +45.1 (SE 10.5) | `model_comparison.csv` |
| 10-fold CV gain over static model | dELPD = +44.3 (SE 10.4), 9/10 folds | `kfold_cv_summary.csv` |
| Tail-only vs location-only ablation | -15.4 elpd for the location variant | `ablation_time_varying_location.csv` |
| ELPD_LOO of the main model | -739.9 (SE 35.6), p_loo = 89.2 | `hb_tv_loo_diagnostics.csv` |
| Pareto-k above 0.7 | 4 of 1,329 trials (0.3%), max 1.007 | `hb_tv_loo_diagnostics.csv` |
| Modal quadrant agreement | 48 of 50, mean confidence 0.489 | `posterior_quadrant_probability.csv` |
| Q2/Q4 crossover at 2.5 s | median 3.36 min (94% ETI 0.17-11.06) | `crossover_time.csv` |
| Ex-Gaussian vs runner-up density | dAIC = 102.17, 95% CI [63.51, 149.13] | `distribution_fit_cluster_bootstrap.csv` |
| Convergence | max r-hat = 1.008, min ESS = 1,266, 2 divergences | `hb_tv_convergence_diagnostics.csv` |

## Data

`data/pseudonymized_raw_trials.csv` holds 50 operators x 30 trials with identifiers
`S01`-`S50` assigned in ascending order of the original participant code. The identity
mapping is not part of this repository and is held by the data controller under the
approved ethics protocol. Only response times and trial order are released; no
demographic, biometric or session-video data are included. `data/DATA_DICTIONARY.md`
documents fields, units and the derivation of the analysis set.

The logger writes a hard ceiling of 6.0 s per trial rather than truncating at the analysis
window: 20 of the 1,500 records sit exactly at 6.000 s and 11 exceed it. The analysis set
uses a 5.0 s window, which is an analyst decision about which trials carry usable tail
information, not a protocol timeout. Cleaned trials therefore stop short of 4.0 s
(max 3.961 s) while the untrimmed raw maximum remains 8.252 s on the foot channel.

## Anonymous review mirror

The code is structured so that it runs unchanged from an anonymous mirror, for example
`https://anonymous.4open.science/r/HB-TV-ExG-Vigilance-Decrement-32AD/`. Every path is
resolved relative to the repository root (`Path(__file__).resolve().parents[1]`), no
script embeds an absolute path, an author name or an institution, and the data files are
the same `S01`-`S50` pseudonyms, so a mirror exposes nothing that this repository does not.

## License and citation

MIT, see `LICENSE`. Citation metadata is in `CITATION.cff`. If you use the model or the
released data, please cite the software record listed there.

Authors: Liyuan Gu, Mingyue Miao (corresponding author), Lihua Huang and Fangxia Zhao,
School of Management Engineering, Capital University of Economics and Business, Beijing,
China. Correspondence should be directed through the journal or the repository issue
tracker rather than by electronic mail.
