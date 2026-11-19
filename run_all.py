"""End-to-end reproduction driver.

Runs the four stages in order, each as an isolated process so that a failure
stops the pipeline with a non-zero exit code:

    01_data_cleaning.py       quality control and trial-attrition funnel
    02_hb_tv_exg_model.py     posterior sampling of HB-TV-ExG, PSIS-LOO diagnostics
    03_model_validation.py    ablation, k-fold CV, bootstrap, sensitivity, PPC
    04_generate_figures.py    six 300 DPI figures (PNG + vector PDF)

After the stages complete, the headline quantities are collected into
``results/tables/reproduction_summary.csv`` and printed as a checksum table.

Usage
-----
    python run_all.py                 full pipeline (four stages, ~40 min on 8 cores)
    python run_all.py --steps 01,02   run a subset
    python run_all.py --list          show the stage identifiers
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TABLES = ROOT / "results" / "tables"

STAGES = [
    ("01", "01_data_cleaning.py", "quality control and attrition funnel"),
    ("02", "02_hb_tv_exg_model.py", "posterior sampling and PSIS-LOO diagnostics"),
    ("03", "03_model_validation.py", "ablation, k-fold CV, bootstrap, sensitivity, PPC"),
    ("04", "04_generate_figures.py", "six figures at 300 DPI (PNG + PDF)"),
]


def run_stage(script: str, label: str) -> None:
    print("\n" + "#" * 78)
    print(f"# {script}  --  {label}")
    print("#" * 78, flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"stage {script} failed with exit code {proc.returncode} "
                         f"after {time.time() - t0:.1f}s")
    print(f"# {script} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def _read(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else None


def summarise() -> pd.DataFrame:
    """Collect the headline reproduction quantities into one flat table."""
    rows: list[dict] = []

    def add(quantity, value, unit, source):
        rows.append({"quantity": quantity, "value": value, "unit": unit,
                     "source_table": source})

    funnel = _read("data_cleaning_funnel.csv")
    if funnel is not None:
        final = funnel[funnel["step"] == "04_final"].iloc[0]
        add("trials_analysed", int(final["rows_after"]), "trials",
            "data_cleaning_funnel.csv")
        add("retention", round(100 * final["rows_after"] / final["rows_before"], 2),
            "%", "data_cleaning_funnel.csv")

    conv = _read("hb_tv_convergence_diagnostics.csv")
    if conv is not None:
        r = conv.iloc[0]
        add("max_r_hat", round(float(r["max_r_hat"]), 4), "-",
            "hb_tv_convergence_diagnostics.csv")
        add("divergences", int(r["divergences"]), "draws",
            "hb_tv_convergence_diagnostics.csv")
        add("theta_pop_mean", round(float(r["theta_pop_mean"]), 4), "s per session",
            "hb_tv_convergence_diagnostics.csv")

    corr = _read("hb_tv_tau_base_vs_p99_correlation.csv")
    if corr is not None:
        c = corr.set_index("method")
        add("pearson_tau_base_vs_p99", round(float(c.loc["Pearson", "r"]), 4), "r",
            "hb_tv_tau_base_vs_p99_correlation.csv")
        add("spearman_tau_base_vs_p99", round(float(c.loc["Spearman", "r"]), 4), "rho",
            "hb_tv_tau_base_vs_p99_correlation.csv")

    quad = _read("hb_tv_quadrant_assignment.csv")
    if quad is not None:
        counts = quad["quadrant"].value_counts()
        add("quadrant_counts",
            "/".join(str(int(counts.get(q, 0))) for q in ("Q1", "Q2", "Q3", "Q4")),
            "operators (Q1/Q2/Q3/Q4)", "hb_tv_quadrant_assignment.csv")

    comp = _read("model_comparison.csv")
    if comp is not None:
        diff = comp[comp["model"].str.contains("vs")].iloc[0]
        add("delta_elpd_loo", diff["note"].split(",")[0].split("=")[1], "elpd",
            "model_comparison.csv")

    boot = _read("distribution_fit_cluster_bootstrap.csv")
    if boot is not None:
        aic = boot[boot["statistic"].str.startswith("delta_AIC")].iloc[0]
        add("delta_aic_runner_up_minus_exgaussian", aic["point_estimate_full_sample"],
            "AIC", "distribution_fit_cluster_bootstrap.csv")

    loo = _read("hb_tv_loo_diagnostics.csv")
    if loo is not None:
        r = loo.iloc[0]
        add("elpd_loo", r["elpd_loo"], "elpd", "hb_tv_loo_diagnostics.csv")
        add("pareto_k_max", r["pareto_k_max"], "k", "hb_tv_loo_diagnostics.csv")
        add("pareto_k_gt_0.7", f"{int(r['n_pareto_k_gt_0.7'])} "
            f"({r['pct_pareto_k_gt_0.7']}%)", "trials",
            "hb_tv_loo_diagnostics.csv")

    abl = _read("ablation_time_varying_location.csv")
    if abl is not None:
        a = abl.set_index("model")
        add("ablation_elpd_static", a.loc["HB-Static-ExG", "elpd_loo"], "elpd",
            "ablation_time_varying_location.csv")
        add("ablation_delta_static_vs_main",
            a.loc["HB-Static-ExG", "d_elpd_loo_vs_main"], "elpd",
            "ablation_time_varying_location.csv")
        add("ablation_delta_both_vs_main",
            a.loc["HB-TV-Both-ExG", "d_elpd_loo_vs_main"], "elpd",
            "ablation_time_varying_location.csv")

    kf = _read("kfold_cv_summary.csv")
    if kf is not None:
        r = kf.iloc[0]
        add("kfold_delta_elpd", r["d_elpd_kfold"], "elpd",
            "kfold_cv_summary.csv")
        add("kfold_delta_elpd_se", r["dse_kfold"], "elpd",
            "kfold_cv_summary.csv")
        add("kfold_pct_folds_favoring_tv", r["pct_folds_favoring_tv"], "%",
            "kfold_cv_summary.csv")

    pq = _read("posterior_quadrant_probability.csv")
    if pq is not None:
        agree = int((pq["quadrant_point"] == pq["quadrant_modal"]).sum())
        add("quadrant_modal_agreement", f"{agree} of {len(pq)}", "operators",
            "posterior_quadrant_probability.csv")
        add("mean_posterior_confidence",
            round(float(pq["posterior_confidence"].mean()), 3), "probability",
            "posterior_quadrant_probability.csv")

    xo = _read("crossover_time.csv")
    if xo is not None:
        r = xo.iloc[0]
        add("q2_q4_crossover_median", r["median_crossover_min"], "min",
            "crossover_time.csv")
        add("q2_q4_crossover_eti94",
            f"{r['eti94_lb_min']}-{r['eti94_ub_min']}", "min", "crossover_time.csv")

    sens = _read("cleaning_sensitivity.csv")
    if sens is not None:
        add("theta_pop_min_across_strategies",
            round(float(sens["theta_pop_mean"].min()), 4), "s per session",
            "cleaning_sensitivity.csv")
        add("theta_pop_max_across_strategies",
            round(float(sens["theta_pop_mean"].max()), 4), "s per session",
            "cleaning_sensitivity.csv")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "reproduction_summary.csv", index=False, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="HB-TV-ExG end-to-end reproduction")
    ap.add_argument("--steps", default="01,02,03,04",
                    help="comma separated stage identifiers (default 01,02,03,04)")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    args = ap.parse_args()

    if args.list:
        for sid, script, label in STAGES:
            print(f"  {sid}  {script:26s} {label}")
        return

    wanted = {s.strip() for s in args.steps.split(",") if s.strip()}
    unknown = wanted - {s[0] for s in STAGES}
    if unknown:
        raise SystemExit(f"unknown stage identifier(s): {sorted(unknown)}")

    t_start = time.time()
    print(f"HB-TV-ExG reproduction | python {sys.version.split()[0]} | "
          f"root {ROOT}")
    for sid, script, label in STAGES:
        if sid in wanted:
            run_stage(script, label)

    if "04" in wanted:
        summary = summarise()
        print("\n" + "=" * 78)
        print("Reproduction checksum")
        print("=" * 78)
        print(summary.to_string(index=False))
        print(f"\n-> {TABLES / 'reproduction_summary.csv'}")
    print(f"\ntotal wall time {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
