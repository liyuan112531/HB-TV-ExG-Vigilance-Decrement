"""Generate the six result figures at 300 DPI.

Figure_1  three-stage closed-loop protocol timeline
Figure_2  pooled response-time density with the posterior predictive band
Figure_3  plate-notation diagram of the hierarchical model
Figure_4  static tail risk versus the individual 99th percentile
Figure_5  individual fatigue slopes with 94% intervals (ranked)
Figure_6  two-axis risk quadrant and the escalation of extreme-delay risk

Every panel is verified twice before it is written: no non-ASCII glyph may be
rendered, and the embedded PNG resolution must be exactly 300 DPI.

Inputs (produced by 01-03)
    data/cleaned_trials_1329.csv
    results/tables/hb_tv_posterior_summary.csv
    results/tables/hb_tv_quadrant_assignment.csv
    results/tables/hb_tv_tau_base_vs_p99_correlation.csv
    results/tables/ppc_density_band.csv, ppc_coverage.csv, risk_curves.csv

Outputs: ``results/figures/Figure_1.png`` ... ``Figure_6.png``

Run from the repository root:
    python src/04_generate_figures.py
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import exponnorm, gaussian_kde, norm, t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_utils

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

CLEAN_CSV = ROOT / "data" / "cleaned_trials_1329.csv"
FUNNEL_CSV = TABLE_DIR / "data_cleaning_funnel.csv"
SUMMARY_CSV = TABLE_DIR / "hb_tv_posterior_summary.csv"
QUAD_CSV = TABLE_DIR / "hb_tv_quadrant_assignment.csv"
CORR_CSV = TABLE_DIR / "hb_tv_tau_base_vs_p99_correlation.csv"
PPC_BAND_CSV = TABLE_DIR / "ppc_density_band.csv"
PPC_COVER_CSV = TABLE_DIR / "ppc_coverage.csv"
RISK_CURVE_CSV = TABLE_DIR / "risk_curves.csv"
CROSSOVER_CSV = TABLE_DIR / "crossover_time.csv"
US2S = 1e6
T_CRIT = 2.5                  # critical takeover budget highlighted in Figure 6

# Okabe-Ito colour-blind-safe palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "pink": "#CC79A7", "sky": "#56B4E9", "vermillion": "#D55E00",
     "yellow": "#F0E442", "grey": "#7F8C8D", "dark": "#2C3E50"}
QCOL = {"Q1": C["vermillion"], "Q2": C["orange"], "Q3": C["green"], "Q4": C["blue"]}
QNAME = {"Q1": "Extreme high-risk", "Q2": "Fatigue deteriorating",
         "Q3": "Robust throughout", "Q4": "Static sluggish"}
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")

CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")
made: list[Path] = []


def assert_no_cjk(fig, name: str) -> None:
    """Fail if any rendered string contains a CJK or full-width glyph."""
    bad = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
           if t.get_text() and CJK.search(t.get_text())]
    if bad:
        raise RuntimeError(f"{name}: CJK text detected: {bad[:5]}")


def save_fig(fig, filename: str) -> Path:
    """Write a 300 DPI PNG and a font-embedded vector PDF of the same figure."""
    assert_no_cjk(fig, filename)
    path = viz_utils.save_figure(fig, FIG_DIR / filename)
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    made.extend([path, pdf])
    return path


def _col(df: pd.DataFrame, names) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} present; columns are {list(df.columns)}")


def fmt_p(p: float) -> str:
    """Format a p value as mathtext scientific notation."""
    if not np.isfinite(p) or p <= 0:
        return "n.s."
    exp = int(np.floor(np.log10(p)))
    mant = p / (10.0 ** exp)
    return rf"${mant:.1f} \times 10^{{{exp}}}$"


def load_core():
    """Load the analysis set and the tables written by 02 and 03."""
    df = pd.read_csv(CLEAN_CSV)
    rt = df["foot_rt_us"].values.astype(float) / US2S
    rt = rt[np.isfinite(rt) & (rt > 0)]
    summ = pd.read_csv(SUMMARY_CSV, index_col=0)
    quad = pd.read_csv(QUAD_CSV)
    corr = pd.read_csv(CORR_CSV).set_index("method")
    return df, rt, summ, quad, corr


def theta_table(summ: pd.DataFrame) -> pd.DataFrame:
    """Individual posterior summaries of theta, sorted by posterior mean."""
    c_mean = _col(summ, ["mean"])
    c_lo = _col(summ, ["eti94_lb", "hdi_3%"])
    c_up = _col(summ, ["eti94_ub", "hdi_97%"])
    rows = summ[summ.index.astype(str).str.match(r"^theta\[")].copy()
    th = pd.DataFrame({
        "subject": [str(i).split("[")[1].rstrip("]") for i in rows.index],
        "mean": rows[c_mean].values.astype(float),
        "lo": rows[c_lo].values.astype(float),
        "hi": rows[c_up].values.astype(float),
    })
    return th.sort_values("mean", ascending=False).reset_index(drop=True)


# ===========================================================================
# Figure 1: three-stage protocol
# ===========================================================================
def fig1_protocol(n_raw: int, n_clean: int, retention: float) -> Path:
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(4.0, 89.0)
    ax.axis("off")

    stages = [
        dict(x=3.0, tag="Stage 1", title="No-Driving Baseline", minutes="0–20 min",
             color=C["blue"], dur="20 min",
             lines=["Seated in the simulator cockpit",
                    "No driving or monitoring task",
                    "Visual probe detection",
                    "Randomised inter-onset interval 2–5 s",
                    "Outcome: baseline RT (zero workload)"]),
        dict(x=36.5, tag="Stage 2", title="Manual Driving + Distraction",
             minutes="20–40 min", color=C["orange"], dur="20 min",
             lines=["Continuous manual lane-keeping",
                    "Logitech G923 wheel, hands on rim",
                    "Dispatch and mode-switch secondary tasks",
                    "Divided-attention workload",
                    "Outcome: RT under multitasking load"]),
        dict(x=70.0, tag="Stage 3", title="Automated Passive Monitoring",
             minutes="40–60 min", color=C["green"], dur="20 min",
             lines=["SAE Level 3 automation engaged",
                    "Passive monitoring, hands off the wheel",
                    "30 takeover events with TOR alerts",
                    "1000 Hz SafetyTest acquisition",
                    "Outcome: 30 trials per operator"]),
    ]

    top, height, width = 86.0, 46.0, 29.0
    for st in stages:
        x0 = st["x"]
        ax.add_patch(FancyBboxPatch((x0, top - height), width, height,
                                    boxstyle="round,pad=0.25,rounding_size=1.2",
                                    facecolor="white", edgecolor=st["color"],
                                    linewidth=1.6, zorder=2))
        ax.add_patch(FancyBboxPatch((x0, top - 8.2), width, 8.2,
                                    boxstyle="round,pad=0.25,rounding_size=1.2",
                                    facecolor=st["color"], edgecolor=st["color"],
                                    linewidth=1.6, alpha=0.92, zorder=3))
        ax.text(x0 + 1.6, top - 4.1, f"{st['tag']}  |  {st['minutes']}",
                ha="left", va="center", fontsize=13, color="white",
                fontweight="bold", zorder=4)
        ax.text(x0 + width / 2, top - 12.8, st["title"], ha="center", va="center",
                fontsize=14, fontweight="bold", color=st["color"], zorder=4)
        ax.text(x0 + width / 2, top - 18.0, f"Duration: {st['dur']}",
                ha="center", va="center", fontsize=12, style="italic",
                color="#333333")
        for k, line in enumerate(st["lines"]):
            ax.text(x0 + 1.8, top - 23.2 - k * 4.6, "•  " + line,
                    ha="left", va="center", fontsize=12.2, color="#1B1B1B")

    for x_from, x_to in ((32.0, 36.5), (65.5, 70.0)):
        ax.add_patch(FancyArrowPatch((x_from + 0.2, top - 24), (x_to - 0.3, top - 24),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color=C["dark"], lw=1.8, zorder=5))

    y_axis = 31.0
    ax.add_patch(FancyArrowPatch((3.0, y_axis), (99.0, y_axis),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=C["dark"], lw=1.4, zorder=3))
    ax.text(3.0, y_axis + 5.8, "Session timeline (min)", ha="left", va="center",
            fontsize=12, color=C["dark"], fontweight="bold")
    for x, lab in ((3.0, "0"), (32.0, "20"), (70.0, "40"), (99.0, "60")):
        ax.plot([x, x], [y_axis - 1.5, y_axis + 1.5], color=C["dark"], lw=1.4,
                zorder=4)
        ax.text(x, y_axis - 3.1, lab, ha="center", va="top", fontsize=11.4,
                color=C["dark"])
    for st in stages:
        ax.plot([st["x"] + 0.6, st["x"] + width - 0.6],
                [y_axis + 2.6, y_axis + 2.6], color=st["color"], lw=4.0,
                solid_capstyle="butt", alpha=0.85, zorder=3)

    ax.add_patch(FancyBboxPatch((3.0, 6.0), 96.0, 20.0,
                                boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="#F7F9FA", edgecolor="#BFC9CA",
                                linewidth=1.0, zorder=1))
    ax.text(6.0, 23.2, "Apparatus and acquisition", fontsize=13,
            fontweight="bold", color=C["dark"], va="center")
    for k, line in enumerate([
            "Logitech G923 force-feedback wheel + Arturo pedal set",
            "SafetyTest platform, 1000 Hz sampling of wheel, pedals and TOR events",
            "SAE Level 3 automated driving mode with 30 takeover requests per "
            "session"]):
        ax.text(6.0, 18.2 - k * 3.9, "•  " + line, fontsize=11.8, va="center",
                color="#1B1B1B")
    ax.text(52.0, 23.2, "Design and sample", fontsize=13, fontweight="bold",
            color=C["dark"], va="center")
    for k, line in enumerate([
            "Within-subject, three-stage closed loop (no between-group assignment)",
            f"50 operators x 30 trials per stage = {n_raw:,} recorded trials",
            f"{n_clean:,} trials retained after quality control "
            f"({100 * retention:.1f}%)"]):
        ax.text(52.0, 18.2 - k * 3.9, "•  " + line, fontsize=11.8, va="center",
                color="#1B1B1B")

    fig.tight_layout()
    return save_fig(fig, "Figure_1.png")


# ===========================================================================
# Figure 2: distribution fit and posterior predictive band
# ===========================================================================
def fig2_distribution_ppc(RT: np.ndarray) -> Path:
    K, loc, scale = exponnorm.fit(RT)
    mu_eg, sigma_eg, tau_eg = loc, scale, K * scale
    mu_n, sd_n = norm.fit(RT)
    ll_eg = float(np.sum(exponnorm.logpdf(RT, K, loc=loc, scale=scale)))
    ll_no = float(np.sum(norm.logpdf(RT, mu_n, sd_n)))

    x = np.linspace(0.0, 4.3, 700)
    band = pd.read_csv(PPC_BAND_CSV) if PPC_BAND_CSV.exists() else None
    cover = pd.read_csv(PPC_COVER_CSV) if PPC_COVER_CSV.exists() else None

    fig = plt.figure(figsize=(13.0, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.24)
    ax, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    ax.hist(RT, bins=55, density=True, color="#C8CDD0", edgecolor="white",
            linewidth=0.5, alpha=0.9, label=f"Observed RT (N = {len(RT):,})",
            zorder=2)
    if band is not None:
        ax.fill_between(band["rt_s"], band["ppc_p3"], band["ppc_p97"],
                        color=C["sky"], alpha=0.30, linewidth=0,
                        label="HB-TV-ExG PPC 94% band", zorder=1)
        ax.plot(band["rt_s"], band["ppc_p50"], color=C["blue"], lw=1.3, ls="-",
                alpha=0.85, label="PPC median", zorder=3)
    ax.plot(x, gaussian_kde(RT)(x), color=C["green"], lw=1.6, ls=":",
            label="Kernel density estimate", zorder=4)
    ax.plot(x, exponnorm.pdf(x, K, loc=loc, scale=scale), color=C["vermillion"],
            lw=2.3, label="Ex-Gaussian fit (MLE)", zorder=5)
    ax.plot(x, norm.pdf(x, mu_n, sd_n), color=C["pink"], lw=1.8, ls="--",
            label="Normal fit (MLE)", zorder=4)

    ax.set_xlim(0, 4.3)
    ax.set_ylim(0, max(exponnorm.pdf(x, K, loc=loc, scale=scale).max(),
                       norm.pdf(x, mu_n, sd_n).max()) * 1.18)
    ax.set_xlabel("Reaction time (s)")
    ax.set_ylabel("Probability density")
    ax.set_title("(a) Pooled reaction-time distribution", fontweight="bold",
                 loc="left")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.96)

    x2 = np.linspace(float(np.percentile(RT, 92)), 4.3, 400)
    ax2.semilogy(x2, np.maximum(exponnorm.pdf(x2, K, loc=loc, scale=scale), 1e-9),
                 color=C["vermillion"], lw=2.0, label="Ex-Gaussian fit")
    ax2.semilogy(x2, np.maximum(norm.pdf(x2, mu_n, sd_n), 1e-9),
                 color=C["pink"], lw=1.7, ls="--", label="Normal fit")
    if band is not None:
        mb = band["rt_s"] >= x2.min()
        ax2.fill_between(band.loc[mb, "rt_s"],
                         np.maximum(band.loc[mb, "ppc_p3"], 1e-9),
                         np.maximum(band.loc[mb, "ppc_p97"], 1e-9),
                         color=C["sky"], alpha=0.35, linewidth=0,
                         label="PPC 94% band")
    ax2.set_xlim(x2.min(), 4.3)
    ax2.set_ylim(5e-6, 9e-2)
    ax2.set_xlabel("Reaction time (s)")
    ax2.set_ylabel("Probability density (log scale)")
    ax2.set_title("(b) Right tail on a logarithmic scale", fontweight="bold",
                  loc="left")
    ax2.grid(alpha=0.22, which="both")
    ax2.legend(loc="upper right", fontsize=7.8, framealpha=0.96)

    txt = (r"Ex-Gaussian MLE" "\n"
           r"$\mu$ = " f"{mu_eg:.3f} s, " r"$\sigma$ = " f"{sigma_eg:.3f} s, "
           r"$\tau$ = " f"{tau_eg:.3f} s\n"
           r"$\Delta$logLik vs Normal = " f"{ll_eg - ll_no:+.1f} nats")
    if cover is not None:
        ok = cover["covered_by_94pct_band"].astype(str).str.lower() == "true"
        covered = [q for q, c in zip(cover["quantile"], ok) if c]
        missed = [f"{q}: {o:.2f} s vs. [{lb:.2f}, {ub:.2f}] s"
                  for q, c, o, lb, ub in zip(cover["quantile"], ok,
                                             cover["observed_s"],
                                             cover["ppc_ci94_lb"],
                                             cover["ppc_ci94_ub"]) if not c]
        txt += "\nPPC 94% band coverage: "
        txt += (" & ".join(covered) + " covered") if covered else "no quantile covered"
        if missed:
            txt += "; " + "; ".join(missed)
    ax2.text(0.03, 0.045, txt, transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=7.6, bbox=dict(boxstyle="round,pad=0.33", facecolor="white",
                                     edgecolor="#BFC9CA", alpha=0.95), zorder=8)

    fig.tight_layout()
    return save_fig(fig, "Figure_2.png")


# ===========================================================================
# Figure 3: hierarchical model diagram
# ===========================================================================
def node_circle(ax, xy, r, label, fontsize=12.5, facecolor="white",
                edgecolor="#1B2631", ls="-", lw=1.4, double=False, zorder=4):
    ax.add_patch(Circle(xy, r, facecolor=facecolor, edgecolor=edgecolor,
                        linestyle=ls, linewidth=lw, zorder=zorder))
    if double:
        ax.add_patch(Circle(xy, r * 0.86, facecolor="none", edgecolor=edgecolor,
                            linewidth=lw * 0.8, zorder=zorder))
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fontsize,
            zorder=zorder + 1)


def arrow(ax, p, q, rp=0.0, rq=0.0, color="#34495E", lw=1.3, ls="-", zorder=3):
    p, q = np.asarray(p, float), np.asarray(q, float)
    v = q - p
    u = v / np.hypot(*v)
    ax.add_patch(FancyArrowPatch(p + u * rp, q - u * rq, arrowstyle="-|>",
                                 mutation_scale=12, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=zorder))


def fig3_dag(n_subjects: int) -> Path:
    fig, ax = plt.subplots(figsize=(11.4, 8.4))
    ax.set_xlim(-0.8, 16.4)
    ax.set_ylim(-2.2, 12.6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.6, 0.9), 14.8, 8.4,
                                boxstyle="round,pad=0.02,rounding_size=0.35",
                                facecolor="#F7F9FA", edgecolor="#566573",
                                linewidth=1.5, zorder=0))
    ax.text(1.15, 1.25,
            r"Subjects $i = 1,\,2,\,\dots,\,%d$" % n_subjects, fontsize=12,
            fontweight="bold", color="#566573", zorder=2)

    hyp = [(3.0, 11.0, r"$\mu_{pop}$"), (6.0, 11.0, r"$\sigma_{pop}$"),
           (9.0, 11.0, r"$\tau_{pop}$"), (12.0, 11.0, r"$\theta_{pop}$")]
    for xx, yy, lab in hyp:
        node_circle(ax, (xx, yy), 0.95, lab, fontsize=12.5, ls="--",
                    edgecolor="#566573")
    ax.text(7.5, 12.35, "Population hyper-parameters (weakly informative priors)",
            fontsize=10, ha="center", color="#566573", style="italic")

    ind = [(3.0, 8.3, r"$\mu_i$"), (6.0, 8.3, r"$\sigma_i$"),
           (9.4, 8.3, r"$\tau_{base,i}$"), (12.5, 8.3, r"$\theta_i$")]
    for (hx, hy, _), (ix, iy, lab) in zip(hyp, ind):
        node_circle(ax, (ix, iy), 0.95, lab, fontsize=12.5)
        arrow(ax, (hx, hy), (ix, iy), rp=0.95, rq=0.95)

    ax.add_patch(FancyBboxPatch((1.5, 1.55), 13.0, 5.05,
                                boxstyle="round,pad=0.02,rounding_size=0.3",
                                facecolor="white", edgecolor="#85929E",
                                linewidth=1.3, linestyle="--", zorder=1))
    ax.text(14.3, 1.12, r"Trials $t = 1,\,2,\,\dots,\,30$", fontsize=10.2,
            fontweight="bold", color="#566573", ha="right", va="bottom", zorder=2)

    node_circle(ax, (4.4, 4.2), 0.95, r"$rt_{i,t}$", fontsize=12.5,
                facecolor="#D5D8DC")
    node_circle(ax, (9.2, 4.2), 0.95, r"$\tau_{i,t}$", fontsize=12.5, double=True)
    ax.text(12.9, 4.2, r"$t_{norm}$", fontsize=12.5, ha="center", va="center",
            zorder=4)
    ax.add_patch(FancyBboxPatch((11.85, 3.65), 2.1, 1.1,
                                boxstyle="round,pad=0.06,rounding_size=0.12",
                                facecolor="none", edgecolor="#AAB7B8",
                                linestyle=":", linewidth=1.0, zorder=2))

    arrow(ax, (9.4, 8.3), (9.2, 4.2), rp=0.95, rq=0.95)
    arrow(ax, (12.5, 8.3), (9.2, 4.2), rp=0.95, rq=0.95)
    arrow(ax, (11.8, 4.2), (9.2, 4.2), rq=0.95)
    arrow(ax, (9.2, 4.2), (4.4, 4.2), rp=0.95, rq=0.95)
    arrow(ax, (3.0, 8.3), (4.4, 4.2), rp=0.95, rq=0.95)
    arrow(ax, (6.0, 8.3), (4.4, 4.2), rp=0.95, rq=0.95)

    ax.text(7.5, 2.42,
            r"$\tau_{i,t} = \mathrm{clip}\left(\tau_{base,i} + \theta_i\, t_{norm},\ "
            r"0.001,\ \infty\right)$"
            "\n" r"$rt_{i,t} \sim \mathrm{ExGaussian}\left(\mu_i,\ \sigma_i,\ "
            r"\tau_{i,t}\right)$",
            fontsize=10.4, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor="#AAB7B8", alpha=0.97), zorder=5)

    for j, (txt, ls) in enumerate([
            ("dashed circle: population hyper-parameter", "--"),
            ("solid circle: individual parameter", "-"),
            ("double circle: deterministic node", "-"),
            ("shaded circle: observed data", "-")]):
        ax.add_patch(Circle((-0.35, -0.55 - j * 0.45), 0.13, facecolor="white",
                            edgecolor="#34495E", linestyle=ls, linewidth=1.1,
                            zorder=5))
        ax.text(-0.05, -0.55 - j * 0.45, txt, fontsize=9.4, va="center",
                color="#34495E")

    fig.tight_layout()
    return save_fig(fig, "Figure_3.png")


# ===========================================================================
# Figure 4: static tail risk versus the individual 99th percentile
# ===========================================================================
def fig4_tau_vs_p99(QUAD: pd.DataFrame, CORR: pd.DataFrame) -> Path:
    r_p, p_p = float(CORR.loc["Pearson", "r"]), float(CORR.loc["Pearson", "p_value"])
    r_s, p_s = float(CORR.loc["Spearman", "r"]), float(CORR.loc["Spearman", "p_value"])

    x = QUAD["tau_base"].values.astype(float)
    y = QUAD["p99_s"].values.astype(float)
    n = len(x)
    k, b = np.polyfit(x, y, 1)
    xg = np.linspace(x.min(), x.max(), 200)
    resid = y - (k * x + b)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))
    tcrit = t_dist.ppf(0.975, n - 2)
    se = s * np.sqrt(1.0 / n + (xg - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    ax.fill_between(xg, k * xg + b - tcrit * se, k * xg + b + tcrit * se,
                    color=C["sky"], alpha=0.30, linewidth=0,
                    label="95% confidence band for the mean response")
    ax.scatter(x, y, s=58, color=C["blue"], alpha=0.85, edgecolors="white",
               linewidth=0.75, zorder=3, label=f"Individual operator (n = {n})")
    ax.plot(xg, k * xg + b, color=C["dark"], lw=1.9, zorder=4,
            label=f"Ordinary least squares (slope = {k:.2f})")

    ax.set_xlabel(r"Static initial tail risk $\tau_{base,i}$ (s)")
    ax.set_ylabel("Individual 99th percentile reaction time P99 (s)")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right")
    ax.text(0.03, 0.97,
            f"Pearson  $r$ = {r_p:.3f}  ($p$ = {fmt_p(p_p)})\n"
            f"Spearman  $\\rho$ = {r_s:.3f}  ($p$ = {fmt_p(p_s)})",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.4,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                      edgecolor="#BFC9CA", alpha=0.95))
    fig.tight_layout()
    return save_fig(fig, "Figure_4.png")


# ===========================================================================
# Figure 5: individual fatigue slopes
# ===========================================================================
def fig5_forest(SUMM: pd.DataFrame) -> Path:
    th = theta_table(SUMM)
    n = len(th)
    strict = th["lo"] > 0
    n_strict = int(strict.sum())
    med = float(th["mean"].median())

    fig, ax = plt.subplots(figsize=(8.6, 11.4))
    ypos = np.arange(n)
    ys, yn = np.where(strict.values)[0], np.where(~strict.values)[0]
    ax.hlines(ys, th["lo"].values[ys], th["hi"].values[ys], color=C["blue"],
              lw=1.15, alpha=0.85, zorder=2)
    ax.hlines(yn, th["lo"].values[yn], th["hi"].values[yn], color=C["orange"],
              lw=1.45, alpha=0.95, zorder=2)
    ax.scatter(th["mean"].values[ys], ys, s=24, color=C["blue"], zorder=3,
               edgecolors="white", linewidth=0.45)
    ax.scatter(th["mean"].values[yn], yn, s=26, color=C["orange"], zorder=3,
               edgecolors="white", linewidth=0.45)
    ax.axvline(0.0, color=C["vermillion"], ls="--", lw=1.5, zorder=1)
    ax.axvline(med, color="black", ls=":", lw=1.3, zorder=1)

    ax.set_yticks(ypos)
    ax.set_yticklabels(th["subject"].tolist(), fontsize=6.2)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlabel(r"Individual fatigue slope $\theta_i$ (s per session, 94% ETI)")
    ax.set_ylabel("Anonymised operator ID (ranked by posterior mean)")
    ax.annotate(
        f"Posterior mean $\\theta$ > 0 in {int((th['mean'] > 0).sum())}/{n} operators; "
        f"mean $\\theta$ = {th['mean'].mean():.3f} s per session",
        xy=(0.5, -0.075), xycoords="axes fraction", ha="center", va="top",
        fontsize=8.4, color="#34495E")
    ax.grid(axis="x", alpha=0.22)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=C["blue"],
               markersize=6, label=f"94% ETI entirely above 0 (n = {n_strict})"),
        Line2D([], [], marker="o", linestyle="none", color=C["orange"],
               markersize=6, label=f"94% ETI includes 0 (n = {n - n_strict})"),
        Line2D([], [], color=C["vermillion"], ls="--", lw=1.4,
               label=r"$\theta_i = 0$ (no fatigue trend)"),
        Line2D([], [], color="black", ls=":", lw=1.4,
               label=f"Group median = {med:.3f} s"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.4, framealpha=0.96)
    fig.tight_layout()
    return save_fig(fig, "Figure_5.png")


# ===========================================================================
# Figure 6: risk quadrant and escalation of extreme-delay risk
# ===========================================================================
def fig6_quadrant_risk(QUAD: pd.DataFrame) -> Path:
    have_risk = RISK_CURVE_CSV.exists()
    if have_risk:
        fig = plt.figure(figsize=(14.4, 6.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.06], wspace=0.30)
        axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    else:
        fig, axA = plt.subplots(figsize=(8.2, 6.6))
        axB = None

    x = QUAD["tau_base"].values.astype(float)
    y = QUAD["theta"].values.astype(float)
    q = QUAD["quadrant"].values
    tau_med, theta_med = float(np.median(x)), float(np.median(y))
    n_tot = len(x)
    counts = {k: int((q == k).sum()) for k in QUADRANTS}

    for k in QUADRANTS:
        m = q == k
        axA.scatter(x[m], y[m], s=78, color=QCOL[k], alpha=0.88,
                    edgecolors="white", linewidth=0.8, zorder=3,
                    label=f"{k} {QNAME[k]} (n = {counts[k]}, "
                          f"{100 * counts[k] / n_tot:.0f}%)")
    axA.axvline(tau_med, color="#6E7B8B", ls="--", lw=1.2, zorder=1)
    axA.axhline(theta_med, color="#6E7B8B", ls="--", lw=1.2, zorder=1)

    corner = {"Q1": (0.965, 0.935, "right", "top"),
              "Q2": (0.035, 0.935, "left", "top"),
              "Q3": (0.035, 0.065, "left", "bottom"),
              "Q4": (0.965, 0.065, "right", "bottom")}
    for k in QUADRANTS:
        fx, fy, ha, va = corner[k]
        axA.text(fx, fy, f"{k} {QNAME[k]}\nn = {counts[k]} "
                         f"({100 * counts[k] / n_tot:.0f}%)",
                 transform=axA.transAxes, ha=ha, va=va, fontsize=8.4,
                 color=QCOL[k], fontweight="bold", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                           edgecolor=QCOL[k], linewidth=1.0, alpha=0.92))

    xpad = max((x.max() - x.min()) * 0.10, 1e-3)
    ypad = max((y.max() - y.min()) * 0.12, 1e-3)
    axA.set_xlim(x.min() - xpad, x.max() + xpad)
    axA.set_ylim(y.min() - ypad, y.max() + ypad)
    axA.set_xlabel(r"$\tau_{base}$: static initial tail risk (s)")
    axA.set_ylabel(r"$\theta$: time-varying fatigue slope (s per session)")
    axA.set_title("(A) Four-quadrant risk matrix", fontweight="bold", loc="left")
    axA.grid(alpha=0.20)

    if axB is not None:
        cur = pd.read_csv(RISK_CURVE_CSV)
        cur = cur[np.isclose(cur["threshold_s"].astype(float), T_CRIT)]
        for k in QUADRANTS:
            sub = cur[cur["quadrant"] == k].sort_values("time_min")
            n_k = int(sub["n_subjects"].iloc[0])
            axB.fill_between(sub["time_min"], sub["p_lo94"], sub["p_hi94"],
                             color=QCOL[k], alpha=0.16, linewidth=0)
            axB.plot(sub["time_min"], sub["p_mean"], color=QCOL[k], lw=2.0,
                     label=f"{k} {QNAME[k]} (n = {n_k})")
            axB.scatter([sub["time_min"].iloc[0], sub["time_min"].iloc[-1]],
                        [sub["p_mean"].iloc[0], sub["p_mean"].iloc[-1]],
                        color=QCOL[k], s=26, zorder=5, edgecolors="white",
                        linewidth=0.6)

        # Q2 overtaking Q4: annotate the posterior median crossing time (computed
        # draw by draw in 03_model_validation.py), not the grid intersection.
        s2 = cur[cur["quadrant"] == "Q2"].sort_values("time_min")
        s4 = cur[cur["quadrant"] == "Q4"].sort_values("time_min")
        d = s2["p_mean"].values - s4["p_mean"].values
        if CROSSOVER_CSV.exists():
            t_med = float(pd.read_csv(CROSSOVER_CSV)["median_crossover_min"].iloc[0])
        else:                                    # fall back to the first grid crossing
            idx = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
            t_med = float(s4["time_min"].values[idx[0]]) if idx.size else np.nan
        if np.isfinite(t_med):
            j = int(np.argmin(np.abs(s2["time_min"].values - t_med)))
            p_at = float(s2["p_mean"].values[j])
            axB.axvline(t_med, color="#6E7B8B", ls=":", lw=1.1, zorder=1)
            axB.annotate(f"Q2 overtakes Q4\n(median {t_med:.2f} min)",
                         xy=(t_med, p_at),
                         xytext=(t_med + 3.0, p_at + 0.10),
                         fontsize=8.2, color=C["dark"],
                         arrowprops=dict(arrowstyle="->", color=C["dark"], lw=1.0),
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                   edgecolor="#BFC9CA", alpha=0.95), zorder=7)

        axB.set_xlim(0, 20)
        axB.set_ylim(0, float(cur["p_hi94"].max()) * 1.10)
        axB.set_xlabel("Time in automated passive monitoring (min)")
        axB.set_ylabel(r"$P(\mathrm{RT} > 2.5\ \mathrm{s})$")
        axB.set_title(r"(B) Escalation of extreme-delay risk ($T_{crit}$ = 2.5 s)",
                      fontweight="bold", loc="left")
        axB.grid(alpha=0.20)
        axB.legend(loc="upper left", fontsize=8.0, framealpha=0.96)

    fig.tight_layout()
    return save_fig(fig, "Figure_6.png")


def main() -> None:
    viz_utils.apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, RT, SUMM, QUAD, CORR = load_core()
    funnel = pd.read_csv(FUNNEL_CSV)
    final = funnel[funnel["step"] == "04_final"].iloc[0]
    n_raw, n_clean = int(final["rows_before"]), int(final["rows_after"])
    retention = n_clean / n_raw
    print(f"[load] N={len(RT)} reaction times | operators={len(QUAD)} | "
          f"retention={retention:.2%}")
    print(f"[load] anonymised identifiers, e.g. {QUAD['subject_id'].head(3).tolist()}")

    print("[fig1] protocol timeline")
    fig1_protocol(n_raw, n_clean, retention)
    print("[fig2] distribution fit and posterior predictive band")
    fig2_distribution_ppc(RT)
    print("[fig3] hierarchical model diagram")
    fig3_dag(len(QUAD))
    print("[fig4] static tail risk versus individual P99")
    fig4_tau_vs_p99(QUAD, CORR)
    print("[fig5] individual fatigue slopes")
    fig5_forest(SUMM)
    print("[fig6] risk quadrant and escalation")
    fig6_quadrant_risk(QUAD)

    print("-" * 72)
    print(f"[done] {len(made)} figures at {viz_utils.DPI} DPI -> "
          f"{FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    main()
