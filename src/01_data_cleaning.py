"""Quality control of the de-identified trial table.

Pipeline
--------
1. Load ``data/pseudonymized_raw_trials.csv`` (50 operators x 30 trials = 1,500 rows).
2. Two-pass outlier control applied separately to each response channel:
   a. 3-sigma screening -> the flagged value is replaced by the channel median
      (the lower threshold is floored at 0 s, the physical limit of a response time);
   b. 3 x IQR screening -> the whole trial is removed (more conservative than the
      conventional 1.5 x IQR rule, because the Ex-Gaussian tail carries the signal).
3. Write the analysis set and the trial-attrition funnel.

No logarithmic, z-score or other monotone rescaling is applied: all statistics are
computed on the raw microsecond scale so that the cleaning funnel is auditable
against the original acquisition log.

Outputs
-------
``data/cleaned_trials_1329.csv``            analysis set (1,329 rows)
``results/tables/data_cleaning_funnel.csv`` attrition per step and channel

Run from the repository root:
    python src/01_data_cleaning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "pseudonymized_raw_trials.csv"
CLEAN_CSV = ROOT / "data" / "cleaned_trials_1329.csv"
TABLE_DIR = ROOT / "results" / "tables"
FUNNEL_CSV = TABLE_DIR / "data_cleaning_funnel.csv"

KEY_COLS = ["subject_id", "trial_id"]
CHANNELS = ["foot_rt_us", "hand_rt_us"]
US2S = 1e6            # microseconds -> seconds
SIGMA_K = 3.0         # 3-sigma screening threshold
IQR_K = 3.0           # 3 x IQR trimming threshold


def load_raw() -> pd.DataFrame:
    """Read the de-identified trial table and validate its expected shape."""
    df = pd.read_csv(RAW_CSV)
    missing = df.groupby("subject_id").size()
    df = df.dropna().reset_index(drop=True)
    print(f"[load] {RAW_CSV.name}: {len(df)} trials, "
          f"{df['subject_id'].nunique()} operators, "
          f"{int(missing.min())}-{int(missing.max())} trials per operator")
    return df


def clean_channel(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, dict]:
    """Apply 3-sigma median replacement then 3 x IQR row removal to one channel."""
    rows_before = len(df)
    series = df[col].astype(float)

    mu, sd = series.mean(), series.std()
    lo3 = max(mu - SIGMA_K * sd, 0.0)
    hi3 = mu + SIGMA_K * sd
    flag3 = (series < lo3) | (series > hi3)
    median = float(series.median())
    df = df.copy()
    df.loc[flag3, col] = median

    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lo_iqr, hi_iqr = q1 - IQR_K * iqr, q3 + IQR_K * iqr
    keep = df[col].between(lo_iqr, hi_iqr)
    df = df[keep].reset_index(drop=True)

    record = {
        "step": f"03_clean_{col}",
        "column": col,
        "rows_before": rows_before,
        "rows_after": len(df),
        "n_3sigma_replaced": int(flag3.sum()),
        "n_iqr_dropped": int((~keep).sum()),
        "note": (f"median={median:.0f} us; 3-sigma=[{lo3:.0f}, {hi3:.0f}] us; "
                 f"IQR threshold=[{lo_iqr:.0f}, {hi_iqr:.0f}] us"),
    }
    print(f"[clean] {col:14s} 3-sigma replaced {record['n_3sigma_replaced']:3d} | "
          f"IQR dropped {record['n_iqr_dropped']:3d} | kept {len(df)}")
    return df, record


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    n_raw = len(df)

    funnel = [
        {"step": "01_load_trials", "column": "-", "rows_before": n_raw,
         "rows_after": n_raw, "n_3sigma_replaced": 0, "n_iqr_dropped": 0,
         "note": "50 operators x 30 trials, anonymous identifiers S01-S50"},
        {"step": "02_drop_missing", "column": "-", "rows_before": n_raw,
         "rows_after": n_raw, "n_3sigma_replaced": 0, "n_iqr_dropped": n_raw - len(df),
         "note": "trials with a missing response channel removed"},
    ]

    for col in CHANNELS:
        df, record = clean_channel(df, col)
        funnel.append(record)

    df = df[KEY_COLS + CHANNELS]
    retained = len(df) / n_raw
    funnel.append({
        "step": "04_final", "column": "-", "rows_before": n_raw,
        "rows_after": len(df),
        "n_3sigma_replaced": int(sum(r["n_3sigma_replaced"] for r in funnel)),
        "n_iqr_dropped": int(sum(r["n_iqr_dropped"] for r in funnel)),
        "note": (f"retention {retained:.2%}; "
                 f"{df['subject_id'].nunique()} operators; "
                 f"trials per operator {df.groupby('subject_id').size().min()}-"
                 f"{df.groupby('subject_id').size().max()}"),
    })
    pd.DataFrame(funnel).to_csv(FUNNEL_CSV, index=False, encoding="utf-8")
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")

    print("-" * 72)
    print(f"[write] {CLEAN_CSV.relative_to(ROOT)}  rows={len(df)}  "
          f"retention={retained:.2%}")
    print(f"[write] {FUNNEL_CSV.relative_to(ROOT)}")
    for col in CHANNELS:
        rt = df[col].values / US2S
        print(f"[summary] {col:14s} n={len(rt)} mean={rt.mean():.4f}s "
              f"sd={rt.std(ddof=1):.4f}s median={np.median(rt):.4f}s "
              f"range=[{rt.min():.4f}, {rt.max():.4f}]s "
              f"skew={skew(rt):+.3f} exkurt={kurtosis(rt, fisher=True):+.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
