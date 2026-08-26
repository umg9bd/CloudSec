"""
Formal statistical validation of synthetic_cloudtrail.csv against
real_dataset_combined.csv -- replaces the original notebook's approach
of eyeballing a percentage-gap against an arbitrary tolerance band.

Two tests, chosen for what they actually compare:
  - Two-sample chi-square test of homogeneity (via a contingency table)
    for categorical fields: read_only, principal_type, mfa_authenticated,
    target_resource (null vs non-null), error_code (null vs non-null),
    event_source, event_name (bucketed to top-N + "other").
  - Two-sample Kolmogorov-Smirnov test for the one continuous quantity
    available without further feature engineering: hour-of-day.

A p-value alone is misleading here: real data has ~46,800 rows and
synthetic has ~7,600, so almost any real difference -- including ones too
small to matter -- will come back "statistically significant" just from
sample size. Every test therefore also reports an effect size (Cramer's V
for chi-square, the KS statistic itself for KS), and the PASS/FLAG verdict
is based on effect size, not the p-value. Effect-size thresholds (0.1) are
the conventional "negligible effect" cutoffs for Cramer's V / KS.

Usage:
    python validate_synthetic_vs_real.py
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

TOP_N_HIGH_CARDINALITY = 15
EFFECT_SIZE_THRESHOLD = 0.10


def cramers_v(contingency_table):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape) - 1
    if k == 0 or n == 0:
        return chi2, p, 0.0
    v = np.sqrt(chi2 / (n * k))
    return chi2, p, v


def chi_square_field(real, synthetic, name, top_n=None, as_null_flag=False):
    r = real.copy()
    s = synthetic.copy()

    if as_null_flag:
        r = r.isnull().map({True: "null", False: "present"})
        s = s.isnull().map({True: "null", False: "present"})
    else:
        r = r.fillna("__null__").astype(str)
        s = s.fillna("__null__").astype(str)

    if top_n:
        top_cats = r.value_counts().head(top_n).index
        r = r.where(r.isin(top_cats), "other")
        s = s.where(s.isin(top_cats), "other")

    categories = sorted(set(r.unique()) | set(s.unique()))
    real_counts = r.value_counts().reindex(categories, fill_value=0)
    syn_counts = s.value_counts().reindex(categories, fill_value=0)
    table = pd.DataFrame({"real": real_counts, "synthetic": syn_counts})

    chi2, p, v = cramers_v(table.values.T)
    verdict = "PASS" if v < EFFECT_SIZE_THRESHOLD else "FLAG"

    print(f"\n{name}  [{verdict}]")
    print(f"  chi2={chi2:.1f}  p={p:.4g}  Cramer's V={v:.4f}  (flag threshold: {EFFECT_SIZE_THRESHOLD})")
    props = pd.DataFrame({
        "real_%": (real_counts / real_counts.sum() * 100).round(2),
        "synthetic_%": (syn_counts / syn_counts.sum() * 100).round(2),
    })
    print(props.sort_values("real_%", ascending=False).head(8).to_string())
    return {"field": name, "test": "chi-square", "p_value": p, "effect_size": v, "verdict": verdict}


def ks_field(real, synthetic, name):
    r = real.dropna()
    s = synthetic.dropna()
    stat, p = ks_2samp(r, s)
    verdict = "PASS" if stat < EFFECT_SIZE_THRESHOLD else "FLAG"
    print(f"\n{name}  [{verdict}]")
    print(f"  KS statistic={stat:.4f}  p={p:.4g}  (flag threshold: {EFFECT_SIZE_THRESHOLD})")
    print(f"  real mean={r.mean():.2f}  synthetic mean={s.mean():.2f}")
    return {"field": name, "test": "KS", "p_value": p, "effect_size": stat, "verdict": verdict}


def main():
    print("Loading datasets...")
    real = pd.read_csv("real_dataset_combined.csv", low_memory=False)
    synthetic = pd.read_csv("synthetic_cloudtrail.csv", low_memory=False)
    real["timestamp"] = pd.to_datetime(real["timestamp"], format="ISO8601", utc=True)
    synthetic["timestamp"] = pd.to_datetime(synthetic["timestamp"], format="ISO8601", utc=True)

    print(f"Real: {len(real)} rows  |  Synthetic: {len(synthetic)} rows")
    print("=" * 70)
    print("CATEGORICAL FIELDS -- two-sample chi-square test of homogeneity")
    print("=" * 70)

    results = []
    results.append(chi_square_field(real["read_only"], synthetic["read_only"], "read_only"))
    results.append(chi_square_field(real["principal_type"], synthetic["principal_type"], "principal_type"))
    results.append(chi_square_field(real["mfa_authenticated"], synthetic["mfa_authenticated"], "mfa_authenticated"))
    results.append(chi_square_field(real["target_resource"], synthetic["target_resource"],
                                     "target_resource (null vs present)", as_null_flag=True))
    results.append(chi_square_field(real["error_code"], synthetic["error_code"],
                                     "error_code (null vs present)", as_null_flag=True))
    results.append(chi_square_field(real["event_source"], synthetic["event_source"],
                                     f"event_source (top {TOP_N_HIGH_CARDINALITY} + other)",
                                     top_n=TOP_N_HIGH_CARDINALITY))
    results.append(chi_square_field(real["event_name"], synthetic["event_name"],
                                     f"event_name (top {TOP_N_HIGH_CARDINALITY} + other)",
                                     top_n=TOP_N_HIGH_CARDINALITY))

    print("\n" + "=" * 70)
    print("CONTINUOUS FIELDS -- two-sample Kolmogorov-Smirnov test")
    print("=" * 70)
    results.append(ks_field(real["timestamp"].dt.hour, synthetic["timestamp"].dt.hour, "hour_of_day"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    n_flagged = (summary["verdict"] == "FLAG").sum()
    print(f"\n{n_flagged} of {len(summary)} fields flagged for a real effect-size gap (not just p-value noise).")


if __name__ == "__main__":
    main()
