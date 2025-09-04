import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# ================== LOAD DATA ==================
rep = pd.read_csv("../data/nudge-replication.csv")      # short-run replication results
sim = pd.read_csv("../data/longitudinal-simulation.csv")         # day2-30 simulations under cycles

# ================== PREP & MERGE META ==================
# Define GPT-3.5 short-run replication success (d>0 & p<.05)
rep["gpt35_llm_cohens_d"] = pd.to_numeric(rep["gpt35_llm_cohens_d"], errors="coerce")
rep["gpt35_llm_p_value"] = pd.to_numeric(rep["gpt35_llm_p_value"], errors="coerce")
rep["short_success"] = (rep["gpt35_llm_cohens_d"] > 0) & (rep["gpt35_llm_p_value"] < 0.05)

meta_cols = ["study_id", "intervention_category", "intervention_technique", "type_experiment", "n_comparison"]
sim = sim.merge(rep[meta_cols + ["short_success"]], on="study_id", how="left")

# Identify effect-size / p-value columns for days 2..30
d_cols = sorted([c for c in sim.columns if c.startswith("d_round_")], key=lambda x: int(x.split("_")[-1]))
p_cols = sorted([c for c in sim.columns if c.startswith("p_round_")], key=lambda x: int(x.split("_")[-1]))
days = np.array([int(c.split("_")[-1]) for c in d_cols])  # [2..30]

# ================== (A) SHORT-RUN vs LONG-RUN (Day30, single) ==================
single = sim[sim["cycle"] == "single nudge"].copy()

long_d30 = pd.to_numeric(single["d_round_30"], errors="coerce")
short_d = rep.set_index("study_id").loc[single["study_id"], "gpt35_llm_cohens_d"].reset_index(drop=True)

mask_pair = long_d30.notna() & short_d.notna()
N_pair = int(mask_pair.sum())

t_A, p_A = stats.ttest_rel(short_d[mask_pair], long_d30[mask_pair], nan_policy="omit")
A_summary = pd.DataFrame({
    "N paired": [N_pair],
    "Short-run mean d (GPT-3.5)": [short_d[mask_pair].mean()],
    "Short-run SD d": [short_d[mask_pair].std(ddof=1)],
    "Long-run day30 mean d (single)": [long_d30[mask_pair].mean()],
    "Long-run day30 SD d": [long_d30[mask_pair].std(ddof=1)],
    "t (paired)": [t_A],
    "p (paired)": [p_A]
}).round(4)

print("=== (A) Short-run vs Long-run (Day 30, single) ===")
print(A_summary.to_string(index=False))

# ================== (B) END-OF-WINDOW FAILURE AMONG SHORT-RUN SUCCESSES ==================
# Fail-by-end (single): Day30 p>=.05 OR d<=0
day30_p = pd.to_numeric(single["p_round_30"], errors="coerce")
day30_d = pd.to_numeric(single["d_round_30"], errors="coerce")
single["fail_end"] = (day30_p >= 0.05) | (day30_d <= 0)

succ_single = single[single["short_success"].fillna(False)].copy()
n_succ_total = int(succ_single.shape[0])
n_fail_end = int(succ_single["fail_end"].sum())
fail_rate_overall = round(100 * n_fail_end / n_succ_total, 1) if n_succ_total > 0 else np.nan

B_overview = pd.DataFrame({
    "Short-run successes (n)": [n_succ_total],
    "Fail-by-end (n)": [n_fail_end],
    "Fail-by-end rate (%)": [fail_rate_overall]
})

print("\n=== (B) End-of-window failure overview (among short-run successes) ===")
print(B_overview.to_string(index=False))

# Failure by intervention_category (within short-run successes)
fail_by_cat = (succ_single.groupby("intervention_category")["fail_end"]
               .agg(fail_n="sum", succ_total="count"))
fail_by_cat["fail_rate_%"] = (fail_by_cat["fail_n"] / fail_by_cat["succ_total"] * 100).round(1)

print("\n=== (B-1) Failure by intervention category ===")
print(fail_by_cat.reset_index().to_string(index=False))

# ================== (C) EXPONENTIAL DECAY FITS ON FAILING CASES BY CATEGORY ==================
# Select failing IDs among short-run successes
fail_ids = succ_single.loc[succ_single["fail_end"], "study_id"].tolist()

def exp_decay(t, A, y0, k):
    # y(t) = A + (y0 - A) * exp(-k t), with t=0 at day2
    return A + (y0 - A) * np.exp(-k * t)

fit_rows = []
for cat, gdf in single[single["study_id"].isin(fail_ids)].groupby("intervention_category"):
    # Mean trajectory (days 2..30) across failing studies of this category
    Y = gdf[d_cols].astype(float).mean(axis=0).values
    t = np.array([d - 2 for d in days], dtype=float)   # t=0 at Day 2

    # Initial guesses
    y0_guess = float(np.nanmean(gdf["d_round_2"]))
    A_guess = float(np.nanmean(gdf[[c for c in d_cols if int(c.split("_")[-1]) >= 25]].mean(axis=1)))
    k_guess = 0.2

    # Drop NaNs
    mask = ~np.isnan(Y)
    t_fit, Y_fit = t[mask], Y[mask]

    # Bounds and fit
    try:
        popt, pcov = curve_fit(exp_decay, t_fit, Y_fit,
                               p0=[A_guess, y0_guess, k_guess],
                               bounds=([-5, 0, 0], [5, 5, 5]),
                               maxfev=10000)
        A_hat, y0_hat, k_hat = popt
        y_pred = exp_decay(t_fit, *popt)
        ss_res = np.sum((Y_fit - y_pred)**2)
        ss_tot = np.sum((Y_fit - np.mean(Y_fit))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    except Exception as e:
        A_hat, y0_hat, k_hat, r2 = np.nan, np.nan, np.nan, np.nan

    fit_rows.append({"intervention_category": cat,
                     "y0_hat": y0_hat, "A_hat": A_hat, "k_hat": k_hat, "R2": r2})

decay_fits = pd.DataFrame(fit_rows).round(4)

print("\n=== (C) Exponential decay fits for failing cases (by category) ===")
print(decay_fits.to_string(index=False))

# ================== (D) REPETITION (3×/5×) VS SINGLE ON FAILING SET ==================
rep3 = sim[sim["cycle"] == "3 repeated nudge"].copy()
rep5 = sim[sim["cycle"] == "5 repeated nudge"].copy()
single_fail = single[single["study_id"].isin(fail_ids)].copy()

# Align study intersection for paired comparisons
ids_common_3 = sorted(set(single_fail["study_id"]).intersection(set(rep3["study_id"])))
ids_common_5 = sorted(set(single_fail["study_id"]).intersection(set(rep5["study_id"])))

single_3 = single_fail[single_fail["study_id"].isin(ids_common_3)].set_index("study_id")
rep3_m   = rep3[rep3["study_id"].isin(ids_common_3)].set_index("study_id")
single_5 = single_fail[single_fail["study_id"].isin(ids_common_5)].set_index("study_id")
rep5_m   = rep5[rep5["study_id"].isin(ids_common_5)].set_index("study_id")

# ---- 修改点：D 的第一项对比为 Day30 效应量提升（重复 vs 单次），而非 MDE ----
# Pairwise t-tests on Day30 d between repeated vs single
d30_single_3 = pd.to_numeric(single_3["d_round_30"], errors="coerce")
d30_rep3     = pd.to_numeric(rep3_m["d_round_30"], errors="coerce")
mask3 = d30_single_3.notna() & d30_rep3.notna()
t_d30_3, p_d30_3 = stats.ttest_rel(d30_rep3[mask3], d30_single_3[mask3], nan_policy="omit")

d30_single_5 = pd.to_numeric(single_5["d_round_30"], errors="coerce")
d30_rep5     = pd.to_numeric(rep5_m["d_round_30"], errors="coerce")
mask5 = d30_single_5.notna() & d30_rep5.notna()
t_d30_5, p_d30_5 = stats.ttest_rel(d30_rep5[mask5], d30_single_5[mask5], nan_policy="omit")

D_d30 = pd.DataFrame({
    "Comparison": ["3× vs single", "5× vs single"],
    "N paired": [int(mask3.sum()), int(mask5.sum())],
    "Mean Day30 d (single)": [d30_single_3[mask3].mean(), d30_single_5[mask5].mean()],
    "Mean Day30 d (repeated)": [d30_rep3[mask3].mean(), d30_rep5[mask5].mean()],
    "ΔDay30 d (repeated − single)": [d30_rep3[mask3].mean() - d30_single_3[mask3].mean(),
                                     d30_rep5[mask5].mean() - d30_single_5[mask5].mean()],
    "t": [t_d30_3, t_d30_5],
    "p": [p_d30_3, p_d30_5]
}).round(4)

print("\n=== (D) Day30 effect-size improvement with repetition (failing set) ===")
print(D_d30.to_string(index=False))

# Recovery at Day30: d>0 & p<.05 among prior failures (kept as in previous)
def recovery_table(df_repeated, ids):
    sub = df_repeated.loc[ids].copy()
    rec = (pd.to_numeric(sub["d_round_30"], errors="coerce") > 0) & \
          (pd.to_numeric(sub["p_round_30"], errors="coerce") < 0.05)
    N = int(rec.notna().sum())
    r_n = int(rec.sum())
    r_pct = round(100 * rec.mean(), 1) if N > 0 else np.nan
    return N, r_n, r_pct, rec

N3, R3, R3pct, rec3 = recovery_table(rep3_m, ids_common_3)
N5, R5, R5pct, rec5 = recovery_table(rep5_m, ids_common_5)

rec_overall_df = pd.DataFrame([
    {"Cycle": "3×", "N": N3, "Recovered (n)": R3, "Recovered (%)": R3pct},
    {"Cycle": "5×", "N": N5, "Recovered (n)": R5, "Recovered (%)": R5pct},
]).round(1)

print("\n=== (D-1) Recovery overall (Day30; among prior failures) ===")
print(rec_overall_df.to_string(index=False))

# Recovery by intervention category
def recovery_by_category(df_repeated, rec_mask_series):
    cats = rep.set_index("study_id").loc[rec_mask_series.index, "intervention_category"]
    tab = pd.crosstab(cats, rec_mask_series)
    rows = []
    for c in tab.index:
        total = int(tab.loc[c].sum())
        rec_n = int(tab.loc[c, True]) if True in tab.columns else 0
        rec_pct = round(100 * rec_n / total, 1) if total > 0 else np.nan
        rows.append({"category": c, "N": total, "Recovered (n)": rec_n, "Recovered (%)": rec_pct})
    return pd.DataFrame(rows)

rec_by_cat_3 = recovery_by_category(rep3_m, rec3)
rec_by_cat_5 = recovery_by_category(rep5_m, rec5)

print("\n=== (D-2) Recovery by category (3×) ===")
print(rec_by_cat_3.to_string(index=False))
print("\n=== (D-3) Recovery by category (5×) ===")
print(rec_by_cat_5.to_string(index=False))

