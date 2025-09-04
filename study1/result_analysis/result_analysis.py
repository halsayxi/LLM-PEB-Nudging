import pandas as pd
import numpy as np
from scipy import stats
from collections import OrderedDict

df = pd.read_csv("../data/nudge-replication.csv")

def success_mask(df, model_prefix):
    d_col = f"{model_prefix}_llm_cohens_d"
    p_col = f"{model_prefix}_llm_p_value"
    d = pd.to_numeric(df[d_col], errors="coerce")
    p = pd.to_numeric(df[p_col], errors="coerce")
    return (d > 0) & (p < 0.05), d, p

models = OrderedDict([
    ("gpt35", "GPT-3.5"),
    ("gpt4o", "GPT-4o"),
    ("claude3", "Claude-3"),
    ("deepseekv3", "DeepSeek-V3"),
])

success_summary = []
for mkey, mname in models.items():
    succ, d, p = success_mask(df, mkey)
    n_total = succ.notna().sum()
    n_succ = succ.sum()
    rate = 100 * n_succ / n_total if n_total > 0 else np.nan
    success_summary.append({"Model": mname,
                            "N (evaluable)": int(n_total),
                            "Success (n)": int(n_succ),
                            "Success rate (%)": round(rate, 1)})
success_df = pd.DataFrame(success_summary)
print(success_df)

cor_rows = []
for mkey, mname in models.items():
    d_h = pd.to_numeric(df["cohens_d"], errors="coerce")
    d_m = pd.to_numeric(df[f"{mkey}_llm_cohens_d"], errors="coerce")
    mask = d_h.notna() & d_m.notna()
    if mask.sum() >= 3:
        r, p = stats.pearsonr(d_h[mask], d_m[mask])
    else:
        r, p = (np.nan, np.nan)
    cor_rows.append({"Model": mname, "r": r, "p": p, "N": int(mask.sum())})
cor_df = pd.DataFrame(cor_rows)
print(cor_df)

paired_rows = []
for mkey, mname in models.items():
    d_h = pd.to_numeric(df["cohens_d"], errors="coerce")
    d_m = pd.to_numeric(df[f"{mkey}_llm_cohens_d"], errors="coerce")
    mask = d_h.notna() & d_m.notna()
    if mask.sum() >= 3:
        t, p = stats.ttest_rel(d_m[mask], d_h[mask], nan_policy="omit")
        delta = (d_m[mask] - d_h[mask]).mean()
    else:
        t, p, delta = (np.nan, np.nan, np.nan)
    paired_rows.append({"Model": mname, "Delta_d (LLM - Human)": delta,
                        "t": t, "p": p, "N": int(mask.sum())})
paired_df = pd.DataFrame(paired_rows)
print(paired_df)

succ_mask_gpt35, d35, p35 = success_mask(df, "gpt35")
valid = succ_mask_gpt35.notna()
df_valid = df.loc[valid].copy()
df_valid["gpt35_success"] = succ_mask_gpt35[valid].values

n_succ = pd.to_numeric(df_valid.loc[df_valid["gpt35_success"], "n_comparison"], errors="coerce").dropna()
n_fail = pd.to_numeric(df_valid.loc[~df_valid["gpt35_success"], "n_comparison"], errors="coerce").dropna()
t_stat, t_p = stats.ttest_ind(n_succ, n_fail, equal_var=False)
print("n_comparison: success mean =", n_succ.mean(), "fail mean =", n_fail.mean(),
      "t =", t_stat, "p =", t_p)

ct_cat = pd.crosstab(df_valid["gpt35_success"], df_valid["intervention_category"])
chi2_cat, p_cat, dof_cat, _ = stats.chi2_contingency(ct_cat)
print("Intervention category chi-square:", chi2_cat, dof_cat, p_cat)
print(ct_cat)

ct_type = pd.crosstab(df_valid["gpt35_success"], df_valid["type_experiment"])
chi2_type, p_type, dof_type, _ = stats.chi2_contingency(ct_type)
print("Type of experiment chi-square:", chi2_type, dof_type, p_type)
print(ct_type)

rates_by_cat = (ct_cat.loc[True] / ct_cat.sum(axis=0) * 100).round(1)
print("Success rates by category (%):")
print(rates_by_cat)

fail_type = ct_type.loc[False].sort_values(ascending=False)
print("Failures by type of experiment:")
print(fail_type)
