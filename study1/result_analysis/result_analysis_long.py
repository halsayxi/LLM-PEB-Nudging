import pandas as pd
import numpy as np
from scipy import stats

# ========= 1) Load CSV (robust to encoding) =========
CSV_PATH = "../data/study_1_long.csv"   

def read_csv_robust(path):
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last try without encoding hint
    return pd.read_csv(path)

df = read_csv_robust(CSV_PATH).copy()

# Sanity-check required columns
required_cols = {"ATE", "llm_ATE", "significance_binary", "llm_p_value"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ========= 2) Significance & directions =========
# Human significance from binary column (1 = significant, 0 = not significant)
df["human_sig"] = df["significance_binary"].astype(int) == 1

# LLM significance: < 0.1 is significant (robust to stray strings)
def parse_llm_sig(p):
    if pd.isna(p):
        return False
    try:
        return float(str(p).replace("<","").replace("=","").replace(">","")) < 0.1
    except Exception:
        return False

df["llm_sig"] = df["llm_p_value"].apply(parse_llm_sig)

# Effect direction helper
def effect_sign(x):
    try:
        v = float(x)
        if np.isnan(v): return 0
        if v > 0: return 1
        if v < 0: return -1
        return 0
    except Exception:
        return 0

df["human_dir"] = df["ATE"].apply(effect_sign)
df["llm_dir"]   = df["llm_ATE"].apply(effect_sign)
df["same_dir"]  = (df["human_dir"] * df["llm_dir"]) > 0  # strictly same non-zero sign

# ========= 3) Replication criteria (English) =========
# Success if:
#   (a) both significant AND same effect direction, OR
#   (b) both non-significant
df["replicated"] = (
    (df["human_sig"] & df["llm_sig"] & df["same_dir"]) |
    (~df["human_sig"] & ~df["llm_sig"])
)

# ========= 4) Correlation & summary (no export) =========
corr_df = df[["ATE", "llm_ATE"]].apply(pd.to_numeric, errors="coerce").dropna()
if len(corr_df) >= 2:
    r, p_corr = stats.pearsonr(corr_df["ATE"], corr_df["llm_ATE"])
else:
    r, p_corr = (np.nan, np.nan)

n_total   = len(df)
n_success = int(df["replicated"].sum())
success_rate = (n_success / n_total) if n_total else np.nan

print("=== Benchmarking LLM vs. Long-run Field Experiments ===")
print(f"Total studies: {n_total}")
print(f"Pearson r (ATE vs. llm_ATE): {r:.2f}, p = {p_corr:.3g}")
print(f"Replication success: {n_success}/{n_total} ({success_rate:.0%})")