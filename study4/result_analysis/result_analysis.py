import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ============== Load ==============
df = pd.read_csv("../data/social-simulation.csv")
paths = {
    "Scale-Free": "../data/social_agent_behavior/hcn_0.5_agent_behavior.csv",   # Barabási–Albert
    "Small-World": "../data/social_agent_behavior/swn_0.5_agent_behavior.csv",  # Watts–Strogatz
    "Community":   "../data/social_agent_behavior/sbn_0.5_agent_behavior.csv",  # Stochastic Block Model
}

networks = ["barabasi_albert", "watts_strogatz", "stochastic_block"]
baseline_key = "no_spread"

# ============== Helpers ==============
def paired_t_with_ci(x, y, alpha=0.05):
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna(); x, y = x[mask], y[mask]
    diff = x - y; n = diff.shape[0]
    t_stat, p_val = stats.ttest_rel(x, y, nan_policy="omit")
    md = diff.mean(); sd = diff.std(ddof=1)
    dz = md / sd if sd > 0 else np.nan
    se = sd / np.sqrt(n) if n > 0 else np.nan
    tcrit = stats.t.ppf(0.975, df=n-1) if n > 1 else np.nan
    ci_low = md - tcrit * se if (tcrit==tcrit) else np.nan
    ci_high = md + tcrit * se if (tcrit==tcrit) else np.nan
    return {"n": int(n), "mean_x": float(x.mean()), "mean_y": float(y.mean()),
            "mean_diff": float(md), "t": float(t_stat), "p": float(p_val),
            "dz": float(dz), "ci_low": float(ci_low), "ci_high": float(ci_high)}

def print_paired_result(title, res, label_x, label_y, unit="d"):
    print(f"\n--- {title} ---")
    print(f"N = {res['n']}")
    print(f"{label_y} mean ({unit}): {res['mean_y']:.3f}")
    print(f"{label_x} mean ({unit}): {res['mean_x']:.3f}")
    print(f"Mean difference ({label_x} − {label_y}) ({unit}): {res['mean_diff']:.3f}")
    print(f"Paired t = {res['t']:.3f}, p = {res['p']:.3f}, dz = {res['dz']:.2f}")
    print(f"95% CI of difference: [{res['ci_low']:.3f}, {res['ci_high']:.3f}]")

# ============== (1) Networks vs no_spread (Day-30 d) ==============
print("=== (1) Network diffusion improves Day-30 effect vs no-diffusion baseline ===")
df_base = df[df["network_type"] == baseline_key].set_index("study_id")
for net in networks:
    df_net = df[df["network_type"] == net].set_index("study_id")
    aligned = df_net.join(df_base, lsuffix="_net", rsuffix="_base", how="inner")
    res = paired_t_with_ci(aligned["d_round_30_net"], aligned["d_round_30_base"])
    print_paired_result(
        title=f"{net} vs no_spread (Day-30 effect size)",
        res=res, label_x=f"{net}", label_y="no_spread", unit="d"
    )

# ============== (2) Day-2 bands (prop mean across networks) → Day-30 outcomes ==============
print("\n=== (2) Early strength (Day-2) bands under diffusion and Day-30 outcomes ===")
df_prop = df[df["network_type"].isin(networks)].copy()
agg = df_prop.groupby("study_id")[["d_round_2", "d_round_30"]].mean().rename(
    columns={"d_round_2":"d2_mean_prop", "d_round_30":"d30_mean_prop"}
)
def band_d2(v):
    if v < 0.20: return "low"
    elif v <= 0.40: return "medium"
    else: return "high"
agg["band"] = agg["d2_mean_prop"].apply(band_d2)

print("Counts by band (studies):", agg["band"].value_counts().to_dict())
band_stats = agg.groupby("band")["d30_mean_prop"].agg(["mean","std","count"]).rename(
    columns={"mean":"Day30 mean d (prop)", "std":"SD", "count":"n"}
).round(3)
print("\nDay-30 outcomes by Day-2 bands (averaged across networks):")
print(band_stats.to_string())

groups = [g["d30_mean_prop"].values for _, g in agg.groupby("band")]
if len(groups) >= 2 and all(len(g)>=1 for g in groups):
    F, p = stats.f_oneway(*groups)
    k, N = len(groups), sum(len(g) for g in groups)
    print(f"\nOne-way ANOVA on Day-30 by bands: F({k-1}, {N-k}) = {F:.2f}, p = {p:.4f}")
else:
    print("\nNot enough groups for ANOVA.")

# Daily trend (β/day, r, p) within each band
day_cols = sorted([c for c in df.columns if c.startswith("d_round_")], key=lambda x: int(x.split("_")[-1]))
days = np.array([int(c.split("_")[-1]) for c in day_cols])
print("\nDaily trend (β per day, Pearson r, p) for each band:")
for band, idx in agg.groupby("band").groups.items():
    sub = df_prop[df_prop["study_id"].isin(list(idx))]
    per_day_mean = sub.groupby("study_id")[day_cols].mean().mean(axis=0)
    mask = ~per_day_mean.isna()
    X = days[mask].astype(float); Y = per_day_mean[mask].values.astype(float)
    slope, intercept, r, p, se = stats.linregress(X, Y)
    print(f"- {band}: β = {slope:.4f} per day, r = {r:.3f}, p = {p:.6f}")

# ============== (3) Consolidation vs conversion (Day-30 rates) ==============
print("\n=== (3) Consolidation vs conversion (Day-30 rates) ===")
def avg_over_nets(df_all, col):
    a = (df_all[df_all["network_type"].isin(networks)]
         .groupby("study_id")[col].mean())
    b = (df_all[df_all["network_type"] == baseline_key]
         .set_index("study_id")[col])
    return a.to_frame("prop").join(b.to_frame("base"), how="inner")

for label, col in [("Initially pro-environmental agents", "group1_round_30"),
                   ("Initially non-pro-environmental agents", "group0_round_30")]:
    xy = avg_over_nets(df, col)
    res = paired_t_with_ci(xy["prop"], xy["base"])
    print_paired_result(
        title=f"{label}: propagation vs no_spread (Day-30 rate)",
        res=res, label_x="propagation", label_y="no_spread", unit="rate"
    )


# ============== (4) dissenting peers ==============
results = []
for name, path in paths.items():
    dfb = pd.read_csv(path)
    d = dfb[dfb['day'].between(3, 30)].copy()
    d['mismatch'] = np.where(
        d['last_choice'] == 0,
        d['choice_1_friends'],
        d['activate_friends'] - d['choice_1_friends']
    )
    d['c01'] = ((d['last_choice'] == 0) & (d['choice'] == 1)).astype(int)
    d['c10'] = ((d['last_choice'] == 1) & (d['choice'] == 0)).astype(int)
    for direction, flag in [('0→1', 'c01'), ('1→0', 'c10')]:
        grp = (
            d[d['last_choice'] == int(direction[0])]
            .groupby('mismatch')[flag]
            .agg(['mean', 'count'])
            .reset_index()
        )
        grp = grp[grp['count'] >= 10]
        if grp.shape[0] < 2:
            continue
        X = sm.add_constant(grp['mismatch'])
        model = sm.OLS(grp['mean'], X).fit()
        beta = model.params.iloc[1]
        ci_low, ci_high = model.conf_int().iloc[1]
        pval = model.pvalues.iloc[1]
        results.append({
            'Network': name,
            'Dir': direction,
            'β': beta,
            'CI': (ci_low, ci_high),
            'p': pval
        })

fig4e = pd.DataFrame(results)
print("Figure 4e:\n", fig4e)
