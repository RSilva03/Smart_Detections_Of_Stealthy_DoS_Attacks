import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_hybrid_roq')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS_DIR, 'roq_hybrid_rskf_all_folds.csv'))

VARIANTS = ["SVM Only", "SVM + Distance", "Fuzzy + SVM", "Fuzzy + SVM + Distance"]
BASELINE = "SVM Only"

PRIMARY_METRICS   = ["FPR", "FNR"]
SECONDARY_METRICS = ["F1", "AUC", "Precision", "Recall"]
ALL_METRICS       = PRIMARY_METRICS + SECONDARY_METRICS

LOWER_IS_BETTER = {"FPR": True, "FNR": True,
                   "F1": False, "AUC": False,
                   "Precision": False, "Recall": False}

PALETTE = {
    "SVM + Distance":         "#F58518",
    "Fuzzy + SVM":            "#54A24B",
    "Fuzzy + SVM + Distance": "#E45756",
}
V3 = ["SVM + Distance", "Fuzzy + SVM", "Fuzzy + SVM + Distance"]


pivot = {}
for m in ALL_METRICS:
    pivot[m] = df.pivot_table(
        index=["Repeat", "Fold"], columns="Model", values=m
    ).reindex(columns=VARIANTS)

def cliffs_delta(x, y):
    
    x, y = np.asarray(x), np.asarray(y)
    greater = np.sum(x[:, None] > y[None, :])
    lesser  = np.sum(x[:, None] < y[None, :])
    return (greater - lesser) / (len(x) * len(y))

def interpret_delta(d):
    d = abs(d)
    if   d < 0.147: return "negligível"
    elif d < 0.330: return "pequeno"
    elif d < 0.474: return "médio"
    else:           return "grande"

def safe_wilcoxon(d, alternative="greater"):
    nonzero = d[d != 0]
    n_nz = len(nonzero)
    if n_nz == 0:
        return np.nan, 1.0
    if n_nz < 5:
        pos = int((d > 0).sum())
        neg = int((d < 0).sum())
        n   = pos + neg
        if n == 0:
            return np.nan, 1.0
        p = stats.binomtest(max(pos, neg), n, 0.5, alternative="greater").pvalue
        return float(pos - neg), float(p)
    try:
        stat, p = stats.wilcoxon(nonzero, zero_method="zsplit", alternative=alternative)
        return float(stat), float(p)
    except Exception:
        return np.nan, 1.0


rows = []

for metric in ALL_METRICS:
    b_vals = pivot[metric][BASELINE].values
    for variant in V3:
        v_vals = pivot[metric][variant].values
        mask   = ~(np.isnan(b_vals) | np.isnan(v_vals))
        b, v   = b_vals[mask], v_vals[mask]

        
        diff_dir = b - v if LOWER_IS_BETTER[metric] else v - b
        _, p_dir = safe_wilcoxon(diff_dir, alternative="greater")
        _, p_two = safe_wilcoxon(b - v,    alternative="two-sided")

        delta = cliffs_delta(v.tolist(), b.tolist())
        if LOWER_IS_BETTER[metric]:
            delta = -delta   

        mean_diff  = float(v.mean() - b.mean())
        rel_change = mean_diff / (b.mean() + 1e-9) * 100

        rows.append({
            "Metric":        metric,
            "Hypothesis":    "primary"   if metric in PRIMARY_METRICS else "secondary",
            "Variant":       variant,
            "Baseline_mean": round(b.mean(), 5),
            "Variant_mean":  round(v.mean(), 5),
            "Mean_diff":     round(mean_diff, 5),
            "Rel_change_%":  round(rel_change, 2),
            "p_raw":         p_dir if not np.isnan(p_dir) else 1.0,
            "p_two_sided":   p_two if not np.isnan(p_two) else 1.0,
            "Cliffs_delta":  round(delta, 4),
            "Effect_size":   interpret_delta(delta),
            "n_pairs":       int(mask.sum()),
        })

results_df = pd.DataFrame(rows)

def apply_correction(sub_df, method, alpha=0.05):
    p_vals = sub_df["p_raw"].values
    reject, p_corr, _, _ = multipletests(p_vals, alpha=alpha, method=method)
    sub_df = sub_df.copy()
    sub_df["p_corrected"] = np.round(p_corr, 5)
    sub_df["p_corrected_method"] = method
    sub_df["reject_H0"] = reject
    sub_df["Significant_raw"]  = sub_df["p_raw"]       < alpha
    sub_df["Significant_corr"] = sub_df["reject_H0"]
    return sub_df

primary_mask   = results_df["Hypothesis"] == "primary"
secondary_mask = results_df["Hypothesis"] == "secondary"

primary_df   = apply_correction(results_df[primary_mask].copy(),   method="holm-sidak")
secondary_df = apply_correction(results_df[secondary_mask].copy(), method="fdr_bh")

results_full = pd.concat([primary_df, secondary_df], ignore_index=True)


results_full = results_full.sort_values(["Hypothesis", "Metric", "Variant"]).reset_index(drop=True)

results_full.to_csv(os.path.join(RESULTS_DIR, 'roq_wilcoxon_bonferroni_results.csv'), index=False)

print("=" * 120)
print("  WILCOXON SIGNED-RANK  +  CLIFF'S DELTA  +  MULTIPLE COMPARISONS CORRECTION")
print("  Primary  (FPR, FNR)  : Holm-Šídák   (controls FWER,  6 tests, α=0.05)")
print("  Secundary (F1, AUC…) : Benjamini-Hochberg (controls FDR, 12 tests, α=0.05)")
print("=" * 120)

for hyp_label, hyp_key, metrics in [
    ("PRIMARY HYPOTHESES   — FPR & FNR  (Holm-Šídák)",   "primary",   PRIMARY_METRICS),
    ("SECONDARY HYPOTHESES — F1, AUC…   (Benjamini-Hochberg)", "secondary", SECONDARY_METRICS),
]:
    print(f"\n{'─'*120}")
    print(f"  {hyp_label}")
    print(f"{'─'*120}")
    header = (f"  {'Metric':<6} {'Variant':<28} {'Base':>7} {'Var':>7} "
              f"{'Δ abs':>9} {'Δ rel%':>7}  {'p_raw':>8}  {'p_corr':>8}  "
              f"{'δ Cliff':>9}  {'Effect':>12}  {'Sig raw':>8}  {'Sig corr':>9}")
    print(header)
    print("  " + "-" * 116)
    sub = results_full[results_full["Hypothesis"] == hyp_key]
    for metric in metrics:
        msub = sub[sub["Metric"] == metric]
        direction = "↓" if LOWER_IS_BETTER[metric] else "↑"
        for _, row in msub.iterrows():
            sr = "✓" if row["Significant_raw"]  else "✗"
            sc = "✓" if row["Significant_corr"] else "✗"
            print(f"  {row['Metric']:<6}{direction}  {row['Variant']:<28} "
                  f"{row['Baseline_mean']:>7.4f} {row['Variant_mean']:>7.4f} "
                  f"{row['Mean_diff']:>+9.5f} {row['Rel_change_%']:>+7.1f}%  "
                  f"{row['p_raw']:>8.4f}  {row['p_corrected']:>8.4f}  "
                  f"{row['Cliffs_delta']:>+9.4f}  {row['Effect_size']:>12}  "
                  f"{sr:>8}  {sc:>9}")


for label, p_col, title_suffix in [
    ("raw",       "p_raw",        "p-values RAW (without correction)"),
    ("corrected", "p_corrected",  "p-values Corrected\n(Primary: Holm-Šídák | Secondary: Benjamini-Hochberg)"),
]:
    piv = results_full.pivot_table(index="Variant", columns="Metric", values=p_col, aggfunc="first")
    piv = piv.reindex(index=V3, columns=ALL_METRICS)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(piv, annot=False, cmap="RdYlGn_r", linewidths=0.6,
                ax=ax, vmin=0, vmax=0.3)

    sig_col  = "Significant_raw" if label == "raw" else "Significant_corr"
    for i, variant in enumerate(V3):
        for j, metric in enumerate(ALL_METRICS):
            row = results_full[(results_full["Variant"] == variant) &
                               (results_full["Metric"]  == metric)]
            if row.empty:
                continue
            val = row[p_col].values[0]
            sig = row[sig_col].values[0]
            txt = f"{val:.4f}" + (" *" if sig else "")
            color = "white" if (not np.isnan(val) and val < 0.08) else "black"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold" if sig else "normal")

    
    ax.axvline(len(PRIMARY_METRICS), color="white", linewidth=3, linestyle="--")
    ax.set_title(f"Wilcoxon — {title_suffix}\n* = p < 0.05 significative  |  Primary: FPR, FNR  |  Secondary: F1, AUC, Precision, Recall",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Metric"); ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"wilcoxon_pvalue_{label}_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()


fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Impact of correction for multiple tests\n(lsolid line = α = 0.05)",
             fontsize=13, fontweight="bold")

for ax, hyp_key, title, metrics, method_label in [
    (axes[0], "primary",   "Primary — Holm-Šídák (FWER)",      PRIMARY_METRICS,   "Holm-Šídák"),
    (axes[1], "secondary", "Secondary — Benjamini-Hochberg (FDR)", SECONDARY_METRICS, "BH"),
]:
    sub = results_full[results_full["Hypothesis"] == hyp_key].copy()
    sub["label"] = sub["Metric"] + "\n" + sub["Variant"].str.replace(" + ", "+", regex=False)

    x = range(len(sub))
    ax.scatter(x, sub["p_raw"],       s=80, zorder=5, color="steelblue",  label="p raw",       marker="o")
    ax.scatter(x, sub["p_corrected"], s=80, zorder=5, color="firebrick",  label=f"p {method_label}", marker="D")

    for xi, (_, row) in zip(x, sub.iterrows()):
        ax.plot([xi, xi], [row["p_raw"], row["p_corrected"]], "grey",
                linewidth=1, alpha=0.5, zorder=3)

    ax.axhline(0.05, color="black", linewidth=1.5, linestyle="--", label="α = 0.05")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["label"].tolist(), fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("p-value"); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.02, min(1.05, sub[["p_raw","p_corrected"]].max().max() * 1.3 + 0.1))
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "wilcoxon_correction_impact.png"), dpi=200, bbox_inches="tight")
plt.close()


piv_d = results_full.pivot_table(index="Variant", columns="Metric",
                                  values="Cliffs_delta", aggfunc="first")
piv_d = piv_d.reindex(index=V3, columns=ALL_METRICS)

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(piv_d, annot=False, cmap="RdYlGn", linewidths=0.6,
            ax=ax, center=0, vmin=-0.5, vmax=0.5)

for i, variant in enumerate(V3):
    for j, metric in enumerate(ALL_METRICS):
        row = results_full[(results_full["Variant"] == variant) &
                           (results_full["Metric"]  == metric)]
        if row.empty:
            continue
        d   = row["Cliffs_delta"].values[0]
        sig = row["Significant_corr"].values[0]
        txt = f"{d:+.4f}" + (" *" if sig else "")
        color = "white" if abs(d) > 0.3 else "black"
        ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                fontsize=10, color=color, fontweight="bold" if sig else "normal")

ax.axvline(len(PRIMARY_METRICS), color="white", linewidth=3, linestyle="--")
ax.set_title("Cliff's Delta — vs SVM Only  |  * = p_corr < 0.05\n"
             "FPR/FNR positive = variant reduces error  |  F1/AUC positive = variant improvement",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Metric"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "wilcoxon_cliffs_delta_corrected.png"), dpi=200, bbox_inches="tight")
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Distribution of paired differences vs SVM Only\n(50 pairs: 5 repetitions × 10 folds)",
             fontsize=13, fontweight="bold")

for ax, metric in zip(axes, ["FPR", "F1"]):
    diff_rows = []
    for v in V3:
        diffs = pivot[metric][v].values - pivot[metric][BASELINE].values
        for d in diffs:
            diff_rows.append({"Variant": v, "Δ": d})
    diff_df = pd.DataFrame(diff_rows)
    sns.boxplot(
        data=diff_df,
        x="Variant",
        y="Δ",
        palette=[PALETTE[v] for v in V3],
        ax=ax,
        width=0.5,
        linewidth=1.3,
    )
    ax.axhline(0, color="black", linewidth=1.8, linestyle="--")
    direction = "↓ better" if LOWER_IS_BETTER[metric] else "↑ better"
    ax.set_title(f"Δ {metric}  ({direction})", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel(f"Δ {metric} (variant − SVM Only)")

    for xi, v in enumerate(V3):
        row = results_full[(results_full["Variant"] == v) &
                           (results_full["Metric"] == metric)]
        if row.empty:
            continue
        p_c = row["p_corrected"].values[0]
        sig = row["Significant_corr"].values[0]
        lbl = f"p={p_c:.4f}" + (" *" if sig else "")
        ymax = diff_df[diff_df["Variant"] == v]["Δ"].max()
        ax.text(
            xi,
            ymax + 0.005,
            lbl,
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            fontstyle="italic",
        )

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "wilcoxon_paired_boxplot_corrected.png"), dpi=200, bbox_inches="tight")
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Paired differences per fold — FPR & FNR\nNegative values = variant is better",
             fontsize=13, fontweight="bold")

for ax, metric in zip(axes, ["FPR", "FNR"]):
    for v in V3:
        diffs = pivot[metric][v].values - pivot[metric][BASELINE].values
        ax.plot(range(1, 51), diffs, alpha=0.7, linewidth=1.3,
                color=PALETTE[v], label=v)
    ax.axhline(0, color="black", linewidth=1.8, linestyle="--")
    ax.set_title(f"Δ {metric} per fold", fontsize=12, fontweight="bold")
    ax.set_xlabel("Fold ratio  (50 = 5 reps × 10 folds)")
    ax.set_ylabel(f"Δ {metric} (variant − SVM Only)")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "wilcoxon_fold_diff_fpr_fnr.png"), dpi=200, bbox_inches="tight")
plt.close()


