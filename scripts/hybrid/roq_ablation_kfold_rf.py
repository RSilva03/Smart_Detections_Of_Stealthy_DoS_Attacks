import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from scipy.spatial.distance import cdist
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

SEED      = 42
N_SPLITS  = 10
N_REPEATS = 5
CI        = 0.95   

FEATURES = [
    'n_pkts','bytes_total','tcp_bytes','pkts_rate','bytes_rate',
    'pkt_size_mean','pkt_size_std','iat_mean','iat_std','iat_cv',
    'burst_peak_ratio','burst_cv','bytes_rate_cv','tcp_ratio'
]

np.random.seed(SEED)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_hybrid_roq')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'roq', 'dataset_roq_window_full.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


df = pd.read_csv(DATASET_PATH)
X  = df[FEATURES].values
y  = df["label"].values

print(f"[DATA] Total samples: {len(X)}")
print(f"[DATA] Attack rate: {y.mean():.3f}")
print(f"[CV] {N_REPEATS} repeats × {N_SPLITS} folds = {N_REPEATS*N_SPLITS} evaluations\n")

def trimf(x, a, b, c):
    left  = np.where((x > a) & (x <= b), (x - a) / (b - a + 1e-9), 0.0)
    right = np.where((x > b) & (x < c),  (c - x) / (c - b + 1e-9), 0.0)
    peak  = np.where(x == b, 1.0, 0.0)
    return np.clip(left + right + peak, 0, 1)

def trapmf(x, a, b, c, d):
    rise = np.where(x <= a, 0.0, np.where(x < b, (x-a)/(b-a+1e-9), 1.0))
    fall = np.where(x >= d, 0.0, np.where(x > c, (d-x)/(d-c+1e-9), 1.0))
    return np.clip(np.minimum(rise, fall), 0, 1)

def make_minmax_scaler(v_ref):
    mn, mx = v_ref.min(), v_ref.max()
    return lambda v: np.clip((v- mn) / (mx - mn + 1e-9), 0, 1)

def fuzzy_svm_distance(svm_score_te, dist_te, svm_score_tr, dist_tr):
    scale_rf = make_minmax_scaler(svm_score_tr)
    scale_dist = make_minmax_scaler(dist_tr)
    rs = scale_rf(svm_score_te)
    ds = scale_dist(dist_te)

    rf_low  = trapmf(rs, 0.00, 0.00, 0.25, 0.45)
    rf_med  = trimf (rs, 0.35, 0.50, 0.65)
    rf_high = trapmf(rs, 0.55, 0.75, 1.00, 1.00)

    d_low   = trapmf(ds, 0.00, 0.00, 0.25, 0.40)
    d_med   = trimf (ds, 0.30, 0.50, 0.70)
    d_high  = trapmf(ds, 0.60, 0.75, 1.00, 1.00)

    u = np.linspace(0, 1, 1000)

    out_benign = trapmf(u, 0.00, 0.00, 0.15, 0.30)
    out_medlow = trimf (u, 0.20, 0.35, 0.50)
    out_med    = trimf (u, 0.35, 0.50, 0.65)
    out_high   = trimf (u, 0.60, 0.78, 0.92)
    out_vhigh  = trapmf(u, 0.82, 0.92, 1.00, 1.00)

    agg = np.zeros((len(rs), len(u)))

    R = lambda s, o: np.minimum(s[:, None], o[None, :])

    agg = np.maximum(agg, R(np.minimum(rf_high, d_high),  out_vhigh))
    agg = np.maximum(agg, R(np.minimum(rf_high, d_med),   out_high))
    agg = np.maximum(agg, R(np.minimum(rf_med,  d_high),  out_high))
    agg = np.maximum(agg, R(np.minimum(rf_low,  d_low),   out_benign))
    agg = np.maximum(agg, R(np.minimum(rf_low,  d_med),   out_medlow))
    agg = np.maximum(agg, R(np.minimum(rf_med,  d_low),   out_medlow))
    agg = np.maximum(agg, R(np.minimum(rf_med,  d_med),   out_med))
    agg = np.maximum(agg, R(np.minimum(rf_high, d_low),   out_med))
    agg = np.maximum(agg, R(np.minimum(rf_low,  d_high),  out_med))

    num = (agg * u[None, :]).sum(axis=1)
    den = agg.sum(axis=1) + 1e-9
    return num / den

def fuzzy_svm_only(svm_score_te, svm_score_tr):
    scale_rf = make_minmax_scaler(svm_score_tr)
    rs = scale_rf(svm_score_te)

    u = np.linspace(0, 1, 1000)

    rf_low  = trapmf(rs, 0.00, 0.00, 0.25, 0.45)
    rf_med  = trimf (rs, 0.35, 0.50, 0.65)
    rf_high = trapmf(rs, 0.55, 0.75, 1.00, 1.00)

    out_benign = trapmf(u, 0.00, 0.00, 0.15, 0.30)
    out_med    = trimf (u, 0.35, 0.50, 0.65)
    out_vhigh  = trapmf(u, 0.82, 0.92, 1.00, 1.00)

    agg = np.zeros((len(rs), len(u)))

    R = lambda s, o: np.minimum(s[:, None], o[None, :])

    agg = np.maximum(agg, R(rf_low, out_benign))
    agg = np.maximum(agg, R(rf_med, out_med))
    agg = np.maximum(agg, R(rf_high, out_vhigh))

    num = (agg * u[None, :]).sum(axis=1)
    den = agg.sum(axis=1) + 1e-9
    return num / den


def best_threshold_on_val(y_true, scores):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (scores >= t).astype(int)
        f1   = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)

def compute_metrics(y_true, y_pred, scores):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "AUC":       roc_auc_score(y_true, scores),
        "FPR":       (((y_pred == 1) & (y_true == 0)).sum() /
                      ((y_true == 0).sum() + 1e-9)),
        "FNR":       (((y_pred == 0) & (y_true == 1)).sum() /
                      ((y_true == 1).sum() + 1e-9)),
    }

def robust_centroid(X_subset):
    Q1 = np.percentile(X_subset, 25, axis=0)
    Q3 = np.percentile(X_subset, 75, axis=0)
    IQR = Q3 - Q1
    mask = np.all((X_subset >= Q1 - 1.5*IQR) & (X_subset <= Q3 + 1.5*IQR), axis=1)
    subset_clean = X_subset[mask]
    return subset_clean.mean(axis=0, keepdims=True) if len(subset_clean) > 0 else X_subset.mean(axis=0, keepdims=True)

rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
)

VARIANTS = ["SVM Only", "SVM + Distance", "Fuzzy + SVM", "Fuzzy + SVM + Distance"]
fold_results = {v: [] for v in VARIANTS}

for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
    rep   = fold_idx // N_SPLITS + 1
    fold  = fold_idx %  N_SPLITS + 1

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    
    sc       = StandardScaler()
    X_tr_s   = sc.fit_transform(X_tr)
    X_te_s   = sc.transform(X_te)

    
    svm  = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)
    svm.fit(X_tr_s, y_tr)
    svm_prob_tr = svm.predict_proba(X_tr_s)[:, 1]
    svm_prob_te = svm.predict_proba(X_te_s)[:, 1]

    
    c_benign = robust_centroid(X_tr_s[y_tr == 0])
    c_attack = robust_centroid(X_tr_s[y_tr == 1])

    def dist_fn(Xs):
        db = cdist(Xs, c_benign, metric="euclidean").ravel()
        da = cdist(Xs, c_attack, metric="euclidean").ravel()
        return db / (db + da + 1e-9)
    
    dist_tr   = dist_fn(X_tr_s)
    dist_te   = dist_fn(X_te_s)

    scale_rf = make_minmax_scaler(svm_prob_tr)
    scale_dist = make_minmax_scaler(dist_tr)

    score_svm = svm_prob_te
    score_svm_dist = 0.7 * scale_rf(svm_prob_te) + 0.3 * scale_dist(dist_te)
    score_fuzzy_svm = fuzzy_svm_only(svm_prob_te, svm_prob_tr)
    score_fuzzy_svm_d = fuzzy_svm_distance(svm_prob_te, dist_te, svm_prob_tr, dist_tr)

    score_svm_dist_tr = 0.7 * scale_rf(svm_prob_tr) + 0.3 * scale_dist(dist_tr)
    score_fuzzy_svm_tr = fuzzy_svm_only(svm_prob_tr, svm_prob_tr)
    score_fuzzy_svm_d_tr = fuzzy_svm_distance(svm_prob_tr, dist_tr, svm_prob_tr, dist_tr)

    t_svm = best_threshold_on_val(y_tr, svm_prob_tr)
    t_svm_dist = best_threshold_on_val(y_tr, score_svm_dist_tr)
    t_fuzzy_svm = best_threshold_on_val(y_tr, score_fuzzy_svm_tr)
    t_fuzzy_svm_d = best_threshold_on_val(y_tr, score_fuzzy_svm_d_tr)

    for variant, score, thresh in [
        ("SVM Only",               score_svm,           t_svm),
        ("SVM + Distance",         score_svm_dist,      t_svm_dist),
        ("Fuzzy + SVM",            score_fuzzy_svm,     t_fuzzy_svm),
        ("Fuzzy + SVM + Distance", score_fuzzy_svm_d,   t_fuzzy_svm_d),
    ]:
        pred = (score >= thresh).astype(int)
        m    = compute_metrics(y_te, pred, score)
        m["Repeat"] = rep
        m["Fold"]   = fold
        fold_results[variant].append(m)

    if fold_idx % 10 == 9:
        print(f"  [Rep {rep:2d}] {N_SPLITS} folds concluded")

print("\n Cross-validation concluded\n")


METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "FPR", "FNR"]

summary_rows = []
fold_dfs     = {}

for variant in VARIANTS:
    fold_df = pd.DataFrame(fold_results[variant])
    fold_dfs[variant] = fold_df

    row = {"Model": variant}
    for m in METRICS:
        vals = fold_df[m].values
        mean = vals.mean()
        std  = vals.std(ddof=1)
        n    = len(vals)
        t_crit = st.t.ppf((1 + CI) / 2, df=n - 1)
        margin = t_crit * std / np.sqrt(n)

        row[f"{m}_mean"]   = round(mean, 5)
        row[f"{m}_std"]    = round(std, 5)
        row[f"{m}_CI_low"] = round(mean - margin, 5)
        row[f"{m}_CI_high"]= round(mean + margin, 5)

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values(
    by=["F1_mean", "AUC_mean"], ascending=False
).reset_index(drop=True)


print("=" * 80)
print(" RESULTs — REPEATED STRATIFIED K-FOLD (5×10 = 50 evals)")
print("=" * 80)
for _, row in summary_df.iterrows():
    print(f"\n  {row['Model']}")
    for m in METRICS:
        print(f"    {m:12s}: {row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}"
              f"  [IC95%: {row[f'{m}_CI_low']:.4f} – {row[f'{m}_CI_high']:.4f}]")

summary_csv = os.path.join(RESULTS_DIR, "roq_hybrid_rskf_summary.csv")
summary_df.to_csv(summary_csv, index=False)
all_folds = pd.concat(
    [pd.DataFrame(v).assign(Model=k) for k, v in fold_results.items()],
    ignore_index=True
)
all_folds_csv = os.path.join(RESULTS_DIR, "roq_hybrid_rskf_all_folds.csv")
all_folds.to_csv(all_folds_csv, index=False)

print(f"\n {summary_csv}")
print(all_folds_csv)


PALETTE = {
    "SVM Only":               "#4C78A8",
    "SVM + Distance":         "#F58518",
    "Fuzzy + SVM":            "#54A24B",
    "Fuzzy + SVM + Distance": "#E45756"
}

ORDER = ["SVM Only", "SVM + Distance", "Fuzzy + SVM", "Fuzzy + SVM + Distance"]


fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle("Repeated Stratified K-Fold (5×10) — Metric Distributions",
             fontsize=16, fontweight="bold")

plot_metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "FPR", "FNR"]
for ax, metric_name in zip(axes.ravel(), plot_metrics):
    data = pd.concat([
        fold_dfs[v][[metric_name]].rename(columns={metric_name: "value"}).assign(Model=v)
        for v in ORDER
    ])
    sns.boxplot(
        data=data, x="Model", y="value", order=ORDER,
        palette=[PALETTE[m] for m in ORDER], ax=ax,
        width=0.5, linewidth=1.2, fliersize=3
    )
    ax.set_title(metric_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=22)
    ax.set_ylim(-0.02, 1.05)

axes.ravel()[-1].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(PLOTS_DIR, "rskf_boxplots.png"), dpi=200, bbox_inches="tight")
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Repeated Stratified K-Fold — Mean ± 95% CI",
             fontsize=14, fontweight="bold")

for ax, metric in zip(axes, ["F1", "AUC"]):
    means  = [summary_df[summary_df["Model"]==v][f"{metric}_mean"].values[0] for v in ORDER]
    lows   = [summary_df[summary_df["Model"]==v][f"{metric}_CI_low"].values[0] for v in ORDER]
    highs  = [summary_df[summary_df["Model"]==v][f"{metric}_CI_high"].values[0] for v in ORDER]
    errs   = [[m - l for m, l in zip(means, lows)],
              [h - m for m, h in zip(means, highs)]]
    colors = [PALETTE[v] for v in ORDER]

    bars = ax.bar(ORDER, means, color=colors, width=0.5,
                  yerr=errs, capsize=6, error_kw={"linewidth": 2})
    ax.set_title(f"Mean {metric} ± 95% CI", fontsize=12, fontweight="bold")
    ax.set_ylim(0.88, 1.01)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rskf_mean_ci_barplot.png"), dpi=200, bbox_inches="tight")
plt.close()

hm_cols = ["F1_mean","F1_std","AUC_mean","AUC_std",
           "Precision_mean","Recall_mean","FPR_mean","FNR_mean"]
hm_df   = summary_df.set_index("Model")[hm_cols].reindex(ORDER)

plt.figure(figsize=(13, 4))
sns.heatmap(hm_df, annot=True, fmt=".4f", cmap="YlGnBu", linewidths=0.5)
plt.title("RSKF Summary Heatmap (5×10 = 50 folds)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rskf_summary_heatmap.png"), dpi=200, bbox_inches="tight")
plt.close()


plt.figure(figsize=(8, 6))
for v in ORDER:
    row = summary_df[summary_df["Model"] == v].iloc[0]
    plt.scatter(row["FPR_mean"], row["F1_mean"], s=200,
                color=PALETTE[v], label=v, zorder=5)
    plt.annotate(v,
                 (row["FPR_mean"], row["F1_mean"]),
                 textcoords="offset points", xytext=(8, 4), fontsize=9)

plt.xlabel("Mean FPR", fontsize=12)
plt.ylabel("Mean F1", fontsize=12)
plt.title("RSKF — FPR vs F1 Trade-off", fontsize=14, fontweight="bold")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rskf_fpr_f1_scatter.png"), dpi=200, bbox_inches="tight")
plt.close()

# 5.5 Violin plots F1
plt.figure(figsize=(10, 5))
data_f1 = pd.concat([
    fold_dfs[v][["F1"]].assign(Model=v) for v in ORDER
])
sns.violinplot(data=data_f1, x="Model", y="F1", order=ORDER,
               palette=[PALETTE[m] for m in ORDER], inner="box",
               linewidth=1.2)
plt.title("F1 Distribution per Variant (5×10 folds)", fontsize=13, fontweight="bold")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rskf_violin_f1.png"), dpi=200, bbox_inches="tight")
plt.close()
