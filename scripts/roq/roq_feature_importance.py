"""
RoQ Feature Importance:
  1. Random Forest Feature Importance (nativa)
  2. SHAP Values (TreeExplainer)
  3. Mutual Information (estatístico)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, json, os
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_feature_importance_roq')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'roq', 'dataset_roq_window_full.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES = [
    "n_pkts", "bytes_total", "tcp_bytes", "pkts_rate", "bytes_rate", "pkt_size_mean", 
    "pkt_size_std", "iat_mean", "iat_std", "iat_cv", "burst_peak_ratio", "burst_cv", 
    "bytes_rate_cv", "tcp_ratio"
]

df = pd.read_csv(DATASET_PATH).dropna(subset=FEATURES + ["label"])
X = df[FEATURES].values
y = df["label"].values

print(f"Dataset: {len(df)} flows | Attack: {y.sum()} | Benign: {(y==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
smote = SMOTE(sampling_strategy=1.0, random_state=RANDOM_STATE,
              k_neighbors=min(5, y_train.sum() - 1))
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\n[1/3] Random Forest Feature Importance...")
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                             random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
rf_importance = rf.feature_importances_

rf_df = pd.DataFrame({
    "feature": FEATURES,
    "rf_importance": rf_importance
}).sort_values("rf_importance", ascending=False)
print(rf_df.to_string(index=False))

print("\n[2/3] SHAP Values...")
try:
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    def _extract_binary_shap_matrix(values):
        if hasattr(values, "values"):
            values = values.values

        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]

        values = np.asarray(values)

        if values.ndim == 3:
            if values.shape[-1] > 1:
                values = values[:, :, 1]
            else:
                values = np.squeeze(values, axis=-1)

        if values.ndim != 2:
            raise ValueError(f"Unexpected SHAP shape: {values.shape}")

        return values

    sv = _extract_binary_shap_matrix(shap_values)

    shap_mean = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": FEATURES,
        "shap_mean_abs": shap_mean
    }).sort_values("shap_mean_abs", ascending=False)
    print(shap_df.to_string(index=False))
    shap_ok = True
except Exception as e:
    print(f"  SHAP unavailable: {e}")
    shap_df = None
    shap_ok = False

print("\n[3/3] Mutual Information...")
mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({
    "feature": FEATURES,
    "mutual_info": mi
}).sort_values("mutual_info", ascending=False)
print(mi_df.to_string(index=False))

combined = rf_df.set_index("feature") \
    .join(mi_df.set_index("feature"), how="outer")
if shap_ok:
    combined = combined.join(shap_df.set_index("feature"), how="outer")
combined = combined.reset_index().rename(columns={"index": "feature"})
feature_csv = os.path.join(RESULTS_DIR, 'roq_feature_importance.csv')
combined.to_csv(feature_csv, index=False)
print(f"\n CSV saved: {feature_csv}")

PALETTE = ["#01696f", "#4f98a3", "#cedcd8", "#7a7974", "#28251d",
           "#437a22", "#006494", "#d19900", "#a13544"]
FIG_W, FIG_H = 9, 5

def make_hbar(df_sorted, col, xlabel, title, fname, color="#01696f"):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("#f7f6f2")
    ax.set_facecolor("#f9f8f5")

    bars = ax.barh(df_sorted["feature"], df_sorted[col],
                   color=color, edgecolor="none", height=0.6)

    for bar, val in zip(bars, df_sorted[col]):
        ax.text(bar.get_width() + df_sorted[col].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=9, color="#28251d")

    ax.set_xlabel(xlabel, fontsize=11, color="#28251d")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#28251d", pad=12)
    ax.tick_params(colors="#28251d", labelsize=10)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#dcd9d5")
    ax.invert_yaxis()
    ax.set_xlim(0, df_sorted[col].max() * 1.18)
    ax.grid(axis="x", color="#dcd9d5", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {fname}")

make_hbar(
    rf_df.sort_values("rf_importance"),
    col="rf_importance",
    xlabel="Importance (Gini)",
    title="Random Forest — Feature Importance (RoQ)",
    fname=os.path.join(PLOTS_DIR, "feat_rf_importance.png"),
    color="#01696f"
)

make_hbar(
    mi_df.sort_values("mutual_info"),
    col="mutual_info",
    xlabel="Mutual Information",
    title="Mutual Information — Relationship Feature–Label (RoQ)",
    fname=os.path.join(PLOTS_DIR, "feat_mutual_info.png"),
    color="#006494"
)

if shap_ok:
    make_hbar(
        shap_df.sort_values("shap_mean_abs"),
        col="shap_mean_abs",
        xlabel="|SHAP value| medium",
        title="SHAP — Global Feature Importance (RoQ)",
        fname=os.path.join(PLOTS_DIR, "feat_shap.png"),
        color="#437a22"
    )

print("\n  Generating comparative chart...")
norm_rf = (rf_df.set_index("feature")["rf_importance"] /
           rf_df["rf_importance"].max()).rename("RF")
norm_mi = (mi_df.set_index("feature")["mutual_info"] /
           mi_df["mutual_info"].max()).rename("MI")
comp = pd.concat([norm_rf, norm_mi], axis=1)

if shap_ok:
    norm_shap = (shap_df.set_index("feature")["shap_mean_abs"] /
                 shap_df["shap_mean_abs"].max()).rename("SHAP")
    comp = pd.concat([comp, norm_shap], axis=1)

comp["mean_rank"] = comp.mean(axis=1)
comp = comp.sort_values("mean_rank", ascending=True)

fig, ax = plt.subplots(figsize=(FIG_W + 1, FIG_H + 1))
fig.patch.set_facecolor("#f7f6f2")
ax.set_facecolor("#f9f8f5")

cols_plot = [c for c in ["RF", "SHAP", "MI"] if c in comp.columns]
colors_map = {"RF": "#01696f", "SHAP": "#437a22", "MI": "#006494"}
bar_h = 0.25
n = len(cols_plot)
y_pos = np.arange(len(comp))

for i, col in enumerate(cols_plot):
    offset = (i - n / 2 + 0.5) * bar_h
    ax.barh(y_pos + offset, comp[col], height=bar_h * 0.9,
            color=colors_map[col], label=col, edgecolor="none")

ax.set_yticks(y_pos)
ax.set_yticklabels(comp.index, fontsize=10, color="#28251d")
ax.set_xlabel("Normalized Importance (0–1)", fontsize=11, color="#28251d")
ax.set_title("Comparison of the 3 Approaches — RoQ Features\n(RF, SHAP, Mutual Information)",
             fontsize=12, fontweight="bold", color="#28251d", pad=12)
ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color("#dcd9d5")
ax.grid(axis="x", color="#dcd9d5", linewidth=0.6)
ax.set_xlim(0, 1.15)
plt.tight_layout()
feat_comparison_path = os.path.join(PLOTS_DIR, 'feat_comparison.png')
plt.savefig(feat_comparison_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {feat_comparison_path}")

print("\n CONCLUDED — files generated:")
for f in [
    feature_csv,
    os.path.join(PLOTS_DIR, "feat_rf_importance.png"),
    os.path.join(PLOTS_DIR, "feat_mutual_info.png"),
    os.path.join(PLOTS_DIR, "feat_shap.png"),
    os.path.join(PLOTS_DIR, "feat_comparison.png"),
]:
    if os.path.exists(f):
        print(f"   {f}")
