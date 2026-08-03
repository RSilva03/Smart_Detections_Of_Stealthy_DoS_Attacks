import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_hybrid_roq')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_style("whitegrid")

df_results = pd.read_csv(os.path.join(RESULTS_DIR, 'roq_hybrid_fuzzy_results.csv'))

df_results = df_results.sort_values(by=["F1-Score", "ROC-AUC"], ascending=False).reset_index(drop=True)

palette = {
    "Fuzzy + RF + Distance": "#1f77b4",
    "Fuzzy + SVM + Distance": "#ff7f0e",
    "Fuzzy + MLP + Distance": "#2ca02c",
    "Fuzzy + AE + Distance": "#d62728",
}

metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

fig, axes = plt.subplots(1, len(metrics), figsize=(24, 5))
fig.suptitle("Hybrid Fuzzy Models - Main Metrics", fontsize=16, fontweight="bold")

for ax, metric in zip(axes, metrics):
    sns.barplot(data=df_results, x="Model", y=metric, palette=[palette.get(m, "#777777") for m in df_results["Model"]], ax=ax)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=30)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(PLOTS_DIR, 'hybrid_metrics_barplot.png'), dpi=200, bbox_inches="tight")
plt.close()


heatmap_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "FPR", "FNR", "Threshold"]
df_heatmap = df_results.set_index("Model")[heatmap_cols]

plt.figure(figsize=(12, 6))
sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
plt.title("Hybrid Fuzzy Models - Metrics Heatmap", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'hybrid_metrics_heatmap.png'), dpi=200, bbox_inches="tight")
plt.close()


n_models = len(df_results)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

if n_models == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, df_results.iterrows()):
    cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False, xticklabels=["Pred Benign", "Pred Attack"], yticklabels=["True Benign", "True Attack"], ax=ax)
    ax.set_title(f"{row['Model']}\nF1={row['F1-Score']:.3f} | AUC={row['ROC-AUC']:.3f}", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'hybrid_confusion_matrices.png'), dpi=200, bbox_inches="tight")
plt.close()


radar_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
labels = radar_metrics
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for _, row in df_results.iterrows():
    values = [row[m] for m in radar_metrics]
    values += values[:1]
    color = palette.get(row["Model"], "#777777")
    ax.plot(angles, values, linewidth=2, label=row["Model"], color=color)
    ax.fill(angles, values, alpha=0.10, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.75, 1.0)
ax.set_title("Hybrid Fuzzy Models - Radar Comparison", fontsize=15, fontweight="bold", pad = 20)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.00))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'hybrid_radar_chart.png'), dpi=200, bbox_inches="tight")
plt.close()


plt.figure(figsize=(8, 6))
for _, row in df_results.iterrows():
    plt.scatter(row["FPR"], row["FNR"], s=180, color=palette.get(row["Model"], "#777777"), label=row["Model"])
    plt.text(row["FPR"] + 0.002, row["FNR"] + 0.002, row["Model"], fontsize=9)
plt.xlabel("False Positive Rate")
plt.ylabel("False Negative Rate")
plt.title("Hybrid Fuzzy Models - FPR vs FNR", fontsize=14, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'hybrid_fpr_fnr_scatter.png'), dpi=200, bbox_inches="tight")
plt.close()


plt.figure(figsize=(10, 5))
ax = sns.barplot(data=df_results, x="Model", y="Threshold", palette=[palette.get(m, "#777777") for m in df_results["Model"]])
plt.title("Optimal Threshold for Hybrid Model", fontsize=14, fontweight="bold")
plt.ylim(0, 1)
plt.xticks(rotation=25)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'hybrid_thresholds.png'), dpi=200, bbox_inches="tight")
plt.close()

