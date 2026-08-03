import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_supervised_slowloris')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'slowloris', 'dataset_A', 'dataset_slowloris.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728',
          '#9467bd','#8c564b','#e377c2','#7f7f7f']

df = pd.read_csv(DATASET_PATH)
features = ['duration','pkts_total','bytes_total','pkt_size_mean',
            'tcp_bytes_total','tcp_pkts','pkts_rate','iat_mean','iat_std']
df_clean = df[features + ['label']].copy()
df_clean['iat_std'] = df_clean['iat_std'].fillna(0)
for col in features:
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

X = df_clean[features].values
y  = df_clean['label'].values

models = {
    'KNN':        KNeighborsClassifier(n_neighbors=5),
    'SVM':        SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'RF':         RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP':        MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
    'GB':         GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LightGBM':   lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'AdaBoost':   AdaBoostClassifier(n_estimators=100, random_state=42),
    'NaiveBayes': GaussianNB(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred_all  = {name: np.zeros_like(y)    for name in models}
y_proba_all = {name: np.zeros(len(y))    for name in models}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    for train_idx, test_idx in cv.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        y_pred_all[name][test_idx]  = pipe.predict(X[test_idx])
        y_proba_all[name][test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    print(f"  {name} done")

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for i, (name, color) in enumerate(zip(models, COLORS)):
    cm = confusion_matrix(y, y_pred_all[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Benign','Slowloris'],
                yticklabels=['Benign','Slowloris'],
                linewidths=0.5, linecolor='gray',
                annot_kws={"size": 14, "weight": "bold"})
    TN, FP, FN, TP = cm.ravel()
    fpr = FP/(FP+TN) if (FP+TN) > 0 else 0
    fnr = FN/(FN+TP) if (FN+TP) > 0 else 0
    axes[i].set_title(f'{name}\nFPR={fpr:.4f}  FNR={fnr:.4f}',
                      fontsize=12, fontweight='bold', pad=8)
    axes[i].set_xlabel('Predicted', fontsize=10)
    axes[i].set_ylabel('Actual', fontsize=10)

plt.suptitle('Confusion Matrices — 5-Fold CV  |  Slowloris Detection',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fig1_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'fig1_confusion_matrices.png')}")


fig, ax = plt.subplots(figsize=(9, 7))

for (name, color) in zip(models, COLORS):
    fpr_arr, tpr_arr, _ = roc_curve(y, y_proba_all[name])
    roc_auc = auc(fpr_arr, tpr_arr)
    ax.plot(fpr_arr, tpr_arr, color=color, lw=2.5,
            label=f'{name}  (AUC = {roc_auc:.4f})')

ax.plot([0,1],[0,1], 'k--', lw=1.2, label='Random Classifier')
ax.fill_between([0,1],[0,1], alpha=0.04, color='gray')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — All Classifiers  |  5-Fold CV',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fig2_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'fig2_roc_curves.png')}")

results_df = pd.read_csv(os.path.join(RESULTS_DIR, 'supervised_results_slowloris.csv'), index_col=0)
metric_cols = ['accuracy','precision','recall','f1','roc_auc']
metric_labels = ['Accuracy','Precision','Recall','F1','ROC-AUC']

x  = np.arange(len(results_df))
n  = len(metric_cols)
bw = 0.13
offsets = np.linspace(-(n-1)/2*bw, (n-1)/2*bw, n)

fig, ax = plt.subplots(figsize=(16, 6))
bar_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

for j, (col, lbl, bc) in enumerate(zip(metric_cols, metric_labels, bar_colors)):
    vals = results_df[col].values
    bars = ax.bar(x + offsets[j], vals, width=bw, label=lbl,
                  color=bc, alpha=0.88, edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(results_df.index, fontsize=11)
ax.set_ylim(0.990, 1.003)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Supervised Classifiers — Performance Metrics  |  5-Fold CV',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, ncol=5, framealpha=0.9)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fig3_metrics_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'fig3_metrics_comparison.png')}")

fig, ax = plt.subplots(figsize=(12, 5))
bw = 0.35
x  = np.arange(len(results_df))

bars_fpr = ax.bar(x - bw/2, results_df['fpr'].values, width=bw,
                  label='FPR (False Alarm Rate)',
                  color='#d62728', alpha=0.85, edgecolor='white')
bars_fnr = ax.bar(x + bw/2, results_df['fnr'].values, width=bw,
                  label='FNR (Miss Detection Rate)',
                  color='#ff7f0e', alpha=0.85, edgecolor='white')

for bar in bars_fpr:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.00005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=9, color='#d62728')

for bar in bars_fnr:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.00005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=9, color='#ff7f0e')

ax.set_xticks(x)
ax.set_xticklabels(results_df.index, fontsize=11)
ax.set_ylim(0, 0.010)
ax.set_ylabel('Error Rate  (lower = better)', fontsize=12)
ax.set_title('FPR & FNR — IDS Error Rates per Classifier  |  5-Fold CV',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fig4_fpr_fnr.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'fig4_fpr_fnr.png')}")

fi_data = {
    'pkts_total':0.2709,'bytes_total':0.2311,'tcp_pkts':0.1708,
    'pkt_size_mean':0.1609,'tcp_bytes_total':0.1300,'iat_std':0.0180,
    'duration':0.0092,'iat_mean':0.0091,'pkts_rate':0.0000,
}
fi = pd.Series(fi_data).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
colors_fi = ['#d62728' if v >= 0.20 else
             '#ff7f0e' if v >= 0.10 else
             '#aec7e8' for v in fi.values]

bars = ax.barh(fi.index, fi.values, color=colors_fi,
               edgecolor='white', linewidth=0.5, height=0.65)

for bar, val in zip(bars, fi.values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)

ax.set_xlim(0, 0.34)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance — Random Forest  |  Slowloris Detection',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

legend_patches = [
    mpatches.Patch(color='#d62728', label='High  (≥ 0.20)'),
    mpatches.Patch(color='#ff7f0e', label='Medium (0.10 – 0.20)'),
    mpatches.Patch(color='#aec7e8', label='Low  (< 0.10)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fig5_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'fig5_feature_importance.png')}")

all_cols    = ['accuracy','precision','recall','f1','roc_auc','fpr','fnr']
all_labels  = ['Accuracy','Precision','Recall','F1','ROC-AUC','FPR','FNR']

heatmap_data = results_df[all_cols].copy().astype(float)
heatmap_viz  = heatmap_data.copy()
heatmap_viz['fpr'] = 1 - heatmap_viz['fpr']   
heatmap_viz['fnr'] = 1 - heatmap_viz['fnr']   
heatmap_viz.columns = all_labels

fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(heatmap_viz, annot=heatmap_data.values, fmt='.4f',
            cmap='RdYlGn', ax=ax, linewidths=0.5,
            vmin=0.990, vmax=1.0,
            annot_kws={"size": 10},
            cbar_kws={'label':'Score (FPR/FNR inverted for coloring)'})

ax.set_title('Full Results Heatmap — All Classifiers × All Metrics  |  5-Fold CV\n'
             '(FPR & FNR: green = closer to 0 = better)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Classifier', fontsize=12)
ax.set_xticklabels(all_labels, fontsize=11)
ax.set_yticklabels(results_df.index, fontsize=11, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fig6_heatmap_all_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'fig6_heatmap_all_metrics.png')}")

print(f"\n  All 6 plots saved to {PLOTS_DIR}")
print("    fig1_confusion_matrices.png")
print("    fig2_roc_curves.png")
print("    fig3_metrics_comparison.png")
print("    fig4_fpr_fnr.png")
print("    fig5_feature_importance.png")
print("    fig6_heatmap_all_metrics.png")
