import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from matplotlib.patches import Patch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_unsupervised_slowloris')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'slowloris', 'dataset_A', 'dataset_slowloris.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

N_RUNS = 20
N_SEEDS = 10

df = pd.read_csv(DATASET_PATH)
features = ['duration','pkts_total','bytes_total','pkt_size_mean',
            'tcp_bytes_total','tcp_pkts','pkts_rate','iat_mean','iat_std']

df_clean = df[features + ['label']].copy()
df_clean['iat_std'] = df_clean['iat_std'].fillna(0)
for col in features:
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

X_all = df_clean[features].values
y_all = df_clean['label'].values
X_benign = X_all[y_all == 0]

scaler = StandardScaler()
X_benign_scaled = scaler.fit_transform(X_benign)
X_all_scaled    = scaler.transform(X_all)

def fpr_fnr(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    return FP/(FP+TN) if (FP+TN)>0 else 0, FN/(FN+TP) if (FN+TP)>0 else 0

def majority_vote_pred(km, cluster_labels, k, y_true):
    ratio = {c: y_true[cluster_labels==c].mean()
             for c in range(k) if (cluster_labels==c).sum()>0}
    attack_c = max(ratio, key=ratio.get)
    return (cluster_labels == attack_c).astype(int)

print("Dataset ready.\n")
best_k = 7

print("=" * 70)
print("K-MEANS — Metrics + Confusion + ROC + Times (mean {N_RUNS} runs)")
print("=" * 70)

km_all_results = {}

for k in [2, best_k]:
    print(f"\n--- K-Means k={k} ---")

    train_times, infer_times = [], []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_benign_scaled)
        train_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        km.predict(X_all_scaled)         
        infer_times.append(time.perf_counter() - t1)

    train_mean_s   = round(np.mean(train_times), 4)
    train_std_s    = round(np.std(train_times), 4)
    infer_mean_ms  = round(np.mean(infer_times) * 1000, 3)
    infer_std_ms   = round(np.std(infer_times) * 1000, 3)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_benign_scaled)
    cluster_labels = km.predict(X_all_scaled)

    distances = km.transform(X_all_scaled)
    min_dist  = distances.min(axis=1)
    auc_val   = roc_auc_score(y_all, min_dist)
    fpr_roc, tpr_roc, _ = roc_curve(y_all, min_dist)

    y_pred = majority_vote_pred(km, cluster_labels, k, y_all)
    fpr_v, fnr_v = fpr_fnr(y_all, y_pred)
    f1   = f1_score(y_all, y_pred)
    prec = precision_score(y_all, y_pred)
    rec  = recall_score(y_all, y_pred)
    sil  = silhouette_score(X_all_scaled, cluster_labels)
    db   = davies_bouldin_score(X_all_scaled, cluster_labels)

    print(f"  Silhouette:       {sil:.4f}")
    print(f"  Davies-Bouldin:   {db:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  F1:               {f1:.4f}")
    print(f"  ROC-AUC:          {auc_val:.4f}")
    print(f"  FPR:              {fpr_v:.4f}")
    print(f"  FNR:              {fnr_v:.4f}")
    print(f"  Train: {train_mean_s}s ± {train_std_s}s  |  "
          f"Infer: {infer_mean_ms}ms ± {infer_std_ms}ms")

    km_all_results[k] = {
        'km': km, 'cluster_labels': cluster_labels, 'y_pred': y_pred,
        'min_dist': min_dist, 'fpr_roc': fpr_roc, 'tpr_roc': tpr_roc,
        'auc': auc_val, 'f1': f1, 'precision': prec, 'recall': rec,
        'fpr': fpr_v, 'fnr': fnr_v, 'sil': sil, 'db': db,
        'train_mean_s': train_mean_s, 'train_std_s': train_std_s,
        'infer_mean_ms': infer_mean_ms, 'infer_std_ms': infer_std_ms,
    }

print(f"\n--- Stability K-Means (k={best_k}, {N_SEEDS} seeds) ---")
sil_scores, f1_scores, auc_scores = [], [], []
for seed in range(N_SEEDS):
    km_s = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    km_s.fit(X_benign_scaled)
    cl = km_s.predict(X_all_scaled)
    dist_s = km_s.transform(X_all_scaled).min(axis=1)
    yp_s = majority_vote_pred(km_s, cl, best_k, y_all)
    sil_scores.append(silhouette_score(X_all_scaled, cl))
    f1_scores.append(f1_score(y_all, yp_s))
    auc_scores.append(roc_auc_score(y_all, dist_s))

km_f1_std  = round(np.std(f1_scores), 4)
km_auc_std = round(np.std(auc_scores), 4)
print(f"\n f1_scores per seed: {[round(v,4) for v in f1_scores]}")
print(f"  auc_socres per seed: {[round(v,4) for v in auc_scores]}")
print(f"  sil_scores per seed: {[round(v,4) for v in sil_scores]}")
print(f"  Silhouette: {np.mean(sil_scores):.4f} ± {np.std(sil_scores):.4f}")
print(f"  F1:         {np.mean(f1_scores):.4f} ± {km_f1_std}")
print(f"  ROC-AUC:    {np.mean(auc_scores):.4f} ± {km_auc_std}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, k in enumerate([2, best_k]):
    cm = confusion_matrix(y_all, km_all_results[k]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Benign','Slowloris'],
                yticklabels=['Benign','Slowloris'],
                linewidths=0.5, annot_kws={"size":14,"weight":"bold"})
    r = km_all_results[k]
    axes[i].set_title(
        f'K-Means (k={k})\nF1={r["f1"]:.4f}  AUC={r["auc"]:.4f}  '
        f'FPR={r["fpr"]:.4f}  FNR={r["fnr"]:.4f}',
        fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
plt.suptitle('K-Means Confusion Matrices  |  Slowloris Detection',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'kmeans_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
for (k, res), color in zip(km_all_results.items(), ['#1f77b4','#ff7f0e']):
    ax.plot(res['fpr_roc'], res['tpr_roc'], color=color, lw=2.5,
            label=f'K-Means k={k}  (AUC={res["auc"]:.4f})')
ax.plot([0,1],[0,1],'k--',lw=1.2,label='Random Classifier')
ax.fill_between([0,1],[0,1],alpha=0.04,color='gray')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — K-Means (score: distance to centroid)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'kmeans_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {os.path.join(PLOTS_DIR, 'kmeans_confusion_matrices.png')} + {os.path.join(PLOTS_DIR, 'kmeans_roc_curves.png')}")


print("\n" + "=" * 70)
print(f"ISOLATION FOREST — Confusion + Stability ({N_SEEDS} seeds) + Times")
print("=" * 70)

if_metrics = {'f1': [], 'auc': [], 'fpr': [], 'fnr': [],
              'train_s': [], 'infer_ms': []}

for seed in range(N_SEEDS):
    t0 = time.perf_counter()
    iforest = IsolationForest(n_estimators=200, contamination=0.05,
                              random_state=seed, n_jobs=-1)
    iforest.fit(X_benign_scaled)
    if_metrics['train_s'].append(time.perf_counter() - t0)

    t1 = time.perf_counter()
    raw_pred = iforest.predict(X_all_scaled)
    if_metrics['infer_ms'].append((time.perf_counter() - t1) * 1000)

    y_pred_s = (raw_pred == -1).astype(int)
    scores_s = -iforest.score_samples(X_all_scaled)
    if_metrics['f1'].append(f1_score(y_all, y_pred_s))
    if_metrics['auc'].append(roc_auc_score(y_all, scores_s))
    fpr_s, fnr_s = fpr_fnr(y_all, y_pred_s)
    if_metrics['fpr'].append(fpr_s)
    if_metrics['fnr'].append(fnr_s)
    print(f"  seed={seed:2d} | F1={if_metrics['f1'][-1]:.4f}  "
          f"AUC={if_metrics['auc'][-1]:.4f}  "
          f"FPR={if_metrics['fpr'][-1]:.4f}  "
          f"FNR={if_metrics['fnr'][-1]:.4f}")

if_f1_mean  = round(np.mean(if_metrics['f1']), 4)
if_f1_std   = round(np.std(if_metrics['f1']), 4)
if_auc_mean = round(np.mean(if_metrics['auc']), 4)
if_auc_std  = round(np.std(if_metrics['auc']), 4)
if_fpr_mean = round(np.mean(if_metrics['fpr']), 4)
if_fpr_std  = round(np.std(if_metrics['fpr']), 4)
if_fnr_mean = round(np.mean(if_metrics['fnr']), 4)
if_fnr_std  = round(np.std(if_metrics['fnr']), 4)
if_train_mean = round(np.mean(if_metrics['train_s']), 4)
if_train_std  = round(np.std(if_metrics['train_s']), 4)
if_infer_mean = round(np.mean(if_metrics['infer_ms']), 3)
if_infer_std  = round(np.std(if_metrics['infer_ms']), 3)

print(f"\n  Summary (mean ± std):")
print(f"    F1:    {if_f1_mean} ± {if_f1_std}")
print(f"    AUC:   {if_auc_mean} ± {if_auc_std}")
print(f"    FPR:   {if_fpr_mean} ± {if_fpr_std}")
print(f"    FNR:   {if_fnr_mean} ± {if_fnr_std}")
print(f"    Train: {if_train_mean}s ± {if_train_std}s")
print(f"    Infer: {if_infer_mean}ms ± {if_infer_std}ms")

iforest_final = IsolationForest(n_estimators=200, contamination=0.05,
                                random_state=42, n_jobs=-1)
iforest_final.fit(X_benign_scaled)
raw_pred_final  = iforest_final.predict(X_all_scaled)
y_pred_if       = (raw_pred_final == -1).astype(int)
if_scores_final = -iforest_final.score_samples(X_all_scaled)
auc_if          = roc_auc_score(y_all, if_scores_final)
fpr_if_roc, tpr_if_roc, _ = roc_curve(y_all, if_scores_final)
fpr_if_v, fnr_if_v = fpr_fnr(y_all, y_pred_if)
f1_if   = f1_score(y_all, y_pred_if)
prec_if = precision_score(y_all, y_pred_if)
rec_if  = recall_score(y_all, y_pred_if)

fig, ax = plt.subplots(figsize=(6, 5))
cm_if = confusion_matrix(y_all, y_pred_if)
sns.heatmap(cm_if, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['Benign','Slowloris'],
            yticklabels=['Benign','Slowloris'],
            linewidths=0.5, annot_kws={"size":14,"weight":"bold"})
ax.set_title(
    f'Isolation Forest (contamination=0.05)\n'
    f'F1={f1_if:.4f}  AUC={auc_if:.4f}  '
    f'FPR={fpr_if_v:.4f}  FNR={fnr_if_v:.4f}',
    fontsize=11, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'iforest_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {os.path.join(PLOTS_DIR, 'iforest_confusion_matrix.png')}")


print("\n" + "=" * 70)
print("ROC UNIFIED — Supervised + Unsupervised")
print("=" * 70)

supervised_models = {
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
sup_roc = {}

for name, model in supervised_models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    y_proba_oof = np.zeros(len(y_all))
    for tr, te in cv.split(X_all, y_all):
        pipe.fit(X_all[tr], y_all[tr])
        y_proba_oof[te] = pipe.predict_proba(X_all[te])[:, 1]
    fpr_s, tpr_s, _ = roc_curve(y_all, y_proba_oof)
    auc_s = roc_auc_score(y_all, y_proba_oof)
    sup_roc[name] = {'fpr': fpr_s, 'tpr': tpr_s, 'auc': auc_s}
    print(f"  {name}: AUC={auc_s:.4f}")

fig, ax = plt.subplots(figsize=(11, 8))
sup_colors = ['#1f77b4','#aec7e8','#2ca02c','#98df8a',
              '#9467bd','#c5b0d5','#8c564b','#c49c94']
for (name, roc), color in zip(sup_roc.items(), sup_colors):
    ax.plot(roc['fpr'], roc['tpr'], color=color, lw=1.5, linestyle='--', alpha=0.8,
            label=f'{name} (AUC={roc["auc"]:.4f})')
ax.plot(km_all_results[2]['fpr_roc'], km_all_results[2]['tpr_roc'],
        color='#ff7f0e', lw=2.5,
        label=f'K-Means k=2 (AUC={km_all_results[2]["auc"]:.4f})')
ax.plot(km_all_results[best_k]['fpr_roc'], km_all_results[best_k]['tpr_roc'],
        color='#d62728', lw=2.5,
        label=f'K-Means k={best_k} (AUC={km_all_results[best_k]["auc"]:.4f})')
ax.plot(fpr_if_roc, tpr_if_roc,
        color='#000000', lw=2.5,
        label=f'Isolation Forest (AUC={auc_if:.4f})')
ax.plot([0,1],[0,1],'k--',lw=1.0,alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — Supervised vs Unsupervised  |  Slowloris Detection\n'
             '(Unsupervised: solid lines  |  Supervised: dashed lines)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_unified_sup_vs_unsup.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {os.path.join(PLOTS_DIR, 'roc_unified_sup_vs_unsup.png')}")


records = []
for k in [2, best_k]:
    r = km_all_results[k]
    records.append({
        'Model':             f'K-Means (k={k})',
        'Type':              'Clustering',
        'Precision':         round(r['precision'], 4),
        'Recall':            round(r['recall'], 4),
        'F1':                round(r['f1'], 4),
        'F1_std':            km_f1_std if k == best_k else '-',
        'ROC_AUC':           round(r['auc'], 4),
        'AUC_std':           km_auc_std if k == best_k else '-',
        'FPR':               round(r['fpr'], 4),
        'FNR':               round(r['fnr'], 4),
        'Silhouette':        round(r['sil'], 4),
        'Davies_Bouldin':    round(r['db'], 4),
        'Train_time_s':      f"{r['train_mean_s']} ± {r['train_std_s']}",
        'Infer_time_ms':     f"{r['infer_mean_ms']} ± {r['infer_std_ms']}",
        'Note': 'Majority vote | Trained on benign only',
    })

records.append({
    'Model':             'Isolation Forest',
    'Type':              'Anomaly Detection',
    'Precision':         round(prec_if, 4),
    'Recall':            round(rec_if, 4),
    'F1':                if_f1_mean,
    'F1_std':            if_f1_std,
    'ROC_AUC':           if_auc_mean,
    'AUC_std':           if_auc_std,
    'FPR':               if_fpr_mean,
    'FNR':               if_fnr_mean,
    'Silhouette':        '-',
    'Davies_Bouldin':    '-',
    'Train_time_s':      f"{if_train_mean} ± {if_train_std}",
    'Infer_time_ms':     f"{if_infer_mean} ± {if_infer_std}",
    'Note': 'contamination=0.05 | Trained on benign only | mean±std de 10 seeds',
})

df_unsup = pd.DataFrame(records)
df_unsup.to_csv(os.path.join(RESULTS_DIR, 'unsupervised_results_slowloris.csv'), index=False)
print(f"\n  {os.path.join(RESULTS_DIR, 'unsupervised_results_slowloris.csv')} updated.")
print(df_unsup[['Model','F1','F1_std','ROC_AUC','AUC_std',
                'Train_time_s','Infer_time_ms']].to_string(index=False))

print("\n Files generated:")
print(f"    {os.path.join(PLOTS_DIR, 'kmeans_confusion_matrices.png')}")
print(f"    {os.path.join(PLOTS_DIR, 'kmeans_roc_curves.png')}")
print(f"    {os.path.join(PLOTS_DIR, 'iforest_confusion_matrix.png')}")
print(f"    {os.path.join(PLOTS_DIR, 'roc_unified_sup_vs_unsup.png')}")
print(f"    {os.path.join(RESULTS_DIR, 'unsupervised_results_slowloris.csv')}")