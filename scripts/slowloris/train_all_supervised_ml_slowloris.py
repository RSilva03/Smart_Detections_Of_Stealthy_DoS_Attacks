import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.contingency_tables import mcnemar
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_supervised_slowloris')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'slowloris', 'dataset_A', 'dataset_slowloris.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


df = pd.read_csv(DATASET_PATH)
features = ['duration','pkts_total','bytes_total','pkt_size_mean',
            'tcp_bytes_total','tcp_pkts','pkts_rate','iat_mean','iat_std']
df_clean = df[features + ['label']].copy()
df_clean['iat_std'] = df_clean['iat_std'].fillna(0)
for col in features:
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

X = df_clean[features].values
y = df_clean['label'].values
print(f"Dataset: {X.shape} | Benign={np.sum(y==0)} | Attack={np.sum(y==1)}\n")


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

def fpr_score(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    return FP / (FP + TN) if (FP + TN) > 0 else 0.0

def fnr_score(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    return FN / (FN + TP) if (FN + TP) > 0 else 0.0

scorers = {
    'accuracy':  make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall':    make_scorer(recall_score),
    'f1':        make_scorer(f1_score),
    'roc_auc':   'roc_auc',
    'fpr':       make_scorer(fpr_score),
    'fnr':       make_scorer(fnr_score),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


print("=" * 75)
print("CROSS-VALIDATION — Metrics with Mean ± Std  |  Training/Inference Times")
print("=" * 75)

results       = {}
y_preds       = {}   

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])

    t0 = time.time()
    cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scorers,
                            return_train_score=False, n_jobs=-1)
    train_time = round(time.time() - t0, 3)

    pipe.fit(X, y)
    scaler_inf = StandardScaler().fit(X)
    X_inf = scaler_inf.transform(X[:555])
    t1 = time.time()
    for _ in range(100):
        pipe.predict(X_inf)
    infer_time_ms = round((time.time() - t1) / 100 * 1000, 3)

    y_pred_oof = np.zeros_like(y)
    for tr, te in cv.split(X, y):
        pipe2 = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        pipe2.fit(X[tr], y[tr])
        y_pred_oof[te] = pipe2.predict(X[te])
    y_preds[name] = y_pred_oof

    results[name] = {
        'accuracy':   cv_res['test_accuracy'].mean(),
        'precision':  cv_res['test_precision'].mean(),
        'recall':     cv_res['test_recall'].mean(),
        'f1':         cv_res['test_f1'].mean(),
        'roc_auc':    cv_res['test_roc_auc'].mean(),
        'fpr':        cv_res['test_fpr'].mean(),
        'fnr':        cv_res['test_fnr'].mean(),
        'f1_std':     cv_res['test_f1'].std(),
        'auc_std':    cv_res['test_roc_auc'].std(),
        'fpr_std':    cv_res['test_fpr'].std(),
        'fnr_std':    cv_res['test_fnr'].std(),
        'train_time_s':    train_time,
        'infer_time_ms':   infer_time_ms,
    }

    r = results[name]
    print(f"\n{name}")
    print(f"  Acc={r['accuracy']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")
    print(f"  F1={r['f1']:.4f} ± {r['f1_std']:.4f}  |  AUC={r['roc_auc']:.4f} ± {r['auc_std']:.4f}")
    print(f"  FPR={r['fpr']:.4f} ± {r['fpr_std']:.4f}  |  FNR={r['fnr']:.4f} ± {r['fnr_std']:.4f}")
    print(f"  Train time: {r['train_time_s']}s  |  Inference: {r['infer_time_ms']}ms/batch")

results_df = pd.DataFrame(results).T.round(4)
results_df.index.name = 'Model'
results_csv = os.path.join(RESULTS_DIR, 'supervised_results_slowloris.csv')
results_df.to_csv(results_csv)
print(f"\n  Saved: {results_csv}\n")

print("=" * 75)
print("HYPERPARAMETER TUNING (GridSearchCV 5-Fold)")
print("=" * 75)

tuning_configs = {
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'clf__C':     [0.1, 1, 10, 100],
            'clf__gamma': ['scale', 'auto', 0.01, 0.1],
            'clf__kernel':['rbf', 'linear'],
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'clf__n_neighbors': [3, 5, 7, 11],
            'clf__weights':     ['uniform', 'distance'],
            'clf__metric':      ['euclidean', 'manhattan'],
        }
    },
    'MLP': {
        'model': MLPClassifier(max_iter=500, random_state=42),
        'params': {
            'clf__hidden_layer_sizes': [(64,32), (128,64), (64,64,32)],
            'clf__alpha':              [0.0001, 0.001, 0.01],
            'clf__learning_rate_init': [0.001, 0.01],
        }
    },
}

tuning_results = {}
for name, cfg in tuning_configs.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', cfg['model'])])
    grid = GridSearchCV(pipe, cfg['params'], cv=cv,
                        scoring='f1', n_jobs=-1, refit=True)
    t0 = time.time()
    grid.fit(X, y)
    elapsed = round(time.time() - t0, 2)

    tuning_results[name] = {
        'best_params': grid.best_params_,
        'best_f1':     round(grid.best_score_, 4),
        'time_s':      elapsed,
    }
    print(f"\n{name} (tuned in {elapsed}s)")
    print(f"  Best F1:     {grid.best_score_:.4f}")
    print(f"  Best params: {grid.best_params_}")

print("\n  Tuning completed\n")

print("=" * 75)
print("McNEMAR'S TEST — Statistical Comparison of Classifiers")
print("=" * 75)

pairs = [
    ('RF',  'NaiveBayes'),
    ('RF',  'KNN'),
    ('RF',  'SVM'),
    ('SVM', 'NaiveBayes'),
    ('KNN', 'NaiveBayes'),
]

mcnemar_results = []
print(f"\n{'Pair':<25} {'b':>5} {'c':>5} {'p-value':>10}  {'Significant?':>12}")
print("-" * 65)

for m1, m2 in pairs:
    pred1 = y_preds[m1]
    pred2 = y_preds[m2]
    b = int(np.sum((pred1 == y) & (pred2 != y)))
    c = int(np.sum((pred1 != y) & (pred2 == y)))
    table = [[0, b], [c, 0]]
    if b + c == 0:
        p_val = 1.0
    else:
        res = mcnemar(table, exact=True)
        p_val = res.pvalue
    sig = "Yes (p<0.05)" if p_val < 0.05 else "No"
    pair_str = f"{m1} vs {m2}"
    print(f"{pair_str:<25} {b:>5} {c:>5} {p_val:>10.4f}  {sig:>12}")
    mcnemar_results.append({'pair': pair_str, 'b': b, 'c': c, 'p_value': round(p_val,4), 'significant': sig})

mcnemar_csv = os.path.join(RESULTS_DIR, 'mcnemar_results_slowloris.csv')
pd.DataFrame(mcnemar_results).to_csv(mcnemar_csv, index=False)
print(f"\n  Saved: {mcnemar_csv}\n")

print("=" * 75)
print("SHAP VALUES — Random Forest")
print("=" * 75)

scaler_shap = StandardScaler()
X_scaled = scaler_shap.fit_transform(X)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_scaled)

if isinstance(shap_values, list):
    sv_attack = shap_values[1]     
else:
    sv_attack = shap_values[:, :, 1] 

plt.figure()
shap.summary_plot(sv_attack, X_scaled,
                  feature_names=features,
                  plot_type='dot',
                  show=False)
plt.title('SHAP Summary Plot — RF  |  Slowloris Detection',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fig7_shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'fig7_shap_summary.png')}")

plt.figure()
shap.summary_plot(sv_attack, X_scaled,
                  feature_names=features,
                  plot_type='bar',
                  show=False)
plt.title('SHAP Global Feature Importance — RF',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fig8_shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'fig8_shap_bar.png')}")

print("\n  SHAP concluded")

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)
print("\nFiles generated:")
print("  results/supervised_results_slowloris.csv  — metrics + std + times")
print("  results/mcnemar_results_slowloris.csv     — statistical tests")
print("  results/plots/fig7_shap_summary.png  — SHAP dot plot")
print("  results/plots/fig8_shap_bar.png      — SHAP bar plot")
print("\nInstall dependencies if necessary:")
print("  pip install shap statsmodels lightgbm")
