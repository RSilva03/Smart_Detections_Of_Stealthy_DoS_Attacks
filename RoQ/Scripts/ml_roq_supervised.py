import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
import warnings, csv
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
FEATURES = ["n_pkts", "bytes_total", "tcp_bytes", "pkts_rate", "bytes_rate", "pkt_size_mean", 
            "pkt_size_std", "iat_mean", "iat_std", "iat_cv", "burst_peak_ratio", "burst_cv", 
            "bytes_rate_cv", "tcp_ratio"]

df = pd.read_csv("../dataset_roq_window_full.csv").dropna(subset=FEATURES + ["label"])
X = df[FEATURES].values
y = df["label"].values

print(f"Total flows : {len(df)}")
print(f"Attack (1) : {y.sum()}")
print(f"Benign (0) : {(y==0).sum()}")
print(f"Ratio : 1:{(y==0).sum()//y.sum()}\n")
print(f"CV : {N_SPLITS}-Fold Stratified")

SCALE_MODELS = ["KNN", "SVM", "MLP"]

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "LightGBM": LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "NaiveBayes": GaussianNB(),
}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

results = []
for name, model in models.items():
    fold_metrics = {k: [] for k in ["acc", "prec", "rec", "f1", "auc", "fpr", "fnr"]}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        smote = SMOTE(sampling_strategy=1.0, random_state=RANDOM_STATE, k_neighbors=min(5, y_tr.sum()-1))
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        if name in SCALE_MODELS:
            scaler = StandardScaler()
            X_tr_res = scaler.fit_transform(X_tr_res)
            X_te_sc = scaler.transform(X_te)
        else:
            X_te_sc = X_te
        
        model.fit(X_tr_res, y_tr_res)
        y_pred = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)[:,1]

        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fold_metrics["acc"].append(accuracy_score(y_te, y_pred))
        fold_metrics["prec"].append(precision_score(y_te, y_pred, zero_division=0))
        fold_metrics["rec"].append(recall_score(y_te, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
        fold_metrics["auc"].append(roc_auc_score(y_te,y_prob))
        fold_metrics["fpr"].append(fp/(fp+tn) if (fp+tn)>0 else 0)
        fold_metrics["fnr"].append(fn/(fn+tp) if (fn+tp)>0 else 0)

    row = {"Model": name}
    for metric, values in fold_metrics.items():
        arr = np.array(values)
        row[metric.upper() + "_mean"] = round(arr.mean(), 4)
        row[metric.upper() + "_std"] = round(arr.std(), 4)
    
    results.append(row)
    print(f"{name:20s} | F1={row['F1_mean']:.4f}±{row['F1_std']:.4f} | AUC={row['AUC_mean']:.4f}±{row['AUC_std']:.4f} | FPR={row['FPR_mean']:.4f}±{row['FPR_std']:.4f} | FNR={row['FNR_mean']:.4f}±{row['FNR_std']:.4f}")

results_df = pd.DataFrame(results).sort_values("F1_mean", ascending=False)
results_df.to_csv("../roq_supervised_results.csv", index=False)
print("\nGuardado em: roq_supervised_results.csv")
print("\n-- Tabela final (média ± std) --------------------")
for _, r in results_df.iterrows():
    print(f"{r['Model']:20s} | F1={r['F1_mean']:.4f}±{r['F1_std']:.4f} |"
          f" Recall={r['REC_mean']:.4f}±{r['REC_std']:.4f} |" 
          f" AUC={r['AUC_mean']:.4f}±{r['AUC_std']:.4f}")
