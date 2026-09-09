import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'roq', 'dataset_roq_window_full.csv')

FEATURES = ['n_pkts', 'bytes_total', 'tcp_bytes', 'pkts_rate', 'bytes_rate', 'pkt_size_mean', 
            'pkt_size_std', 'iat_mean', 'iat_std', 'iat_cv', 'burst_peak_ratio', 'burst_cv', 
            'bytes_rate_cv', 'tcp_ratio']

df = pd.read_csv(DATASET_PATH)
X = df[FEATURES].values
y = df['label'].values

X_trainval, X_te, y_trainval, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_val, y_tr, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SEED)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)
X_te_s = scaler.transform(X_te)

def robust_centroid(X_subset):
    Q1 = np.percentile(X_subset, 25, axis=0)
    Q3 = np.percentile(X_subset, 75, axis=0)
    IQR = Q3 - Q1
    mask = np.all((X_subset >= Q1 - 1.5*IQR) & (X_subset <= Q3 + 1.5*IQR), axis=1)
    subset_clean = X_subset[mask]
    return subset_clean.mean(axis=0, keepdims=True) if len(subset_clean) > 0 else X_subset.mean(axis=0, keepdims=True)

centroid_benign = robust_centroid(X_tr_s[y_tr == 0])
centroid_attack = robust_centroid(X_tr_s[y_tr == 1])

def distance_score_fn(X_scaled):
    db = cdist(X_scaled, centroid_benign, metric='euclidean').ravel()
    da = cdist(X_scaled, centroid_attack, metric='euclidean').ravel()
    return db / (db + da + 1e-9)

dist_val = distance_score_fn(X_val_s)
dist_te = distance_score_fn(X_te_s)

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1)

svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)

knn.fit(X_tr_s, y_tr)
rf.fit(X_tr_s, y_tr)
svm.fit(X_tr_s, y_tr)

knn_score_val = knn.predict_proba(X_val_s)[:, 1]
rf_score_val = rf.predict_proba(X_val_s)[:, 1]
svm_score_val = svm.predict_proba(X_val_s)[:, 1]
knn_score_te = knn.predict_proba(X_te_s)[:, 1]
rf_score_te = rf.predict_proba(X_te_s)[:, 1]
svm_score_te = svm.predict_proba(X_te_s)[:, 1]

class AutoencoderV1(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, n_in)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

ae = AutoencoderV1(len(FEATURES))
opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

X_tr_benign = torch.tensor(X_tr_s[y_tr == 0], dtype=torch.float32)

for _ in range(200):
    ae.train()
    opt.zero_grad()
    recon = ae(X_tr_benign)
    loss = loss_fn(recon, X_tr_benign)
    loss.backward()
    opt.step()

ae.eval()
with torch.no_grad():
    ae_mse_val = ((ae(torch.tensor(X_val_s, dtype=torch.float32))- torch.tensor(X_val_s, dtype=torch.float32))**2).mean(1).numpy()
    ae_mse_te = ((ae(torch.tensor(X_te_s, dtype=torch.float32))- torch.tensor(X_te_s, dtype=torch.float32))**2).mean(1).numpy()

ae_thresh_norm = np.percentile(ae_mse_val, 99) + 1e-9
ae_score_val = np.clip(ae_mse_val / ae_thresh_norm, 0, 1)
ae_score_te = np.clip(ae_mse_te / ae_thresh_norm, 0, 1)

def trimf(x, a, b, c):
    left = np.where((x > a) & (x <= b), (x - a) / (b - a + 1e-9), 0.0)
    right = np.where((x > b) & (x < c), (c - x) / (c -b + 1e-9), 0.0)
    peak = np.where(x == b, 1.0, 0.0)
    return np.clip(left + right + peak, 0, 1)

def trapmf(x, a, b, c, d):
    rise = np.where(x <= a, 0.0, np.where(x < b, (x - a) / (b - a + 1e-9), 1.0))
    fall = np.where(x >= d, 0.0, np.where(x > c, (d - x) / (d - c + 1e-9), 1.0))
    return np.clip(np.minimum(rise, fall), 0, 1)

def make_scaler(v_ref):
    mn, mx = v_ref.min(), v_ref.max()
    return lambda v: np.clip((v- mn) / (mx - mn + 1e-9), 0, 1)

def fuzzy_fusion(cs, ds):
    u = np.linspace(0, 1, 1000)

    cls_low = trapmf(cs, 0.0, 0.0, 0.25, 0.45)
    cls_med = trimf(cs, 0.35, 0.50, 0.65)
    cls_high = trapmf(cs, 0.55, 0.75, 1.0, 1.0)

    dist_low = trapmf(ds, 0.0, 0.0, 0.25, 0.40)
    dist_med = trimf(ds, 0.30, 0.50, 0.70)
    dist_high = trapmf(ds, 0.60, 0.75, 1.0, 1.0)

    out_benign = trapmf(u, 0.0, 0.0, 0.15, 0.30)
    out_medlow = trimf(u, 0.20, 0.35, 0.50)
    out_med = trimf(u, 0.35, 0.50, 0.65)
    out_high = trimf(u, 0.60, 0.78, 0.92)
    out_vhigh = trapmf(u, 0.82, 0.92, 1.0, 1.0)

    agg = np.zeros((len(cs), len(u)))

    R = lambda s, o: np.minimum(s[:, None], o[None, :])

    agg = np.maximum(agg, R(np.minimum(cls_high, dist_high), out_vhigh))
    agg = np.maximum(agg, R(np.minimum(cls_high, dist_med), out_high))
    agg = np.maximum(agg, R(np.minimum(cls_med, dist_high), out_high))
    agg = np.maximum(agg, R(np.minimum(cls_med, dist_med), out_med))
    agg = np.maximum(agg, R(np.minimum(cls_low, dist_low), out_benign))
    agg = np.maximum(agg, R(np.minimum(cls_low, dist_med), out_medlow))
    agg = np.maximum(agg, R(np.minimum(cls_med, dist_low), out_medlow))
    agg = np.maximum(agg, R(np.minimum(cls_high, dist_low), out_med))
    agg = np.maximum(agg, R(np.minimum(cls_low, dist_high), out_med))

    num = (agg * u[None, :]).sum(axis=1)
    den = agg.sum(axis=1) + 1e-9
    return num / den

def best_threshold_on_val(scores_val, y_val):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (scores_val >= t).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)

def evaluate_variants(name, cls_val, cls_te, dist_val, dist_te, y_val, y_te):
    scale_cls = make_scaler(cls_val)
    scale_dist = make_scaler(dist_val)
    hybrid_val = fuzzy_fusion(scale_cls(cls_val), scale_dist(dist_val))
    hybrid_te = fuzzy_fusion(scale_cls(cls_te), scale_dist(dist_te))
    t = best_threshold_on_val(hybrid_val, y_val)
    pred = (hybrid_te >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "F1-Score": f1_score(y_te, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_te, hybrid_te),
        "FPR": fp / (fp + tn + 1e-9),
        "FNR": fn / (fn + tp + 1e-9),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Threshold": round(t, 3)
    }

variants = [
    ("Fuzzy + KNN + Distance", knn_score_val, knn_score_te),
    ("Fuzzy + RF + Distance", rf_score_val, rf_score_te),
    ("Fuzzy + SVM + Distance", svm_score_val, svm_score_te),
    ("Fuzzy + AE + Distance", ae_score_val, ae_score_te),
]

results = [evaluate_variants(name, vs, ts, dist_val, dist_te, y_val, y_te) for name, vs, ts in variants]

df_results = pd.DataFrame(results).sort_values(by=["F1-Score", "ROC-AUC"], ascending=False)
results_csv = os.path.join(RESULTS_DIR, 'roq_hybrid_fuzzy_results.csv')
df_results.to_csv(results_csv, index=False)

print(df_results.to_string(index=False))
print(f"\nSaved in: {results_csv}")
