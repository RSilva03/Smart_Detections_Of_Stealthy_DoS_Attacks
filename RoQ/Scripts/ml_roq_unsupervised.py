import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, silhouette_score
)

C_TEAL  = "#01696f"; C_BLUE  = "#006494"; C_GREEN = "#437a22"
C_RED   = "#a13544"; C_GOLD  = "#d19900"; C_PURPLE= "#7a39bb"
C_BG    = "#f7f6f2"; C_SURF  = "#f9f8f5"; C_TEXT  = "#28251d"
C_MUTED = "#7a7974"; C_DIV   = "#dcd9d5"
COLORS_MODELS = [C_TEAL, C_BLUE, C_GREEN]
COLORS_AE     = [C_TEAL, C_BLUE, C_GREEN] 

RANDOM_STATE = 42
RECALL_MIN   = 0.95 

ALL_FEATURES = [
    "n_pkts", "bytes_total", "tcp_bytes", "pkts_rate", "bytes_rate",
    "pkt_size_mean", "pkt_size_std", "iat_mean", "iat_std", "iat_cv",
    "burst_peak_ratio", "burst_cv", "bytes_rate_cv", "tcp_ratio"
]
TOP_FEATURES = [
    "pkt_size_std", "tcp_ratio", "pkt_size_mean", "burst_cv", "burst_peak_ratio"
]

def compute_metrics(y_true, y_pred, y_score, model_name=""):
    cm_v = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm_v.ravel()
    return {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC":       round(roc_auc_score(y_true, y_score), 4),
        "FPR":       round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        "FNR":       round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

def styled_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C_SURF)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(C_DIV)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", color=C_TEXT, pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=C_MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=C_MUTED)
    ax.grid(color=C_DIV, linewidth=0.5, linestyle="--")

def save_fig(fig, fname):
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {fname}")

def get_best_thresholds(errors, y_true, recall_min=RECALL_MIN):
    """Sweep percentil 50–99.75; retorna (thresh_maxf1, thresh_lowfnr, sweep_df)"""
    rows = []
    for pct in np.arange(50, 100, 0.25):
        t    = np.percentile(errors, pct)
        pred = (errors > t).astype(int)
        f1   = f1_score(y_true, pred, zero_division=0)
        rec  = recall_score(y_true, pred, zero_division=0)
        cm_v = confusion_matrix(y_true, pred).ravel()
        tn_, fp_, fn_, tp_ = cm_v
        fpr  = fp_ / (fp_ + tn_) if (fp_ + tn_) > 0 else 0
        rows.append({"pct": pct, "thresh": t, "F1": f1, "Recall": rec, "FPR": fpr})
    df_s = pd.DataFrame(rows)
    t_f1  = df_s.loc[df_s["F1"].idxmax(), "thresh"]
    cands = df_s[df_s["Recall"] >= recall_min]
    if len(cands) == 0:
        cands = df_s[df_s["Recall"] == df_s["Recall"].max()]
    t_fnr = cands.loc[cands["FPR"].idxmin(), "thresh"]
    return t_f1, t_fnr, df_s

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH  = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  PyTorch {torch.__version__} | device: {DEVICE}")
except ImportError:
    TORCH = False
    print("  [AVISO] PyTorch não disponível — fallback MLPRegressor")

if TORCH:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout=0.0):
            super().__init__()
            dims = [input_dim] + hidden_dims + [input_dim]
            mid  = len(hidden_dims) // 2 + 1
            enc, dec = [], []
            for i in range(mid):
                enc += [nn.Linear(dims[i], dims[i+1]),
                        nn.BatchNorm1d(dims[i+1]), nn.ReLU()]
                if dropout > 0:
                    enc.append(nn.Dropout(dropout))
            for i in range(mid, len(dims) - 1):
                dec += [nn.Linear(dims[i], dims[i+1])]
                if i < len(dims) - 2:
                    dec += [nn.BatchNorm1d(dims[i+1]), nn.ReLU()]
                    if dropout > 0:
                        dec.append(nn.Dropout(dropout))
            self.encoder = nn.Sequential(*enc)
            self.decoder = nn.Sequential(*dec)
        def forward(self, x):
            return self.decoder(self.encoder(x))

def train_autoencoder(X_benign, X_all, hidden, dropout=0.0,
                      epochs=100, batch=64, lr=1e-3, label="AE"):
    input_dim = X_benign.shape[1]
    if TORCH:
        model = Autoencoder(input_dim, hidden, dropout).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        loader  = DataLoader(
            TensorDataset(torch.tensor(X_benign, dtype=torch.float32),
                          torch.tensor(X_benign, dtype=torch.float32)),
            batch_size=batch, shuffle=True
        )
        losses = []
        model.train()
        for ep in range(epochs):
            ep_loss = 0
            for xb, _ in loader:
                xb = xb.to(DEVICE)
                opt.zero_grad()
                l = loss_fn(model(xb), xb)
                l.backward(); opt.step()
                ep_loss += l.item()
            losses.append(ep_loss / len(loader))
            if (ep + 1) % 25 == 0:
                print(f"    [{label}] Epoch {ep+1:3d}/{epochs} | Loss: {losses[-1]:.6f}")
        model.eval()
        with torch.no_grad():
            X_t   = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
            recon = model(X_t).cpu().numpy()
    else:
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=tuple(hidden), max_iter=200,
                           random_state=RANDOM_STATE)
        mlp.fit(X_benign, X_benign)
        recon  = mlp.predict(X_all)
        losses = None
    errors = np.mean((X_all - recon) ** 2, axis=1)
    return errors, losses


print("=" * 65)
print("  RoQ Unsupervised Detection Pipeline (v2 — AE Variants)")
print("=" * 65)

df = pd.read_csv(".././dataset_roq_window_full.csv")
avail_all = [f for f in ALL_FEATURES if f in df.columns]
avail_top = [f for f in TOP_FEATURES if f in df.columns]
miss = [f for f in ALL_FEATURES if f not in df.columns]
if miss: print(f"  [AVISO] Features não encontradas (ignoradas): {miss}")

df = df.dropna(subset=avail_all + ["label"])
X_all_raw = df[avail_all].values
X_top_raw = df[avail_top].values
y         = df["label"].values

print(f"  Flows : {len(df)} | Attack: {y.sum()} | Benign: {(y==0).sum()}")

scaler_all = StandardScaler()
X_all_sc   = scaler_all.fit_transform(X_all_raw)
scaler_top = StandardScaler()
X_top_sc   = scaler_top.fit_transform(X_top_raw)

pca2   = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca  = pca2.fit_transform(X_all_sc)

results = []


print("\n[1/3] K-Means ...")
inertias, silhouettes, k_range = [], [], range(2, 9)
for k in k_range:
    km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    lbs = km.fit_predict(X_all_sc)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_all_sc, lbs,
                                        sample_size=min(5000, len(X_all_sc))))

best_k     = list(k_range)[np.argmax(silhouettes)]
km_final   = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
km_labels  = km_final.fit_predict(X_all_sc)

cluster_attack_rate = [y[km_labels == c].mean() for c in range(best_k)]
attack_cluster = np.argmax(cluster_attack_rate)
km_pred  = (km_labels == attack_cluster).astype(int)
km_score = -km_final.transform(X_all_sc)[:, attack_cluster]

km_m = compute_metrics(y, km_pred, km_score, "KMeans")
results.append(km_m)
print(f"  k={best_k} | F1={km_m['F1']} | AUC={km_m['AUC']} | "
      f"Recall={km_m['Recall']} | FPR={km_m['FPR']}")

print("\n[2/3] Isolation Forest ...")
sweep_rows = []
for c in np.arange(0.02, 0.51, 0.02):
    ifo      = IsolationForest(contamination=c, n_estimators=200,
                                random_state=RANDOM_STATE, n_jobs=-1)
    ifo_pred = (ifo.fit_predict(X_all_sc) == -1).astype(int)
    sweep_rows.append({
        "contamination": round(c, 3),
        "F1":        round(f1_score(y, ifo_pred, zero_division=0), 4),
        "Precision": round(precision_score(y, ifo_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y, ifo_pred, zero_division=0), 4),
    })

sweep_df    = pd.DataFrame(sweep_rows)
sweep_df.to_csv("../unsup_iforest_sweep.csv", index=False)
best_contam = sweep_df.loc[sweep_df["F1"].idxmax(), "contamination"]

ifo_final = IsolationForest(contamination=best_contam, n_estimators=300,
                             random_state=RANDOM_STATE, n_jobs=-1)
ifo_final.fit(X_all_sc)
ifo_pred  = (ifo_final.predict(X_all_sc) == -1).astype(int)
ifo_score = -ifo_final.decision_function(X_all_sc)

ifo_m = compute_metrics(y, ifo_pred, ifo_score, "IsolationForest")
results.append(ifo_m)
print(f"  contam={best_contam} | F1={ifo_m['F1']} | AUC={ifo_m['AUC']} | "
      f"Recall={ifo_m['Recall']} | FPR={ifo_m['FPR']}")


print("\n[3/3] Autoencoder — 3 variantes ...")

AE_VARIANTS = [
    {"name": "V1 — Original",       "X": X_all_sc, "hidden": [64,32,16,32,64], "dropout": 0.0},
    {"name": "V2 — Top Features",   "X": X_top_sc, "hidden": [32,16, 8,16,32], "dropout": 0.0},
    {"name": "V3 — BN+Dropout",     "X": X_all_sc, "hidden": [32,16, 8,16,32], "dropout": 0.2},
]

ae_variant_results = []
best_ae_f1      = -1
best_ae_variant = None

for var in AE_VARIANTS:
    print(f"\n  → {var['name']}")
    X_sc    = var["X"]
    X_ben   = X_sc[y == 0]
    errors, _ = train_autoencoder(
        X_ben, X_sc,
        hidden=var["hidden"], dropout=var["dropout"],
        epochs=100, label=var["name"]
    )

    t_f1, t_fnr, sweep_v = get_best_thresholds(errors, y)

    
    pred_f1 = (errors > t_f1).astype(int)
    m_f1    = compute_metrics(y, pred_f1, errors, var["name"])
    m_f1["Threshold"] = "max-F1"

    
    pred_fnr = (errors > t_fnr).astype(int)
    m_fnr    = compute_metrics(y, pred_fnr, errors, var["name"])
    m_fnr["Threshold"] = f"Recall≥{RECALL_MIN}"

    ae_variant_results.append(m_f1)
    ae_variant_results.append(m_fnr)

    print(f"    max-F1  → F1={m_f1['F1']} | AUC={m_f1['AUC']} | "
          f"FNR={m_f1['FNR']} | FPR={m_f1['FPR']}")
    print(f"    low-FNR → F1={m_fnr['F1']} | AUC={m_fnr['AUC']} | "
          f"FNR={m_fnr['FNR']} | FPR={m_fnr['FPR']}")
    
    var["errors"]   = errors
    var["t_f1"]     = t_f1
    var["t_fnr"]    = t_fnr
    var["sweep"]    = sweep_v
    var["pred_f1"]  = pred_f1
    var["pred_fnr"] = pred_fnr
    var["m_f1"]     = m_f1
    var["m_fnr"]    = m_fnr

    if m_f1["F1"] > best_ae_f1:
        best_ae_f1      = m_f1["F1"]
        best_ae_variant = var

ae_vars_df = pd.DataFrame(ae_variant_results)
ae_vars_df.to_csv("../unsup_ae_variants_results.csv", index=False)

best_name = best_ae_variant["name"]
print(f"\n Melhor variante AE: {best_name} (F1={best_ae_f1})")

ae_errors_best = best_ae_variant["errors"]
ae_pred_best   = best_ae_variant["pred_f1"]
ae_m_best      = dict(best_ae_variant["m_f1"])
ae_m_best["Model"] = f"Autoencoder ({best_name})"
results.append(ae_m_best)

cols = ["Model","Accuracy","Precision","Recall","F1","AUC","FPR","FNR","TP","TN","FP","FN"]
results_df = pd.DataFrame(results)[cols]
results_df.to_csv("../roq_unsupervised_results.csv", index=False)
print("\n roq_unsupervised_results.csv guardado")
print(results_df[["Model","F1","AUC","FPR","FNR","Recall"]].to_string(index=False))


print("\nA gerar gráficos...")

model_names  = ["KMeans", "IsolationForest", f"Autoencoder\n({best_name})"]
preds_list   = [km_pred, ifo_pred, ae_pred_best]
scores_list  = [km_score, ifo_score, ae_errors_best]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("Matrizes de Confusão — Modelos Não Supervisionados (RoQ)",
             fontsize=13, fontweight="bold", color=C_TEXT, y=1.02)

for ax, name, pred, color in zip(axes, model_names, preds_list, COLORS_MODELS):
    cm_v = confusion_matrix(y, pred)
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", [C_SURF, color])
    ax.imshow(cm_v, cmap=cmap, aspect="auto")
    for i in range(2):
        for j in range(2):
            v = cm_v[i, j]
            ax.text(j, i, f"{v:,}\n({v/cm_v.sum()*100:.1f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if v > cm_v.max() * 0.5 else C_TEXT)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred Benigno","Pred Ataque"], fontsize=8, color=C_TEXT)
    ax.set_yticklabels(["Real Benigno","Real Ataque"], fontsize=8, color=C_TEXT,
                       rotation=90, va="center")
    ax.set_title(name, fontsize=10, fontweight="bold", color=C_TEXT, pad=8)
    ax.spines[:].set_color(C_DIV)

plt.tight_layout()
save_fig(fig, "../unsup_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_SURF)
ax.plot([0,1],[0,1], "--", color=C_MUTED, linewidth=1, label="Random (AUC=0.50)")

roc_names = ["KMeans", "IsolationForest", f"Autoencoder ({best_name})"]
for name, score, color in zip(roc_names, scores_list, COLORS_MODELS):
    fpr_a, tpr_a, _ = roc_curve(y, score)
    auc_v = roc_auc_score(y, score)
    ax.plot(fpr_a, tpr_a, color=color, linewidth=2.5,
            label=f"{name} (AUC={auc_v:.4f})")

ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("False Positive Rate", fontsize=10, color=C_MUTED)
ax.set_ylabel("True Positive Rate",  fontsize=10, color=C_MUTED)
ax.set_title("Curvas ROC — Modelos Não Supervisionados (RoQ)",
             fontsize=11, fontweight="bold", color=C_TEXT, pad=10)
ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
ax.tick_params(colors=C_TEXT)
ax.grid(color=C_DIV, linewidth=0.5, linestyle="--")
ax.spines[["top","right"]].set_visible(False)
ax.spines[["left","bottom"]].set_color(C_DIV)
plt.tight_layout()
save_fig(fig, "../unsup_roc_curves.png")

titles_pca = ["K-Means (clusters)", "Isolation Forest (anomalias)",
              f"Autoencoder — {best_name}"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("PCA 2D — Separação Ataque vs Benigno (RoQ)",
             fontsize=13, fontweight="bold", color=C_TEXT, y=1.02)

for ax, pred, title in zip(axes, preds_list, titles_pca):
    idx = np.random.choice(len(X_pca), min(3000, len(X_pca)), replace=False)
    xp, yp, pp, yt = X_pca[idx,0], X_pca[idx,1], pred[idx], y[idx]
    mask_b = pp == 0
    ax.scatter(xp[mask_b], yp[mask_b], c=C_SURF, edgecolors=C_DIV,
               s=8, linewidths=0.3, alpha=0.6, label="Pred Benigno", zorder=1)
    correct = (pp == yt)
    tp_m = (pp == 1) & correct
    fp_m = (pp == 1) & ~correct
    ax.scatter(xp[tp_m], yp[tp_m], c=C_GREEN, s=20, alpha=0.8, label="TP", zorder=3)
    ax.scatter(xp[fp_m], yp[fp_m], c=C_RED,   s=20, alpha=0.8, label="FP", zorder=3)
    var_e = pca2.explained_variance_ratio_
    styled_ax(ax, title=title,
              xlabel=f"PC1 ({var_e[0]*100:.1f}%)",
              ylabel=f"PC2 ({var_e[1]*100:.1f}%)")
    ax.legend(fontsize=7, markerscale=1.5, framealpha=0.8)
plt.tight_layout()
save_fig(fig, "../unsup_pca_clusters.png")

best_t_f1  = best_ae_variant["t_f1"]
best_sweep = best_ae_variant["sweep"]
best_pct   = best_sweep.loc[best_sweep["F1"].idxmax(), "pct"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(C_BG)
fig.suptitle(f"Threshold Sweep — IForest e Autoencoder [{best_name}] (RoQ)",
             fontsize=12, fontweight="bold", color=C_TEXT, y=1.02)

for ax, df_s, xcol, xlabel, best_val, color, title in [
    (ax1, sweep_df,    "contamination", "Contamination", best_contam, C_BLUE,  "Isolation Forest"),
    (ax2, best_sweep,  "pct",           "Percentil (%)", best_pct,   C_GREEN, f"Autoencoder — {best_name}"),
]:
    ax.set_facecolor(C_SURF)
    ax.plot(df_s[xcol], df_s["F1"],       color=color, linewidth=2,   label="F1")
    if "Precision" in df_s.columns:
        ax.plot(df_s[xcol], df_s["Precision"],color=C_GOLD, linewidth=1.5, linestyle="--", label="Precision")
    ax.plot(df_s[xcol], df_s["Recall"],   color=C_RED,  linewidth=1.5, linestyle=":",  label="Recall")
    ax.axvline(x=best_val, color=C_TEXT, linewidth=1, linestyle="-.", alpha=0.7,
               label=f"Melhor ({xcol}={best_val:.2f})")
    styled_ax(ax, title=title, xlabel=xlabel, ylabel="Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, framealpha=0.9)
plt.tight_layout()
save_fig(fig, "../unsup_threshold_sweep.png")

best_thresh = best_ae_variant["t_f1"]
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_SURF)
cap = np.percentile(ae_errors_best, 99)
errs_plot = np.clip(ae_errors_best, 0, cap)
ax.hist(errs_plot[y==0], bins=80, alpha=0.6, color=C_BLUE, label="Benigno", density=True)
ax.hist(errs_plot[y==1], bins=80, alpha=0.7, color=C_RED,  label="Ataque",  density=True)
ax.axvline(x=min(best_thresh, cap), color=C_TEXT, linewidth=1.5, linestyle="-.",
           label=f"Threshold max-F1 ({best_thresh:.4f})")
ax.axvline(x=min(best_ae_variant["t_fnr"], cap), color=C_GREEN, linewidth=1.5, linestyle=":",
           label=f"Threshold Recall≥{RECALL_MIN} ({best_ae_variant['t_fnr']:.4f})")
styled_ax(ax, title=f"Autoencoder — Distribuição Erro de Reconstrução [{best_name}] (RoQ)",
          xlabel="Erro MSE", ylabel="Densidade")
ax.legend(fontsize=9, framealpha=0.9)
plt.tight_layout()
save_fig(fig, "../unsup_ae_reconstruction.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("K-Means — Seleção do Número de Clusters (RoQ)",
             fontsize=12, fontweight="bold", color=C_TEXT, y=1.02)
for ax, vals, ylabel, title, color in [
    (ax1, inertias,   "Inércia (WCSS)",  "Elbow Method",      C_TEAL),
    (ax2, silhouettes,"Silhouette Score","Silhouette Analysis",C_BLUE),
]:
    ax.plot(list(k_range), vals, "o-", color=color, linewidth=2, markersize=7)
    ax.axvline(x=best_k, color=C_TEXT, linewidth=1, linestyle="-.", alpha=0.7,
               label=f"k={best_k} (escolhido)")
    styled_ax(ax, title=title, xlabel="Número de clusters (k)", ylabel=ylabel)
    ax.legend(fontsize=8)
plt.tight_layout()
save_fig(fig, "../unsup_kmeans_selection.png")

metrics_show  = ["F1", "AUC", "FPR", "FNR"]
metric_colors = [C_TEAL, C_BLUE, C_RED, C_GOLD]
modes         = ["max-F1", f"Recall≥{RECALL_MIN}"]
var_labels    = ["V1\nOriginal", "V2\nTop Feat.", "V3\nBN+Dropout"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("Comparação das 3 Variantes do Autoencoder — RoQ",
             fontsize=13, fontweight="bold", color=C_TEXT, y=1.02)

for ax, mode in zip(axes, modes):
    df_m = ae_vars_df[ae_vars_df["Threshold"] == mode]
    x    = np.arange(len(var_labels))
    bw   = 0.18
    for i, (metric, color) in enumerate(zip(metrics_show, metric_colors)):
        vals = df_m[metric].values
        bars = ax.bar(x + i*bw - 1.5*bw, vals, width=bw*0.85,
                      color=color, label=metric, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=C_TEXT)
    ax.set_xticks(x); ax.set_xticklabels(var_labels, fontsize=9)
    ax.set_ylim(0, 1.15)

    best_idx = [i for i, v in enumerate(AE_VARIANTS) if v["name"] == best_name]
    if best_idx:
        ax.axvspan(best_idx[0] - 0.45, best_idx[0] + 0.45, alpha=0.07,
                   color=C_GREEN, zorder=0)
        ax.text(best_idx[0], 1.10, "★ Melhor", ha="center", fontsize=8,
                color=C_GREEN, fontweight="bold")

    styled_ax(ax, title=f"Threshold: {mode}", xlabel="Variante", ylabel="Score")
    ax.legend(fontsize=8, framealpha=0.9)

plt.tight_layout()
save_fig(fig, "../unsup_ae_variants_comparison.png")


print("\n" + "=" * 65)
print(f"  Melhor Autoencoder: {best_name}")
print(f"  F1={best_ae_f1} | AUC={ae_m_best['AUC']} | "
      f"FNR={ae_m_best['FNR']} | FPR={ae_m_best['FPR']}")
print("\n  Ficheiros gerados:")
for f in [
    "roq_unsupervised_results.csv", "unsup_ae_variants_results.csv",
    "unsup_iforest_sweep.csv",
    "unsup_confusion_matrices.png", "unsup_roc_curves.png",
    "unsup_pca_clusters.png",       "unsup_threshold_sweep.png",
    "unsup_ae_reconstruction.png",  "unsup_kmeans_selection.png",
    "unsup_ae_variants_comparison.png",
]:
    status = "✓" if os.path.exists(f) else "–"
    print(f"    {status}  {f}")
print("=" * 65)