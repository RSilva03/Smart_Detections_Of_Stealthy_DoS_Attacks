import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve)

tf.random.set_seed(42)
np.random.seed(42)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]
results_dir = project_root / 'results'
plots_dir = results_dir / 'plots_autoencoder_slowloris'
dataset_path = project_root / 'dataset' / 'slowloris' / 'dataset_A' / 'dataset_slowloris.csv'
results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(dataset_path)

FEATURES = ['duration', 'pkts_total', 'bytes_total', 'pkt_size_mean',
            'tcp_bytes_total', 'tcp_pkts', 'pkts_rate', 'iat_mean', 'iat_std']


valid_feats = [f for f in FEATURES if df[f].notna().mean() > 0.01]
print(f"Used Features ({len(valid_feats)}): {valid_feats}")

df_clean = df[valid_feats + ['label']].copy()
for col in valid_feats:
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

X_all = df_clean[valid_feats].values
y_all = df_clean['label'].values
n_features = X_all.shape[1]


X_benign = X_all[y_all == 0]
X_attack = X_all[y_all == 1]


np.random.seed(42)
idx = np.random.permutation(len(X_benign))
split = int(0.8 * len(X_benign))
X_train_raw = X_benign[idx[:split]]
X_val_raw   = X_benign[idx[split:]]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_all)  

print(f"Train (benign): {X_train.shape[0]} | Val (benign): {X_val.shape[0]}")
print(f"Test (all):      {X_test.shape[0]}  | Attack: {X_attack.shape[0]} | Benign: {X_benign.shape[0]}")


def build_autoencoder(n_features, architecture='medium', l2_reg=1e-4):
    reg = regularizers.l2(l2_reg)
    inp = keras.Input(shape=(n_features,), name='input')

    if architecture == 'shallow':
        x = layers.Dense(8, activation='relu', kernel_regularizer=reg, name='enc_1')(inp)
        x = layers.BatchNormalization(name='bn_enc_1')(x)
        bottleneck = layers.Dense(3, activation='relu', name='bottleneck')(x)
        
        x = layers.Dense(8, activation='relu', kernel_regularizer=reg, name='dec_1')(bottleneck)
        x = layers.BatchNormalization(name='bn_dec_1')(x)

    elif architecture == 'medium':
        x = layers.Dense(16, activation='relu', kernel_regularizer=reg, name='enc_1')(inp)
        x = layers.BatchNormalization(name='bn_enc_1')(x)
        x = layers.Dense(8, activation='relu', kernel_regularizer=reg, name='enc_2')(x)
        x = layers.BatchNormalization(name='bn_enc_2')(x)
        bottleneck = layers.Dense(4, activation='relu', name='bottleneck')(x)

        x = layers.Dense(8, activation='relu', kernel_regularizer=reg, name='dec_1')(bottleneck)
        x = layers.BatchNormalization(name='bn_dec_1')(x)
        x = layers.Dense(16, activation='relu', kernel_regularizer=reg, name='dec_2')(x)
        x = layers.BatchNormalization(name='bn_dec_2')(x)

    elif architecture == 'deep':
        x = layers.Dense(32, activation='relu', kernel_regularizer=reg, name='enc_1')(inp)
        x = layers.BatchNormalization(name='bn_enc_1')(x)
        x = layers.Dropout(0.1, name='drop_enc_1')(x)
        x = layers.Dense(16, activation='relu', kernel_regularizer=reg, name='enc_2')(x)
        x = layers.BatchNormalization(name='bn_enc_2')(x)
        x = layers.Dense(8, activation='relu', kernel_regularizer=reg, name='enc_3')(x)
        bottleneck = layers.Dense(4, activation='relu', name='bottleneck')(x)

        x = layers.Dense(8, activation='relu', kernel_regularizer=reg, name='dec_1')(bottleneck)
        x = layers.Dense(16, activation='relu', kernel_regularizer=reg, name='dec_2')(x)
        x = layers.BatchNormalization(name='bn_dec_2')(x)
        x = layers.Dropout(0.1, name='drop_dec_1')(x)
        x = layers.Dense(32, activation='relu', kernel_regularizer=reg, name='dec_3')(x)
        x = layers.BatchNormalization(name='bn_dec_3')(x)

    output = layers.Dense(n_features, activation='linear', name='output')(x)
    model = keras.Model(inputs=inp, outputs=output, name=f'AE_{architecture}')
    return model


def train_autoencoder(X_train, X_val, architecture='medium', epochs=200,
                      batch_size=32, seed=42, verbose=0):
    tf.random.set_seed(seed)
    model = build_autoencoder(n_features, architecture=architecture)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse')

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                 restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=10, min_lr=1e-6, verbose=0),
    ]

    t0 = time.perf_counter()
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=cb, verbose=verbose
    )
    train_time = time.perf_counter() - t0
    return model, history, train_time

def reconstruction_error(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean((X - X_pred) ** 2, axis=1)

def evaluate_at_threshold(errors, y_true, threshold):
    y_pred = (errors > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
    f1  = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                f1=f1, precision=prec, recall=rec,
                fpr=fpr_val, fnr=fnr_val, threshold=threshold, y_pred=y_pred)


print("\n" + "="*60)
print("FASE 1 — Architecture Comparison (seed=42)")
print("="*60)

arch_results = {}
for arch in ['shallow', 'medium', 'deep']:
    print(f"  Training {arch}...", end=' ')
    model_a, hist_a, t_a = train_autoencoder(X_train, X_val, architecture=arch, seed=42)
    errors_a = reconstruction_error(model_a, X_test)
    thr_a = np.percentile(reconstruction_error(model_a, X_train), 95)
    res_a = evaluate_at_threshold(errors_a, y_all, thr_a)
    auc_a = roc_auc_score(y_all, errors_a)
    arch_results[arch] = {**res_a, 'auc': auc_a, 'train_time': t_a,
                          'history': hist_a, 'model': model_a,
                          'errors': errors_a, 'threshold': thr_a}
    print(f"F1={res_a['f1']:.4f} AUC={auc_a:.4f} FPR={res_a['fpr']:.4f} "
          f"FNR={res_a['fnr']:.4f} ({t_a:.2f}s)")

best_arch = max(arch_results, key=lambda k: arch_results[k]['f1'])
print(f"\n  Best Architecture: {best_arch} (F1={arch_results[best_arch]['f1']:.4f})")

print("\n" + "="*60)
print(f"FASE 2 — Threshold sweep [{best_arch}] (percentis 90–99)")
print("="*60)

best_model  = arch_results[best_arch]['model']
best_errors = arch_results[best_arch]['errors']
train_errors = reconstruction_error(best_model, X_train)

percentiles = list(range(90, 100))
sweep_results = []
for p in percentiles:
    thr = np.percentile(train_errors, p)
    res = evaluate_at_threshold(best_errors, y_all, thr)
    auc = roc_auc_score(y_all, best_errors)
    sweep_results.append({'percentile': p, 'threshold': thr, **res, 'auc': auc})
    print(f"  p{p}: thr={thr:.6f} F1={res['f1']:.4f} "
          f"FPR={res['fpr']:.4f} FNR={res['fnr']:.4f}")

sweep_df = pd.DataFrame(sweep_results)
best_p_row = sweep_df.loc[sweep_df['f1'].idxmax()]
best_p     = int(best_p_row['percentile'])
best_thr   = best_p_row['threshold']
print(f"\n  Best threshold: percentil {best_p} "
      f"(thr={best_thr:.6f}, F1={best_p_row['f1']:.4f})")

print("\n" + "="*60)
print(f"FASE 3 — Multi-Seed Robustness (10 runs, arch={best_arch}, p{best_p})")
print("="*60)

seed_metrics = []
for seed in range(10):
    model_s, _, t_s = train_autoencoder(X_train, X_val, architecture=best_arch,
                                         seed=seed, verbose=0)
    errs_s = reconstruction_error(model_s, X_test)
    train_errs_s = reconstruction_error(model_s, X_train)
    thr_s = np.percentile(train_errs_s, best_p)
    res_s = evaluate_at_threshold(errs_s, y_all, thr_s)
    auc_s = roc_auc_score(y_all, errs_s)
    t_infer = time.perf_counter()
    _ = model_s.predict(X_test, verbose=0)
    t_infer = (time.perf_counter() - t_infer) * 1000

    seed_metrics.append({
        'seed': seed, 'f1': res_s['f1'], 'precision': res_s['precision'],
        'recall': res_s['recall'], 'auc': auc_s,
        'fpr': res_s['fpr'], 'fnr': res_s['fnr'],
        'train_time': t_s, 'infer_time_ms': t_infer,
        'threshold': thr_s
    })
    print(f"  seed {seed}: F1={res_s['f1']:.4f} AUC={auc_s:.4f} "
          f"FPR={res_s['fpr']:.4f} FNR={res_s['fnr']:.4f} ({t_s:.2f}s)")

seed_df = pd.DataFrame(seed_metrics)
print(f"\n  SUMMARY (mean ± std):")
for col in ['f1','precision','recall','auc','fpr','fnr','train_time','infer_time_ms']:
    print(f"    {col:15s}: {seed_df[col].mean():.4f} ± {seed_df[col].std():.4f}")

median_f1 = seed_df['f1'].median()
best_seed  = int(seed_df.loc[(seed_df['f1'] - median_f1).abs().idxmin(), 'seed'])
print(f"\n  Representative Seed for Plots: {best_seed}")

final_model, final_hist, final_train_time = train_autoencoder(
    X_train, X_val, architecture=best_arch, seed=best_seed, verbose=0)
final_errors = reconstruction_error(final_model, X_test)
final_train_errors = reconstruction_error(final_model, X_train)
final_thr = np.percentile(final_train_errors, best_p)
final_res = evaluate_at_threshold(final_errors, y_all, final_thr)
final_auc = roc_auc_score(y_all, final_errors)
t_infer_final = time.perf_counter()
_ = final_model.predict(X_test, verbose=0)
final_infer_ms = (time.perf_counter() - t_infer_final) * 1000

COLORS = {'benign': '#2196F3', 'attack': '#F44336',
          'ae': '#FF9800', 'threshold': '#9C27B0',
          'train': '#4CAF50', 'val': '#FF5722'}

fig, ax = plt.subplots(figsize=(9, 4))
epochs_ran = len(final_hist.history['loss'])
ax.plot(range(1, epochs_ran+1), final_hist.history['loss'],
        color=COLORS['train'], linewidth=2, label='Train Loss')
ax.plot(range(1, epochs_ran+1), final_hist.history['val_loss'],
        color=COLORS['val'], linewidth=2, linestyle='--', label='Val Loss')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title(f'Autoencoder Training — {best_arch.capitalize()} Architecture', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'ae_training_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  {plots_dir / 'ae_training_curve.png'}")

benign_errors = final_errors[y_all == 0]
attack_errors = final_errors[y_all == 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(benign_errors, bins=60, alpha=0.7, color=COLORS['benign'],
        label=f'Benign (n={len(benign_errors)})', density=True)
ax.hist(attack_errors, bins=60, alpha=0.7, color=COLORS['attack'],
        label=f'Slowloris (n={len(attack_errors)})', density=True)
ax.axvline(final_thr, color=COLORS['threshold'], linewidth=2, linestyle='--',
           label=f'Threshold (p{best_p}={final_thr:.5f})')

p999_benign = np.percentile(benign_errors, 99.9)
p999_attack = np.percentile(attack_errors, 99.9)
xlim_max = max(p999_benign, p999_attack, final_thr) * 1.3
ax.set_xlim(left=0, right=xlim_max)

ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Reconstruction Error Distribution', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
bp = ax.boxplot([benign_errors, attack_errors],
                labels=['Benign', 'Slowloris'],
                patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor(COLORS['benign'])
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(COLORS['attack'])
bp['boxes'][1].set_alpha(0.7)
ax.axhline(final_thr, color=COLORS['threshold'], linewidth=2, linestyle='--',
           label=f'Threshold (p{best_p})')
ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12)
ax.set_title('Error Distribution — Boxplot', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

fig.suptitle(f'Autoencoder — Reconstruction Error Analysis | {best_arch.capitalize()} | p{best_p} threshold',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_dir / 'ae_reconstruction_error.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  {plots_dir / 'ae_reconstruction_error.png'}")

from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_all, final_res['y_pred'], labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Slowloris'])
disp.plot(ax=ax, colorbar=True, cmap='Oranges')
ax.set_title(
    f'Autoencoder — Confusion Matrix\n'
    f'FPR={final_res["fpr"]:.4f}  FNR={final_res["fnr"]:.4f}  '
    f'F1={final_res["f1"]:.4f}  AUC={final_auc:.4f}',
    fontsize=12)
plt.tight_layout()
plt.savefig(plots_dir / 'ae_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  {plots_dir / 'ae_confusion_matrix.png'}")

fpr_roc, tpr_roc, _ = roc_curve(y_all, final_errors)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_roc, tpr_roc, color=COLORS['ae'], linewidth=2.5,
        label=f'Autoencoder ({best_arch}) — AUC={final_auc:.4f}')
ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Autoencoder | Slowloris Detection', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'ae_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  {plots_dir / 'ae_roc_curve.png'}")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(sweep_df['percentile'], sweep_df['f1'],
         'o-', color='#4CAF50', linewidth=2.5, label='F1')
ax2.plot(sweep_df['percentile'], sweep_df['fpr'],
         's--', color=COLORS['attack'], linewidth=2, label='FPR', alpha=0.85)
ax2.plot(sweep_df['percentile'], sweep_df['fnr'],
         '^--', color='#FF9800', linewidth=2, label='FNR', alpha=0.85)
ax1.axvline(best_p, color=COLORS['threshold'], linewidth=2, linestyle=':',
            label=f'Best p{best_p}')
ax1.set_xlabel('Threshold Percentile', fontsize=12)
ax1.set_ylabel('F1 Score', fontsize=12, color='#4CAF50')
ax2.set_ylabel('FPR / FNR', fontsize=12)
ax1.set_xticks(percentiles)
ax1.tick_params(axis='y', labelcolor='#4CAF50')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='lower left')
ax1.set_title('Threshold Sensitivity — F1, FPR, FNR vs Percentile', fontsize=13)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'ae_threshold_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  {plots_dir / 'ae_threshold_sweep.png'}")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_arch = ['f1', 'auc', 'fpr']
titles_arch  = ['F1 Score', 'ROC-AUC', 'FPR (False Alarm Rate)']
arch_names = list(arch_results.keys())
colors_arch = ['#42A5F5', '#66BB6A', '#EF5350']

for i, (metric, title) in enumerate(zip(metrics_arch, titles_arch)):
    vals = [arch_results[a][metric] for a in arch_names]
    bars = axes[i].bar(arch_names, vals, color=colors_arch, alpha=0.85, edgecolor='white')
    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].set_ylim(0, max(vals) * 1.15)
    axes[i].grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=11)

fig.suptitle('Architecture Comparison — Shallow vs Medium vs Deep', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_dir / 'ae_architecture_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  {plots_dir / 'ae_architecture_comparison.png'}")

fig, axes = plt.subplots(1, 4, figsize=(14, 5))
metrics_seed = ['f1', 'auc', 'fpr', 'fnr']
titles_seed  = ['F1', 'ROC-AUC', 'FPR', 'FNR']
for i, (m, t) in enumerate(zip(metrics_seed, titles_seed)):
    axes[i].boxplot(seed_df[m], patch_artist=True,
                    boxprops=dict(facecolor='#FF9800', alpha=0.7))
    axes[i].set_title(t, fontsize=12, fontweight='bold')
    axes[i].set_xticks([])
    axes[i].grid(alpha=0.3, axis='y')
    mean_v = seed_df[m].mean()
    std_v  = seed_df[m].std()
    axes[i].set_xlabel(f'{mean_v:.4f} ± {std_v:.4f}', fontsize=10)

fig.suptitle(f'Autoencoder Robustness — 10 Seeds | {best_arch.capitalize()} | p{best_p}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_dir / 'ae_multiseed_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  {plots_dir / 'ae_multiseed_robustness.png'}")

mean_s = seed_df.mean()
std_s  = seed_df.std()

ae_row = {
    'Model': 'Autoencoder',
    'Type': 'Anomaly Detection',
    'Precision':  round(mean_s['precision'], 4),
    'Recall':     round(mean_s['recall'],    4),
    'F1':         round(mean_s['f1'],        4),
    'F1_std':     round(std_s['f1'],         4),
    'ROC_AUC':    round(mean_s['auc'],       4),
    'AUC_std':    round(std_s['auc'],        4),
    'FPR':        round(mean_s['fpr'],       4),
    'FNR':        round(mean_s['fnr'],       4),
    'Silhouette': '-',
    'Davies_Bouldin': '-',
    'Train_time_s':    f"{mean_s['train_time']:.4f} ± {std_s['train_time']:.4f}",
    'Infer_time_ms':   f"{mean_s['infer_time_ms']:.3f} ± {std_s['infer_time_ms']:.3f}",
    'Note': f"threshold=p{best_p} | arch={best_arch} | 10 seeds | trained on benign only"
}

try:
    prev = pd.read_csv(results_dir / 'unsupervised_results_complete.csv')
    prev = prev[prev['Model'] != 'Autoencoder']
    updated = pd.concat([prev, pd.DataFrame([ae_row])], ignore_index=True)
except FileNotFoundError:
    updated = pd.DataFrame([ae_row])

updated.to_csv(results_dir / 'unsupervised_results_complete.csv', index=False)

pd.DataFrame(seed_metrics).to_csv(results_dir / 'autoencoder_seed_results.csv', index=False)
sweep_df.drop(columns=['y_pred'], errors='ignore').to_csv(
    results_dir / 'autoencoder_threshold_sweep.csv', index=False)

print("\n" + "="*60)
print("AUTOENCODER — FINAL RESULTS")
print("="*60)
print(f"  Architecture:  {best_arch.capitalize()} | Bottleneck={'4' if best_arch!='shallow' else '3'}")
print(f"  Threshold:     percentile {best_p} = {final_thr:.6f}")
print(f"  Seeds:         10 (representative seed = {best_seed})")
print(f"\n  {'Metric':<15} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*33}")
for m in ['f1','precision','recall','auc','fpr','fnr']:
    print(f"  {m:<15} {seed_df[m].mean():>8.4f} {seed_df[m].std():>8.4f}")
print(f"  {'train_time(s)':<15} {seed_df['train_time'].mean():>8.2f} {seed_df['train_time'].std():>8.2f}")
print(f"  {'infer_time(ms)':<15} {seed_df['infer_time_ms'].mean():>8.3f} {seed_df['infer_time_ms'].std():>8.3f}")
print(f"\n  Files generated:")
print(f"    plots_autoencoder/ae_training_curve.png")
print(f"    {plots_dir / 'ae_reconstruction_error.png'}")
print(f"    {plots_dir / 'ae_confusion_matrix.png'}")
print(f"    {plots_dir / 'ae_roc_curve.png'}")
print(f"    {plots_dir / 'ae_threshold_sweep.png'}")
print(f"    {plots_dir / 'ae_architecture_comparison.png'}")
print(f"    {plots_dir / 'ae_multiseed_robustness.png'}")
print(f"    {results_dir / 'autoencoder_seed_results.csv'}")
print(f"    {results_dir / 'autoencoder_threshold_sweep.csv'}")
print(f"    {results_dir / 'unsupervised_results_complete.csv'}  (updated with Autoencoder)")