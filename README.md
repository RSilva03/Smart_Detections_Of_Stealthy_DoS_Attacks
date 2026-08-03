# Smart Detection of Stealthy DoS Attacks
**Master's Dissertation — NOVA University Lisbon**  
Rodrigo António Gonçalves Silva

---

## Overview

This repository contains all the experimental scripts used in the dissertation
**"Smart Detection of Stealthy DoS Attacks"**, covering:

- **Slowloris** attack detection (supervised + unsupervised ML)
- **RoQ (Reduction of Quality)** attack detection (supervised + unsupervised ML)
- **Hybrid Detection Model** combining the best classifiers found, Euclidean distance scoring,
  and fuzzy logic inference for RoQ traffic

---

## Repository Structure

```
project/
│
├── dataset/
│   ├── slowloris/
│   │   ├── dataset_A/          
|   |   |   ├── dataset_slowloris.csv  # Laboratory Dataset
|   |   └── dataset_B/
|   |   |   ├── dataset_IFTO.csv       # IFTO Dataset
│   └── roq/
│       └── dataset_roq_window_full.csv      # RoQ dataset (window-level features)
│
├── scripts/
│   ├── slowloris/
│   │   ├── train_all_supervised_ml_slowloris.py   # Supervised ML benchmark (Slowloris)
│   │   ├── autoencoder_slowloris.py               # Autoencoder unsupervised (Slowloris)
│   │   ├── unsupervised_corrections.py            # K-Means + IForest (Slowloris)
│   │   ├── slowloris_feature_importance.py        # Feature importance (Slowloris)
│   │   └── visualize_results_supervised.py        # Visualisation (Slowloris)
│   │
│   ├── roq/
│   |   ├── ml_roq_supervised.py                   # Supervised ML benchmark (RoQ)
│   |   ├── ml_roq_unsupervised.py                 # Unsupervised ML (RoQ)
|   |   ├── roq_generator.py                       # RoQ traffic generator (RoQ)
│   |   └── roq_feature_importance.py              # Feature importance + SHAP (RoQ)
│   |
|   └── hybrid/
|   |   ├── roq_hybrid_fuzzy.py                    # Hybrid Fuzzy RF model (Hybrid)
│   |   ├── roq_ablation_kfold_rf.py               # RSKF ablation RF variants (Hybrid)
|   |   ├── visualize_hybrid_results.py            # Visualisation (Hybrid)
│   |   └── wilcoxon_multiple_corrections.py       # Statistical significance tests (Hybrid)
│
├── results/
│   └── (auto-generated CSVs and plots)
│
├── requirements.txt
└── README.md
```

---

## Datasets Required

The scripts expect two datasets, which are included in this repository as follows:

| Path | Description |
|------|-------------|
| `dataset/slowloris/dataset_A/dataset_slowloris.csv` | Slowloris labelled flow dataset (Laboratory) |
| `dataset/slowloris/dataset_B/dataset_IFTO.csv` | Slowloris labelled flow dataset (IFTO)
| `dataset/roq/dataset_roq_window_full.csv` | RoQ window-level feature dataset |

Both files contain a `label` column (0 = benign, 1 = attack).

---

## Installation

### 1. Clone / download the repository

```bash
git clone <repo-url>
cd project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        
venv\Scripts\activate          
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If you do not have a GPU, PyTorch CPU-only is sufficient:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Running the Experiments

All scripts are self-contained and can be run independently.
Results (CSVs and plots) are saved automatically in the `results/` folder.

### Slowloris — Supervised ML (training)

```bash
python scripts/slowloris/train_all_supervised_ml_slowloris.py
```
Outputs:
- `results/supervised_results_slowloris.csv`
- `results/mcnemar_results_slowloris.csv`
- `results/plots_supervised_slowloris/fig7_shap_summary.png`
- `results/plots_supervised_slowloris/fig8_shap_bar.png`

### Slowloris — Supervised ML (visualization)

```bash
python scripts/slowloris/visualize_results_supervised.py
```
Outputs:
- `results/plots_supervised_slowloris/fig1_confusion_matrices.png`
- `results/plots_supervised_slowloris/fig2_roc_curves.png`
- `results/plots_supervised_slowloris/fig3_metrics_comparison.png`
- `results/plots_supervised_slowloris/fig4_fpr_fnr.png`
- `results/plots_supervised_slowloris/fig5_feature_importance.png`
- `results/plots_supervised_slowloris/fig6_heatmap_all_metrics.png`

### Slowloris — Unsupervised ML (K-Means + Isolation Forest)

```bash
python scripts/slowloris/unsupervised_corrections.py
```
Outputs:
- `results/unsupervised_results_slowloris.csv`
- `results/plots_unsupervised_slowloris/kmeans_confusion_matrices.png`
- `results/plots_unsupervised_slowloris/kmeans_roc_curves.png`
- `results/plots_unsupervised_slowloris/iforest_confusion_matrix.png`
- `results/plots_unsupervised_slowloris/roc_unified_sup_vs_unsup.png`

### Slowloris — Autoencoders

```bash
python scripts/slowloris/autoencoder_slowloris.py
```
Outputs:
- `results/autoencoder_seed_results.csv`
- `results/autoencoder_threshold_sweep.csv`
- `results/unsupervised_results_complete.csv` (updated)
- `results/plots_autoencoder_slowloris/ae_training_curve.png`
- `results/plots_autoencoder_slowloris/ae_reconstruction_error.png`
- `results/plots_autoencoder_slowloris/ae_confusion_matrix.png`
- `results/plots_autoencoder_slowloris/ae_roc_curve.png`
- `results/plots_autoencoder_slowloris/ae_threshold_sweep.png`
- `results/plots_autoencoder_slowloris/ae_architecture_comparison.png`
- `results/plots_autoencoder_slowloris/ae_multiseed_robustness.png`

### Slowloris — Feature Importance

```bash
python scripts/slowloris/slowloris_feature_importance.py
```
Outputs:
- `results/slowloris_feature_importance.csv`
- `results/slowloris_feature_importance.json`
- `results/plots_slowloris_feature_importance/slowloris_feature_importance.png`

### RoQ — Supervised ML

```bash
python scripts/roq/ml_roq_supervised.py
```
Outputs:
- `results/roq_supervised_results.csv`

### RoQ — Unsupervised ML

```bash
python scripts/roq/ml_roq_unsupervised.py
```
Outputs:
- `results/roq_unsupervised_results_roq.csv`
- `results/unsup_ae_variants_results_roq.csv`
- `results/unsup_iforest_sweep_roq.csv`
- `results/plots_unsupervised_roq/unsup_confusion_matrices.png`
- `results/plots_unsupervised_roq/unsup_roc_curves.png`
- `results/plots_unsupervised_roq/unsup_pca_clusters.png`
- `results/plots_unsupervised_roq/unsup_threshold_sweep.png`
- `results/plots_unsupervised_roq/unsup_ae_reconstruction.png`
- `results/plots_unsupervised_roq/unsup_kmeans_selection.png`
- `results/plots_unsupervised_roq/unsup_ae_variants_comparison.png`

### RoQ — Feature Importance

```bash
python scripts/roq/roq_feature_importance.py
```
Outputs:
- `results/roq_feature_importance.csv`
- `results/plots_feature_importance_roq/feat_rf_importance.png`
- `results/plots_feature_importance_roq/feat_mutual_info.png`
- `results/plots_feature_importance_roq/feat_shap.png` (if SHAP available)
- `results/plots_feature_importance_roq/feat_comparison.png`

### RoQ — Hybrid Fuzzy Model

```bash
python scripts/hybrid/roq_hybrid_fuzzy.py
```
Outputs:
- `results/roq_hybrid_fuzzy_results.csv`

### RoQ — RSKF Ablation (RF variants)

```bash
python scripts/hybrid/roq_ablation_kfold_rf.py
```
Outputs:
- `results/roq_hybrid_rskf_summary.csv`
- `results/roq_hybrid_rskf_all_folds.csv`
- `results/plots_hybrid_roq/rskf_boxplots.png`
- `results/plots_hybrid_roq/rskf_mean_ci_barplot.png`
- `results/plots_hybrid_roq/rskf_summary_heatmap.png`
- `results/plots_hybrid_roq/rskf_fpr_f1_scatter.png`
- `results/plots_hybrid_roq/rskf_violin_f1.png`

### Hybrid — Results Visualization

```bash
python scripts/hybrid/visualize_hybrid_results.py
```
Outputs:
- `results/plots_hybrid_roq/hybrid_metrics_barplot.png`
- `results/plots_hybrid_roq/hybrid_metrics_heatmap.png`
- `results/plots_hybrid_roq/hybrid_confusion_matrices.png`
- `results/plots_hybrid_roq/hybrid_radar_chart.png`
- `results/plots_hybrid_roq/hybrid_fpr_fnr_scatter.png`
- `results/plots_hybrid_roq/hybrid_thresholds.png`

### Statistical Analysis (Wilcoxon + corrections)

```bash
python scripts/hybrid/wilcoxon_multiple_corrections.py
```
Outputs:
- `results/roq_wilcoxon_bonferroni_results.csv`
- `results/plots_hybrid_roq/wilcoxon_pvalue_raw_heatmap.png`
- `results/plots_hybrid_roq/wilcoxon_pvalue_corrected_heatmap.png`
- `results/plots_hybrid_roq/wilcoxon_correction_impact.png`
- `results/plots_hybrid_roq/wilcoxon_cliffs_delta_corrected.png`
- `results/plots_hybrid_roq/wilcoxon_paired_boxplot_corrected.png`
- `results/plots_hybrid_roq/wilcoxon_fold_diff_fpr_fnr.png`

### RoQ Traffic Generator (data collection utility)

```bash
python scripts/roq/roq_generator.py --target <ip> --port 80 --url /index.html
```
Outputs:
- `roq_attack_log.csv` (saved in current working directory by default, or custom `--log` path)

### Run all experiments sequentially

```bash
python scripts/slowloris/train_all_supervised_ml_slowloris.py && \
python scripts/slowloris/visualize_results_supervised.py && \
python scripts/slowloris/unsupervised_corrections.py && \
python scripts/slowloris/autoencoder_slowloris.py && \
python scripts/slowloris/slowloris_feature_importance.py && \
python scripts/roq/ml_roq_supervised.py && \
python scripts/roq/ml_roq_unsupervised.py && \
python scripts/roq/roq_feature_importance.py && \
python scripts/hybrid/roq_hybrid_fuzzy.py && \
python scripts/hybrid/visualize_hybrid_results.py && \
python scripts/hybrid/roq_ablation_kfold_rf.py && \
python scripts/hybrid/wilcoxon_multiple_corrections.py
```

### Setup on a fresh Ubuntu 22.04 VM

```bash

sudo apt update && sudo apt install -y python3-pip python3-venv git

git clone <repo-url>
cd project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

scp -i key.pem dataset/ ubuntu@<vm-ip>:~/project/dataset/

bash run_all.sh
```

---

## Expected Runtimes (approximate, 2 vCPU / 4 GB RAM)

| Script | Est. Runtime |
|--------|-------------|
| Supervised Slowloris | ~2–5 min |
| Unsupervised Slowloris | ~3–8 min |
| Supervised RoQ | ~3–6 min |
| Unsupervised RoQ | ~5–15 min |
| Hybrid Fuzzy Model | ~5–10 min |
| RSKF Ablation (50 folds) | ~15–30 min |
| Wilcoxon + corrections | ~1–2 min |
| **Total** | **~35–75 min** |

---

## Dependencies Summary

| Package | Purpose |
|---------|---------|
| scikit-learn | ML classifiers, cross-validation, metrics |
| lightgbm | LightGBM classifier |
| imbalanced-learn | SMOTE oversampling |
| torch | Autoencoder (deep learning) |
| scikit-fuzzy | Fuzzy logic inference system |
| shap | Feature importance (SHAP values) |
| statsmodels | McNemar test |
| scipy | Wilcoxon signed-rank test |
| pandas / numpy | Data handling |
| matplotlib / seaborn | Visualisation |

---

## Contact

Rodrigo António Gonçalves Silva  
NOVA School of Science and Technology  
Supervisor: Prof. Henrique Domingos
