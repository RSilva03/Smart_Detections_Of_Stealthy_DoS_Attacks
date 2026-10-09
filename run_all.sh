# run_all.sh - Run all dissertation experiments sequentially

set -e

echo "==========================="
echo " Smart Detection of Stealthy DoS Attacks"
echo " Running all experiments..."
echo "==========================="

echo ""
echo ">>> [1/12] Slowloris - Supervised ML"
python scripts/slowloris/train_all_supervised_ml_slowloris.py

echo ""
echo ">>> [2/12] Slowloris - Autoencoder"
python scripts/slowloris/autoencoder_slowloris.py

echo ""
echo ">>> [3/12] Slowloris - Unsupervised ML"
python scripts/slowloris/unsupervised_corrections.py

echo ""
echo ">>> [4/12] Slowloris - Visualisation Results"
python scripts/slowloris/visualize_results_supervised.py

echo ""
echo ">>> [5/12] Slowloris - Feature Importance"
python scripts/slowloris/slowloris_feature_importance.py

echo ""
echo ">>> [6/12] RoQ - Supervised ML"
python scripts/roq/ml_roq_supervised.py

echo ""
echo ">>> [7/12] RoQ - Unsupervised ML"
python scripts/roq/ml_roq_unsupervised.py 

echo ""
echo ">>> [8/12] RoQ - Feature Importance"
python scripts/roq/roq_feature_importance.py

echo ""
echo ">>> [9/12] RoQ - Hybrid Fuzzy Model"
python scripts/hybrid/roq_hybrid_fuzzy.py

echo ""
echo ">>> [10/12] RoQ - RSKF Ablation (SVM Variants)"
python scripts/hybrid/roq_ablation_kfold_svm.py

echo ""
echo ">>> [11/12] RoQ - Statistical Analysis (Wilcoxon Test)"
python scripts/hybrid/wilcoxon_test.py

echo ""
echo ">>> [12/12] RoQ - Visualisation Results"
python scripts/hybrid/visualize_hybrid_results.py


echo ""
echo "==========================="
echo " All experiments completed!"
echo " Results saved in : results/"
echo "==========================="

