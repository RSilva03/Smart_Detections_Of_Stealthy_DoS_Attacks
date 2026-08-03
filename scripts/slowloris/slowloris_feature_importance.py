import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

FEATURES = [
    "duration", "pkts_total", "bytes_total", "pkt_size_mean", "tcp_bytes_total", "tcp_pkts",
    "pkts_rate", "iat_mean", "iat_std", ]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parents[1]

    data_path = root_dir / "dataset" / "slowloris" / "dataset_A" / "dataset_slowloris.csv"
    results_dir = root_dir / "results"
    plots_dir = results_dir / "plots_slowloris_feature_importance"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path).dropna(subset=FEATURES + ["label"])
    X = df[FEATURES].values
    y = df["label"].values

    print(
        f"Dataset: {len(df)} flows | Attack: {int(y.sum())} | Benign: {int((y == 0).sum())}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    gini_importance = rf.feature_importances_
    mi_importance = mutual_info_classif(
        X_train,
        y_train,
        random_state=RANDOM_STATE,
        discrete_features=False,
    )

    perm = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_importance = perm.importances_mean

    corr_importance = np.array(
        [abs(np.corrcoef(df[f].values, y)[0, 1]) for f in FEATURES], dtype=float
    )
    corr_importance = np.nan_to_num(corr_importance, nan=0.0)

    combined = (
        _minmax(gini_importance)
        + _minmax(mi_importance)
        + _minmax(perm_importance)
        + _minmax(corr_importance)
    ) / 4.0

    out_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "rf_gini": gini_importance,
            "mutual_info": mi_importance,
            "permutation": perm_importance,
            "abs_corr_label": corr_importance,
            "combined_score": combined,
        }
    ).sort_values("combined_score", ascending=False)

    csv_path = results_dir / "slowloris_feature_importance.csv"
    out_df.to_csv(csv_path, index=False)

    json_path = results_dir / "slowloris_feature_importance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(data_path),
                "n_flows": int(len(df)),
                "n_attack": int(y.sum()),
                "n_benign": int((y == 0).sum()),
                "ranking": out_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    plot_df = out_df.iloc[::-1]
    y_pos = np.arange(len(plot_df))
    h = 0.18

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y_pos - 1.5 * h, plot_df["rf_gini"], height=h, label="RF (Gini)")
    ax.barh(y_pos - 0.5 * h, plot_df["mutual_info"], height=h, label="Mutual Info")
    ax.barh(y_pos + 0.5 * h, plot_df["permutation"], height=h, label="Permutation")
    ax.barh(y_pos + 1.5 * h, plot_df["combined_score"], height=h, label="Combined")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Importance score")
    ax.set_title("Slowloris Feature Importance Ranking")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()

    plot_path = plots_dir / "slowloris_feature_importance.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("\nTop features by combined score:")
    print(out_df[["feature", "combined_score"]].head(5).to_string(index=False))
    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()