from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".deps")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

import diabetes_project as project


RESULTS_DIR = Path("results")
ASSETS_DIR = RESULTS_DIR / "report_assets"

MODEL_LABELS = {
    "Logistic Regression": "LR",
    "Decision Tree": "DT",
    "KNN": "KNN",
    "Gaussian Naive Bayes": "NB",
    "SVM": "SVM",
    "AdaBoost": "AdaB",
    "Gradient Boosting": "GBM",
    "Random Forest": "RF",
    "Extra Trees": "ET",
    "MLP": "MLP",
}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)

    df = project.load_and_prepare_data()
    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=project.RANDOM_STATE)
    models = project.get_models(list(x.columns))

    fold_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    plot_data: list[list[float]] = []
    plot_labels: list[str] = []

    ordered_names = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Gaussian Naive Bayes",
        "SVM",
        "AdaBoost",
        "Gradient Boosting",
        "Random Forest",
        "Extra Trees",
        "MLP",
    ]

    for model_name in ordered_names:
        pipeline = models[model_name]
        scores = cross_val_score(
            pipeline,
            x,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )

        for fold_idx, score in enumerate(scores, start=1):
            fold_records.append(
                {
                    "model": model_name,
                    "label": MODEL_LABELS[model_name],
                    "fold": fold_idx,
                    "accuracy": round(float(score), 4),
                }
            )

        summary_records.append(
            {
                "model": model_name,
                "label": MODEL_LABELS[model_name],
                "mean_accuracy": round(float(scores.mean()), 4),
                "std_accuracy": round(float(scores.std()), 4),
                "min_accuracy": round(float(scores.min()), 4),
                "max_accuracy": round(float(scores.max()), 4),
            }
        )

        plot_data.append(scores.tolist())
        plot_labels.append(MODEL_LABELS[model_name])

    pd.DataFrame(fold_records).to_csv(RESULTS_DIR / "cv_accuracy_scores.csv", index=False)
    summary_df = pd.DataFrame(summary_records).sort_values(
        by=["mean_accuracy", "std_accuracy"],
        ascending=[False, True],
    )
    summary_df.to_csv(RESULTS_DIR / "cv_accuracy_summary.csv", index=False)

    plt.figure(figsize=(13, 6))
    plt.boxplot(plot_data, labels=plot_labels, patch_artist=True)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "model_accuracy_boxplot.png", dpi=220)
    plt.close()

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
