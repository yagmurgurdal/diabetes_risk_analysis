from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".deps")

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("diabetes.csv")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42


def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    invalid_zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for column in invalid_zero_columns:
        df.loc[df[column] == 0, column] = pd.NA

    df["BMI_Obese_Flag"] = (df["BMI"].fillna(df["BMI"].median()) >= 30).astype(int)
    df["Glucose_High_Flag"] = (df["Glucose"].fillna(df["Glucose"].median()) >= 140).astype(int)
    df["Age_Risk_Flag"] = (df["Age"] >= 35).astype(int)
    df["Pregnancy_Risk_Flag"] = (df["Pregnancies"] >= 4).astype(int)
    df["Glucose_BMI_Interaction"] = df["Glucose"].fillna(df["Glucose"].median()) * df["BMI"].fillna(df["BMI"].median())
    df["Age_Glucose_Interaction"] = df["Age"] * df["Glucose"].fillna(df["Glucose"].median())
    df["Insulin_Glucose_Ratio"] = df["Insulin"].fillna(df["Insulin"].median()) / (df["Glucose"].fillna(df["Glucose"].median()) + 1)
    df["Pregnancy_Age_Ratio"] = df["Pregnancies"] / (df["Age"] + 1)
    df["BMI_Age_Product"] = df["BMI"].fillna(df["BMI"].median()) * df["Age"]
    df["Glucose_Squared"] = df["Glucose"].fillna(df["Glucose"].median()) ** 2

    return df


def build_preprocessors(feature_columns: list[str]) -> tuple[ColumnTransformer, ColumnTransformer]:
    scaled_preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ]
    )

    tree_preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_columns,
            )
        ]
    )

    return scaled_preprocessor, tree_preprocessor


def get_models(feature_columns: list[str]) -> dict[str, Pipeline]:
    scaled_preprocessor, tree_preprocessor = build_preprocessors(feature_columns)

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "KNN": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                ("model", KNeighborsClassifier(n_neighbors=11, weights="distance", metric="euclidean")),
            ]
        ),
        "SVM": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                ("model", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                ("model", ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=RANDOM_STATE)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                ("model", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE)),
            ]
        ),
        "AdaBoost": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                ("model", AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE)),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                ("model", GaussianNB()),
            ]
        ),
        "MLP": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, learning_rate_init=0.001, max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
    }


def find_best_threshold(y_true: pd.Series, probabilities) -> float:
    best_threshold = 0.50
    best_f1 = -1.0
    best_accuracy = -1.0

    for step in range(25, 76):
        threshold = step / 100
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, predictions, zero_division=0)
        accuracy = accuracy_score(y_true, predictions)

        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-9 and accuracy > best_accuracy):
            best_f1 = f1
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    models = get_models(list(x.columns))
    records: list[dict[str, object]] = []
    best_model_payload: dict[str, object] | None = None

    for model_name, pipeline in models.items():
        pipeline.fit(x_train, y_train)

        val_probabilities = pipeline.predict_proba(x_val)[:, 1]
        threshold = find_best_threshold(y_val, val_probabilities)

        test_probabilities = pipeline.predict_proba(x_test)[:, 1]
        test_predictions = (test_probabilities >= threshold).astype(int)

        metrics = {
            "model": model_name,
            "threshold": round(threshold, 2),
            "accuracy": round(float(accuracy_score(y_test, test_predictions)), 4),
            "precision": round(float(precision_score(y_test, test_predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, test_predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, test_predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, test_probabilities)), 4),
            "confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
        }
        records.append(metrics)

        is_better = False
        if best_model_payload is None:
            is_better = True
        else:
            best_metrics = best_model_payload["metrics"]
            if (
                metrics["f1_score"],
                metrics["accuracy"],
                metrics["roc_auc"],
            ) > (
                best_metrics["f1_score"],
                best_metrics["accuracy"],
                best_metrics["roc_auc"],
            ):
                is_better = True

        if is_better:
            best_model_payload = {
                "model_name": model_name,
                "pipeline": pipeline,
                "metrics": metrics,
                "probabilities": test_probabilities,
                "predictions": test_predictions,
                "actual": y_test.reset_index(drop=True),
            }

    if best_model_payload is None:
        raise RuntimeError("No model evaluation result was produced.")

    comparison_df = pd.DataFrame(records).sort_values(
        by=["f1_score", "accuracy", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    return comparison_df, best_model_payload


def save_outputs(comparison_df: pd.DataFrame, best_model_payload: dict[str, object]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    pd.DataFrame(
        {
            "actual": best_model_payload["actual"],
            "predicted": best_model_payload["predictions"],
            "probability": pd.Series(best_model_payload["probabilities"]).round(4),
        }
    ).to_csv(RESULTS_DIR / "best_model_predictions.csv", index=False)

    metrics = best_model_payload["metrics"]
    payload = {
        "selected_model": best_model_payload["model_name"],
        "metrics": metrics,
        "ranking": comparison_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_dict(orient="records"),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines = [
        "# Diyabet Riski Tahmini - 10 Algoritma Karsilastirmasi",
        "",
        "## Kullanilan Algoritmalar",
        "- Logistic Regression",
        "- KNN",
        "- SVM",
        "- Decision Tree",
        "- Random Forest",
        "- Extra Trees",
        "- Gradient Boosting",
        "- AdaBoost",
        "- Gaussian Naive Bayes",
        "- MLP",
        "",
        "## En Iyi Model",
        f"- Model: {best_model_payload['model_name']}",
        f"- Accuracy: {metrics['accuracy']}",
        f"- Precision: {metrics['precision']}",
        f"- Recall: {metrics['recall']}",
        f"- F1-Score: {metrics['f1_score']}",
        f"- ROC-AUC: {metrics['roc_auc']}",
        f"- Confusion Matrix: {metrics['confusion_matrix']}",
        "",
        "## Dosyalar",
        "- model_comparison.csv",
        "- best_model_predictions.csv",
        "- metrics.json",
    ]
    (RESULTS_DIR / "project_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    df = load_and_prepare_data()
    comparison_df, best_model_payload = evaluate_models(df)
    save_outputs(comparison_df, best_model_payload)

    print("10 algoritma egitildi ve karsilastirildi.")
    print(comparison_df.to_string(index=False))
    print()
    print("En iyi model:", best_model_payload["model_name"])
    print(json.dumps(best_model_payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
