# Diabetes Risk Analysis

This repository contains a machine learning project for **diabetes risk prediction** using the `diabetes.csv` dataset.

The project was built as an end-to-end mini data science workflow:
- data inspection
- data cleaning
- feature engineering
- multi-model classification
- model comparison
- report asset generation

## Project Goal

The main objective is to predict whether an individual is in the diabetes risk group (`Outcome = 1`) or not (`Outcome = 0`) based on clinical measurements.

The project does not rely on a single model. Instead, it compares multiple classification algorithms and evaluates them with several performance metrics to identify the most suitable approach for this problem.

## Dataset

The dataset used in this project is:

- `diabetes.csv`

Target variable:

- `Outcome`

Selected input features include:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

## Data Preparation

Before training, the project applies several preprocessing steps:

- biologically unrealistic `0` values in selected columns are treated as missing values
- missing-like values are filled with median imputation
- scaled preprocessing is used for models that are sensitive to feature magnitude
- additional engineered features are created to improve predictive signal

Examples of engineered features:

- `BMI_Obese_Flag`
- `Glucose_High_Flag`
- `Age_Risk_Flag`
- `Pregnancy_Risk_Flag`
- `Glucose_BMI_Interaction`
- `Age_Glucose_Interaction`
- `Insulin_Glucose_Ratio`
- `Pregnancy_Age_Ratio`
- `BMI_Age_Product`
- `Glucose_Squared`

## Models Included

The project compares 10 classification algorithms:

1. Logistic Regression
2. KNN
3. SVM
4. Decision Tree
5. Random Forest
6. Extra Trees
7. Gradient Boosting
8. AdaBoost
9. Gaussian Naive Bayes
10. MLP

## Evaluation Strategy

Two evaluation perspectives are used in the project:

1. Train / validation / test split for final model comparison
2. Stratified 10-fold cross validation for video-style accuracy comparison

Performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

The project gives special attention to **recall**, because in a health-related classification problem, missing a risky patient can be more critical than producing a false alarm.

## Repository Structure

Main files in this repository:

- `diabetes_project.py`
  Main training and evaluation pipeline for the 10 models.

- `generate_cv_comparison.py`
  Produces 10-fold cross validation accuracy results and the boxplot used for model comparison.

- `generate_report_assets.py`
  Creates visual assets such as class distribution, confusion matrix, ROC curve, and comparison graphics.

- `edit_docx_report.py`
  Prepares the final `.docx` report from the generated project outputs.

- `diabetes.csv`
  Dataset used for training and evaluation.

## How to Run

Use the bundled Python runtime that was used during development:

```powershell
C:\Users\yagmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe diabetes_project.py
```

To generate the cross-validation comparison:

```powershell
C:\Users\yagmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe generate_cv_comparison.py
```

To generate report visuals:

```powershell
C:\Users\yagmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe generate_report_assets.py
```

To generate the final report document:

```powershell
C:\Users\yagmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe edit_docx_report.py
```

## Generated Outputs

When the scripts are run locally, they generate outputs under `results/`, such as:

- `model_comparison.csv`
- `best_model_predictions.csv`
- `metrics.json`
- `cv_accuracy_scores.csv`
- `cv_accuracy_summary.csv`
- report visual assets under `results/report_assets/`

These generated artifacts are not committed to the repository by default.

## Current Best Result

In the current project version, the strongest model under the final split-based comparison is:

- Model: `Logistic Regression`
- Accuracy: `0.7532`
- Precision: `0.6053`
- Recall: `0.8519`
- F1-Score: `0.7077`
- ROC-AUC: `0.8161`

Under the 10-fold cross validation comparison used to mirror the video-style boxplot, the average accuracy ranking may differ from the final split-based ranking. This is expected and is discussed in the report.

## Notes

- The repository includes the project code and dataset.
- Large local environments and generated runtime folders are ignored with `.gitignore`.
- The final report document itself is intentionally excluded from version control.

## Author

GitHub: [yagmurgurdal](https://github.com/yagmurgurdal)
