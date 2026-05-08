# Diabetes Risk Analysis

This repository presents an end-to-end machine learning study on **diabetes risk prediction** using the `diabetes.csv` dataset. The project was developed as a compact healthcare analytics workflow that moves from raw clinical measurements to model comparison, evaluation, and report-ready outputs.

Rather than relying on a single algorithm, the project compares multiple classification approaches, applies targeted preprocessing, and discusses model quality from a health decision-support perspective.

## Why This Project Matters

Diabetes risk estimation is not just a modeling exercise. In real decision-support settings, the cost of missing a high-risk patient can be more important than maximizing a single headline metric such as accuracy.

This project focuses on that idea by combining:

- data cleaning for medically unrealistic values
- feature engineering for stronger clinical signal
- comparison across 10 classification models
- evaluation with recall, F1-score, ROC-AUC, and confusion matrix analysis

## Project Highlights

- uses `diabetes.csv` as the core dataset
- treats biologically unrealistic `0` values as missing-like entries in selected variables
- applies median-based imputation
- creates engineered risk-oriented features and interaction terms
- compares 10 classification algorithms in one pipeline
- includes both split-based evaluation and 10-fold cross validation
- generates report assets such as ROC curves, confusion matrix visuals, and model comparison charts

## Dataset

The dataset contains clinical variables commonly used in diabetes risk analysis:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

Target variable:

- `Outcome`

The prediction task is binary classification:

- `0`: lower observed diabetes risk group
- `1`: diabetes risk group

## Data Preparation

The preprocessing workflow is one of the most important parts of the project.

Applied steps:

- inspection of missing-like values
- replacement of unrealistic `0` values in selected medical fields
- median imputation
- scaling for models sensitive to feature magnitude
- creation of additional engineered features

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

## Models Compared

The project evaluates 10 classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors
3. Support Vector Machine
4. Decision Tree
5. Random Forest
6. Extra Trees
7. Gradient Boosting
8. AdaBoost
9. Gaussian Naive Bayes
10. Multi-Layer Perceptron

## Evaluation Strategy

Two evaluation views are used:

1. final train/test style comparison for model selection
2. stratified 10-fold cross validation for broader stability comparison

Main metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

The project gives special attention to **recall**, because false negatives are especially important in healthcare-oriented prediction problems.

## Current Best Result

In the latest split-based evaluation, the strongest model is:

- Model: `Logistic Regression`
- Accuracy: `0.7532`
- Precision: `0.6053`
- Recall: `0.8519`
- F1-Score: `0.7077`
- ROC-AUC: `0.8161`

The cross-validation ranking can differ from the final split-based ranking, which is expected in small and moderately noisy tabular datasets.

## Repository Structure

- `diabetes_project.py`
  Main training and evaluation pipeline for all 10 models.

- `generate_cv_comparison.py`
  Produces 10-fold cross-validation accuracy results and boxplot-style comparison outputs.

- `generate_report_assets.py`
  Generates visuals such as class distribution, confusion matrix, ROC curve, and performance charts.

- `edit_docx_report.py`
  Builds the final report document from the generated outputs.

- `diabetes.csv`
  Dataset used for training and evaluation.

## How to Run

Create a Python environment and install the main dependencies used in the project:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl python-docx
```

Run the full model comparison pipeline:

```bash
python diabetes_project.py
```

Run the cross-validation comparison:

```bash
python generate_cv_comparison.py
```

Generate report visuals:

```bash
python generate_report_assets.py
```

Generate the report document:

```bash
python edit_docx_report.py
```

## Generated Outputs

When executed locally, the project produces artifacts under `results/`, such as:

- `model_comparison.csv`
- `best_model_predictions.csv`
- `metrics.json`
- `cv_accuracy_scores.csv`
- `cv_accuracy_summary.csv`
- report images under `results/report_assets/`

Generated outputs are intentionally excluded from version control where appropriate.

## What This Repository Shows

- applied machine learning on a healthcare classification problem
- practical preprocessing for imperfect medical data
- feature engineering on tabular clinical variables
- model comparison beyond a single accuracy score
- report-oriented analysis for academic or portfolio presentation

## Author

GitHub: [yagmurgurdal](https://github.com/yagmurgurdal)
