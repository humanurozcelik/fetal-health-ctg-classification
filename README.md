# Fetal Health Classification from CTG-Derived Features

A machine-learning bootcamp project for multiclass classification of fetal health using **features extracted from cardiotocography (CTG) examinations**.

The project compares Logistic Regression, Support Vector Machine (SVM), and Random Forest models while emphasizing class imbalance, leakage-aware preprocessing, cross-validation, and hold-out evaluation.

> This repository works with tabular features derived from CTG recordings; it does not process raw CTG waveforms.

## Project Context

This project was developed as part of a data science / machine learning bootcamp to practice an end-to-end classification workflow on biomedical data.

The main learning goals were:

- handling an imbalanced multiclass dataset
- keeping preprocessing inside scikit-learn pipelines
- comparing models using cross-validation rather than the test set
- selecting a metric suitable for class imbalance
- tuning the selected model with GridSearchCV
- interpreting a confusion matrix and feature-importance output without treating them as clinical evidence

A longer Turkish write-up is available on Medium:
https://medium.com/@humanurozcelik555/ctg-verileriyle-fetal-sa%C4%9Fl%C4%B1k-s%C4%B1n%C4%B1fland%C4%B1rmas%C4%B1-%C3%BC%C3%A7-makine-%C3%B6%C4%9Frenmesi-modelinin-kar%C5%9F%C4%B1la%C5%9Ft%C4%B1r%C4%B1lmas%C4%B1-c5b962e5379a?sharedUserId=humanurozcelik555

## Dataset

The project uses the **Fetal Health Classification** dataset distributed on Kaggle:

https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

The dataset contains **2,126 records** and **21 CTG-derived input features**, with the target variable `fetal_health` represented by three classes:

- `1.0` — Normal
- `2.0` — Suspect
- `3.0` — Pathological

The dataset description credits Ayres de Campos et al. and the SisPorto 2.0 work. In this project, 13 duplicate rows are removed before modeling, leaving **2,113 unique observations**.

## Methodology

The workflow is intentionally simple and reproducible:

1. Load the dataset and inspect missing values and duplicates.
2. Remove duplicate observations.
3. Visualize class distribution and feature correlations.
4. Create a stratified **80/20 train-test split**.
5. Compare three classifiers on the training set using **5-fold cross-validation** and **Macro F1**:
   - Logistic Regression
   - Support Vector Machine
   - Random Forest
6. Use `class_weight="balanced"` to reduce majority-class dominance.
7. Keep `StandardScaler` inside scikit-learn `Pipeline` objects for Logistic Regression and SVM.
8. Tune Random Forest hyperparameters using `GridSearchCV` on the training set.
9. Evaluate the selected model once on the untouched hold-out test set.
10. Inspect the confusion matrix and Random Forest feature importances.

## Reported Results

The original project run produced the following results:

- **Random Forest CV Macro F1:** approximately 0.885
- **Hold-out accuracy:** approximately 94.6%
- **Hold-out Macro F1:** approximately 0.91
- **Pathological-class recall:** approximately 97% (34 of 35 pathological test samples)

These values describe this specific train/test split and modeling workflow; they should not be interpreted as external clinical validation.

The Random Forest assigned relatively high importance to features including `abnormal_short_term_variability` and `percentage_of_time_with_abnormal_long_term_variability`. Feature importance indicates model reliance, not biological or clinical causality.

## Visual Outputs

### Class Distribution

![Class distribution](1_class_distribution.png)

### Model Comparison

![Model comparison](5_model_comparison.png)

### Confusion Matrix

![Confusion matrix](3_confusion_matrix.PNG)

### Feature Importance

![Feature importance](4_feature_importance.png)

## Repository Structure

```text
.
├── fetal_health_classification.py
├── fetal_health.xls
├── requirements.txt
├── 1_class_distribution.png
├── 2_correlation_heatmap.png
├── 3_confusion_matrix.PNG
├── 4_feature_importance.png
├── 5_model_comparison.png
├── README.md
└── LICENSE
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run

Place `fetal_health.xls` in the repository root and run:

```bash
python fetal_health_classification.py
```

The script prints model-comparison and final evaluation results and regenerates the analysis figures in the repository directory.

## Scope and Limitations

This is an **educational machine-learning project**, not a diagnostic system.

Important limitations include:

- a single public dataset
- no external validation
- no raw CTG waveform processing
- a limited model and hyperparameter search space
- feature importance is model-specific and non-causal
- reported performance is specific to the chosen data split and evaluation procedure

The project should therefore be interpreted as a demonstration of biomedical-data classification methodology rather than evidence of clinical performance.
