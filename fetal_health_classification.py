"""
Fetal Health Classification from CTG-Derived Features

Bootcamp machine-learning project for three-class classification:
1 = Normal, 2 = Suspect, 3 = Pathological.

Workflow:
- duplicate removal
- stratified train/test split
- leakage-aware scaling with scikit-learn Pipeline
- 5-fold cross-validation using Macro F1
- Random Forest hyperparameter tuning
- final hold-out evaluation

This is an educational project and not a clinically validated diagnostic system.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
DATA_FILE = Path(__file__).resolve().parent / "fetal_health.xls"


def save_class_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="fetal_health",
        hue="fetal_health",
        palette="Set2",
        legend=False,
    )
    plt.title("Fetal Health Class Distribution")
    plt.ylabel("Number of Samples")
    plt.xlabel("Fetal Health Class (1: Normal, 2: Suspect, 3: Pathological)")
    plt.tight_layout()
    plt.savefig("1_class_distribution.png", dpi=300)
    plt.close()


def save_correlation_heatmap(df):
    plt.figure(figsize=(14, 12))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of CTG-Derived Features")
    plt.tight_layout()
    plt.savefig("2_correlation_heatmap.png", dpi=300)
    plt.close()


def build_models():
    logistic_regression = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return {
        "Logistic Regression": logistic_regression,
        "SVM": svm,
        "Random Forest": random_forest,
    }, random_forest


def save_model_comparison(cv_results):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=list(cv_results.keys()),
        y=list(cv_results.values()),
        hue=list(cv_results.keys()),
        palette="viridis",
        legend=False,
    )
    plt.title("5-Fold Cross-Validation Performance")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)

    for index, value in enumerate(cv_results.values()):
        plt.text(
            index,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("5_model_comparison.png", dpi=300)
    plt.close()


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Suspect", "Pathological"],
        yticklabels=["Normal", "Suspect", "Pathological"],
    )
    plt.title("Final Model Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig("3_confusion_matrix.PNG", dpi=300)
    plt.close()


def save_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=importances[indices],
        y=feature_names[indices],
        hue=feature_names[indices],
        palette="viridis",
        legend=False,
    )
    plt.title("Top 10 Random Forest Feature Importances")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("4_feature_importance.png", dpi=300)
    plt.close()


def main():
    print("--- Step 1: Load and clean data ---")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place fetal_health.xls in the repository root before running the script."
        )

    df = pd.read_excel(DATA_FILE)

    print(f"Initial dataset shape: {df.shape}")
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dataset shape after duplicate removal: {df.shape}")

    save_class_distribution(df)
    save_correlation_heatmap(df)

    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]

    # The hold-out test set is separated before model comparison and tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n--- Step 2: Compare models with 5-fold cross-validation ---")

    models, random_forest = build_models()
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
        )
        cv_results[name] = scores.mean()
        print(f"{name}: CV Macro F1 = {scores.mean():.4f}")

    save_model_comparison(cv_results)

    print("\n--- Step 3: Tune Random Forest ---")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=random_forest,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)
    print(f"Best CV Macro F1: {grid_search.best_score_:.4f}")

    print("\n--- Step 4: Final hold-out evaluation ---")

    y_pred = best_rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    save_confusion_matrix(y_test, y_pred)
    save_feature_importance(best_rf, X.columns)

    print("\nAnalysis complete. Figures have been saved to the repository directory.")


if __name__ == "__main__":
    main()
