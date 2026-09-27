from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"

RANDOM_STATE = 42

TEXT_COLUMN = "Consumer complaint narrative"
TARGET_COLUMN = "Product"


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    return train, validation, test


def build_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=50000,
    )


def build_logistic_regression():
    return Pipeline(
        [
            (
                "tfidf",
                build_vectorizer(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_linear_svm():
    return Pipeline(
        [
            (
                "tfidf",
                build_vectorizer(),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_precision": precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_precision": precision_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": recall_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def main():

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading processed data...")

    train, validation, test = load_data()

    X_train = train[TEXT_COLUMN].fillna("")
    y_train = train[TARGET_COLUMN]

    X_val = validation[TEXT_COLUMN].fillna("")
    y_val = validation[TARGET_COLUMN]

    X_test = test[TEXT_COLUMN].fillna("")
    y_test = test[TARGET_COLUMN]

    # Combine training and validation after model configuration.
    X_train_final = pd.concat(
        [X_train, X_val],
        ignore_index=True,
    )

    y_train_final = pd.concat(
        [y_train, y_val],
        ignore_index=True,
    )

    models = {
        "Logistic Regression": build_logistic_regression(),
        "Linear SVM": build_linear_svm(),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():

        print("\n" + "=" * 60)
        print(f"Training {name}")
        print("=" * 60)

        start = time.perf_counter()

        model.fit(
            X_train_final,
            y_train_final,
        )

        training_time = time.perf_counter() - start

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        metrics["model"] = name
        metrics["training_time_seconds"] = training_time

        results.append(metrics)
        fitted_models[name] = model

        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Macro F1:     {metrics['macro_f1']:.4f}")
        print(f"Weighted F1:  {metrics['weighted_f1']:.4f}")
        print(f"Training time: {training_time:.2f}s")

    results_df = pd.DataFrame(results)

    # Select based primarily on macro F1.
    # Macro F1 gives each complaint category equal importance.
    best_row = results_df.sort_values(
        "macro_f1",
        ascending=False,
    ).iloc[0]

    best_model_name = best_row["model"]

    best_model = fitted_models[best_model_name]

    print("\n" + "=" * 60)
    print(f"Selected model: {best_model_name}")
    print("=" * 60)

    # Save complete fitted pipeline.
    model_path = (
        MODELS_DIR /
        "complaint_classifier.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    # Save comparison.
    results_df.to_csv(
        REPORTS_DIR /
        "model_comparison.csv",
        index=False,
    )

    # Save metadata.
    metadata = {
        "selected_model": best_model_name,
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "training_records": int(len(X_train_final)),
        "test_records": int(len(X_test)),
        "metrics": results,
    }

    with open(
        REPORTS_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(f"\nSaved model to:")
    print(model_path)

    print("\nModel comparison:")
    print(
        results_df[
            [
                "model",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "training_time_seconds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
