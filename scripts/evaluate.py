from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "complaint_classifier.joblib"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

TEXT_COLUMN = "Consumer complaint narrative"
TARGET_COLUMN = "Product"


def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python scripts/train.py"
        )

    model = joblib.load(MODEL_PATH)

    print("Loading test data...")

    test = pd.read_csv(TEST_PATH)

    X_test = test[TEXT_COLUMN].fillna("")
    y_test = test[TARGET_COLUMN]

    print("Generating predictions...")

    predictions = model.predict(X_test)

    # Overall metrics
    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "macro_precision": float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    print("\nOverall Metrics")
    print("=" * 50)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Classification report
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    report_path = (
        REPORTS_DIR /
        "classification_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"\nClassification report saved to:\n"
        f"{report_path}"
    )

    # Save JSON metrics
    metrics_path = (
        REPORTS_DIR /
        "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # Confusion matrix
    labels = sorted(y_test.unique())

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    plt.figure(
        figsize=(14, 11)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(
        "Complaint Classification Confusion Matrix"
    )

    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.yticks(
        rotation=0,
    )

    plt.tight_layout()

    confusion_path = (
        FIGURES_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Confusion matrix saved to:\n"
        f"{confusion_path}"
    )

    # Per-class metrics
    report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    per_class = []

    for category in labels:

        row = report_dict.get(
            category,
            {},
        )

        per_class.append(
            {
                "Category": category,
                "Precision": row.get(
                    "precision",
                    0,
                ),
                "Recall": row.get(
                    "recall",
                    0,
                ),
                "F1": row.get(
                    "f1-score",
                    0,
                ),
                "Support": row.get(
                    "support",
                    0,
                ),
            }
        )

    per_class_df = pd.DataFrame(
        per_class
    )

    per_class_df.to_csv(
        REPORTS_DIR /
        "per_class_metrics.csv",
        index=False,
    )

    print("\nEvaluation complete.")

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
