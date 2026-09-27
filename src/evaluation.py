from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(
    y_true,
    y_pred,
    training_time: float | None = None,
) -> Dict:
    """Calculate multi-class classification metrics."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
    }

    if training_time is not None:
        result["training_time_seconds"] = float(training_time)

    return result


def get_classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """Return sklearn classification report as a DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).transpose()


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path: Path,
    figsize: tuple[int, int] = (12, 10),
):
    """Save a confusion matrix image."""
    labels = sorted(set(y_true) | set(y_pred))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title("Complaint Classification Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def get_top_terms(
    pipeline,
    category: str,
    n_terms: int = 10,
) -> list[str]:
    """
    Get representative learned features for a category.

    These are model-associated terms, not causal explanations.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = np.array(classifier.classes_)

    class_index = np.where(classes == category)[0]

    if len(class_index) == 0:
        return []

    class_index = class_index[0]

    coefficients = classifier.coef_

    # Binary classifiers may expose a single coefficient row.
    if coefficients.shape[0] == 1 and len(classes) == 2:
        if class_index == 0:
            weights = -coefficients[0]
        else:
            weights = coefficients[0]
    else:
        weights = coefficients[class_index]

    top_indices = np.argsort(weights)[::-1][:n_terms]

    return feature_names[top_indices].tolist()


def build_category_keyword_table(
    pipeline,
    categories,
    n_terms: int = 10,
) -> pd.DataFrame:
    rows = []

    for category in sorted(categories):
        terms = get_top_terms(
            pipeline,
            category,
            n_terms=n_terms,
        )

        rows.append(
            {
                "category": category,
                "top_terms": ", ".join(terms),
            }
        )

    return pd.DataFrame(rows)
