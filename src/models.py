from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import create_tfidf_vectorizer


def build_logistic_regression() -> Pipeline:
    """TF-IDF + Logistic Regression."""
    return Pipeline(
        [
            ("tfidf", create_tfidf_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_linear_svm() -> Pipeline:
    """TF-IDF + Linear SVM."""
    return Pipeline(
        [
            ("tfidf", create_tfidf_vectorizer()),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def get_models() -> dict:
    return {
        "Logistic Regression": build_logistic_regression(),
        "Linear SVM": build_linear_svm(),
    }
