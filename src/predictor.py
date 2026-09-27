from pathlib import Path
from typing import Dict

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "complaint_classifier.joblib"


def load_model_bundle():
    """Load the fitted model bundle."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found at models/complaint_classifier.joblib. "
            "Run preprocessing, training and evaluation first."
        )

    return joblib.load(MODEL_PATH)


def predict_complaint(text: str) -> Dict:
    """
    Predict a complaint category.

    Returns probability information for Logistic Regression and
    decision-score information for Linear SVM.
    """
    if text is None or not str(text).strip():
        raise ValueError("Complaint text cannot be empty.")

    bundle = load_model_bundle()

    pipeline = bundle["pipeline"]
    model_name = bundle["model_name"]

    text = str(text).strip()

    predicted_category = pipeline.predict([text])[0]

    result = {
        "predicted_category": str(predicted_category),
        "model_name": model_name,
        "score_type": None,
        "score": None,
        "top_predictions": [],
    }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_

        order = np.argsort(probabilities)[::-1][:3]

        result["score_type"] = "Probability"
        result["score"] = float(probabilities[order[0]])

        result["top_predictions"] = [
            {
                "category": str(classes[i]),
                "score": float(probabilities[i]),
            }
            for i in order
        ]

    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function([text])

        if scores.ndim == 1:
            # Binary SVM.
            classes = pipeline.classes_
            values = np.array([-scores[0], scores[0]])
        else:
            values = scores[0]
            classes = pipeline.classes_

        order = np.argsort(values)[::-1][:3]

        result["score_type"] = "Decision score"
        result["score"] = float(values[order[0]])

        result["top_predictions"] = [
            {
                "category": str(classes[i]),
                "score": float(values[i]),
            }
            for i in order
        ]

    return result


def get_representative_terms(category: str, n_terms: int = 8):
    """Return representative learned terms for a category."""
    bundle = load_model_bundle()

    # Import here to keep the public prediction function lightweight.
    from .evaluation import get_top_terms

    return get_top_terms(
        bundle["pipeline"],
        category,
        n_terms=n_terms,
    )
