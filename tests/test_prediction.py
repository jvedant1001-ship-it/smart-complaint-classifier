import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import load_model_bundle, predict_complaint


def test_model_loads():
    bundle = load_model_bundle()

    assert bundle is not None
    assert "pipeline" in bundle
    assert "model_name" in bundle


def test_prediction_returns_category():
    result = predict_complaint(
        "I have a problem with my financial account and need help."
    )

    assert isinstance(result, dict)
    assert result["predicted_category"]
    assert result["model_name"]


def test_prediction_structure():
    result = predict_complaint(
        "I was charged an unexpected amount."
    )

    expected_keys = {
        "predicted_category",
        "model_name",
        "score_type",
        "score",
        "top_predictions",
    }

    assert expected_keys.issubset(result.keys())


def test_empty_input():
    with pytest.raises(ValueError):
        predict_complaint("")


def test_whitespace_input():
    with pytest.raises(ValueError):
        predict_complaint("   ")
