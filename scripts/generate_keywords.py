from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "complaint_classifier.joblib"
OUTPUT_PATH = (
    ROOT /
    "outputs" /
    "reports" /
    "category_keywords.csv"
)


def main():

    model = joblib.load(MODEL_PATH)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    feature_names = vectorizer.get_feature_names_out()

    rows = []

    for class_index, category in enumerate(
        classifier.classes_
    ):

        coefficients = classifier.coef_[
            class_index
        ]

        top_indices = coefficients.argsort()[
            -10:
        ][::-1]

        terms = [
            feature_names[index]
            for index in top_indices
        ]

        rows.append(
            {
                "Category": category,
                "Top Terms": ", ".join(terms),
            }
        )

    result = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved category keywords to:\n"
        f"{OUTPUT_PATH}"
    )

    print("\nRepresentative terms:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
