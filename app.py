from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "complaint_classifier.joblib"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
DEMO_PATH = ROOT / "data" / "demo_complaints.csv"

REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"


st.set_page_config(
    page_title="Smart Complaint Classifier",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def top_predictions(model, text, n=3):
    scores = model.decision_function([text])[0]

    results = sorted(
        zip(model.classes_, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return results[:n]


def representative_terms(model, category, n=10):

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    features = vectorizer.get_feature_names_out()

    class_index = list(
        classifier.classes_
    ).index(category)

    coefficients = classifier.coef_[class_index]

    indices = coefficients.argsort()[-n:][::-1]

    return [
        features[i]
        for i in indices
    ]


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🧠 Smart Complaint Classifier")

st.markdown(
    "### AI-powered customer complaint routing and analysis"
)

st.divider()


# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():

    st.error(
        f"Model not found at:\n\n"
        f"`{MODEL_PATH}`"
    )

    st.code(
        "python scripts/train.py"
    )

    st.stop()


try:
    model = load_model()
    train, test = load_data()

except Exception as e:

    st.error(
        f"Could not load the trained system:\n\n{e}"
    )

    st.stop()


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Classify Complaint",
        "📊 Dataset Insights",
        "🔎 Category Explorer",
        "📈 Model Performance",
    ]
)


# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.header("Classify a Customer Complaint")

    complaint = st.text_area(
        "Enter customer complaint",
        height=180,
        placeholder=(
            "Example: I was charged a fee on my "
            "checking account that I do not understand."
        ),
    )

    if st.button(
        "🚀 Classify Complaint",
        type="primary",
    ):

        if not complaint.strip():

            st.warning(
                "Please enter a complaint."
            )

        else:

            prediction = model.predict(
                [complaint]
            )[0]

            candidates = top_predictions(
                model,
                complaint,
            )

            terms = representative_terms(
                model,
                prediction,
            )

            st.success(
                "Complaint classified successfully!"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Category",
                    prediction,
                )

            with col2:

                st.metric(
                    "Classification Score",
                    f"{candidates[0][1]:.3f}",
                )

            st.caption(
                "This is a Linear SVM decision-function "
                "score, not a probability."
            )

            st.subheader(
                "Top Candidate Categories"
            )

            for i, (category, score) in enumerate(
                candidates,
                1,
            ):

                st.write(
                    f"**{i}. {category}** — "
                    f"{score:.3f}"
                )

            st.subheader(
                "Representative Terms"
            )

            st.write(
                " • ".join(
                    f"`{term}`"
                    for term in terms
                )
            )

            st.info(
                "These terms are learned features "
                "associated with the category. "
                "They are not causal explanations "
                "of an individual prediction."
            )


# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.header("Dataset Insights")

    total = len(train) + len(test)

    categories = train["Product"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Clean Complaints",
        f"{total:,}",
    )

    col2.metric(
        "Categories",
        categories,
    )

    col3.metric(
        "Test Complaints",
        f"{len(test):,}",
    )

    st.subheader(
        "Complaint Category Distribution"
    )

    counts = pd.concat(
        [
            train["Product"],
            test["Product"],
        ]
    ).value_counts()

    st.bar_chart(counts)

    st.dataframe(
        counts.rename(
            "Complaints"
        ).reset_index(),
        use_container_width=True,
    )

    st.subheader("EDA Charts")

    category_chart = (
        FIGURES_DIR /
        "category_distribution.png"
    )

    length_chart = (
        FIGURES_DIR /
        "complaint_length_distribution.png"
    )

    words_chart = (
        FIGURES_DIR /
        "top_words.png"
    )

    if category_chart.exists():
        st.image(
            str(category_chart),
            caption="Category Distribution",
            use_container_width=True,
        )

    if length_chart.exists():
        st.image(
            str(length_chart),
            caption="Complaint Length Distribution",
            use_container_width=True,
        )

    if words_chart.exists():
        st.image(
            str(words_chart),
            caption="Most Common Words",
            use_container_width=True,
        )


# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.header("Category Explorer")

    categories = sorted(
        train["Product"].unique()
    )

    category = st.selectbox(
        "Select category",
        categories,
    )

    all_data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    category_data = all_data[
        all_data["Product"] == category
    ]

    st.metric(
        "Complaints",
        f"{len(category_data):,}",
    )

    st.subheader(
        "Representative Terms"
    )

    terms = representative_terms(
        model,
        category,
        12,
    )

    st.write(
        " • ".join(
            f"`{term}`"
            for term in terms
        )
    )

    st.subheader(
        "Example Complaints"
    )

    examples = category_data.head(3)

    for _, row in examples.iterrows():

        with st.expander(
            "View complaint"
        ):

            st.write(
                row["Consumer complaint narrative"]
            )


# ==================================================
# TAB 4
# ==================================================

with tab4:

    st.header("Model Performance")

    comparison_path = (
        REPORTS_DIR /
        "model_comparison.csv"
    )

    if comparison_path.exists():

        comparison = pd.read_csv(
            comparison_path
        )

        display = comparison.copy()

        for column in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
        ]:

            if column in display.columns:

                display[column] = (
                    display[column] * 100
                ).round(2)

        st.dataframe(
            display,
            use_container_width=True,
        )

        best = comparison.loc[
            comparison["macro_f1"].idxmax()
        ]

        st.success(
            f"Selected model: "
            f"**{best['model']}**"
        )

    confusion_path = (
        FIGURES_DIR /
        "confusion_matrix.png"
    )

    if confusion_path.exists():

        st.subheader(
            "Confusion Matrix"
        )

        st.image(
            str(confusion_path),
            use_container_width=True,
        )

    report_path = (
        REPORTS_DIR /
        "classification_report.txt"
    )

    if report_path.exists():

        with st.expander(
            "View Classification Report"
        ):

            st.code(
                report_path.read_text(
                    encoding="utf-8"
                )
            )


st.divider()

st.caption(
    "Smart Complaint Classification • "
    "TF-IDF + Linear Machine Learning"
)
