import re
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def normalize_text(text: str) -> str:
    """
    Lightweight text normalization.

    We intentionally do not remove stopwords or aggressively stem/lemmatize.
    Complaint wording such as 'not received', 'charged twice', and 'can't'
    can contain useful classification information.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize URLs without deleting the surrounding context.
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Normalize line breaks and repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Clean complaint text and target labels."""
    data = df[[text_column, target_column]].copy()

    data.columns = ["text", "category"]

    data["text"] = data["text"].fillna("").map(normalize_text)
    data["category"] = data["category"].fillna("").astype(str).str.strip()

    # Remove empty text and missing labels.
    data = data[
        (data["text"].str.len() > 0)
        & (data["category"].str.len() > 0)
    ]

    # Exact duplicate complaint narratives can otherwise leak across splits.
    data = data.drop_duplicates(subset=["text"])

    return data.reset_index(drop=True)


def filter_and_sample_classes(
    df: pd.DataFrame,
    min_samples: int = 50,
    max_samples_per_class: int = 2500,
) -> pd.DataFrame:
    """
    Preserve class diversity while keeping local training practical.
    """
    counts = df["category"].value_counts()

    valid_classes = counts[counts >= min_samples].index

    data = df[df["category"].isin(valid_classes)].copy()

    if max_samples_per_class:
        data = (
            data.groupby("category", group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=min(len(group), max_samples_per_class),
                    random_state=RANDOM_STATE,
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

    return data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    64/16/20 train/validation/test split.

    The final test set remains untouched during model selection.
    """
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["category"],
        random_state=RANDOM_STATE,
    )

    train, validation = train_test_split(
        train_val,
        test_size=0.20,
        stratify=train_val["category"],
        random_state=RANDOM_STATE,
    )

    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
