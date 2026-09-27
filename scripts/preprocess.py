from pathlib import Path
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_FIGURES = ROOT / "outputs" / "figures"
OUTPUT_REPORTS = ROOT / "outputs" / "reports"
DEMO_FILE = ROOT / "data" / "demo_complaints.csv"

RANDOM_STATE = 42

TEXT_COLUMN = "Consumer complaint narrative"
TARGET_COLUMN = "Product"


def make_directories():
    for directory in [
        PROCESSED_DIR,
        OUTPUT_FIGURES,
        OUTPUT_REPORTS,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def find_best_csv():
    """Select the CSV containing the most complaint narratives."""
    csv_files = list(RAW_DIR.glob("*.csv"))

    candidates = []

    for file in csv_files:
        try:
            df = pd.read_csv(
                file,
                usecols=[TEXT_COLUMN, TARGET_COLUMN],
                low_memory=False,
            )

            narrative_count = df[TEXT_COLUMN].notna().sum()

            print(
                f"{file.name}: "
                f"{len(df):,} rows | "
                f"{narrative_count:,} narratives"
            )

            if narrative_count > 0:
                candidates.append((file, narrative_count))

        except Exception as exc:
            print(f"Skipping {file.name}: {exc}")

    if not candidates:
        raise FileNotFoundError(
            "No usable CFPB CSV with complaint narratives was found "
            "inside data/raw/."
        )

    best_file = max(candidates, key=lambda x: x[1])[0]

    print(f"\nUsing dataset: {best_file.name}")

    return best_file


def clean_text(text):
    """Light normalization that preserves useful language."""
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_eda(df):
    """Generate basic exploratory charts."""

    sns.set_theme(style="whitegrid")

    # Category distribution.
    plt.figure(figsize=(12, 7))

    counts = df[TARGET_COLUMN].value_counts()

    sns.barplot(
        x=counts.values,
        y=counts.index,
        color="#2563eb",
    )

    plt.title("Complaint Category Distribution")
    plt.xlabel("Number of Complaints")
    plt.ylabel("Category")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURES / "category_distribution.png",
        dpi=150,
    )
    plt.close()

    # Text length.
    df["text_length"] = df[TEXT_COLUMN].str.split().str.len()

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["text_length"].clip(upper=500),
        bins=50,
        color="#7c3aed",
    )

    plt.title("Complaint Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Complaints")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURES / "complaint_length_distribution.png",
        dpi=150,
    )
    plt.close()

    # Common words.
    from collections import Counter

    words = []

    for text in df[TEXT_COLUMN]:
        words.extend(
            word
            for word in text.split()
            if len(word) >= 3
        )

    common_words = Counter(words).most_common(20)

    word_df = pd.DataFrame(
        common_words,
        columns=["word", "count"],
    )

    plt.figure(figsize=(10, 7))

    sns.barplot(
        data=word_df,
        x="count",
        y="word",
        color="#059669",
    )

    plt.title("Most Common Complaint Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURES / "top_words.png",
        dpi=150,
    )
    plt.close()

    return word_df


def create_demo_file(df):
    """Create demo complaints directly from real dataset examples."""

    demo = (
        df.groupby(TARGET_COLUMN, group_keys=False)
        .head(1)
        [[TEXT_COLUMN, TARGET_COLUMN]]
        .rename(
            columns={
                TEXT_COLUMN: "text",
                TARGET_COLUMN: "expected_category",
            }
        )
    )

    demo.to_csv(DEMO_FILE, index=False)

    print(f"Demo complaints saved to: {DEMO_FILE}")


def main():
    make_directories()

    print("Loading CFPB narrative dataset...")

    csv_file = find_best_csv()

    df = pd.read_csv(
        csv_file,
        usecols=[TEXT_COLUMN, TARGET_COLUMN],
        low_memory=False,
    )

    print(f"Raw records: {len(df):,}")

    # Keep only rows with both text and target.
    df = df.dropna(
        subset=[TEXT_COLUMN, TARGET_COLUMN]
    )

    # Normalize text.
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

    # Remove empty text.
    df = df[df[TEXT_COLUMN].str.len() > 20]

    # Remove duplicates.
    df = df.drop_duplicates(
        subset=[TEXT_COLUMN, TARGET_COLUMN]
    )

    print(f"After cleaning: {len(df):,}")

    # Keep all categories represented in the archive.
    # There are only ~10k narrative records, so no aggressive
    # class filtering is necessary.
    print(
        f"Categories: {df[TARGET_COLUMN].nunique()}"
    )

    print("\nCategory distribution:")
    print(df[TARGET_COLUMN].value_counts())

    # Create EDA.
    common_words = create_eda(df)

    # Stratified split.
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False,
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False,
    )

    # Save a small validation set from training data.
    train_df, validation_df = train_test_split(
        train_df,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=train_df[TARGET_COLUMN],
    )

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False,
    )

    # Demo complaints.
    create_demo_file(df)

    # Dataset summary.
    summary = {
        "source_file": csv_file.name,
        "total_clean_records": int(len(df)),
        "number_of_categories": int(
            df[TARGET_COLUMN].nunique()
        ),
        "train_records": int(len(train_df)),
        "validation_records": int(len(validation_df)),
        "test_records": int(len(test_df)),
        "category_counts": {
            str(k): int(v)
            for k, v in df[TARGET_COLUMN]
            .value_counts()
            .items()
        },
        "average_words": float(
            df[TEXT_COLUMN]
            .str.split()
            .str.len()
            .mean()
        ),
        "median_words": float(
            df[TEXT_COLUMN]
            .str.split()
            .str.len()
            .median()
        ),
    }

    with open(
        OUTPUT_REPORTS / "dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    common_words.to_csv(
        OUTPUT_REPORTS / "common_words.csv",
        index=False,
    )

    print("\nPreprocessing complete.")
    print(
        f"Train: {len(train_df):,} | "
        f"Validation: {len(validation_df):,} | "
        f"Test: {len(test_df):,}"
    )


if __name__ == "__main__":
    main()
