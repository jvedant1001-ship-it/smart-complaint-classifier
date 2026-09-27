from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def find_raw_csv(raw_dir: Optional[Path] = None) -> Path:
    """Find the first CSV under data/raw."""
    raw_dir = raw_dir or RAW_DIR

    csv_files = sorted(raw_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file was found in data/raw/. "
            "Run `python scripts/download_data.py` or place a CFPB CSV there."
        )

    return csv_files[0]


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw complaint CSV."""
    path = path or find_raw_csv()

    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def load_processed_split(name: str) -> pd.DataFrame:
    """Load train/validation/test processed data."""
    path = PROCESSED_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python scripts/preprocess.py` first."
        )

    return pd.read_csv(path)


def find_column(df: pd.DataFrame, desired: str) -> str:
    """
    Find a column case-insensitively and ignoring minor punctuation differences.
    """
    normalized = {
        str(column).strip().lower().replace("_", " "): column
        for column in df.columns
    }

    key = desired.strip().lower().replace("_", " ")

    if key in normalized:
        return normalized[key]

    # Fuzzy matching for CFPB-style column names.
    for normalized_name, original_name in normalized.items():
        if key in normalized_name or normalized_name in key:
            return original_name

    raise KeyError(
        f"Could not find column '{desired}'. "
        f"Available columns: {list(df.columns)}"
    )
