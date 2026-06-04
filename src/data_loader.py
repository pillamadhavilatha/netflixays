"""
src/data_loader.py
==================
Data ingestion and preprocessing for the Netflix dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Netflix titles CSV and return a raw DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download from: https://www.kaggle.com/datasets/shivamb/netflix-shows"
        )
    df = pd.read_csv(filepath)
    print(f"✅  Loaded {len(df):,} rows × {df.shape[1]} columns from '{path.name}'")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return a tidy DataFrame.

    Steps
    -----
    1. Fill missing categorical columns with sensible defaults.
    2. Drop exact duplicate rows.
    3. Strip leading / trailing whitespace from text columns.
    4. Parse date_added into datetime; extract year_added.
    5. Normalise duration into numeric minutes (movies) or season count (shows).
    """
    df = df.copy()

    # ── 1. Fill nulls ─────────────────────────────────────────────────────────
    fill_map = {
        "director": "Unknown",
        "cast":     "Unknown",
        "country":  "Unknown",
        "rating":   "Not Rated",
        "description": "",
        "listed_in":   "Unknown",
    }
    df.fillna(fill_map, inplace=True)

    # ── 2. Drop duplicates ───────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"🗑️   Removed {dropped} duplicate row(s).")

    # ── 3. Strip text ────────────────────────────────────────────────────────
    for col in ["title", "director", "cast", "country", "listed_in", "description"]:
        df[col] = df[col].str.strip()

    # ── 4. Date parsing ──────────────────────────────────────────────────────
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year

    # ── 5. Numeric duration ──────────────────────────────────────────────────
    def _parse_duration(row):
        if pd.isna(row["duration"]):
            return np.nan
        val = row["duration"]
        if "min" in str(val):
            return int(str(val).replace(" min", "").strip())
        if "Season" in str(val):
            return int(str(val).split(" ")[0])
        return np.nan

    df["duration_numeric"] = df.apply(_parse_duration, axis=1)

    print(f"✅  Cleaning complete. Final shape: {df.shape}")
    return df


def get_summary(df: pd.DataFrame) -> None:
    """Print a concise dataset summary."""
    print("\n" + "═" * 50)
    print("  Dataset Summary")
    print("═" * 50)
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"  Movies       : {(df['type']=='Movie').sum():,}")
    print(f"  TV Shows     : {(df['type']=='TV Show').sum():,}")
    print(f"  Null values  :\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    print("═" * 50 + "\n")
