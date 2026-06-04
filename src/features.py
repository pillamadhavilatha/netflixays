"""
src/features.py
===============
Feature engineering for the content-based recommender.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def build_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single text blob per title by concatenating:
      listed_in + description + director + cast
    This becomes the input for TF-IDF vectorisation.
    """
    df = df.copy()
    df["combined_features"] = (
        df["listed_in"].fillna("")   + " " +
        df["description"].fillna("") + " " +
        df["director"].fillna("")    + " " +
        df["cast"].fillna("")
    )
    print("✅  Combined feature column built.")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode categorical columns for potential downstream ML use.
    Adds: type_enc, rating_enc, country_enc
    """
    df = df.copy()
    le = LabelEncoder()
    for col in ["type", "rating", "country"]:
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    print("✅  Categorical columns encoded.")
    return df


def compute_tfidf(df: pd.DataFrame,
                  column: str = "combined_features",
                  max_features: int = 10_000) -> tuple:
    """
    Fit a TF-IDF vectoriser on `column`.

    Returns
    -------
    tfidf_matrix : scipy sparse matrix  (n_titles × n_terms)
    vectorizer   : fitted TfidfVectorizer
    """
    tfidf = TfidfVectorizer(stop_words="english",
                            max_features=max_features,
                            ngram_range=(1, 2))
    matrix = tfidf.fit_transform(df[column].fillna(""))
    print(f"✅  TF-IDF matrix shape: {matrix.shape}  "
          f"(titles × terms)")
    return matrix, tfidf
