"""
tests/test_recommender.py
=========================
Unit tests for the Netflix Recommender pipeline.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np

from src.data_loader  import clean_data
from src.features     import build_combined_features, compute_tfidf
from src.recommender  import NetflixRecommender


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal in-memory DataFrame that mirrors netflix_titles.csv structure."""
    data = {
        "show_id":      ["s1", "s2", "s3", "s4", "s5"],
        "type":         ["Movie", "TV Show", "Movie", "Movie", "TV Show"],
        "title":        ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "director":     ["Dir A", "Dir B", None, "Dir D", "Dir E"],
        "cast":         ["Actor X", "Actor Y", "Actor Z", None, "Actor W"],
        "country":      ["US", None, "UK", "US", "India"],
        "date_added":   ["January 1, 2020", "March 5, 2019", None, "June 10, 2021", "December 1, 2022"],
        "release_year": [2019, 2018, 2020, 2021, 2022],
        "rating":       ["TV-MA", None, "PG-13", "R", "TV-14"],
        "duration":     ["120 min", "2 Seasons", "90 min", "100 min", "1 Season"],
        "listed_in":    ["Drama, Thriller", "Comedy", "Action", "Drama", "Romance"],
        "description":  [
            "A thriller about secrets",
            "A comedy about life",
            "Explosive action scenes",
            "Deep emotional drama",
            "A love story unfolds",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def clean_df(sample_df):
    return clean_data(sample_df)


@pytest.fixture
def featured_df(clean_df):
    return build_combined_features(clean_df)


# ── Data Loader Tests ─────────────────────────────────────────────────────────

class TestCleanData:
    def test_fills_null_director(self, clean_df):
        assert clean_df["director"].isna().sum() == 0

    def test_fills_null_rating(self, clean_df):
        assert "Not Rated" in clean_df["rating"].values

    def test_no_duplicates(self, clean_df):
        assert clean_df.duplicated().sum() == 0

    def test_year_added_parsed(self, clean_df):
        non_null = clean_df["year_added"].dropna()
        assert all(non_null > 2000)

    def test_duration_numeric_movies(self, clean_df):
        movies = clean_df[clean_df["type"] == "Movie"]
        assert movies["duration_numeric"].dropna().gt(0).all()


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatures:
    def test_combined_features_exists(self, featured_df):
        assert "combined_features" in featured_df.columns

    def test_combined_features_no_null(self, featured_df):
        assert featured_df["combined_features"].isna().sum() == 0

    def test_combined_features_contains_genre(self, featured_df):
        assert "Drama" in featured_df.loc[0, "combined_features"]

    def test_tfidf_shape(self, featured_df):
        matrix, _ = compute_tfidf(featured_df)
        assert matrix.shape[0] == len(featured_df)
        assert matrix.shape[1] > 0


# ── Recommender Tests ─────────────────────────────────────────────────────────

class TestRecommender:
    def test_not_fitted_raises(self):
        rec = NetflixRecommender.__new__(NetflixRecommender)
        rec._fitted = False
        with pytest.raises(RuntimeError):
            rec.recommend("Alpha")

    def test_fit_and_recommend(self, tmp_path, sample_df):
        csv = tmp_path / "netflix_titles.csv"
        sample_df.to_csv(csv, index=False)
        rec = NetflixRecommender(str(csv))
        rec.fit()
        result = rec.recommend("Alpha", top_n=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 3
        assert "title" in result.columns
        assert "similarity" in result.columns

    def test_recommend_excludes_self(self, tmp_path, sample_df):
        csv = tmp_path / "netflix_titles.csv"
        sample_df.to_csv(csv, index=False)
        rec = NetflixRecommender(str(csv))
        rec.fit()
        result = rec.recommend("Alpha")
        assert "Alpha" not in result["title"].values

    def test_invalid_title_raises(self, tmp_path, sample_df):
        csv = tmp_path / "netflix_titles.csv"
        sample_df.to_csv(csv, index=False)
        rec = NetflixRecommender(str(csv))
        rec.fit()
        with pytest.raises(ValueError):
            rec.recommend("This Title Does Not Exist")

    def test_similarity_scores_between_0_and_1(self, tmp_path, sample_df):
        csv = tmp_path / "netflix_titles.csv"
        sample_df.to_csv(csv, index=False)
        rec = NetflixRecommender(str(csv))
        rec.fit()
        result = rec.recommend("Beta", top_n=4)
        assert (result["similarity"] >= 0).all()
        assert (result["similarity"] <= 1).all()

    def test_search(self, tmp_path, sample_df):
        csv = tmp_path / "netflix_titles.csv"
        sample_df.to_csv(csv, index=False)
        rec = NetflixRecommender(str(csv))
        rec.fit()
        hits = rec.search("alp")
        assert "Alpha" in hits
