"""
src/recommender.py
==================
Content-based Netflix recommendation engine.

Usage
-----
    from src.recommender import NetflixRecommender

    rec = NetflixRecommender("data/netflix_titles.csv")
    rec.fit()
    print(rec.recommend("Stranger Things"))
    print(rec.recommend("The Irishman", top_n=5))
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_data, clean_data
from src.features   import build_combined_features, compute_tfidf


class NetflixRecommender:
    """Content-based recommender using TF-IDF + cosine similarity."""

    def __init__(self, data_path: str = "data/netflix_titles.csv"):
        self.data_path  = data_path
        self.df         = None
        self.tfidf_mat  = None
        self.cosine_sim = None
        self._indices   = None
        self._fitted    = False

    # ── Build ──────────────────────────────────────────────────────────────────
    def fit(self) -> "NetflixRecommender":
        """Load data, engineer features, and compute the similarity matrix."""
        print("\n🚀  Building recommendation engine …")

        raw = load_data(self.data_path)
        self.df = clean_data(raw)
        self.df = build_combined_features(self.df)

        self.tfidf_mat, _ = compute_tfidf(self.df)

        print("⏳  Computing cosine similarity matrix …")
        self.cosine_sim = cosine_similarity(self.tfidf_mat, self.tfidf_mat)
        print(f"✅  Similarity matrix: {self.cosine_sim.shape}")

        # Map title → index (drop duplicate titles — keep first occurrence)
        self._indices = pd.Series(
            self.df.index, index=self.df["title"]
        ).drop_duplicates()

        self._fitted = True
        print("🎬  Recommender ready.\n")
        return self

    # ── Recommend ──────────────────────────────────────────────────────────────
    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Return the top-N most similar titles.

        Parameters
        ----------
        title : str   — exact title as it appears in the dataset
        top_n : int   — number of recommendations (default 10)

        Returns
        -------
        pd.DataFrame with columns: title, type, listed_in, rating, similarity
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .recommend().")

        if title not in self._indices.index:
            # Fuzzy hint
            close = [t for t in self._indices.index
                     if title.lower() in t.lower()][:5]
            hint = f"\n  Did you mean: {close}" if close else ""
            raise ValueError(f"Title '{title}' not found in dataset.{hint}")

        idx    = self._indices[title]
        scores = list(enumerate(self.cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [(i, s) for i, s in scores if i != idx][:top_n]

        rec_idx = [i for i, _ in scores]
        sims    = [round(s, 4) for _, s in scores]

        result = self.df[["title", "type", "listed_in", "rating"]].iloc[rec_idx].copy()
        result["similarity"] = sims
        result.reset_index(drop=True, inplace=True)
        result.index += 1   # 1-based rank
        return result

    # ── Batch ──────────────────────────────────────────────────────────────────
    def batch_recommend(self, titles: list, top_n: int = 5) -> dict:
        """Run recommend() for multiple titles. Returns dict {title: DataFrame}."""
        results = {}
        for t in titles:
            try:
                results[t] = self.recommend(t, top_n=top_n)
            except ValueError as e:
                results[t] = str(e)
        return results

    # ── Search ─────────────────────────────────────────────────────────────────
    def search(self, query: str) -> list:
        """Return all titles containing `query` (case-insensitive)."""
        return [t for t in self._indices.index
                if query.lower() in t.lower()]


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from src.eda import run_all_eda

    # ── 1. EDA ────────────────────────────────────────────────────────────────
    from src.data_loader import load_data, clean_data
    raw = load_data("data/netflix_titles.csv")
    df  = clean_data(raw)
    run_all_eda(df)

    # ── 2. Recommendations ───────────────────────────────────────────────────
    rec = NetflixRecommender("data/netflix_titles.csv")
    rec.fit()

    demo_titles = ["Stranger Things", "The Irishman"]
    for title in demo_titles:
        try:
            recs = rec.recommend(title, top_n=10)
            print(f"\n🎬  Top 10 recommendations for '{title}':")
            print(recs.to_string())
        except ValueError as e:
            print(f"⚠️  {e}")
