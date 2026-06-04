"""
src/eda.py
==========
Exploratory Data Analysis: all visualisations for the Netflix dataset.
Each function saves a PNG to `outputs/` and optionally displays inline.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
DARK   = "#141414"    # Netflix black
RED    = "#E50914"    # Netflix red
LIGHT  = "#1f1f1f"
MUTED  = "#888888"
WHITE  = "#F5F5F1"
PALETTE = [RED, "#F5F5F1", "#E87040", "#B81D24", "#831010", "#FABC2A", "#3A86FF"]

def _save(name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"📊  Saved → {path}")

def _style_ax(ax):
    ax.set_facecolor(LIGHT)
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.title.set_color(WHITE)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


# ── 1. Movies vs TV Shows ─────────────────────────────────────────────────────
def plot_type_distribution(df: pd.DataFrame) -> None:
    counts = df["type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK)
    bars = ax.bar(counts.index, counts.values,
                  color=[RED, WHITE], edgecolor="#333", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40, f"{val:,}",
                ha="center", color=WHITE, fontsize=11)
    ax.set_title("Movies vs TV Shows", fontsize=14, pad=12)
    ax.set_ylabel("Count")
    _style_ax(ax)
    _save("01_type_distribution.png")


# ── 2. Top 15 Genres ─────────────────────────────────────────────────────────
def plot_top_genres(df: pd.DataFrame, top_n: int = 15) -> None:
    genres = df["listed_in"].str.split(", ").explode()
    top = genres.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK)
    colors = [RED if i == 0 else "#555" for i in range(top_n)]
    top.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
    ax.set_title(f"Top {top_n} Genres on Netflix", fontsize=14, pad=12)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of Titles")
    plt.xticks(rotation=45, ha="right")
    _style_ax(ax)
    _save("02_top_genres.png")


# ── 3. Content Added Per Year ─────────────────────────────────────────────────
def plot_content_over_time(df: pd.DataFrame) -> None:
    year_counts = df["year_added"].dropna().astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
    ax.fill_between(year_counts.index, year_counts.values,
                    alpha=0.3, color=RED)
    ax.plot(year_counts.index, year_counts.values,
            color=RED, lw=2.5, marker="o", markersize=5)
    ax.set_title("Content Added to Netflix Per Year", fontsize=14, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Titles Added")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _style_ax(ax)
    _save("03_content_over_time.png")


# ── 4. Release Year Distribution ─────────────────────────────────────────────
def plot_release_year(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
    sns.histplot(df["release_year"].dropna(), bins=30,
                 color=RED, edgecolor="none", ax=ax)
    ax.set_title("Release Year Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Count")
    _style_ax(ax)
    ax.set_facecolor(LIGHT)
    _save("04_release_year.png")


# ── 5. Rating Distribution ────────────────────────────────────────────────────
def plot_ratings(df: pd.DataFrame) -> None:
    order = df["rating"].value_counts().index
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK)
    colors = [RED if i == 0 else "#555" for i in range(len(order))]
    sns.countplot(y="rating", data=df, order=order,
                  palette=colors, ax=ax)
    ax.set_title("Content Rating Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Count")
    ax.set_ylabel("Rating")
    _style_ax(ax)
    _save("05_ratings.png")


# ── 6. Movies vs Shows — Genre Heatmap ───────────────────────────────────────
def plot_genre_type_heatmap(df: pd.DataFrame, top_n: int = 10) -> None:
    genres_exp = df[["type", "listed_in"]].copy()
    genres_exp = genres_exp.assign(
        genre=genres_exp["listed_in"].str.split(", ")
    ).explode("genre")

    top_genres = genres_exp["genre"].value_counts().head(top_n).index
    filtered   = genres_exp[genres_exp["genre"].isin(top_genres)]
    pivot      = filtered.groupby(["genre", "type"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK)
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Reds",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Genre × Content Type Heatmap", fontsize=14, pad=12)
    ax.tick_params(colors=WHITE, labelsize=10)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    _save("06_genre_type_heatmap.png")


# ── 7. Top Countries ─────────────────────────────────────────────────────────
def plot_top_countries(df: pd.DataFrame, top_n: int = 10) -> None:
    countries = df[df["country"] != "Unknown"]["country"]
    countries = countries.str.split(", ").explode()
    top = countries.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
    colors = [RED if i == 0 else "#555" for i in range(top_n)]
    top.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
    ax.set_title(f"Top {top_n} Countries by Content Volume", fontsize=14, pad=12)
    ax.set_xlabel("Country")
    ax.set_ylabel("Titles")
    plt.xticks(rotation=30, ha="right")
    _style_ax(ax)
    _save("07_top_countries.png")


# ── Run all ───────────────────────────────────────────────────────────────────
def run_all_eda(df: pd.DataFrame) -> None:
    print("\n🔍  Running EDA …")
    plot_type_distribution(df)
    plot_top_genres(df)
    plot_content_over_time(df)
    plot_release_year(df)
    plot_ratings(df)
    plot_genre_type_heatmap(df)
    plot_top_countries(df)
    print("✅  All EDA charts saved to outputs/\n")
