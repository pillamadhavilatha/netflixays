# 🎬 Netflix Content Recommender System
### Data Analyst Internship Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?logo=pandas)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

A content-based recommendation engine built on the Netflix Titles dataset (~8,800 titles).  
The system recommends similar movies or TV shows using **TF-IDF vectorisation** and **cosine similarity** across genres, descriptions, directors, and cast.

Key deliverables:
- Exploratory Data Analysis with 10+ visualisations
- Feature engineering pipeline
- Content-based recommendation engine (top-10 results)
- Clean, modular Python code ready for extension

---

## 📁 Project Structure

```
netflix_recommender/
├── data/
│   └── netflix_titles.csv          # Raw dataset (Kaggle)
├── notebooks/
│   └── netflix_recommender.ipynb   # Full analysis notebook
├── src/
│   ├── data_loader.py              # Data ingestion & cleaning
│   ├── eda.py                      # Exploratory Data Analysis
│   ├── features.py                 # Feature engineering
│   └── recommender.py              # Recommendation engine
├── outputs/
│   └── *.png                       # Generated charts
├── tests/
│   └── test_recommender.py         # Unit tests
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

**Source:** [Netflix Movies and TV Shows — Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)

| Column | Description |
|---|---|
| `show_id` | Unique ID |
| `type` | Movie or TV Show |
| `title` | Title of the content |
| `director` | Director name(s) |
| `cast` | Main cast |
| `country` | Country of origin |
| `date_added` | Date added to Netflix |
| `release_year` | Original release year |
| `rating` | Content rating (TV-MA, PG-13 …) |
| `duration` | Runtime (min) or seasons |
| `listed_in` | Genre tags |
| `description` | Synopsis |

---

## 🔍 Exploratory Data Analysis

| Analysis | Insight |
|---|---|
| Content split | ~70% Movies, ~30% TV Shows |
| Top genre | Dramas dominate (~1,900 titles) |
| Content growth | Massive spike in additions 2018–2020 |
| Rating | TV-MA is the most common rating |
| Release year | Most content released after 2015 |

---

## 🧠 Recommendation Engine

**Algorithm:** Content-Based Filtering using TF-IDF + Cosine Similarity

```
Input title
    └─► Combine: genre + description + director + cast
            └─► TF-IDF matrix  (n_titles × n_terms)
                    └─► Cosine similarity matrix  (n × n)
                            └─► Rank & return Top-10 similar titles
```

### Example

```python
from src.recommender import NetflixRecommender

rec = NetflixRecommender("data/netflix_titles.csv")
rec.fit()
print(rec.recommend("Stranger Things"))
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/netflix-recommender.git
cd netflix-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python src/recommender.py

# 4. Or open the notebook
jupyter notebook notebooks/netflix_recommender.ipynb
```

---

## 📊 Key Visualisations

- Movies vs TV Shows (countplot)
- Top 15 Genre Distribution (bar chart)
- Content Added Per Year (line chart)
- Release Year Distribution (histogram)
- Rating Distribution (horizontal bar)

All charts saved to `outputs/`.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | TF-IDF vectorisation, cosine similarity |

---

## 📈 Future Improvements

- [ ] Collaborative filtering using user ratings
- [ ] Hybrid model (content + collaborative)
- [ ] Streamlit web app interface
- [ ] Deep learning embeddings (Sentence-BERT)
- [ ] A/B testing framework for recommendation quality

---

## 👤 Author

**Shanmukha** — Data Analyst Internship Project  
*Built as part of a data science internship programme to demonstrate end-to-end analytical and ML skills.*

---

## 📄 License

MIT License — free to use, modify, and distribute.
