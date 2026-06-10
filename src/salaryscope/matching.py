"""Matching « Where do I stand » : offres proches + percentile marché + tension.

Transforme le modèle en PRODUIT : un candidat saisit un intitulé, on lui rend
les k offres les plus proches (cosinus TF-IDF), son percentile salarial sur le
marché, et un indice de tension par segment.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_KEEP = ["title", "company", "city", "departement", "contract", "profession",
         "sal_min_eur", "sal_max_eur", "has_salary", "url"]


def build_index(df: pd.DataFrame) -> dict:
    frame = df[[c for c in _KEEP if c in df.columns]].reset_index(drop=True)
    vec = TfidfVectorizer(max_features=4000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform(frame["title"].fillna(""))
    return {"vectorizer": vec, "matrix": matrix, "frame": frame}


def nearest(index: dict, title: str, k: int = 5) -> list[dict]:
    q = index["vectorizer"].transform([title or ""])
    sims = cosine_similarity(q, index["matrix"])[0]
    top = np.argsort(sims)[::-1][:k]
    rows = []
    for i in top:
        r = index["frame"].iloc[int(i)].to_dict()
        r["similarite"] = round(float(sims[i]), 3)
        for kk in ("sal_min_eur", "sal_max_eur"):
            if pd.notna(r.get(kk)):
                r[kk] = int(r[kk])
            else:
                r[kk] = None
        rows.append(r)
    return rows


def market_percentile(salary_distribution: list[float], value: float) -> float:
    """Position d'un salaire dans la distribution marché (0-100)."""
    dist = sorted(salary_distribution)
    if not dist:
        return 0.0
    pos = bisect.bisect_right(dist, value)
    return round(100.0 * pos / len(dist), 1)


def tension(df: pd.DataFrame, profession: str | None = None, departement: str | None = None) -> dict:
    """Indice de tension simple = volume d'offres ouvertes sur un segment."""
    sub = df
    if profession:
        sub = sub[sub["profession"] == profession]
    if departement:
        sub = sub[sub["departement"] == departement]
    return {
        "n_offres": int(len(sub)),
        "part_marche_pct": round(100.0 * len(sub) / max(len(df), 1), 2),
    }
