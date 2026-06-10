"""Service d'inférence : charge les modèles servis et répond aux requêtes.

Point d'entrée unique réutilisé par l'API FastAPI et la démo Streamlit.
Les modèles (regressor/classifier/index) sont chargés une fois, en cache.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from salaryscope import classifier as clf_mod
from salaryscope import config, explain, features, matching, regressor


@lru_cache(maxsize=1)
def _models() -> dict:
    return {
        "reg": joblib.load(config.MODELS / "regressor.joblib"),
        "clf": joblib.load(config.MODELS / "classifier.joblib"),
        "idx": joblib.load(config.MODELS / "match_index.joblib"),
    }


def _row(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "source": payload.get("source", "WTTJ"),
        "title": payload.get("title", ""),
        "company": payload.get("company") or "__unknown__",
        "contract": payload.get("contract", "Non précisé"),
        "exp_bucket": payload.get("exp_bucket", "Non précisé"),
        "education": payload.get("education", "Non précisé"),
        "departement": str(payload.get("departement", "Non précisé")),
        "remote": payload.get("remote", "Non précisé"),
        "is_idf": bool(payload.get("is_idf", True)),
        "skills": payload.get("skills", []) or [],
    }])


def estimate(payload: dict) -> dict:
    c = _models()
    x = features.build_features(_row(payload))
    pred = regressor.predict(c["reg"], x).iloc[0]
    drivers = explain.drivers(c["reg"], x, top_k=6)[0]
    pct = matching.market_percentile(c["reg"]["salary_distribution"], float(pred["point"]))
    return {
        "salaire_estime_eur": int(pred["point"]),
        "intervalle_80_eur": [int(pred["low"]), int(pred["high"])],
        "percentile_marche": pct,
        "drivers": drivers,
    }


def classify(title: str) -> list[dict]:
    return clf_mod.predict_famille(_models()["clf"]["model"], title)


def match(title: str, k: int = 5) -> list[dict]:
    return matching.nearest(_models()["idx"], title, k)


def where_do_i_stand(payload: dict) -> dict:
    """Réponse produit complète : estimation + famille + offres proches."""
    est = estimate(payload)
    return {
        **est,
        "famille_predite": classify(payload.get("title", "")),
        "offres_proches": match(payload.get("title", ""), k=5),
    }
