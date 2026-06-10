"""Explicabilité SHAP : transformer une prédiction € en drivers lisibles.

SHAP TreeExplainer sur le LightGBM (cible en €) -> contributions additives en
euros : "Senior +8 k, Paris +5 k, Stage -22 k". C'est ce qui transforme une
prédiction en conversation avec le recruteur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _pretty(name: str) -> str:
    """Nettoie les noms de features du ColumnTransformer pour l'affichage."""
    n = name.split("__", 1)[-1]
    n = n.replace("skill_", "").replace("_", " ")
    if n.startswith("contract"):
        return n.replace("contract ", "contrat=")
    if n.startswith("departement"):
        return n.replace("departement ", "dépt=")
    if n.startswith("exp bucket"):
        return n.replace("exp bucket ", "exp=")
    if n.startswith("remote"):
        return n.replace("remote ", "TT=")
    if n.startswith("education"):
        return n.replace("education ", "diplôme=")
    if n.startswith("source"):
        return n.replace("source ", "source=")
    if n in ("is senior", "is junior titre"):
        return {"is senior": "titre=senior", "is junior titre": "titre=junior"}[n]
    return n


def _parts(bundle: dict):
    pipe = bundle["point"]
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    return prep, model


def drivers(bundle: dict, X: pd.DataFrame, top_k: int = 6) -> list[list[dict]]:
    """Pour chaque ligne de X, les top_k contributions SHAP (en €)."""
    prep, model = _parts(bundle)
    names = [_pretty(n) for n in prep.get_feature_names_out()]
    xt = prep.transform(X)
    xt = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
    sv = shap.TreeExplainer(model).shap_values(xt)
    out = []
    for row in np.atleast_2d(sv):
        order = np.argsort(np.abs(row))[::-1][:top_k]
        out.append([{"feature": names[i], "impact_eur": round(float(row[i]))} for i in order])
    return out


def global_importance(bundle: dict, X: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Importance globale = moyenne des |SHAP| (pour la figure récap)."""
    prep, model = _parts(bundle)
    names = [_pretty(n) for n in prep.get_feature_names_out()]
    xt = prep.transform(X)
    xt = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
    sv = shap.TreeExplainer(model).shap_values(xt)
    imp = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
