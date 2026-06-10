"""Feature engineering : tabulaire interprétable + séniorité/stack depuis le titre.

On garde des features LISIBLES (one-hot + flags) pour que SHAP produise des
drivers parlants ("Senior + Paris + CDI = +Xk"), plus un TF-IDF du titre géré
côté pipeline (regressor.py).
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from salaryscope import config

SENIOR_RE = re.compile(
    r"\b(?:senior|sr|lead|principal|staff|head|architect|expert|confirm|director|directeur)\b"
)
JUNIOR_RE = re.compile(
    r"\b(?:junior|jr|stage|stagiaire|intern|alternance|alternant|apprenti|graduate|debutant|entry)\b"
)


def _strip(text: str) -> str:
    norm = unicodedata.normalize("NFD", str(text).lower())
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def skill_col(skill: str) -> str:
    return "skill_" + skill.replace(" ", "_").replace("-", "_")


def feature_columns() -> dict[str, list[str]]:
    """Colonnes par type, pour le ColumnTransformer."""
    skill_cols = [skill_col(s) for s in config.SKILL_FLAGS]
    return {
        "categorical": config.CAT_FEATURES,
        "boolean": config.BOOL_FEATURES + skill_cols,
        "numeric": config.NUM_FEATURES,
        "text": "title",
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice de features (sans la cible) à partir des offres GOLD."""
    out = pd.DataFrame(index=df.index)

    for col in config.CAT_FEATURES:
        out[col] = df[col].fillna("Non précisé").astype(str) if col in df else "Non précisé"

    out["is_idf"] = df.get("is_idf", False).fillna(False).astype(bool) if "is_idf" in df else False

    title = df.get("title", pd.Series("", index=df.index)).fillna("").map(_strip)
    out["is_senior"] = title.str.contains(SENIOR_RE)
    out["is_junior_titre"] = title.str.contains(JUNIOR_RE)

    skills = df.get("skills", pd.Series([[]] * len(df), index=df.index)).apply(
        lambda s: list(s) if s is not None else []
    )
    out["n_skills"] = skills.apply(len)
    for s in config.SKILL_FLAGS:
        out[skill_col(s)] = skills.apply(lambda lst, sk=s: sk in lst)

    out["title"] = df.get("title", pd.Series("", index=df.index)).fillna("")
    return out


def groups(df: pd.DataFrame) -> pd.Series:
    """Clé de groupe pour la CV anti-fuite : l'entreprise (offres d'un même employeur)."""
    return df["company"].fillna("__unknown__").astype(str)
