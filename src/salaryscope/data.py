"""Chargement des offres depuis le GOLD du module 01 (colonne vertébrale)."""

from __future__ import annotations

import pandas as pd

from salaryscope import config


def load_offres() -> pd.DataFrame:
    """Charge fct_offres (GOLD). Lève une erreur claire si le GOLD est absent."""
    path = config.resolve_gold()
    if not path.exists():
        raise FileNotFoundError(
            f"GOLD introuvable : {path}.\n"
            "Lancez d'abord le module 01 (py -m cartodata_de.pipeline) ou "
            "définissez $GOLD_PARQUET."
        )
    df = pd.read_parquet(path)
    # skills peut être un ndarray (Parquet list) -> liste Python.
    if "skills" in df.columns:
        df["skills"] = df["skills"].apply(lambda s: list(s) if s is not None else [])
    return df


def load_salaried(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sous-ensemble exploitable pour la régression : salaire annualisé plausible."""
    if df is None:
        df = load_offres()
    m = (
        df["has_salary"]
        & df[config.TARGET].notna()
        & df[config.TARGET].between(config.SALARY_MIN, config.SALARY_MAX)
    )
    return df.loc[m].reset_index(drop=True)
