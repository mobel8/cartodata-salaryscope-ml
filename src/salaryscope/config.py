"""Configuration : chemins, résolution du GOLD (module 01), constantes ML."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # …/03-salaryscope-ml
MONOREPO = ROOT.parent                               # …/cartodata-idf

MODELS = ROOT / "models"
FIGURES = ROOT / "docs" / "figures"
METRICS = ROOT / "metrics"
DATA = ROOT / "data"

RANDOM_STATE = 42
TARGET = "sal_mid_eur"

# Hygiène : on n'entraîne que sur des salaires annualisés plausibles.
SALARY_MIN, SALARY_MAX = 8_000, 300_000

# Features tabulaires interprétables (lisibles dans SHAP).
CAT_FEATURES = ["source", "contract", "exp_bucket", "education", "departement", "remote"]
BOOL_FEATURES = ["is_idf", "is_senior", "is_junior_titre"]
NUM_FEATURES = ["n_skills"]
# Compétences les plus fréquentes -> flags binaires (drivers SHAP lisibles).
SKILL_FLAGS = [
    "Python", "SQL", "Machine Learning", "Power BI", "Spark", "AWS", "Azure", "GCP",
    "Docker", "ETL", "LLM", "Deep Learning", "Tableau", "Java", "dbt", "Snowflake",
]


def resolve_gold() -> Path:
    """Localise le Parquet GOLD fct_offres produit par le module 01.

    Ordre : $GOLD_PARQUET -> sibling module 01 -> échantillon commité.
    """
    env = os.environ.get("GOLD_PARQUET")
    if env and Path(env).exists():
        return Path(env)
    sibling = MONOREPO / "01-data-engineering" / "data" / "gold" / "fct_offres.parquet"
    if sibling.exists():
        return sibling
    return DATA / "sample_gold.parquet"


def ensure_dirs() -> None:
    for d in (MODELS, FIGURES, METRICS, DATA):
        d.mkdir(parents=True, exist_ok=True)
