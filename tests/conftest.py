"""Fixtures de tests : offres synthétiques (aucune dépendance au GOLD)."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth_offers() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 180
    contracts = rng.choice(["CDI", "Stage", "Alternance", "Non précisé"], n, p=[0.5, 0.2, 0.15, 0.15])
    exp = rng.choice(
        ["Junior (< 3 ans)", "Confirmé (3-5 ans)", "Senior (5+ ans)", "Non précisé"], n
    )
    titles = rng.choice(
        ["data scientist", "data engineer senior", "data analyst junior",
         "machine learning engineer", "consultant bi", "stagiaire data"], n
    )
    base = {"CDI": 50000, "Stage": 15000, "Alternance": 20000, "Non précisé": 45000}
    bonus = {"Junior (< 3 ans)": -5000, "Confirmé (3-5 ans)": 5000,
             "Senior (5+ ans)": 18000, "Non précisé": 0}
    sal = np.clip(
        [base[c] + bonus[e] + rng.normal(0, 4000) for c, e in zip(contracts, exp, strict=True)],
        9000, 200000,
    )
    return pd.DataFrame({
        "source": rng.choice(["WTTJ", "Indeed"], n),
        "title": titles,
        "company": rng.choice([f"corp{i}" for i in range(12)], n),
        "city": "Paris",
        "departement": rng.choice(["75", "92", "93", "94", "Non précisé"], n),
        "is_idf": True,
        "profession": rng.choice(["Technologie et ingénierie", "Gestion de projets"], n),
        "contract": contracts,
        "exp_bucket": exp,
        "education": "Non précisé",
        "remote": rng.choice(["Non", "Partiel", "Non précisé"], n),
        "skills": [list(rng.choice(["Python", "SQL", "Machine Learning"], 2, replace=False))
                   for _ in range(n)],
        "sal_mid_eur": np.round(sal),
        "has_salary": True,
        "url": [f"https://x/{i}" for i in range(n)],
    })
