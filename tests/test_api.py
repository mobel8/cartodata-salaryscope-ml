"""Tests API : nécessitent les modèles servis (models/*.joblib commités)."""

import pytest
from fastapi.testclient import TestClient

from salaryscope import config
from salaryscope.api import app

pytestmark = pytest.mark.skipif(
    not (config.MODELS / "regressor.joblib").exists(),
    reason="modèles non entraînés (py scripts/train.py)",
)

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_salary():
    r = client.post("/predict_salary", json={
        "title": "Data Scientist Senior NLP", "contract": "CDI",
        "exp_bucket": "Senior (5+ ans)", "departement": "75",
        "skills": ["Python", "Machine Learning"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["salaire_estime_eur"] > 0
    assert len(body["intervalle_80_eur"]) == 2
    assert body["drivers"]


def test_where_do_i_stand():
    r = client.post("/where_do_i_stand", json={"title": "Data Engineer", "contract": "CDI"})
    assert r.status_code == 200
    assert "famille_predite" in r.json()
    assert "offres_proches" in r.json()
