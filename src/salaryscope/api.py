"""API FastAPI de scoring SalaryScope.

Lancer : uvicorn salaryscope.api:app --reload   ->  http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from salaryscope import __version__, service

app = FastAPI(
    title="SalaryScope IDF — API",
    version=__version__,
    description="Estimation salariale + explicabilité SHAP + matching du marché Data/IA francilien.",
)


class SalaryRequest(BaseModel):
    title: str = Field(..., examples=["Data Scientist Senior NLP"])
    contract: str = "Non précisé"
    exp_bucket: str = "Non précisé"
    education: str = "Non précisé"
    departement: str = "Non précisé"
    remote: str = "Non précisé"
    skills: list[str] = []
    source: str = "WTTJ"


class TitleRequest(BaseModel):
    title: str = Field(..., examples=["Data Engineer"])
    k: int = 5


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/predict_salary")
def predict_salary(req: SalaryRequest) -> dict:
    return service.estimate(req.model_dump())


@app.post("/classify")
def classify(req: TitleRequest) -> list[dict]:
    return service.classify(req.title)


@app.post("/match")
def match(req: TitleRequest) -> list[dict]:
    return service.match(req.title, req.k)


@app.post("/where_do_i_stand")
def where_do_i_stand(req: SalaryRequest) -> dict:
    return service.where_do_i_stand(req.model_dump())
