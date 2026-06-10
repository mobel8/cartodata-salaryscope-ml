"""Classifieur de famille de métier (NLP multi-classe, TF-IDF).

Ground truth PROPRE = profession.fr (taxonomie WTTJ). Volontairement TF-IDF +
régression logistique (pas de CamemBERT) : honnête sur la taille de données et
CPU-friendly. CamemBERT documenté en option (model-card).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from salaryscope import config

MIN_PER_CLASS = 15


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)),
    ])


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["profession"].notna() & df["title"].notna()].copy()
    vc = sub["profession"].value_counts()
    keep = vc[vc >= MIN_PER_CLASS].index
    return sub[sub["profession"].isin(keep)].reset_index(drop=True)


def train_classifier(df: pd.DataFrame) -> dict:
    sub = labeled(df)
    x, y = sub["title"].fillna(""), sub["profession"]
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE
    )
    pipe = build_pipeline().fit(x_tr, y_tr)
    pred = pipe.predict(x_te)
    labels = sorted(y.unique())
    metrics = {
        "n_total": int(len(sub)),
        "n_train": int(len(x_tr)),
        "n_test": int(len(x_te)),
        "n_classes": len(labels),
        "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 3),
        "f1_micro": round(float(f1_score(y_te, pred, average="micro")), 3),
    }
    cm = confusion_matrix(y_te, pred, labels=labels).tolist()
    final = clone(build_pipeline()).fit(x, y)  # refit sur tout pour le service
    return {"model": final, "metrics": metrics, "labels": labels, "confusion": cm}


def predict_famille(model, title: str, top: int = 3) -> list[dict]:
    proba = model.predict_proba([title])[0]
    classes = model.classes_
    order = proba.argsort()[::-1][:top]
    return [{"famille": str(classes[i]), "proba": round(float(proba[i]), 3)} for i in order]
