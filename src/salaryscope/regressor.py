"""Régresseur de salaire : LightGBM vs baselines, validation anti-fuite, intervalles.

Décisions clés (défendables) :
  - Cible = salaire annualisé médian EN EUROS (pas de log) -> les valeurs SHAP
    s'expriment directement en € ("Senior = +8 k").
  - Objectif LightGBM = L1 (MAE) : robuste aux salaires sales/asymétriques.
  - Validation = GroupKFold par ENTREPRISE : deux offres d'un même employeur ne
    peuvent pas être à la fois en train et en test (anti-fuite via l'ATS).
  - Intervalles de prédiction = régression quantile LightGBM (q10/q90 -> 80 %).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from salaryscope import config, features

N_SPLITS = 5


class MedianByGroup(BaseEstimator, RegressorMixin):
    """Baseline métier : médiane du salaire par modalité d'une colonne (ex. exp_bucket)."""

    def __init__(self, col: str = "exp_bucket"):
        self.col = col

    def fit(self, X, y):
        y = np.asarray(y)
        self.global_ = float(np.median(y))
        self.map_ = pd.Series(y, index=X[self.col].to_numpy()).groupby(level=0).median().to_dict()
        return self

    def predict(self, X):
        return X[self.col].map(self.map_).fillna(self.global_).to_numpy()


def build_preprocessor() -> ColumnTransformer:
    cols = features.feature_columns()
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cols["categorical"]),
            ("bool", "passthrough", cols["boolean"]),
            ("num", "passthrough", cols["numeric"]),
            ("txt", TfidfVectorizer(max_features=150, min_df=5, ngram_range=(1, 2)), cols["text"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _lgbm(objective: str = "regression_l1", alpha: float = 0.5) -> LGBMRegressor:
    params = dict(
        n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    if objective == "quantile":
        return LGBMRegressor(objective="quantile", alpha=alpha, **params)
    return LGBMRegressor(objective=objective, **params)


def build_regressor() -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", _lgbm())])


def build_ridge() -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", Ridge(alpha=1.0))])


def _scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y, pred)), 1),
        "mape": round(float(np.mean(np.abs((y - pred) / y)) * 100), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 1),
    }


def evaluate(df: pd.DataFrame) -> dict:
    """Validation GroupKFold : LightGBM vs baselines, global + par segment de contrat."""
    X = features.build_features(df)
    y = df[config.TARGET].to_numpy(dtype=float)
    grp = features.groups(df)
    cv = GroupKFold(n_splits=N_SPLITS)

    candidates = {
        "median_global": DummyRegressor(strategy="median"),
        "median_exp_bucket": MedianByGroup("exp_bucket"),
        "ridge": build_ridge(),
        "lightgbm": build_regressor(),
    }
    out = {"n": int(len(df)), "n_groups": int(grp.nunique()), "models": {}}
    lgbm_oof = None
    for name, est in candidates.items():
        pred = cross_val_predict(est, X, y, cv=cv, groups=grp)
        res = _scores(y, pred)
        if name == "lightgbm":
            lgbm_oof = pred
            res["by_contract"] = {}
            for contract, idx in df.groupby("contract").groups.items():
                ii = df.index.get_indexer(idx)
                if len(ii) >= 20:
                    res["by_contract"][str(contract)] = {"n": int(len(ii)), **_scores(y[ii], pred[ii])}
        out["models"][name] = res

    out["target_median"] = round(float(np.median(y)))
    out["target_q1"] = round(float(np.quantile(y, 0.25)))
    out["target_q3"] = round(float(np.quantile(y, 0.75)))
    if lgbm_oof is not None:
        out["uplift_vs_median_pct"] = round(
            100 * (out["models"]["median_global"]["mae"] - out["models"]["lightgbm"]["mae"])
            / out["models"]["median_global"]["mae"], 1
        )
    return out


def fit_final(df: pd.DataFrame) -> dict:
    """Entraîne le modèle servi : point (L1) + quantiles q10/q90 (intervalle 80 %)."""
    X = features.build_features(df)
    y = df[config.TARGET].to_numpy(dtype=float)

    point = build_regressor().fit(X, y)
    q_low = Pipeline([("prep", build_preprocessor()), ("model", _lgbm("quantile", 0.1))]).fit(X, y)
    q_high = Pipeline([("prep", build_preprocessor()), ("model", _lgbm("quantile", 0.9))]).fit(X, y)

    return {
        "point": point,
        "q_low": q_low,
        "q_high": q_high,
        "salary_distribution": np.sort(y).tolist(),
        "feature_columns": features.feature_columns(),
        "trained_on": int(len(df)),
    }


def predict(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    point = bundle["point"].predict(X)
    low = bundle["q_low"].predict(X)
    high = bundle["q_high"].predict(X)
    # On garantit low <= point <= high.
    low = np.minimum(low, point)
    high = np.maximum(high, point)
    return pd.DataFrame({"point": np.round(point), "low": np.round(low), "high": np.round(high)})
