"""Tracking d'expériences léger : JSON local toujours écrit, MLflow si dispo.

On ne dépend pas de MLflow pour tourner (MLOps junior pragmatique) : les
métriques sont versionnées en JSON dans metrics/, et loggées en plus dans
MLflow si le paquet est installé (mode best-effort).
"""

from __future__ import annotations

import json

from salaryscope import config


def log_run(name: str, params: dict, metrics: dict) -> None:
    config.ensure_dirs()
    payload = {"run": name, "params": params, "metrics": metrics}
    (config.METRICS / f"run_{name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:  # best-effort : ne casse jamais le pipeline si MLflow absent
        import mlflow

        mlflow.set_tracking_uri((config.ROOT / "mlruns").as_uri())
        mlflow.set_experiment("salaryscope")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    except Exception:  # noqa: BLE001
        pass
