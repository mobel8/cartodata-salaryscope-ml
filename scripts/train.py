"""Entraînement complet SalaryScope : évaluation -> modèles servis -> figures -> métriques.

Usage : py scripts/train.py
Produit : models/*.joblib, metrics/*.json, docs/figures/*.png
"""

from __future__ import annotations

import json
import sys
import warnings

import joblib

warnings.filterwarnings("ignore", message="X does not have valid feature names")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from salaryscope import (  # noqa: E402
    classifier,
    config,
    data,
    explain,
    features,
    matching,
    regressor,
    tracking,
)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass


def _fig_shap(bundle, sal):
    sample = sal.sample(min(500, len(sal)), random_state=config.RANDOM_STATE)
    gi = explain.global_importance(bundle, features.build_features(sample))
    plt.figure(figsize=(8, 5))
    plt.barh(gi["feature"][::-1], gi["mean_abs_shap"][::-1], color="#34507a")
    plt.xlabel("Impact moyen |SHAP| (€)")
    plt.title("SalaryScope — drivers du salaire (importance globale SHAP)")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "shap_global.png", dpi=120)
    plt.close()


def _fig_segment(reg_metrics):
    seg = reg_metrics["models"]["lightgbm"].get("by_contract", {})
    if not seg:
        return
    names = list(seg)
    maes = [seg[n]["mae"] for n in names]
    plt.figure(figsize=(7, 4))
    plt.bar(names, maes, color="#34507a")
    plt.ylabel("MAE (€)")
    plt.title("MAE LightGBM par segment de contrat (GroupKFold)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "mae_by_segment.png", dpi=120)
    plt.close()


def _fig_confusion(clf):
    import numpy as np

    cm = np.array(clf["confusion"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = [lbl[:18] for lbl in clf["labels"]]
    plt.figure(figsize=(8, 7))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="taux (normalisé par ligne)")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.title(f"Classifieur famille — matrice de confusion (F1 macro {clf['metrics']['f1_macro']})")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "confusion_famille.png", dpi=120)
    plt.close()


def main() -> int:
    config.ensure_dirs()
    df = data.load_offres()
    sal = data.load_salaried(df)
    print(f"Offres GOLD: {len(df)} | salaires exploitables: {len(sal)}")

    print("== Évaluation régresseur (GroupKFold par entreprise) ==")
    reg_metrics = regressor.evaluate(sal)
    m = reg_metrics["models"]
    for name in ("median_global", "median_exp_bucket", "ridge", "lightgbm"):
        print(f"  {name:18s} MAE {m[name]['mae']:>8.0f} €  MAPE {m[name]['mape']:>5.1f}%")
    print(f"  -> uplift LightGBM vs médiane : {reg_metrics.get('uplift_vs_median_pct')}%")
    (config.METRICS / "regressor.json").write_text(
        json.dumps(reg_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tracking.log_run(
        "regressor",
        {"model": "lightgbm", "objective": "l1", "cv": "GroupKFold5/company"},
        {
            "mae_lightgbm": m["lightgbm"]["mae"], "mape_lightgbm": m["lightgbm"]["mape"],
            "mae_median": m["median_global"]["mae"], "mae_ridge": m["ridge"]["mae"],
            "uplift_pct": reg_metrics.get("uplift_vs_median_pct", 0),
        },
    )

    print("== Modèle servi (point + intervalle q10/q90) ==")
    bundle = regressor.fit_final(sal)
    joblib.dump(bundle, config.MODELS / "regressor.joblib")
    _fig_shap(bundle, sal)
    _fig_segment(reg_metrics)

    print("== Classifieur famille (TF-IDF + LogReg) ==")
    clf = classifier.train_classifier(df)
    print(f"  F1 macro {clf['metrics']['f1_macro']} | micro {clf['metrics']['f1_micro']} "
          f"| {clf['metrics']['n_classes']} classes")
    joblib.dump({"model": clf["model"], "labels": clf["labels"], "metrics": clf["metrics"]},
                config.MODELS / "classifier.joblib")
    (config.METRICS / "classifier.json").write_text(
        json.dumps(clf["metrics"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _fig_confusion(clf)

    print("== Index de matching (cosinus TF-IDF) ==")
    joblib.dump(matching.build_index(df), config.MODELS / "match_index.joblib")

    print("OK — modèles, métriques et figures écrits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
