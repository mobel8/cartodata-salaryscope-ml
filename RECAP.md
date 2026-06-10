# 🗂️ RÉCAP — SalaryScope IDF (module Data Science / ML)

> Fiche de récapitulation **exhaustive** : quoi, pourquoi, comment, chiffres, décisions, état, lancement. (À jour au 2026-06-10.)

## 1. En une phrase
**Module #03** de CartoData IDF : un **moteur d'estimation salariale** Data/IA francilien (régression LightGBM + **explicabilité SHAP** + classifieur de famille NLP + **matching marché**), servi en **API FastAPI** et **démo Streamlit**, qui **consomme le GOLD du module 01**.

## 2. Objectif & usage
Projet portfolio famille **Data Science / Machine Learning**, pour compenser le manque d'expérience par un produit ML de bout en bout sur données réelles sales. Pitch méta : « j'ai modélisé le salaire de marché de VOTRE poste sur ~1 800 offres concurrentes, avant même de postuler ».

## 3. État : ✅ CONSTRUIT, ENTRAÎNÉ & VÉRIFIÉ
- Entraînement complet exécuté sur le GOLD réel (1 845 salaires).
- **pytest 13/13 · ruff clean** ; modèles + figures + métriques générés et commités.
- API vérifiée (TestClient) ; démo Streamlit (syntaxe validée).
- Dépôt **public-safe** : aucune information personnelle.

## 4. Stack
Python 3.11 · scikit-learn · **LightGBM** · **SHAP** · TF-IDF · pandas/numpy · **FastAPI** · **Streamlit** · MLflow/JSON (tracking) · joblib · matplotlib.

## 5. Arborescence
```
03-salaryscope-ml/
├── src/salaryscope/  data.py · features.py · regressor.py · explain.py · classifier.py
│                     matching.py · service.py · api.py · cost_of_living.py · tracking.py · config.py
├── app.py            démo Streamlit
├── scripts/train.py  entraînement complet
├── models/           regressor.joblib (3,4 Mo) · classifier.joblib · match_index.joblib (servis, commités)
├── metrics/          regressor.json · classifier.json · run_*.json (chiffres réels)
├── docs/figures/     shap_global.png · mae_by_segment.png · confusion_famille.png
├── tests/            conftest + 5 fichiers (features/régresseur/matching/classifieur/API)
├── .github/workflows/ci.yml · pyproject.toml · requirements.txt · model-card.md · README.md
```

## 6. Pipeline ML
GOLD (module 01) → `load_salaried` (salaires 8 k–300 k) → `build_features` (one-hot contrat/exp/edu/dépt/TT/source + flags compétences + séniorité titre + TF-IDF) → **LightGBM L1** + quantiles q10/q90 → **SHAP** drivers € → classifieur TF-IDF/LogReg (profession.fr) → index matching cosinus. Tracking MLflow/JSON. Sérialisation joblib. Service unique (API + démo).

## 7. Chiffres clés (réels)
- **1 845** salaires · **859** entreprises (groupes) · médiane 49 k€ (Q1 29,7 k / Q3 65 k).
- **LightGBM MAE 11 567 € · MAPE 25,5 % · RMSE 19 285 €** ; baseline médiane 24 121 € → **−52 %** ; Ridge 13 055 €.
- Par segment : **CDI MAE 9 160 € (15,0 %)** · Stage 5 490 € · sans contrat 12 363 €.
- Classifieur famille : **F1 macro 0,647 · micro 0,809** (10 classes).

## 8. Décisions d'ingénierie (défendables)
Cible en € (SHAP en €) · objectif **L1** robuste · **GroupKFold par entreprise** (anti-fuite ATS) · baselines triviales explicites · intervalles quantiles · features **interprétables** pour des drivers SHAP lisibles · TF-IDF plutôt que CamemBERT (volume + CPU, honnêteté) · fusion DVF en feature contextuelle au département (backlog DVF complet) · modèles servis commités pour démo reproductible.

## 9. Lancer
```powershell
py -m pip install -r requirements.txt ; py -m pip install -e .
py scripts/train.py ; uvicorn salaryscope.api:app --reload ; streamlit run app.py
py -m ruff check . ; py -m pytest
```

## 10. Postes visés
Data Scientist (stage/alternance/junior) · Ingénieur Machine Learning · Ingénieur IA & Data Science appliquée · Data Analyst & Data Scientist · Quantitative Analyst junior · Data Scientist Intern.

## 11. Limites & next steps
Biais de sélection assumé · couverture salaire 24 % · labels 81 % Indeed (métadonnées pauvres) · géo au département (BAN en backlog) · agent LLM (bonus) renvoyé au module 05 · DVF complet en backlog. Détails : `model-card.md`.

## 12. Confidentialité
Dépôt **public-safe** : aucune information personnelle. Le coût de la vie est traité **au département** (pas de ville de domicile). Données = offres publiques.

## 13. Place dans la plateforme
Module **#03** du monorepo **CartoData IDF** : consomme le GOLD du **#01 (Data Eng)**, complète le **#02 (BI)**. Le régresseur et l'index pourront servir d'outils à l'agent du **#05 (CartoIA)**.
