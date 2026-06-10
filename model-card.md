# Model Card — SalaryScope IDF (régresseur de salaire)

## Usage prévu
Estimer un **ordre de grandeur** du salaire annuel brut d'une offre Data/IA en
Île-de-France, avec un intervalle et une explication, pour situer un poste sur
le marché (candidat ou recruteur). **Aide à la décision, pas une vérité** :
toute estimation s'accompagne d'un intervalle et d'un percentile marché.

## Données d'entraînement
- Source : couche **GOLD** du module 01 (CartoData IDF) = offres Data/IA IDF
  scrapées (Welcome to the Jungle + Indeed), dédupliquées, salaires annualisés.
- **1 845** offres avec salaire annualisé plausible (8 k–300 k €), sur 7 643.
- Répartition des labels : **~81 % Indeed, ~19 % WTTJ** ; **859 entreprises**.
- Cible : `sal_mid_eur` (milieu de fourchette annualisée). Médiane 49 k€.

## Modèle
- LightGBM, objectif **L1 (MAE)**, 400 arbres, features one-hot (contrat, exp,
  diplôme, département, télétravail, source) + flags compétences + séniorité
  extraite du titre + TF-IDF du titre. Cible en € (SHAP additif en €).
- Intervalle 80 % : régresseurs quantiles LightGBM q10/q90.

## Évaluation (GroupKFold=5 par entreprise — anti-fuite ATS)
| Métrique | Valeur |
|---|--:|
| MAE | 11 567 € |
| MAPE | 25,5 % |
| RMSE | 19 285 € |
| MAE CDI | 9 160 € (MAPE 15,0 %) |
| MAE Stage | 5 490 € |
| Baseline médiane (MAE) | 24 121 € |

Le **GroupKFold par entreprise** est essentiel : sans lui, des offres d'un même
employeur (souvent au même salaire via l'ATS) fuiteraient entre train et test
et gonfleraient artificiellement la performance.

## Biais & limites (à assumer à l'oral)
- **Biais de sélection** : seules ~24 % des offres affichent un salaire ; celles
  qui le font ne sont pas représentatives (secteurs/tailles spécifiques).
- **Dominance Indeed** : 81 % des labels viennent d'offres à métadonnées pauvres
  (souvent contrat « Non précisé », pas d'expérience/diplôme) → le modèle
  s'appuie beaucoup sur l'intitulé et le département.
- **Annualisation à hypothèses** : mensuel ×12, TJM ×218 j ouvrés — un freelance
  peut être sur/sous-estimé.
- **Petit volume** : ~1 800 lignes → PAS de deep learning ; LightGBM + features
  interprétables, c'est le bon dimensionnement.
- **Géo grossière** : département, pas commune (géocodage BAN en backlog).

## Hors-périmètre / extensions documentées
- CamemBERT pour le classifieur de famille (gain marginal attendu sur ce volume,
  coût CPU élevé) — non retenu, documenté.
- Fusion DVF complète (prix €/m² par commune) — ici approximée au **département**
  (`cost_of_living.py`), intégration fine en backlog.

## Éthique
Ne pas utiliser pour fixer une rémunération individuelle. Données = offres
publiques, aucune donnée personnelle. Repo public-safe.
