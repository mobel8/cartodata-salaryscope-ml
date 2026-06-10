"""Feature contextuelle de coût de la vie (€/m²) par département IDF.

Greffe territoriale BORNÉE et assumée : un repère de prix immobilier médian au
m² par département (ordres de grandeur publics, données DVF/marché ~2024),
joignable comme feature contextuelle au régresseur et affichable sur une carte.
Ce n'est PAS le cœur du projet (le volume salaire suffit) — c'est un
enrichissement géo documenté. L'intégration DVF complète (9M lignes) reste un
backlog.
"""

from __future__ import annotations

# €/m² médian approximatif par département IDF (ordre de grandeur public).
DEPT_PRICE_EUR_M2 = {
    "75": 9800,   # Paris
    "92": 6500,   # Hauts-de-Seine
    "93": 3700,   # Seine-Saint-Denis
    "94": 4700,   # Val-de-Marne
    "78": 4200,   # Yvelines
    "91": 3300,   # Essonne
    "95": 3200,   # Val-d'Oise
    "77": 3000,   # Seine-et-Marne
}
DEFAULT_PRICE = 4500  # IDF / non précisé


def price_m2(departement: str | None) -> int:
    return DEPT_PRICE_EUR_M2.get(str(departement), DEFAULT_PRICE)
