"""Démo interactive SalaryScope (Streamlit).

Lancer : streamlit run app.py
"colle un intitulé -> salaire estimé + intervalle + percentile marché + drivers SHAP + offres proches"
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from salaryscope import config, service

st.set_page_config(page_title="SalaryScope IDF", page_icon="📊", layout="wide")
st.title("📊 SalaryScope IDF — où se situe ce poste sur le marché Data/IA francilien ?")
st.caption(
    "Modèle entraîné sur ~1 800 salaires d'offres Data/IA réelles d'Île-de-France "
    "(GOLD du module Data Engineering). Estimation + explication + offres proches."
)

with st.sidebar:
    st.header("Décrivez le poste")
    title = st.text_input("Intitulé", "Data Scientist Senior NLP")
    contract = st.selectbox("Contrat", ["Non précisé", "CDI", "CDD", "Stage", "Alternance", "Freelance", "VIE"])
    exp = st.selectbox("Expérience", ["Non précisé", "Junior (< 3 ans)", "Confirmé (3-5 ans)", "Senior (5+ ans)", "Autre"])
    edu = st.selectbox("Diplôme", ["Non précisé", "bac", "bac_2", "bac_3", "bac_5"])
    dept = st.selectbox("Département", ["Non précisé", "75", "92", "93", "94", "78", "91", "95", "77", "IDF"])
    remote = st.selectbox("Télétravail", ["Non précisé", "Non", "Ponctuel", "Partiel", "Total"])
    skills = st.multiselect("Compétences", config.SKILL_FLAGS, default=["Python", "Machine Learning"])
    go = st.button("Estimer", type="primary")

if go:
    payload = {
        "title": title, "contract": contract, "exp_bucket": exp, "education": edu,
        "departement": dept, "remote": remote, "skills": skills, "source": "WTTJ",
    }
    try:
        res = service.where_do_i_stand(payload)
    except FileNotFoundError:
        st.error("Modèles absents. Lancez d'abord `py scripts/train.py`.")
        st.stop()

    lo, hi = res["intervalle_80_eur"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Salaire estimé", f"{res['salaire_estime_eur']:,} €".replace(",", " "))
    c2.metric("Intervalle 80 %", f"{lo:,}–{hi:,} €".replace(",", " "))
    c3.metric("Percentile marché", f"{res['percentile_marche']} %")

    st.subheader("Pourquoi ? (drivers SHAP, en €)")
    drv = pd.DataFrame(res["drivers"])
    st.bar_chart(drv.set_index("feature")["impact_eur"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Famille de métier prédite")
        st.dataframe(pd.DataFrame(res["famille_predite"]), hide_index=True)
    with col_b:
        st.subheader("Offres les plus proches")
        near = pd.DataFrame(res["offres_proches"])
        cols = [c for c in ["title", "company", "city", "contract", "similarite"] if c in near]
        st.dataframe(near[cols], hide_index=True)

    st.caption(
        "⚠️ Couverture salaire ~24 % du marché ; biais de sélection (les offres qui "
        "affichent un salaire ne sont pas représentatives). Voir model-card.md."
    )
else:
    st.info("Renseignez le poste dans la barre latérale puis cliquez **Estimer**.")
