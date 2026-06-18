"""Frontend Streamlit : tester l'API de classification Titanic.

Lancement : `uv run streamlit run src/frontend/app.py`
En docker compose, API_URL est injecte via la variable d'environnement.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Titanic — Survie", layout="wide")
st.title("Prédiction de survie — Titanic")

api_url = st.text_input("URL de l'API", value=API_URL)

about_tab, predict_tab, history_tab = st.tabs(["À propos", "Prédiction", "Historique"])

with about_tab:
    # Cherche README.md : d'abord à côté de app.py (Docker), puis à la racine du projet (local)
    here = Path(__file__).parent
    candidates = [here.parent / "README.md", here.parent.parent / "README.md"]
    readme_path = next((p for p in candidates if p.exists()), None)

    if readme_path:
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.warning("README.md introuvable.")

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pclass = st.selectbox("Classe (Pclass)", [1, 2, 3], index=2)
            age = st.number_input("Âge", min_value=0.0, max_value=120.0, value=29.0, step=1.0)
            fare = st.number_input("Tarif (Fare)", min_value=0.0, value=30.0, step=1.0)

        with col2:
            sibsp = st.number_input("Frères/conjoints à bord (SibSp)", min_value=0, value=0)
            parch = st.number_input("Parents/enfants à bord (Parch)", min_value=0, value=0)

        with col3:
            sex = st.selectbox("Sexe (Sex)", ["male", "female"])
            embarked = st.selectbox("Port d'embarquement (Embarked)", ["S", "C", "Q"])
            title = st.selectbox("Titre (Title)", ["Mr", "Mrs", "Miss", "Master", "Rare"])

        submitted = st.form_submit_button("Prédire")

    if submitted:
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        payload = {
            "Pclass": pclass,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "FamilySize": family_size,
            "IsAlone": is_alone,
            "Sex": sex,
            "Embarked": embarked,
            "Title": title,
        }
        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel à l'API impossible : {exc}")
        else:
            prediction = result["prediction"]
            probability = result["probability"]

            if prediction == 1:
                st.success("Survie prédite")
            else:
                st.error("Décès prédit")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Prédiction", "Survivant" if prediction == 1 else "Décédé")
            with col_b:
                st.metric("Probabilité de survie", f"{probability:.1%}")

            st.progress(probability)

with history_tab:
    st.subheader("Historique des prévisions")
    st.info("Aucun journal de prévisions : ajoutez un endpoint /predictions à l'API (bonus).")
    _ = pd
