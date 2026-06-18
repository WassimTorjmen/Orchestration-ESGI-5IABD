"""Frontend Streamlit : tester l'API de classification Titanic.

Lancement : `uv run streamlit run src/frontend/app.py`
En docker compose, API_URL est injecte via la variable d'environnement.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="🚢 Titanic — Survie",
    page_icon="🚢",
    layout="wide",
)

st.markdown("""
<style>
.survivor-banner {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border-left: 6px solid #10b981;
    border-radius: 10px;
    padding: 18px 24px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin: 12px 0;
}
.death-banner {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border-left: 6px solid #ef4444;
    border-radius: 10px;
    padding: 18px 24px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin: 12px 0;
}
.section-header {
    font-size: 15px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚢 Prédiction de survie — Titanic")
st.caption("Modèle ML entraîné sur le dataset Titanic · Données historiques 1912")

api_url = st.text_input("🔗 URL de l'API", value=API_URL)

about_tab, predict_tab, history_tab = st.tabs(["📖 À propos", "🎯 Prédiction", "📊 Historique"])

# ── Onglet À propos ──────────────────────────────────────────────────────────
with about_tab:
    here = Path(__file__).parent
    candidates = [here.parent / "README.md", here.parent.parent / "README.md"]
    readme_path = next((p for p in candidates if p.exists()), None)
    if readme_path:
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.warning("⚠️ README.md introuvable.")

# ── Onglet Prédiction ────────────────────────────────────────────────────────
with predict_tab:
    st.subheader("🎯 Simuler la survie d'un passager")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="section-header">🎫 Voyage</div>', unsafe_allow_html=True)
            pclass = st.selectbox("🏷️ Classe", [1, 2, 3], index=2,
                                  format_func=lambda x: {1: "1ère — Luxe 👑", 2: "2ème — Confort 🎩", 3: "3ème — Économique 🧳"}[x])
            fare = st.number_input("💰 Tarif (£)", min_value=0.0, value=30.0, step=1.0)

        with col2:
            st.markdown('<div class="section-header">👤 Passager</div>', unsafe_allow_html=True)
            age = st.number_input("🎂 Âge", min_value=0.0, max_value=120.0, value=29.0, step=1.0)
            sex = st.selectbox("⚧️ Sexe", ["male", "female"],
                               format_func=lambda x: "♂️ Homme" if x == "male" else "♀️ Femme")
            title = st.selectbox("🎩 Titre", ["Mr", "Mrs", "Miss", "Master", "Rare"])

        with col3:
            st.markdown('<div class="section-header">👨‍👩‍👧 Famille</div>', unsafe_allow_html=True)
            sibsp = st.number_input("💑 Frères / conjoints à bord", min_value=0, value=0)
            parch = st.number_input("👶 Parents / enfants à bord", min_value=0, value=0)
            embarked = st.selectbox("⚓ Port d'embarquement", ["S", "C", "Q"],
                                    format_func=lambda x: {"S": "🇬🇧 Southampton", "C": "🇫🇷 Cherbourg", "Q": "🇮🇪 Queenstown"}[x])

        submitted = st.form_submit_button("🔮 Lancer la prédiction", use_container_width=True, type="primary")

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
            st.error(f"❌ Appel à l'API impossible : {exc}")
        else:
            prediction = result["prediction"]
            probability = result["probability"]

            st.divider()

            if prediction == 1:
                st.markdown(
                    '<div class="survivor-banner">🟢 SURVIVANT · Ce passager aurait survécu 🛟</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="death-banner">🔴 DÉCÉDÉ · Ce passager n\'aurait pas survécu ⚓</div>',
                    unsafe_allow_html=True,
                )

            col_metrics, col_gauge = st.columns([1, 1])

            with col_metrics:
                st.markdown("#### 📊 Métriques")
                st.metric(
                    "🎯 Verdict",
                    "✅ Survivant" if prediction == 1 else "❌ Décédé",
                )
                st.metric(
                    "📈 Probabilité de survie",
                    f"{probability:.1%}",
                    delta=f"{probability - 0.5:+.1%} vs aléatoire",
                    delta_color="normal",
                )
                st.progress(probability, text=f"Survie : {probability:.1%}")

                # Donut chart
                fig_donut = go.Figure(go.Pie(
                    labels=["🟢 Survie", "🔴 Décès"],
                    values=[probability, 1 - probability],
                    hole=0.55,
                    marker_colors=["#10b981", "#ef4444"],
                    textinfo="label+percent",
                    textfont_size=13,
                ))
                fig_donut.update_layout(
                    title={"text": "Répartition probabiliste", "x": 0.5},
                    height=280,
                    margin=dict(t=40, b=0, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_gauge:
                st.markdown("#### 🌡️ Jauge de survie")
                color = "#10b981" if prediction == 1 else "#ef4444"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(probability * 100, 1),
                    number={"suffix": "%", "font": {"size": 36}},
                    delta={"reference": 50, "suffix": "%"},
                    title={"text": "Probabilité de survie", "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": color, "thickness": 0.25},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0, 33], "color": "#fca5a5"},
                            {"range": [33, 66], "color": "#fde68a"},
                            {"range": [66, 100], "color": "#86efac"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.85,
                            "value": 50,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    height=300,
                    margin=dict(t=60, b=20, l=30, r=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with st.expander("🔍 Détails du profil analysé"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("🏷️ Classe", {1: "1ère 👑", 2: "2ème 🎩", 3: "3ème 🧳"}[pclass])
                    st.metric("💰 Tarif", f"£{fare:.0f}")
                with c2:
                    st.metric("🎂 Âge", f"{age:.0f} ans")
                    st.metric("⚧️ Sexe", "♂️ Homme" if sex == "male" else "♀️ Femme")
                with c3:
                    st.metric("👨‍👩‍👧 Taille famille", f"{family_size} pers.")
                    st.metric("🏠 Seul à bord", "✅ Oui" if is_alone else "❌ Non")

# ── Onglet Historique ────────────────────────────────────────────────────────
with history_tab:
    st.subheader("📊 Historique des prévisions")

    try:
        resp = httpx.get(f"{api_url}/predictions", timeout=5.0)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            df_hist = pd.DataFrame(rows)
            st.dataframe(df_hist, use_container_width=True)
            counts = df_hist["prediction"].value_counts().rename({1: "✅ Survivant", 0: "❌ Décédé"})
            st.bar_chart(counts)
        else:
            st.info("📭 Aucune prévision enregistrée pour l'instant.")
    except httpx.HTTPError:
        st.info("💡 L'endpoint `/predictions` n'existe pas encore (bonus TP S14bis).")

    st.divider()
    st.markdown("### 🗂️ Données historiques du Titanic")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("🚢 Passagers à bord", "2 224")
    col_s2.metric("💀 Victimes", "1 502", delta="-67 %", delta_color="inverse")
    col_s3.metric("🟢 Survivants", "722", delta="+32 %")
    col_s4.metric("👩 Femmes sauvées", "~74 %", delta="priorité évacuation")

    st.markdown("#### 📉 Taux de survie par classe")
    hist_df = pd.DataFrame({
        "Classe": ["1ère 👑", "2ème 🎩", "3ème 🧳"],
        "Taux de survie (%)": [62, 43, 25],
    })
    fig_bar = go.Figure(go.Bar(
        x=hist_df["Classe"],
        y=hist_df["Taux de survie (%)"],
        marker_color=["#10b981", "#f59e0b", "#ef4444"],
        text=hist_df["Taux de survie (%)"].apply(lambda v: f"{v} %"),
        textposition="outside",
    ))
    fig_bar.update_layout(
        yaxis={"range": [0, 80], "title": "Taux de survie (%)"},
        height=320,
        margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### ⚧️ Taux de survie par sexe")
    sex_df = pd.DataFrame({
        "Sexe": ["♀️ Femmes", "♂️ Hommes"],
        "Taux de survie (%)": [74, 19],
    })
    fig_sex = go.Figure(go.Bar(
        x=sex_df["Sexe"],
        y=sex_df["Taux de survie (%)"],
        marker_color=["#818cf8", "#38bdf8"],
        text=sex_df["Taux de survie (%)"].apply(lambda v: f"{v} %"),
        textposition="outside",
    ))
    fig_sex.update_layout(
        yaxis={"range": [0, 90], "title": "Taux de survie (%)"},
        height=280,
        margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    st.plotly_chart(fig_sex, use_container_width=True)
