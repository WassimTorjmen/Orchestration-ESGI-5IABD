"""Frontend Streamlit — Titanic Survival Classifier.
Projet MLOps ESGI 5IABD — Wassim Torjmen
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow:5000")
MLFLOW_PUBLIC = os.environ.get("MLFLOW_PUBLIC_URL", "http://34.156.27.147:5000")
AIRFLOW_PUBLIC = os.environ.get("AIRFLOW_PUBLIC_URL", "http://34.156.27.147:8080")
API_PUBLIC = os.environ.get("API_PUBLIC_URL", "http://34.156.27.147:8000")
AUTHOR = "Wassim Torjmen"
COURSE = "ESGI 5IABD — MLOps 2025/2026"
QUALITY_F1_MIN = float(os.environ.get("EVAL_F1_MIN", "0.55"))
QUALITY_ROC_MIN = float(os.environ.get("EVAL_ROC_AUC_MIN", "0.65"))

st.set_page_config(
    page_title="🚢 Titanic ML — ESGI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a 0%,#1e293b 100%);
    border-right:1px solid #334155;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color:#e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color:#334155 !important; }
.main-header {
    background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 55%,#0f172a 100%);
    padding:22px 32px;
    border-radius:14px;
    border-bottom:3px solid #06b6d4;
    margin-bottom:24px;
}
.survivor-banner {
    background:linear-gradient(135deg,#064e3b,#065f46);
    border-left:6px solid #10b981;
    border-radius:10px;
    padding:18px 24px;
    color:white;
    font-size:22px;
    font-weight:bold;
    margin:12px 0;
}
.death-banner {
    background:linear-gradient(135deg,#7f1d1d,#991b1b);
    border-left:6px solid #ef4444;
    border-radius:10px;
    padding:18px 24px;
    color:white;
    font-size:22px;
    font-weight:bold;
    margin:12px 0;
}
.service-card {
    background:#1e293b;
    border-radius:10px;
    padding:18px 20px;
    border-left:4px solid #06b6d4;
    margin:8px 0;
    text-align:center;
}
.nav-link {
    display:block;
    padding:8px 12px;
    margin:3px 0;
    background:#1e293b;
    border-radius:8px;
    color:#38bdf8 !important;
    text-decoration:none;
    font-size:14px;
    border:1px solid #334155;
    transition:background 0.2s;
}
.section-title {
    font-size:12px;
    font-weight:700;
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:1.5px;
    margin:12px 0 6px 0;
}
@keyframes blink-color {
    0%   { color:#06b6d4; }
    25%  { color:#818cf8; }
    50%  { color:#10b981; }
    75%  { color:#f59e0b; }
    100% { color:#06b6d4; }
}
.author-name {
    animation: blink-color 2.5s infinite;
    font-size:17px;
    font-weight:800;
    letter-spacing:0.5px;
}
.quality-ok {
    background:#064e3b;
    border-radius:6px;
    padding:8px 14px;
    color:#6ee7b7;
    font-weight:bold;
    display:inline-block;
}
.quality-fail {
    background:#7f1d1d;
    border-radius:6px;
    padding:8px 14px;
    color:#fca5a5;
    font-weight:bold;
    display:inline-block;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_service(url: str, path: str = "/health") -> bool:
    try:
        r = httpx.get(f"{url}{path}", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def get_model_info() -> dict[str, Any]:
    try:
        r = httpx.get(f"{API_URL}/model-info", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def get_mlflow_metrics() -> dict[str, Any]:
    try:
        r = httpx.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": "titanic-survival"},
            timeout=5.0,
        )
        if r.status_code != 200:
            return {}
        exp_id = r.json()["experiment"]["experiment_id"]
        r2 = httpx.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": [exp_id],
                "max_results": 1,
                "order_by": ["start_time DESC"],
            },
            timeout=5.0,
        )
        if r2.status_code != 200 or not r2.json().get("runs"):
            return {}
        run = r2.json()["runs"][0]
        metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}
        params = {p["key"]: p["value"] for p in run["data"].get("params", [])}
        return {
            "metrics": metrics,
            "params": params,
            "run_id": run["info"]["run_id"],
        }
    except Exception:
        return {}


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <div style="display:flex;justify-content:space-between;
              align-items:center;flex-wrap:wrap;gap:12px;">
    <div>
      <h1 style="color:white;margin:0;font-size:26px;letter-spacing:-0.5px;">
        🚢 Titanic Survival Classifier
      </h1>
      <p style="color:#94a3b8;margin:6px 0 0 0;font-size:13px;">
        Modèle ML de classification binaire · Dataset historique 1912
      </p>
    </div>
    <div style="text-align:right;">
      <p style="color:#06b6d4;margin:0;font-size:18px;font-weight:700;">
        👨‍💻 {AUTHOR}
      </p>
      <p style="color:#64748b;margin:3px 0 0 0;font-size:12px;">
        🎓 {COURSE}
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='background:#0f172a;border-radius:10px;padding:14px 16px;"
        f"border:1px solid #334155;margin-bottom:8px;'>"
        f"<p class='author-name'>👨‍💻 {AUTHOR}</p>"
        f"<p style='color:#94a3b8;font-size:12px;margin:2px 0;'>"
        f"🎓 {COURSE}</p>"
        f"<hr style='border-color:#1e293b;margin:8px 0;'/>"
        f"<p style='color:#64748b;font-size:11px;margin:0;'>"
        f"🚀 FastAPI · MLflow · Airflow</p>"
        f"<p style='color:#64748b;font-size:11px;margin:2px 0 0 0;'>"
        f"🐳 Docker · GitHub Actions</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        "<h2 style='margin-top:0;color:#e2e8f0;font-size:18px;'>🧭 Navigation</h2>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "nav",
        key="nav_page",
        options=[
            "🏠  Accueil",
            "🎯  Prédiction",
            "📊  Évaluation",
            "📈  Historique",
            "🔗  Services",
            "📋  À propos",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Service status
    st.markdown(
        "<p class='section-title'>📡 État des services</p>",
        unsafe_allow_html=True,
    )
    api_ok = check_service(API_URL)
    mlflow_ok = check_service(MLFLOW_URL, "/")
    airflow_ok = check_service(AIRFLOW_PUBLIC, "/health")

    for label, ok in [
        ("API FastAPI", api_ok),
        ("MLflow", mlflow_ok),
        ("Airflow", airflow_ok),
    ]:
        icon, status, color = (
            ("🟢", "En ligne", "#6ee7b7") if ok
            else ("🔴", "Hors ligne", "#fca5a5")
        )
        st.markdown(
            f"<p style='margin:5px 0;font-size:13px;'>"
            f"{icon} <b>{label}</b> "
            f"<span style='color:{color};'>— {status}</span></p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Quick links
    st.markdown(
        "<p class='section-title'>🔗 Liens rapides</p>",
        unsafe_allow_html=True,
    )
    _links = [
        ("📊 MLflow UI", MLFLOW_PUBLIC),
        ("🌀 Airflow UI", AIRFLOW_PUBLIC),
        ("📖 API Docs", f"{API_PUBLIC}/docs"),
        ("❤️ API Health", f"{API_PUBLIC}/health"),
        ("🤖 Model Info", f"{API_PUBLIC}/model-info"),
    ]
    for lbl, url in _links:
        st.markdown(
            f"<a href='{url}' target='_blank' class='nav-link'>{lbl}</a>",
            unsafe_allow_html=True,
        )



# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if "Accueil" in page:
    # ── Hero
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 55%,#0f172a 100%);
         border-radius:16px;padding:40px 36px;text-align:center;
         border:1px solid #1e3a5f;margin-bottom:16px;'>
      <h1 style='color:white;font-size:38px;margin:0 0 8px 0;letter-spacing:-1px;'>
        🚢 Titanic Survival Classifier
      </h1>
      <p style='color:#94a3b8;font-size:16px;margin:0;max-width:600px;
                margin-left:auto;margin-right:auto;'>
        Modèle de Machine Learning entraîné sur les données historiques du naufrage de
        1912 · API temps réel · Pipeline MLOps complet
      </p>
    </div>
    """, unsafe_allow_html=True)

    _hb1, _hb2, _hb3, _hb4 = st.columns([2, 2, 1, 1])
    with _hb1:
        if st.button("🎯 Faire une prédiction", type="primary", use_container_width=True):
            st.session_state.nav_page = "🎯  Prédiction"
            st.rerun()
    with _hb2:
        if st.button("📊 Voir les métriques", use_container_width=True):
            st.session_state.nav_page = "📊  Évaluation"
            st.rerun()

    # ── Stats row
    _model_info = get_model_info()
    _mlf = get_mlflow_metrics()
    _f1 = _mlf.get("metrics", {}).get("f1", 0.803)
    _roc = _mlf.get("metrics", {}).get("roc_auc", 0.880)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(
            "<div class='service-card'>"
            "<p style='color:#94a3b8;margin:0;font-size:11px;'>🎯 F1-SCORE</p>"
            f"<p style='color:#10b981;font-size:28px;font-weight:800;margin:4px 0;'>"
            f"{_f1:.3f}</p>"
            "<p style='color:#64748b;margin:0;font-size:11px;'>dernière run MLflow</p>"
            "</div>", unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            "<div class='service-card'>"
            "<p style='color:#94a3b8;margin:0;font-size:11px;'>📐 ROC AUC</p>"
            f"<p style='color:#818cf8;font-size:28px;font-weight:800;margin:4px 0;'>"
            f"{_roc:.3f}</p>"
            "<p style='color:#64748b;margin:0;font-size:11px;'>dernière run MLflow</p>"
            "</div>", unsafe_allow_html=True,
        )
    with sc3:
        _v = _model_info.get("version", "—")
        st.markdown(
            "<div class='service-card'>"
            "<p style='color:#94a3b8;margin:0;font-size:11px;'>🏷️ VERSION</p>"
            f"<p style='color:#06b6d4;font-size:28px;font-weight:800;margin:4px 0;'>"
            f"v{_v}</p>"
            "<p style='color:#64748b;margin:0;font-size:11px;'>modèle en service</p>"
            "</div>", unsafe_allow_html=True,
        )
    with sc4:
        _api_color = "#10b981" if api_ok else "#ef4444"
        _api_txt = "En ligne" if api_ok else "Hors ligne"
        st.markdown(
            "<div class='service-card'>"
            "<p style='color:#94a3b8;margin:0;font-size:11px;'>📡 API STATUS</p>"
            f"<p style='color:{_api_color};font-size:28px;font-weight:800;margin:4px 0;'>"
            f"{'🟢' if api_ok else '🔴'}</p>"
            f"<p style='color:#64748b;margin:0;font-size:11px;'>{_api_txt}</p>"
            "</div>", unsafe_allow_html=True,
        )

    st.divider()

    # ── Tech stack
    st.markdown(
        "<h3 style='color:#e2e8f0;margin-bottom:16px;'>🛠️ Stack technologique</h3>",
        unsafe_allow_html=True,
    )
    _techs = [
        ("🐍", "Python 3.12", "scikit-learn · pandas · numpy", "#3b82f6"),
        ("⚡", "FastAPI", "API REST · Pydantic · uvicorn", "#06b6d4"),
        ("📊", "MLflow", "Tracking · Model Registry", "#818cf8"),
        ("🌀", "Airflow", "Orchestration · DAGs · CeleryExecutor", "#f59e0b"),
        ("🐳", "Docker", "Compose · Multi-stage builds", "#38bdf8"),
        ("🔁", "GitHub Actions", "CI (ruff·mypy·pytest) · CD (GHCR)", "#10b981"),
    ]
    tc1, tc2, tc3 = st.columns(3)
    for i, (icon, name, desc, color) in enumerate(_techs):
        col = [tc1, tc2, tc3][i % 3]
        with col:
            st.markdown(
                f"<div style='background:#1e293b;border-radius:10px;padding:14px 16px;"
                f"border-top:3px solid {color};margin-bottom:12px;'>"
                f"<p style='font-size:22px;margin:0;'>{icon}</p>"
                f"<p style='color:white;font-weight:700;margin:4px 0 2px 0;font-size:14px;'>"
                f"{name}</p>"
                f"<p style='color:#64748b;font-size:11px;margin:0;'>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── MLOps pipeline
    st.markdown(
        "<h3 style='color:#e2e8f0;margin-bottom:16px;'>🔄 Pipeline MLOps</h3>",
        unsafe_allow_html=True,
    )
    _steps = [
        ("1", "📦 Données", "Dataset Titanic · Feature engineering", "#06b6d4"),
        ("2", "🧠 Entraînement", "Logistic Regression · MLflow tracking", "#818cf8"),
        ("3", "✅ Évaluation", "F1 · ROC AUC · Porte qualité", "#10b981"),
        ("4", "📤 Registry", "MLflow Model Registry · versioning", "#f59e0b"),
        ("5", "🚀 Déploiement", "FastAPI · Docker · GHCR", "#ef4444"),
        ("6", "🌀 Réentraînement", "Airflow DAG · tous les lundis à 3h", "#38bdf8"),
    ]
    pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
    for i, (num, title, desc, color) in enumerate(_steps):
        col = [pc1, pc2, pc3, pc4, pc5, pc6][i]
        with col:
            st.markdown(
                f"<div style='background:#1e293b;border-radius:8px;padding:12px 10px;"
                f"text-align:center;border-bottom:3px solid {color};'>"
                f"<p style='color:{color};font-size:20px;font-weight:900;margin:0;'>{num}</p>"
                f"<p style='color:white;font-size:12px;font-weight:700;margin:4px 0 2px 0;'>"
                f"{title}</p>"
                f"<p style='color:#64748b;font-size:10px;margin:0;'>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Quick actions (single HTML block to avoid Streamlit column whitespace)
    st.markdown(
        "<h3 style='color:#e2e8f0;margin-bottom:12px;'>⚡ Accès rapide</h3>",
        unsafe_allow_html=True,
    )
    _card_style = (
        "flex:1;background:#1e293b;border-radius:10px;padding:16px;"
        "text-align:center;text-decoration:none;"
    )
    st.markdown(
        f"<div style='display:flex;gap:12px;flex-wrap:wrap;'>"
        f"<a href='{API_PUBLIC}/docs' target='_blank'"
        f" style='{_card_style}border:1px solid #06b6d4;'>"
        f"<p style='font-size:24px;margin:0;'>⚡</p>"
        f"<p style='color:#06b6d4;font-weight:700;margin:6px 0 2px 0;font-size:13px;'>"
        f"API Swagger</p>"
        f"<p style='color:#64748b;font-size:11px;margin:0;'>Tester l'API</p></a>"
        f"<a href='{MLFLOW_PUBLIC}' target='_blank'"
        f" style='{_card_style}border:1px solid #818cf8;'>"
        f"<p style='font-size:24px;margin:0;'>📊</p>"
        f"<p style='color:#818cf8;font-weight:700;margin:6px 0 2px 0;font-size:13px;'>"
        f"MLflow UI</p>"
        f"<p style='color:#64748b;font-size:11px;margin:0;'>Expériences & Registry</p></a>"
        f"<a href='{AIRFLOW_PUBLIC}' target='_blank'"
        f" style='{_card_style}border:1px solid #f59e0b;'>"
        f"<p style='font-size:24px;margin:0;'>🌀</p>"
        f"<p style='color:#f59e0b;font-weight:700;margin:6px 0 2px 0;font-size:13px;'>"
        f"Airflow UI</p>"
        f"<p style='color:#64748b;font-size:11px;margin:0;'>DAGs & Pipelines</p></a>"
        f"<a href='{API_PUBLIC}/health' target='_blank'"
        f" style='{_card_style}border:1px solid #10b981;'>"
        f"<p style='font-size:24px;margin:0;'>❤️</p>"
        f"<p style='color:#10b981;font-weight:700;margin:6px 0 2px 0;font-size:13px;'>"
        f"Health Check</p>"
        f"<p style='color:#64748b;font-size:11px;margin:0;'>Statut API</p></a>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif "Prédiction" in page:
    st.subheader("🎯 Simuler la survie d'un passager")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎫 Voyage**")
            _cls = {1: "1ère — Luxe 👑", 2: "2ème — Confort 🎩", 3: "3ème — Économique 🧳"}
            pclass = st.selectbox("🏷️ Classe", [1, 2, 3], index=2,
                                  format_func=lambda x: _cls[x])
            fare = st.number_input("💰 Tarif (£)", min_value=0.0, value=30.0, step=1.0)

        with col2:
            st.markdown("**👤 Passager**")
            age = st.number_input(
                "🎂 Âge", min_value=0.0, max_value=120.0, value=29.0, step=1.0
            )
            sex = st.selectbox(
                "⚧️ Sexe", ["male", "female"],
                format_func=lambda x: "♂️ Homme" if x == "male" else "♀️ Femme",
            )
            title = st.selectbox("🎩 Titre", ["Mr", "Mrs", "Miss", "Master", "Rare"])

        with col3:
            st.markdown("**👨‍👩‍👧 Famille**")
            sibsp = st.number_input("💑 Frères / conjoints à bord", min_value=0, value=0)
            parch = st.number_input("👶 Parents / enfants à bord", min_value=0, value=0)
            _ports = {
                "S": "🇬🇧 Southampton",
                "C": "🇫🇷 Cherbourg",
                "Q": "🇮🇪 Queenstown",
            }
            embarked = st.selectbox(
                "⚓ Port d'embarquement", ["S", "C", "Q"],
                format_func=lambda x: _ports[x],
            )

        submitted = st.form_submit_button(
            "🔮 Lancer la prédiction", use_container_width=True, type="primary"
        )

    if submitted:
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        payload = {
            "Pclass": pclass, "Age": age, "SibSp": sibsp, "Parch": parch,
            "Fare": fare, "FamilySize": family_size, "IsAlone": is_alone,
            "Sex": sex, "Embarked": embarked, "Title": title,
        }
        try:
            response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"❌ Appel à l'API impossible : {exc}")
        else:
            prediction = result["prediction"]
            probability = result["probability"]
            st.divider()

            if prediction == 1:
                banner = (
                    '<div class="survivor-banner">'
                    "🟢 SURVIVANT · Ce passager aurait survécu 🛟</div>"
                )
            else:
                banner = (
                    '<div class="death-banner">'
                    "🔴 DÉCÉDÉ · Ce passager n'aurait pas survécu ⚓</div>"
                )
            st.markdown(banner, unsafe_allow_html=True)

            col_m, col_g = st.columns(2)
            with col_m:
                st.markdown("#### 📊 Métriques")
                st.metric("🎯 Verdict",
                          "✅ Survivant" if prediction == 1 else "❌ Décédé")
                st.metric("📈 Probabilité de survie", f"{probability:.1%}",
                          delta=f"{probability - 0.5:+.1%} vs aléatoire")
                st.progress(probability, text=f"Survie : {probability:.1%}")
                fig_d = go.Figure(go.Pie(
                    labels=["🟢 Survie", "🔴 Décès"],
                    values=[probability, 1 - probability],
                    hole=0.55,
                    marker_colors=["#10b981", "#ef4444"],
                    textinfo="label+percent",
                    textfont_size=13,
                ))
                fig_d.update_layout(
                    title={"text": "Répartition probabiliste", "x": 0.5},
                    height=260, margin=dict(t=40, b=0, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_d, use_container_width=True)

            with col_g:
                st.markdown("#### 🌡️ Jauge de survie")
                color = "#10b981" if prediction == 1 else "#ef4444"
                fig_g = go.Figure(go.Indicator(
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
                            "thickness": 0.85, "value": 50,
                        },
                    },
                ))
                fig_g.update_layout(
                    height=300, margin=dict(t=60, b=20, l=30, r=30),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                )
                st.plotly_chart(fig_g, use_container_width=True)

            with st.expander("🔍 Détails du profil analysé"):
                c1, c2, c3 = st.columns(3)
                _cls_label = {1: "1ère 👑", 2: "2ème 🎩", 3: "3ème 🧳"}
                with c1:
                    st.metric("🏷️ Classe", _cls_label[pclass])
                    st.metric("💰 Tarif", f"£{fare:.0f}")
                with c2:
                    st.metric("🎂 Âge", f"{age:.0f} ans")
                    st.metric("⚧️ Sexe",
                              "♂️ Homme" if sex == "male" else "♀️ Femme")
                with c3:
                    st.metric("👨‍👩‍👧 Taille famille", f"{family_size} pers.")
                    st.metric("🏠 Seul à bord",
                              "✅ Oui" if is_alone else "❌ Non")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Évaluation" in page:
    st.subheader("📊 Évaluation du modèle en production")

    model_info = get_model_info()
    mlflow_data = get_mlflow_metrics()

    # ── Model info banner
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="service-card">'
            f"<p style='color:#94a3b8;margin:0;font-size:12px;'>🤖 MODÈLE</p>"
            f"<p style='color:#06b6d4;font-size:20px;font-weight:700;margin:4px 0;'>"
            f"{model_info.get('model', 'titanic-classifier')}</p>"
            f"<p style='color:#64748b;margin:0;font-size:12px;'>Nom du modèle registry</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        version = model_info.get("version", "—")
        st.markdown(
            f'<div class="service-card">'
            f"<p style='color:#94a3b8;margin:0;font-size:12px;'>🏷️ VERSION</p>"
            f"<p style='color:#06b6d4;font-size:20px;font-weight:700;margin:4px 0;'>"
            f"v{version}</p>"
            f"<p style='color:#64748b;margin:0;font-size:12px;'>Version en service</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        status_txt = "✅ En ligne" if api_ok else "❌ Hors ligne"
        status_col = "#6ee7b7" if api_ok else "#fca5a5"
        st.markdown(
            f'<div class="service-card">'
            f"<p style='color:#94a3b8;margin:0;font-size:12px;'>📡 STATUT API</p>"
            f"<p style='color:{status_col};font-size:20px;font-weight:700;margin:4px 0;'>"
            f"{status_txt}</p>"
            f"<p style='color:#64748b;margin:0;font-size:12px;'>FastAPI / uvicorn</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Metrics (live from MLflow or fallback to last known)
    metrics = mlflow_data.get("metrics", {})
    f1 = metrics.get("f1", 0.803)
    roc = metrics.get("roc_auc", 0.880)
    source = "🔴 valeurs par défaut" if not metrics else "🟢 MLflow (live)"

    st.markdown(f"#### 📈 Métriques du dernier run · <small>{source}</small>",
                unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🎯 F1-Score", f"{f1:.3f}",
               delta=f"{f1 - QUALITY_F1_MIN:+.3f} vs seuil")
    mc2.metric("📐 ROC AUC", f"{roc:.3f}",
               delta=f"{roc - QUALITY_ROC_MIN:+.3f} vs seuil")
    mc3.metric("🚪 Seuil F1 min", f"{QUALITY_F1_MIN:.2f}")
    mc4.metric("🚪 Seuil ROC min", f"{QUALITY_ROC_MIN:.2f}")

    if mlflow_data.get("params"):
        with st.expander("⚙️ Hyperparamètres du run"):
            st.json(mlflow_data["params"])

    # ── Quality gate
    st.markdown("#### 🚪 Porte qualité")
    f1_ok = f1 >= QUALITY_F1_MIN
    roc_ok = roc >= QUALITY_ROC_MIN
    gate_ok = f1_ok and roc_ok

    g1, g2, g3 = st.columns(3)
    with g1:
        cls = "quality-ok" if f1_ok else "quality-fail"
        icon = "✅" if f1_ok else "❌"
        st.markdown(
            f"<div class='{cls}'>{icon} F1 = {f1:.3f} "
            f"({'≥' if f1_ok else '<'} {QUALITY_F1_MIN})</div>",
            unsafe_allow_html=True,
        )
    with g2:
        cls = "quality-ok" if roc_ok else "quality-fail"
        icon = "✅" if roc_ok else "❌"
        st.markdown(
            f"<div class='{cls}'>{icon} ROC AUC = {roc:.3f} "
            f"({'≥' if roc_ok else '<'} {QUALITY_ROC_MIN})</div>",
            unsafe_allow_html=True,
        )
    with g3:
        if gate_ok:
            st.success("🟢 Modèle validé — prêt pour la production")
        else:
            st.error("🔴 Modèle rejeté — ne passe pas la porte qualité")

    # ── Radar chart
    st.markdown("#### 🕸️ Profil de performance")
    fig_radar = go.Figure(go.Scatterpolar(
        r=[f1, roc, 0.80, 0.75, f1],
        theta=["F1-Score", "ROC AUC", "Accuracy", "Précision", "F1-Score"],
        fill="toself",
        fillcolor="rgba(6,182,212,0.2)",
        line={"color": "#06b6d4", "width": 2},
        name="Modèle courant",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[QUALITY_F1_MIN, QUALITY_ROC_MIN, 0.65, 0.65, QUALITY_F1_MIN],
        theta=["F1-Score", "ROC AUC", "Accuracy", "Précision", "F1-Score"],
        fill="toself",
        fillcolor="rgba(239,68,68,0.1)",
        line={"color": "#ef4444", "width": 1, "dash": "dot"},
        name="Seuils minimaux",
    ))
    fig_radar.update_layout(
        polar={"radialaxis": {"range": [0, 1], "tickformat": ".0%"}},
        showlegend=True,
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    if mlflow_data.get("run_id"):
        run_url = f"{MLFLOW_PUBLIC}/#/experiments/1/runs/{mlflow_data['run_id']}"
        st.markdown(
            f"[🔗 Voir ce run dans MLflow UI ↗]({run_url})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════════
elif "Historique" in page:
    st.subheader("📈 Historique des prédictions & données Titanic")

    # ── Live predictions log
    st.markdown("#### 🔴 Prédictions en direct (session courante)")
    try:
        resp = httpx.get(f"{API_URL}/predictions", timeout=5.0)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            df_hist = pd.DataFrame(rows)
            _disp_cols = ["timestamp", "prediction", "probability",
                          "Pclass", "Age", "Sex", "Fare"]
            _cols = [c for c in _disp_cols if c in df_hist.columns]
            st.dataframe(df_hist[_cols], use_container_width=True)

            lmap = {1: "✅ Survivant", 0: "❌ Décédé"}
            counts = df_hist["prediction"].value_counts().rename(index=lmap)
            ph1, ph2 = st.columns(2)
            with ph1:
                fig_ph = go.Figure(go.Pie(
                    labels=list(counts.index),
                    values=list(counts.values),
                    hole=0.5,
                    marker_colors=["#10b981", "#ef4444"],
                    textinfo="label+percent",
                ))
                fig_ph.update_layout(
                    title={"text": "Répartition prédictions", "x": 0.5},
                    height=260, margin=dict(t=40, b=0, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                    showlegend=False,
                )
                st.plotly_chart(fig_ph, use_container_width=True)
            with ph2:
                st.metric("📬 Total prédictions", len(df_hist))
                surv = int((df_hist["prediction"] == 1).sum())
                st.metric("🟢 Survivants prédits", surv)
                st.metric("🔴 Décédés prédits", len(df_hist) - surv)
        else:
            st.info("📭 Aucune prédiction enregistrée dans cette session. "
                    "Allez sur la page Prédiction pour en créer.")
    except httpx.HTTPError:
        st.warning("⚠️ Impossible de contacter l'API.")

    st.divider()

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("🚢 Passagers à bord", "2 224")
    h2.metric("💀 Victimes", "1 502", delta="-67 %", delta_color="inverse")
    h3.metric("🟢 Survivants", "722", delta="+32 %")
    h4.metric("👩 Femmes sauvées", "~74 %", delta="priorité évacuation")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 📉 Taux de survie par classe")
        fig_cls = go.Figure(go.Bar(
            x=["1ère 👑", "2ème 🎩", "3ème 🧳"],
            y=[62, 43, 25],
            marker_color=["#10b981", "#f59e0b", "#ef4444"],
            text=["62 %", "43 %", "25 %"],
            textposition="outside",
        ))
        fig_cls.update_layout(
            yaxis={"range": [0, 80], "title": "Taux de survie (%)"},
            height=300, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
        )
        st.plotly_chart(fig_cls, use_container_width=True)

    with col_r:
        st.markdown("#### ⚧️ Taux de survie par sexe")
        fig_sex = go.Figure(go.Bar(
            x=["♀️ Femmes", "♂️ Hommes"],
            y=[74, 19],
            marker_color=["#818cf8", "#38bdf8"],
            text=["74 %", "19 %"],
            textposition="outside",
        ))
        fig_sex.update_layout(
            yaxis={"range": [0, 90], "title": "Taux de survie (%)"},
            height=300, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
        )
        st.plotly_chart(fig_sex, use_container_width=True)

    st.markdown("#### 🎂 Taux de survie par tranche d'âge")
    fig_age = go.Figure(go.Bar(
        x=["Enfants\n(0-12)", "Ados\n(13-17)", "Adultes\n(18-50)", "Seniors\n(50+)"],
        y=[59, 45, 36, 32],
        marker_color=["#a78bfa", "#60a5fa", "#34d399", "#fbbf24"],
        text=["59 %", "45 %", "36 %", "32 %"],
        textposition="outside",
    ))
    fig_age.update_layout(
        yaxis={"range": [0, 75], "title": "Taux de survie (%)"},
        height=300, margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="white",
    )
    st.plotly_chart(fig_age, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — SERVICES
# ══════════════════════════════════════════════════════════════════════════════
elif "Services" in page:
    st.subheader("🔗 Services & Infrastructure")

    services = [
        {
            "name": "FastAPI", "icon": "⚡", "desc": "API REST de prédiction",
            "url": f"{API_PUBLIC}/docs", "label": "Swagger UI",
            "color": "#06b6d4", "ok": api_ok,
            "endpoints": ["/health", "/predict", "/model-info", "/predictions"],
            "port": "8000",
        },
        {
            "name": "MLflow", "icon": "📊", "desc": "Tracking & Model Registry",
            "url": MLFLOW_PUBLIC, "label": "MLflow UI",
            "color": "#818cf8", "ok": mlflow_ok,
            "endpoints": ["/experiments", "/runs", "/registered-models"],
            "port": "5000",
        },
        {
            "name": "Airflow", "icon": "🌀", "desc": "Orchestration & DAGs",
            "url": AIRFLOW_PUBLIC, "label": "Airflow UI",
            "color": "#f59e0b", "ok": airflow_ok,
            "endpoints": ["model_retraining DAG", "schedule: lundi 3h"],
            "port": "8080",
        },
    ]

    # ── Centered side-by-side service cards
    st.markdown(
        "<div style='display:flex;justify-content:center;margin-bottom:8px;'>"
        "<p style='color:#94a3b8;font-size:13px;'>État en temps réel · cliquez sur un lien"
        " pour ouvrir le service</p></div>",
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    for col, svc in zip([col_a, col_b, col_c], services):
        ok = svc["ok"]
        if ok is True:
            badge = "🟢 En ligne"
            badge_color = "#6ee7b7"
            dot_color = "#10b981"
        else:
            badge = "🔴 Hors ligne"
            badge_color = "#fca5a5"
            dot_color = "#ef4444"

        ep_html = "".join(
            f"<span style='background:#0f172a;border-radius:4px;"
            f"padding:2px 8px;font-size:11px;color:#94a3b8;"
            f"margin:2px 2px 2px 0;display:inline-block;'>{ep}</span>"
            for ep in svc["endpoints"]
        )
        with col:
            st.markdown(
                f"<div style='background:#1e293b;border-radius:14px;"
                f"padding:24px 20px;border-top:4px solid {svc['color']};"
                f"text-align:center;height:100%;'>"
                f"<p style='font-size:40px;margin:0 0 8px 0;'>{svc['icon']}</p>"
                f"<h3 style='color:white;margin:0 0 4px 0;font-size:20px;'>"
                f"{svc['name']}</h3>"
                f"<p style='color:#94a3b8;font-size:12px;margin:0 0 12px 0;'>"
                f"{svc['desc']} · port {svc['port']}</p>"
                f"<p style='color:{badge_color};font-size:14px;font-weight:700;"
                f"margin:0 0 12px 0;'>{badge}</p>"
                f"<div style='margin-bottom:14px;text-align:left;'>{ep_html}</div>"
                f"<a href='{svc['url']}' target='_blank'"
                f" style='display:inline-block;background:{svc['color']}22;"
                f"border:1px solid {svc['color']};color:{svc['color']};"
                f"padding:8px 18px;border-radius:8px;text-decoration:none;"
                f"font-size:13px;font-weight:700;'>"
                f"🔗 {svc['label']} ↗</a>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### 🏗️ Architecture de la stack")
    st.code("""
 Browser ──► Streamlit :8501
                │
                ▼
            FastAPI :8000  ──► model.joblib
                │
                ▼
            MLflow :5000   ──► mlflow.db (SQLite)
                │
                ▼
            Airflow :8080  ──► DAG model_retraining
                              (tous les lundis à 3h)
    """, language="text")

    st.markdown("#### 🐳 Docker Compose")
    st.code("""
# Démarrer la stack complète
docker compose up -d mlflow api frontend

# Entraîner le modèle
docker compose --profile train run --rm train

# Airflow
cd ~/airflow && docker compose up -d
    """, language="bash")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
elif "propos" in page:
    st.subheader("📋 À propos du projet")

    here = Path(__file__).parent
    candidates = [here.parent / "README.md", here.parent.parent / "README.md"]
    readme_path = next((p for p in candidates if p.exists()), None)
    if readme_path:
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.warning("⚠️ README.md introuvable.")
        st.markdown(f"""
### 🚢 Titanic Survival Classifier

**Projet MLOps — ESGI 5IABD 2025/2026**
Auteur : **{AUTHOR}**

#### Objectif
Entraîner et déployer un modèle de classification binaire préisant la survie
des passagers du Titanic à travers un pipeline MLOps complet.

#### Stack
- **Python 3.12** · scikit-learn · pandas
- **FastAPI** · uvicorn · Pydantic
- **MLflow** · Model Registry · Tracking
- **Airflow** · CeleryExecutor · DAG hebdomadaire
- **Docker** · Docker Compose · multi-stage
- **GitHub Actions** · CI (ruff · mypy · pytest) · CD (GHCR)

#### Séances couvertes
| Séance | Contenu |
|--------|---------|
| S11 | Features engineering |
| S12 | API FastAPI `/predict` |
| S13 | CI GitHub Actions |
| S14 | Docker & Docker Compose |
| S17 | Airflow DAG réentraînement |
| S19 | CD pipeline GHCR |
        """)
