# Projet MLOps — Prédiction de Survie sur le Titanic

## Problématique

**Peut-on prédire si un passager du Titanic a survécu au naufrage, à partir de ses caractéristiques personnelles et de son billet ?**

Le naufrage du RMS Titanic (15 avril 1912) est l'une des catastrophes maritimes les plus meurtrières. Sur 2 224 passagers et membres d'équipage, 1 502 ont péri. L'analyse des données révèle que certains groupes (femmes, enfants, passagers de 1ère classe) avaient une probabilité de survie significativement plus élevée, en raison de l'ordre d'évacuation et du nombre limité de canots de sauvetage.

**Cible binaire** :
- `1` → passager survécu
- `0` → passager décédé

L'intérêt de ce problème est double : il illustre un cas de classification binaire déséquilibrée (38 % de survivants), et il soulève des questions d'équité (biais de genre, de classe sociale) que les métriques classiques comme la précision peuvent masquer.

## Dataset

**Source** : [Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/competitions/titanic)

| Colonne | Type | Description |
|---|---|---|
| `Survived` | Cible (0/1) | 0 = décédé, 1 = survivant |
| `Pclass` | Numérique | Classe du billet (1 = 1ère, 2 = 2ème, 3 = 3ème) |
| `Sex` | Catégorielle | Sexe du passager |
| `Age` | Numérique | Âge en années |
| `SibSp` | Numérique | Nombre de frères/sœurs ou conjoints à bord |
| `Parch` | Numérique | Nombre de parents ou enfants à bord |
| `Fare` | Numérique | Tarif du billet |
| `Embarked` | Catégorielle | Port d'embarquement (C = Cherbourg, Q = Queenstown, S = Southampton) |

891 passagers, 12 colonnes (après exclusion de `Name`, `Ticket`, `Cabin`).

## Stack technique

- **Python 3.12** — géré par [uv](https://docs.astral.sh/uv/)
- **scikit-learn** — pipeline de préprocessing + modèles baseline
- **MLflow** — tracking des expériences + Model Registry
- **Optuna** — optimisation des hyperparamètres
- **FastAPI + uvicorn** — API de prédiction
- **Streamlit** — frontend de démonstration
- **Airflow** — DAG de ré-entraînement planifié
- **Docker + docker-compose** — conteneurisation de la stack complète
- **GitHub Actions** — CI/CD

## Arborescence

```
.
├── README.md
├── Makefile                     # commandes du projet (install, train, api…)
├── pyproject.toml               # dépendances Python (uv)
├── uv.lock
├── models/                      # modèles entraînés (gitignore)
├── .github/
│   └── workflows/
│       └── ci_cd.yaml           # pipeline CI GitHub Actions
└── src/
    ├── data/                    # Titanic-Dataset.csv (gitignore)
    ├── mlproject/
    │   ├── config.py            # configuration dataset et chemins  ← TP S0
    │   ├── data.py              # chargement et split
    │   ├── features.py          # pipeline de préprocessing
    │   ├── tracking.py          # setup_experiment, log_dataset     ← TP S5
    │   ├── train.py             # entraînement baseline (MLflow)    ← TP S5
    │   ├── train_optuna.py      # optimisation Optuna + Registry    ← TP S6
    │   ├── train_models.py      # comparaison de modèles + SHAP     ← TP S7
    │   ├── api.py               # API FastAPI                       ← TP S12
    │   └── evaluation.py        # métriques et plots SHAP
    ├── frontend/
    │   └── app.py               # interface Streamlit               ← TP S14bis
    ├── docker/
    │   ├── Dockerfile.train                                         ← TP S8
    │   ├── Dockerfile.api
    │   └── Dockerfile.frontend
    ├── docker-compose.yml                                           ← TP S14
    └── dags/
        └── retrain_dag.py       # DAG Airflow de ré-entraînement   ← TP S17
```

## Résultats

### S0 — Baseline (régression logistique, features brutes)

| Modèle | F1 | ROC AUC |
|---|---|---|
| Baseline (features brutes) | 0.724 | 0.844 |
| + feature engineering & RobustScaler | **0.803** | **0.880** |

**Améliorations apportées (`features.py`) :**
- `FamilySize = SibSp + Parch + 1` — la taille du groupe familial influe sur la priorité d'évacuation
- `IsAlone` — les voyageurs seuls avaient significativement moins de chances de survie
- `Title` extrait de `Name` (Mr / Mrs / Miss / Master / Rare) — prédicteur fort lié au genre et à l'âge
- `StandardScaler` → `RobustScaler` — robuste aux valeurs aberrantes de `Fare` (plage 0–512)

### S5 — MLflow Tracking (régression logistique)

Suivi des expériences dans `train.py` : paramètres (`C`, `max_iter`), métriques (`f1`, `roc_auc`), modèle et matrice de confusion loggués dans MLflow.

**Partie 2 — Centralisation (`tracking.py`) :**
- `setup_experiment()` : configure `tracking_uri`, crée/sélectionne l'expérience et y applique description + tags (lus depuis `.env` via `config.py`) — idempotent.
- `log_dataset()` : trace le DataFrame source dans l'onglet "Datasets" de l'UI MLflow (`mlflow.log_input`).
- `train.py` et `train_models.py` utilisent désormais ces deux fonctions au lieu de dupliquer la configuration MLflow.

### S6 — Optimisation Optuna (TPE, n_trials=30, cv=5)

Chaque famille (Random Forest, XGBoost, LightGBM) est optimisée par une étude Optuna indépendante (sampler TPE). Les 30 essais de chaque famille sont tracés dans MLflow comme runs imbriqués (`trial-0` … `trial-29`), et le meilleur modèle toutes familles confondues est enregistré dans le Model Registry avec description et tags.

**Structure MLflow :** `optuna-compare` → `random_forest` / `xgboost` / `lightgbm` → `trial-N`

| Famille | Espace de recherche |
|---|---|
| Random Forest | `n_estimators` [100-300], `max_depth` {None,10,20,30}, `min_samples_leaf` [1-5] |
| XGBoost | `n_estimators` [100-300], `max_depth` [3-10], `learning_rate` log[0.01-0.3] |
| LightGBM | `n_estimators` [50-300], `num_leaves` [15-127], `learning_rate` log[0.01-0.3], `max_depth` [3-12] |

### S7 — Comparaison de modèles (GridSearchCV, cv=5, scoring=roc_auc)

| Modèle | F1 | ROC AUC | Meilleurs hyperparamètres |
|---|---|---|---|
| Random Forest | 0.766 | 0.842 | `max_depth=None, min_samples_leaf=2, n_estimators=100` |
| **XGBoost** ✓ | 0.739 | **0.859** | `learning_rate=0.01, max_depth=3, n_estimators=200` |
| LightGBM | 0.766 | 0.832 | `learning_rate=0.1, n_estimators=100, num_leaves=31` |

**XGBoost** est le meilleur modèle (ROC AUC = 0.859). Le modèle gagnant est enregistré dans le Model Registry MLflow sous `titanic-classifier` et sauvegardé dans `models/model.joblib`. Chaque run inclut : matrice de confusion, rapport de classification, et summary plot SHAP (importance des variables).

### S12 — API FastAPI (`src/mlproject/api.py`)

Le modèle est exposé via une API HTTP REST :

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Statut de l'API |
| `/predict` | POST | Prédiction de survie |
| `/model-info` | GET | Nom et version du modèle servi (bonus) |

Le modèle est chargé **une seule fois au démarrage** via le mécanisme `lifespan` de FastAPI. Les entrées sont validées par Pydantic (`Pclass` ∈ [1,3], `Age` ∈ [0,120], etc.) — une entrée invalide renvoie automatiquement un **422**.

La version servie est lue automatiquement depuis le **MLflow Model Registry** (SQLite local `mlflow.db`) au démarrage, sans nécessiter de serveur MLflow actif. Fallback sur la variable d'env `MODEL_VERSION`.

**Exemples de requêtes :**
```bash
# Prédiction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 1, "Age": 29.0, "SibSp": 0, "Parch": 0, "Fare": 211.3,
       "FamilySize": 1, "IsAlone": 1, "Sex": "female", "Embarked": "S", "Title": "Miss"}'
# → {"prediction": 1, "probability": 0.9723}

# Version du modèle
curl http://127.0.0.1:8000/model-info
# → {"version": "3", "model": "titanic-classifier"}
```

## Mise en route

```bash
make install                  # installe les dépendances (uv)
cp .env.example .env          # copier la configuration (adapter si besoin)

# Terminal 1 — serveur MLflow
make mlflow                   # http://localhost:5000

# Terminal 2 — entraînements
make train                    # baseline logreg (C=1.0)
make train C=0.1              # variante hyperparamètre
make train-models             # RF / XGBoost / LightGBM (GridSearchCV, cv=5)
make train-optuna N_TRIALS=30 # RF / XGBoost / LightGBM (Optuna TPE)
make evaluate                 # porte qualité sur le modèle du registry

# Terminal 3 — API d'inférence
make api                      # http://localhost:8000/docs

# Qualité du code
make lint                     # ruff check
make format                   # ruff format
```

## Feuille de route des TP

| Séance | Fichier à compléter | Objectif |
|---|---|---|
| S0 | `src/mlproject/config.py` | Brancher le dataset Titanic ✅ |
| S5 | `src/mlproject/train.py`, `tracking.py` | Suivi MLflow + centralisation config + dataset lineage ✅ |
| S6 | `src/mlproject/train_optuna.py` | Optimisation Optuna (TPE) + Model Registry ✅ |
| S7 | `src/mlproject/train_models.py` | Comparaison de modèles (GridSearchCV) + SHAP ✅ |
| S8 | `src/docker/Dockerfile.train` | Conteneuriser l'entraînement |
| S12 | `src/mlproject/api.py` | Exposer le modèle via FastAPI ✅ |
| S14 | `src/docker-compose.yml` | Orchestrer la stack |
| S14bis | `src/frontend/app.py` | Frontend Streamlit |
| S8 | `src/docker/Dockerfile.train` | Conteneuriser l'entraînement |
| S12 | `src/mlproject/api.py` | Exposer le modèle via FastAPI ✅ |
| S14 | `src/docker-compose.yml` | Orchestrer la stack |
| S14bis | `src/frontend/app.py` | Frontend Streamlit |
| S17 | `src/dags/retrain_dag.py` | Planifier le ré-entraînement avec Airflow |
| S18 | `.github/workflows/ci_cd.yaml` | Pipeline CI/CD GitHub Actions ✅ |

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci_cd.yaml`) se déclenche sur chaque push et PR vers `master` :

| Étape | Outil | Détail |
|---|---|---|
| Lint | `ruff check` | Erreurs de syntaxe et logiques |
| Format | `ruff format --check` | Cohérence du style |
| Import check | `python -c "..."` | Vérifie que le projet s'importe |

## Suivi GitHub

L'enseignant doit être ajouté comme collaborateur :
**Settings > Collaborators > Add people > `lewishkpv`**
