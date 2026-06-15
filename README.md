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
└── src/
    ├── data/                    # Titanic-Dataset.csv (gitignore)
    ├── mlproject/
    │   ├── config.py            # configuration dataset et chemins  ← TP S0
    │   ├── data.py              # chargement et split
    │   ├── features.py          # pipeline de préprocessing
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

## Mise en route

```bash
make install                                    # installe les dépendances (uv)
uv run python -m mlproject.train               # entraîne la baseline -> affiche f1 et roc_auc
```

## Feuille de route des TP

| Séance | Fichier à compléter | Objectif |
|---|---|---|
| S0 | `src/mlproject/config.py` | Brancher le dataset Titanic |
| S5 | `src/mlproject/train.py` | Suivi d'expériences MLflow |
| S6 | `src/mlproject/train_optuna.py` | Optimisation Optuna + Model Registry |
| S7 | `src/mlproject/train_models.py` | Comparaison de modèles (GridSearchCV) + SHAP |
| S8 | `src/docker/Dockerfile.train` | Conteneuriser l'entraînement |
| S12 | `src/mlproject/api.py` | Exposer le modèle via FastAPI |
| S14 | `src/docker-compose.yml` | Orchestrer la stack |
| S14bis | `src/frontend/app.py` | Frontend Streamlit |
| S17 | `src/dags/retrain_dag.py` | Planifier le ré-entraînement avec Airflow |

## Suivi GitHub

L'enseignant doit être ajouté comme collaborateur :
**Settings > Collaborators > Add people > `lewishkpv`**
