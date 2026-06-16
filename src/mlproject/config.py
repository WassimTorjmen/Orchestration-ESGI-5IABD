"""Configuration centrale du projet de classification.

C'est le SEUL fichier a adapter pour brancher votre propre jeu de donnees :
data.py, features.py et les scripts d'entrainement lisent toutes leurs
colonnes via ces constantes. Voir tp/TP_S0_projet_personnel.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# TODO (S0-1) : chemin vers votre fichier de donnees (CSV) place dans data/
DATA_PATH = ROOT / "src" / "data" / "Titanic-Dataset.csv"
MODEL_DIR = ROOT / "models"

# TODO (S0-2) : nom de la colonne cible binaire (valeurs 0/1)
TARGET = "Survived"

# TODO (S0-3) : colonnes numeriques de votre dataset
NUMERIC_FEATURES: list[str] = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]

# TODO (S0-4) : colonnes categorielles (peut rester vide : [])
CATEGORICAL_FEATURES: list[str] = ["Sex", "Embarked", "Title"]

RANDOM_STATE = 42

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "titanic-survival")
MLFLOW_EXPERIMENT_DESCRIPTION = os.getenv("MLFLOW_EXPERIMENT_DESCRIPTION", "")
_tags_raw = os.getenv("MLFLOW_EXPERIMENT_TAGS", "")
MLFLOW_EXPERIMENT_TAGS: dict[str, str] = dict(
    pair.split("=", 1) for pair in _tags_raw.split(",") if "=" in pair
)
MODEL_NAME = os.getenv("MODEL_NAME", "titanic-classifier")

# Seuils de la porte qualite (mlproject.evaluate)
EVAL_ROC_AUC_MIN = float(os.getenv("EVAL_ROC_AUC_MIN", "0.65"))
EVAL_F1_MIN = float(os.getenv("EVAL_F1_MIN", "0.55"))
