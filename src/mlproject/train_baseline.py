"""Entrainement de la baseline personnelle (régression logistique + feature engineering).

Script autonome, sans MLflow. Lance l'entraînement complet sur le dataset Titanic
avec le pipeline de préprocessing défini dans features.py et config.py.

Usage :
    uv run python -m mlproject.train_baseline
    uv run python -m mlproject.train_baseline --c 0.5 --max-iter 500
    make train-baseline
"""
from __future__ import annotations

import argparse

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from mlproject.config import CATEGORICAL_FEATURES, MODEL_DIR, NUMERIC_FEATURES, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", LogisticRegression(C=c, max_iter=max_iter, random_state=RANDOM_STATE)),
    ])


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    print(f"Train : {len(x_train)} lignes | Test : {len(x_test)} lignes")
    print(f"Features numériques  : {NUMERIC_FEATURES}")
    print(f"Features catégorielles : {CATEGORICAL_FEATURES}")
    print(f"Hyperparamètres : C={c}, max_iter={max_iter}")
    print("-" * 50)

    model = build_model(c=c, max_iter=max_iter)
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")
    print()
    print("Rapport de classification :")
    print(classification_report(y_test, preds, target_names=["Décédé (0)", "Survivant (1)"]))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, preds))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model_baseline.joblib"
    joblib.dump(model, model_path)
    print(f"\nModèle sauvegardé → {model_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement baseline — régression logistique")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse de la régularisation (défaut : 1.0)")
    parser.add_argument("--max-iter", type=int, default=1000, help="Nombre max d'itérations (défaut : 1000)")
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
