"""DAG Airflow - pipeline de re-entrainement du modele.

Seance 17 - TP Airflow
    Pipeline : preparation des donnees -> entrainement -> controle qualite.
    Planifie tous les lundis a 3h du matin.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    # S17-1 : verifier que le dataset est lisible et logger ses statistiques
    from mlproject.data import load_data

    df = load_data()
    logger.info("Dataset charge : %d lignes, %d colonnes", len(df), len(df.columns))
    logger.info("Colonnes : %s", list(df.columns))


def task_train(**context) -> None:
    # S17-2 : entrainer le modele et pousser le f1 dans XCom
    from mlproject.train import train

    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    logger.info("Entrainement termine : f1=%.3f  roc_auc=%.3f", metrics["f1"], metrics["roc_auc"])


def task_check_quality(**context) -> None:
    # S17-3 : recuperer f1 depuis XCom et appliquer la porte qualite
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(f"Qualite insuffisante : f1={f1:.3f} < seuil {QUALITY_THRESHOLD}")
    logger.info("Qualite OK : f1=%.3f >= seuil %.3f", f1, QUALITY_THRESHOLD)


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine le modele et controle sa qualite",
    schedule="0 3 * * 1",  # S17-4 : tous les lundis a 3h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    # S17-5 : ordre d'execution
    prepare >> train_task >> check
