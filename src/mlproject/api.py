"""API d'inference d'un modele de classification (FastAPI).

Seance 12 - TP FastAPI
    /health est fourni et fonctionne. A vous d'implementer le schema d'entree
    (adapte a VOTRE jeu de donnees), le schema de sortie, le chargement du
    modele et l'endpoint /predict (voir les TODO S12-n).
    Lancement : `uvicorn mlproject.api:app --reload`
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.config import MODEL_DIR, MODEL_NAME, ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = joblib.load(MODEL_DIR / "model.joblib")
    logger.info("Modèle chargé")
    try:
        mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow.db")
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME)
        ml["version"] = str(versions[0].version) if versions else "unknown"
    except Exception:
        ml["version"] = os.environ.get("MODEL_VERSION", "unknown")
    yield
    ml.clear()


app = FastAPI(title="Classification API", version="0.1.0", lifespan=lifespan)


class Features(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Age: float = Field(..., ge=0, le=120)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    FamilySize: int = Field(..., ge=1)
    IsAlone: int = Field(..., ge=0, le=1)
    Sex: str
    Embarked: str
    Title: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pclass": 1,
                    "Age": 29.0,
                    "SibSp": 0,
                    "Parch": 0,
                    "Fare": 211.3,
                    "FamilySize": 1,
                    "IsAlone": 1,
                    "Sex": "female",
                    "Embarked": "S",
                    "Title": "Miss",
                }
            ]
        }
    }


class PredictionOut(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict:
    return {"version": ml.get("version", "unknown")}
