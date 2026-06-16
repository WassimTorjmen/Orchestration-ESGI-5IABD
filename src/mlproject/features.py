"""Construction du pre-processing."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

_TITLE_RARE = {
    "Lady",
    "Countess",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer",
    "Dona",
}
_TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Génère les features dérivées spécifiques au dataset Titanic.

    Colonnes créées :
    - FamilySize : SibSp + Parch + 1
    - IsAlone    : 1 si FamilySize == 1, sinon 0
    - Title      : titre extrait de Name (Mr, Mrs, Miss, Master, Rare…)
    """

    def fit(self, X: pd.DataFrame, y=None) -> FeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if {"SibSp", "Parch"}.issubset(X.columns):
            X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
            X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        if "Name" in X.columns:
            X["Title"] = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
            X["Title"] = X["Title"].replace(list(_TITLE_RARE), "Rare")
            X["Title"] = X["Title"].replace(_TITLE_MAP)

        return X


def build_preprocessor() -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("feature_engineer", FeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
