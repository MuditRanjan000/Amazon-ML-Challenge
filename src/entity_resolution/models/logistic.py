"""Train-only-preprocessed regularised Logistic Regression for candidate pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Explicit L2 Logistic Regression settings; no calibration is applied."""

    c: float = 1.0
    max_iter: int = 500
    class_weight: str | None = None
    random_state: int = 0


def fit_logistic_regression(
    features: pd.DataFrame, labels: pd.Series, feature_columns: list[str], config: LogisticRegressionConfig | None = None
) -> Pipeline:
    """Fit scaling and Logistic Regression on labelled training pairs only."""

    config = config or LogisticRegressionConfig()
    if labels.nunique() < 2:
        raise ValueError("Logistic Regression requires both positive and negative training pairs.")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=config.c,
                    max_iter=config.max_iter,
                    class_weight=config.class_weight,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    model.fit(features.loc[:, feature_columns], labels)
    return model


def predict_match_probabilities(model: Pipeline, features: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Return Logistic Regression ``predict_proba`` estimates in candidate-row order."""

    probabilities = model.predict_proba(features.loc[:, feature_columns])[:, 1]
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Logistic Regression produced invalid probability estimates.")
    return probabilities
