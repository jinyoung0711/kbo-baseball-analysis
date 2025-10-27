"""Utilities for training and evaluating ordinary least squares (OLS) models.

The helpers in this module centralise the logic for splitting a dataset, running
cross-validation and reporting evaluation metrics.  They are intentionally
written in a composable style so that other modelling scripts can reuse the
individual pieces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


@dataclass
class OLSResult:
    """Container for artefacts that result from training an OLS model."""

    model: LinearRegression
    metrics: Dict[str, float]
    cross_validation_scores: Optional[List[float]] = None


def _select_features(
    data: pd.DataFrame, target_column: str, feature_columns: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the provided dataframe into features and the target series.

    Parameters
    ----------
    data:
        Dataframe containing the modelling data.
    target_column:
        Name of the column to predict.
    feature_columns:
        Optional subset of columns to use as features.  When omitted all
        remaining columns (excluding ``target_column``) are used.
    """

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' missing from dataframe")

    if feature_columns is None:
        feature_columns = [c for c in data.columns if c != target_column]
    else:
        missing = [c for c in feature_columns if c not in data.columns]
        if missing:
            raise ValueError(f"Feature columns missing from dataframe: {missing}")

    X = data.loc[:, feature_columns]
    y = data.loc[:, target_column]
    return X, y


def split_train_test(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[Sequence[str]] = None,
    *,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare a train/test split for an OLS regression model."""

    X, y = _select_features(data, target_column, feature_columns)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit an ordinary least squares regression model."""

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Return common regression metrics for a trained OLS model."""

    predictions = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> List[float]:
    """Execute K-fold cross validation for the dataset and return the scores."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation")

    estimator = LinearRegression()
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2")
    return scores.tolist()


def run_ols_pipeline(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[Sequence[str]] = None,
    *,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    cv_splits: int = 5,
) -> OLSResult:
    """Train an OLS model and return evaluation artefacts."""

    X_train, X_test, y_train, y_test = split_train_test(
        data,
        target_column=target_column,
        feature_columns=feature_columns,
        test_size=test_size,
        random_state=random_state,
    )
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    cross_validation_scores: Optional[List[float]] = None
    if cv_splits and cv_splits > 1:
        X, y = _select_features(data, target_column, feature_columns)
        cross_validation_scores = run_cross_validation(
            X, y, n_splits=cv_splits, random_state=random_state
        )

    return OLSResult(model=model, metrics=metrics, cross_validation_scores=cross_validation_scores)
