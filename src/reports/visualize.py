"""Helper functions for producing common diagnostic plots."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import RegressorMixin


sns.set_theme(style="whitegrid")


def _ensure_axis(ax: Optional[plt.Axes]) -> Tuple[plt.Figure, plt.Axes]:
    """Return a valid figure/axes pair, creating them when necessary."""

    if ax is not None:
        return ax.figure, ax
    fig, axis = plt.subplots()
    return fig, axis


def plot_correlation_matrix(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    *,
    annot: bool = False,
    cmap: str = "coolwarm",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot a correlation matrix for the given dataframe."""

    if columns is not None:
        missing = [col for col in columns if col not in data.columns]
        if missing:
            raise ValueError(f"Columns missing from dataframe: {missing}")
        data = data.loc[:, columns]

    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap=cmap, square=True, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Produce a residual scatter plot with a zero reference line."""

    fig, axis = _ensure_axis(ax)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    axis.scatter(y_pred, residuals, alpha=0.7)
    axis.axhline(0, color="red", linestyle="--", linewidth=1)
    axis.set_xlabel("Predicted Values")
    axis.set_ylabel("Residuals (y_true - y_pred)")
    axis.set_title("Residual Plot")
    fig.tight_layout()
    return fig


def plot_feature_importance(
    model: RegressorMixin,
    feature_names: Sequence[str],
    *,
    top_n: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Visualise feature importance using model coefficients."""

    if not hasattr(model, "coef_"):
        raise AttributeError("The provided model does not expose 'coef_' coefficients")

    coef = np.asarray(model.coef_)
    if coef.ndim > 1:
        coef = coef.ravel()

    if len(coef) != len(feature_names):
        raise ValueError("Number of coefficients does not match feature names")

    importance = pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=False)
    if top_n is not None:
        importance = importance.head(top_n)

    fig, axis = _ensure_axis(ax)
    importance.iloc[::-1].plot(kind="barh", ax=axis, color="steelblue")
    axis.set_xlabel("|Coefficient|")
    axis.set_ylabel("Feature")
    axis.set_title("Feature Importance (OLS Coefficients)")
    fig.tight_layout()
    return fig
