"""Held-out evaluation metrics and diagnostics for delay-regression models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Calculate robust regression metrics, omitting R-squared for n < 2."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Evaluation requires equal, non-empty target and prediction arrays")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Evaluation targets and predictions must be finite")
    return {
        "sample_count": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r_squared": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
    }


def grouped_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: pd.Series,
    *,
    minimum_count: int = 5,
) -> list[dict[str, Any]]:
    """Summarize held-out errors for groups with enough observations."""
    frame = pd.DataFrame(
        {
            "actual": np.asarray(actual, dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
            "group": groups.astype("string").fillna("<missing>").to_numpy(),
        }
    )
    rows: list[dict[str, Any]] = []
    for name, group in frame.groupby("group", dropna=False, sort=True):
        if len(group) < minimum_count:
            continue
        metrics = regression_metrics(
            group["actual"].to_numpy(), group["predicted"].to_numpy()
        )
        rows.append({"group": str(name), **metrics})
    return rows


def save_diagnostic_charts(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_dir: str | Path,
) -> list[Path]:
    """Write only diagnostics supported by the held-out sample size."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if len(y_true) != len(y_pred) or not len(y_true):
        return []
    residuals = y_true - y_pred
    frame = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "residual": residuals,
            "absolute_error": np.abs(residuals),
        }
    )
    paths: list[Path] = []
    if len(frame) >= 2:
        figures = {
            "actual_vs_predicted": px.scatter(
                frame,
                x="actual",
                y="predicted",
                title="Held-out actual versus predicted delay",
                labels={
                    "actual": "Actual scheduling delay (unverified source units)",
                    "predicted": "Predicted scheduling delay (unverified source units)",
                },
            ),
            "residual_distribution": px.histogram(
                frame,
                x="residual",
                title="Held-out residual distribution",
                labels={"residual": "Actual minus predicted (source units)"},
            ),
            "absolute_error_distribution": px.histogram(
                frame,
                x="absolute_error",
                title="Held-out absolute error distribution",
                labels={"absolute_error": "Absolute error (source units)"},
            ),
            "error_vs_prediction": px.scatter(
                frame,
                x="predicted",
                y="residual",
                title="Held-out residual versus predicted delay",
                labels={
                    "predicted": "Predicted delay (source units)",
                    "residual": "Residual (source units)",
                },
            ),
        }
        for name, figure in figures.items():
            path = output / f"{name}.html"
            figure.write_html(path, include_plotlyjs="cdn", full_html=True)
            paths.append(path)
    return paths
