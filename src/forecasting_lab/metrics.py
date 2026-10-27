"""Métricas de exactitud y tablas resumen para predicciones de kWh."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

METRIC_COLUMNS = ["mae", "rmse", "wape", "mape", "mase"]


def _values(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _paired(actual: Iterable[float], predicted: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = _values(actual)
    y_hat = _values(predicted)
    if y.shape != y_hat.shape:
        raise ValueError("actual and predicted must have the same length")
    return y, y_hat


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Error absoluto medio, expresado en la unidad original de kWh."""
    y, y_hat = _paired(actual, predicted)
    return float(np.mean(np.abs(y - y_hat))) if len(y) else float("nan")


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Raíz del error cuadrático medio, expresada en la unidad original de kWh."""
    y, y_hat = _paired(actual, predicted)
    return float(np.sqrt(np.mean((y - y_hat) ** 2))) if len(y) else float("nan")


def wape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Error porcentual absoluto ponderado; un total real cero no está definido."""
    y, y_hat = _paired(actual, predicted)
    denominator = float(np.sum(np.abs(y)))
    return float("nan") if denominator == 0 else float(np.sum(np.abs(y - y_hat)) / denominator)


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Error porcentual absoluto medio; cualquier valor real cero lo deja sin definir."""
    y, y_hat = _paired(actual, predicted)
    if not len(y) or np.any(y == 0):
        return float("nan")
    return float(np.mean(np.abs((y - y_hat) / y)))


def _seasonal_scale(history: Iterable[float], seasonal_period: int) -> float:
    values = _values(history)
    if seasonal_period < 1:
        raise ValueError("seasonal_period must be positive")
    if len(values) <= seasonal_period:
        return float("nan")
    return float(np.mean(np.abs(values[seasonal_period:] - values[:-seasonal_period])))


def mase(
    actual: Iterable[float],
    predicted: Iterable[float],
    history: Iterable[float] | None = None,
    seasonal_period: int = 24,
) -> float:
    """MASE estacional que usa la historia previa como escala ingenua estacional."""
    if history is None:
        return float("nan")
    scale = _seasonal_scale(history, seasonal_period)
    return float("nan") if not np.isfinite(scale) or scale == 0 else mae(actual, predicted) / scale


def _summary(frame: pd.DataFrame, history: Iterable[float] | None = None, seasonal_period: int = 24) -> dict[str, float]:
    if not {"y", "y_hat"}.issubset(frame.columns):
        raise ValueError("forecast data must contain y and y_hat columns")
    result = {
        "mae": mae(frame["y"], frame["y_hat"]),
        "rmse": rmse(frame["y"], frame["y_hat"]),
        "wape": wape(frame["y"], frame["y_hat"]),
        "mape": mape(frame["y"], frame["y_hat"]),
    }
    if "mase_scale" in frame:
        scales = pd.to_numeric(frame["mase_scale"], errors="coerce").to_numpy(dtype=float)
        if len(scales) == 0 or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            result["mase"] = float("nan")
        else:
            errors = np.abs(frame["y"].to_numpy(dtype=float) - frame["y_hat"].to_numpy(dtype=float))
            result["mase"] = float(np.mean(errors / scales))
    else:
        result["mase"] = mase(frame["y"], frame["y_hat"], history, seasonal_period)
    return result


def aggregate_metrics(
    forecasts: pd.DataFrame, history: Iterable[float] | None = None, seasonal_period: int = 24
) -> pd.DataFrame:
    """Devuelve una fila con todas las métricas admitidas para una tabla de pronósticos."""
    return pd.DataFrame([_summary(forecasts, history, seasonal_period)], columns=METRIC_COLUMNS)


def _grouped_metrics(
    forecasts: pd.DataFrame,
    columns: str | list[str],
    history_by_client: Mapping[Any, Sequence[float]] | None = None,
    seasonal_period: int = 24,
) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    if not set(columns).issubset(forecasts.columns):
        raise ValueError(f"forecast data must contain grouping columns: {columns}")
    rows: list[dict[str, Any]] = []
    grouped = forecasts.groupby(columns, sort=True, dropna=False)
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(columns, keys))
        history = None
        if history_by_client is not None and columns == ["unique_id"]:
            history = history_by_client.get(keys[0])
        row.update(_summary(group, history, seasonal_period))
        rows.append(row)
    return pd.DataFrame(rows, columns=[*columns, *METRIC_COLUMNS])


def per_client_metrics(
    forecasts: pd.DataFrame,
    history_by_client: Mapping[Any, Sequence[float]] | None = None,
    seasonal_period: int = 24,
) -> pd.DataFrame:
    return _grouped_metrics(forecasts, "unique_id", history_by_client, seasonal_period)


def per_horizon_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    return _grouped_metrics(forecasts, "horizon")


def hour_of_day_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    dated = forecasts.copy()
    dated["hour"] = pd.to_datetime(dated["ds"]).dt.hour
    return _grouped_metrics(dated, "hour")


def christmas_week_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Devuelve métricas diarias para la semana navideña inclusiva de 2014."""
    dated = forecasts.copy()
    dated["ds"] = pd.to_datetime(dated["ds"])
    window = dated.loc[(dated["ds"] >= "2014-12-22") & (dated["ds"] < "2014-12-29")].copy()
    window["ds"] = window["ds"].dt.normalize()
    return _grouped_metrics(window, "ds")
