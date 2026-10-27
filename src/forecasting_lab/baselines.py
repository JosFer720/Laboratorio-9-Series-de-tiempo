"""Pronosticadores deterministas de referencia y un punto de integración diferida."""

from __future__ import annotations

import importlib
from typing import Iterable

import numpy as np
import pandas as pd


def _history_array(history: Iterable[float]) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("history must be a non-empty one-dimensional sequence")
    return values


def naive_forecast(history: Iterable[float], horizon: int) -> np.ndarray:
    """Repite el último valor observado en cada paso futuro."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return np.full(horizon, _history_array(history)[-1], dtype=float)


def seasonal_naive_forecast(history: Iterable[float], horizon: int, season_length: int) -> np.ndarray:
    """Repite el ciclo estacional más reciente, por ejemplo de 24 o 168 pasos."""
    values = _history_array(history)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if season_length <= 0 or values.size < season_length:
        raise ValueError("history must contain at least season_length observations")
    return np.resize(values[-season_length:], horizon).astype(float, copy=False)


class BaselineForecaster:
    """Protocolo mínimo ``fit``/``predict`` para ciclos de comparación."""

    def __init__(self, method: str = "naive", season_length: int | None = None):
        if method not in {"naive", "seasonal_naive"}:
            raise ValueError("method must be 'naive' or 'seasonal_naive'")
        if method == "seasonal_naive" and season_length not in {24, 168} and (season_length is None or season_length <= 0):
            raise ValueError("seasonal_naive requires a positive season_length (typically 24 or 168)")
        self.method, self.season_length = method, season_length
        self.history_: np.ndarray | None = None

    def fit(self, history: Iterable[float]) -> "BaselineForecaster":
        self.history_ = _history_array(history)
        return self

    def _direct_predict(self, values: Iterable[float], horizon: int) -> np.ndarray:
        if self.method == "naive":
            return naive_forecast(values, horizon)
        return seasonal_naive_forecast(values, horizon, int(self.season_length))

    def predict(
        self,
        history: Iterable[float] | pd.DataFrame | None = None,
        horizon: int = 1,
        future_covariates: pd.DataFrame | None = None,
    ) -> np.ndarray | pd.Series:
        """Pronostica arreglos directos o el protocolo tabular usado por ``backtest``."""
        if isinstance(history, pd.DataFrame):
            required = {"unique_id", "ds", "y"}
            missing = required.difference(history.columns)
            if missing:
                raise ValueError(f"history no contiene las columnas requeridas: {sorted(missing)}")
            if future_covariates is None or not isinstance(future_covariates.index, pd.MultiIndex):
                raise ValueError("future_covariates debe tener un MultiIndex unique_id/ds")
            future_index = future_covariates.index
            if future_index.names != ["unique_id", "ds"]:
                raise ValueError("future_covariates debe estar indexado por unique_id y ds")
            predictions: dict[object, np.ndarray] = {}
            for unique_id in future_index.get_level_values("unique_id").unique():
                values = history.loc[history["unique_id"].eq(unique_id)].sort_values("ds")["y"]
                if len(future_index[future_index.get_level_values("unique_id") == unique_id]) != horizon:
                    raise ValueError("future_covariates debe contener horizon filas por unique_id")
                predictions[unique_id] = self._direct_predict(values.to_numpy(), horizon)
            offsets = {unique_id: 0 for unique_id in predictions}
            output = []
            for unique_id in future_index.get_level_values("unique_id"):
                output.append(predictions[unique_id][offsets[unique_id]])
                offsets[unique_id] += 1
            return pd.Series(output, index=future_index, name="y_hat", dtype=float)

        values = self.history_ if history is None else _history_array(history)
        if values is None:
            raise ValueError("fit must be called or history must be supplied")
        return self._direct_predict(values, horizon)


class StatsForecastMSTLAutoETSForecaster:
    """Punto opcional para un modelo MSTL + AutoETS de StatsForecast.

    Importar este módulo es ligero; ``statsforecast`` solo se instala cuando el
    experimento selecciona esta integración.
    """

    def __init__(self, season_length: int | list[int] = 24):
        self.season_length = season_length

    def build_models(self):
        try:
            models = importlib.import_module("statsforecast.models")
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError("StatsForecast support requires `pip install statsforecast`.") from exc
        return [models.MSTL(season_length=self.season_length, trend_forecaster=models.AutoETS())]
