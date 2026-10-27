"""Backtesting de orígenes móviles sin fuga de datos para el laboratorio."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

RESERVED_FORECAST_COLUMNS = {"y", "y_hat", "unique_id", "ds", "origin", "horizon", "modelo"}


@runtime_checkable
class ForecastModel(Protocol):
    """Modelo que predice un bloque futuro de 24 horas desde observaciones pasadas."""

    def predict(
        self, history: pd.DataFrame, horizon: int, future_covariates: pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        """Devuelve predicciones indexadas por unique_id y marca temporal futura."""


def _mase_scale(history: pd.Series, seasonal_period: int) -> float:
    if len(history) <= seasonal_period:
        return float("nan")
    return float(np.mean(np.abs(history.iloc[seasonal_period:].to_numpy() - history.iloc[:-seasonal_period].to_numpy())))


def _validate_hourly_history(history: pd.DataFrame, expected_clients: pd.Index, origin: pd.Timestamp) -> None:
    expected_end = origin - pd.Timedelta(1, unit="h")
    for unique_id in expected_clients:
        timestamps = history.loc[history["unique_id"].eq(unique_id), "ds"]
        differences = timestamps.diff().iloc[1:]
        if (
            timestamps.empty
            or timestamps.duplicated().any()
            or not differences.eq(pd.Timedelta(1, unit="h")).all()
            or timestamps.iloc[-1] != expected_end
        ):
            raise ValueError(f"historia horaria inválida para {unique_id!r} antes de {origin}")


def _prediction_values(
    prediction: pd.Series | pd.DataFrame | Sequence[float], expected_index: pd.MultiIndex
) -> np.ndarray:
    if isinstance(prediction, pd.DataFrame):
        if "y_hat" not in prediction:
            raise ValueError("prediction DataFrame must contain a y_hat column")
        prediction = prediction["y_hat"]
    if not isinstance(prediction, pd.Series) or not isinstance(prediction.index, pd.MultiIndex):
        raise ValueError("model prediction must be a Series indexed by unique_id and ds")
    if prediction.index.nlevels != 2 or not prediction.index.is_unique:
        raise ValueError("model prediction MultiIndex must contain unique unique_id and ds pairs")
    prediction = prediction.reindex(expected_index)
    values = prediction.to_numpy(dtype=float)
    if len(values) != len(expected_index) or not np.isfinite(values).all():
        raise ValueError("model prediction must align to every unique_id and contain only finite values")
    return values


def backtest(
    model: ForecastModel,
    data: pd.DataFrame,
    future_covariate_columns: Sequence[str] = (),
    seasonal_period: int = 24,
) -> pd.DataFrame:
    """Ejecuta 92 pronósticos diarios de 24 horas entre 2014-10-01 y 2014-12-31."""
    forbidden = RESERVED_FORECAST_COLUMNS.intersection(future_covariate_columns)
    if forbidden:
        raise ValueError(f"future_covariate_columns contiene columnas reservadas: {sorted(forbidden)}")
    if seasonal_period < 1:
        raise ValueError("seasonal_period debe ser positivo")
    required = {"unique_id", "ds", "y", *future_covariate_columns}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")

    observations = data.copy()
    # Normaliza la resolución para que índices horarios equivalentes comparen igual.
    observations["ds"] = pd.to_datetime(observations["ds"]).astype("datetime64[ns]")
    observations = observations.sort_values(["unique_id", "ds"])
    origins = pd.date_range("2014-10-01", "2014-12-31", freq="D")
    model_name = str(getattr(model, "name", model.__class__.__name__))
    outputs: list[pd.DataFrame] = []

    for origin in origins:
        history = observations.loc[observations["ds"] < origin].copy()
        _validate_hourly_history(history, observations["unique_id"].unique(), origin)
        future = observations.loc[
            (observations["ds"] >= origin)
            & (observations["ds"] < origin + pd.Timedelta(24, unit="h"))
        ].copy()
        expected_timestamps = pd.date_range(origin, periods=24, freq="h")
        expected_index = pd.MultiIndex.from_product(
            [observations["unique_id"].unique(), expected_timestamps], names=["unique_id", "ds"]
        )
        future_index = pd.MultiIndex.from_frame(future[["unique_id", "ds"]])
        if not future_index.equals(expected_index):
            raise ValueError(f"data lacks a complete 24-hour truth block for {origin.date()}")

        future_covariates = future.loc[:, list(future_covariate_columns)].copy()
        future_covariates.index = future_index
        prediction = model.predict(history, 24, future_covariates)
        values = _prediction_values(prediction, expected_index)
        scale_by_client = {
            unique_id: _mase_scale(client_history["y"], seasonal_period)
            for unique_id, client_history in history.groupby("unique_id", sort=False)
        }
        outputs.append(
            pd.DataFrame(
                {
                    "unique_id": future["unique_id"].to_numpy(),
                    "ds": future["ds"].to_numpy(),
                    "modelo": model_name,
                    "y_hat": values,
                    "y": future["y"].to_numpy(dtype=float),
                    "origin": origin,
                    "horizon": future.groupby("unique_id", sort=False).cumcount().to_numpy() + 1,
                    "mase_scale": future["unique_id"].map(scale_by_client).to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(
        columns=["unique_id", "ds", "modelo", "y_hat", "y", "origin", "horizon", "mase_scale"]
    )
