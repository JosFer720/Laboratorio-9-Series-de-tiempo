"""Adaptador NeuralForecast diferido con contrato explícito de formato largo."""

from __future__ import annotations

import importlib

import pandas as pd


LONG_FORMAT_COLUMNS = ("unique_id", "ds", "y")
MODEL_CLASSES = {
    "nhits": "NHiTS", "nbeats": "NBEATS", "tcn": "TCN", "patchtst": "PatchTST",
    "tft": "TFT", "itransformer": "iTransformer", "deepar": "DeepAR",
}


def validate_long_format(data: pd.DataFrame) -> pd.DataFrame:
    """Valida el formato largo obligatorio ``unique_id, ds, y`` de NeuralForecast."""
    missing = [column for column in LONG_FORMAT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"long-format data is missing required columns: {', '.join(missing)}")
    frame = data.copy()
    frame["ds"] = pd.to_datetime(frame["ds"], errors="raise")
    if frame.loc[:, list(LONG_FORMAT_COLUMNS)].isna().any().any():
        raise ValueError("unique_id, ds, and y must not contain missing values")
    return frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def validate_future_format(data: pd.DataFrame) -> pd.DataFrame:
    """Valida ``futr_df`` con identificadores, fechas y covariables, pero sin objetivo."""
    required = ("unique_id", "ds")
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"futr_df no contiene las columnas requeridas: {', '.join(missing)}")
    if "y" in data.columns:
        raise ValueError("futr_df no debe contener y")
    frame = data.copy()
    frame["ds"] = pd.to_datetime(frame["ds"], errors="raise")
    if frame.isna().any().any():
        raise ValueError("futr_df no debe contener valores ausentes")
    return frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)


class NeuralForecastAdapter:
    """Construye y ejecuta modelos NeuralForecast solo si está instalado el paquete opcional.

    ``make_model`` ofrece enlaces para N-HiTS/N-BEATS, TCN, PatchTST, TFT,
    iTransformer y DeepAR. Los datos de entrada deben usar ``unique_id, ds, y``.
    """

    def __init__(self, models, freq: str = "h"):
        self.models, self.freq, self.forecaster_ = list(models), freq, None

    @staticmethod
    def _models_module():
        try:
            return importlib.import_module("neuralforecast.models")
        except ImportError as exc:  # pragma: no cover - depende del entorno local
            raise ImportError("NeuralForecast support requires `pip install neuralforecast`.") from exc

    @classmethod
    def make_model(cls, name: str, **kwargs):
        normalized = name.lower().replace("-", "")
        if normalized not in MODEL_CLASSES:
            raise ValueError(f"unsupported NeuralForecast model '{name}'")
        model_cls = getattr(cls._models_module(), MODEL_CLASSES[normalized])
        return model_cls(**kwargs)

    def fit(self, data: pd.DataFrame) -> "NeuralForecastAdapter":
        frame = validate_long_format(data)
        try:
            forecaster_cls = importlib.import_module("neuralforecast").NeuralForecast
        except ImportError as exc:  # pragma: no cover - depende del entorno local
            raise ImportError("NeuralForecast support requires `pip install neuralforecast`.") from exc
        self.forecaster_ = forecaster_cls(models=self.models, freq=self.freq)
        self.forecaster_.fit(df=frame)
        return self

    def predict(self, future_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.forecaster_ is None:
            raise ValueError("fit must be called before predict")
        if future_df is not None:
            future_df = validate_future_format(future_df)
            return self.forecaster_.predict(futr_df=future_df)
        return self.forecaster_.predict()
