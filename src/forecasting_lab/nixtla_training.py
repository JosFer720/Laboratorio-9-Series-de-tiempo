"""Entrenamiento de modelos NeuralForecast con el mismo panel, splits y backtest."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .dataset import SeriesStandardizer
from .features import make_calendar_features
from .neuralforecast_adapter import NeuralForecastAdapter, validate_long_format
from .training import PanelTensors, daily_origins, history_start, TrainingConfig

MODEL_CLASSES = {
    "nhits": "NHITS", "tcn": "TCN", "patchtst": "PatchTST",
    "itransformer": "iTransformer", "deepar": "DeepAR",
}
CALENDAR_COLUMNS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_holiday"]


def long_with_calendar(panel: pd.DataFrame, calendar: bool) -> pd.DataFrame:
    """Formato ``unique_id, ds, y`` más covariables de calendario conocidas a futuro."""

    frame = panel.rename_axis("ds").reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y")
    if calendar:
        features = make_calendar_features(pd.DatetimeIndex(panel.index)).rename_axis("ds").reset_index()
        frame = frame.merge(features, on="ds", how="left")
    return validate_long_format(frame)


def _trajectory_log(model) -> pd.DataFrame:
    """Pérdida de validación por chequeo y pérdida media de entrenamiento entre chequeos."""

    train = pd.DataFrame(model.train_trajectories, columns=["step", "train_loss"])
    valid = pd.DataFrame(model.valid_trajectories, columns=["step", "validation_loss"])
    if valid.empty:
        return valid.assign(train_loss=np.nan)
    bins = np.searchsorted(valid["step"].to_numpy(), train["step"].to_numpy(), side="left")
    means = train.assign(check=bins).groupby("check")["train_loss"].mean()
    valid["train_loss"] = [means.get(index, np.nan) for index in range(len(valid))]
    return valid.astype(float)


@dataclass
class NixtlaRun:
    name: str
    family: str
    forecaster: object
    uses_calendar: bool
    input_size: int
    history_months: int
    log: pd.DataFrame
    mean: pd.Series
    scale: pd.Series
    training_seconds: float
    peak_gpu_memory_mb: float
    multivariate: bool = False

    @property
    def parameters(self) -> tuple[int, int]:
        model = self.forecaster.models[0]
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return int(total), int(trainable)

    def scale_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Aplica el z-score por serie ajustado en entrenamiento a la columna ``y``."""

        clients = frame["unique_id"].astype(str).str.split("|").str[0]
        scaled = frame.copy()
        scaled["y"] = (frame["y"].to_numpy() - self.mean.reindex(clients).to_numpy()) / self.scale.reindex(clients).to_numpy()
        return scaled

    def unscale_forecasts(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Revierte la escala en todas las columnas de pronóstico (puntual y cuantiles)."""

        clients = raw["unique_id"].astype(str).str.split("|").str[0]
        mean, scale = self.mean.reindex(clients).to_numpy(), self.scale.reindex(clients).to_numpy()
        result = raw.copy()
        for column in [c for c in raw.columns if c == self.name or c.startswith(f"{self.name}-")]:
            result[column] = raw[column].to_numpy() * scale + mean
        return result

    def point_column(self, forecasts: pd.DataFrame) -> str:
        """Usa la mediana cuando el modelo es probabilístico."""

        median = f"{self.name}-median"
        return median if median in forecasts.columns else self.name


def fit_nixtla(
    name: str,
    family: str,
    model_name: str,
    panel: pd.DataFrame,
    *,
    uses_calendar: bool,
    input_size: int = 168,
    history_months: int = 30,
    experiment: ExperimentConfig | None = None,
    seed: int = 3092,
    train_end: str | pd.Timestamp | None = None,
    fixed_steps: int | None = None,
    **model_kwargs,
) -> NixtlaRun:
    """Ajusta con train y hace early stopping sobre el bloque de validación (92 días).

    Con ``fixed_steps`` reentrena sin validación hasta ``train_end`` durante ese
    número de pasos, como en el reentrenamiento mensual con ventana expandible.
    """

    experiment = experiment or ExperimentConfig()
    train_end = pd.Timestamp(train_end or experiment.train_end)
    start = history_start(TrainingConfig("dlinear", history_months=history_months), experiment, train_end)
    if fixed_steps is None:
        block = panel.loc[start : experiment.validation_end]
        validation_size = len(panel.loc[experiment.validation_start : experiment.validation_end])
    else:
        block = panel.loc[start:train_end]
        validation_size = 0
        model_kwargs.update(max_steps=fixed_steps, early_stop_patience_steps=-1)
    # El escalado por ventana de NeuralForecast divide por eps en ventanas constantes
    # (clientes con largos tramos en cero); se usa el z-score por serie de train.
    standardizer = SeriesStandardizer().fit(panel.loc[start:train_end])
    kwargs = dict(
        h=24, input_size=input_size, random_seed=seed, alias=name,
        val_check_steps=100, early_stop_patience_steps=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        enable_progress_bar=False, enable_model_summary=False, logger=False,
        scaler_type="identity",
    )
    if uses_calendar:
        kwargs["futr_exog_list"] = CALENDAR_COLUMNS
    kwargs.update(model_kwargs)
    # neuralforecast.models expone NHITS (no NHiTS), así que se resuelve aquí por nombre real.
    model_cls = getattr(NeuralForecastAdapter._models_module(), MODEL_CLASSES[model_name])
    model = model_cls(**kwargs)
    adapter = NeuralForecastAdapter([model], freq="h")
    from neuralforecast import NeuralForecast

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter.forecaster_ = NeuralForecast(models=adapter.models, freq="h")
    adapter.forecaster_.fit(
        df=long_with_calendar(standardizer.transform(block), uses_calendar), val_size=validation_size
    )
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    log = _trajectory_log(adapter.forecaster_.models[0])
    return NixtlaRun(
        name, family, adapter.forecaster_, uses_calendar, input_size, history_months, log,
        standardizer.mean_.rename(index=str), standardizer.scale_.rename(index=str), elapsed, peak,
        multivariate=bool(getattr(model, "MULTIVARIATE", False)),
    )


def _pseudo_series(
    data: PanelTensors, origins: pd.DatetimeIndex, context: int, uses_calendar: bool
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Convierte cada par ``(cliente, origen)`` en una serie corta para predecir en un solo lote."""

    values = data.panel.to_numpy(np.float32)
    frames, futures = [], []
    for origin in origins:
        position = data.position(origin)
        hours = data.index[position - context : position]
        block = values[position - context : position]
        ids = [f"{client}|{origin:%Y-%m-%d}" for client in data.clients]
        frame = pd.DataFrame(
            {
                "unique_id": np.repeat(ids, context),
                "ds": np.tile(hours.to_numpy(), len(ids)),
                "y": block.T.reshape(-1),
            }
        )
        future = pd.DataFrame(
            {
                "unique_id": np.repeat(ids, 24),
                "ds": np.tile(pd.date_range(origin, periods=24, freq="h").to_numpy(), len(ids)),
            }
        )
        if uses_calendar:
            calendar = data.calendar_frame
            frame = frame.join(calendar, on="ds")
            future = future.join(calendar, on="ds")
        frames.append(frame)
        futures.append(future)
    return pd.concat(frames, ignore_index=True), pd.concat(futures, ignore_index=True) if uses_calendar else None


def nixtla_predict(run: NixtlaRun, history: pd.DataFrame, future: pd.DataFrame | None) -> pd.DataFrame:
    """Predice en kWh con ``df`` explícito; el modelo solo ve la historia entregada."""

    kwargs = {"df": run.scale_history(history)}
    if run.uses_calendar:
        kwargs["futr_df"] = future
    raw = run.forecaster.predict(**kwargs)
    raw["unique_id"] = raw["unique_id"].astype(str)
    return run.unscale_forecasts(raw)


def nixtla_rolling_forecasts(
    run: NixtlaRun, data: PanelTensors, origins: pd.DatetimeIndex, context: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pronósticos diarios en validación con el contrato de ``backtest``.

    Devuelve además las columnas crudas de NeuralForecast (cuantiles incluidos).
    """

    context = context or max(run.input_size, 24)
    groups = [origins] if not run.multivariate else [origins[i : i + 1] for i in range(len(origins))]
    raw_parts = []
    for group in groups:
        history, future = _pseudo_series(data, group, context, run.uses_calendar)
        if run.multivariate:
            history["unique_id"] = history["unique_id"].str.split("|").str[0]
            if future is not None:
                future["unique_id"] = future["unique_id"].str.split("|").str[0]
            raw = nixtla_predict(run, history, future)
            raw["unique_id"] = raw["unique_id"].astype(str) + f"|{group[0]:%Y-%m-%d}"
        else:
            raw = nixtla_predict(run, history, future)
        raw_parts.append(raw)
    raw = pd.concat(raw_parts, ignore_index=True)
    parts = raw["unique_id"].astype(str).str.split("|", expand=True)
    raw["unique_id"], raw["origin"] = parts[0], pd.to_datetime(parts[1])
    raw["ds"] = pd.to_datetime(raw["ds"])
    raw = raw.sort_values(["origin", "unique_id", "ds"]).reset_index(drop=True)
    truth = data.panel.rename_axis("ds").reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y")
    raw = raw.merge(truth, on=["unique_id", "ds"], how="left", validate="many_to_one")
    raw["horizon"] = raw.groupby(["origin", "unique_id"]).cumcount() + 1
    column_of = {client: index for index, client in enumerate(data.clients)}
    scales = {origin: data.mase_scale(data.position(origin)) for origin in origins}
    raw["mase_scale"] = [scales[o][column_of[u]] for o, u in zip(raw["origin"], raw["unique_id"])]
    point = run.point_column(raw)
    forecasts = raw.assign(modelo=run.name, y_hat=raw[point])[
        ["unique_id", "ds", "modelo", "y_hat", "y", "origin", "horizon", "mase_scale"]
    ]
    return forecasts, raw


class NixtlaBacktestModel:
    """Adapta un modelo NeuralForecast al protocolo ``ForecastModel`` de ``backtest``.

    Conserva las columnas probabilísticas de cada origen en ``quantiles_``.
    """

    def __init__(self, run: NixtlaRun) -> None:
        self.run = run
        self.name = run.name
        self.quantiles_: list[pd.DataFrame] = []

    def predict(self, history: pd.DataFrame, horizon: int, future_covariates: pd.DataFrame) -> pd.Series:
        context = max(self.run.input_size, 24)
        tail = history.groupby("unique_id", sort=False).tail(context).loc[:, ["unique_id", "ds", "y"]]
        future = None
        if self.run.uses_calendar:
            calendar = make_calendar_features(pd.DatetimeIndex(sorted(set(tail["ds"]) | set(future_covariates.index.get_level_values("ds")))))
            tail = tail.join(calendar, on="ds")
            future = future_covariates.index.to_frame(index=False).join(calendar, on="ds")
        raw = nixtla_predict(self.run, tail.reset_index(drop=True), future)
        raw["ds"] = pd.to_datetime(raw["ds"])
        self.quantiles_.append(raw)
        series = raw.set_index(["unique_id", "ds"])[self.run.point_column(raw)]
        return series.rename("y_hat").reindex(future_covariates.index)
