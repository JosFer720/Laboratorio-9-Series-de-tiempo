"""Adaptador LightGBM diferido para pronóstico global con rezagos y calendario."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("unique_id", "ds", "y")


def _validate_long_frame(data: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"long-format data is missing required columns: {', '.join(missing)}")
    frame = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["ds"] = pd.to_datetime(frame["ds"], errors="raise")
    if frame[["unique_id", "ds", "y"]].isna().any().any():
        raise ValueError("unique_id, ds, and y must not contain missing values")
    return frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)


class LightGBMGlobalAdapter:
    """Regresor global con rezagos 1--24 y 168, más características de calendario."""

    def __init__(self, lags: tuple[int, ...] | None = None, model_params: dict | None = None, model=None):
        self.lags = tuple(range(1, 25)) + (168,) if lags is None else tuple(lags)
        if (
            any(not isinstance(lag, int) or isinstance(lag, bool) or lag <= 0 for lag in self.lags)
            or len(set(self.lags)) != len(self.lags)
        ):
            raise ValueError("los lags deben ser enteros positivos y únicos")
        if set(range(1, 25)).difference(self.lags) or 168 not in self.lags:
            raise ValueError("lags must include 1 through 24 and 168")
        self.model_params = {} if model_params is None else dict(model_params)
        self.model_ = model
        self.feature_columns_ = [f"lag_{lag}" for lag in self.lags] + ["hour", "dayofweek", "month"]

    @staticmethod
    def _calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["hour"] = result["ds"].dt.hour
        result["dayofweek"] = result["ds"].dt.dayofweek
        result["month"] = result["ds"].dt.month
        return result

    def make_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea características causales: cada rezago en *t* proviene de antes de *t*."""
        frame = _validate_long_frame(data)
        for lag in self.lags:
            frame[f"lag_{lag}"] = frame.groupby("unique_id", sort=False)["y"].shift(lag)
        return self._calendar_features(frame)

    @staticmethod
    def _lightgbm_regressor():
        try:
            module = importlib.import_module("lightgbm")
        except ImportError as exc:  # pragma: no cover - depende del entorno local
            raise ImportError("LightGBM adapter requires `pip install lightgbm`.") from exc
        return module.LGBMRegressor

    def fit(self, data: pd.DataFrame) -> "LightGBMGlobalAdapter":
        featured = self.make_features(data).dropna().reset_index(drop=True)
        if featured.empty:
            raise ValueError("not enough history to construct lags through 168")
        if self.model_ is None:
            regressor = self._lightgbm_regressor()
            self.model_ = regressor(**self.model_params)
        self.model_.fit(featured[self.feature_columns_], featured["y"])
        return self

    def predict(self, history: pd.DataFrame, horizon: int, future_covariates: pd.DataFrame) -> pd.Series:
        """Pronostica sobre el índice futuro ``unique_id/ds`` definido por ``backtest``."""
        if self.model_ is None:
            raise ValueError("fit must be called before predict")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        frame = _validate_long_frame(history)
        if not isinstance(future_covariates, pd.DataFrame) or not isinstance(future_covariates.index, pd.MultiIndex):
            raise ValueError("future_covariates debe tener un MultiIndex unique_id/ds")
        future_index = future_covariates.index
        if future_index.names != ["unique_id", "ds"]:
            raise ValueError("future_covariates debe estar indexado por unique_id y ds")
        future_rows = future_index.to_frame(index=False)
        future_rows["ds"] = pd.to_datetime(future_rows["ds"], errors="raise")
        history_ids = set(frame["unique_id"].unique())
        if set(future_rows["unique_id"].unique()) != history_ids:
            raise ValueError("future_covariates debe contener los mismos unique_id que history")
        counts = future_rows.groupby("unique_id", sort=False).size()
        if not counts.eq(horizon).all():
            raise ValueError("future_covariates debe contener horizon filas por unique_id")
        outputs: dict[tuple[object, pd.Timestamp], float] = {}
        for timestamp in future_rows["ds"].drop_duplicates():
            rows = []
            step_ids = future_rows.loc[future_rows["ds"].eq(timestamp), "unique_id"].tolist()
            for unique_id in step_ids:
                group = frame.loc[frame["unique_id"].eq(unique_id)]
                values = group["y"].to_numpy()
                if len(values) < max(self.lags):
                    raise ValueError("history must contain at least 168 observations per unique_id")
                row = {"unique_id": unique_id, "ds": timestamp, "y": np.nan}
                row.update({f"lag_{lag}": values[-lag] for lag in self.lags})
                rows.append(row)
            step = self._calendar_features(pd.DataFrame(rows))
            predictions = self.model_.predict(step[self.feature_columns_])
            step["y"] = predictions
            frame = pd.concat([frame, step.loc[:, list(REQUIRED_COLUMNS)]], ignore_index=True)
            outputs.update({(row.unique_id, row.ds): float(row.y) for row in step.itertuples(index=False)})
        values = [outputs[(unique_id, pd.Timestamp(ds))] for unique_id, ds in future_index]
        return pd.Series(values, index=future_index, name="y_hat", dtype=float)
