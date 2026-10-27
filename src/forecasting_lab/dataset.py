"""Escalado solo con train y ventanas PyTorch eficientes en memoria."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SeriesStandardizer:
    """Z-score por serie con estadísticos ajustados solo en entrenamiento."""

    def __init__(self) -> None:
        self.mean_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.fit_start_: pd.Timestamp | None = None
        self.fit_end_: pd.Timestamp | None = None

    def fit(self, train: pd.DataFrame) -> "SeriesStandardizer":
        if train.empty or train.isna().any().any():
            raise ValueError("El bloque de entrenamiento debe ser completo")
        self.mean_ = train.mean(axis=0)
        scale = train.std(axis=0, ddof=0)
        self.scale_ = scale.mask(scale.eq(0), 1.0)
        if isinstance(train.index, pd.DatetimeIndex):
            self.fit_start_ = train.index.min()
            self.fit_end_ = train.index.max()
        return self

    def _check_fitted(self) -> tuple[pd.Series, pd.Series]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("El normalizador todavía no fue ajustado")
        return self.mean_, self.scale_

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        mean, scale = self._check_fitted()
        if panel.columns.tolist() != mean.index.tolist():
            raise ValueError("Las series no coinciden con las usadas al ajustar")
        return (panel - mean) / scale

    def inverse_transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        mean, scale = self._check_fitted()
        return panel * scale + mean


@dataclass(frozen=True)
class WindowRef:
    unique_id: str
    target_position: int
    target_start: pd.Timestamp


class PanelWindowDataset(Dataset):
    """Ventanas deslizantes del panel sin prematerializar tensores."""

    def __init__(
        self,
        panel: pd.DataFrame,
        calendar_features: pd.DataFrame,
        lookback: int,
        horizon: int,
        target_start: str | pd.Timestamp | None = None,
        target_end: str | pd.Timestamp | None = None,
        stride: int = 1,
    ) -> None:
        if lookback <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError("L, H y stride deben ser positivos")
        if not panel.index.equals(calendar_features.index):
            raise ValueError("Panel y covariables deben estar alineadas exactamente")
        if panel.isna().any().any() or calendar_features.isna().any().any():
            raise ValueError("Panel y covariables deben estar completos")
        expected = pd.date_range(panel.index.min(), panel.index.max(), freq="h")
        if not panel.index.equals(expected):
            raise ValueError("El panel debe tener frecuencia horaria continua")
        self.panel = panel.astype(np.float32)
        self.calendar_features = calendar_features.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        start = pd.Timestamp(target_start) if target_start is not None else panel.index[lookback]
        end = pd.Timestamp(target_end) if target_end is not None else panel.index[-1]
        self.windows: list[WindowRef] = []
        for unique_id in panel.columns:
            for position in range(lookback, len(panel) - horizon + 1, stride):
                first = panel.index[position]
                last = panel.index[position + horizon - 1]
                if first >= start and last <= end:
                    self.windows.append(WindowRef(str(unique_id), position, first))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        ref = self.windows[index]
        start = ref.target_position - self.lookback
        stop = ref.target_position
        target_stop = stop + self.horizon
        history_y = self.panel[ref.unique_id].iloc[start:stop].to_numpy(copy=True)[:, None]
        history_cov = self.calendar_features.iloc[start:stop].to_numpy(copy=True)
        future_cov = self.calendar_features.iloc[stop:target_stop].to_numpy(copy=True)
        target = self.panel[ref.unique_id].iloc[stop:target_stop].to_numpy(copy=True)
        x = torch.as_tensor(np.concatenate([history_y, history_cov], axis=1), dtype=torch.float32)
        return (
            x,
            torch.as_tensor(future_cov, dtype=torch.float32),
            torch.as_tensor(target, dtype=torch.float32),
            {"unique_id": ref.unique_id, "target_start": ref.target_start},
        )


def count_windows(
    panel: pd.DataFrame,
    lookback: int,
    horizon: int,
    target_start: str | pd.Timestamp,
    target_end: str | pd.Timestamp,
    stride: int = 1,
) -> int:
    """Cuenta ventanas cuyo objetivo queda contenido en el split."""

    expected = pd.date_range(panel.index.min(), panel.index.max(), freq="h")
    if not panel.index.equals(expected):
        raise ValueError("El panel debe tener frecuencia horaria continua")
    start = pd.Timestamp(target_start)
    end = pd.Timestamp(target_end)
    count_per_series = 0
    for position in range(lookback, len(panel) - horizon + 1, stride):
        if panel.index[position] >= start and panel.index[position + horizon - 1] <= end:
            count_per_series += 1
    return count_per_series * panel.shape[1]
