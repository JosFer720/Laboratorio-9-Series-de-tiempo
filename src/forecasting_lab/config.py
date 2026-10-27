"""Fechas centrales del experimento y opciones de reproducibilidad."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    """Definición inmutable del experimento compartido de forecasting."""

    data_start: str = "2012-01-01 00:00"
    train_end: str = "2014-06-30 23:00"
    validation_start: str = "2014-07-01 00:00"
    validation_end: str = "2014-09-30 23:00"
    test_start: str = "2014-10-01 00:00"
    test_end: str = "2014-12-31 23:00"
    lookback: int = 168
    horizon: int = 24
    seed: int = 3092
    minimum_clients: int = 100

    def timestamp(self, field: str) -> pd.Timestamp:
        """Devuelve un límite configurado como marca de tiempo de pandas."""

        return pd.Timestamp(getattr(self, field))
