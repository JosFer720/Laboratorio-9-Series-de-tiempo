"""Covariables de calendario conocidas por completo al pronosticar."""

from __future__ import annotations

from datetime import date, timedelta
from math import pi
from typing import Iterable

import numpy as np
import pandas as pd


def _easter_sunday(year: int) -> date:
    """Cómputo gregoriano de Pascua (Meeus/Jones/Butcher)."""

    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def portugal_holidays(years: Iterable[int]) -> pd.DatetimeIndex:
    """Feriados nacionales portugueses, incluida la suspensión de 2013–2015."""

    days: set[date] = set()
    for year in years:
        easter = _easter_sunday(year)
        days.update(
            {
                date(year, 1, 1),
                easter - timedelta(days=2),  # Viernes Santo
                easter,
                date(year, 4, 25),
                date(year, 5, 1),
                date(year, 6, 10),
                date(year, 8, 15),
                date(year, 12, 8),
                date(year, 12, 25),
            }
        )
        # Corpus Christi, República, Todos los Santos y Restauración de la
        # Independencia se suspendieron a nivel nacional entre 2013 y 2015.
        if year <= 2012 or year >= 2016:
            days.update(
                {
                    easter + timedelta(days=60),
                    date(year, 10, 5),
                    date(year, 11, 1),
                    date(year, 12, 1),
                }
            )
    return pd.DatetimeIndex(sorted(pd.Timestamp(day) for day in days))


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Codifica hora, día y mes cíclicamente, más un indicador de feriado."""

    index = pd.DatetimeIndex(index)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Las fechas de covariables deben estar ordenadas y ser únicas")
    holidays = portugal_holidays(sorted(set(index.year)))
    normalized = index.normalize()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * pi * index.hour / 24),
            "hour_cos": np.cos(2 * pi * index.hour / 24),
            "dow_sin": np.sin(2 * pi * index.dayofweek / 7),
            "dow_cos": np.cos(2 * pi * index.dayofweek / 7),
            "month_sin": np.sin(2 * pi * (index.month - 1) / 12),
            "month_cos": np.cos(2 * pi * (index.month - 1) / 12),
            "is_holiday": normalized.isin(holidays).astype(float),
        },
        index=index,
        dtype=float,
    )
