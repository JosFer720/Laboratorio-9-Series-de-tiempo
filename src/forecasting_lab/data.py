"""Carga, validación, agregación, selección y división temporal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .config import ExperimentConfig

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/321/"
    "electricityloaddiagrams20112014.zip"
)


@dataclass(frozen=True)
class PanelSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def download_uci_data(data_dir: str | Path = "data") -> Path:
    """Descarga y extrae una vez los datos UCI; devuelve la ruta del texto."""

    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    text_path = destination / "LD2011_2014.txt"
    if text_path.exists():
        return text_path
    archive = destination / "electricity.zip"
    if not archive.exists():
        urlretrieve(UCI_URL, archive)
    with ZipFile(archive) as bundle:
        member = next(
            (name for name in bundle.namelist() if name.endswith("LD2011_2014.txt")),
            None,
        )
        if member is None:
            raise ValueError("El ZIP de UCI no contiene LD2011_2014.txt")
        with bundle.open(member) as source, text_path.open("wb") as target:
            target.write(source.read())
    return text_path


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Carga la matriz UCI con separador punto y coma e índice validado."""

    frame = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("El índice crudo debe ser cronológico y sin duplicados")
    return frame.astype(float)


def aggregate_quarter_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Convierte lecturas promedio de 15 minutos en kW a kWh horarios.

    El archivo UCI contiene 96 intervalos civiles todos los días, incluso en
    cambios de horario. Por eso se conservan sus etiquetas canónicas en vez de
    localizar zona horaria, lo que fabricaría intervalos faltantes o repetidos.
    """

    if not isinstance(raw.index, pd.DatetimeIndex):
        raise TypeError("El índice debe ser DatetimeIndex")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("El índice debe ser cronológico y sin duplicados")
    # Las marcas UCI indican el final del cuarto de hora; 01:00 pertenece al
    # consumo de 00:00–01:00. El desplazamiento conserva 24 horas por fecha.
    hour = (raw.index - pd.Timedelta(15, unit="min")).floor("h")
    counts = pd.Series(1, index=raw.index).groupby(hour).sum()
    if not counts.eq(4).all():
        bad = counts[counts.ne(4)].index[0]
        raise ValueError(f"Cada hora requiere cuatro observaciones; falla {bad}")
    expected_hours = pd.date_range(hour.min(), hour.max(), freq="h")
    if not counts.index.equals(expected_hours):
        raise ValueError("La serie horaria resultante debe ser continua")
    # kW promedio de cada intervalo de 15 minutos -> kWh = kW * 0.25 h.
    hourly = raw.mul(0.25).groupby(hour).sum(min_count=4)
    hourly.index.name = "ds"
    return hourly


def filter_complete_clients(
    panel: pd.DataFrame,
    start: str | pd.Timestamp = "2012-01-01",
) -> pd.DataFrame:
    """Conserva clientes observados y ya activos al inicio del experimento."""

    start_ts = pd.Timestamp(start)
    if panel.empty or start_ts not in panel.index:
        raise ValueError("El panel debe contener exactamente el inicio solicitado")
    analysis = panel.loc[start_ts:]
    no_missing = analysis.notna().all(axis=0)
    activation = {}
    for column in panel.columns:
        positive = panel.index[panel[column].fillna(0).gt(0)]
        activation[column] = positive.min() if len(positive) else pd.NaT
    active = pd.Series(activation).le(start_ts)
    keep = panel.columns[no_missing.reindex(panel.columns).to_numpy() & active.reindex(panel.columns).to_numpy()]
    if len(keep) == 0:
        raise ValueError("Ningún cliente tiene datos completos y actividad desde el inicio")
    return panel.loc[start_ts:, keep].copy()


def select_clients(
    panel: pd.DataFrame,
    n_clients: int | None = None,
    seed: int = 3092,
    minimum: int = 100,
) -> pd.DataFrame:
    """Elige un subconjunto determinista compartido por todos los modelos."""

    if n_clients is None or n_clients == panel.shape[1]:
        return panel.copy()
    if n_clients < minimum:
        raise ValueError(f"El subconjunto debe contener al menos {minimum} clientes")
    if n_clients > panel.shape[1]:
        raise ValueError("No se pueden seleccionar más clientes que los disponibles")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(panel.columns.to_numpy(), size=n_clients, replace=False))
    return panel.loc[:, chosen].copy()


def split_panel(panel: pd.DataFrame, config: ExperimentConfig) -> PanelSplits:
    """Crea bloques exactos, ordenados y disjuntos de train/validación/prueba."""

    expected = pd.date_range(
        config.timestamp("data_start"), config.timestamp("test_end"), freq="h"
    )
    experiment = panel.loc[
        config.timestamp("data_start") : config.timestamp("test_end")
    ]
    if not experiment.index.equals(expected):
        raise ValueError("El panel experimental debe ser continuo y cubrir todos los límites")
    train = experiment.loc[config.timestamp("data_start") : config.timestamp("train_end")]
    validation = experiment.loc[
        config.timestamp("validation_start") : config.timestamp("validation_end")
    ]
    test = experiment.loc[config.timestamp("test_start") : config.timestamp("test_end")]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Los tres bloques temporales deben contener observaciones")
    return PanelSplits(train=train, validation=validation, test=test)


def to_long(panel: pd.DataFrame) -> pd.DataFrame:
    """Devuelve datos largos compatibles con Nixtla (`unique_id`, `ds`, `y`)."""

    named = panel.rename_axis("ds").reset_index()
    return named.melt(id_vars="ds", var_name="unique_id", value_name="y")
