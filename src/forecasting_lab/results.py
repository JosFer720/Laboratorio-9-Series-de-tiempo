"""Contratos de artefactos y visualizaciones comparables entre modelos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PREDICTION_COLUMNS = ["unique_id", "ds", "modelo", "y_hat"]


def _output_path(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def write_predictions(forecasts: pd.DataFrame, path: str | Path) -> Path:
    """Escribe pronósticos con el esquema exacto solicitado por la guía."""

    missing = set(PREDICTION_COLUMNS).difference(forecasts.columns)
    if missing:
        raise ValueError(f"Faltan columnas de pronóstico: {sorted(missing)}")
    output = forecasts.loc[:, PREDICTION_COLUMNS].copy()
    output["ds"] = pd.to_datetime(output["ds"])
    output = output.sort_values(["unique_id", "ds", "modelo"], kind="stable")
    destination = _output_path(path)
    output.to_csv(destination, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return destination


def build_comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Ordena las mejores configuraciones por MASE de prueba."""

    if "mase" not in runs.columns or "modelo" not in runs.columns:
        raise ValueError("La tabla de corridas requiere modelo y mase")
    return runs.sort_values(["mase", "modelo"], kind="stable").reset_index(drop=True)


def _save_figure(fig: plt.Figure, path: str | Path) -> Path:
    destination = _output_path(path)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_pareto(runs: pd.DataFrame, path: str | Path) -> Path:
    """Grafica MASE frente a tiempo de entrenamiento en escala logarítmica."""

    required = {"modelo", "mase", "training_seconds"}
    if missing := required.difference(runs.columns):
        raise ValueError(f"Faltan columnas para Pareto: {sorted(missing)}")
    if runs["training_seconds"].le(0).any():
        raise ValueError("Los tiempos de entrenamiento deben ser positivos")
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(runs["training_seconds"], runs["mase"], color="#176B87")
    for row in runs.itertuples(index=False):
        axis.annotate(str(row.modelo), (row.training_seconds, row.mase), xytext=(4, 4), textcoords="offset points")
    axis.set_xscale("log")
    axis.set_xlabel("Tiempo de entrenamiento (s, escala log)")
    axis.set_ylabel("MASE de prueba")
    axis.set_title("Costo frente a precisión")
    axis.grid(alpha=0.25)
    return _save_figure(fig, path)


def plot_horizon_mae(rows: pd.DataFrame, path: str | Path) -> Path:
    """Grafica una curva de MAE para cada modelo a lo largo del horizonte."""

    required = {"modelo", "horizon", "mae"}
    if missing := required.difference(rows.columns):
        raise ValueError(f"Faltan columnas de horizonte: {sorted(missing)}")
    fig, axis = plt.subplots(figsize=(7, 4))
    for model, group in rows.groupby("modelo", sort=True):
        ordered = group.sort_values("horizon")
        axis.plot(ordered["horizon"], ordered["mae"], marker="o", label=model)
    axis.set_xlabel("Paso del horizonte (h)")
    axis.set_ylabel("MAE (kWh)")
    axis.set_title("Error por paso del horizonte")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    return _save_figure(fig, path)


def plot_history_sensitivity(rows: pd.DataFrame, path: str | Path) -> Path:
    """Grafica MASE frente a meses de historia para cada modelo."""

    required = {"modelo", "history_months", "mase"}
    if missing := required.difference(rows.columns):
        raise ValueError(f"Faltan columnas de historia: {sorted(missing)}")
    fig, axis = plt.subplots(figsize=(7, 4))
    for model, group in rows.groupby("modelo", sort=True):
        ordered = group.sort_values("history_months")
        axis.plot(ordered["history_months"], ordered["mase"], marker="o", label=model)
    axis.set_xlabel("Meses de historia")
    axis.set_ylabel("MASE")
    axis.set_title("Sensibilidad a la historia de entrenamiento")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    return _save_figure(fig, path)

