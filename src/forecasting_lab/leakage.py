"""Evidencia ejecutable de las cuatro protecciones contra fugas."""

from __future__ import annotations

import pandas as pd

from .config import ExperimentConfig


def run_leakage_audit(
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    scaler_fit_end: pd.Timestamp,
    early_stopping_end: pd.Timestamp,
    origin: pd.Timestamp,
    history_end: pd.Timestamp,
    future_covariate_end: pd.Timestamp,
    covariates_known_in_advance: bool,
) -> dict[str, bool]:
    """Devuelve protecciones verificables por máquina, no solo una lista textual."""

    values = {
        "temporal_split": train_end < validation_end < origin,
        "train_only_scaling": scaler_fit_end <= train_end,
        "validation_only_early_stopping": early_stopping_end <= validation_end,
        "aligned_known_future_covariates": (
            history_end < origin
            and future_covariate_end >= origin
            and covariates_known_in_advance
        ),
    }
    return {name: bool(value) for name, value in values.items()}


def assert_no_leakage(**audit_inputs: object) -> dict[str, bool]:
    """Ejecuta la auditoría y rechaza el pipeline si falla una protección."""

    report = run_leakage_audit(**audit_inputs)  # type: ignore[arg-type]
    failed = [name for name, passed in report.items() if not passed]
    if failed:
        raise ValueError(f"Auditoría de fugas fallida: {', '.join(failed)}")
    return report


def audit_prepared_pipeline(
    panel: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    scaler: object,
    calendar_features: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, bool]:
    """Audita los objetos reales preparados, no fechas declaradas a mano."""

    expected = pd.date_range(config.data_start, config.test_end, freq="h")
    temporal_ok = (
        panel.index.equals(expected)
        and train.index.min() == pd.Timestamp(config.data_start)
        and train.index.max() == pd.Timestamp(config.train_end)
        and validation.index.min() == pd.Timestamp(config.validation_start)
        and validation.index.max() == pd.Timestamp(config.validation_end)
    )
    scaling_ok = (
        getattr(scaler, "fit_start_", None) == pd.Timestamp(config.data_start)
        and getattr(scaler, "fit_end_", None) == pd.Timestamp(config.train_end)
    )
    covariates_ok = calendar_features.index.equals(panel.index) and "y" not in calendar_features.columns
    report = run_leakage_audit(
        train_end=pd.Timestamp(config.train_end),
        validation_end=pd.Timestamp(config.validation_end),
        scaler_fit_end=getattr(scaler, "fit_end_", pd.NaT),
        early_stopping_end=validation.index.max(),
        origin=pd.Timestamp(config.test_start),
        history_end=validation.index.max(),
        future_covariate_end=calendar_features.index.max(),
        covariates_known_in_advance=covariates_ok,
    )
    report["temporal_split"] = report["temporal_split"] and temporal_ok
    report["train_only_scaling"] = report["train_only_scaling"] and scaling_ok
    failed = [name for name, passed in report.items() if not passed]
    if failed:
        raise ValueError(f"Auditoría de fugas fallida: {', '.join(failed)}")
    return report
