"""Ejecuta por etapas la preparación y los modelos no neuronales del laboratorio."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from forecasting_lab.config import ExperimentConfig
from forecasting_lab.data import (
    aggregate_quarter_hourly,
    download_uci_data,
    filter_complete_clients,
    load_raw_data,
    select_clients,
    split_panel,
)
from forecasting_lab.dataset import SeriesStandardizer, count_windows
from forecasting_lab.features import make_calendar_features
from forecasting_lab.leakage import audit_prepared_pipeline
from forecasting_lab.metrics import aggregate_metrics, per_horizon_metrics
from forecasting_lab.results import (
    build_comparison_table,
    plot_history_sensitivity,
    plot_horizon_mae,
    plot_pareto,
    write_predictions,
)


def file_sha256(path: str | Path) -> str:
    """Calcula SHA-256 por bloques para no cargar el archivo completo en memoria."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_artifacts(
    raw: pd.DataFrame,
    raw_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    *,
    config: ExperimentConfig | None = None,
    n_clients: int = 100,
) -> dict[str, object]:
    """Prepara el panel, ejecuta la auditoría y guarda metadatos reproducibles."""

    config = config or ExperimentConfig()
    destination = Path(artifacts_dir)
    cache = destination / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    hourly = aggregate_quarter_hourly(raw)
    complete = filter_complete_clients(hourly, config.data_start)
    selected = select_clients(
        complete,
        n_clients=n_clients,
        seed=config.seed,
        minimum=config.minimum_clients,
    )
    splits = split_panel(selected, config)
    scaler = SeriesStandardizer().fit(splits.train)
    features = make_calendar_features(selected.index)
    leakage = audit_prepared_pipeline(
        selected, splits.train, splits.validation, scaler, features, config
    )
    windows = {
        "train": count_windows(
            selected, config.lookback, config.horizon, config.data_start, config.train_end
        ),
        "validation": count_windows(
            selected,
            config.lookback,
            config.horizon,
            config.validation_start,
            config.validation_end,
        ),
        "test": count_windows(
            selected, config.lookback, config.horizon, config.test_start, config.test_end
        ),
    }
    metadata: dict[str, object] = {
        "sha256_datos_crudos": file_sha256(raw_path),
        "forma_cruda": list(raw.shape),
        "forma_horaria": list(hourly.shape),
        "clientes_completos": int(complete.shape[1]),
        "clientes_seleccionados": int(selected.shape[1]),
        "horas_por_serie": int(selected.shape[0]),
        "observaciones_panel": int(selected.size),
        "semilla": config.seed,
        "L": config.lookback,
        "H": config.horizon,
        "ventanas": windows,
        "auditoria_fugas": leakage,
    }
    selected.rename_axis("ds").to_csv(
        cache / "panel_seleccionado.csv.gz",
        compression={"method": "gzip", "mtime": 0},
        date_format="%Y-%m-%d %H:%M:%S",
    )
    pd.Series(selected.columns, name="unique_id").to_csv(
        destination / "clientes_seleccionados.csv", index=False
    )
    (destination / "metadata_preparacion.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def command_prepare(args: argparse.Namespace) -> None:
    """Implementa la etapa de preparación del comando."""

    raw_path = Path(args.raw) if args.raw else download_uci_data(args.data_dir)
    metadata = prepare_artifacts(
        load_raw_data(raw_path), raw_path, args.artifacts, n_clients=args.clients
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def _load_cached_panel(artifacts_dir: str | Path) -> pd.DataFrame:
    """Carga el panel auditado para evitar repetir la preparación de 710 MB."""

    path = Path(artifacts_dir) / "cache" / "panel_seleccionado.csv.gz"
    if not path.exists():
        raise FileNotFoundError("Ejecute primero la etapa prepare")
    return pd.read_csv(path, index_col="ds", parse_dates=["ds"])


def _long_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Transforma el panel ancho al contrato `unique_id, ds, y`."""

    return (
        panel.rename_axis("ds")
        .reset_index()
        .melt(id_vars="ds", var_name="unique_id", value_name="y")
        .sort_values(["unique_id", "ds"], kind="stable")
        .reset_index(drop=True)
    )


def _baseline_backtest(
    panel: pd.DataFrame, name: str, season_length: int | None
) -> pd.DataFrame:
    """Evalúa un baseline de forma vectorizada con el mismo contrato de salida."""

    import numpy as np

    origins = pd.date_range("2014-10-01", "2014-12-31", freq="D")
    outputs: list[pd.DataFrame] = []
    for origin in origins:
        history = panel.loc[: origin - pd.Timedelta(1, unit="h")]
        truth = panel.loc[origin : origin + pd.Timedelta(23, unit="h")]
        if len(truth) != 24:
            raise ValueError(f"Falta verdad completa para {origin.date()}")
        if season_length is None:
            predicted = np.repeat(history.iloc[-1].to_numpy()[:, None], 24, axis=1)
        else:
            if len(history) < season_length:
                raise ValueError("La historia no cubre la estacionalidad solicitada")
            seasonal_start = len(history) - season_length
            predicted = history.iloc[seasonal_start : seasonal_start + 24].to_numpy().T
        seasonal_errors = history.diff(24).abs().iloc[24:]
        scales = seasonal_errors.mean(axis=0).to_numpy(dtype=float)
        expected_index = pd.MultiIndex.from_product(
            [panel.columns, truth.index], names=["unique_id", "ds"]
        )
        outputs.append(
            pd.DataFrame(
                {
                    "unique_id": expected_index.get_level_values("unique_id"),
                    "ds": expected_index.get_level_values("ds"),
                    "modelo": name,
                    "y_hat": predicted.reshape(-1),
                    "y": truth.to_numpy().T.reshape(-1),
                    "origin": origin,
                    "horizon": np.tile(np.arange(1, 25), panel.shape[1]),
                    "mase_scale": np.repeat(scales, 24),
                }
            )
        )
    return pd.concat(outputs, ignore_index=True)


def _lightgbm_backtest(panel: pd.DataFrame, adapter: object) -> pd.DataFrame:
    """Ejecuta recursión LightGBM por matrices anchas sin copiar el panel largo."""

    import numpy as np

    origins = pd.date_range("2014-10-01", "2014-12-31", freq="D")
    outputs: list[pd.DataFrame] = []
    for origin in origins:
        history = panel.loc[: origin - pd.Timedelta(1, unit="h")].copy()
        truth = panel.loc[origin : origin + pd.Timedelta(23, unit="h")]
        predictions: list[np.ndarray] = []
        for timestamp in truth.index:
            features = pd.DataFrame(
                {
                    **{
                        f"lag_{lag}": history.iloc[-lag].to_numpy()
                        for lag in adapter.lags
                    },
                    "hour": timestamp.hour,
                    "dayofweek": timestamp.dayofweek,
                    "month": timestamp.month,
                },
                index=panel.columns,
            )
            step = np.asarray(
                adapter.model_.predict(features[adapter.feature_columns_]), dtype=float
            )
            predictions.append(step)
            history.loc[timestamp] = step
        predicted = np.stack(predictions, axis=0).T
        scales = panel.loc[: origin - pd.Timedelta(1, unit="h")].diff(24).abs().iloc[24:].mean().to_numpy()
        expected_index = pd.MultiIndex.from_product(
            [panel.columns, truth.index], names=["unique_id", "ds"]
        )
        outputs.append(
            pd.DataFrame(
                {
                    "unique_id": expected_index.get_level_values("unique_id"),
                    "ds": expected_index.get_level_values("ds"),
                    "modelo": str(getattr(adapter, "name", "LightGBM")),
                    "y_hat": predicted.reshape(-1),
                    "y": truth.to_numpy().T.reshape(-1),
                    "origin": origin,
                    "horizon": np.tile(np.arange(1, 25), panel.shape[1]),
                    "mase_scale": np.repeat(scales, 24),
                }
            )
        )
    return pd.concat(outputs, ignore_index=True)


def command_baselines(args: argparse.Namespace) -> None:
    """Ejecuta naive y seasonal naive con el backtest único de 92 orígenes."""

    destination = Path(args.artifacts)
    panel = _load_cached_panel(destination)
    specifications = [
        ("Naive", "naive", None),
        ("SeasonalNaive24", "seasonal_naive", 24),
        ("SeasonalNaive168", "seasonal_naive", 168),
    ]
    all_forecasts: list[pd.DataFrame] = []
    result_rows: list[dict[str, object]] = []
    horizon_rows: list[pd.DataFrame] = []
    for name, method, season in specifications:
        started = time.perf_counter()
        forecasts = _baseline_backtest(panel, name, season)
        elapsed = time.perf_counter() - started
        summary = aggregate_metrics(forecasts).iloc[0].to_dict()
        result_rows.append(
            {
                "modelo": name,
                "familia": "Baseline",
                "L": season or 1,
                "estrategia": "directa",
                "usa_covariables": False,
                "total_parameters": 0,
                "trainable_parameters": 0,
                "training_seconds": 1e-9,
                "inference_seconds": elapsed,
                "hardware": "CPU",
                **summary,
            }
        )
        per_horizon = per_horizon_metrics(forecasts)
        per_horizon.insert(0, "modelo", name)
        horizon_rows.append(per_horizon)
        all_forecasts.append(forecasts)
    combined = pd.concat(all_forecasts, ignore_index=True)
    results = build_comparison_table(pd.DataFrame(result_rows))
    write_predictions(combined, destination / "pronosticos_baselines.csv")
    results.to_csv(destination / "resultados_baselines.csv", index=False)
    pd.concat(horizon_rows, ignore_index=True).to_csv(
        destination / "mae_horizonte_baselines.csv", index=False
    )
    best = str(results.iloc[0]["modelo"])
    monthly = []
    selected = combined.loc[combined["modelo"].eq(best)].copy()
    for month, group in selected.groupby(selected["origin"].dt.to_period("M")):
        monthly.append({"mes_reentrenamiento": str(month), "modelo": best, **aggregate_metrics(group).iloc[0].to_dict()})
    pd.DataFrame(monthly).to_csv(destination / "reentrenamiento_mensual_baseline.csv", index=False)
    print(results.to_string(index=False))


def command_lightgbm(args: argparse.Namespace) -> None:
    """Ajusta LightGBM global hasta septiembre y evalúa una sola vez en prueba."""

    from forecasting_lab.backtest import backtest
    from forecasting_lab.lightgbm_adapter import LightGBMGlobalAdapter

    destination = Path(args.artifacts)
    panel = _load_cached_panel(destination)
    long_panel = _long_panel(panel)
    training = long_panel.loc[long_panel["ds"].le(pd.Timestamp("2014-09-30 23:00"))]
    model = LightGBMGlobalAdapter(
        model_params={"n_estimators": args.estimators, "random_state": 3092, "n_jobs": -1}
    )
    started = time.perf_counter()
    model.fit(training)
    training_seconds = time.perf_counter() - started
    model.name = "LightGBM"
    started = time.perf_counter()
    forecasts = _lightgbm_backtest(panel, model)
    inference_seconds = time.perf_counter() - started
    write_predictions(forecasts, destination / "pronosticos_lightgbm.csv")
    row = {
        "modelo": "LightGBM",
        "familia": "Árboles",
        "L": 168,
        "estrategia": "autorregresiva",
        "usa_covariables": True,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "hardware": "CPU",
        **aggregate_metrics(forecasts).iloc[0].to_dict(),
    }
    pd.DataFrame([row]).to_csv(destination / "resultados_lightgbm.csv", index=False)
    print(pd.Series(row).to_string())


def command_statsforecast(args: argparse.Namespace) -> None:
    """Ejecuta los baselines oficiales de StatsForecast con 92 ventanas."""

    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, Naive, SeasonalNaive

    destination = Path(args.artifacts)
    panel = _load_cached_panel(destination)
    long_panel = _long_panel(panel)
    models = [
        Naive(alias="NaiveSF"),
        SeasonalNaive(24, alias="SeasonalNaive24SF"),
        SeasonalNaive(168, alias="SeasonalNaive168SF"),
        AutoETS(season_length=24, alias="AutoETS24"),
    ]
    engine = StatsForecast(models=models, freq="h", n_jobs=-1)
    started = time.perf_counter()
    cross_validation = engine.cross_validation(
        h=24,
        df=long_panel,
        n_windows=92,
        step_size=24,
        refit=False,
    )
    elapsed = time.perf_counter() - started
    all_forecasts: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for model in ("NaiveSF", "SeasonalNaive24SF", "SeasonalNaive168SF", "AutoETS24"):
        forecast = cross_validation.loc[:, ["unique_id", "ds", "cutoff", "y", model]].rename(
            columns={model: "y_hat", "cutoff": "history_end"}
        )
        # La unidad explícita evita la conversión ambigua de enteros de NumPy.
        forecast["origin"] = pd.to_datetime(forecast["history_end"]) + pd.Timedelta(1, unit="h")
        forecast["modelo"] = model
        forecast["horizon"] = forecast.groupby(
            ["unique_id", "origin"], sort=False
        ).cumcount() + 1
        scales: dict[tuple[object, pd.Timestamp], float] = {}
        for cutoff in pd.to_datetime(forecast["history_end"]).unique():
            history = panel.loc[:cutoff]
            values = history.diff(24).abs().iloc[24:].mean()
            scales.update({(uid, pd.Timestamp(cutoff)): float(value) for uid, value in values.items()})
        forecast["mase_scale"] = [
            scales[(uid, pd.Timestamp(cutoff))]
            for uid, cutoff in zip(forecast["unique_id"], forecast["history_end"])
        ]
        summary = aggregate_metrics(forecast).iloc[0].to_dict()
        rows.append(
            {
                "modelo": model,
                "familia": "StatsForecast",
                "L": 168 if "168" in model else 24,
                "estrategia": "directa",
                "usa_covariables": False,
                "training_seconds": elapsed,
                "inference_seconds": elapsed,
                "hardware": "CPU",
                **summary,
            }
        )
        all_forecasts.append(forecast)
    combined = pd.concat(all_forecasts, ignore_index=True)
    write_predictions(combined, destination / "pronosticos_statsforecast.csv")
    build_comparison_table(pd.DataFrame(rows)).to_csv(
        destination / "resultados_statsforecast.csv", index=False
    )
    print(build_comparison_table(pd.DataFrame(rows)).to_string(index=False))


def command_chronos(args: argparse.Namespace) -> None:
    """Ejecuta Chronos-Bolt zero-shot por lotes y conserva la mediana P50."""

    import numpy as np

    from forecasting_lab.backtest import backtest
    from forecasting_lab.chronos_adapter import ChronosBoltAdapter

    class ChronosBacktestModel:
        name = "Chronos-Bolt"

        def __init__(self) -> None:
            self.adapter = ChronosBoltAdapter(model_id=args.model_id)

        def predict(self, history, horizon, future_covariates):
            ids = future_covariates.index.get_level_values("unique_id").unique()
            contexts = np.stack(
                [
                    history.loc[history["unique_id"].eq(uid)].sort_values("ds")["y"].tail(args.context_length).to_numpy()
                    for uid in ids
                ]
            )
            result = self.adapter.predict(contexts, horizon)
            quantiles = result[0] if isinstance(result, tuple) else result
            if hasattr(quantiles, "detach"):
                quantiles = quantiles.detach().cpu().numpy()
            values = np.asarray(quantiles, dtype=float)[:, :, 1].reshape(-1)
            return pd.Series(values, index=future_covariates.index, name="y_hat")

    destination = Path(args.artifacts)
    panel = _load_cached_panel(destination)
    long_panel = _long_panel(panel)
    model = ChronosBacktestModel()
    started = time.perf_counter()
    forecasts = backtest(model, long_panel)
    inference_seconds = time.perf_counter() - started
    write_predictions(forecasts, destination / "pronosticos_chronos.csv")
    device = next(model.adapter.pipeline.model.parameters()).device.type.upper()
    row = {
        "modelo": "Chronos-Bolt",
        "familia": "Fundacional",
        "L": args.context_length,
        "estrategia": "zero-shot",
        "usa_covariables": False,
        "training_seconds": 1e-9,
        "inference_seconds": inference_seconds,
        "hardware": device,
        **aggregate_metrics(forecasts).iloc[0].to_dict(),
    }
    pd.DataFrame([row]).to_csv(destination / "resultados_chronos.csv", index=False)
    print(pd.Series(row).to_string())


def consolidate_artifacts(
    artifacts_dir: str | Path = "artifacts",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une resultados y pronósticos disponibles sin duplicar baselines equivalentes."""

    destination = Path(artifacts_dir)
    duplicate_models = {"NaiveSF", "SeasonalNaive24SF", "SeasonalNaive168SF"}
    result_paths = [
        destination / name
        for name in (
            "resultados_baselines.csv",
            "resultados_statsforecast.csv",
            "resultados_lightgbm.csv",
            "resultados_chronos.csv",
        )
        if (destination / name).exists()
    ]
    prediction_paths = [
        destination / name
        for name in (
            "pronosticos_baselines.csv",
            "pronosticos_statsforecast.csv",
            "pronosticos_lightgbm.csv",
            "pronosticos_chronos.csv",
        )
        if (destination / name).exists()
    ]
    if not result_paths or not prediction_paths:
        raise FileNotFoundError("No hay resultados y pronósticos suficientes para consolidar")

    results = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    results = results.loc[~results["modelo"].isin(duplicate_models)].copy()
    results = build_comparison_table(results)
    results.to_csv(destination / "resultados_modelos.csv", index=False)

    forecasts = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)
    forecasts = forecasts.loc[~forecasts["modelo"].isin(duplicate_models)].copy()
    forecasts["ds"] = pd.to_datetime(forecasts["ds"])
    forecasts = forecasts.sort_values(["unique_id", "ds", "modelo"], kind="stable").reset_index(drop=True)
    write_predictions(forecasts, destination / "pronosticos_modelos.csv")
    return results, forecasts


def _per_horizon_from_forecasts(
    forecasts: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """Reconstruye MAE por paso al unir pronósticos con el panel observado."""

    truth = (
        panel.rename_axis("ds")
        .reset_index()
        .melt(id_vars="ds", var_name="unique_id", value_name="y")
    )
    truth["ds"] = pd.to_datetime(truth["ds"])
    values = forecasts.copy()
    values["ds"] = pd.to_datetime(values["ds"])
    joined = values.merge(truth, on=["unique_id", "ds"], how="left", validate="many_to_one")
    if joined["y"].isna().any():
        raise ValueError("Hay pronósticos sin observación real correspondiente")
    joined["horizon"] = joined["ds"].dt.hour + 1
    joined["error_absoluto"] = (joined["y"] - joined["y_hat"]).abs()
    return (
        joined.groupby(["modelo", "horizon"], sort=True)["error_absoluto"]
        .mean()
        .rename("mae")
        .reset_index()
    )


def build_available_result_figures(
    artifacts_dir: str | Path,
    panel: pd.DataFrame,
    results: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> None:
    """Genera las figuras cuyo insumo real ya está disponible."""

    destination = Path(artifacts_dir)
    plot_pareto(results, destination / "pareto_mase_tiempo.png")
    horizon = _per_horizon_from_forecasts(forecasts, panel)
    horizon.to_csv(destination / "mae_horizonte_modelos.csv", index=False)
    plot_horizon_mae(horizon, destination / "mae_por_horizonte.png")
    sensitivity_path = destination / "sensibilidad_historia.csv"
    if sensitivity_path.exists():
        plot_history_sensitivity(
            pd.read_csv(sensitivity_path), destination / "sensibilidad_historia.png"
        )


def command_report(_: argparse.Namespace) -> None:
    """Regenera notebook y reporte con los artefactos disponibles."""

    try:
        from scripts.build_notebook import build_notebook
        from scripts.build_report import build_report
    except ModuleNotFoundError as exc:
        # Al ejecutar el archivo directamente, Python agrega `scripts` y no su padre.
        if not str(exc.name).startswith("scripts"):
            raise
        from build_notebook import build_notebook
        from build_report import build_report

    results, forecasts = consolidate_artifacts("artifacts")
    build_available_result_figures(
        "artifacts", _load_cached_panel("artifacts"), results, forecasts
    )
    build_notebook()
    build_report("report/reporte.md", "report/reporte_laboratorio.pdf")


def build_parser() -> argparse.ArgumentParser:
    """Crea la interfaz por etapas para poder reanudar corridas largas."""

    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="stage", required=True)
    prepare = subcommands.add_parser("prepare", help="prepara y audita el panel")
    prepare.add_argument("--raw", help="ruta a LD2011_2014.txt; se descarga si se omite")
    prepare.add_argument("--data-dir", default="data")
    prepare.add_argument("--artifacts", default="artifacts")
    prepare.add_argument("--clients", type=int, default=100)
    prepare.set_defaults(handler=command_prepare)
    baselines = subcommands.add_parser("baselines", help="ejecuta baselines deterministas")
    baselines.add_argument("--artifacts", default="artifacts")
    baselines.set_defaults(handler=command_baselines)
    statsforecast = subcommands.add_parser(
        "statsforecast", help="ejecuta Naive, SeasonalNaive y AutoETS oficiales"
    )
    statsforecast.add_argument("--artifacts", default="artifacts")
    statsforecast.set_defaults(handler=command_statsforecast)
    lightgbm = subcommands.add_parser("lightgbm", help="ejecuta LightGBM global")
    lightgbm.add_argument("--artifacts", default="artifacts")
    lightgbm.add_argument("--estimators", type=int, default=300)
    lightgbm.set_defaults(handler=command_lightgbm)
    chronos = subcommands.add_parser("chronos", help="ejecuta Chronos-Bolt zero-shot")
    chronos.add_argument("--artifacts", default="artifacts")
    chronos.add_argument("--model-id", default="amazon/chronos-bolt-small")
    chronos.add_argument("--context-length", type=int, default=2048)
    chronos.set_defaults(handler=command_chronos)
    report = subcommands.add_parser("report", help="regenera notebook y PDF")
    report.set_defaults(handler=command_report)
    return parser


def main() -> None:
    """Punto de entrada del ejecutor por etapas."""

    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
