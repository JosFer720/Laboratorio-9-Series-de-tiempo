"""Ejecuta por etapas el entrenamiento de las redes, sus iteraciones y la evaluación en prueba."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forecasting_lab.backtest import backtest
from forecasting_lab.config import ExperimentConfig
from forecasting_lab.metrics import (
    aggregate_metrics,
    christmas_week_metrics,
    hour_of_day_metrics,
    per_client_metrics,
    per_horizon_metrics,
)
from forecasting_lab.results import build_comparison_table, write_predictions
from forecasting_lab.training import (
    FAMILY,
    ITERATION_COLUMNS,
    STRATEGY,
    USES_COVARIATES,
    PanelTensors,
    TorchBacktestModel,
    TrainingConfig,
    build_model,
    daily_origins,
    forecast_errors,
    predict_positions,
    rolling_forecasts,
    train_model,
    Normalizer,
)

CONFIG = ExperimentConfig()
VALIDATION_ORIGINS = daily_origins(CONFIG.validation_start, CONFIG.validation_end)
TEST_ORIGINS = daily_origins(CONFIG.test_start, CONFIG.test_end)
MODEL_DIR = "cache/modelos"


# ---------------------------------------------------------------- utilidades


def load_panel(artifacts: Path) -> pd.DataFrame:
    path = artifacts / "cache" / "panel_seleccionado.csv.gz"
    if not path.exists():
        raise FileNotFoundError("Falta el panel auditado: ejecute primero la etapa prepare del pipeline de datos")
    return pd.read_csv(path, index_col="ds", parse_dates=["ds"])


def long_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.rename_axis("ds").reset_index()
        .melt(id_vars="ds", var_name="unique_id", value_name="y")
        .sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)
    )


def hardware() -> str:
    return f"GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


@contextmanager
def file_lock(path: Path, timeout: float = 600.0):
    """Candado por archivo para que etapas en paralelo no pisen el mismo CSV."""

    lock = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"candado ocupado: {lock}")
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(handle)
        lock.unlink(missing_ok=True)


def append_csv(frame: pd.DataFrame, path: Path, key: list[str]) -> pd.DataFrame:
    """Agrega filas reemplazando las corridas previas con la misma llave."""

    with file_lock(path):
        if path.exists():
            previous = pd.read_csv(path)
            merged = previous.merge(frame[key].drop_duplicates(), on=key, how="left", indicator=True)
            previous = previous.loc[merged["_merge"].eq("left_only").to_numpy()]
            frame = pd.concat([previous, frame], ignore_index=True)
        frame.to_csv(path, index=False)
    return frame


def save_run(run, artifacts: Path, run_id: str) -> None:
    destination = artifacts / MODEL_DIR
    destination.mkdir(parents=True, exist_ok=True)
    config = asdict(run.config)
    torch.save(
        {
            "config": config,
            "state_dict": run.model.state_dict(),
            "best_epoch": run.best_epoch,
            "training_seconds": run.training_seconds,
            "peak_gpu_memory_mb": run.peak_gpu_memory_mb,
        },
        destination / f"{run_id}.pt",
    )


def load_run(artifacts: Path, run_id: str, data: PanelTensors):
    """Reconstruye una corrida guardada sin volver a entrenar."""

    from forecasting_lab.training import TrainingRun

    payload = torch.load(artifacts / MODEL_DIR / f"{run_id}.pt", map_location=data.device, weights_only=False)
    config = payload["config"]
    if config.get("clients") is not None:
        config["clients"] = tuple(config["clients"])
    config = TrainingConfig(**config)
    model = build_model(config).to(data.device)
    model.load_state_dict(payload["state_dict"])
    subset = data.subset(list(config.clients)) if config.clients else data
    start = max(
        pd.Timestamp(CONFIG.train_end) + pd.Timedelta(1, unit="h") - pd.DateOffset(months=config.history_months),
        pd.Timestamp(CONFIG.data_start),
    )
    normalizer = Normalizer(config.normalization).fit(subset.panel.loc[start : CONFIG.train_end], subset.device)
    return TrainingRun(
        config, model, normalizer, subset, pd.DataFrame(), payload["best_epoch"], None,
        payload["training_seconds"], payload["peak_gpu_memory_mb"],
    )


def run_id_of(config: TrainingConfig) -> str:
    return slug(f"{config.label}_{config.describe()}_seed{config.seed}").replace("=", "").replace(" ", "_")


def test_metrics(forecasts: pd.DataFrame) -> dict[str, float]:
    """Métricas de prueba, incluida la semana del 22 al 28 de diciembre."""

    summary = aggregate_metrics(forecasts).iloc[0].to_dict()
    ds = pd.to_datetime(forecasts["ds"])
    christmas = forecasts.loc[(ds >= "2014-12-22") & (ds < "2014-12-29")]
    summary["mase_navidad"] = float(aggregate_metrics(christmas).iloc[0]["mase"])
    summary["mase_h168"] = float("nan")
    return summary


def save_test_outputs(forecasts: pd.DataFrame, artifacts: Path, row: dict[str, object]) -> None:
    """Guarda pronósticos, fila de resultados y desgloses de error de un modelo."""

    name = str(row["modelo"])
    write_predictions(forecasts, artifacts / f"pronosticos_{slug(name)}.csv")
    append_csv(pd.DataFrame([row]), artifacts / "resultados_redes.csv", ["modelo"])
    detail = artifacts / "detalle_redes"
    detail.mkdir(exist_ok=True)
    for label, frame in {
        "horizonte": per_horizon_metrics(forecasts),
        "hora": hour_of_day_metrics(forecasts),
        "cliente": per_client_metrics(forecasts),
        "navidad": christmas_week_metrics(forecasts),
    }.items():
        frame.insert(0, "modelo", name)
        if "ds" in frame:
            # Mismo formato de fecha para todos los modelos del CSV acumulado.
            frame["ds"] = pd.to_datetime(frame["ds"]).dt.strftime("%Y-%m-%d")
        append_csv(frame, detail / f"metricas_{label}.csv", ["modelo"])
    print(pd.Series(row).to_string(), flush=True)


def evaluate_on_test(model, panel: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Única evaluación de prueba: la función ``backtest`` compartida con los 92 orígenes."""

    started = time.perf_counter()
    forecasts = backtest(model, long_panel(panel))
    return forecasts, time.perf_counter() - started


def fast_test(run, name: str, origins: pd.DatetimeIndex = TEST_ORIGINS) -> pd.DataFrame:
    """Evaluador vectorizado equivalente a ``backtest`` (ver tests/test_training.py).

    Se usa en los análisis repetidos (semillas, local/global, mensual) para no
    pagar once minutos de ``backtest`` por corrida.
    """

    return rolling_forecasts(
        lambda p, c: predict_positions(run.model, run.config, run.data, run.normalizer, p, c),
        run.data, origins, name,
    )


# ---------------------------------------------------------------- iteraciones propias


def base_configs() -> list[TrainingConfig]:
    return [
        TrainingConfig("mlp", name="MLP", hidden_size=256, dropout=0.1),
        TrainingConfig("dlinear", name="DLinear"),
        TrainingConfig("nlinear", name="NLinear"),
        TrainingConfig("lstm", name="LSTM"),
        TrainingConfig("lstm_ar", name="LSTM-AR"),
        TrainingConfig("seq2seq", name="Seq2Seq"),
    ]


def iteration_grid() -> list[TrainingConfig]:
    """Variaciones de un factor a la vez alrededor de la configuración base (sección 5.2)."""

    configs = base_configs()
    lstm = configs[3]
    dlinear = configs[1]
    mlp = configs[0]
    configs += [replace(lstm, lookback=value) for value in (48, 336)]
    configs += [replace(lstm, history_months=value) for value in (3, 12)]
    configs += [replace(lstm, normalization=value) for value in ("global", "ventana")]
    configs += [replace(lstm, hidden_size=value) for value in (32, 128)]
    configs += [replace(lstm, num_layers=2), replace(lstm, num_layers=2, dropout=0.2)]
    configs += [replace(lstm, learning_rate=value) for value in (3e-4, 3e-3)]
    configs += [replace(lstm, batch_size=value) for value in (256, 4096)]
    configs += [replace(dlinear, lookback=value) for value in (48, 336)]
    configs += [replace(dlinear, history_months=value) for value in (3, 12)]
    configs += [replace(dlinear, normalization=value) for value in ("global", "ventana")]
    configs += [replace(mlp, hidden_size=64, dropout=0.0), replace(mlp, lookback=336)]
    return configs


def command_iterations(args: argparse.Namespace) -> None:
    artifacts = Path(args.artifacts)
    panel = load_panel(artifacts)
    data = PanelTensors(panel, "cuda" if torch.cuda.is_available() else "cpu")
    configs = iteration_grid()
    if args.only:
        configs = [c for c in configs if c.label in args.only]
    summary_path = artifacts / "iteraciones_resumen.csv"
    done = set(pd.read_csv(summary_path)["run_id"]) if summary_path.exists() else set()
    for config in configs:
        run_id = run_id_of(config)
        if run_id in done and not args.force:
            print(f"ya existe {run_id}", flush=True)
            continue
        run = train_model(config, data)
        save_run(run, artifacts, run_id)
        history = run.history.assign(run_id=run_id)
        append_csv(history, artifacts / "iteraciones_entrenamiento.csv", ["run_id"])
        summary = run.summary()
        summary["run_id"] = run_id
        append_csv(pd.DataFrame([summary]), summary_path, ["run_id"])


def candidates(artifacts: Path) -> pd.DataFrame:
    """Corridas elegibles: semilla común; las de NeuralForecast solo con historia completa."""

    summary = pd.read_csv(artifacts / "iteraciones_resumen.csv")
    own = summary["kind"].isin(list(STRATEGY))
    eligible = summary["seed"].eq(CONFIG.seed) & ~summary["run_id"].str.endswith("_semilla")
    return summary.loc[eligible & (own | summary["history_months"].eq(30))]


def best_runs(artifacts: Path) -> pd.DataFrame:
    """Mejor configuración de validación de cada red propia."""

    summary = candidates(artifacts)
    summary = summary.loc[summary["kind"].isin(list(STRATEGY))]
    return summary.sort_values("validation_mase").groupby("modelo", sort=False).head(1)


def command_test_own(args: argparse.Namespace) -> None:
    """Evalúa una sola vez en prueba la mejor configuración de cada red propia."""

    artifacts = Path(args.artifacts)
    panel = load_panel(artifacts)
    data = PanelTensors(panel, "cuda" if torch.cuda.is_available() else "cpu")
    for row in best_runs(artifacts).itertuples(index=False):
        if args.only and row.modelo not in args.only:
            continue
        run = load_run(artifacts, row.run_id, data)
        forecasts, inference = evaluate_on_test(TorchBacktestModel(run, row.modelo), panel)
        total, trainable = run.parameters
        save_test_outputs(
            forecasts,
            artifacts,
            {
                "modelo": row.modelo,
                "familia": FAMILY[run.config.kind],
                "L": run.config.lookback,
                "estrategia": STRATEGY[run.config.kind],
                "usa_covariables": run.config.kind in USES_COVARIATES,
                "total_parameters": total,
                "trainable_parameters": trainable,
                "training_seconds": run.training_seconds,
                "inference_seconds": inference,
                "hardware": hardware(),
                "configuracion": run.config.describe(),
                "validation_mase": row.validation_mase,
                **test_metrics(forecasts),
            },
        )


# ---------------------------------------------------------------- NeuralForecast


def nixtla_specs(steps: int) -> list[dict[str, object]]:
    common = {"max_steps": steps, "learning_rate": 1e-3, "windows_batch_size": 1024}
    return [
        {"name": "N-HiTS", "family": "Residuales y convolucionales", "model": "nhits", "calendar": True,
         "kwargs": {**common}},
        {"name": "TCN", "family": "Residuales y convolucionales", "model": "tcn", "calendar": True,
         "kwargs": {**common, "windows_batch_size": 512, "encoder_hidden_size": 64, "context_size": 24,
                    "decoder_hidden_size": 64}},
        {"name": "PatchTST", "family": "Atención", "model": "patchtst", "calendar": False,
         "kwargs": {**common, "windows_batch_size": 512, "patch_len": 24, "stride": 12}},
        {"name": "iTransformer", "family": "Atención", "model": "itransformer", "calendar": True,
         "kwargs": {"max_steps": steps, "learning_rate": 1e-3, "n_series": 100}},
        {"name": "DeepAR", "family": "Probabilísticos", "model": "deepar", "calendar": True,
         "kwargs": {**common, "windows_batch_size": 512, "lstm_hidden_size": 64,
                    "inference_windows_batch_size": 256, "valid_batch_size": 8}},
    ]


def _nixtla_log(run, val_mae: float, val_mase: float, run_id: str, inference: float) -> pd.DataFrame:
    total, trainable = run.parameters
    log = run.log.copy()
    log = log.rename(columns={"step": "epoch"})
    log["modelo"] = run.name
    log["configuracion"] = f"L={run.input_size} hist={run.history_months}m steps_por_chequeo=100"
    log["L"], log["H"], log["history_months"] = run.input_size, 24, run.history_months
    log["validation_mae"], log["validation_mase"] = np.nan, np.nan
    if len(log):
        log.loc[log.index[-1], ["validation_mae", "validation_mase"]] = [val_mae, val_mase]
    log["total_parameters"], log["trainable_parameters"] = total, trainable
    log["epoch_seconds"] = run.training_seconds / max(len(log), 1)
    log["training_seconds"] = run.training_seconds
    log["inference_seconds"] = inference
    log["peak_gpu_memory_mb"] = run.peak_gpu_memory_mb
    log["seed"] = CONFIG.seed
    return log.reindex(columns=ITERATION_COLUMNS).assign(run_id=run_id)


def run_nixtla_validation(spec, panel, data, artifacts, history_months=30, seed=CONFIG.seed, tag=""):
    from forecasting_lab.nixtla_training import fit_nixtla, nixtla_rolling_forecasts

    run = fit_nixtla(
        spec["name"], spec["family"], spec["model"], panel, uses_calendar=spec["calendar"],
        history_months=history_months, seed=seed, **spec["kwargs"],
    )
    started = time.perf_counter()
    validation, raw = nixtla_rolling_forecasts(run, data, VALIDATION_ORIGINS)
    inference = time.perf_counter() - started
    val_mae, val_mase = forecast_errors(validation)
    run_id = slug(f"{spec['name']}_hist{history_months}_seed{seed}{tag}")
    append_csv(_nixtla_log(run, val_mae, val_mase, run_id, inference), artifacts / "iteraciones_entrenamiento.csv", ["run_id"])
    total, trainable = run.parameters
    append_csv(
        pd.DataFrame([{
            "run_id": run_id, "modelo": run.name, "kind": spec["model"], "configuracion": json.dumps(spec["kwargs"]),
            "lookback": run.input_size, "horizon": 24, "history_months": history_months, "seed": seed,
            "validation_mae": val_mae, "validation_mase": val_mase, "total_parameters": total,
            "trainable_parameters": trainable, "training_seconds": run.training_seconds,
            "inference_seconds": inference, "peak_gpu_memory_mb": run.peak_gpu_memory_mb,
            "epochs_run": int(run.log["step"].max()) if len(run.log) else np.nan,
        }]),
        artifacts / "iteraciones_resumen.csv", ["run_id"],
    )
    print(f"[{run.name}] hist={history_months} seed={seed} val_mase={val_mase:.4f} ({run.training_seconds:.0f}s)", flush=True)
    return run, validation, raw


def pinball_and_coverage(raw: pd.DataFrame, name: str, truth: pd.DataFrame) -> dict[str, float]:
    """Pinball promedio en P10/P50/P90 y cobertura empírica del intervalo P10-P90."""

    joined = raw.merge(truth, on=["unique_id", "ds"], how="left", validate="one_to_one")
    y = joined["y"].to_numpy()
    quantiles = {0.1: f"{name}-lo-80", 0.5: f"{name}-median", 0.9: f"{name}-hi-80"}
    losses = []
    for level, column in quantiles.items():
        error = y - joined[column].to_numpy()
        losses.append(np.mean(np.maximum(level * error, (level - 1) * error)))
    covered = (y >= joined[quantiles[0.1]].to_numpy()) & (y <= joined[quantiles[0.9]].to_numpy())
    return {"pinball_loss": float(np.mean(losses)), "cobertura_p10_p90": float(covered.mean())}


def command_nixtla(args: argparse.Namespace) -> None:
    from forecasting_lab.nixtla_training import NixtlaBacktestModel

    artifacts = Path(args.artifacts)
    panel = load_panel(artifacts)
    data = PanelTensors(panel, "cpu")
    truth = long_panel(panel)
    results_path = artifacts / "resultados_redes.csv"
    finished = set(pd.read_csv(results_path)["modelo"]) if results_path.exists() else set()
    for spec in nixtla_specs(args.steps):
        if args.only and spec["name"] not in args.only:
            continue
        if spec["name"] in finished and not args.force:
            print(f"ya existe {spec['name']}", flush=True)
            continue
        run, validation, _ = run_nixtla_validation(spec, panel, data, artifacts)
        model = NixtlaBacktestModel(run)
        forecasts, inference = evaluate_on_test(model, panel)
        total, trainable = run.parameters
        row = {
            "modelo": run.name, "familia": run.family, "L": run.input_size,
            "estrategia": "autorregresiva" if spec["model"] == "deepar" else "multi-salida",
            "usa_covariables": spec["calendar"], "total_parameters": total, "trainable_parameters": trainable,
            "training_seconds": run.training_seconds, "inference_seconds": inference, "hardware": hardware(),
            "configuracion": json.dumps(spec["kwargs"]), "validation_mase": forecast_errors(validation)[1],
            **test_metrics(forecasts),
        }
        if spec["model"] == "deepar":
            raw = pd.concat(model.quantiles_, ignore_index=True)
            row.update(pinball_and_coverage(raw, run.name, truth))
            quantile_frame = raw.rename(
                columns={f"{run.name}-lo-80": "p10", f"{run.name}-median": "p50", f"{run.name}-hi-80": "p90"}
            )[["unique_id", "ds", "p10", "p50", "p90"]]
            quantile_frame.to_csv(artifacts / "cuantiles_deepar.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
        save_test_outputs(forecasts, artifacts, row)


# ---------------------------------------------------------------- historia de entrenamiento


def _lightgbm_validation(panel: pd.DataFrame, months: int) -> pd.DataFrame:
    """LightGBM global con la misma recursión que en prueba, evaluado en validación."""

    from forecasting_lab.lightgbm_adapter import LightGBMGlobalAdapter

    start = max(pd.Timestamp(CONFIG.validation_start) - pd.DateOffset(months=months), pd.Timestamp(CONFIG.data_start))
    training = long_panel(panel.loc[start : CONFIG.train_end])
    adapter = LightGBMGlobalAdapter(model_params={"n_estimators": 300, "random_state": CONFIG.seed, "n_jobs": -1, "verbose": -1})
    adapter.fit(training)
    data = PanelTensors(panel, "cpu")

    def predict(positions: np.ndarray, columns: np.ndarray) -> np.ndarray:
        values = panel.to_numpy(np.float64)
        output = np.zeros((len(positions), 24))
        for origin in np.unique(positions):
            mask = positions == origin
            cols = columns[mask]
            buffer = list(values[origin - 168 : origin, cols])
            for step in range(24):
                timestamp = panel.index[origin + step]
                features = pd.DataFrame(
                    {**{f"lag_{lag}": buffer[-lag] for lag in adapter.lags},
                     "hour": timestamp.hour, "dayofweek": timestamp.dayofweek, "month": timestamp.month}
                )
                prediction = adapter.model_.predict(features[adapter.feature_columns_])
                buffer.append(prediction)
                output[mask, step] = prediction
        return output

    return rolling_forecasts(predict, data, VALIDATION_ORIGINS, "LightGBM")


def _chronos_validation(panel: pd.DataFrame) -> pd.DataFrame:
    from forecasting_lab.chronos_adapter import ChronosBoltAdapter

    data = PanelTensors(panel, "cpu")
    adapter = ChronosBoltAdapter(model_id="amazon/chronos-bolt-small")
    values = panel.to_numpy(np.float32)

    def predict(positions: np.ndarray, columns: np.ndarray) -> np.ndarray:
        context = np.stack([values[p - 2048 : p, c] for p, c in zip(positions, columns)])
        outputs = []
        for start in range(0, len(context), 1024):
            quantiles, _ = adapter.predict(context[start : start + 1024], 24)
            outputs.append(np.asarray(quantiles.detach().cpu(), dtype=float)[:, :, 1])
        return np.concatenate(outputs)

    return rolling_forecasts(predict, data, VALIDATION_ORIGINS, "Chronos-Bolt")


def command_history(args: argparse.Namespace) -> None:
    """Sensibilidad del MASE de validación a 3, 12 y 30 meses de historia."""

    artifacts = Path(args.artifacts)
    panel = load_panel(artifacts)
    data = PanelTensors(panel, "cpu")
    rows = []
    values = panel.to_numpy(np.float64)
    seasonal = rolling_forecasts(lambda p, c: np.stack([values[q - 24 : q, k] for q, k in zip(p, c)]),
                                 data, VALIDATION_ORIGINS, "SeasonalNaive24")
    chronos = _chronos_validation(panel)
    for months in (3, 12, 30):
        rows.append({"modelo": "SeasonalNaive24", "history_months": months, "mase": forecast_errors(seasonal)[1]})
        rows.append({"modelo": "Chronos-Bolt (zero-shot)", "history_months": months, "mase": forecast_errors(chronos)[1]})
        lightgbm = _lightgbm_validation(panel, months)
        rows.append({"modelo": "LightGBM", "history_months": months, "mase": forecast_errors(lightgbm)[1]})
        print(rows[-3:], flush=True)
    # Mejor recurrente según validación con la configuración base de cada una.
    summary = pd.read_csv(artifacts / "iteraciones_resumen.csv").set_index("run_id")
    references = {config.label: config for config in base_configs()}
    recurrent = min(
        ("LSTM", "LSTM-AR", "Seq2Seq"),
        key=lambda name: summary.loc[run_id_of(references[name]), "validation_mase"],
    )
    gpu = PanelTensors(panel, "cuda" if torch.cuda.is_available() else "cpu")
    for model in ("DLinear", recurrent):
        for months in (3, 12, 30):
            config = replace(references[model], history_months=months)
            run_id = run_id_of(config)
            if run_id not in summary.index:
                run = train_model(config, gpu)
                save_run(run, artifacts, run_id)
                append_csv(run.history.assign(run_id=run_id), artifacts / "iteraciones_entrenamiento.csv", ["run_id"])
                append_csv(pd.DataFrame([{**run.summary(), "run_id": run_id}]), artifacts / "iteraciones_resumen.csv", ["run_id"])
                summary = pd.read_csv(artifacts / "iteraciones_resumen.csv").set_index("run_id")
            rows.append({"modelo": model, "history_months": months, "mase": float(summary.loc[run_id, "validation_mase"])})
    for months in (3, 12, 30):
        run_id = slug(f"PatchTST_hist{months}_seed{CONFIG.seed}")
        if run_id in summary.index:
            rows.append({"modelo": "PatchTST", "history_months": months, "mase": float(summary.loc[run_id, "validation_mase"])})
    frame = pd.DataFrame(rows).drop_duplicates(["modelo", "history_months"], keep="last")
    frame.to_csv(artifacts / "sensibilidad_historia.csv", index=False)
    print(frame.to_string(index=False))


def command_patchtst_history(args: argparse.Namespace) -> None:
    artifacts = Path(args.artifacts)
    panel = load_panel(artifacts)
    data = PanelTensors(panel, "cpu")
    spec = [s for s in nixtla_specs(args.steps) if s["name"] == "PatchTST"][0]
    for months in (3, 12):
        run_nixtla_validation(spec, panel, data, artifacts, history_months=months)


# ---------------------------------------------------------------- local contra global, semillas y mensual


def best_own_run(artifacts: Path) -> pd.Series:
    return best_runs(artifacts).sort_values("validation_mase").iloc[0]


def command_local_global(args: argparse.Namespace) -> None:
    """Mejor red propia: un modelo por cliente (10 clientes) contra el modelo global."""

    artifacts = Path(args.artifacts)
    panel = load_panel(artifacts)
    data = PanelTensors(panel, "cuda" if torch.cuda.is_available() else "cpu")
    best = best_own_run(artifacts)
    global_run = load_run(artifacts, best["run_id"], data)
    rng = np.random.default_rng(CONFIG.seed)
    clients = sorted(rng.choice(panel.columns.to_numpy(), size=10, replace=False).tolist())
    global_forecasts = fast_test(global_run, "global")
    rows = []
    for client in clients:
        config = replace(global_run.config, clients=(client,), batch_size=min(global_run.config.batch_size, 256))
        local = train_model(config, data, verbose=False)
        local_forecasts = fast_test(local, "local")
        global_client = global_forecasts.loc[global_forecasts["unique_id"].eq(client)]
        rows.append({
            "unique_id": client,
            "modelo_base": best["modelo"],
            "consumo_medio_kwh": float(panel.loc[:CONFIG.train_end, client].mean()),
            "mase_local": float(aggregate_metrics(local_forecasts).iloc[0]["mase"]),
            "mase_global": float(aggregate_metrics(global_client).iloc[0]["mase"]),
            "validation_mase_local": float(local.history["validation_mase"].min()),
            "epochs_local": local.best_epoch,
            "training_seconds_local": local.training_seconds,
        })
        print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame["mejora_global"] = frame["mase_local"] - frame["mase_global"]
    frame.to_csv(artifacts / "local_vs_global.csv", index=False)
    print(frame.to_string(index=False))


def _best_overall(artifacts: Path) -> pd.Series:
    """Mejor red por MASE de validación entre propias y NeuralForecast."""

    return candidates(artifacts).sort_values("validation_mase").iloc[0]


def command_seeds(args: argparse.Namespace) -> None:
    """Repite la mejor red con tres semillas: validación y prueba."""

    artifacts = Path(args.artifacts)
    panel = load_panel(artifacts)
    best = _best_overall(artifacts)
    rows = []
    for seed in (CONFIG.seed, 1, 2):
        if best["kind"] in STRATEGY:
            data = PanelTensors(panel, "cuda" if torch.cuda.is_available() else "cpu")
            if seed == CONFIG.seed:
                run = load_run(artifacts, best["run_id"], data)
                validation_mase = float(best["validation_mase"])
            else:
                reference = load_run(artifacts, best["run_id"], data).config
                run = train_model(replace(reference, seed=seed), data, verbose=False)
                validation_mase = float(run.history["validation_mase"].min())
            forecasts = fast_test(run, best["modelo"])
        else:
            spec = [s for s in nixtla_specs(args.steps) if s["name"] == best["modelo"]][0]
            run, validation, _ = run_nixtla_validation(
                spec, panel, PanelTensors(panel, "cpu"), artifacts, seed=seed, tag="_semilla"
            )
            from forecasting_lab.nixtla_training import nixtla_rolling_forecasts

            validation_mase = forecast_errors(validation)[1]
            forecasts, _ = nixtla_rolling_forecasts(run, PanelTensors(panel, "cpu"), TEST_ORIGINS)
        rows.append({"modelo": best["modelo"], "seed": seed, "validation_mase": validation_mase,
                     "test_mase": float(aggregate_metrics(forecasts).iloc[0]["mase"])})
        print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(artifacts / "semillas.csv", index=False)
    print(frame.to_string(index=False))
    print("std validación", frame["validation_mase"].std(ddof=1), "std prueba", frame["test_mase"].std(ddof=1))


def _monthly_forecasts(best: pd.Series, panel: pd.DataFrame, artifacts: Path, month_start: pd.Timestamp,
                       origins: pd.DatetimeIndex, args: argparse.Namespace) -> pd.DataFrame:
    """Reentrena con todo lo observado antes del mes y pronostica solo ese mes."""

    cutoff = month_start - pd.Timedelta(1, unit="h")
    # Ventana expandible: historia completa hasta el corte y el mismo presupuesto elegido en validación.
    months = (cutoff.year - 2012) * 12 + cutoff.month
    name = f"{best['modelo']} mensual"
    if best["kind"] in STRATEGY:
        data = PanelTensors(panel, "cuda" if torch.cuda.is_available() else "cpu")
        reference = load_run(artifacts, best["run_id"], data)
        config = replace(reference.config, history_months=months)
        run = train_model(config, data, train_end=cutoff, fixed_epochs=reference.best_epoch, verbose=False)
        return fast_test(run, name, origins)
    from forecasting_lab.nixtla_training import fit_nixtla, nixtla_rolling_forecasts

    spec = [s for s in nixtla_specs(args.steps) if s["name"] == best["modelo"]][0]
    run = fit_nixtla(
        spec["name"], spec["family"], spec["model"], panel, uses_calendar=spec["calendar"],
        history_months=months, train_end=cutoff, fixed_steps=int(best["epochs_run"]), **spec["kwargs"],
    )
    forecasts, _ = nixtla_rolling_forecasts(run, PanelTensors(panel, "cpu"), origins)
    return forecasts.assign(modelo=name)


def command_monthly(args: argparse.Namespace) -> None:
    """Reentrena al inicio de octubre, noviembre y diciembre la mejor red propia y la mejor red global."""

    artifacts = Path(args.artifacts)
    panel = load_panel(artifacts)
    truth = long_panel(panel)
    targets = [best_own_run(artifacts)]
    overall = _best_overall(artifacts)
    if overall["kind"] not in STRATEGY:
        targets.append(overall)
    for best in targets:
        if args.only and best["modelo"] not in args.only:
            continue
        parts, rows = [], []
        for month in ("2014-10", "2014-11", "2014-12"):
            month_start = pd.Timestamp(f"{month}-01")
            origins = TEST_ORIGINS[(TEST_ORIGINS >= month_start) & (TEST_ORIGINS < month_start + pd.offsets.MonthBegin(1))]
            forecasts = _monthly_forecasts(best, panel, artifacts, month_start, origins, args)
            parts.append(forecasts)
            rows.append({"mes_reentrenamiento": month, "modelo": best["modelo"], **aggregate_metrics(forecasts).iloc[0].to_dict()})
            print(rows[-1], flush=True)
        monthly = pd.concat(parts, ignore_index=True)
        rows.append({"mes_reentrenamiento": "total", "modelo": best["modelo"], **aggregate_metrics(monthly).iloc[0].to_dict()})
        single = pd.read_csv(artifacts / f"pronosticos_{slug(best['modelo'])}.csv", parse_dates=["ds"])
        single = single.merge(truth, on=["unique_id", "ds"])
        single = single.merge(monthly[["unique_id", "ds", "mase_scale"]], on=["unique_id", "ds"])
        rows.append({"mes_reentrenamiento": "sin reentrenar", "modelo": best["modelo"], **aggregate_metrics(single).iloc[0].to_dict()})
        append_csv(pd.DataFrame(rows), artifacts / "reentrenamiento_mensual_redes.csv", ["modelo", "mes_reentrenamiento"])
        write_predictions(monthly, artifacts / f"pronosticos_{slug(best['modelo'])}_mensual.csv")
        print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, handler, help_text in (
        ("iteraciones", command_iterations, "entrena la grilla de la sección 5.2 (solo validación)"),
        ("prueba-propias", command_test_own, "evalúa en prueba la mejor configuración de cada red propia"),
        ("nixtla", command_nixtla, "entrena N-HiTS, TCN, PatchTST, iTransformer y DeepAR"),
        ("patchtst-historia", command_patchtst_history, "PatchTST con 3 y 12 meses de historia"),
        ("historia", command_history, "tabla de sensibilidad a meses de historia"),
        ("local-global", command_local_global, "modelo por cliente contra modelo global"),
        ("semillas", command_seeds, "mejor red con tres semillas"),
        ("mensual", command_monthly, "reentrenamiento mensual de la mejor red"),
    ):
        stage = sub.add_parser(name, help=help_text)
        stage.add_argument("--artifacts", default="artifacts")
        stage.add_argument("--only", nargs="*")
        stage.add_argument("--force", action="store_true")
        stage.add_argument("--steps", type=int, default=3000)
        stage.set_defaults(handler=handler)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
