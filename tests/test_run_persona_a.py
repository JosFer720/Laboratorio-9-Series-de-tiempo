from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_lab.config import ExperimentConfig
from forecasting_lab.backtest import backtest
from forecasting_lab.baselines import BaselineForecaster
from forecasting_lab.lightgbm_adapter import LightGBMGlobalAdapter
from scripts.run_persona_a import (
    _baseline_backtest,
    _per_horizon_from_forecasts,
    _lightgbm_backtest,
    _long_panel,
    consolidate_artifacts,
    prepare_artifacts,
)


def test_prepare_artifacts_records_checksum_counts_and_reusable_panel(tmp_path: Path) -> None:
    index = pd.date_range("2014-01-01 00:15", periods=4 * 96, freq="15min")
    raw = pd.DataFrame({"a": np.ones(len(index)), "b": np.full(len(index), 2.0)}, index=index)
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path)
    config = ExperimentConfig(
        data_start="2014-01-01 00:00",
        train_end="2014-01-01 23:00",
        validation_start="2014-01-02 00:00",
        validation_end="2014-01-02 23:00",
        test_start="2014-01-03 00:00",
        test_end="2014-01-04 23:00",
        lookback=4,
        horizon=2,
        minimum_clients=1,
    )
    artifacts = tmp_path / "artifacts"

    metadata = prepare_artifacts(raw, raw_path, artifacts, config=config, n_clients=2)

    saved = json.loads((artifacts / "metadata_preparacion.json").read_text(encoding="utf-8"))
    cached = pd.read_csv(
        artifacts / "cache" / "panel_seleccionado.csv.gz", index_col="ds", parse_dates=["ds"]
    )
    assert metadata == saved
    assert len(saved["sha256_datos_crudos"]) == 64
    assert saved["clientes_completos"] == 2
    assert saved["clientes_seleccionados"] == 2
    assert saved["horas_por_serie"] == 96
    assert saved["ventanas"]["train"] == 19 * 2
    assert cached.shape == (96, 2)


def test_vectorized_baseline_matches_the_shared_backtest() -> None:
    index = pd.date_range("2014-09-23", "2014-12-31 23:00", freq="h")
    panel = pd.DataFrame(
        {"a": np.arange(len(index), dtype=float), "b": np.arange(len(index), dtype=float) + 10},
        index=index,
    )
    model = BaselineForecaster("seasonal_naive", 24)
    model.name = "SeasonalNaive24"

    generic = backtest(model, _long_panel(panel))
    vectorized = _baseline_backtest(panel, "SeasonalNaive24", 24)

    columns = ["unique_id", "ds", "y", "y_hat", "horizon"]
    pd.testing.assert_frame_equal(
        generic[columns].reset_index(drop=True), vectorized[columns].reset_index(drop=True)
    )


def test_vectorized_lightgbm_matches_adapter_backtest_with_fake_regressor() -> None:
    class FakeRegressor:
        def predict(self, features):
            return features["lag_24"].to_numpy()

    index = pd.date_range("2014-09-23", "2014-12-31 23:00", freq="h")
    panel = pd.DataFrame(
        {"a": np.arange(len(index), dtype=float), "b": np.arange(len(index), dtype=float) + 10},
        index=index,
    )
    model = LightGBMGlobalAdapter(model=FakeRegressor())
    model.name = "LightGBM"

    generic = backtest(model, _long_panel(panel))
    vectorized = _lightgbm_backtest(panel, model)

    columns = ["unique_id", "ds", "y", "y_hat", "horizon"]
    pd.testing.assert_frame_equal(
        generic[columns].reset_index(drop=True), vectorized[columns].reset_index(drop=True)
    )


def test_consolidate_artifacts_omits_duplicate_statsforecast_baselines(tmp_path: Path) -> None:
    """La entrega conjunta conserva AutoETS, pero no duplica baselines equivalentes."""
    result_columns = {
        "familia": "base",
        "L": 24,
        "estrategia": "directa",
        "usa_covariables": False,
        "training_seconds": 0.01,
        "inference_seconds": 0.01,
        "hardware": "CPU",
        "mae": 1.0,
        "rmse": 1.0,
        "wape": 1.0,
        "mape": 1.0,
        "mase": 1.0,
    }
    pd.DataFrame([{"modelo": "Naive", **result_columns}]).to_csv(
        tmp_path / "resultados_baselines.csv", index=False
    )
    pd.DataFrame(
        [
            {"modelo": "NaiveSF", **result_columns},
            {"modelo": "AutoETS24", **result_columns, "mase": 0.9},
        ]
    ).to_csv(tmp_path / "resultados_statsforecast.csv", index=False)
    predictions = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": ["2014-10-01", "2014-10-01"],
            "modelo": ["NaiveSF", "AutoETS24"],
            "y_hat": [1.0, 2.0],
        }
    )
    predictions.to_csv(tmp_path / "pronosticos_statsforecast.csv", index=False)

    results, forecasts = consolidate_artifacts(tmp_path)

    assert results["modelo"].tolist() == ["AutoETS24", "Naive"]
    assert forecasts["modelo"].tolist() == ["AutoETS24"]
    assert (tmp_path / "resultados_modelos.csv").exists()
    assert (tmp_path / "pronosticos_modelos.csv").exists()


def test_per_horizon_metrics_join_predictions_to_panel_truth() -> None:
    """La curva de horizonte debe reconstruir la verdad sin guardarla en el CSV final."""
    panel = pd.DataFrame(
        {"a": [1.0, 3.0]},
        index=pd.to_datetime(["2014-10-01 00:00", "2014-10-01 01:00"]),
    )
    forecasts = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.to_datetime(["2014-10-01 00:00", "2014-10-01 01:00"]),
            "modelo": ["modelo", "modelo"],
            "y_hat": [0.0, 1.0],
        }
    )

    result = _per_horizon_from_forecasts(forecasts, panel)

    assert result.to_dict("records") == [
        {"modelo": "modelo", "horizon": 1, "mae": 1.0},
        {"modelo": "modelo", "horizon": 2, "mae": 2.0},
    ]
