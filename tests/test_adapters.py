"""Pruebas de comportamiento para bases deterministas y adaptadores opcionales."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting_lab.baselines import (  # noqa: E402
    BaselineForecaster,
    naive_forecast,
    seasonal_naive_forecast,
)
import forecasting_lab.baselines as baselines_module  # noqa: E402
import forecasting_lab.chronos_adapter as chronos_module  # noqa: E402
import forecasting_lab.lightgbm_adapter as lightgbm_module  # noqa: E402
import forecasting_lab.neuralforecast_adapter as neuralforecast_module  # noqa: E402
from forecasting_lab.backtest import backtest  # noqa: E402
from forecasting_lab.chronos_adapter import ChronosBoltAdapter  # noqa: E402
from forecasting_lab.lightgbm_adapter import LightGBMGlobalAdapter  # noqa: E402
from forecasting_lab.neuralforecast_adapter import (  # noqa: E402
    NeuralForecastAdapter,
    validate_future_format,
    validate_long_format,
)


def _complete_hourly_data(start="2014-09-23"):
    """Crea verdad horaria completa para los 92 orígenes del backtest."""
    timestamps = pd.date_range(start, "2014-12-31 23:00", freq="h")
    return pd.DataFrame(
        {
            "unique_id": "cliente-1",
            "ds": timestamps,
            "y": np.arange(len(timestamps), dtype=float),
            "temperatura": 20.0,
        }
    )


def test_baselines_produce_hand_checked_deterministic_forecasts():
    """Cambiar la regla del último valor o ciclo estacional debe fallar estas salidas."""
    history = np.array([1.0, 2.0, 3.0, 4.0])

    assert naive_forecast(history, horizon=3).tolist() == [4.0, 4.0, 4.0]
    assert seasonal_naive_forecast(history, horizon=5, season_length=2).tolist() == [3.0, 4.0, 3.0, 4.0, 3.0]
    assert BaselineForecaster(method="seasonal_naive", season_length=2).predict(
        history, horizon=3
    ).tolist() == [3.0, 4.0, 3.0]


def test_seasonal_baseline_rejects_insufficient_history():
    """Quitar la historia mínima debe fallar en vez de inventar una estación."""
    with pytest.raises(ValueError, match="season_length"):
        seasonal_naive_forecast(np.array([1.0]), horizon=2, season_length=24)


def test_baseline_es_compatible_con_backtest_y_su_multiindex():
    """Romper la firma del protocolo debe impedir ejecutar los 92 orígenes reales."""
    result = backtest(BaselineForecaster(), _complete_hourly_data())

    assert len(result) == 92 * 24
    first_origin = result.loc[result["origin"].eq(pd.Timestamp("2014-10-01"))]
    assert first_origin["y_hat"].eq(191.0).all()


def test_lightgbm_adapter_validates_long_format_before_optional_import():
    """Quitar las columnas obligatorias debe aceptar datos globales mal formados."""
    with pytest.raises(ValueError, match="unique_id"):
        LightGBMGlobalAdapter().fit(pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=2), "y": [1, 2]}))


def test_lightgbm_adapter_explains_missing_optional_dependency(monkeypatch):
    """Reemplazar el error de importación útil debe fallar este contrato de dependencia."""
    data = pd.DataFrame(
        {
            "unique_id": ["a"] * 170,
            "ds": pd.date_range("2024-01-01", periods=170, freq="h"),
            "y": np.arange(170, dtype=float),
        }
    )
    def importacion_ausente(name):
        if name == "lightgbm":
            raise ImportError("ausente")
        return __import__(name)

    monkeypatch.setattr(lightgbm_module.importlib, "import_module", importacion_ausente)
    adapter = LightGBMGlobalAdapter()
    with pytest.raises(ImportError, match="pip install lightgbm"):
        adapter.fit(data)


@pytest.mark.parametrize(
    "lags",
    [
        tuple(range(1, 25)) + (168, 0),
        tuple(range(1, 25)) + (168, -1),
        tuple(range(1, 25)) + (168, 168),
    ],
)
def test_lightgbm_rechaza_rezagos_no_positivos_o_repetidos(lags):
    """Aceptar rezagos inválidos debe contaminar las características causales."""
    with pytest.raises(ValueError, match="positivos y únicos"):
        LightGBMGlobalAdapter(lags=lags)


def test_lightgbm_con_modelo_falso_es_compatible_con_backtest():
    """Ignorar el índice futuro debe romper la alineación exigida por backtest."""
    class ModeloUltimoRezago:
        def predict(self, features):
            return features["lag_1"].to_numpy()

    adapter = LightGBMGlobalAdapter(model=ModeloUltimoRezago())
    result = backtest(
        adapter,
        _complete_hourly_data(),
        future_covariate_columns=["temperatura"],
    )

    assert len(result) == 92 * 24
    assert result["y_hat"].notna().all()
    assert result.loc[0, "y_hat"] == 191.0


def test_chronos_adapter_forwards_quantile_request_to_injected_pipeline():
    """Cambiar el horizonte o cuantiles de Chronos debe fallar este contrato de borde."""
    class Pipeline:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            assert inputs.tolist() == [1.0, 2.0]
            assert prediction_length == 3
            assert quantile_levels == [0.1, 0.5, 0.9]
            return "quantiles"

    result = ChronosBoltAdapter(pipeline=Pipeline()).predict(np.array([1.0, 2.0]), 3)

    assert result == "quantiles"


def test_chronos_adapter_converts_numpy_context_to_torch_tensor():
    """La tubería oficial rechaza arreglos NumPy aunque sus valores sean válidos."""
    import torch

    class Pipeline:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            assert isinstance(inputs, torch.Tensor)
            return inputs

    result = ChronosBoltAdapter(pipeline=Pipeline()).predict(np.array([[1.0, 2.0]]), 1)

    assert result.dtype == torch.float32


def test_chronos_adapter_rejects_non_positive_prediction_length():
    """Quitar la validación debe permitir una petición que la API Chronos no admite."""
    with pytest.raises(ValueError, match="prediction_length"):
        ChronosBoltAdapter(pipeline=object()).predict(np.array([1.0]), 0)


def test_chronos_explains_missing_optional_dependency(monkeypatch):
    """Perder la guía de instalación debe ocultar cómo habilitar Chronos-Bolt."""
    def importacion_ausente(name):
        if name == "chronos":
            raise ImportError("ausente")
        return __import__(name)

    monkeypatch.setattr(chronos_module.importlib, "import_module", importacion_ausente)
    with pytest.raises(ImportError, match="pip install chronos-forecasting"):
        ChronosBoltAdapter().predict(np.array([1.0]), 1)


def test_statsforecast_explains_missing_optional_dependency(monkeypatch):
    """Perder la guía de instalación debe ocultar cómo habilitar MSTL/AutoETS."""
    def importacion_ausente(name):
        if name == "statsforecast.models":
            raise ImportError("ausente")
        return __import__(name)

    monkeypatch.setattr(baselines_module.importlib, "import_module", importacion_ausente)
    with pytest.raises(ImportError, match="pip install statsforecast"):
        baselines_module.StatsForecastMSTLAutoETSForecaster().build_models()


def test_neuralforecast_adapter_enforces_long_format():
    """Quitar las verificaciones debe aceptar datos sin la columna objetivo y."""
    malformed = pd.DataFrame({"unique_id": ["a"], "ds": ["2024-01-01"]})

    with pytest.raises(ValueError, match="y"):
        validate_long_format(malformed)


def test_neuralforecast_valida_futuro_sin_exigir_objetivo():
    """Reutilizar el validador de entrenamiento debe exigir y indebidamente en el futuro."""
    future = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": ["2024-01-02"],
            "temperatura": [21.0],
        }
    )

    validated = validate_future_format(future)

    assert list(validated.columns) == ["unique_id", "ds", "temperatura"]
    with pytest.raises(ValueError, match="no debe contener y"):
        validate_future_format(future.assign(y=1.0))


def test_neuralforecast_adapter_explains_missing_optional_dependency(monkeypatch):
    """Reemplazar la guía de importación debe fallar este contrato de construcción."""
    def importacion_ausente(name):
        if name == "neuralforecast.models":
            raise ImportError("ausente")
        return __import__(name)

    monkeypatch.setattr(neuralforecast_module.importlib, "import_module", importacion_ausente)
    with pytest.raises(ImportError, match="pip install neuralforecast"):
        NeuralForecastAdapter.make_model("nhits", h=4, input_size=12)
