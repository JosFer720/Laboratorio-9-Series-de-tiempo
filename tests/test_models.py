"""Pruebas de contrato para los modelos ligeros de pronóstico con PyTorch."""

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting_lab.models import (  # noqa: E402
    AutoregressiveLSTM,
    DLinear,
    LSTMForecaster,
    MultiOutputMLP,
    NLinear,
    Seq2SeqForecaster,
)


@pytest.mark.parametrize(
    "model",
    [
        MultiOutputMLP(context_length=12, horizon=4, hidden_size=8),
        DLinear(context_length=12, horizon=4),
        NLinear(context_length=12, horizon=4),
        LSTMForecaster(input_size=1, horizon=4, hidden_size=8),
        AutoregressiveLSTM(input_size=1, horizon=4, hidden_size=8),
    ],
)
def test_point_forecasters_return_batch_by_horizon(model):
    """Quitar la proyección de horizonte debe fallar este contrato de forma pública."""
    history = torch.randn(3, 12)

    forecast = model(history)

    assert forecast.shape == (3, 4)


def test_seq2seq_uses_future_covariates_without_target_values():
    """Ignorar covariables del decodificador o el horizonte debe fallar este contrato."""
    model = Seq2SeqForecaster(
        input_size=1, decoder_covariate_size=2, horizon=4, hidden_size=8
    )
    history = torch.randn(3, 12, 1)
    future_covariates = torch.randn(3, 4, 2)

    forecast = model(history, future_covariates)

    assert forecast.shape == (3, 4)


def test_seq2seq_rejects_decoder_covariates_for_a_different_horizon():
    """Eliminar esta validación debe fallar antes de que el modelo filtre objetivos."""
    model = Seq2SeqForecaster(
        input_size=1, decoder_covariate_size=2, horizon=4, hidden_size=8
    )

    with pytest.raises(ValueError, match="horizon"):
        model(torch.randn(2, 12, 1), torch.randn(2, 3, 2))


@pytest.mark.parametrize("kernel_size", [0, -3, 4, 3.5, True])
def test_dlinear_rejects_kernel_no_positivo_o_par(kernel_size):
    """Aceptar un kernel no positivo o par debe fallar antes del cálculo de tendencia."""
    with pytest.raises(ValueError, match="kernel_size"):
        DLinear(context_length=12, horizon=4, kernel_size=kernel_size)
