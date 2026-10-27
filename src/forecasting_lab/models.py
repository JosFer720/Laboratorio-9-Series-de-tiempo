"""Redes neuronales pequeñas, por lotes primero, para pronósticos puntuales."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _sequence(x: Tensor, input_size: int, name: str = "history") -> Tensor:
    """Normaliza datos univariados de ``(lote, tiempo)`` a tres dimensiones."""
    if x.ndim == 2 and input_size == 1:
        x = x.unsqueeze(-1)
    if x.ndim != 3 or x.shape[-1] != input_size:
        raise ValueError(f"{name} must have shape (batch, time, {input_size})")
    return x


class MultiOutputMLP(nn.Module):
    """MLP directo multi-horizonte; usa únicamente valores del contexto."""

    def __init__(self, context_length: int, horizon: int, hidden_size: int = 64, input_size: int = 1):
        super().__init__()
        self.context_length, self.horizon, self.input_size = context_length, horizon, input_size
        self.network = nn.Sequential(
            nn.Linear(context_length * input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon)
        )

    def forward(self, history: Tensor) -> Tensor:
        history = _sequence(history, self.input_size)
        if history.shape[1] != self.context_length:
            raise ValueError("history time dimension must equal context_length")
        return self.network(history.reshape(history.shape[0], -1))


class DLinear(nn.Module):
    """Modelo lineal directo con descomposición para series univariadas."""

    def __init__(self, context_length: int, horizon: int, kernel_size: int = 25):
        super().__init__()
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser un entero positivo e impar")
        self.context_length, self.horizon = context_length, horizon
        self.kernel_size = kernel_size
        self.seasonal = nn.Linear(context_length, horizon)
        self.trend = nn.Linear(context_length, horizon)

    def forward(self, history: Tensor) -> Tensor:
        history = _sequence(history, 1).squeeze(-1)
        if history.shape[1] != self.context_length:
            raise ValueError("history time dimension must equal context_length")
        padding = (self.kernel_size - 1) // 2
        trend = F.avg_pool1d(F.pad(history.unsqueeze(1), (padding, padding), mode="replicate"), self.kernel_size, stride=1)
        trend = trend.squeeze(1)
        return self.seasonal(history - trend) + self.trend(trend)


class NLinear(nn.Module):
    """Modelo lineal directo centrado en el último objetivo observado."""

    def __init__(self, context_length: int, horizon: int):
        super().__init__()
        self.context_length, self.horizon = context_length, horizon
        self.linear = nn.Linear(context_length, horizon)

    def forward(self, history: Tensor) -> Tensor:
        history = _sequence(history, 1).squeeze(-1)
        if history.shape[1] != self.context_length:
            raise ValueError("history time dimension must equal context_length")
        last = history[:, -1:].detach()
        return self.linear(history - last) + last


class LSTMForecaster(nn.Module):
    """Pronosticador puntual LSTM de un paso para historiales por lotes."""

    def __init__(self, input_size: int = 1, horizon: int = 1, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.input_size, self.horizon = input_size, horizon
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, history: Tensor) -> Tensor:
        history = _sequence(history, self.input_size)
        _, (hidden, _) = self.encoder(history)
        return self.head(hidden[-1])


class AutoregressiveLSTM(nn.Module):
    """LSTM que avanza con sus propias predicciones, sin objetivos futuros."""

    def __init__(self, input_size: int = 1, horizon: int = 1, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.input_size, self.horizon = input_size, horizon
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, history: Tensor) -> Tensor:
        history = _sequence(history, self.input_size)
        _, state = self.encoder(history)
        step = history[:, -1:, :]
        forecasts: list[Tensor] = []
        for _ in range(self.horizon):
            decoded, state = self.decoder(step, state)
            prediction = self.head(decoded[:, -1, :])
            forecasts.append(prediction)
            step = step.clone()
            step[:, :, 0] = prediction
        return torch.cat(forecasts, dim=1)


class Seq2SeqForecaster(nn.Module):
    """LSTM codificador-decodificador que usa solo covariables futuras conocidas."""

    def __init__(
        self, input_size: int, decoder_covariate_size: int, horizon: int, hidden_size: int = 64, num_layers: int = 1
    ):
        super().__init__()
        self.input_size, self.decoder_covariate_size, self.horizon = input_size, decoder_covariate_size, horizon
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(decoder_covariate_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, history: Tensor, future_covariates: Tensor) -> Tensor:
        history = _sequence(history, self.input_size)
        if future_covariates.ndim != 3 or future_covariates.shape[-1] != self.decoder_covariate_size:
            raise ValueError("future_covariates must have shape (batch, horizon, decoder_covariate_size)")
        if future_covariates.shape[0] != history.shape[0] or future_covariates.shape[1] != self.horizon:
            raise ValueError("future_covariates batch and horizon must match the configured horizon")
        _, state = self.encoder(history)
        decoded, _ = self.decoder(future_covariates, state)
        return self.head(decoded).squeeze(-1)
