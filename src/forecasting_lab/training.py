"""Entrenamiento en GPU de las redes propias con registro de iteraciones por epoch."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .dataset import SeriesStandardizer
from .features import make_calendar_features
from .models import (
    AutoregressiveLSTM,
    DLinear,
    LSTMForecaster,
    MultiOutputMLP,
    NLinear,
    Seq2SeqForecaster,
)

ITERATION_COLUMNS = [
    "modelo", "configuracion", "L", "H", "history_months",
    "epoch", "train_loss", "validation_loss", "validation_mae",
    "validation_mase", "total_parameters", "trainable_parameters",
    "epoch_seconds", "training_seconds", "inference_seconds",
    "peak_gpu_memory_mb", "seed",
]
FORECAST_COLUMNS = ["unique_id", "ds", "modelo", "y_hat", "y", "origin", "horizon", "mase_scale"]
MODEL_KINDS = ("mlp", "dlinear", "nlinear", "lstm", "lstm_ar", "seq2seq")
# Las redes univariadas reciben solo el objetivo; las demás, objetivo y calendario.
USES_COVARIATES = {"mlp", "lstm", "lstm_ar", "seq2seq"}
STRATEGY = {
    "mlp": "multi-salida", "dlinear": "multi-salida", "nlinear": "multi-salida",
    "lstm": "multi-salida", "lstm_ar": "autorregresiva", "seq2seq": "encoder-decoder",
}
FAMILY = {
    "mlp": "MLP y lineales", "dlinear": "MLP y lineales", "nlinear": "MLP y lineales",
    "lstm": "Recurrentes", "lstm_ar": "Recurrentes", "seq2seq": "Encoder-decoder",
}


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparámetros de una corrida; ``history_months`` cuenta hacia atrás desde el fin de train."""

    kind: str
    name: str = ""
    lookback: int = 168
    horizon: int = 24
    history_months: int = 30
    normalization: str = "serie"
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 25
    patience: int = 4
    seed: int = 3092
    train_stride: int = 1
    clients: tuple[str, ...] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in MODEL_KINDS:
            raise ValueError(f"modelo desconocido: {self.kind}")
        if self.normalization not in {"serie", "global", "ventana"}:
            raise ValueError("normalization debe ser serie, global o ventana")

    @property
    def label(self) -> str:
        return self.name or self.kind

    def describe(self) -> str:
        """Resume la configuración en una cadena estable para la tabla de iteraciones."""

        return (
            f"L={self.lookback} hist={self.history_months}m norm={self.normalization} "
            f"hidden={self.hidden_size} capas={self.num_layers} dropout={self.dropout} "
            f"lr={self.learning_rate} batch={self.batch_size}"
        )


class CalendarAutoregressiveLSTM(AutoregressiveLSTM):
    """Variante autorregresiva que reemplaza el calendario del paso realimentado.

    La clase base copia las covariables de la última hora observada en cada paso;
    aquí cada paso recibe el calendario de la hora que acaba de predecir, que se
    conoce de antemano.
    """

    def forward(self, history: Tensor, future_covariates: Tensor | None = None) -> Tensor:
        if future_covariates is None or self.input_size == 1:
            return super().forward(history)
        _, state = self.encoder(history)
        step = history[:, -1:, :]
        forecasts: list[Tensor] = []
        for position in range(self.horizon):
            decoded, state = self.decoder(step, state)
            prediction = self.head(decoded[:, -1, :])
            forecasts.append(prediction)
            if position + 1 < self.horizon:
                step = torch.cat([prediction[:, None, :], future_covariates[:, position : position + 1, :]], dim=2)
        return torch.cat(forecasts, dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: TrainingConfig, covariate_count: int = 7) -> nn.Module:
    """Instancia las clases de ``models.py`` según la configuración."""

    inputs = 1 + covariate_count
    L, H = config.lookback, config.horizon
    if config.kind == "mlp":
        model = MultiOutputMLP(context_length=L, horizon=H, hidden_size=config.hidden_size, input_size=inputs)
        if config.dropout:
            model.network.insert(2, nn.Dropout(config.dropout))
    elif config.kind == "dlinear":
        model = DLinear(context_length=L, horizon=H)
    elif config.kind == "nlinear":
        model = NLinear(context_length=L, horizon=H)
    elif config.kind == "lstm":
        model = LSTMForecaster(input_size=inputs, horizon=H, hidden_size=config.hidden_size, num_layers=config.num_layers)
    elif config.kind == "lstm_ar":
        model = CalendarAutoregressiveLSTM(
            input_size=inputs, horizon=H, hidden_size=config.hidden_size, num_layers=config.num_layers
        )
    else:
        model = Seq2SeqForecaster(
            input_size=inputs, decoder_covariate_size=covariate_count, horizon=H,
            hidden_size=config.hidden_size, num_layers=config.num_layers,
        )
    # nn.LSTM lee su atributo dropout en cada forward; solo actúa entre capas apiladas.
    for module in model.modules():
        if isinstance(module, nn.LSTM) and module.num_layers > 1:
            module.dropout = float(config.dropout)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


class PanelTensors:
    """Panel ancho en kWh y calendario residentes en el dispositivo de entrenamiento."""

    def __init__(self, panel: pd.DataFrame, device: str | torch.device = "cpu") -> None:
        expected = pd.date_range(panel.index.min(), panel.index.max(), freq="h")
        if not panel.index.equals(expected):
            raise ValueError("El panel debe tener frecuencia horaria continua")
        self.panel = panel
        self.index = pd.DatetimeIndex(panel.index)
        self.clients = pd.Index(panel.columns.astype(str))
        self.device = torch.device(device)
        self.calendar_frame = make_calendar_features(self.index)
        self.values = torch.as_tensor(panel.to_numpy(np.float32), device=self.device)
        self.calendar = torch.as_tensor(self.calendar_frame.to_numpy(np.float32), device=self.device)
        # Escala MASE de cada origen: error naive estacional medio sobre toda la historia previa.
        values = panel.to_numpy(np.float64)
        absolute = np.abs(values[24:] - values[:-24])
        self._cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(absolute, axis=0)])

    def position(self, timestamp: str | pd.Timestamp) -> int:
        return int(self.index.get_loc(pd.Timestamp(timestamp)))

    def mase_scale(self, origin_position: int) -> np.ndarray:
        """Replica ``backtest._mase_scale`` con historia hasta ``origin_position - 1``."""

        count = origin_position - 24
        return self._cumulative[count] / count

    def subset(self, clients: list[str]) -> "PanelTensors":
        return PanelTensors(self.panel.loc[:, clients], self.device)


class Normalizer:
    """Normalización global, por serie (z-score de train) o por ventana reversible."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.loc_: Tensor | None = None
        self.scale_: Tensor | None = None
        self.standardizer_: SeriesStandardizer | None = None

    def fit(self, train: pd.DataFrame, device: torch.device) -> "Normalizer":
        if self.mode == "serie":
            self.standardizer_ = SeriesStandardizer().fit(train)
            loc, scale = self.standardizer_.mean_.to_numpy(), self.standardizer_.scale_.to_numpy()
        elif self.mode == "global":
            values = train.to_numpy(np.float64)
            loc = np.full(train.shape[1], values.mean())
            scale = np.full(train.shape[1], values.std() or 1.0)
        else:
            loc = np.zeros(train.shape[1])
            scale = np.ones(train.shape[1])
        self.loc_ = torch.as_tensor(loc, dtype=torch.float32, device=device)
        self.scale_ = torch.as_tensor(scale, dtype=torch.float32, device=device)
        return self

    def statistics(self, history: Tensor, columns: Tensor) -> tuple[Tensor, Tensor]:
        """Devuelve ubicación y escala ``(lote, 1)`` calculables en el origen."""

        if self.mode == "ventana":
            loc = history.mean(dim=1, keepdim=True)
            scale = history.std(dim=1, keepdim=True).clamp_min(1e-2)
            return loc, scale
        return self.loc_[columns][:, None], self.scale_[columns][:, None]


@dataclass
class Batch:
    inputs: Tensor
    future_covariates: Tensor
    target: Tensor | None
    loc: Tensor
    scale: Tensor


def make_batch(
    data: PanelTensors,
    normalizer: Normalizer,
    config: TrainingConfig,
    positions: Tensor,
    columns: Tensor,
    with_target: bool = True,
) -> Batch:
    """Construye ventanas ``(lote, L, atributos)`` desde posiciones de inicio del objetivo."""

    L, H = config.lookback, config.horizon
    history_index = positions[:, None] + torch.arange(-L, 0, device=positions.device)
    future_index = positions[:, None] + torch.arange(H, device=positions.device)
    history = data.values[history_index, columns[:, None]]
    loc, scale = normalizer.statistics(history, columns)
    scaled = (history - loc) / scale
    if config.kind in USES_COVARIATES:
        inputs = torch.cat([scaled[..., None], data.calendar[history_index]], dim=2)
    else:
        inputs = scaled
    target = (data.values[future_index, columns[:, None]] - loc) / scale if with_target else None
    return Batch(inputs, data.calendar[future_index], target, loc, scale)


def forward(model: nn.Module, config: TrainingConfig, batch: Batch) -> Tensor:
    if config.kind in {"seq2seq", "lstm_ar"}:
        return model(batch.inputs, batch.future_covariates)
    return model(batch.inputs)


@torch.no_grad()
def predict_positions(
    model: nn.Module,
    config: TrainingConfig,
    data: PanelTensors,
    normalizer: Normalizer,
    positions: np.ndarray,
    columns: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    """Predice en kWh para pares ``(posición de origen, columna)``."""

    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(positions), batch_size):
        p = torch.as_tensor(positions[start : start + batch_size], device=data.device)
        c = torch.as_tensor(columns[start : start + batch_size], device=data.device)
        batch = make_batch(data, normalizer, config, p, c, with_target=False)
        prediction = forward(model, config, batch) * batch.scale + batch.loc
        outputs.append(prediction.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def daily_origins(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")


def rolling_forecasts(
    predict: "callable",
    data: PanelTensors,
    origins: pd.DatetimeIndex,
    name: str,
    horizon: int = 24,
) -> pd.DataFrame:
    """Evalúa orígenes diarios con el mismo contrato de salida que ``backtest``.

    ``predict(positions, columns)`` devuelve ``(n, horizon)`` en kWh. Sirve para
    validación (julio-septiembre), donde ``backtest`` no aplica porque fija los
    orígenes de prueba.
    """

    n_clients = len(data.clients)
    origin_positions = np.array([data.position(origin) for origin in origins])
    positions = np.repeat(origin_positions, n_clients)
    columns = np.tile(np.arange(n_clients), len(origins))
    y_hat = predict(positions, columns)
    truth_index = positions[:, None] + np.arange(horizon)
    y = data.panel.to_numpy(np.float64)[truth_index, columns[:, None]]
    scales = np.stack([data.mase_scale(position) for position in origin_positions])
    return pd.DataFrame(
        {
            "unique_id": np.repeat(data.clients.to_numpy()[columns], horizon),
            "ds": data.index.to_numpy()[truth_index.reshape(-1)],
            "modelo": name,
            "y_hat": y_hat.reshape(-1),
            "y": y.reshape(-1),
            "origin": np.repeat(data.index.to_numpy()[positions], horizon),
            "horizon": np.tile(np.arange(1, horizon + 1), len(positions)),
            "mase_scale": np.repeat(scales.reshape(-1), horizon),
        }
    )


def forecast_errors(forecasts: pd.DataFrame) -> tuple[float, float]:
    """MAE en kWh y MASE con la escala por origen y cliente."""

    errors = np.abs(forecasts["y"].to_numpy() - forecasts["y_hat"].to_numpy())
    return float(errors.mean()), float(np.mean(errors / forecasts["mase_scale"].to_numpy()))


def history_start(config: TrainingConfig, experiment: ExperimentConfig, train_end: pd.Timestamp) -> pd.Timestamp:
    """Primer instante usable: ``history_months`` antes del fin de entrenamiento."""

    start = (train_end + pd.Timedelta(1, unit="h")) - pd.DateOffset(months=config.history_months)
    return max(pd.Timestamp(start), pd.Timestamp(experiment.data_start))


@dataclass
class TrainingRun:
    config: TrainingConfig
    model: nn.Module
    normalizer: Normalizer
    data: PanelTensors
    history: pd.DataFrame
    best_epoch: int
    validation_forecasts: pd.DataFrame | None
    training_seconds: float
    peak_gpu_memory_mb: float

    @property
    def parameters(self) -> tuple[int, int]:
        return count_parameters(self.model)

    def summary(self) -> dict[str, object]:
        best = self.history.loc[self.history["epoch"].eq(self.best_epoch)].iloc[0]
        config = asdict(self.config)
        config.pop("clients")
        return {
            **config,
            "modelo": self.config.label,
            "configuracion": self.config.describe(),
            "best_epoch": self.best_epoch,
            "epochs_run": int(self.history["epoch"].max()),
            "validation_mae": float(best["validation_mae"]),
            "validation_mase": float(best["validation_mase"]),
            "total_parameters": int(best["total_parameters"]),
            "trainable_parameters": int(best["trainable_parameters"]),
            "epoch_seconds_mean": float(self.history["epoch_seconds"].mean()),
            "training_seconds": self.training_seconds,
            "inference_seconds": float(best["inference_seconds"]),
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }


def train_model(
    config: TrainingConfig,
    panel: pd.DataFrame | PanelTensors,
    *,
    experiment: ExperimentConfig | None = None,
    train_end: str | pd.Timestamp | None = None,
    validation_end: str | pd.Timestamp | None = None,
    fixed_epochs: int | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> TrainingRun:
    """Entrena con L1 en escala normalizada y early stopping por MASE de validación.

    Con ``fixed_epochs`` no hay validación: se entrena ese número de epochs sobre
    todo el bloque hasta ``train_end`` (reentrenamiento mensual).
    """

    experiment = experiment or ExperimentConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = panel if isinstance(panel, PanelTensors) else PanelTensors(panel, device)
    if config.clients is not None:
        data = data.subset(list(config.clients))
    train_end = pd.Timestamp(train_end or experiment.train_end)
    validation_end = pd.Timestamp(validation_end or experiment.validation_end)
    start = history_start(config, experiment, train_end)

    set_seed(config.seed)
    normalizer = Normalizer(config.normalization).fit(data.panel.loc[start:train_end], data.device)
    # La primera entrada debe caer dentro de la historia permitida.
    first_target = max(data.position(start) + config.lookback, config.lookback)
    last_target = data.position(train_end) - config.horizon + 1
    target_positions = torch.arange(first_target, last_target + 1, config.train_stride, device=data.device)
    n_clients = len(data.clients)
    pairs = torch.cartesian_prod(target_positions, torch.arange(n_clients, device=data.device))

    model = build_model(config).to(data.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.L1Loss()
    total, trainable = count_parameters(model)
    validation_origins = daily_origins(train_end + pd.Timedelta(1, unit="h"), validation_end - pd.Timedelta(23, unit="h"))
    if data.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(data.device)

    rows: list[dict[str, object]] = []
    best_score, best_epoch, best_state, stale = float("inf"), 0, None, 0
    best_forecasts: pd.DataFrame | None = None
    started = time.perf_counter()
    epochs = fixed_epochs or config.max_epochs
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(pairs), device=data.device)
        running, batches = torch.zeros((), device=data.device), 0
        for begin in range(0, len(order), config.batch_size):
            chosen = pairs[order[begin : begin + config.batch_size]]
            batch = make_batch(data, normalizer, config, chosen[:, 0], chosen[:, 1])
            loss = loss_fn(forward(model, config, batch), batch.target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.detach()
            batches += 1
        train_loss = float(running / batches)
        row: dict[str, object] = {
            "modelo": config.label, "configuracion": config.describe(), "L": config.lookback,
            "H": config.horizon, "history_months": config.history_months, "epoch": epoch,
            "train_loss": train_loss, "total_parameters": total, "trainable_parameters": trainable,
            "seed": config.seed,
        }
        if fixed_epochs is None:
            inference_started = time.perf_counter()
            forecasts = rolling_forecasts(
                lambda p, c: predict_positions(model, config, data, normalizer, p, c),
                data, validation_origins, config.label, config.horizon,
            )
            row["inference_seconds"] = time.perf_counter() - inference_started
            row["validation_mae"], row["validation_mase"] = forecast_errors(forecasts)
            # Pérdida de validación en la misma escala normalizada que la de entrenamiento.
            origin_positions = np.array([data.position(o) for o in validation_origins])
            p = torch.as_tensor(np.repeat(origin_positions, n_clients), device=data.device)
            c = torch.as_tensor(np.tile(np.arange(n_clients), len(origin_positions)), device=data.device)
            with torch.no_grad():
                model.eval()
                batch = make_batch(data, normalizer, config, p, c)
                row["validation_loss"] = float(loss_fn(forward(model, config, batch), batch.target))
        row["epoch_seconds"] = time.perf_counter() - epoch_started
        row["training_seconds"] = time.perf_counter() - started
        row["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated(data.device) / 2**20 if data.device.type == "cuda" else 0.0
        )
        rows.append(row)
        if verbose:
            print(
                f"[{config.label}] epoch {epoch:02d} train={train_loss:.4f} "
                f"val_mase={row.get('validation_mase', float('nan')):.4f} ({row['epoch_seconds']:.1f}s)",
                flush=True,
            )
        if fixed_epochs is not None:
            continue
        if row["validation_mase"] < best_score - 1e-4:
            best_score, best_epoch, stale = row["validation_mase"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            best_forecasts = forecasts
        else:
            stale += 1
            if stale >= config.patience:
                break
    training_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    history = pd.DataFrame(rows).reindex(columns=ITERATION_COLUMNS)
    peak = float(history["peak_gpu_memory_mb"].max())
    return TrainingRun(config, model, normalizer, data, history, best_epoch, best_forecasts, training_seconds, peak)


class TorchBacktestModel:
    """Expone una red entrenada con el protocolo ``ForecastModel`` de ``backtest``."""

    def __init__(self, run: TrainingRun, name: str | None = None) -> None:
        self.run = run
        self.name = name or run.config.label
        self.column_of = {client: position for position, client in enumerate(run.data.clients)}

    def predict(self, history: pd.DataFrame, horizon: int, future_covariates: pd.DataFrame) -> pd.Series:
        config = self.run.config
        if horizon != config.horizon:
            raise ValueError("El horizonte debe coincidir con el entrenado")
        tail = history.groupby("unique_id", sort=False).tail(config.lookback)
        wide = tail.pivot(index="ds", columns="unique_id", values="y")
        if len(wide) != config.lookback or wide.isna().any().any():
            raise ValueError("La historia no cubre la ventana L de todos los clientes")
        origin = wide.index.max() + pd.Timedelta(1, unit="h")
        clients = [str(client) for client in wide.columns]
        device = self.run.data.device
        hours = pd.date_range(wide.index.min(), periods=config.lookback + horizon, freq="h")
        calendar = torch.as_tensor(make_calendar_features(hours).to_numpy(np.float32), device=device)
        values = torch.as_tensor(wide.to_numpy(np.float32), device=device)
        columns = torch.as_tensor([self.column_of[client] for client in clients], device=device)
        model = self.run.model.eval()
        with torch.no_grad():
            observed = values.T
            loc, scale = self.run.normalizer.statistics(observed, columns)
            scaled = (observed - loc) / scale
            if config.kind in USES_COVARIATES:
                inputs = torch.cat([scaled[..., None], calendar[: config.lookback].expand(len(clients), -1, -1)], dim=2)
            else:
                inputs = scaled
            future = calendar[config.lookback :].expand(len(clients), -1, -1)
            batch = Batch(inputs, future, None, loc, scale)
            prediction = (forward(model, config, batch) * scale + loc).float().cpu().numpy()
        index = pd.MultiIndex.from_product(
            [clients, pd.date_range(origin, periods=horizon, freq="h")], names=["unique_id", "ds"]
        )
        return pd.Series(prediction.reshape(-1), index=index, name="y_hat").reindex(future_covariates.index)
