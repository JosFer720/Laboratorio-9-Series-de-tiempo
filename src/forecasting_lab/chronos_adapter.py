"""Adaptador diferido para la predicción de cuantiles de Chronos-Bolt."""

from __future__ import annotations

import importlib
from typing import Iterable

import numpy as np
import torch


class ChronosBoltAdapter:
    """Llama a ``BaseChronosPipeline.predict_quantiles`` sin importar Chronos al inicio."""

    def __init__(self, pipeline=None, model_id: str = "amazon/chronos-bolt-small", quantile_levels=(0.1, 0.5, 0.9)):
        self.pipeline = pipeline
        self.model_id = model_id
        self.quantile_levels = tuple(quantile_levels)
        if not self.quantile_levels or any(level < 0 or level > 1 for level in self.quantile_levels):
            raise ValueError("quantile_levels must be non-empty values between 0 and 1")

    def _pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        try:
            module = importlib.import_module("chronos")
            pipeline_cls = module.BaseChronosPipeline
        except (ImportError, AttributeError) as exc:  # pragma: no cover - depende del entorno local
            raise ImportError("Chronos-Bolt support requires `pip install chronos-forecasting`.") from exc
        self.pipeline = pipeline_cls.from_pretrained(self.model_id)
        return self.pipeline

    def predict(self, context: Iterable[float], prediction_length: int):
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        values = np.asarray(context, dtype=float)
        if values.ndim not in {1, 2} or values.size == 0:
            raise ValueError("context must be a non-empty one- or two-dimensional array")
        # Chronos 2.x valida explícitamente que el lote sea un tensor.
        tensor = torch.as_tensor(values, dtype=torch.float32)
        return self._pipeline().predict_quantiles(
            inputs=tensor,
            prediction_length=prediction_length,
            quantile_levels=list(self.quantile_levels),
        )
