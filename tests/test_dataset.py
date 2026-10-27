from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from forecasting_lab.dataset import PanelWindowDataset, SeriesStandardizer, count_windows
from forecasting_lab.features import make_calendar_features


def _panel(periods: int = 12) -> pd.DataFrame:
    index = pd.date_range("2014-01-01", periods=periods, freq="h")
    return pd.DataFrame({"a": np.arange(periods), "b": np.arange(periods) + 10}, index=index)


def test_standardizer_uses_supplied_training_values_and_handles_constant_series() -> None:
    train = pd.DataFrame({"a": [0.0, 2.0], "constant": [5.0, 5.0]})
    later = pd.DataFrame({"a": [100.0], "constant": [5.0]})

    scaler = SeriesStandardizer().fit(train)
    transformed = scaler.transform(later)

    assert scaler.mean_["a"] == 1.0
    assert scaler.scale_["constant"] == 1.0
    assert transformed.loc[0, "a"] == 99.0
    assert transformed.loc[0, "constant"] == 0.0


def test_standardizer_records_the_actual_fit_interval() -> None:
    train = _panel(8)

    scaler = SeriesStandardizer().fit(train)

    assert scaler.fit_start_ == train.index.min()
    assert scaler.fit_end_ == train.index.max()


def test_window_dataset_has_batch_first_sample_shapes_and_aligned_future_covariates() -> None:
    panel = _panel()
    features = make_calendar_features(panel.index)
    dataset = PanelWindowDataset(
        panel,
        features,
        lookback=4,
        horizon=2,
        target_start=panel.index[6],
        target_end=panel.index[9],
    )

    x, future, y, metadata = dataset[0]

    assert x.shape == (4, 1 + features.shape[1])
    assert future.shape == (2, features.shape[1])
    assert y.shape == (2,)
    assert x.dtype == future.dtype == y.dtype == torch.float32
    assert metadata["target_start"] == panel.index[6]


def test_window_targets_stay_inside_split_and_counts_are_exact() -> None:
    panel = _panel()
    features = make_calendar_features(panel.index)
    dataset = PanelWindowDataset(
        panel,
        features,
        lookback=4,
        horizon=2,
        target_start=panel.index[6],
        target_end=panel.index[9],
    )

    assert len(dataset) == 6  # 3 orígenes por serie × 2 series
    assert count_windows(panel, 4, 2, panel.index[6], panel.index[9]) == 6
    assert all(panel.index[6] <= item.target_start <= panel.index[8] for item in dataset.windows)


def test_window_dataset_rejects_misaligned_covariates() -> None:
    panel = _panel()
    features = make_calendar_features(panel.index[:-1])

    with pytest.raises(ValueError, match="alineadas"):
        PanelWindowDataset(panel, features, lookback=4, horizon=2)


def test_count_windows_rejects_an_hourly_gap() -> None:
    panel = _panel().drop(pd.Timestamp("2014-01-01 05:00"))

    with pytest.raises(ValueError, match="continua"):
        count_windows(panel, 4, 2, panel.index[4], panel.index[-1])
