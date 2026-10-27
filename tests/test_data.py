from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_lab.config import ExperimentConfig
from forecasting_lab.data import (
    aggregate_quarter_hourly,
    filter_complete_clients,
    select_clients,
    split_panel,
)


def test_aggregate_quarter_hourly_sums_four_kw_measurements_into_kwh() -> None:
    # UCI marca el final de cada intervalo: 00:15–01:00 forman la hora 00:00.
    index = pd.date_range("2012-03-25 00:15", periods=96, freq="15min")
    raw = pd.DataFrame({"MT_001": np.full(96, 2.0)}, index=index)

    hourly = aggregate_quarter_hourly(raw)

    assert len(hourly) == 24
    assert hourly.index[0] == pd.Timestamp("2012-03-25 00:00")
    assert hourly.index[-1] == pd.Timestamp("2012-03-25 23:00")
    assert hourly["MT_001"].tolist() == [2.0] * 24


def test_aggregate_rejects_an_hour_without_exactly_four_quarters() -> None:
    index = pd.date_range("2012-01-01", periods=7, freq="15min")
    raw = pd.DataFrame({"MT_001": np.ones(7)}, index=index)

    with pytest.raises(ValueError, match="cuatro observaciones"):
        aggregate_quarter_hourly(raw)


def test_aggregate_rejects_a_completely_missing_civil_hour() -> None:
    index = pd.date_range("2012-03-25 00:15", periods=96, freq="15min")
    raw = pd.DataFrame({"MT_001": np.ones(96)}, index=index)
    raw = raw.drop(pd.date_range("2012-03-25 02:15", periods=4, freq="15min"))

    with pytest.raises(ValueError, match="continua"):
        aggregate_quarter_hourly(raw)


def test_filter_complete_clients_excludes_late_activation_and_missing_values() -> None:
    index = pd.date_range("2011-12-31", periods=12, freq="6h")
    frame = pd.DataFrame(
        {
            "complete": [1.0] * 12,
            "late": [0.0] * 6 + [2.0] * 6,
            "preperiod_missing": [1.0, np.nan] + [1.0] * 10,
            "missing_after": [1.0] * 5 + [np.nan] + [1.0] * 6,
        },
        index=index,
    )

    filtered = filter_complete_clients(frame, start="2012-01-01")

    assert filtered.columns.tolist() == ["complete", "preperiod_missing"]
    assert filtered.index.min() == pd.Timestamp("2012-01-01")


def test_select_clients_is_deterministic_and_validates_minimum() -> None:
    panel = pd.DataFrame(np.ones((2, 105)), columns=[f"c{i:03d}" for i in range(105)])

    first = select_clients(panel, n_clients=100, seed=3092)
    second = select_clients(panel, n_clients=100, seed=3092)

    assert first.columns.tolist() == second.columns.tolist()
    assert len(first.columns) == 100
    with pytest.raises(ValueError, match="100"):
        select_clients(panel, n_clients=99, seed=3092)


def test_split_panel_uses_exact_non_overlapping_boundaries() -> None:
    idx = pd.date_range("2012-01-01 00:00", "2014-12-31 23:00", freq="h")
    panel = pd.DataFrame({"MT_001": np.arange(len(idx), dtype=float)}, index=idx)
    config = ExperimentConfig()

    splits = split_panel(panel, config)

    assert splits.train.index.max() == pd.Timestamp("2014-06-30 23:00")
    assert splits.validation.index.min() == pd.Timestamp("2014-07-01 00:00")
    assert splits.validation.index.max() == pd.Timestamp("2014-09-30 23:00")
    assert splits.test.index.min() == pd.Timestamp("2014-10-01 00:00")
    assert splits.test.index.max() == pd.Timestamp("2014-12-31 23:00")

    with pytest.raises(ValueError, match="continuo"):
        split_panel(panel.drop(pd.Timestamp("2014-08-01 05:00")), config)
