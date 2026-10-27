from __future__ import annotations

import pandas as pd
import pytest

from forecasting_lab.config import ExperimentConfig
from forecasting_lab.dataset import SeriesStandardizer
from forecasting_lab.features import make_calendar_features
from forecasting_lab.leakage import audit_prepared_pipeline, run_leakage_audit


def test_leakage_audit_proves_all_four_guards() -> None:
    report = run_leakage_audit(
        train_end=pd.Timestamp("2014-06-30 23:00"),
        validation_end=pd.Timestamp("2014-09-30 23:00"),
        scaler_fit_end=pd.Timestamp("2014-06-30 23:00"),
        early_stopping_end=pd.Timestamp("2014-09-30 23:00"),
        origin=pd.Timestamp("2014-10-01 00:00"),
        history_end=pd.Timestamp("2014-09-30 23:00"),
        future_covariate_end=pd.Timestamp("2014-10-01 23:00"),
        covariates_known_in_advance=True,
    )

    assert report == {
        "temporal_split": True,
        "train_only_scaling": True,
        "validation_only_early_stopping": True,
        "aligned_known_future_covariates": True,
    }


def test_pipeline_audit_reads_real_splits_scaler_and_covariate_index() -> None:
    config = ExperimentConfig(
        data_start="2014-01-01 00:00",
        train_end="2014-01-02 23:00",
        validation_start="2014-01-03 00:00",
        validation_end="2014-01-03 23:00",
        test_start="2014-01-04 00:00",
        test_end="2014-01-04 23:00",
    )
    index = pd.date_range(config.data_start, config.test_end, freq="h")
    panel = pd.DataFrame({"a": range(len(index))}, index=index, dtype=float)
    train = panel.loc[: config.train_end]
    validation = panel.loc[config.validation_start : config.validation_end]
    scaler = SeriesStandardizer().fit(train)
    features = make_calendar_features(index)

    report = audit_prepared_pipeline(panel, train, validation, scaler, features, config)

    assert all(report.values())
    scaler.fit_end_ = pd.Timestamp(config.validation_start)
    with pytest.raises(ValueError, match="train_only_scaling"):
        audit_prepared_pipeline(panel, train, validation, scaler, features, config)
