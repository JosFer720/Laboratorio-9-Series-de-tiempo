"""Pruebas del entrenamiento propio y su equivalencia con el backtest compartido."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from forecasting_lab.backtest import backtest  # noqa: E402
from forecasting_lab.nixtla_training import NixtlaRun  # noqa: E402
from forecasting_lab.training import (  # noqa: E402
    CalendarAutoregressiveLSTM,
    Normalizer,
    PanelTensors,
    TorchBacktestModel,
    TrainingConfig,
    build_model,
    daily_origins,
    make_batch,
    predict_positions,
    rolling_forecasts,
    train_model,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    index = pd.date_range("2012-01-01", "2014-12-31 23:00", freq="h")
    rng = np.random.default_rng(0)
    daily = np.sin(2 * np.pi * index.hour.to_numpy() / 24)
    return pd.DataFrame(
        {
            "MT_A": 10 + 3 * daily + rng.normal(0, 0.3, len(index)),
            "MT_B": 200 + 50 * daily + rng.normal(0, 5, len(index)),
        },
        index=index,
    ).rename_axis("ds")


@pytest.fixture(scope="module")
def trained(panel):
    config = TrainingConfig("dlinear", max_epochs=1, train_stride=48, batch_size=256)
    return train_model(config, panel, device="cpu", verbose=False)


def test_fast_rolling_forecasts_match_shared_backtest(panel, trained):
    """El evaluador vectorizado debe reproducir pronósticos y escala MASE de ``backtest``."""

    long = panel.reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y")
    reference = backtest(TorchBacktestModel(trained), long)
    fast = rolling_forecasts(
        lambda p, c: predict_positions(trained.model, trained.config, trained.data, trained.normalizer, p, c),
        trained.data, daily_origins("2014-10-01", "2014-12-31"), trained.config.label,
    )
    joined = reference.merge(fast, on=["unique_id", "ds"], suffixes=("", "_rapido"), validate="one_to_one")
    assert len(joined) == len(reference) == 2 * 92 * 24
    np.testing.assert_allclose(joined["y_hat"], joined["y_hat_rapido"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(joined["mase_scale"], joined["mase_scale_rapido"], rtol=1e-9)
    assert (joined["horizon"] == joined["horizon_rapido"]).all()


def test_training_log_has_one_row_per_epoch_with_shared_columns(trained):
    history = trained.history
    assert list(history["epoch"]) == [1]
    assert history[["train_loss", "validation_loss", "validation_mae", "validation_mase"]].notna().all().all()
    assert trained.summary()["total_parameters"] == sum(p.numel() for p in trained.model.parameters())


def test_training_windows_never_reach_validation(panel):
    """Con historia reducida, el normalizador se ajusta solo con esos meses de train."""

    config = TrainingConfig("nlinear", history_months=3, max_epochs=1, train_stride=24)
    run = train_model(config, panel, device="cpu", verbose=False)
    fitted = run.normalizer.standardizer_
    assert fitted.fit_start_ == pd.Timestamp("2014-04-01")
    assert fitted.fit_end_ == pd.Timestamp("2014-06-30 23:00")


def test_window_normalization_is_reversible(panel):
    data = PanelTensors(panel, "cpu")
    config = TrainingConfig("dlinear", normalization="ventana")
    normalizer = Normalizer("ventana").fit(panel.loc[:"2014-06-30 23:00"], data.device)
    positions = torch.tensor([data.position("2014-07-01")] * 2)
    batch = make_batch(data, normalizer, config, positions, torch.tensor([0, 1]))
    restored = batch.target * batch.scale + batch.loc
    expected = torch.as_tensor(panel.loc["2014-07-01"].to_numpy(np.float32).T)
    torch.testing.assert_close(restored, expected, rtol=1e-4, atol=1e-3)
    assert batch.inputs.mean(dim=1).abs().max() < 1e-4


def test_autoregressive_variant_reads_future_calendar():
    torch.manual_seed(0)
    model = CalendarAutoregressiveLSTM(input_size=3, horizon=4, hidden_size=8)
    history = torch.randn(2, 12, 3)
    future = torch.randn(2, 4, 2)
    first = model(history, future)
    changed = future.clone()
    changed[:, 1:, :] += 1.0
    assert first.shape == (2, 4)
    torch.testing.assert_close(first[:, :2], model(history, changed)[:, :2])
    assert not torch.allclose(first[:, 2:], model(history, changed)[:, 2:])


def test_dropout_is_applied_between_stacked_lstm_layers():
    model = build_model(TrainingConfig("lstm", num_layers=2, dropout=0.2))
    assert model.encoder.dropout == pytest.approx(0.2)


def test_nixtla_scaling_round_trip():
    run = NixtlaRun(
        "M", "f", None, False, 4, 30, pd.DataFrame(),
        pd.Series({"A": 10.0, "B": 100.0}), pd.Series({"A": 2.0, "B": 20.0}), 0.0, 0.0,
    )
    history = pd.DataFrame({"unique_id": ["A|2014-10-01", "B"], "ds": pd.Timestamp("2014-01-01"), "y": [12.0, 60.0]})
    assert run.scale_history(history)["y"].tolist() == [1.0, -2.0]
    raw = pd.DataFrame({"unique_id": ["A", "B"], "M": [1.0, -2.0], "M-lo-80": [0.0, 0.0]})
    restored = run.unscale_forecasts(raw)
    assert restored["M"].tolist() == [12.0, 60.0]
    assert restored["M-lo-80"].tolist() == [10.0, 100.0]
