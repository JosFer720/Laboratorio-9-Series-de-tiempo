import pandas as pd
import pytest

from forecasting_lab.backtest import backtest


class SpyModel:
    """Registra cada límite de historia recibido y devuelve un pronóstico constante."""

    name = "spy"

    def __init__(self):
        self.calls = []

    def predict(self, history, horizon, future_covariates):
        self.calls.append((history.copy(), horizon, future_covariates.copy()))
        future_index = future_covariates.index
        return pd.Series(1.0, index=future_index, name="y_hat")


def complete_data():
    """Crea datos horarios completos que cubren todos los orígenes del backtest."""
    timestamps = pd.date_range("2014-09-29", "2014-12-31 23:00", freq="h")
    return pd.DataFrame(
        {
            "unique_id": "client-1",
            "ds": timestamps,
            "y": range(len(timestamps)),
            "temperature": 25.0,
            "y_hat": -1.0,
            "origin": pd.Timestamp("2014-09-29"),
            "horizon": 1,
            "modelo": "dato-no-covariable",
        }
    )


def test_backtest_has_92_daily_origins_and_never_leaks_future_history():
    """Debe fallar si se pasan filas de la fecha de origen o posteriores a la historia."""
    timestamps = pd.date_range("2014-09-30", "2015-01-01 23:00", freq="h")
    client_one = pd.DataFrame(
        {
            "unique_id": "client-1",
            "ds": timestamps,
            "y": range(len(timestamps)),
            "temperature": 25.0,
        }
    )
    client_two = client_one.assign(unique_id="client-2", y=range(10_000, 10_000 + len(timestamps)))
    data = pd.concat([client_one, client_two], ignore_index=True)
    model = SpyModel()

    result = backtest(model, data, future_covariate_columns=["temperature"])

    assert len(model.calls) == 92
    assert len(result) == 92 * 24 * 2
    assert {"unique_id", "ds", "modelo", "y_hat", "y"}.issubset(result.columns)
    assert set(result["unique_id"]) == {"client-1", "client-2"}
    assert result["ds"].min() == pd.Timestamp("2014-10-01")
    assert result["ds"].max() == pd.Timestamp("2014-12-31 23:00")
    assert result["y_hat"].eq(1.0).all()
    assert result["y"].iloc[0] == 24

    for origin, (history, horizon, future_covariates) in zip(
        pd.date_range("2014-10-01", "2014-12-31", freq="D"), model.calls
    ):
        assert history["ds"].max() == origin - pd.Timedelta(1, unit="h")
        assert history["ds"].lt(origin).all()
        assert horizon == 24
        assert future_covariates.index.names == ["unique_id", "ds"]
        assert future_covariates.index.get_level_values("ds").unique().tolist() == list(
            pd.date_range(origin, periods=24, freq="h")
        )


def test_backtest_accepts_valid_timestamps_with_microsecond_resolution():
    """La resolución física del dtype no debe invalidar un panel horario correcto."""
    data = complete_data()
    data["ds"] = data["ds"].astype("datetime64[us]")

    result = backtest(SpyModel(), data, future_covariate_columns=["temperature"])

    assert len(result) == 92 * 24


@pytest.mark.parametrize("forbidden", ["y", "y_hat", "unique_id", "ds", "origin", "horizon", "modelo"])
def test_backtest_rejects_target_and_reserved_columns_as_future_covariates(forbidden):
    """Debe fallar si la verdad futura o una columna reservada llega al modelo."""
    model = SpyModel()

    with pytest.raises(ValueError, match="reserv"):
        backtest(model, complete_data(), future_covariate_columns=[forbidden])

    assert model.calls == []


class InvalidPredictionModel:
    """Devuelve deliberadamente una predicción que viola el contrato."""

    def __init__(self, mode):
        self.mode = mode

    def predict(self, history, horizon, future_covariates):
        if self.mode == "sin_indice":
            return [1.0] * len(future_covariates)
        value = float("nan") if self.mode == "nan" else float("inf")
        return pd.Series(value, index=future_covariates.index)


@pytest.mark.parametrize("mode", ["sin_indice", "nan", "inf"])
def test_backtest_rejects_predictions_without_multiindex_or_with_nonfinite_values(mode):
    """Debe fallar si el modelo rompe el índice o devuelve NaN o infinito."""
    with pytest.raises(ValueError, match="prediction"):
        backtest(InvalidPredictionModel(mode), complete_data(), future_covariate_columns=["temperature"])


@pytest.mark.parametrize("defect", ["duplicate", "gap"])
def test_backtest_rejects_non_hourly_client_history_before_seasonal_lag(defect):
    """Debe fallar si la historia de un cliente contiene duplicados o huecos horarios."""
    data = complete_data()
    defective_timestamp = pd.Timestamp("2014-09-30 12:00")
    row = data.loc[data["ds"].eq(defective_timestamp)]
    if defect == "duplicate":
        data = pd.concat([data, row], ignore_index=True)
    else:
        data = data.loc[~data["ds"].eq(defective_timestamp)].copy()

    with pytest.raises(ValueError, match="historia horaria"):
        backtest(SpyModel(), data, future_covariate_columns=["temperature"])
