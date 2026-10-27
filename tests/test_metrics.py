import math

import numpy as np
import pandas as pd
import pytest

from forecasting_lab.metrics import (
    aggregate_metrics,
    christmas_week_metrics,
    hour_of_day_metrics,
    mae,
    mape,
    mase,
    per_client_metrics,
    per_horizon_metrics,
    rmse,
    wape,
)


def test_metrics_return_hand_calculated_values_in_original_kwh():
    """Debe fallar si cambia una fórmula de error absoluto, cuadrático o porcentual."""
    actual = [100.0, 200.0]
    predicted = [110.0, 180.0]

    assert mae(actual, predicted) == pytest.approx(15.0)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(250.0))
    assert wape(actual, predicted) == pytest.approx(0.1)
    assert mape(actual, predicted) == pytest.approx(0.1)


def test_zero_percentage_and_scale_denominators_return_nan_without_epsilon():
    """Debe fallar si se reemplaza un denominador cero con epsilon."""
    assert math.isnan(wape([0.0, 0.0], [1.0, 2.0]))
    assert math.isnan(mape([0.0, 0.0], [1.0, 2.0]))
    assert math.isnan(mase([1.0, 1.0], [2.0, 0.0], history=[5.0, 5.0, 5.0], seasonal_period=1))


def test_mase_uses_hand_calculated_seasonal_naive_scale():
    """Debe fallar si usa errores de pronóstico en vez de la escala ingenua estacional."""
    actual = [12.0, 9.0]
    predicted = [10.0, 11.0]
    history = [2.0, 5.0, 4.0, 8.0]

    assert mase(actual, predicted, history, seasonal_period=2) == pytest.approx(0.8)


def test_grouped_summaries_cover_client_horizon_hour_and_christmas_week():
    """Debe fallar si omite una clave de agrupación o un límite de semana navideña."""
    rows = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(
                ["2014-12-21 23:00", "2014-12-22 00:00", "2014-12-28 23:00", "2014-12-29 00:00"]
            ),
            "horizon": [1, 2, 1, 2],
            "y": [10.0, 20.0, 30.0, 40.0],
            "y_hat": [8.0, 24.0, 27.0, 44.0],
            "mase_scale": [2.0, 4.0, 3.0, 4.0],
        }
    )

    assert aggregate_metrics(rows).iloc[0]["mae"] == pytest.approx(3.25)
    assert per_client_metrics(rows).set_index("unique_id").loc["a", "mae"] == pytest.approx(3.0)
    assert per_client_metrics(rows).set_index("unique_id").loc["a", "mase"] == pytest.approx(1.0)
    assert per_horizon_metrics(rows).set_index("horizon").loc[1, "mae"] == pytest.approx(2.5)
    assert per_horizon_metrics(rows).set_index("horizon").loc[1, "mase"] == pytest.approx(1.0)
    assert hour_of_day_metrics(rows).set_index("hour").loc[0, "mae"] == pytest.approx(4.0)
    assert hour_of_day_metrics(rows).set_index("hour").loc[0, "mase"] == pytest.approx(1.0)

    christmas = christmas_week_metrics(rows)
    assert christmas["ds"].dt.date.tolist() == [pd.Timestamp("2014-12-22").date(), pd.Timestamp("2014-12-28").date()]
    assert christmas["mae"].tolist() == pytest.approx([4.0, 3.0])
    assert christmas["mase"].tolist() == pytest.approx([1.0, 1.0])


def test_mase_with_row_scales_is_mean_of_scaled_absolute_errors():
    """Debe fallar si divide MAE por la media de escalas heterogéneas."""
    rows = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "horizon": [1, 2],
            "y": [2.0, 2.0],
            "y_hat": [0.0, 0.0],
            "mase_scale": [1.0, 100.0],
        }
    )

    assert aggregate_metrics(rows).iloc[0]["mase"] == pytest.approx(1.01)
    assert per_client_metrics(rows).iloc[0]["mase"] == pytest.approx(1.01)
    assert per_horizon_metrics(rows).set_index("horizon")["mase"].to_dict() == pytest.approx({1: 2.0, 2: 0.02})


@pytest.mark.parametrize("invalid_scale", [0.0, np.nan])
def test_invalid_row_scale_makes_affected_mase_summary_nan(invalid_scale):
    """Debe fallar si una escala cero o no numérica se omite silenciosamente."""
    rows = pd.DataFrame(
        {
            "unique_id": ["valid", "invalid"],
            "y": [2.0, 2.0],
            "y_hat": [0.0, 0.0],
            "mase_scale": [1.0, invalid_scale],
        }
    )

    assert math.isnan(aggregate_metrics(rows).iloc[0]["mase"])
    grouped = per_client_metrics(rows).set_index("unique_id")
    assert grouped.loc["valid", "mase"] == pytest.approx(2.0)
    assert math.isnan(grouped.loc["invalid", "mase"])


def test_empty_grouped_summaries_keep_stable_schemas():
    """Debe fallar si un resumen vacío pierde sus columnas canónicas."""
    rows = pd.DataFrame(columns=["unique_id", "ds", "horizon", "y", "y_hat", "mase_scale"])
    metrics = ["mae", "rmse", "wape", "mape", "mase"]

    assert aggregate_metrics(rows).columns.tolist() == metrics
    assert per_client_metrics(rows).columns.tolist() == ["unique_id", *metrics]
    assert per_horizon_metrics(rows).columns.tolist() == ["horizon", *metrics]
    assert hour_of_day_metrics(rows).columns.tolist() == ["hour", *metrics]
    assert christmas_week_metrics(rows).columns.tolist() == ["ds", *metrics]
