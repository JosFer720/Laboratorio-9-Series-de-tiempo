from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from forecasting_lab.results import (
    build_comparison_table,
    plot_history_sensitivity,
    plot_horizon_mae,
    plot_pareto,
    write_predictions,
)


def test_write_predictions_enforces_exact_schema_and_stable_order(tmp_path: Path) -> None:
    forecasts = pd.DataFrame(
        {
            "modelo": ["z", "a"],
            "y_hat": [2.0, 1.0],
            "ds": pd.to_datetime(["2014-10-01 01:00", "2014-10-01 00:00"]),
            "unique_id": ["b", "a"],
            "y": [2.5, 1.5],
        }
    )

    output = write_predictions(forecasts, tmp_path / "pronosticos.csv")
    saved = pd.read_csv(output)

    assert saved.columns.tolist() == ["unique_id", "ds", "modelo", "y_hat"]
    assert saved["unique_id"].tolist() == ["a", "b"]


def test_write_predictions_rejects_missing_contract_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="y_hat"):
        write_predictions(
            pd.DataFrame({"unique_id": ["a"], "ds": ["2014-10-01"], "modelo": ["m"]}),
            tmp_path / "pronosticos.csv",
        )


def test_comparison_table_is_ordered_by_mase() -> None:
    runs = pd.DataFrame(
        {
            "modelo": ["lento", "mejor"],
            "familia": ["red", "baseline"],
            "mase": [1.2, 0.8],
        }
    )

    result = build_comparison_table(runs)

    assert result["modelo"].tolist() == ["mejor", "lento"]


def test_required_plots_create_nonempty_files(tmp_path: Path) -> None:
    runs = pd.DataFrame(
        {
            "modelo": ["naive", "lightgbm"],
            "mase": [1.0, 0.8],
            "training_seconds": [0.01, 2.0],
        }
    )
    horizon = pd.DataFrame(
        {
            "modelo": ["naive"] * 2 + ["lightgbm"] * 2,
            "horizon": [1, 2, 1, 2],
            "mae": [1.0, 1.2, 0.8, 1.0],
        }
    )
    history = pd.DataFrame(
        {
            "modelo": ["naive", "naive", "chronos", "chronos"],
            "history_months": [3, 12, 3, 12],
            "mase": [1.1, 1.0, 0.9, 0.9],
        }
    )

    paths = [
        plot_pareto(runs, tmp_path / "pareto.png"),
        plot_horizon_mae(horizon, tmp_path / "horizonte.png"),
        plot_history_sensitivity(history, tmp_path / "historia.png"),
    ]

    assert all(path.exists() and path.stat().st_size > 0 for path in paths)

