from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_lab.features import make_calendar_features, portugal_holidays


def test_calendar_features_are_known_from_future_timestamp_only() -> None:
    index = pd.DatetimeIndex(["2014-12-25 00:00", "2014-12-25 06:00"])

    features = make_calendar_features(index)

    assert features.index.equals(index)
    assert features["is_holiday"].tolist() == [1.0, 1.0]
    assert np.isclose(features.loc[index[0], "hour_sin"], 0.0)
    assert np.isclose(features.loc[index[1], "hour_sin"], 1.0)
    assert set(features.columns) == {
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_holiday",
    }


def test_portugal_holidays_cover_christmas_and_new_year() -> None:
    holidays = portugal_holidays((2012, 2013, 2014))

    assert pd.Timestamp("2012-01-01") in holidays
    assert pd.Timestamp("2013-12-25") in holidays
    assert pd.Timestamp("2014-12-25") in holidays

