import numpy as np
import pandas as pd

from stplfcnn import QuantileLevels, TimeHorizon
from stplfcnn.estimators.persistence1 import Persistence1Estimator


def test_horizon_6h_resolution_1h():
    data = pd.DataFrame(
        data=[(i - i % 4) / 16 for i in range(10 * 24 * 4)],
        index=pd.DatetimeIndex(freq="15min", start="2018-01-01", periods=10 * 24 * 4),
        columns=["load"],
    )
    quantile_levels = [25, 50, 75]
    estimator = Persistence1Estimator(
        time_horizon=TimeHorizon(0, 6 * 60, 60),
        quantile_levels=QuantileLevels(quantile_levels),
    )

    issue_times = pd.date_range("2018-01-01 00", "2018-01-10 00", freq="1d")
    expected = np.percentile([i for i in range(10 * 24) if 0 <= i % 24 < 6], quantile_levels)
    train_and_check(estimator, data, issue_times, expected)

    issue_times = pd.date_range("2018-01-03 07", "2018-01-08 07", freq="1d")
    expected = np.percentile([i for i in range(2 * 24, 8 * 24) if 7 <= i % 24 < 13], quantile_levels)
    train_and_check(estimator, data, issue_times, expected)

    issue_times = pd.date_range("2018-01-01 00", "2018-01-10 18", freq="6h")
    expected = np.percentile(range(10 * 24), quantile_levels)
    train_and_check(estimator, data, issue_times, expected)


def train_and_check(estimator, data, issue_times, expected):
    estimator.train(data, issue_times)
    load_predictions_per_issue_time = estimator.predict(data, issue_times)

    assert np.all(estimator.quantile_forecasts == expected)
    assert list(load_predictions_per_issue_time.index.get_level_values("issue_time").unique()) == list(issue_times)
    assert (list(load_predictions_per_issue_time.index.get_level_values("offset").unique())
            == list(estimator.time_horizon.deltas))
    assert np.all(load_predictions_per_issue_time.groupby("issue_time").count() == len(estimator.time_horizon))
    assert np.all(load_predictions_per_issue_time.groupby("offset").count() == len(issue_times))
    for issue_time in issue_times:
        load_predictions = load_predictions_per_issue_time.loc[issue_time]
        assert load_predictions.shape == (len(estimator.time_horizon), len(estimator.quantile_levels))
        assert np.all(load_predictions.columns == estimator.quantile_levels.numerators)
        assert np.all(load_predictions == expected)
