import numpy as np
import pandas as pd

from stplfcnn import QuantileLevels, TimeHorizon
from stplfcnn.estimators.persistence1 import Persistence1Estimator
from stplfcnn.estimators.persistence2 import Persistence2Estimator
from stplfcnn.measures import Measures


def test_horizon_6h_resolution_1h():
    data = pd.DataFrame(
        data=[(i - i % 4) / 16 for i in range(10 * 24 * 4)],
        index=pd.date_range("2018-01-01", freq="15min", periods=10 * 24 * 4),
        columns=["load"],
    )
    quantile_levels = QuantileLevels([25, 50, 75])

    # test on estimator1
    estimator1 = Persistence1Estimator(
        time_horizon=TimeHorizon(0, 6 * 60, 60),
        quantile_levels=quantile_levels,
    )
    issue_times = pd.date_range("2018-01-01 00", "2018-01-10 00", freq="1d")
    loss = train_and_get_loss(estimator1, data, issue_times)
    unique_rows = np.unique(loss, axis=0)
    assert unique_rows.shape == (6, 3)
    assert np.all(loss == loss.values[::-1, ::-1])

    # test on estimator2
    estimator2 = Persistence2Estimator(
        time_horizon=TimeHorizon(0, 6 * 60, 60),
        quantile_levels=quantile_levels,
    )
    issue_times = pd.date_range("2018-01-01 00", "2018-01-10 00", freq="1d")
    loss = train_and_get_loss(estimator2, data, issue_times)
    unique_rows = np.unique(loss, axis=0)
    assert unique_rows.shape == (1, 3)
    assert unique_rows[0, 0] == unique_rows[0, 2]


def train_and_get_loss(estimator, data, issue_times):
    estimator.train(data, issue_times)
    load_predictions_per_issue_time = estimator.predict(data, issue_times)

    return Measures(
        estimator.time_horizon,
        estimator.quantile_levels,
        data,
        load_predictions_per_issue_time,
    ).pinball_loss.groupby("offset").mean()


def test_calculate_pinball_loss():
    time_horizon = TimeHorizon(0, 6 * 60, 60)
    quantile_levels = QuantileLevels([25, 50, 75])
    data = pd.DataFrame(
        data=[50] * 12,
        index=pd.date_range("2018-01-01", freq="60min", periods=12),
        columns=["load"],
    )
    data2 = pd.DataFrame(
        data=[5, 10, 15, 20] * 12,
        index=pd.date_range("2018-01-01", freq="15min", periods=12 * 4),
        columns=["load"],
    )
    issue_time_index = np.repeat(pd.date_range("2018-01-01", freq="360 min", periods=2), len(time_horizon))
    offset_index = np.tile(time_horizon.deltas, 2)
    index = pd.MultiIndex.from_arrays(
        [issue_time_index, issue_time_index + offset_index, offset_index],
        names=["issue_time", "lead_time", "offset"],
    )
    predicted_loads_per_issue_time = pd.DataFrame(
        data=[
            [50, 50, 50],
            [20, 30, 40],
            [10, 25, 60],
            [40, 60, 90],
            [50, 70, 75],
            [70, 80, 100],

            [40, 40, 40],
            [10, 20, 30],
            [10, 15, 50],
            [50, 70, 100],
            [60, 80, 85],
            [80, 90, 110],
        ],
        index=index,
        columns=[25, 50, 75],
    )

    # test first issue time
    loss = Measures(time_horizon, quantile_levels, data, predicted_loads_per_issue_time.iloc[:6]).pinball_loss
    loss2 = Measures(time_horizon, quantile_levels, data2, predicted_loads_per_issue_time.iloc[:6]).pinball_loss
    expected_loss = pd.DataFrame(
        data=[
            [0, 0, 0],
            [7.5, 10, 7.5],
            [10, 12.5, 2.5],
            [2.5, 5, 10],
            [0, 10, 6.25],
            [15, 15, 12.5],
        ],
        index=time_horizon.deltas,
        columns=[25, 50, 75],
    )
    assert np.all(loss.groupby("offset").mean() == expected_loss)
    assert np.all(loss2.groupby("offset").mean() == expected_loss)

    # test second issue time
    loss = Measures(time_horizon, quantile_levels, data, predicted_loads_per_issue_time.iloc[6:]).pinball_loss
    loss2 = Measures(time_horizon, quantile_levels, data2, predicted_loads_per_issue_time.iloc[6:]).pinball_loss
    expected_loss = pd.DataFrame(
        data=[
            [2.5, 5, 7.5],
            [10, 15, 15],
            [10, 17.5, 0],
            [0, 10, 12.5],
            [7.5, 15, 8.75],
            [22.5, 20, 15],
        ],
        index=time_horizon.deltas,
        columns=[25, 50, 75],
    )
    assert np.all(loss.groupby("offset").mean() == expected_loss)
    assert np.all(loss2.groupby("offset").mean() == expected_loss)

    # test mean
    loss = Measures(time_horizon, quantile_levels, data, predicted_loads_per_issue_time).pinball_loss
    loss2 = Measures(time_horizon, quantile_levels, data2, predicted_loads_per_issue_time).pinball_loss
    expected_loss = pd.DataFrame(
        data=[
            [1.25, 2.5, 3.75],
            [8.75, 12.5, 11.25],
            [10, 15, 1.25],
            [1.25, 7.5, 11.25],
            [3.75, 12.5, 7.5],
            [18.75, 17.5, 13.75],
        ],
        index=time_horizon.deltas,
        columns=[25, 50, 75],
    )
    assert np.all(loss.groupby("offset").mean() == expected_loss)
    assert np.all(loss2.groupby("offset").mean() == expected_loss)
