import numpy as np
import pandas as pd

from stplfcnn import QuantileLevels, TimeHorizon
from stplfcnn.measures import Measures


def test_calculate_winkler_score():
    time_horizon = TimeHorizon(0, 6 * 60, 60)
    quantile_levels = QuantileLevels([10, 25, 50, 75, 90])
    data = pd.DataFrame(
        data=[50] * 12,
        index=pd.DatetimeIndex(freq="60min", start="2018-01-01", periods=12),
        columns=["load"],
    )
    data2 = pd.DataFrame(
        data=[5, 10, 15, 20] * 12,
        index=pd.DatetimeIndex(freq="15min", start="2018-01-01", periods=12 * 4),
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
            [50, 50, 50, 50, 50],
            [10, 20, 30, 40, 45],
            [5, 10, 25, 60, 65],
            [30, 40, 60, 90, 100],
            [50, 50, 70, 75, 80],
            [60, 70, 80, 100, 140],

            [40, 40, 40, 40, 40],
            [0, 10, 20, 30, 35],
            [5, 10, 15, 50, 55],
            [40, 50, 70, 100, 110],
            [60, 60, 80, 85, 90],
            [70, 80, 90, 110, 150],
        ],
        index=index,
        columns=[10, 25, 50, 75, 90],
    )

    # test first issue time
    score = Measures(time_horizon, quantile_levels, data, predicted_loads_per_issue_time.iloc[:6]).winkler_score
    score2 = Measures(time_horizon, quantile_levels, data2, predicted_loads_per_issue_time.iloc[:6]).winkler_score
    expected_score = pd.DataFrame(
        data=[
            [0, 0],
            [85, 60],
            [60, 50],
            [70, 50],
            [30, 25],
            [180, 110],
        ],
        index=time_horizon.deltas,
        columns=[20, 50],
    )
    assert np.all(score.groupby("offset").mean() == expected_score)
    assert np.all(score2.groupby("offset").mean() == expected_score)

    # test second issue time
    score = Measures(time_horizon, quantile_levels, data, predicted_loads_per_issue_time.iloc[6:]).winkler_score
    score2 = Measures(time_horizon, quantile_levels, data2, predicted_loads_per_issue_time.iloc[6:]).winkler_score
    expected_score = pd.DataFrame(
        data=[
            [100, 40],
            [185, 100],
            [50, 40],
            [70, 50],
            [130, 65],
            [280, 150],
        ],
        index=time_horizon.deltas,
        columns=[20, 50],
    )
    assert np.all(score.groupby("offset").mean() == expected_score)
    assert np.all(score2.groupby("offset").mean() == expected_score)

    # test mean
    score = Measures(time_horizon, quantile_levels, data, predicted_loads_per_issue_time).winkler_score
    score2 = Measures(time_horizon, quantile_levels, data2, predicted_loads_per_issue_time).winkler_score
    expected_score = pd.DataFrame(
        data=[
            [50, 20],
            [135, 80],
            [55, 45],
            [70, 50],
            [80, 45],
            [230, 130],
        ],
        index=time_horizon.deltas,
        columns=[20, 50],
    )
    assert np.all(score.groupby("offset").mean() == expected_score)
    assert np.all(score2.groupby("offset").mean() == expected_score)
