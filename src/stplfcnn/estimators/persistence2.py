import pathlib
import pickle

import numpy as np
import pandas as pd

from . import PredictionDataFrameBuilder, QuantileEstimator
from .. import QuantileLevels, TimeHorizon


# TODO Is 'persistence' model the best name for this?

class Persistence2Estimator(QuantileEstimator):
    """
    An estimator that determines the quantiles empirically independent of
    any conditions but dependent on the lead time offset.
    """

    def __init__(self, time_horizon: TimeHorizon, quantile_levels: QuantileLevels) -> None:
        super().__init__(time_horizon, quantile_levels)
        self.quantile_forecasts = None

    def train(self, observed_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> None:
        resampled_loads = self.resample_data(observed_data).load
        indices = issue_times.values[:, None] + self.time_horizon.deltas.values
        loads = resampled_loads.loc[indices.flat].values.reshape(indices.shape)
        self.quantile_forecasts = np.percentile(loads, self.quantile_levels.percentiles, axis=0).T

    def predict(self, input_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> pd.DataFrame:
        return PredictionDataFrameBuilder(self, issue_times).build(
            np.tile(self.quantile_forecasts, (len(issue_times), 1)),
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.quantile_forecasts = None

    def load_trained_state(self, path: pathlib.Path) -> None:
        with open(path / "state.pickle", "rb") as f:
            state = pickle.load(f)
            assert isinstance(state, np.ndarray)
            self.quantile_forecasts = state

    def dump_trained_state(self, path: pathlib.Path) -> None:
        with open(path / "state.pickle", "wb") as f:
            pickle.dump(self.quantile_forecasts, f)
