import pathlib
import pickle

import numpy as np
import pandas as pd

from . import PredictionDataFrameBuilder, QuantileEstimator
from .. import QuantileLevels, TimeHorizon
from ..utils import cached_getter


class NaiveEstimator(QuantileEstimator):
    """
    An estimator that models the load y_(t+k) as op(y_(t+k-o), noise_k), where
    op is either addition or multiplication and o is an offset (e.g. one day
    or one week). The model learns the noise by calculating the quantiles on
    the difference or quotient of y_(t+k) and y_(t+k-o).

    The load predictions are clipped at 0, as negative loads are not possible.
    """

    def __init__(
            self,
            time_horizon: TimeHorizon,
            quantile_levels: QuantileLevels,
            load_offset: int,
            noise_type: str = "additive",
    ) -> None:
        super().__init__(time_horizon, quantile_levels)
        if load_offset <= 0:
            raise ValueError("load_offset must be positive")
        self.load_offset = load_offset
        if noise_type not in ("additive", "multiplicative"):
            raise ValueError("noise_type must be additive or multiplicative")
        self.noise_type = noise_type
        self.quantile_forecasts = None

    def train(self, observed_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> None:
        extractor = FeatureExtractor(self, observed_data, issue_times)
        historical_loads = extractor.historical_loads
        loads = extractor.loads

        if self.noise_type == "additive":
            diff_loads = loads - historical_loads
        elif self.noise_type == "multiplicative":
            diff_loads = np.divide(loads, historical_loads, out=np.zeros_like(loads), where=historical_loads != 0)
        else:
            assert False
        self.quantile_forecasts = np.percentile(diff_loads, self.quantile_levels.percentiles, axis=0).T

    def predict(self, input_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> pd.DataFrame:
        extractor = FeatureExtractor(self, input_data, issue_times)
        historical_loads = extractor.historical_loads

        if self.noise_type == "additive":
            predicted_loads = historical_loads[:, :, None] + self.quantile_forecasts[None, :, :]
            predicted_loads = np.maximum(predicted_loads, 0)
        elif self.noise_type == "multiplicative":
            predicted_loads = historical_loads[:, :, None] * self.quantile_forecasts[None, :, :]
        else:
            assert False

        return PredictionDataFrameBuilder(self, issue_times).build(
            predicted_loads.reshape(-1, predicted_loads.shape[-1]),
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


class FeatureExtractor:
    def __init__(self, estimator: NaiveEstimator, data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> None:
        self.estimator = estimator
        self.data = estimator.resample_data(data)
        self.indices = issue_times.values[:, None] + estimator.time_horizon.deltas.values
        historical_indices = self.indices - pd.Timedelta(self.estimator.load_offset, "m")
        self.historical_loads = self.data.load.loc[historical_indices.flat].values.reshape(historical_indices.shape)

    @cached_getter
    def loads(self) -> np.ndarray:
        return self.data.load.loc[self.indices.flat].values.reshape(self.indices.shape)
