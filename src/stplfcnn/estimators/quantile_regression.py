import pathlib
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from patsy.highlevel import dmatrices, dmatrix

from . import PredictionDataFrameBuilder, QuantileEstimator
from .. import QuantileLevels, TimeHorizon
from ..sklearn_quantile import QuantileRegressor


class QuantileRegressionEstimator(QuantileEstimator):
    """
    A quantile regression estimator depending on time and weather variables.

    The model is inspired by Tao Hong's vanilla model from Short Term Electric
    Load Forecasting (2010, North Carolina State University).

    Including the month can cause problems if a month does not occur in the
    training data because the month is a categorical variable.
    """

    def __init__(
            self,
            time_horizon: TimeHorizon,
            quantile_levels: QuantileLevels,
            max_iter: int = 100,
            include_month: bool = False,
            temperature_shift: float = 0,
            temperature_scale: float = 1,
    ) -> None:
        super().__init__(time_horizon, quantile_levels)
        self.max_iter = max_iter
        self.include_month = include_month
        self.temperature_shift = temperature_shift
        self.temperature_scale = temperature_scale

        self.daysofweek = range(7)
        self.months = range(1, 13)
        self.minutes = range(0, 24 * 60, self.time_horizon.step)
        exog = [
            "C(timestamp.dt.dayofweek, levels=self.daysofweek):C(self.day_progress(timestamp), levels=self.minutes)",
            "C(self.day_progress(timestamp), levels=self.minutes):self.standardize(temperature)",
            "C(self.day_progress(timestamp), levels=self.minutes):I((self.standardize(temperature))**2)",
            "C(self.day_progress(timestamp), levels=self.minutes):I((self.standardize(temperature))**3)",
        ]
        if include_month:
            exog += [
                "C(timestamp.dt.month, levels=self.months)",
                "C(timestamp.dt.month, levels=self.months):self.standardize(temperature)",
                "C(timestamp.dt.month, levels=self.months):I((self.standardize(temperature))**2)",
                "C(timestamp.dt.month, levels=self.months):I((self.standardize(temperature))**3)",
            ]
        self.exog = " + ".join(exog)
        self.formula = "load ~ " + self.exog

        self.models: Optional[List[QuantileRegressor]] = None

    def train(self, observed_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> None:
        resampled_data, unique_inverse = self.unique_data(observed_data, issue_times)
        y, X = dmatrices(self.formula, resampled_data)
        self.models = []
        for quantile_level in self.quantile_levels.fractions:
            model = QuantileRegressor(
                quantile=quantile_level,
                max_iter=self.max_iter,

            ).fit(X, y.flatten())
            if model.n_iter_ >= self.max_iter:
                print(f"Training for model {quantile_level} stopped due to iteration limit.")
            else:
                print(f"Training for model {quantile_level} finished.")
            print("Iterations:  ", model.n_iter_)
            print("Coefficients:")
            print(model.coef_)
            print("Intercept:   ", model.intercept_)
            print("Gamma:       ", model.gamma_)
            self.models.append(model)

    def predict(self, input_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> pd.DataFrame:
        resampled_data, unique_inverse = self.unique_data(input_data, issue_times)
        X = dmatrix(self.exog, resampled_data)
        return PredictionDataFrameBuilder(self, issue_times).build(
            np.array([
                np.maximum(model.predict(X), 0)[unique_inverse]
                for model in self.models
            ]).T,
        )

    def unique_data(self, data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> Tuple[pd.DataFrame, np.ndarray]:
        indices = (issue_times.values[:, None] + self.time_horizon.deltas.values).flatten()
        indices_unique, unique_inverse = np.unique(indices, return_inverse=True)
        resampled_data = self.resample_data(data).loc[indices_unique].reset_index()
        return resampled_data, unique_inverse

    def day_progress(self, timestamp: pd.Series) -> pd.Series:
        return 60 * timestamp.dt.hour + timestamp.dt.minute

    def standardize(self, temperature: pd.Series) -> pd.Series:
        return (temperature + self.temperature_shift) * self.temperature_scale

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.models = None

    def load_trained_state(self, path: pathlib.Path) -> None:
        with open(path / "state.pickle", "rb") as f:
            state = pickle.load(f)
            self.models = state

    def dump_trained_state(self, path: pathlib.Path) -> None:
        with open(path / "state.pickle", "wb") as f:
            pickle.dump(self.models, f)
