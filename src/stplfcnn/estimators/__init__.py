import abc
import importlib
import pathlib
from typing import Generator, Optional, Type, TypeVar

import numpy as np
import pandas as pd

from .. import QuantileLevels, TimeHorizon

T_Estimator = TypeVar("T_Estimator", bound="QuantileEstimator")


class QuantileEstimator(metaclass=abc.ABCMeta):
    """
    A quantile estimator represents a model that can be trained on observed
    load data or used to predict multiple load quantiles.

    An estimator is a context manager. The context has to be active for any
    calls of train and predict. It is guaranteed that the predict method can
    use the trained parameters of the last train call in the same active
    context.
    """

    def __init__(self, time_horizon: TimeHorizon, quantile_levels: QuantileLevels) -> None:
        """
        Initialize the quantile estimator.

        The time horizon is divided into lead times. The values of the slice
        are offsets from the issue time in minutes.

        The quantile levels have to in ascending order.
        """
        if (time_horizon.stop - time_horizon.start) % time_horizon.step != 0:
            raise ValueError("time_horizon has to be divided into equally sized parts")
        self.time_horizon = time_horizon
        self.quantile_levels = quantile_levels

    @abc.abstractmethod
    def train(self, observed_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> Optional[Generator[int, None, None]]:
        """
        Train the model on the observed data.

        The observed data should at least have two columns: the index
        (datetime) and the actual load.

        If a generator is returned, it can be used to observe the training
        iterations and to stop the training.
        """

    @abc.abstractmethod
    def predict(self, input_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Predict the loads of the input data.

        The input data should at least have an index column (datetime).

        The loads in the forecast horizon are predicted separately for each
        issue time and returned as separate data frames. This is important for
        overlapping time intervals.
        """
        # TODO update documentation: no separate data frames

    def __enter__(self: T_Estimator) -> T_Estimator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data frame to the time horizon frequency.
        """
        # TODO check columns
        # TODO additional columns
        return data.resample(self.time_horizon.deltas.freq).agg({
            "load": "sum",
            "temperature": "mean",
        })

    @classmethod
    def from_params(cls: Type[T_Estimator], **params) -> T_Estimator:
        """
        Create an estimator from params.
        """
        params["time_horizon"] = TimeHorizon(**params["time_horizon"])
        if isinstance(params["quantile_levels"], list):
            params["quantile_levels"] = QuantileLevels(params["quantile_levels"])
        else:
            params["quantile_levels"] = QuantileLevels(**params["quantile_levels"])
        return cls(**params)

    @abc.abstractmethod
    def load_trained_state(self, path: pathlib.Path) -> None:
        """
        Load a trained state into the estimator.
        """

    @abc.abstractmethod
    def dump_trained_state(self, path: pathlib.Path) -> None:
        """
        Dump the trained state into files.
        """


class PredictionDataFrameBuilder:
    def __init__(self, estimator: QuantileEstimator, issue_times: pd.DatetimeIndex) -> None:
        self.quantile_levels = estimator.quantile_levels
        issue_time_index = np.repeat(issue_times, len(estimator.time_horizon))
        offset_index = np.tile(estimator.time_horizon.deltas, len(issue_times))
        self.index = pd.MultiIndex.from_arrays(
            [issue_time_index, issue_time_index + offset_index, offset_index],
            names=["issue_time", "lead_time", "offset"],
        )

    def build(self, data) -> pd.DataFrame:
        return pd.DataFrame(
            data=data,
            index=self.index,
            columns=self.quantile_levels.numerators,
        )


_class_modules = dict(
    Persistence1Estimator="persistence1",
    Persistence2Estimator="persistence2",
    NaiveEstimator="naive",
    QuantileRegressionEstimator="quantile_regression",
    STLQFCNNEstimator="stlqfcnn",
)


def get_class(class_name: str) -> Type[QuantileEstimator]:
    """
    Load the estimator class from its module.
    """
    module = importlib.import_module(f".{_class_modules[class_name]}", __package__)
    return getattr(module, class_name)
