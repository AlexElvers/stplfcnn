import datetime
import itertools
import lzma
import pathlib
import pickle
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml
from pandas.tseries.frequencies import to_offset

from . import IssueTimesPartition, QuantileLevels, TimeHorizon, datareaders, estimators, hyperopt as hyperopt_, \
    partitioners
from .utils import HyperoptLoader, Reference, ReferenceDumper, ReferenceLoader


def _issue_times_from_params(
        params_issue_times: Sequence[Dict[str, Union[datetime.datetime, datetime.date]]],
        freq: str,
) -> pd.DatetimeIndex:
    issue_times = pd.DatetimeIndex([])
    for params_part in params_issue_times:
        assert isinstance(params_part, dict)
        start = pd.Timestamp(params_part.pop("start"))
        end = pd.Timestamp(params_part.pop("end"))
        assert len(params_part) == 0
        part = pd.date_range(start, end, freq=freq)
        issue_times |= part
    return issue_times


def _issue_times_to_params(
        issue_times: pd.DatetimeIndex,
        freq: pd.DateOffset,
        diffs: pd.TimedeltaIndex,
) -> Sequence[Dict[str, Union[datetime.datetime, datetime.date]]]:
    skip_indices = 1 + np.where(diffs != freq)[0]
    return [
        dict(start=issue_times[i].to_pydatetime(), end=issue_times[j - 1].to_pydatetime())
        for i, j in zip(
            itertools.chain([0], skip_indices),
            itertools.chain(skip_indices, [len(issue_times)]),
        )
    ]


def _load_params(path: pathlib.Path, loader_class=yaml.SafeLoader) -> Any:
    with open(path) as f:
        return yaml.load(f, loader_class)


def _dump_params(
        path: pathlib.Path,
        params: Any,
        overwrite_cb: Optional[Callable[[pathlib.Path], None]] = None,
        dumper_class=yaml.SafeDumper,
        **kwargs,
) -> None:
    """
    Dump parameters to file.

    If the file exists, the overwrite callback is called. The file is not
    written if the callback returns False.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite_cb is not None and overwrite_cb(path) is False:
        return
    kwargs = {"default_flow_style": False, **kwargs}
    with open(path, "w") as f:
        yaml.dump(params, f, dumper_class, **kwargs)


class Database:
    data_reader: str
    estimator: str
    issue_times: str
    model: str
    partition_number: int
    partitioner: str
    partitions: str
    predictions: str
    quantile_levels: str
    time_horizon: str
    hyperopt: str

    def __init__(
            self,
            *,
            base_path: pathlib.Path = None,
            params_path: pathlib.Path = None,
            models_path: pathlib.Path = None,
            overwrite_cb: Optional[Callable[[pathlib.Path], None]] = None,
            **kwargs,
    ) -> None:
        base_path = base_path or pathlib.Path()
        self.params_path = params_path or base_path / "params"
        self.models_path = models_path or base_path / "models"
        self.overwrite_cb = overwrite_cb

        self.apply(**kwargs)

    def apply(self, **kwargs) -> None:
        not_existing = set(kwargs.keys()) - (
                set(self.__annotations__.keys()) - {"base_path", "params_path", "models_path"})
        if len(not_existing) == 0:
            self.__dict__.update(kwargs)
            if all(hasattr(self, k) for k in ["estimator", "data_reader", "partitions"]) and (
                    not hasattr(self, "model") or getattr(self, "_generated_model", False)):
                self.model = f"estimator={self.estimator},data_reader={self.data_reader},partitions={self.partitions}"
                self._generated_model = True
        else:
            raise ValueError(f"some keys do not exist: {', '.join(sorted(not_existing))}")

    @property
    def issue_times_params_path(self) -> pathlib.Path:
        return self.params_path / "issue_times" / f"{self.issue_times}.yaml"

    @property
    def partitioner_params_path(self) -> pathlib.Path:
        return self.params_path / "partitioners" / f"{self.partitioner}.yaml"

    @property
    def partitions_params_path(self) -> pathlib.Path:
        return self.params_path / "partitions" / f"{self.partitions}.yaml"

    @property
    def data_reader_params_path(self) -> pathlib.Path:
        return self.params_path / "data_readers" / f"{self.data_reader}.yaml"

    @property
    def estimator_params_path(self) -> pathlib.Path:
        return self.params_path / "estimators" / f"{self.estimator}.yaml"

    @property
    def time_horizon_params_path(self) -> pathlib.Path:
        return self.params_path / "time_horizons" / f"{self.time_horizon}.yaml"

    @property
    def quantile_levels_params_path(self) -> pathlib.Path:
        return self.params_path / "quantile_levels" / f"{self.quantile_levels}.yaml"

    @property
    def hyperopt_params_path(self) -> pathlib.Path:
        return self.params_path / "hyperopt" / f"{self.hyperopt}.yaml"

    @property
    def hyperopt_trials_path(self) -> pathlib.Path:
        return self.models_path / "trials" / f"{self.hyperopt}.pickle.xz"

    @property
    def model_path(self) -> pathlib.Path:
        return self.models_path / self.model

    @property
    def model_estimator_path(self) -> pathlib.Path:
        return self.model_path / "estimator.pickle"

    @property
    def model_partition_path(self) -> pathlib.Path:
        return self.model_path / f"partition={self.partition_number}"

    @property
    def errors_by_iteration_pickle_path(self) -> pathlib.Path:
        return self.model_partition_path / "errors_by_iteration.pickle.xz"

    @property
    def training_duration_struct_path(self) -> pathlib.Path:
        return self.model_partition_path / "training_duration.struct"

    @property
    def predictions_path(self) -> pathlib.Path:
        return self.model_partition_path / "predictions" / self.predictions

    @property
    def errors_pickle_path(self) -> pathlib.Path:
        return self.predictions_path / "errors.pickle.xz"

    @property
    def predictions_pickle_path(self) -> pathlib.Path:
        return self.predictions_path / "predictions.pickle.xz"

    def _check_overwrite(self, path: pathlib.Path) -> bool:
        return path.exists() and self.overwrite_cb is not None and self.overwrite_cb(path) is False

    def load_issue_times(self) -> pd.DatetimeIndex:
        """
        Load issue times from yaml.
        """
        params = _load_params(self.issue_times_params_path)
        assert isinstance(params, dict)
        freq = params.pop("freq")
        params_issue_times = params.pop("issue_times")
        assert len(params) == 0
        assert isinstance(freq, str)
        assert isinstance(params_issue_times, list)
        return _issue_times_from_params(params_issue_times, freq)

    def dump_issue_times(self, issue_times: pd.DatetimeIndex) -> None:
        """
        Dump issue times to yaml.
        """
        diffs = issue_times[1:] - issue_times[:-1]
        freq = to_offset(diffs.min())
        params_issue_times = _issue_times_to_params(issue_times, freq, diffs)
        params = dict(
            freq=freq.freqstr,
            issue_times=params_issue_times,
        )

        _dump_params(self.issue_times_params_path, params, self.overwrite_cb)

    def create_partitioner(self) -> partitioners.Partitioner:
        """
        Create partitioner from yaml.
        """
        params = _load_params(self.partitioner_params_path)
        assert isinstance(params, dict)
        assert "type" in params
        return partitioners.get_class(params.pop("type")).from_params(**params)

    def load_partitions(self) -> Sequence[IssueTimesPartition]:
        """
        Load partitions from yaml.
        """
        params = _load_params(self.partitions_params_path)
        assert isinstance(params, dict)
        freq = params.pop("freq")
        params_partitions = params.pop("partitions")
        assert len(params) == 0
        assert isinstance(freq, str)
        assert isinstance(params_partitions, list)
        partitions = []
        for params_partition in params_partitions:
            assert isinstance(params_partition, dict)
            partitions.append(IssueTimesPartition(
                train=_issue_times_from_params(params_partition.pop("train"), freq),
                test=_issue_times_from_params(params_partition.pop("test"), freq),
            ))
            assert len(params_partition) == 0
        return partitions

    def dump_partitions(self, partitions: Sequence[IssueTimesPartition]) -> None:
        """
        Dump partitions to yaml.
        """
        freq = None
        for partition in partitions:
            for issue_times in [partition.train, partition.test]:
                diffs = issue_times[1:] - issue_times[:-1]
                current_freq = to_offset(diffs.min())
                if freq is None or current_freq < freq:
                    freq = current_freq

        params_partitions = []
        for partition in partitions:
            params_partition = {}
            for set_name, issue_times in [("train", partition.train), ("test", partition.test)]:
                diffs = issue_times[1:] - issue_times[:-1]
                params_partition[set_name] = _issue_times_to_params(issue_times, freq, diffs)
            params_partitions.append(params_partition)
        params = dict(
            freq=freq.freqstr,
            partitions=params_partitions,
        )

        _dump_params(self.partitions_params_path, params, self.overwrite_cb)

    def create_data_reader(self) -> datareaders.DataReader:
        """
        Create data reader from yaml.
        """
        params = _load_params(self.data_reader_params_path)
        assert isinstance(params, dict)
        assert "type" in params
        return datareaders.get_class(params.pop("type")).from_params(**params)

    def load_time_horizon(self) -> TimeHorizon:
        """
        Load time horizon from yaml.
        """
        params = _load_params(self.time_horizon_params_path)
        return TimeHorizon(**params)

    def load_quantile_levels(self) -> QuantileLevels:
        """
        Load quantile levels from yaml.
        """
        params = _load_params(self.quantile_levels_params_path)
        if isinstance(params, list):
            return QuantileLevels(params)
        else:
            return QuantileLevels(**params)

    def create_estimator(self) -> estimators.QuantileEstimator:
        """
        Create estimator from yaml.
        """
        params = _load_params(self.estimator_params_path, ReferenceLoader)
        assert isinstance(params, dict)
        assert "type" in params
        assert "time_horizon" in params
        assert "quantile_levels" in params
        if isinstance(params["time_horizon"], Reference):
            self.time_horizon = params["time_horizon"].name
            params["time_horizon"] = _load_params(self.time_horizon_params_path)
        if isinstance(params["quantile_levels"], Reference):
            self.quantile_levels = params["quantile_levels"].name
            params["quantile_levels"] = _load_params(self.quantile_levels_params_path)
        return estimators.get_class(params.pop("type")).from_params(**params)

    def dump_estimator_params(self, params: Dict[str, Any]) -> None:
        """
        Dump estimator params to yaml.
        """
        _dump_params(self.estimator_params_path, params, self.overwrite_cb, ReferenceDumper)

    def load_untrained_estimator(self) -> estimators.QuantileEstimator:
        """
        Load untrained estimator from pickle.
        """
        with open(self.model_estimator_path, "rb") as f:
            estimator = pickle.load(f)
        assert isinstance(estimator, estimators.QuantileEstimator)
        return estimator

    def dump_untrained_estimator(self, estimator: estimators.QuantileEstimator) -> None:
        """
        Dump untrained estimator to pickle.
        """
        path = self.model_estimator_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._check_overwrite(path):
            return
        with open(path, "wb") as f:
            pickle.dump(estimator, f, protocol=-1)

    def load_trained_state(self, estimator: estimators.QuantileEstimator) -> None:
        """
        Load the trained state from the database into the estimator.
        """
        estimator.load_trained_state(self.model_partition_path)

    def dump_trained_state(self, estimator: estimators.QuantileEstimator) -> None:
        """
        Dump the trained state from the estimator to the database.
        """
        path = self.model_partition_path
        if self._check_overwrite(path):
            return
        path.mkdir(parents=True, exist_ok=True)
        estimator.dump_trained_state(path)

    def load_errors(self) -> Dict[str, pd.DataFrame]:
        """
        Load errors.
        """
        with lzma.open(self.errors_pickle_path) as f:
            errors = pickle.load(f)
        assert isinstance(errors, dict)
        assert set(errors) == {"coverage", "pinball_loss", "winkler_score"}
        return errors

    def dump_errors(self, errors: Dict[str, pd.DataFrame]) -> None:
        """
        Dump errors.
        """
        path = self.errors_pickle_path
        if self._check_overwrite(path):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with lzma.open(path, "wb") as f:
            pickle.dump(errors, f)

    def load_errors_by_iteration(self) -> Dict[str, Union[List[int], Dict[str, List[pd.DataFrame]]]]:
        """
        Load errors by iteration.
        """
        with lzma.open(self.errors_by_iteration_pickle_path) as f:
            errors_by_iteration = pickle.load(f)
        assert isinstance(errors_by_iteration, dict)
        assert {"iterations", "train", "test"} <= set(errors_by_iteration)
        return errors_by_iteration

    def dump_errors_by_iteration(self, errors_by_iteration: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """
        Dump errors by iteration.
        """
        path = self.errors_by_iteration_pickle_path
        if self._check_overwrite(path):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with lzma.open(path, "wb") as f:
            pickle.dump(errors_by_iteration, f)

    def load_training_duration(self) -> float:
        """
        Load training duration.
        """
        with open(self.training_duration_struct_path, "rb") as f:
            buffer = f.read()
        return struct.unpack("<d", buffer)[0]

    def dump_training_duration(self, training_duration: float) -> None:
        """
        Dump training duration.
        """
        path = self.training_duration_struct_path
        if self._check_overwrite(path):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(struct.pack("<d", training_duration))

    def load_predictions(self) -> pd.DataFrame:
        """
        Load predictions.
        """
        path = self.predictions_pickle_path
        return pd.read_pickle(str(path))

    def dump_predictions(self, predictions: pd.DataFrame) -> None:
        """
        Dump predictions.
        """
        path = self.predictions_pickle_path
        if self._check_overwrite(path):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        return predictions.to_pickle(str(path))

    def create_hyperparameter_optimizer(self) -> "hyperopt_.HyperparameterOptimizer":
        """
        Create hyperparameter optimizer from yaml.
        """
        params = _load_params(self.hyperopt_params_path, HyperoptLoader)
        assert isinstance(params, dict)
        assert "type" in params
        assert "time_horizon" in params
        assert "quantile_levels" in params
        assert "space" in params
        return hyperopt_.get_class(params.pop("type")).from_params(self, **params)

    def load_hyperopt_trials(self) -> Any:
        """
        Load Hyperopt trials.
        """
        with lzma.open(self.hyperopt_trials_path) as f:
            return pickle.load(f)

    def dump_hyperopt_trials(self, trials: Any) -> None:
        """
        Dump Hyperopt trials.
        """
        path = self.hyperopt_trials_path
        if self._check_overwrite(path):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with lzma.open(path, "wb") as f:
            pickle.dump(trials, f)
