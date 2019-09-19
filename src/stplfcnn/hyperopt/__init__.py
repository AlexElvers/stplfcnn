import abc
import hashlib
import importlib
import inspect
import pickle
from typing import Any, Dict, Iterator, Optional, Tuple, Type, TypeVar

import numpy as np
import yaml

from .. import database
from ..estimators import QuantileEstimator
from ..utils import Reference, ReferenceDumper


class BadEstimatorParams(Exception):
    pass


T_HyperparameterOptimizer = TypeVar("T_HyperparameterOptimizer", bound="HyperparameterOptimizer")


class HyperparameterOptimizer(metaclass=abc.ABCMeta):
    """
    A hyperparameter optimizer using Hyperopt.
    """

    estimator_class: Type[QuantileEstimator]

    def __init__(
            self,
            db: "database.Database",
            time_horizon: str,
            quantile_levels: str,
            space,
            max_evals: int,
            overall_max_evals: Optional[int] = None,
            overall_max_evals_factor: Optional[float] = 1,
            max_chunk_size: int = 100,
    ) -> None:
        """
        Initialize the hyperparameter optimizer.

        `max_evals` includes the trails with status 'ok', `overall_max_evals`
        additionally includes failed trials. If `overall_max_evals` is not
        set, it is calculated from the `overall_max_evals_factor` (default 1).
        """
        self.db = db
        self.time_horizon = time_horizon
        self.quantile_levels = quantile_levels
        self.max_evals = max_evals
        if overall_max_evals is None:
            self.overall_max_evals = int(max_evals * overall_max_evals_factor)
        else:
            self.overall_max_evals = overall_max_evals
        self.max_chunk_size = max_chunk_size
        self.space = space

        self.trials = None
        self.dump_trained_state = False
        self.dump_predictions = False

    @classmethod
    def from_params(
            cls: Type[T_HyperparameterOptimizer],
            db: "database.Database",
            **params,
    ) -> T_HyperparameterOptimizer:
        """
        Create a hyperparameter optimizer from params.
        """
        return cls(db, **params)

    def optimize(self) -> Iterator[Tuple[int, int, Dict[str, Any]]]:
        """
        Search for the optimal hyperparameters.

        The optimization stops when `max_evals` trials with status 'ok' are
        reached or when `overall_max_evals` trials are reached.
        """
        from hyperopt import fmin, tpe, Trials
        if self.trials is None:
            self.trials = Trials()

        num_trials = len(self.trials)
        num_ok_trials = sum(1 for t in self.trials if t["result"]["status"] == "ok")
        finished = False
        while not finished:
            print(f"\033]0;Hyperopt status: {num_ok_trials} trials\007")
            rest_max_chunk_size = self.max_chunk_size
            while rest_max_chunk_size > 0:
                missing_ok_trials = self.max_evals - num_ok_trials
                overall_missing_trials = self.overall_max_evals - num_trials
                # maximum number of evals to not exceed max_evals
                current_chunk_size = min(rest_max_chunk_size, missing_ok_trials, overall_missing_trials)
                if current_chunk_size == 0:
                    finished = True
                    break
                print("training chunk of size", current_chunk_size)
                rest_max_chunk_size -= current_chunk_size
                # train the chunk
                fmin(
                    fn=self.train_single_estimator,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=num_trials + current_chunk_size,
                    trials=self.trials,
                    return_argmin=False,
                )
                num_trials = len(self.trials)
                num_ok_trials = sum(1 for t in self.trials if t["result"]["status"] == "ok")
            yield num_trials, num_ok_trials, self.trials.best_trial["result"] if num_ok_trials > 0 else None

    def train_single_estimator(self, params) -> Any:
        try:
            params = self.build_estimator_params(**params)
        except BadEstimatorParams as e:
            fail_reason = "bad parameters" + (f": {e}" if e.args else "")
            print(20 * "=")
            print("Trial", len(self.trials.trials) - 1)
            print("SKIPPING:", fail_reason)
            print("original parameters:")
            print(yaml.dump(params, Dumper=ReferenceDumper)[:-1])
            print(20 * "=")
            return dict(
                status="fail",
                fail_reason=fail_reason,
                original_params=params,
            )
        params_hash = self.calculate_params_hash(params)
        estimator_name = f"{self.db.hyperopt}-gen-{params_hash}"

        print(20 * "=")
        print("Trial", len(self.trials.trials) - 1, estimator_name)
        print(yaml.dump(params, Dumper=ReferenceDumper)[:-1])
        print(20 * "=")

        self.db.apply(estimator=estimator_name)
        if self.db.estimator_params_path.exists():
            # should not happen if we are using random floats
            print("Already trained. Skipping training.")
        else:
            # dump params
            self.db.dump_estimator_params(params)

            # run training
            from ..cli import f_train
            f_train(
                self.db,
                dump_trained_state=self.dump_trained_state,
                dump_predictions=self.dump_predictions,
            )

        mean_pinball_loss_train, mean_pinball_loss_test, mean_training_duration = self.calculate_loss_and_duration()

        loss = mean_pinball_loss_test
        print("\nLoss (used in Hyperopt):", loss)

        return dict(
            status="ok",
            loss=loss,
            estimator=estimator_name,
            params=params,
            mean_pinball_loss_train=mean_pinball_loss_train,
            mean_pinball_loss_test=mean_pinball_loss_test,
            mean_training_duration=mean_training_duration,
        )

    def build_estimator_params(self, **kwargs) -> Dict[str, Any]:
        """
        Build estimator parameters.
        """
        valid_params = {
            p.name
            for p in inspect.signature(self.estimator_class).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        valid_params -= {"time_horizon", "quantile_levels"}
        invalid_params = [p for p in kwargs if p not in valid_params]
        if invalid_params:
            raise ValueError(f"invalid params: {', '.join(invalid_params)}")
        return {
            "type": self.estimator_class.__name__,
            "time_horizon": Reference(self.time_horizon),
            "quantile_levels": Reference(self.quantile_levels),
            **kwargs,
        }

    @classmethod
    def calculate_params_hash(cls, params: Dict[str, Any]) -> str:
        """
        Calculate the hash of the estimator params.
        """

        def canonicalize(obj):
            if isinstance(obj, dict):
                return sorted((k, canonicalize(v)) for k, v in obj.items())
            elif isinstance(obj, (tuple, list)):
                return [canonicalize(e) for e in obj]
            return obj

        canonical_params = canonicalize(params)
        pickled_params = pickle.dumps(canonical_params, pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(pickled_params).hexdigest()

    def calculate_loss_and_duration(self) -> Tuple[float, float, float]:
        # load errors and training durations
        num_partitions = len(self.db.load_partitions())
        errors_train = []
        errors_test = []
        training_durations = []
        for partition in range(num_partitions):
            self.db.apply(partition_number=partition)
            for predictions_name, errors in [("train", errors_train), ("test", errors_test)]:
                self.db.apply(predictions=predictions_name)
                errors.append(self.db.load_errors())
            training_durations.append(self.db.load_training_duration())
        # calculate mean pinball losses and mean training duration
        mean_pinball_loss_train = float(np.mean([
            errors["pinball_loss"].values
            for errors in errors_train
        ]))
        mean_pinball_loss_test = float(np.mean([
            errors["pinball_loss"].values
            for errors in errors_test
        ]))
        mean_training_duration = float(np.mean(training_durations))
        return mean_pinball_loss_train, mean_pinball_loss_test, mean_training_duration


_class_modules = dict(
    STLQFCNNHyperparameterOptimizer="stlqfcnn",
)


def get_class(class_name: str) -> Type[HyperparameterOptimizer]:
    """
    Load the hyperparameter optimizer class from its module.
    """
    module = importlib.import_module(f".{_class_modules[class_name]}", __package__)
    return getattr(module, class_name)
