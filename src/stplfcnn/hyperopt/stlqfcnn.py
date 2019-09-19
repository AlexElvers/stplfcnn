import math
from typing import Any, Dict, Optional

from . import BadEstimatorParams, HyperparameterOptimizer
from ..database import Database
from ..estimators.stlqfcnn import STLQFCNNEstimator
from ..utils import cached_getter


class STLQFCNNHyperparameterOptimizer(HyperparameterOptimizer):
    """
    A hyperparameter optimizer for the STLQFCNN estimator.
    """

    estimator_class = STLQFCNNEstimator

    def __init__(
            self,
            db: Database,
            time_horizon: str,
            quantile_levels: str,
            space,
            max_evals: int,
            overall_max_evals: Optional[int] = None,
            overall_max_evals_factor: Optional[float] = 1,
            max_chunk_size: int = 100,
            max_number_of_weights_between_layers: Optional[int] = None,
    ) -> None:
        super().__init__(
            db,
            time_horizon,
            quantile_levels,
            space,
            max_evals,
            overall_max_evals,
            overall_max_evals_factor,
            max_chunk_size,
        )

        self.max_number_of_weights_between_layers = max_number_of_weights_between_layers

    @cached_getter
    def time_horizon_len(self) -> int:
        self.db.apply(time_horizon=self.time_horizon)
        return len(self.db.load_time_horizon())

    @cached_getter
    def quantile_levels_len(self) -> int:
        self.db.apply(quantile_levels=self.quantile_levels)
        return len(self.db.load_quantile_levels())

    def build_estimator_params(self, **kwargs) -> Dict[str, Any]:
        current_history_shape = kwargs.get("history_shape", (7, 24))
        for i, layer_params in enumerate(kwargs["conv_layers_params"]):
            filter_shape = layer_params["filter_shape"]
            if filter_shape[0] > current_history_shape[0] or filter_shape[1] > current_history_shape[1]:
                raise BadEstimatorParams(
                    f"trying to apply filter on smaller input data"
                    f" (conv layer={i},"
                    f" filter shape={filter_shape},"
                    f" input shape={current_history_shape})"
                )
            pooling_shape = layer_params.get("pooling_shape", None)
            if pooling_shape is not None:
                current_history_shape = (
                    math.ceil(current_history_shape[0] / pooling_shape[0]),
                    math.ceil(current_history_shape[1] / pooling_shape[1]),
                )

        if self.max_number_of_weights_between_layers is not None:
            num_filters = kwargs["conv_layers_params"][-1]["num_filters"]
            num_conv_inputs = num_filters * current_history_shape[0] * current_history_shape[1]
            num_horizon_variables = len(kwargs.get("horizon_inputs", []))
            num_horizon_inputs = num_horizon_variables * self.time_horizon_len
            num_issue_time_inputs = kwargs.get("issue_time_inputs", False) * 2
            num_inputs = num_conv_inputs + num_horizon_inputs + num_issue_time_inputs
            num_outputs = kwargs["fc_layers_params"][0].get("num_outputs")
            if num_outputs is None:
                num_outputs = self.time_horizon_len * self.quantile_levels_len

            num_weights = num_inputs * num_outputs
            if num_weights > self.max_number_of_weights_between_layers:
                raise BadEstimatorParams(
                    f"too many weights in first fully connected layer"
                    f" (threshold={self.max_number_of_weights_between_layers},"
                    f" num weights={num_weights},"
                    f" num inputs={num_inputs},"
                    f" num outputs={num_outputs}"
                    f" (quantiles={self.quantile_levels_len}, horizon={self.time_horizon_len}),"
                    f" num horizon inputs={num_horizon_inputs}"
                    f" (variables={num_horizon_variables}, horizon={self.time_horizon_len}),"
                    f" num issue time inputs={num_issue_time_inputs},"
                    f" num conv inputs={num_conv_inputs}"
                    f" (filters={num_filters}, shape={current_history_shape}))"
                )

        return super().build_estimator_params(**kwargs)
