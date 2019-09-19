from typing import Any, Dict, Generator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.image import extract_patches

from . import PredictionDataFrameBuilder
from .tensorflow import TensorFlowEstimator
from .. import QuantileLevels, TimeHorizon
from ..utils import cached_getter


class STLQFCNNEstimator(TensorFlowEstimator):
    """
    A CNN estimator depending on time and weather variables and previous
    loads.
    """

    def __init__(
            self,
            time_horizon: TimeHorizon,
            quantile_levels: QuantileLevels,
            tf_seed: Optional[int] = None,
            np_seed: Optional[int] = None,
            *,
            history_shape: Sequence[int] = (7, 24),
            iterations: int = 1000,
            learning_rate: float = .0001,
            batch_size: int = 64,
            conv_layers_params: Sequence[Dict[str, Any]],
            fc_layers_params: Sequence[Dict[str, Any]],
            historical_inputs: Optional[Sequence[str]] = None,
            horizon_inputs: Optional[Sequence[str]] = None,
            issue_time_inputs: bool = False,
            regularization_weight: float = 0,
            monotonous_forecasts: bool = False,
    ) -> None:
        super().__init__(time_horizon, quantile_levels, tf_seed)
        self.np_seed = np_seed
        self.history_shape = history_shape
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.conv_layers_params = conv_layers_params
        self.fc_layers_params = fc_layers_params
        self.historical_inputs: Sequence[str] = historical_inputs or []
        self.horizon_inputs: Sequence[str] = horizon_inputs or []
        self.issue_time_inputs = issue_time_inputs
        self.regularization_weight = regularization_weight
        self.monotonous_forecasts = monotonous_forecasts

    def train(self, observed_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> Generator[int, None, None]:
        extractor = FeatureExtractor(self, observed_data, issue_times)
        y_load = extractor.y_load
        batch_size = min(len(y_load), self.batch_size)

        seed = None if self.np_seed is None else np.append(np.int64(issue_times) // 10**9, self.np_seed)
        np_rand = np.random.RandomState(seed)

        self.session.run(self.initializer)
        i = 0
        try:
            for i in range(self.iterations):
                batch_indices = np_rand.choice(len(y_load), size=batch_size, replace=False)
                feed_dict = {
                    **extractor.create_feed_dict(batch_indices),
                    self.y_true: y_load[batch_indices],
                }
                self.session.run(self.optimizer, feed_dict)
                if i % 100 == 0:
                    yield i
        except KeyboardInterrupt:
            print(f"Canceled training in iteration {i} by keyboard interrupt.")
        if i % 100 != 0:
            yield i

    def predict(self, input_data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> pd.DataFrame:
        extractor = FeatureExtractor(self, input_data, issue_times)

        feed_dict = extractor.create_feed_dict()
        y_pred = self.session.run(self.y_pred, feed_dict)

        return PredictionDataFrameBuilder(self, issue_times).build(
            y_pred.reshape(-1, len(self.quantile_levels)),
        )

    def create_graph_nodes(self) -> None:
        """
        Create the TensorFlow graph and its nodes.
        """
        # historical loads
        self.x_load = tf.placeholder(tf.float32, shape=[None, *self.history_shape], name="x_load")

        # other historical inputs
        self.x_hist = {
            column: tf.placeholder(tf.float32, shape=[None, *self.history_shape], name=f"x_hist.{column}")
            for column in self.historical_inputs
        }
        x_hist = tf.stack([self.x_load, *self.x_hist.values()], axis=-1, name="x_hist")

        # horizon inputs (parallel to y)
        self.x_horizon = {
            column: tf.placeholder(tf.float32, shape=[None, len(self.time_horizon)], name=f"x_horizon.{column}")
            for column in self.horizon_inputs
        }

        # issue time inputs
        if self.issue_time_inputs:
            self.x_issue_time = tf.placeholder(tf.float32, shape=[None, 2], name="x_issue_time")

        # true loads
        self.y_true = tf.placeholder(tf.float32, shape=[None, len(self.time_horizon)], name="y_true")
        y_true_reshaped = tf.reshape(self.y_true, [-1, len(self.time_horizon), 1])

        # weights for regularization
        self.weights = []

        # convolutional layers
        layer = x_hist
        for i, conv_layer_params in enumerate(self.conv_layers_params):
            layer = self.create_convolutional_layer(
                input=layer,
                **conv_layer_params,
                name=f"conv{i}",
            )

        # flat layer
        layer = self.create_flatten_layer(layer, name="flatten")
        layer = tf.concat(
            [layer, *self.x_horizon.values()] + ([self.x_issue_time] if self.issue_time_inputs else []),
            axis=1,
        )

        # fully connected layers
        for i, fc_layer_params in enumerate(self.fc_layers_params):
            if i == len(self.fc_layers_params) - 1:
                fc_layer_params = {
                    **fc_layer_params,
                    "num_outputs": len(self.time_horizon) * len(self.quantile_levels),
                }
                if self.monotonous_forecasts:
                    # the last fully connected layer defaults to softplus if
                    # monotonous forecasts are required
                    fc_layer_params.setdefault("activation", "softplus")
            layer = self.create_fully_connected_layer(
                input=layer,
                **fc_layer_params,
                name=f"fully_connected{i}",
            )

        layer = tf.reshape(
            layer,
            [-1, len(self.time_horizon), len(self.quantile_levels)],
            name="y_pred" + self.monotonous_forecasts * "_diff",
        )
        if self.monotonous_forecasts:
            # forecasts are monotonous because previous layer uses ReLU or Softplus
            self.y_pred = tf.cumsum(layer, axis=2, name="y_pred")
        else:
            self.y_pred = layer
        self.pinball_loss = self.create_pinball_loss_layer(y_true_reshaped, self.y_pred)
        self.regularization_loss = self.regularization_weight * tf.reduce_sum([tf.nn.l2_loss(w) for w in self.weights])
        self.optimizer = self.create_optimizer(self.pinball_loss + self.regularization_loss)
        self.initializer = tf.global_variables_initializer()

    def create_convolutional_layer(
            self,
            input: tf.Tensor,
            filter_shape: Tuple[int, int],
            num_filters: int,
            pooling_shape: Optional[Tuple[int, int]] = None,
            *,
            name: str,
    ) -> tf.Tensor:
        with tf.name_scope(name):
            num_input_channels = input.shape[3].value
            weights = tf.Variable(
                initial_value=tf.truncated_normal(
                    shape=[*filter_shape, num_input_channels, num_filters],
                    stddev=.05,
                ),
                name="weights",
            )
            self.weights.append(weights)
            biases = tf.Variable(
                initial_value=tf.constant(.05, shape=[num_filters]),
                name="biases",
            )
            layer = tf.nn.conv2d(
                input=input,
                filter=weights,
                strides=[1, 1, 1, 1],
                padding="SAME",
            ) + biases

            if pooling_shape is not None:
                layer = tf.nn.max_pool(
                    value=layer,
                    ksize=[1, *pooling_shape, 1],
                    strides=[1, *pooling_shape, 1],
                    padding="SAME",
                )

            layer = tf.nn.relu(layer)

            return layer

    def create_flatten_layer(
            self,
            input: tf.Tensor,
            *,
            name: str,
    ) -> tf.Tensor:
        with tf.name_scope(name):
            num_features = input.shape[1:].num_elements()
            layer = tf.reshape(input, [-1, num_features])
            return layer

    def create_fully_connected_layer(
            self,
            input: tf.Tensor,
            num_outputs: int,
            activation: Optional[str] = None,
            *,
            name: str,
    ) -> tf.Tensor:
        with tf.name_scope(name):
            num_inputs = input.shape[1].value
            weights = tf.Variable(
                initial_value=tf.truncated_normal(
                    shape=[num_inputs, num_outputs],
                    stddev=.05,
                ),
                name="weights",
            )
            self.weights.append(weights)
            biases = tf.Variable(
                initial_value=tf.constant(.05, shape=[num_outputs]),
                name="biases",
            )
            layer = input @ weights + biases

            activation_function = {
                None: lambda l: l,
                "relu": tf.nn.relu,
                "softplus": tf.nn.softplus,
            }[activation]
            layer = activation_function(layer)

            return layer

    def create_pinball_loss_layer(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
    ) -> tf.Tensor:
        quantile_levels_tensor = tf.constant(
            value=self.quantile_levels.fractions.reshape(1, 1, -1).astype(np.float32),
            name="quantile_levels",
        )
        with tf.name_scope("pinball_loss"):
            diff = tf.subtract(y_true, y_pred, name="diff")
            with tf.name_scope("is_overpredicted"):
                is_overpredicted = tf.to_float(diff < 0)
            layer = tf.reduce_mean(
                input_tensor=diff * (quantile_levels_tensor - is_overpredicted),
                name="mean_pinball_loss",
            )
            return layer

    def create_optimizer(self, loss: tf.Tensor) -> tf.Operation:
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return optimizer

    def get_additional_errors(self) -> Dict[str, Any]:
        return dict(regularization_loss=self.session.run(self.regularization_loss))


class FeatureExtractor:
    def __init__(self, estimator: STLQFCNNEstimator, data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> None:
        self.estimator = estimator
        self.data = estimator.resample_data(data)
        self.issue_times = issue_times
        self.issue_time_indices = self.data.index.get_indexer(issue_times)
        self._x_hist = {}
        self._x_horizon = {}

    @cached_getter
    def x_load(self) -> np.ndarray:
        return self.select_historical_patches(self.data.load, self.issue_time_indices)

    def get_x_hist(self, column: str) -> np.ndarray:
        if column not in self._x_hist:
            self._x_hist[column] = self.select_historical_patches(self.data[column], self.issue_time_indices)
        return self._x_hist[column]

    def get_x_horizon(self, column: str) -> np.ndarray:
        if column not in self._x_horizon:
            hor_indices = self.horizon_indices
            self._x_horizon[column] = self.data[column].loc[hor_indices.flat].values.reshape(hor_indices.shape)
        return self._x_horizon[column]

    @cached_getter
    def x_issue_time(self) -> np.ndarray:
        return np.stack([
            # day of week
            1. * self.issue_times.dayofweek,
            # day progress
            (60 * self.issue_times.hour + self.issue_times.minute) / 1440,
        ], axis=1)

    @cached_getter
    def y_load(self) -> np.ndarray:
        return self.data.load.loc[self.horizon_indices.flat].values.reshape(self.horizon_indices.shape)

    @cached_getter
    def horizon_indices(self) -> np.ndarray:
        return self.issue_times.values[:, None] + self.estimator.time_horizon.deltas.values

    def select_historical_patches(self, data_column: pd.DataFrame, issue_time_indices: np.ndarray) -> np.ndarray:
        """
        Select patches of historical data (before the issue time) by issue
        time indices.
        """
        # Extracting patches is approximately O(1), but further operations are heavy.
        # Output shape is (a, b) where a == n - prod(history_shape) + 1 and b == prod(history_shape),
        # where n is the length of resampled_data.
        patches = extract_patches(data_column.values, patch_shape=np.prod(self.estimator.history_shape))
        # The first patch belongs to the issue time equal to the timestamp of the
        # data point at index prod(history_shape), the second to last patch belongs
        # to the issue time equal to the timestamp of the last data point. The last
        # h patches are invalid where h is the length from the issue time to the last
        # lead time of the time horizon since the horizon has to be included in the data.
        # Select the patches and reshape them, this should include the heavy operation.
        return patches[issue_time_indices - np.prod(self.estimator.history_shape)] \
            .reshape(-1, *self.estimator.history_shape)

    def create_feed_dict(self, indices: slice = slice(None)) -> Dict[tf.Tensor, Any]:
        feed_dict = {
            self.estimator.x_load: self.x_load[indices],
        }
        for column, tensor in self.estimator.x_hist.items():
            feed_dict[tensor] = self.get_x_hist(column)[indices]
        for column, tensor in self.estimator.x_horizon.items():
            feed_dict[tensor] = self.get_x_horizon(column)[indices]
        if self.estimator.issue_time_inputs:
            feed_dict[self.estimator.x_issue_time] = self.x_issue_time[indices]
        return feed_dict
