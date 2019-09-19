import abc
import pathlib
from typing import Optional, TypeVar

import tensorflow as tf
import yaml

from . import QuantileEstimator
from .. import QuantileLevels, TimeHorizon

T_Estimator = TypeVar("T_Estimator", bound="QuantileEstimator")


class TensorFlowEstimator(QuantileEstimator):
    """
    A TensorFlow estimator has a graph and a session using that graph.
    """

    def __init__(
            self,
            time_horizon: TimeHorizon,
            quantile_levels: QuantileLevels,
            tf_seed: Optional[int] = None,
    ) -> None:
        super().__init__(time_horizon, quantile_levels)
        self.tf_seed = tf_seed

        self.graph: Optional[tf.Graph] = None
        self.session: Optional[tf.Session] = None
        self.saver: Optional[tf.train.Saver] = None
        self.config_path: Optional[pathlib.Path] = pathlib.Path("tensorflow.yaml")

    def create_graph(self) -> None:
        """
        Create the TensorFlow graph and its nodes. Assign the graph to the
        `graph` attribute.
        """
        if self.graph is not None:
            raise ValueError("graph already exists")
        with tf.Graph().as_default() as self.graph:
            if self.tf_seed is not None:
                tf.set_random_seed(self.tf_seed)
            self.create_graph_nodes()
            if self.saver is None:
                self.saver = tf.train.Saver()

    @abc.abstractmethod
    def create_graph_nodes(self) -> None:
        """
        Create the nodes of the TensorFlow graph.

        This method should only be called by `create_graph`.
        """

    def load_config(self) -> tf.ConfigProto:
        """
        Load config from file.
        """
        if self.config_path is not None and self.config_path.exists():
            with open(self.config_path) as f:
                config_params = yaml.safe_load(f)
                if config_params is None:
                    return None
                return tf.ConfigProto(**config_params)
        return None

    def create_session(self) -> None:
        """
        Create the TensorFlow session that is using the graph.
        """
        if self.session is not None:
            raise ValueError("session already exists")
        if self.graph is None:
            raise ValueError("graph has to be created before the session")
        self.session = tf.Session(graph=self.graph, config=self.load_config())

    def close_session(self) -> None:
        """
        Close and delete the TensorFlow session.
        """
        if self.session is not None:
            self.session.close()
        self.session = None

    def __enter__(self: T_Estimator) -> T_Estimator:
        """
        Create the TensorFlow graph and the session.
        """
        if self.graph is None:
            self.create_graph()
        self.create_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Close and delete the TensorFlow session.
        """
        self.close_session()

    def load_trained_state(self, path: pathlib.Path) -> None:
        self.saver.restore(self.session, str(path / "state"))

    def dump_trained_state(self, path: pathlib.Path) -> None:
        self.saver.save(self.session, str(path / "state"))
