import abc
import datetime
import importlib
import pathlib
from typing import Dict, Optional, Type

import pandas as pd

from .. import QuantileLevels


class Plotter(metaclass=abc.ABCMeta):
    """
    A plotter plots data from estimators in a single or multiple documents.
    """

    def __init__(
            self,
            output_path: pathlib.Path,
            quantile_levels: QuantileLevels,
            selection: Optional[Dict[str, datetime.datetime]],
            name: Optional[str] = None,
    ) -> None:
        self.output_path = output_path
        self.quantile_levels = quantile_levels
        self.selection = selection
        self.name = name

    @abc.abstractmethod
    def plot(self, data: pd.DataFrame, predicted_loads: pd.DataFrame, estimator_name: str) -> None:
        """
        Add a plot to the plotter.
        """

    def save(self) -> None:
        """
        Save the document(s) if not done yet.
        """


_class_modules = dict(
    MatplotlibPlotter="matplotlib",
)


def get_class(class_name: str) -> Type[Plotter]:
    """
    Load the plotter class from its module.
    """
    module = importlib.import_module(f".{_class_modules[class_name]}", __package__)
    return getattr(module, class_name)
