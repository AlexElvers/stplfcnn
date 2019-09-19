import abc
import importlib
from typing import Iterable, Type, TypeVar

import pandas as pd

from .. import IssueTimesPartition

T_Partitioner = TypeVar("T_Partitioner", bound="Partitioner")


class Partitioner(metaclass=abc.ABCMeta):
    """
    A partitioner divides the issue times into training and testing data.
    """

    @abc.abstractmethod
    def apply_to_issue_times(self, issue_times: pd.DatetimeIndex) -> Iterable[IssueTimesPartition]:
        """
        Split the issue times into one or multiple pairs of training and
        testing issue times.
        """

    @classmethod
    def from_params(cls: Type[T_Partitioner], **params) -> T_Partitioner:
        """
        Create a partitioner from params.
        """
        return cls(**params)


_class_modules = dict(
    CrossValidationPartitioner="cross_validation",
    SimpleRatioPartitioner="simple_ratio",
)


def get_class(class_name: str) -> Type[Partitioner]:
    """
    Load the data reader class from its module.
    """
    module = importlib.import_module(f".{_class_modules[class_name]}", __package__)
    return getattr(module, class_name)
