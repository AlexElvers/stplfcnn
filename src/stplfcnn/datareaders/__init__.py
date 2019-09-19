import abc
import importlib
from typing import Type, TypeVar

import pandas as pd

T_DataReader = TypeVar("T_DataReader", bound="DataReader")


class DataReader(metaclass=abc.ABCMeta):
    """
    An abstract data reader.
    """

    @abc.abstractmethod
    def read_data(self) -> pd.DataFrame:
        """
        Read the data.
        """

    @classmethod
    def from_params(cls: Type[T_DataReader], **params) -> T_DataReader:
        """
        Create a data reader from params.
        """
        return cls(**params)


_class_modules = dict(
    PecanStreetReader="pecanstreet",
)


def get_class(class_name: str) -> Type[DataReader]:
    """
    Load the data reader class from its module.
    """
    module = importlib.import_module(f".{_class_modules[class_name]}", __package__)
    return getattr(module, class_name)
