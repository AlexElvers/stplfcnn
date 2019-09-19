import pathlib
from typing import Optional, Sequence, Union

import pandas as pd

from . import DataReader


class PecanStreetReader(DataReader):
    """
    A data reader for the Pecan Street data.
    """

    def __init__(
            self,
            base_path: pathlib.Path,
            city: str,
            resolution: str,
            aggregation: int = 1,
            columns: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    ) -> None:
        """
        Initialize the data reader.

        In the resulting data frame, the selected columns are summed up. If no
        columns are selected, all columns are summed up.
        """
        self.base_path = base_path
        self.city = city
        self.resolution = resolution
        self.aggregation = aggregation
        if isinstance(columns, (int, str)):
            self.columns = [columns]
        else:
            self.columns = columns

    def read_data(self) -> pd.DataFrame:
        city_path = self.base_path / self.city
        agg_str = f"_agg_{self.aggregation}" if self.aggregation > 1 else ""
        load_file = city_path / self.resolution / f"{self.city}_{self.resolution}_load{agg_str}.csv"
        weather_file = city_path / f"{self.city}_weather.csv"

        load_data: pd.DataFrame = pd.read_csv(load_file)
        load_data.rename(columns={load_data.columns[0]: "timestamp"}, inplace=True)
        load_data.timestamp = pd.to_datetime(load_data.timestamp)
        load_data.set_index("timestamp", inplace=True)
        column_sel = self.columns if self.columns is not None else slice(None)
        load_data = load_data.loc[:, column_sel].sum(axis=1).rename("load")

        weather_data: pd.DataFrame = pd.read_csv(weather_file)
        weather_data.rename(columns={weather_data.columns[0]: "timestamp"}, inplace=True)
        weather_data.timestamp = pd.to_datetime(weather_data.timestamp)
        weather_data.set_index("timestamp", inplace=True)
        if self.resolution == "15min":
            extra_item = pd.DataFrame(index=[weather_data.index[-1] + pd.Timedelta("1h")])
            weather_data = weather_data.append(extra_item, sort=False).resample("15min").interpolate().iloc[:-1]

        if not load_data.index.equals(weather_data.index):
            raise ValueError("indices of load and weather data are not equal")

        data = pd.concat([load_data, weather_data], axis=1)
        return data

    @classmethod
    def from_params(cls, **params) -> "PecanStreetReader":
        params["base_path"] = pathlib.Path(params["base_path"])
        return cls(**params)
