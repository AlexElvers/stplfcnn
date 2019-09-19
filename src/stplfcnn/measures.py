import numpy as np
import pandas as pd

from . import QuantileLevels, TimeHorizon
from .utils import cached_getter


class Measures:
    def __init__(
            self,
            time_horizon: TimeHorizon,
            quantile_levels: QuantileLevels,
            observed_data: pd.DataFrame,
            predicted_loads_per_issue_time: pd.DataFrame,
    ):
        self.time_horizon = time_horizon
        self.quantile_levels = quantile_levels
        self.observed_data = observed_data.resample(time_horizon.deltas.freq).sum()
        self.predicted_loads_per_issue_time = predicted_loads_per_issue_time

    @cached_getter
    def coverage(self) -> pd.DataFrame:
        """
        Calculate the coverage.
        """
        return self.diff < 0

    @cached_getter
    def pinball_loss(self) -> pd.DataFrame:
        """
        Calculate the pinball loss.
        """
        return self.diff * (self.quantile_levels.fractions - (self.diff < 0))

    @cached_getter
    def winkler_score(self) -> pd.DataFrame:
        """
        Calculate the Winkler score.
        """
        lower_quantiles = self.quantile_levels.intervals.lower
        upper_quantiles = self.quantile_levels.intervals.upper
        quantile_forecast_lower = pd.DataFrame(
            data=self.predicted_loads_per_issue_time[lower_quantiles].values,
            index=self.predicted_loads_per_issue_time.index,
            columns=self.quantile_levels.intervals.outside_coverage,
        )
        quantile_forecast_upper = pd.DataFrame(
            data=self.predicted_loads_per_issue_time[upper_quantiles].values,
            index=self.predicted_loads_per_issue_time.index,
            columns=self.quantile_levels.intervals.outside_coverage,
        )
        selected_observed_loads = self.observed_loads

        quantile_forecast_diff = quantile_forecast_upper - quantile_forecast_lower
        below_interval = -selected_observed_loads.values[:, None] + quantile_forecast_lower
        above_interval = selected_observed_loads.values[:, None] - quantile_forecast_upper
        return quantile_forecast_diff + (
                2 * np.maximum(np.maximum(below_interval, above_interval), 0)
                / (self.quantile_levels.intervals.outside_coverage / self.quantile_levels.denominator)
        )

    @cached_getter
    def diff(self) -> pd.DataFrame:
        """
        Calculate the difference between the observed and predicted data.
        """
        return self.observed_loads.values[:, None] - self.predicted_loads_per_issue_time

    @cached_getter
    def observed_loads(self) -> pd.DataFrame:
        """
        Select the observed loads by indices of predicted loads.
        """
        indices = self.predicted_loads_per_issue_time.index.get_level_values("lead_time")
        return self.observed_data.load.loc[indices]
