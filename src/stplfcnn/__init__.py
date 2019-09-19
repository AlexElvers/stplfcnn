from typing import Sequence

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class TimeHorizon:
    """
    A time horizon is divided into lead times. The values are offsets from the
    issue time in minutes.

    The start value is included, the stop value is excluded.
    """
    start: int
    stop: int
    step: int
    deltas: pd.TimedeltaIndex = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.deltas = pd.timedelta_range(
            start=f"{self.start} minutes",
            end=f"{self.stop} minutes",
            freq=f"{self.step}min",
            closed="left",
        )

    def __len__(self) -> int:
        return len(self.deltas)


@dataclass(frozen=True)
class IntervalLevels:
    """
    A set of central prediction intervals.
    """
    coverage: np.ndarray
    outside_coverage: np.ndarray = field(repr=False)
    lower: np.ndarray = field(repr=False)
    upper: np.ndarray = field(repr=False)


@dataclass(init=False)
class QuantileLevels:
    """
    A set of quantile levels and central prediction intervals.

    The nominal coverage rate of prediction intervals is 1-beta, so
    lower alpha = beta/2
    """
    numerators: np.ndarray
    denominator: int
    fractions: np.ndarray = field(repr=False)
    percentiles: np.ndarray = field(repr=False)
    intervals: IntervalLevels = field(repr=False)

    def __init__(self, numerators: Sequence[int], denominator: int = 100) -> None:
        if list(numerators) != sorted(numerators):
            raise ValueError("numerators have to be in ascending order")
        self.numerators = np.asarray(numerators)
        self.denominator = denominator
        self.fractions = self.numerators / denominator
        self.percentiles = 100 * self.numerators / denominator
        central_intervals = np.reshape([
            [denominator - 2 * a, 2 * a, a, denominator - a]
            for a in numerators
            if 2 * a < denominator and denominator - a in numerators
        ], newshape=(-1, 4))
        self.intervals = IntervalLevels(
            coverage=central_intervals[:, 0],
            outside_coverage=central_intervals[:, 1],
            lower=central_intervals[:, 2],
            upper=central_intervals[:, 3],
        )

    def __eq__(self, other) -> bool:
        if type(other) is QuantileLevels:
            return (np.all(self.numerators == other.numerators)
                    and self.denominator == other.denominator)
        return False

    def __len__(self) -> int:
        return len(self.numerators)


@dataclass
class IssueTimesPartition:
    """
    A pair of training and testing issue times.
    """
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
