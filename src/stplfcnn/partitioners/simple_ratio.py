from typing import Iterable

import pandas as pd

from . import Partitioner
from .. import IssueTimesPartition


class SimpleRatioPartitioner(Partitioner):
    def __init__(self, train_ratio: float) -> None:
        self.train_ratio = train_ratio

    def apply_to_issue_times(self, issue_times: pd.DatetimeIndex) -> Iterable[IssueTimesPartition]:
        split_point = round(self.train_ratio * len(issue_times))
        train_set = issue_times[:split_point]
        test_set = issue_times[split_point:]
        yield IssueTimesPartition(train_set, test_set)
