from typing import Iterable

import numpy as np
import pandas as pd

from . import Partitioner
from .. import IssueTimesPartition


class CrossValidationPartitioner(Partitioner):
    def __init__(self, num_folds: int, num_repeats: int = 1) -> None:
        # TODO num_folds and num_repeats are ambiguous names
        self.num_folds = num_folds
        self.num_repeats = num_repeats

    def apply_to_issue_times(self, issue_times: pd.DatetimeIndex) -> Iterable[IssueTimesPartition]:
        num_partitions = self.num_folds * self.num_repeats
        partition_size = len(issue_times) / num_partitions
        partitions = []
        for i in range(num_partitions):
            start = round(i * partition_size)
            stop = round((i + 1) * partition_size)
            partitions.append(issue_times[start:stop])
        for i in range(self.num_folds):
            test_set = pd.DatetimeIndex(np.concatenate(partitions[i::self.num_folds]))
            train_set = issue_times.difference(test_set)
            yield IssueTimesPartition(train_set, test_set)
