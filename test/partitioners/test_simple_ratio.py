import pandas as pd

from stplfcnn.partitioners.simple_ratio import SimpleRatioPartitioner


def test_apply_to_issue_times():
    start = pd.Timestamp("2010-01-01")
    end = start + pd.Timedelta(4 * 30 - 1, unit="d")
    issue_times = pd.date_range(start, end, freq="1d")

    p = SimpleRatioPartitioner(train_ratio=.75)
    pp = list(p.apply_to_issue_times(issue_times))
    assert len(pp) == 1
    assert set(pp[0].train) == set(issue_times[:90])
    assert set(pp[0].test) == set(issue_times[90:])
