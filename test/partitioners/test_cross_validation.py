import pandas as pd

from stplfcnn.partitioners.cross_validation import CrossValidationPartitioner


def test_apply_to_issue_times():
    start = pd.Timestamp("2010-01-01")
    end = start + pd.Timedelta(4 * 30 - 1, unit="d")
    issue_times = pd.date_range(start, end, freq="1d")

    # num_repeats=1
    p = CrossValidationPartitioner(
        num_folds=4,
    )
    pp = list(p.apply_to_issue_times(issue_times))
    assert len(pp) == 4
    expected_test = pd.date_range("2010-01-01", "2010-01-30")
    assert set(pp[0].train) == set(issue_times.difference(expected_test))
    assert set(pp[0].test) == set(expected_test)
    expected_test = pd.date_range("2010-01-31", "2010-03-01")
    assert set(pp[1].train) == set(issue_times.difference(expected_test))
    assert set(pp[1].test) == set(expected_test)
    expected_test = pd.date_range("2010-03-02", "2010-03-31")
    assert set(pp[2].train) == set(issue_times.difference(expected_test))
    assert set(pp[2].test) == set(expected_test)
    expected_test = pd.date_range("2010-04-01", "2010-04-30")
    assert set(pp[3].train) == set(issue_times.difference(expected_test))
    assert set(pp[3].test) == set(expected_test)

    # num_repeats=2
    p = CrossValidationPartitioner(
        num_folds=4,
        num_repeats=2,
    )
    pp = list(p.apply_to_issue_times(issue_times))
    assert len(pp) == 4
    expected_test = (pd.date_range("2010-01-01", "2010-01-15")
                     .append(pd.date_range("2010-03-02", "2010-03-16")))
    assert set(pp[0].train) == set(issue_times.difference(expected_test))
    assert set(pp[0].test) == set(expected_test)
    expected_test = (pd.date_range("2010-01-16", "2010-01-30")
                     .append(pd.date_range("2010-03-17", "2010-03-31")))
    assert set(pp[1].train) == set(issue_times.difference(expected_test))
    assert set(pp[1].test) == set(expected_test)
    expected_test = (pd.date_range("2010-01-31", "2010-02-14")
                     .append(pd.date_range("2010-04-01", "2010-04-15")))
    assert set(pp[2].train) == set(issue_times.difference(expected_test))
    assert set(pp[2].test) == set(expected_test)
    expected_test = (pd.date_range("2010-02-15", "2010-03-01")
                     .append(pd.date_range("2010-04-16", "2010-04-30")))
    assert set(pp[3].train) == set(issue_times.difference(expected_test))
    assert set(pp[3].test) == set(expected_test)
