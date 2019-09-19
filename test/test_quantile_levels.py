import numpy as np

from stplfcnn import QuantileLevels


def test_quantile_levels():
    ql = QuantileLevels([25, 50, 75])
    assert np.all(ql.numerators == [25, 50, 75])
    assert ql.denominator == 100
    assert np.all(ql.fractions == [.25, .5, .75])
    assert np.all(ql.percentiles == [25, 50, 75])
    assert np.all(ql.intervals.outside_coverage == [50])

    ql = QuantileLevels([1, 2, 5, 10, 20, 50, 80, 95, 99])
    assert np.all(ql.numerators == [1, 2, 5, 10, 20, 50, 80, 95, 99])
    assert ql.denominator == 100
    assert np.all(ql.fractions == [.01, .02, .05, .10, .20, .50, .80, .95, .99])
    assert np.all(ql.percentiles == [1, 2, 5, 10, 20, 50, 80, 95, 99])
    assert np.all(ql.intervals.outside_coverage == [2, 10, 40])

    ql = QuantileLevels([1, 3, 5, 6, 8, 9], 10)
    assert np.all(ql.numerators == [1, 3, 5, 6, 8, 9])
    assert ql.denominator == 10
    assert np.all(ql.fractions == [.1, .3, .5, .6, .8, .9])
    assert np.all(ql.percentiles == [10, 30, 50, 60, 80, 90])
    assert np.all(ql.intervals.outside_coverage == [2])

    ql = QuantileLevels([50, 75])
    assert np.all(ql.numerators == [50, 75])
    assert ql.denominator == 100
    assert np.all(ql.fractions == [.5, .75])
    assert np.all(ql.percentiles == [50, 75])
    assert np.all(ql.intervals.outside_coverage == [])

    ql = QuantileLevels([50])
    assert np.all(ql.numerators == [50])
    assert ql.denominator == 100
    assert np.all(ql.fractions == [.5])
    assert np.all(ql.percentiles == [50])
    assert np.all(ql.intervals.outside_coverage == [])

    ql = QuantileLevels([])
    assert np.all(ql.numerators == [])
    assert ql.denominator == 100
    assert np.all(ql.fractions == [])
    assert np.all(ql.percentiles == [])
    assert np.all(ql.intervals.outside_coverage == [])
