import copy
import datetime
import pathlib

import pandas as pd
import pytest
import yaml

from stplfcnn import IssueTimesPartition, QuantileLevels, TimeHorizon
from stplfcnn.database import Database
from stplfcnn.datareaders.pecanstreet import PecanStreetReader
from stplfcnn.estimators.naive import NaiveEstimator
from stplfcnn.hyperopt.stlqfcnn import STLQFCNNHyperparameterOptimizer
from stplfcnn.partitioners.cross_validation import CrossValidationPartitioner
from stplfcnn.utils import Reference, ReferenceLoader


def test_paths():
    # test for exceptions
    db = Database()
    pytest.raises(AttributeError, getattr, db, "issue_times_params_path")
    pytest.raises(AttributeError, getattr, db, "partitioner_params_path")
    pytest.raises(AttributeError, getattr, db, "partitions_params_path")
    pytest.raises(AttributeError, getattr, db, "data_reader_params_path")
    pytest.raises(AttributeError, getattr, db, "estimator_params_path")
    pytest.raises(AttributeError, getattr, db, "time_horizon_params_path")
    pytest.raises(AttributeError, getattr, db, "quantile_levels_params_path")
    pytest.raises(AttributeError, getattr, db, "model_path")
    pytest.raises(AttributeError, getattr, db, "model_estimator_path")
    pytest.raises(AttributeError, getattr, db, "model_partition_path")
    pytest.raises(AttributeError, getattr, db, "predictions_path")
    pytest.raises(AttributeError, getattr, db, "predictions_pickle_path")


def test_apply():
    db = Database()
    assert not hasattr(db, "data_reader")
    assert not hasattr(db, "estimator")
    assert not hasattr(db, "issue_times")
    assert not hasattr(db, "model")
    assert not hasattr(db, "partition_number")
    assert not hasattr(db, "partitioner")
    assert not hasattr(db, "partitions")
    assert not hasattr(db, "quantile_levels")
    assert not hasattr(db, "time_horizon")
    db.apply(estimator="estimator", data_reader="data_reader", partitions="partitions")
    assert hasattr(db, "estimator")
    assert hasattr(db, "data_reader")
    assert hasattr(db, "partitions")
    assert hasattr(db, "model")
    with pytest.raises(ValueError) as e:
        db.apply(foo="bar")
    assert str(e.value) == "some keys do not exist: foo"


def test_load_issue_times(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    params = dict(
        freq="1d",
        issue_times=[
            dict(start=datetime.date(2010, 1, 1), end=datetime.date(2010, 1, 30)),
            dict(start=datetime.date(2010, 3, 2), end=datetime.date(2010, 3, 31)),
        ],
    )
    path = pathlib.Path("params/issue_times/foo.yaml")
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(params))

    db = Database(issue_times="foo")
    issue_times = db.load_issue_times()
    assert isinstance(issue_times, pd.DatetimeIndex)
    assert list(issue_times) == list(
        pd.date_range("2010-01-01", "2010-01-30")
        | pd.date_range("2010-03-02", "2010-03-31")
    )


def test_dump_issue_times(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    issue_times = pd.date_range("2010-01-01", "2010-01-30") | pd.date_range("2010-03-02", "2010-03-31")
    db = Database(issue_times="foo")
    db.dump_issue_times(issue_times)

    path = pathlib.Path("params/issue_times/foo.yaml")
    params = yaml.safe_load(path.read_text())
    assert params == dict(
        freq="D",
        issue_times=[
            dict(start=datetime.datetime(2010, 1, 1), end=datetime.datetime(2010, 1, 30)),
            dict(start=datetime.datetime(2010, 3, 2), end=datetime.datetime(2010, 3, 31)),
        ],
    )


def test_create_partitioner(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    params = dict(
        type="CrossValidationPartitioner",
        num_folds=4,
    )
    path = pathlib.Path("params/partitioners/foo.yaml")
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(params))

    db = Database(partitioner="foo")
    partitioner = db.create_partitioner()
    assert isinstance(partitioner, CrossValidationPartitioner)


def test_load_partitions(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    params_issue_times1 = [
        dict(start=datetime.date(2010, 1, 1), end=datetime.date(2010, 1, 30)),
        dict(start=datetime.date(2010, 3, 2), end=datetime.date(2010, 3, 31)),
    ]
    params_issue_times2 = [
        dict(start=datetime.date(2010, 1, 31), end=datetime.date(2010, 3, 1)),
        dict(start=datetime.date(2010, 4, 1), end=datetime.date(2010, 4, 30)),
    ]
    params = dict(
        freq="1d",
        partitions=[
            dict(
                train=copy.deepcopy(params_issue_times1),
                test=copy.deepcopy(params_issue_times2),
            ),
            dict(
                train=copy.deepcopy(params_issue_times2),
                test=copy.deepcopy(params_issue_times1),
            ),
        ],
    )
    path = pathlib.Path("params/partitions/foo.yaml")
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(params))

    db = Database(partitions="foo")
    partitions = db.load_partitions()
    assert len(partitions) == 2
    issue_times1 = list(pd.date_range("2010-01-01", "2010-01-30") | pd.date_range("2010-03-02", "2010-03-31"))
    issue_times2 = list(pd.date_range("2010-01-31", "2010-03-01") | pd.date_range("2010-04-01", "2010-04-30"))
    assert list(partitions[0].train) == issue_times1
    assert list(partitions[0].test) == issue_times2
    assert list(partitions[1].train) == issue_times2
    assert list(partitions[1].test) == issue_times1


def test_dump_partitions(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    issue_times1 = pd.date_range("2010-01-01", "2010-01-30") | pd.date_range("2010-03-02", "2010-03-31")
    issue_times2 = pd.date_range("2010-01-31", "2010-03-01") | pd.date_range("2010-04-01", "2010-04-30")
    partitions = [
        IssueTimesPartition(train=issue_times1, test=issue_times2),
        IssueTimesPartition(train=issue_times2, test=issue_times1),
    ]
    db = Database(partitions="foo")
    db.dump_partitions(partitions)

    path = pathlib.Path("params/partitions/foo.yaml")
    params = yaml.safe_load(path.read_text())

    params_issue_times1 = [
        dict(start=datetime.datetime(2010, 1, 1), end=datetime.datetime(2010, 1, 30)),
        dict(start=datetime.datetime(2010, 3, 2), end=datetime.datetime(2010, 3, 31)),
    ]
    params_issue_times2 = [
        dict(start=datetime.datetime(2010, 1, 31), end=datetime.datetime(2010, 3, 1)),
        dict(start=datetime.datetime(2010, 4, 1), end=datetime.datetime(2010, 4, 30)),
    ]
    assert params == dict(
        freq="D",
        partitions=[
            dict(
                train=params_issue_times1,
                test=params_issue_times2,
            ),
            dict(
                train=params_issue_times2,
                test=params_issue_times1,
            ),
        ],
    )


def test_create_data_reader(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    params = dict(
        type="PecanStreetReader",
        base_path="data/pecanstreet",
        city="austin",
        resolution="15min",
    )
    path = pathlib.Path("params/data_readers/foo.yaml")
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(params))

    db = Database(data_reader="foo")
    data_reader = db.create_data_reader()
    assert isinstance(data_reader, PecanStreetReader)
    assert data_reader.base_path == pathlib.Path("data/pecanstreet")
    assert data_reader.city == "austin"
    assert data_reader.resolution == "15min"
    assert data_reader.aggregation == 1
    assert data_reader.columns is None


def test_create_estimator(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    params = dict(
        type="NaiveEstimator",
        time_horizon=dict(start=0, stop=600, step=60),
        quantile_levels=dict(numerators=[1, 2, 8, 9], denominator=10),
        load_offset=1440,
        noise_type="multiplicative",
    )
    path = pathlib.Path("params/estimators/foo.yaml")
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(params))

    db = Database(estimator="foo")
    estimator = db.create_estimator()
    assert isinstance(estimator, NaiveEstimator)
    assert estimator.time_horizon == TimeHorizon(0, 600, 60)
    assert estimator.quantile_levels == QuantileLevels([1, 2, 8, 9], 10)
    assert estimator.load_offset == 1440
    assert estimator.noise_type == "multiplicative"

    path = pathlib.Path("params/estimators/bar.yaml")
    path.write_text("\n".join([
        "type: NaiveEstimator",
        "time_horizon: !ref bar_horizon",
        "quantile_levels: !ref bar_quantiles",
        "load_offset: 1440",
        "noise_type: multiplicative",
    ]))
    path = pathlib.Path("params/time_horizons/bar_horizon.yaml")
    path.parent.mkdir(parents=True)
    path.write_text("\n".join([
        "start: 0",
        "stop: 1440",
        "step: 60",
    ]))
    path = pathlib.Path("params/quantile_levels/bar_quantiles.yaml")
    path.parent.mkdir(parents=True)
    path.write_text("\n".join([
        "[1, 5, 10, 25, 50, 75, 90, 95, 99]",
    ]))

    db = Database(estimator="bar")
    estimator = db.create_estimator()
    assert isinstance(estimator, NaiveEstimator)
    assert estimator.time_horizon == TimeHorizon(0, 1440, 60)
    assert estimator.quantile_levels == QuantileLevels([1, 5, 10, 25, 50, 75, 90, 95, 99])
    assert estimator.load_offset == 1440
    assert estimator.noise_type == "multiplicative"


def test_dump_estimator_params(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    params = dict(
        type="STLQFCNNEstimator",
        time_horizon=Reference("time_horizon"),
        quantile_levels=Reference("quantile_levels"),
        history_shape=[7, 24],
    )
    db = Database(estimator="foo")
    db.dump_estimator_params(params)

    path = pathlib.Path("params/estimators/foo.yaml")
    observed_params = yaml.load(path.read_text(), ReferenceLoader)

    assert observed_params == params


def test_create_hyperparameter_optimizer(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    path = pathlib.Path("params/hyperopt/foo.yaml")
    path.parent.mkdir(parents=True)
    path.write_text("\n".join([
        "type: STLQFCNNHyperparameterOptimizer",
        "time_horizon: time_horizon",
        "quantile_levels: quantile_levels",
        "max_evals: 1000",
        "space: !hp.uniform [0, 5]",
    ]))

    db = Database(hyperopt="foo")
    optimizer = db.create_hyperparameter_optimizer()
    assert isinstance(optimizer, STLQFCNNHyperparameterOptimizer)

    path.write_text("\n".join([
        "type: STLQFCNNHyperparameterOptimizer",
        "time_horizon: time_horizon",
        "quantile_levels: quantile_levels",
        "max_evals: 1000",
        "space: !hp.uniform {low: 0, high: 5}",
    ]))

    db = Database(hyperopt="foo")
    optimizer = db.create_hyperparameter_optimizer()
    assert isinstance(optimizer, STLQFCNNHyperparameterOptimizer)
