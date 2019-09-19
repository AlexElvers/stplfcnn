import io
from contextlib import redirect_stdout

from hyperopt import hp

from stplfcnn import cli
from stplfcnn.database import Database
from stplfcnn.hyperopt import BadEstimatorParams
from stplfcnn.hyperopt.stlqfcnn import STLQFCNNHyperparameterOptimizer
from stplfcnn.utils import Reference


def test_build_estimator_params():
    optimizer = STLQFCNNHyperparameterOptimizer(
        db=Database(),
        time_horizon="time_horizon",
        quantile_levels="quantile_levels",
        max_evals=1000,
        space=None,
    )
    params = optimizer.build_estimator_params(
        history_shape=[7, 24],
        conv_layers_params=[
            dict(filter_shape=(3, 5), num_filters=25),
        ],
        fc_layers_params=[
            dict(num_outpus=64, activation="relu"),
            dict(),
        ],
    )
    assert params == dict(
        type="STLQFCNNEstimator",
        time_horizon=Reference("time_horizon"),
        quantile_levels=Reference("quantile_levels"),
        history_shape=[7, 24],
        conv_layers_params=[
            dict(filter_shape=(3, 5), num_filters=25),
        ],
        fc_layers_params=[
            dict(num_outpus=64, activation="relu"),
            dict(),
        ],
    )


def test_warm_start(tmpdir, monkeypatch, mocker):
    monkeypatch.chdir(tmpdir)
    mocker.patch.object(cli, "f_train")
    mocker.patch.object(STLQFCNNHyperparameterOptimizer, "calculate_loss_and_duration", return_value=(0, 0, 0))
    mocker.patch.object(Database, "dump_estimator_params")
    db = Database(hyperopt="test")
    space = dict(
        history_shape=[hp.choice("a", [7, 14]), 24],
        learning_rate=hp.loguniform("lr", -13, -4),
        conv_layers_params=[
            dict(
                filter_shape=[hp.choice("b", [1, 3, 5]), hp.choice("c", [3, 5, 7, 9])],
                num_filters=hp.choice("d", [4, 8, 16, 32, 64]),
            ),
        ],
        fc_layers_params=[
            dict(num_outpus=64, activation="relu"),
            dict(),
        ],
    )
    optimizer = STLQFCNNHyperparameterOptimizer(
        db=db,
        time_horizon="time_horizon",
        quantile_levels="quantile_levels",
        max_evals=1,
        space=space,
    )
    f = io.StringIO()
    with redirect_stdout(f):
        for _ in optimizer.optimize():
            pass
    assert cli.f_train.call_count == 1
    assert len(optimizer.trials) == 1

    cli.f_train.reset_mock()
    optimizer.max_evals = optimizer.overall_max_evals = 4
    with redirect_stdout(f):
        for _ in optimizer.optimize():
            pass
    assert cli.f_train.call_count == 3
    assert len(optimizer.trials) == 4


def test_optimize(tmpdir, monkeypatch, mocker):
    orig_build_estimator_params = STLQFCNNHyperparameterOptimizer.build_estimator_params

    def mock_build_estimator_params(self, **kwargs):
        if fail_predicate():
            raise BadEstimatorParams
        return orig_build_estimator_params(self, **kwargs)

    monkeypatch.chdir(tmpdir)
    mocker.patch.object(cli, "f_train")
    mocker.patch.object(STLQFCNNHyperparameterOptimizer, "calculate_loss_and_duration", return_value=(0, 0, 0))
    monkeypatch.setattr(STLQFCNNHyperparameterOptimizer, "build_estimator_params", mock_build_estimator_params)
    mocker.patch.object(Database, "dump_estimator_params")
    db = Database(hyperopt="test")
    space = dict(
        history_shape=[hp.choice("a", [7, 14]), 24],
        learning_rate=hp.loguniform("lr", -13, -4),
        conv_layers_params=[
            dict(
                filter_shape=[hp.choice("b", [1, 3, 5]), hp.choice("c", [3, 5, 7, 9])],
                num_filters=hp.choice("d", [4, 8, 16, 32, 64]),
            ),
        ],
        fc_layers_params=[
            dict(num_outpus=64, activation="relu"),
            dict(),
        ],
    )

    # all ok
    optimizer = STLQFCNNHyperparameterOptimizer(
        db=db,
        time_horizon="time_horizon",
        quantile_levels="quantile_levels",
        max_evals=37,
        max_chunk_size=10,
        space=space,
    )
    fail_predicate = lambda: False
    expected_trials = [
        (10, 10),
        (20, 20),
        (30, 30),
        (37, 37),
    ]
    f = io.StringIO()
    with redirect_stdout(f):
        for i, (num_trials, num_ok_trials, best_trial) in enumerate(optimizer.optimize()):
            assert (num_trials, num_ok_trials) == expected_trials[i]

    # all fail
    optimizer = STLQFCNNHyperparameterOptimizer(
        db=db,
        time_horizon="time_horizon",
        quantile_levels="quantile_levels",
        max_evals=37,
        max_chunk_size=10,
        space=space,
    )
    fail_predicate = lambda: True
    expected_trials = [
        (10, 0),
        (20, 0),
        (30, 0),
        (37, 0),
    ]
    f = io.StringIO()
    with redirect_stdout(f):
        for i, (num_trials, num_ok_trials, best_trial) in enumerate(optimizer.optimize()):
            assert (num_trials, num_ok_trials) == expected_trials[i]

    # 1/3 fail, stop by overall_max_evals
    optimizer = STLQFCNNHyperparameterOptimizer(
        db=db,
        time_horizon="time_horizon",
        quantile_levels="quantile_levels",
        max_evals=37,
        overall_max_evals_factor=1,
        max_chunk_size=10,
        space=space,
    )
    fail_predicate = lambda: len(optimizer.trials) % 3 == 1  # trails contain unfinished tasks
    expected_trials = [
        (10, 6),
        (20, 13),
        (30, 20),
        (37, 24),
    ]
    f = io.StringIO()
    with redirect_stdout(f):
        for i, (num_trials, num_ok_trials, best_trial) in enumerate(optimizer.optimize()):
            assert (num_trials, num_ok_trials) == expected_trials[i]

    # 1/3 fail, stop by max_evals
    optimizer = STLQFCNNHyperparameterOptimizer(
        db=db,
        time_horizon="time_horizon",
        quantile_levels="quantile_levels",
        max_evals=37,
        overall_max_evals_factor=2,
        max_chunk_size=10,
        space=space,
    )
    fail_predicate = lambda: len(optimizer.trials) % 3 == 1
    expected_trials = [
        (10, 6),
        (20, 13),
        (30, 20),
        (40, 26),
        (50, 33),
        (56, 37),
    ]
    f = io.StringIO()
    with redirect_stdout(f):
        for i, (num_trials, num_ok_trials, best_trial) in enumerate(optimizer.optimize()):
            assert (num_trials, num_ok_trials) == expected_trials[i]
