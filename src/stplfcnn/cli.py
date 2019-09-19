import pathlib
import time
from typing import Optional

import click
import numpy as np
import yaml

from .database import Database
from .measures import Measures
from .utils import ReferenceDumper, format_duration, get_path_kwargs_from_env
from .visualization.matplotlib import plot_errors_by_iteration


@click.group()
def cli():
    pass


def overwrite_cb(path: pathlib.Path) -> bool:
    return click.confirm(f"Overwrite {path}?")


@cli.command()
@click.option("--partitioner", "partitioner_name",
              required=True,
              help="base name of the partitioner params")
@click.option("--issue_times", "issue_times_name",
              required=True,
              help="base name of the issue times params")
@click.option("--output", "output_name",
              help="base name of the output (partitions) params (default: ISSUE_TIMES-PARTITIONER)")
@click.option("--overwrite",
              is_flag=True,
              help="overwrite output file if it exists")
def apply_partitioner(partitioner_name: str, issue_times_name: str, output_name: Optional[str], overwrite: bool):
    """
    Apply a partitioner to a range of issue times.
    """
    if output_name is None:
        output_name = f"{issue_times_name}-{partitioner_name}"
    db = Database(
        partitioner=partitioner_name,
        issue_times=issue_times_name,
        partitions=output_name,
        overwrite_cb=None if overwrite else overwrite_cb,
        **get_path_kwargs_from_env(),
    )
    partitioner = db.create_partitioner()
    issue_times = db.load_issue_times()
    partitions = list(partitioner.apply_to_issue_times(issue_times))
    db.dump_partitions(partitions)


@cli.command()
@click.option("--estimator", "estimator_name",
              required=True,
              help="base name of the estimator params")
@click.option("--data_reader", "data_reader_name",
              required=True,
              help="base name of the data reader params")
@click.option("--partitions", "partitions_name",
              required=True,
              help="base name of the partitions params")
@click.option("--overwrite",
              is_flag=True,
              help="overwrite output file if it exists")
def train(estimator_name: str, data_reader_name: str, partitions_name: str, overwrite: bool):
    """
    Train a model on a dataset with given rounds of training and testing data.
    """
    db = Database(
        estimator=estimator_name,
        data_reader=data_reader_name,
        partitions=partitions_name,
        overwrite_cb=None if overwrite else overwrite_cb,
        **get_path_kwargs_from_env(),
    )
    f_train(db)


def f_train(db: Database, dump_trained_state: bool = True, dump_predictions: bool = True) -> None:
    estimator = db.create_estimator()
    data_reader = db.create_data_reader()
    data = data_reader.read_data()
    print(f"{len(data)} data points from {data.index[0]} to {data.index[-1]}")
    partitions = db.load_partitions()
    print(f"Training {len(partitions)} partitions")
    db.dump_untrained_estimator(estimator)
    get_additional_errors = getattr(estimator, "get_additional_errors", None)

    time_horizon = estimator.time_horizon
    quantile_levels = estimator.quantile_levels

    for i, partition in enumerate(partitions):
        print(f"\n\nPartition {i}:")
        print(f"Training: {len(partition.train)} data points")
        print(f"Testing: {len(partition.test)} data points")
        with estimator:
            # train
            start_time = time.perf_counter()
            train_gen = estimator.train(data, partition.train)
            errors_by_iteration = {}
            if train_gen is not None:
                # training with status
                errors_iterations = []
                errors_train = dict(coverage=[], pinball_loss=[], winkler_score=[])
                errors_test = dict(coverage=[], pinball_loss=[], winkler_score=[])
                additional_errors = {}
                for j in train_gen:
                    print(f"\niter {j}  ({format_duration(time.perf_counter() - start_time)})")

                    predictions_train = estimator.predict(data, partition.train)
                    predictions_test = estimator.predict(data, partition.test)

                    measures_train = Measures(time_horizon, quantile_levels, data, predictions_train)
                    coverage_train = measures_train.coverage
                    pinball_loss_train = measures_train.pinball_loss
                    winkler_score_train = measures_train.winkler_score
                    measures_test = Measures(time_horizon, quantile_levels, data, predictions_test)
                    coverage_test = measures_test.coverage
                    pinball_loss_test = measures_test.pinball_loss
                    winkler_score_test = measures_test.winkler_score

                    errors_iterations.append(j)
                    errors_train["coverage"].append(coverage_train.groupby("offset").mean())
                    errors_train["pinball_loss"].append(pinball_loss_train.groupby("offset").mean())
                    errors_train["winkler_score"].append(winkler_score_train.groupby("offset").mean())
                    errors_test["coverage"].append(coverage_test.groupby("offset").mean())
                    errors_test["pinball_loss"].append(pinball_loss_test.groupby("offset").mean())
                    errors_test["winkler_score"].append(winkler_score_test.groupby("offset").mean())
                    if get_additional_errors:
                        for k, v in get_additional_errors().items():
                            additional_errors.setdefault(k, []).append(v)

                    print("\nTraining errors:")
                    print_measures(measures_train, show_winkler_score=False)
                    print("\nTesting errors:")
                    print_measures(measures_test, show_winkler_score=False)

                errors_by_iteration = dict(
                    iterations=errors_iterations,
                    train=errors_train,
                    test=errors_test,
                    **additional_errors,
                )

            duration = time.perf_counter() - start_time
            print("\nTraining duration:", format_duration(duration))

            # predict
            predictions_train = estimator.predict(data, partition.train)
            predictions_test = estimator.predict(data, partition.test)

            measures_train = Measures(time_horizon, quantile_levels, data, predictions_train)
            coverage_train = measures_train.coverage
            pinball_loss_train = measures_train.pinball_loss
            winkler_score_train = measures_train.winkler_score
            measures_test = Measures(time_horizon, quantile_levels, data, predictions_test)
            coverage_test = measures_test.coverage
            pinball_loss_test = measures_test.pinball_loss
            winkler_score_test = measures_test.winkler_score

            # show errors
            print("\nTraining errors:")
            print_measures(measures_train)
            print("\nTesting errors:")
            print_measures(measures_test)

            # dump trained state and predictions
            db.apply(partition_number=i)
            db.dump_training_duration(duration)
            if dump_trained_state:
                db.dump_trained_state(estimator)
            if errors_by_iteration:
                db.dump_errors_by_iteration(errors_by_iteration)
                plot_errors_by_iteration(db.model_partition_path, errors_by_iteration)
            db.apply(predictions="train")
            db.dump_errors(dict(
                coverage=coverage_train.groupby("offset").mean(),
                pinball_loss=pinball_loss_train.groupby("offset").mean(),
                winkler_score=winkler_score_train.groupby("offset").mean(),
            ))
            if dump_predictions:
                db.dump_predictions(predictions_train)
            db.apply(predictions="test")
            db.dump_errors(dict(
                coverage=coverage_test.groupby("offset").mean(),
                pinball_loss=pinball_loss_test.groupby("offset").mean(),
                winkler_score=winkler_score_test.groupby("offset").mean(),
            ))
            if dump_predictions:
                db.dump_predictions(predictions_test)


@cli.command()
@click.option("--model_dir",
              required=True,
              help="name of the model directory")
@click.option("--partition", "partition_number", type=int,
              required=True,
              help="partition number")
@click.option("--data_reader", "data_reader_name",
              required=True,
              help="base name of the data reader params")
@click.option("--issue_times", "issue_times_name",
              required=True,
              help="base name of the issue times params")
@click.option("--overwrite",
              is_flag=True,
              help="overwrite output file if it exists")
def predict(model_dir: str, partition_number: int, data_reader_name: str, issue_times_name: str, overwrite: bool):
    """
    Predict loads using a model of a given round on a dataset using the given
    issue times.
    """
    db = Database(
        model=model_dir,
        partition_number=partition_number,
        data_reader=data_reader_name,
        issue_times=issue_times_name,
        predictions=f"data_reader={data_reader_name},issue_times={issue_times_name}",
        overwrite_cb=None if overwrite else overwrite_cb,
        **get_path_kwargs_from_env(),
    )
    estimator = db.load_untrained_estimator()
    data_reader = db.create_data_reader()
    data = data_reader.read_data()
    print(f"{len(data)} data points from {data.index[0]} to {data.index[-1]}")
    issue_times = db.load_issue_times()
    print(len(issue_times), "issue times")

    time_horizon = estimator.time_horizon
    quantile_levels = estimator.quantile_levels

    with estimator:
        db.load_trained_state(estimator)
        predictions = estimator.predict(data, issue_times)
        measures = Measures(time_horizon, quantile_levels, data, predictions)
        print("\nErrors:")
        print_measures(measures)

        db.dump_errors(dict(
            coverage=measures.coverage.groupby("offset").mean(),
            pinball_loss=measures.pinball_loss.groupby("offset").mean(),
            winkler_score=measures.winkler_score.groupby("offset").mean(),
        ))
        db.dump_predictions(predictions)


def print_measures(
        measures: Measures,
        show_coverage: bool = True,
        show_pinball_loss: bool = True,
        show_winkler_score: bool = True,
) -> None:
    if show_coverage:
        print("Coverage:")
        print((measures.quantile_levels.denominator * measures.coverage.mean()).to_string())
    if show_pinball_loss:
        print("Pinball loss:")
        print(measures.pinball_loss.mean().to_string())
    if show_winkler_score:
        print("Winkler score:")
        print(measures.winkler_score.mean().to_string())


@cli.command()
@click.option("--data_reader", "data_reader_name",
              required=True,
              help="base name of the data reader params")
def data_summary(data_reader_name: str):
    """
    Show a summary of the data.
    """
    db = Database(data_reader=data_reader_name, **get_path_kwargs_from_env())
    data_reader = db.create_data_reader()
    data = data_reader.read_data()
    print("length:", len(data))
    print("start:", data.index[0])
    print("end:", data.index[-1])
    diffs = data.index[1:] - data.index[:-1]
    freq = diffs.min()
    print("freq:", freq)
    print("load:")
    print("  min:", data.load.min())
    print("  max:", data.load.max())
    print("  mean:", data.load.mean())
    print("  median:", data.load.median())
    print("  std:", data.load.std())
    print("  var:", data.load.var())
    print("columns:", ", ".join(data.columns))
    missing = np.where(diffs != freq)[0]
    if len(missing) != 0:
        print("missing entries between:")
        for gap_start, gap_end in zip(data.index[missing], data.index[missing + 1]):
            print(f"- {gap_start} and {gap_end}")


@cli.group("hyperopt")
def hyperopt_cli():
    """
    Hyperparameter optimization with Hyperopt.
    """


@hyperopt_cli.command()
@click.option("--params", "params_name",
              required=True,
              help="base name of the optimization params")
@click.option("--data_reader", "data_reader_name",
              required=True,
              help="base name of the data reader params")
@click.option("--partitions", "partitions_name",
              required=True,
              help="base name of the partitions params")
@click.option("--overwrite",
              is_flag=True,
              help="overwrite output file if it exists")
@click.option("--warm_start",
              is_flag=True,
              help="load Hyperopt trials")
@click.option("--dump_trained_state",
              is_flag=True,
              help="dump trained state of each model")
@click.option("--dump_predictions",
              is_flag=True,
              help="dump predictions of each model")
def optimize(
        params_name: str,
        data_reader_name: str,
        partitions_name: str,
        overwrite: bool,
        warm_start: bool,
        dump_trained_state: bool,
        dump_predictions: bool,
):
    """
    Run hyperparameter optimization.
    """
    db = Database(
        hyperopt=params_name,
        data_reader=data_reader_name,
        partitions=partitions_name,
        overwrite_cb=None if overwrite else overwrite_cb,
        **get_path_kwargs_from_env(),
    )
    optimizer = db.create_hyperparameter_optimizer()
    if warm_start:
        try:
            optimizer.trials = db.load_hyperopt_trials()
        except FileNotFoundError:
            print("Trials for warm start not found. Ignoring.")
    optimizer.dump_trained_state = dump_trained_state
    optimizer.dump_predictions = dump_predictions

    def dump(trials):
        if len(trials) > 0:
            db.dump_hyperopt_trials(trials)
            print(f"Dumped {len(trials)} trials.")
        else:
            print("No trials to dump.")

    try:
        for num_trials, num_ok_trials, best_trial in optimizer.optimize():
            l = max(len(str(num_trials)), len(str(num_ok_trials)))
            progress_ok = num_ok_trials / optimizer.max_evals
            progress_all = num_trials / optimizer.overall_max_evals
            click.secho(
                f"\nProgress:\n"
                f"  ok:  {num_ok_trials:{l}} trials  ({progress_ok:.2%} of max. {optimizer.max_evals})\n"
                f"  all: {num_trials:{l}} trials  ({progress_all:.2%} of max. {optimizer.overall_max_evals})\n"
                f"  ratio ok: {num_ok_trials / num_trials:.2%}\n"
                f"best:",
                fg="blue",
                bold=True,
            )
            click.secho(f"{yaml.dump(best_trial, Dumper=ReferenceDumper)}\n", fg="blue", bold=True)
            dump(optimizer.trials)
    except KeyboardInterrupt:
        dump(optimizer.trials)
