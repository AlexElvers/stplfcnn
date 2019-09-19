import pathlib
from typing import Dict, Union, List

import matplotlib.pyplot as plt
import pandas as pd

from . import Plotter


class MatplotlibPlotter(Plotter):
    def plot(self, data: pd.DataFrame, predicted_loads: pd.DataFrame, estimator_name: str) -> None:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_title("load predictions")
        resampled_loads = data.load.resample("1h").sum()
        selected_lead_times = resampled_loads.index
        if self.selection is not None:
            if "start" in self.selection:
                selected_lead_times = selected_lead_times[selected_lead_times >= self.selection["start"]]
            if "stop" in self.selection:
                selected_lead_times = selected_lead_times[selected_lead_times < self.selection["stop"]]
        resampled_loads[selected_lead_times].plot(
            label="observed", ax=ax,
            linewidth=1.5, color="black",
        )
        predicted_loads.groupby("lead_time").mean().reindex(selected_lead_times).rename(
            columns=lambda c: f"${{{c}}}_{{/{self.quantile_levels.denominator}}}$",
        ).plot(
            ax=ax,
            style=[":", "--", "-.", "--", ":"], linewidth=1,
            color="blue", alpha=.8,
        )
        plt.legend()
        plt.tight_layout()
        plot_filename = f"{estimator_name}.png"
        if self.name:
            plot_filename = f"{self.name}_{plot_filename}"
        plt.savefig(self.output_path / plot_filename)
        plt.close()


def plot_errors_by_iteration(
        path: pathlib.Path,
        errors_by_iteration: Dict[str, Union[List[int], Dict[str, List[pd.DataFrame]]]],
) -> None:
    fig, ax = plt.subplots(nrows=3, figsize=(16, 16))
    fig.suptitle("Errors by Iteration")
    markers = "oxdvsPd>p*^"
    x = errors_by_iteration["iterations"]

    ax[0].set_title("Coverage")
    ax[0].set_ylim(-.05, 1.05)
    y = pd.concat([df.mean() for df in errors_by_iteration["train"]["coverage"]], axis=1).T
    for col in y.columns:
        # TODO denominator
        ax[0].axhline(col / 100, linestyle="--", color="grey")
    ax[0].plot(x[1:], y.mean(axis=1)[1:], color="blue", label="training avg.")
    for col, m in zip(y.columns, markers):
        ax[0].plot(x[1:], y[col][1:], color="blue", marker=m, label=f"training {col}")
    y = pd.concat([df.mean() for df in errors_by_iteration["test"]["coverage"]], axis=1).T
    ax[0].plot(x[1:], y.mean(axis=1)[1:], color="green", label="validation avg.")
    for col, m in zip(y.columns, markers):
        ax[0].plot(x[1:], y[col][1:], color="green", marker=m, label=f"validation {col}")
    ax[0].legend()

    ax[1].set_title("Pinball loss")
    y = pd.concat([df.mean() for df in errors_by_iteration["train"]["pinball_loss"]], axis=1).T
    ax[1].plot(x[1:], y.mean(axis=1)[1:], color="blue", label="training avg.")
    for col, m in zip(y.columns, markers):
        ax[1].plot(x[1:], y[col][1:], color="blue", marker=m, label=f"training {col}")
    y = pd.concat([df.mean() for df in errors_by_iteration["test"]["pinball_loss"]], axis=1).T
    ax[1].plot(x[1:], y.mean(axis=1)[1:], color="green", label="validation avg.")
    for col, m in zip(y.columns, markers):
        ax[1].plot(x[1:], y[col][1:], color="green", marker=m, label=f"validation {col}")
    ax[1].legend()

    ax[2].set_title("Winkler score")
    y = pd.concat([df.mean() for df in errors_by_iteration["train"]["winkler_score"]], axis=1).T
    ax[2].plot(x[1:], y.mean(axis=1)[1:], color="blue", label="training avg.")
    for col, m in zip(y.columns, markers):
        ax[2].plot(x[1:], y[col][1:], color="blue", marker=m, label=f"training {col}")
    y = pd.concat([df.mean() for df in errors_by_iteration["test"]["winkler_score"]], axis=1).T
    ax[2].plot(x[1:], y.mean(axis=1)[1:], color="green", label="validation avg.")
    for col, m in zip(y.columns, markers):
        ax[2].plot(x[1:], y[col][1:], color="green", marker=m, label=f"validation {col}")
    ax[2].legend()

    plt.savefig(path / "errors_by_iteration.png", dpi=200)
    plt.close()
