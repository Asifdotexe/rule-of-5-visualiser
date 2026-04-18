"""
Module responsible for visualisation of simulation results.
"""

import matplotlib.axes as maxes
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np

from src.settings import (
    DEFAULT_FIGURE_SIZE,
    FAILURE_COLOR,
    FIGURE_DPI,
    HISTOGRAM_BINS,
    MATPLOTLIB_STYLE,
    MAX_VISUALIZATIONS,
    MEDIAN_COLOR,
    POPULATION_COLOR,
    SAMPLE_SIZE,
    SUCCESS_COLOR,
)
from src.simulate import SimulationResults

plt.style.use(MATPLOTLIB_STYLE)


def plot_simulation(
    population: np.ndarray,
    simulation_results: SimulationResults,
    num_simulations: int,
) -> mfigure.Figure:
    """
    Plot the simulation results on a horizontal range visualization.

    :param population: The population data array.
    :param simulation_results: The simulation results containing sample data,
        medians, and success flags.
    :param num_simulations: The total number of simulations run.
    :return: The matplotlib Figure object.
    :raises ValueError: If num_simulations is less than 1.
    """
    if num_simulations < 1:
        raise ValueError("num_simulations must be greater than 0")

    fig: mfigure.Figure
    axes: maxes.Axes
    fig, axes = plt.subplots(figsize=DEFAULT_FIGURE_SIZE, dpi=FIGURE_DPI)

    axes.hist(
        population,
        bins=HISTOGRAM_BINS,
        color=POPULATION_COLOR,
        alpha=0.15,
        density=True,
        label="Population",
    )
    axes.axvline(
        simulation_results.population_median,
        color=MEDIAN_COLOR,
        linestyle="--",
        linewidth=2,
        label="True Median",
        zorder=2,
    )

    num_trials_to_visualize: int = min(num_simulations, MAX_VISUALIZATIONS)
    y_axis_maximum: float = axes.get_ylim()[1]
    trial_y_positions: np.ndarray = np.linspace(
        y_axis_maximum * 0.1, y_axis_maximum * 0.9, num_trials_to_visualize
    )

    for trial_index in range(num_trials_to_visualize):
        is_median_within_range: bool = simulation_results.is_median_within_range_arr[
            trial_index
        ]
        trial_color: str = SUCCESS_COLOR if is_median_within_range else FAILURE_COLOR
        trial_y_position: float = trial_y_positions[trial_index]

        sample_min_value: float = simulation_results.sample_mins[trial_index]
        sample_max_value: float = simulation_results.sample_maxs[trial_index]

        axes.plot(
            [sample_min_value, sample_max_value],
            [trial_y_position, trial_y_position],
            color=trial_color,
            linewidth=3,
            alpha=0.8,
            zorder=3,
        )

        cap_height: float = y_axis_maximum * 0.02
        axes.vlines(
            [sample_min_value, sample_max_value],
            trial_y_position - cap_height,
            trial_y_position + cap_height,
            color=trial_color,
            linewidth=2,
        )

        if num_simulations <= 10 or trial_index < 3:
            sample_points: np.ndarray = simulation_results.sample_data_arr[trial_index]
            axes.scatter(
                sample_points,
                [trial_y_position] * SAMPLE_SIZE,
                color=trial_color,
                s=40,
                edgecolors="white",
                linewidth=0.5,
                zorder=5,
            )

    axes.set_title(
        f"Rule of Five: {num_simulations} Trials",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axes.set_xlabel("Value")
    axes.get_yaxis().set_visible(False)
    axes.legend(loc="upper right", frameon=True, facecolor="white", fontsize="small")

    for spine in ["top", "right", "left"]:
        axes.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig
