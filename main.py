"""Main entry point for the rule-of-5-visualiser application."""

import matplotlib.figure as mfigure
import pandas as pd
from src.visualize import plot_simulation

from src.settings import POPULATION_SIZE, DistributionType
from src.simulate import (
    SimulationResults,
    simulate_rule_of_five,
)


def run_simulations(
    distribution_type: DistributionType, num_simulations: int, history_df: pd.DataFrame
) -> None:
    """Orchestrates the simulation and visualization of the rule-of-5 distribution."""
    simulation_results: SimulationResults = simulate_rule_of_five(
        distribution_type, num_simulations
    )
    fig: mfigure.Figure = plot_simulation(
        simulation_results.population_data_arr, simulation_results, num_simulations
    )
