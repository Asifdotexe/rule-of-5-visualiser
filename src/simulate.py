"""
Module responsible for simulation
"""

from dataclasses import dataclass

import numpy as np

from src.settings import DISTRIBUTIONS, POPULATION_SIZE, SAMPLE_SIZE, DistributionType


@dataclass
class SimulationResults:
    """Data container for simulation results."""

    population_data_arr: np.ndarray
    population_median: np.float64
    sample_data_arr: np.ndarray
    sample_mins: np.ndarray
    sample_maxs: np.ndarray
    is_median_within_range_arr: np.ndarray
    num_of_simulations_within_range: np.int64
    rule_of_five_success_rate: np.float64


def generate_population_data(
    distribution_type: DistributionType, size: int = POPULATION_SIZE
) -> np.ndarray:
    """
    Generate synthetic population data based on the given distribution type.

    :param distribution_type: The type of distribution to use.
    :param size: The number of data points to generate.
    :return: An array of generated data points.
    """
    return DISTRIBUTIONS[distribution_type](size)


def simulate_rule_of_five(
    distribution_type: DistributionType,
    num_of_simulations: int,
) -> SimulationResults:
    """
    Simulate population data based on the given distribution type.

    :param distribution_type: The type of distribution to use.
    :param num_of_simulations: The number of simulations to run.
    :return: Simulation results.
    """
    if num_of_simulations < 1:
        raise ValueError("num_of_simulations must be greater than 0")

    population_data_arr: np.ndarray = generate_population_data(distribution_type)
    population_median: np.float64 = np.median(population_data_arr)
    sample_data_arr: np.ndarray = np.random.choice(
        population_data_arr, size=(num_of_simulations, SAMPLE_SIZE)
    )

    sample_mins: np.ndarray = np.min(sample_data_arr, axis=1)
    sample_maxs: np.ndarray = np.max(sample_data_arr, axis=1)

    is_median_within_range_arr: np.ndarray = np.logical_and(
        sample_mins <= population_median, sample_maxs >= population_median
    )

    num_of_simulations_within_range: np.int64 = np.sum(is_median_within_range_arr)
    rule_of_five_success_rate: np.float64 = (
        num_of_simulations_within_range / num_of_simulations
    ) * 100

    return SimulationResults(
        population_data_arr=population_data_arr,
        population_median=population_median,
        sample_data_arr=sample_data_arr,
        sample_mins=sample_mins,
        sample_maxs=sample_maxs,
        is_median_within_range_arr=is_median_within_range_arr,
        num_of_simulations_within_range=num_of_simulations_within_range,
        rule_of_five_success_rate=rule_of_five_success_rate,
    )
