"""
Place to house all the backend functions
"""

from enum import Enum

import numpy as np

POPULATION_SIZE: int = 10_000


class DistributionType(Enum):
    """Supported distribution types for population data generation."""

    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"


DISTRIBUTIONS = {
    DistributionType.NORMAL: lambda size: np.random.normal(loc=50, scale=15, size=size),
    DistributionType.UNIFORM: lambda size: np.random.uniform(low=0, high=100, size=size),
    DistributionType.EXPONENTIAL: lambda size: np.random.exponential(scale=20, size=size),
}


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


def simulate(
    distribution_type: DistributionType, num_of_simulations: int, current_history: list
):
    """
    Simulate population data based on the given distribution type.

    :param distribution_type: The type of distribution to use.
    :param num_of_simulations: The number of simulations to run.
    :param current_history: The current history of data points.
    :return: Updated plot, stats string, and history table.
    """
    # Keep population size constant rather than scaling it with num_of_simulations.
    # This is because the population represents the underlying truth we are trying to estimate.
    # Regardless of how many simulations we run, the Truth should be based on a large, stable population.
    # to ensure the true median is statistically significant
    population_data_arr: np.ndarray = generate_population_data(distribution_type)
    population_median: float = np.median(population_data_arr)
    # generate all samples at once in (N, 5) matrix
    sample_data_arr: np.ndarray = np.random.choice(
        population_data_arr, size=(num_of_simulations, 5)
    )

    sample_min: np.float64 = np.min(sample_data_arr, axis=1)
    sample_max: np.float64 = np.max(sample_data_arr, axis=1)

    # check if the population median likes within the range of the sample
    # the result is a boolean array of shape (num_simulations,)
    is_median_within_range_arr: np.ndarray = np.logical_and(sample_min <= population_median,
                                                            sample_max >= population_median)

    num_of_simulations_within_range: np.int64 = np.sum(is_median_within_range_arr)
    rule_of_five_success_rate: np.float64 = (num_of_simulations_within_range / num_of_simulations)*100
    return rule_of_five_success_rate


def main():
    print(generate_population_data(DistributionType.NORMAL, 10_000))


if __name__ == "__main__":
    main()
