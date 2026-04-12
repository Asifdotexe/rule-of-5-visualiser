"""
Centralized configuration settings for the Rule of Five Visualiser project.

This module contains all tunable constants used across the project, organized by
their intended use case. Each constant includes documentation explaining its purpose
and rationale.
"""

from collections.abc import Callable
from enum import Enum

import numpy as np

# =============================================================================
# Simulation Configuration
# =============================================================================

#: The number of data points in the population dataset.
#: 10,000 provides a large enough sample to ensure the true median is
#: statistically significant while remaining computationally efficient.
#: Used in: src/simulate.py::generate_population_data()
POPULATION_SIZE: int = 10_000

#: The number of samples drawn from the population in each simulation run.
#: A sample size of 5 is used because the Rule of Five states that the median
#: of a population will fall between the min and max of 5 randomly sampled
#: points with 93.75% confidence.
#: refer: https://pm.dartus.fr/posts/2024/rule-of-five/
#: Used in: src/simulate.py::simulate_rule_of_five()
SAMPLE_SIZE: int = 5


# =============================================================================
# Distribution Configuration
# =============================================================================


class DistributionType(Enum):
    """Supported distribution types for population data generation.

    These distributions are chosen to represent different types of real-world
    data patterns:
    - NORMAL: Symmetric data with no extreme outliers (e.g., heights, test scores)
    - UNIFORM: Data uniformly distributed across a range (e.g., waiting times)
    - EXPONENTIAL: Skewed data with a long tail (e.g., service times, lifetimes)

    Used in: src/simulate.py
    """

    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"


#: Distribution parameters for generating synthetic population data.
#: Each distribution is tuned to have a similar median (~50) for fair comparison.
#: - Normal: loc=50, scale=15 -> median ≈ 50
#: - Uniform: low=0, high=100 -> median = 50
#: - Exponential: scale=20 -> median ≈ 13.86 (natural median, not forced to 50)
#:
#: Used in: src/simulate.py::generate_population_data()
DISTRIBUTIONS: dict[DistributionType, Callable[[int], np.ndarray]] = {
    DistributionType.NORMAL: lambda size: np.random.normal(loc=50, scale=15, size=size),
    DistributionType.UNIFORM: lambda size: np.random.uniform(
        low=0, high=100, size=size
    ),
    DistributionType.EXPONENTIAL: lambda size: np.random.exponential(
        scale=20, size=size
    ),
}


# =============================================================================
# Visualisation Configuration
# =============================================================================

#: Matplotlib style to use for plots.
#: 'bmh' provides a clean, professional look with grid lines.
#: Used in: src/visualise.py
MATPLOTLIB_STYLE: str = "bmh"

#: DPI (dots per inch) for figure exports.
#: 150 provides a good balance between file size and quality for screen viewing.
#: Used in: src/visualise.py
FIGURE_DPI: int = 150

#: Default figure size (width, height) in inches.
#: 12x8 provides ample space for data visualization while fitting on standard screens.
#: Used in: src/visualise.py
DEFAULT_FIGURE_SIZE: tuple[float, float] = (12, 8)
