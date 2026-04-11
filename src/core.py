"""
Place to house all the backend functions
"""

import numpy as np


def generate_population_data(distribution_type: str, size: int) -> np.ndarray:
    if distribution_type == "normal":
        return np.random.normal(loc=50, scale=15, size=size)
    elif distribution_type == "uniform":
        return np.random.uniform(low=0, high=100, size=size)
    elif distribution_type == "exponential":
        return np.random.exponential(scale=20, size=size)
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
