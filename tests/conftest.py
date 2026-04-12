"""Test configuration and fixtures."""

import numpy as np
import pytest

TEST_SEED: int = 42

POPULATION_SIZES: tuple[int, ...] = (100, 1000, 10_000)

NUM_SIMULATIONS: tuple[int, ...] = (1, 10, 1000)


@pytest.fixture(autouse=True)
def seeded_rng() -> None:
    """Set random seed before each test for deterministic results."""
    np.random.seed(TEST_SEED)
