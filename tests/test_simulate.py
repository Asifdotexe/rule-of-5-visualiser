"""Test the simulate module."""

import numpy as np
import pytest

from src.settings import SAMPLE_SIZE, DistributionType
from src.simulate import (
    SimulationResults,
    generate_population_data,
    simulate_rule_of_five,
)
from tests.conftest import NUM_SIMULATIONS, POPULATION_SIZES, TEST_SEED


class TestGeneratePopulationData:
    """Test the generate_population_data function."""

    @pytest.mark.parametrize("distribution", list(DistributionType))
    @pytest.mark.parametrize("size", POPULATION_SIZES)
    def test_generate_population_data_size(self, distribution, size):
        """Test that generated population data has the correct shape."""
        result = generate_population_data(distribution, size)
        assert result.shape == (size,)

    @pytest.mark.parametrize("distribution", list(DistributionType))
    def test_generate_population_data_validity(self, distribution):
        """Test that generated population data contains no NaN or Inf values."""
        result = generate_population_data(distribution)
        assert not np.any(np.isnan(result)), "Result contains NaN values"
        assert not np.any(np.isinf(result)), "Result contains Inf values"


class TestSimulateOutputTypes:
    """Test the output types of simulate_rule_of_five function."""

    @pytest.mark.parametrize("distribution", list(DistributionType))
    @pytest.mark.parametrize("num_sims", NUM_SIMULATIONS)
    def test_simulate_returns_simulation_results(self, distribution, num_sims):
        """Test that simulate returns a SimulationResults instance."""
        result = simulate_rule_of_five(distribution, num_sims)
        assert isinstance(result, SimulationResults)

    @pytest.mark.parametrize("distribution", list(DistributionType))
    @pytest.mark.parametrize("num_sims", NUM_SIMULATIONS)
    def test_simulate_dtypes(self, distribution, num_sims):
        """Test that simulate returns correct numpy dtypes."""
        result = simulate_rule_of_five(distribution, num_sims)
        assert result.population_median.dtype == np.float64
        assert result.num_of_simulations_within_range.dtype == np.int64
        assert result.rule_of_five_success_rate.dtype == np.float64


class TestSimulateShapes:
    """Test the output shapes of simulate_rule_of_five function."""

    @pytest.mark.parametrize("distribution", list(DistributionType))
    @pytest.mark.parametrize("num_sims", NUM_SIMULATIONS)
    def test_simulate_shapes(self, distribution, num_sims):
        """Test that simulate returns arrays with correct shapes."""
        result = simulate_rule_of_five(distribution, num_sims)
        assert result.sample_data_arr.shape == (num_sims, SAMPLE_SIZE)
        assert result.sample_mins.shape == (num_sims,)
        assert result.sample_maxs.shape == (num_sims,)
        assert result.is_median_within_range_arr.shape == (num_sims,)

    @pytest.mark.parametrize("distribution", list(DistributionType))
    @pytest.mark.parametrize("num_sims", NUM_SIMULATIONS)
    def test_simulate_boolean_array_dtype(self, distribution, num_sims):
        """Test that is_median_within_range_arr is a boolean array."""
        result = simulate_rule_of_five(distribution, num_sims)
        assert result.is_median_within_range_arr.dtype == np.bool_


class TestSimulateInvariants:
    """Test invariants that should always hold for simulate output."""

    @pytest.mark.parametrize("distribution", list(DistributionType))
    @pytest.mark.parametrize("num_sims", NUM_SIMULATIONS)
    def test_within_range_count_bounds(self, distribution, num_sims):
        """Test that within range count is between 0 and num_sims."""
        result = simulate_rule_of_five(distribution, num_sims)
        assert 0 <= result.num_of_simulations_within_range <= num_sims

    @pytest.mark.parametrize("distribution", list(DistributionType))
    @pytest.mark.parametrize("num_sims", NUM_SIMULATIONS)
    def test_success_rate_bounds(self, distribution, num_sims):
        """Test that success rate is between 0 and 100."""
        result = simulate_rule_of_five(distribution, num_sims)
        assert 0 <= result.rule_of_five_success_rate <= 100

    @pytest.mark.parametrize("distribution", list(DistributionType))
    @pytest.mark.parametrize("num_sims", NUM_SIMULATIONS)
    def test_within_range_count_matches_boolean_sum(self, distribution, num_sims):
        """Test that count matches sum of boolean array."""
        result = simulate_rule_of_five(distribution, num_sims)
        expected_count = np.sum(result.is_median_within_range_arr)
        assert result.num_of_simulations_within_range == expected_count


class TestSimulateExpectedValues:
    """Test expected value calculations with fixed seed."""

    @pytest.mark.parametrize("distribution", list(DistributionType))
    def test_simulate_expected_values_100_sims(self, distribution):
        """Test that count and rate calculations are consistent."""
        result = simulate_rule_of_five(distribution, 100)
        expected_count = np.sum(result.is_median_within_range_arr)
        expected_rate = (expected_count / 100) * 100

        assert result.num_of_simulations_within_range == expected_count
        assert result.rule_of_five_success_rate == expected_rate

    @pytest.mark.parametrize("distribution", list(DistributionType))
    def test_simulate_median_within_sample_range(self, distribution):
        """Test that median is within sample range when flagged."""
        result = simulate_rule_of_five(distribution, 100)

        for i in range(100):
            sample_min = result.sample_mins[i]
            sample_max = result.sample_maxs[i]
            median = result.population_median

            if result.is_median_within_range_arr[i]:
                assert sample_min <= median <= sample_max


class TestSimulateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_simulate_single_simulation(self):
        """Test simulate with exactly one simulation."""
        result = simulate_rule_of_five(DistributionType.NORMAL, 1)
        assert result.sample_data_arr.shape == (1, SAMPLE_SIZE)
        assert result.sample_mins.shape == (1,)
        assert result.sample_maxs.shape == (1,)
        assert result.is_median_within_range_arr.shape == (1,)

    def test_simulate_within_range_count_single_simulation(self):
        """Test that single simulation count is either 0 or 1."""
        result = simulate_rule_of_five(DistributionType.NORMAL, 1)
        assert result.num_of_simulations_within_range in [0, 1]


class TestConstants:
    """Test that configuration constants are properly defined."""

    def test_population_sizes_not_empty(self):
        """Test that POPULATION_SIZES is not empty."""
        assert len(POPULATION_SIZES) > 0

    def test_num_simulations_not_empty(self):
        """Test that NUM_SIMULATIONS is not empty."""
        assert len(NUM_SIMULATIONS) > 0

    def test_test_seed_is_int(self):
        """Test that TEST_SEED is an integer."""
        assert isinstance(TEST_SEED, int)
