"""Test the visualise module."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.settings import DistributionType, SAMPLE_SIZE
from src.simulate import SimulationResults, simulate_rule_of_five
from src.visualise import plot_simulation


class TestPlotSimulation:
    """Test the plot_simulation function."""

    def test_plot_simulation_returns_figure(self):
        """Test that plot_simulation returns a matplotlib Figure."""
        np.random.seed(42)
        results: SimulationResults = simulate_rule_of_five(DistributionType.NORMAL, 10)
        population: np.ndarray = results.population_data_arr

        fig: plt.Figure = plot_simulation(population, results, 10)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_simulation_with_different_num_simulations(self):
        """Test plot_simulation with different simulation counts."""
        np.random.seed(42)

        for num_sims in [1, 10, 100]:
            results: SimulationResults = simulate_rule_of_five(
                DistributionType.NORMAL, num_sims
            )
            population: np.ndarray = results.population_data_arr

            fig: plt.Figure = plot_simulation(population, results, num_sims)

            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_plot_simulation_raises_on_invalid_num_simulations(self):
        """Test that plot_simulation raises ValueError for invalid num_simulations."""
        np.random.seed(42)
        results: SimulationResults = simulate_rule_of_five(DistributionType.NORMAL, 10)
        population: np.ndarray = results.population_data_arr

        with pytest.raises(ValueError, match="num_simulations must be greater than 0"):
            plot_simulation(population, results, 0)

    def test_plot_simulation_raises_on_negative_num_simulations(self):
        """Test that plot_simulation raises ValueError for negative num_simulations."""
        np.random.seed(42)
        results: SimulationResults = simulate_rule_of_five(DistributionType.NORMAL, 10)
        population: np.ndarray = results.population_data_arr

        with pytest.raises(ValueError, match="num_simulations must be greater than 0"):
            plot_simulation(population, results, -1)


class TestVisualiseConstants:
    """Test that visualise constants are properly defined."""

    def test_histogram_bins_is_positive(self):
        """Test that HISTOGRAM_BINS is a positive integer."""
        from src.settings import HISTOGRAM_BINS

        assert isinstance(HISTOGRAM_BINS, int)
        assert HISTOGRAM_BINS > 0

    def test_max_visualizations_is_positive(self):
        """Test that MAX_VISUALIZATIONS is a positive integer."""
        from src.settings import MAX_VISUALIZATIONS

        assert isinstance(MAX_VISUALIZATIONS, int)
        assert MAX_VISUALIZATIONS > 0

    def test_colors_are_valid_hex(self):
        """Test that color constants are valid hex strings."""
        from src.settings import (
            FAILURE_COLOR,
            MEDIAN_COLOR,
            POPULATION_COLOR,
            SUCCESS_COLOR,
        )

        hex_pattern = r"^#[0-9A-Fa-f]{6}$"
        for color in [POPULATION_COLOR, MEDIAN_COLOR, SUCCESS_COLOR, FAILURE_COLOR]:
            assert isinstance(color, str)
            assert len(color) == 7
            assert color.startswith("#")
