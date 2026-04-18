"""Main entry point for the rule-of-5-visualiser application."""

import gradio as gr
import matplotlib.figure as mfigure
import pandas as pd

from src.settings import DistributionType
from src.simulate import (
    SimulationResults,
    simulate_rule_of_five,
)
from src.visualise import plot_simulation


def run_simulations(
    distribution_type: str,
    num_simulations: int,
    historical_datapoints_df: pd.DataFrame,
) -> tuple[mfigure.Figure, pd.DataFrame, str]:
    """Orchestrates the simulation and visualization of the rule-of-5 distribution.

    :param distribution_type: The type of distribution to simulate (as string).
    :param num_simulations: The number of simulations to run.
    :param historical_datapoints_df: A dataframe of historical datapoints to include in the plot.
    :return: A tuple of the figure, updated history dataframe, and simulation stats text.
    :raises ValueError: If num_simulations is less than 1.
    """
    if num_simulations < 1:
        raise ValueError("num_simulations must be greater than 0")

    distribution_type_enum: DistributionType = DistributionType(distribution_type)

    simulation_results: SimulationResults = simulate_rule_of_five(
        distribution_type_enum, num_simulations
    )
    fig: mfigure.Figure = plot_simulation(
        simulation_results.population_data_arr, simulation_results, num_simulations
    )

    new_datapoints_df: pd.DataFrame = pd.DataFrame(
        [
            {
                "n_simulations": num_simulations,
                "n_success": simulation_results.num_of_simulations_within_range,
                "accuracy": simulation_results.rule_of_five_success_rate,
                "distribution_type": distribution_type_enum.value,
            }
        ]
    )

    updated_history: pd.DataFrame = pd.concat(
        [new_datapoints_df, historical_datapoints_df], ignore_index=True
    )

    simulation_stats_text: str = (
        f"### Simulation Stats\n"
        f"**Accuracy Rate:** {simulation_results.rule_of_five_success_rate:.2f}%  \n"
        f"**Hits/Trials:** {simulation_results.num_of_simulations_within_range} / {num_simulations}"
    )

    return fig, updated_history, simulation_stats_text


DISTRIBUTION_CHOICES: list[str] = [dist.value for dist in DistributionType]


with gr.Blocks(
    theme=gr.Theme.from_hub("miittnnss/green"), title="Rule of 5 Visualiser"
) as app:
    gr.Markdown("# Rule of 5 Visualiser")
    gr.Markdown(
        "A random sample of 5 has a 93.75% chance of containing the population median."
    )

    historical_state = gr.State(
        pd.DataFrame(
            columns=["n_simulations", "n_success", "accuracy", "distribution_type"]
        )
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                distribution_selection = gr.Dropdown(
                    DISTRIBUTION_CHOICES,
                    value=DistributionType.NORMAL.value,
                    label="Distribution",
                )
                with gr.Row():
                    b1 = gr.Button("1x", variant="primary")
                    b10 = gr.Button("10x", variant="primary")
                with gr.Row():
                    b100 = gr.Button("100x", variant="primary")
                    b1000 = gr.Button("1000x", variant="primary")
                stats_box = gr.Markdown("Ready to simulate...")

            log_table = gr.DataFrame(label="Simulation Log", interactive=False)

        with gr.Column(scale=2):
            gradio_plot = gr.Plot()

        for btn, value in zip([b1, b10, b100, b1000], [1, 10, 100, 1000]):
            btn.click(
                fn=run_simulations,
                inputs=[distribution_selection, gr.State(value), historical_state],
                outputs=[gradio_plot, log_table, stats_box],
            ).then(fn=lambda x: x, inputs=[log_table], outputs=[historical_state])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
