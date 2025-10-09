from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from configs.config import Config
from src.utils.vehicles_sold_per_year_extraction import save_vehicles_data_to_csv


class VehicleSalesVisualizerAgent:
    """Visualize vehicle sales data per year (2020+)."""

    def __init__(self) -> None:
        """Initialize the visualizer with the output chart path."""
        self.output_img: Path = Config.paths.GRAPH_DIR / "vehicles_sold_per_year.png"

    def load_data(self) -> pd.DataFrame:
        """Extract and load vehicle sales data.

        Returns
        -------
            pd.DataFrame: DataFrame containing 'year' and 'vehicles_sold' columns.

        """
        sales_df = save_vehicles_data_to_csv()

        if sales_df.empty:
            print("No vehicle sales data to plot.")

        return sales_df

    def plot(self, streamlit: bool | None = False) -> None:
        """Create and display or save a bar chart of vehicles sold per year.

        Args:
        ----
            streamlit (bool, optional): If True, render inside Streamlit.
                If False, save to file.

        """
        # === Load the vehicle sales data ===
        sales_df = self.load_data()
        if sales_df.empty:
            return  # Exit if no data available

        # === Create the bar chart ===
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            sales_df["year"],  # x-axis: years
            sales_df["vehicles_sold"],  # y-axis: number of vehicles sold
            color="steelblue",
            edgecolor="black",
        )

        # === Annotate each bar with its value ===
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # === Final chart formatting ===
        ax.set_title("Vehicles Sold Per Year (2020+)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Vehicles Sold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()

        # === Save or render the plot ===
        if streamlit:
            import streamlit as st

            st.pyplot(fig)
        else:
            plt.savefig(self.output_img)
            plt.close()
            print(f"Vehicle sales chart saved to: {self.output_img}")
