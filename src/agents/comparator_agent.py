from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from configs.config import Config
from src.utils.announcement_dates_extraction import merge_and_save_all_dates


class ComparatorAgent:
    """Compare Renault's stock price with the CAC40 index on earnings announcement dates.

    Provides methods to load announcement dates and plot Renault vs CAC40 stock performance.
    """

    def __init__(self) -> None:
        """Initialize the ComparatorAgent with the output image path."""
        self.output_img: Path = Config.paths.GRAPH_DIR / "renault_stock_vs_cac40.png"

    def load_data(self) -> pd.DataFrame:
        """Load or generate Renault's financial announcement dates."""
        output_csv = Config.paths.FINANCIAL_DIR / Config.paths.ANNOUNCEMENT_DATE_FILE_NAME

        if output_csv.exists():
            # Use cached announcement dates if available
            announcement_dates_df = pd.read_csv(output_csv)
            return announcement_dates_df

        # Generate announcement dates from PDFs and HTML if not already saved
        announcement_dates_df = merge_and_save_all_dates(
            pdf_dir=Config.paths.PDF_DIR,
        )

        if announcement_dates_df.empty:
            print("No extracted data to plot.")

        return announcement_dates_df

    def plot(self, streamlit: bool | None = False) -> None:
        """Generate a plot comparing Renault's stock price with the CAC40 index."""
        # === Load and filter relevant announcement dates ===
        announcement_dates_df = self.load_data()
        if announcement_dates_df.empty:
            return

        # Filter only records from 2020 onwards
        announcement_dates_df = announcement_dates_df[announcement_dates_df["date"] >= "2020-01-01"]

        # === Download stock and index data ===
        renault_stock = yf.download("RNO.PA", start="2020-01-01")
        cac40_index = yf.download("^FCHI", start="2020-01-01")

        renault_stock.index = pd.to_datetime(renault_stock.index)
        cac40_index.index = pd.to_datetime(cac40_index.index)

        # === Collect stock prices on announcement dates ===
        price_records = [
            {
                "date": pd.to_datetime(row["date"]),
                "renault": renault_stock.loc[row["date"]]["Close"],
                "cac40": cac40_index.loc[row["date"]]["Close"],
            }
            for _, row in announcement_dates_df.iterrows()
            if pd.to_datetime(row["date"]) in renault_stock.index and pd.to_datetime(row["date"]) in cac40_index.index
        ]

        prices_df = pd.DataFrame(price_records).sort_values("date")
        if prices_df.empty:
            print("No overlapping financial data found.")
            return

        # === Plotting ===
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(prices_df["date"], prices_df["renault"], label="Renault Stock", marker="o")
        ax.plot(prices_df["date"], prices_df["cac40"], label="CAC40 Index", marker="s")

        ax.set_title("Renault Stock vs CAC40 on Earnings Dates (2020+)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if streamlit:
            import streamlit as st

            st.pyplot(fig)
        else:
            plt.savefig(self.output_img)
            plt.close()
            print(f"Stock comparison graph saved to: {self.output_img}")
