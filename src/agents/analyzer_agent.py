from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from configs.config import Config
from src.utils.announcement_dates_extraction import merge_and_save_all_dates
from src.utils.vehicles_sold_per_year_extraction import save_vehicles_data_to_csv


class AnalyzerAgent:
    """Analyze the correlation between Renault's vehicle sales and stock price on result days.

    Provides methods to load or generate required CSV data and plot the correlation
    either to disk or within a Streamlit app.
    """

    START_YEAR: int = 2020

    def __init__(self) -> None:
        """Initialize the AnalyzerAgent with output image path."""
        self.output_img: Path = Config.paths.GRAPH_DIR / "correlation_sales_vs_stock.png"

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load or generate the required CSVs for analysis.

        Returns
        -------
            Tuple[pd.DataFrame, pd.DataFrame]: (dates_df, sales_df)

        """
        dates_csv = Config.paths.FINANCIAL_DIR / Config.paths.ANNOUNCEMENT_DATE_FILE_NAME
        dates_df = pd.read_csv(dates_csv) if dates_csv.exists() else merge_and_save_all_dates(pdf_dir=Config.paths.PDF_DIR)

        sales_csv = Config.paths.FINANCIAL_DIR / Config.paths.VEHICLE_SOLD_FILE_NAME
        sales_df = pd.read_csv(sales_csv) if sales_csv.exists() else save_vehicles_data_to_csv()

        if dates_df.empty or sales_df.empty:
            print("No extracted data to plot.")

        return dates_df, sales_df

    def plot(self, streamlit: bool | None = False) -> None:
        """Visualize the correlation between Renault's stock price and vehicle sales."""
        # === Load data ===
        dates_df, sales_df = self.load_data()
        if dates_df.empty or sales_df.empty:
            return

        # Filter to years >= START_YEAR
        sales_df = sales_df[sales_df["year"] >= self.START_YEAR]
        dates_df = dates_df[dates_df["date"] >= f"{self.START_YEAR}-01-01"]

        # Convert date and extract year
        dates_df["date"] = pd.to_datetime(dates_df["date"])
        dates_df["year"] = dates_df["date"].dt.year

        renault_stock = yf.download("RNO.PA", start=f"{self.START_YEAR}-01-01")
        renault_stock.index = pd.to_datetime(renault_stock.index)

        # === Average stock price per year on result dates ===
        stock_by_year = [
            {
                "year": year,
                "avg_stock_price": sum(renault_stock.loc[date]["Close"] for date in group["date"] if date in renault_stock.index)
                / len([date for date in group["date"] if date in renault_stock.index]),
            }
            for year, group in dates_df.groupby("year")
            if any(date in renault_stock.index for date in group["date"])
        ]

        stock_df = pd.DataFrame(stock_by_year)
        merged_df = sales_df.merge(stock_df, on="year", how="inner")
        if merged_df.empty:
            print("No overlapping data to compute correlation.")
            return

        # === Compute correlation ===
        correlation = merged_df["vehicles_sold"].corr(merged_df["avg_stock_price"])
        print(f"Correlation between vehicle sales and Renault stock price on result days: {correlation:.3f}")

        # === Plotting ===
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(merged_df["vehicles_sold"], merged_df["avg_stock_price"], color="blue")

        for _, row in merged_df.iterrows():
            ax.text(row["vehicles_sold"], row["avg_stock_price"], str(row["year"]), fontsize=9)

        ax.set_title("Correlation: Vehicle Sales vs Renault Stock Price on Result Days (2020+)")
        ax.set_xlabel("Vehicles Sold")
        ax.set_ylabel("Average Stock Price (on result days)")
        ax.grid(True)
        fig.tight_layout()

        if streamlit:
            import streamlit as st

            st.pyplot(fig)
        else:
            plt.savefig(self.output_img)
            plt.close()
            print(f"Graph saved to: {self.output_img}")
