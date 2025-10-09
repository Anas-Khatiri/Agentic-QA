from pathlib import Path

import pandas as pd

from configs.config import Config


def save_vehicles_data_to_csv() -> pd.DataFrame:
    """Create and save a CSV file containing the number of Renault vehicles sold. Covers years 2020 to 2024. Returns the corresponding DataFrame."""
    # === Manually defined data ===
    # List of years
    years = ["2020", "2021", "2022", "2023", "2024"]
    # Vehicles sold per year
    vehicles_sold_counts = [
        2951971,
        2696401,
        2051174,
        2235000,
        2264815,
    ]

    # === Create a DataFrame from the data ===
    sales_df = pd.DataFrame({"year": years, "vehicles_sold": vehicles_sold_counts})

    # === Define output path and save as CSV ===
    output_path: Path = Config.paths.FINANCIAL_DIR / Config.paths.VEHICLE_SOLD_FILE_NAME
    sales_df.to_csv(output_path, index=False)  # Save without row index

    # === Confirmation message ===
    print(f"Data saved to {output_path}")

    # === Return the created DataFrame for further use ===
    return sales_df
