from pathlib import Path

import pandas as pd
import pdfplumber

from configs.config import Config


def extract_pdf_content(file_path: Path, pdf_name: str) -> tuple[str, list[pd.DataFrame]]:
    """Extract textual and tabular content from a PDF file."""
    full_text = ""  # Accumulate all page texts
    tables_data: list[pd.DataFrame] = []  # Store extracted tables

    # Open the PDF file
    with pdfplumber.open(file_path) as pdf:
        # Loop through each page of the PDF
        for i, page in enumerate(pdf.pages, 1):
            # === Extract and append text ===
            text = page.extract_text()
            if text:
                full_text += text + "\n"

            # === Extract tables if any ===
            table = page.extract_table()
            if table:
                table_df = pd.DataFrame(table[1:], columns=table[0] if len(table) > 1 else None)
                tables_data.append(table_df)

                # Save CSV using Path.open() if needed
                csv_path = Config.paths.TABLE_DIR / f"{pdf_name}_table_page_{i}.csv"
                table_df.to_csv(csv_path, index=False)

    # Print summary of the operation
    print(f"Extracted text and {len(tables_data)} tables from {file_path.name}")

    # Return the text and list of tables
    return full_text, tables_data
