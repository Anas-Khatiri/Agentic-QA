import re
from pathlib import Path

import pandas as pd
import pdfplumber
from pdfplumber.pdfparser import PDFSyntaxError

from configs.config import Config


def extract_financial_announcements(text: str) -> list[str]:
    """Extract financial announcement dates from raw text using regex patterns."""
    results: list[str] = []

    # Search for patterns like "Agenda 2023 des annonces financières"
    for agenda_match in re.finditer(Config.announcement_patterns.AGENDA_PATTERN, text, re.IGNORECASE):
        year = agenda_match.group(1)
        agenda_start = agenda_match.end()

        # Extract 300 characters after the "Agenda" line
        date_block = text[agenda_start : agenda_start + 300]

        # Extract French-style date patterns (e.g., "25 janvier")
        date_matches = re.findall(Config.announcement_patterns.FRENCH_DATE_PATTERN, date_block, re.IGNORECASE)

        for day, month in date_matches:
            jour = day.zfill(2)
            mois = month.lower()
            mois_num = Config.announcement_patterns.FRENCH_MONTHS.get(mois)
            if mois_num:
                results.append(f"{year}-{mois_num}-{jour}")

    return results


def extract_dates_from_txt_dir(directory: Path, source_label: str) -> list[dict[str, str]]:
    """Extract dates from all .txt files in a directory."""
    all_dates: list[dict[str, str]] = []

    for file_path in directory.glob("*.txt"):
        print(file_path.name)
        text = file_path.read_text(encoding="utf-8")
        dates = extract_financial_announcements(text)
        for d in dates:
            all_dates.append({"date": d, "source": source_label})

    return all_dates


def extract_dates_from_pdf_dir(directory: Path) -> list[dict[str, str]]:
    """Extract dates from all .pdf files in a directory using pdfplumber."""
    all_dates: list[dict[str, str]] = []

    for file_path in directory.glob("*.pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except (PDFSyntaxError, OSError) as e:
            print(f"Failed to read PDF {file_path.name}: {e}")
            continue

        dates = extract_financial_announcements(text)
        for d in dates:
            all_dates.append({"date": d, "source": "pdf"})

    return all_dates


def merge_and_save_all_dates(pdf_dir: Path) -> pd.DataFrame:
    """Extract, merge, clean, and save financial announcement dates from HTML and PDF content."""
    all_dates: list[dict[str, str]] = []
    # Extract from PDFs and label source
    all_dates += extract_dates_from_pdf_dir(pdf_dir)

    if not all_dates:
        print("❌ No dates were extracted.")
        return pd.DataFrame()

    merged_dates_df = pd.DataFrame(all_dates).drop_duplicates()
    merged_dates_df = merged_dates_df.dropna(subset=["date"]).sort_values(by="date")

    # Save results to CSV
    output_csv = Config.paths.FINANCIAL_DIR / Config.paths.ANNOUNCEMENT_DATE_FILE_NAME
    merged_dates_df.to_csv(output_csv, index=False)

    print(f"{len(merged_dates_df)} dates saved to: {output_csv}")
    return merged_dates_df
