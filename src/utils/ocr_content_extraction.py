import re
from pathlib import Path

import cv2
import pandas as pd
import pytesseract
from PIL import Image

from configs.config import Config


def clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR text to avoid long unwanted shifts."""
    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple newlines with a single newline
    text = re.sub(r"\n\s*\n+", "\n", text)

    # Strip leading/trailing spaces on each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join([line for line in lines if line])  # remove empty lines

    return text.strip()


def extract_image_content(file_path: Path, img_name: str) -> tuple[str, list[pd.DataFrame]]:
    """Extract text and tables from an image using Tesseract OCR."""
    full_text = ""  # Extracted text
    tables_data: list[pd.DataFrame] = []  # Detected tables

    # Ensure output dirs exist
    Config.paths.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    Config.paths.TXT_FROM_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # === Extract raw text with Tesseract ===
    image = Image.open(file_path)
    raw_text = pytesseract.image_to_string(image, lang="eng")

    if raw_text:
        # ✅ Clean text to avoid long shifts
        text = clean_ocr_text(raw_text)

        # Save cleaned text to file using Path.open()
        text_path = Config.paths.TXT_FROM_IMAGE_DIR / f"txt_{img_name}.txt"
        with text_path.open("w", encoding="utf-8") as f:
            f.write(text)

        full_text = text

    # === Optional: detect tables ===
    img_cv = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if img_cv is not None:
        _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours, 1):
            x, y, w, h = cv2.boundingRect(cnt)
            roi = img_cv[y : y + h, x : x + w]

            # OCR on each detected region
            table_text = pytesseract.image_to_string(roi, lang="eng")
            table_text = clean_ocr_text(table_text)

            if table_text.strip():
                rows = [r.split() for r in table_text.split("\n") if r.strip()]
                table_df = pd.DataFrame(rows)  # avoid generic 'df'
                tables_data.append(table_df)

                # Save CSV
                csv_path = Config.paths.TABLE_DIR / f"{img_name}_table_{i}.csv"
                table_df.to_csv(csv_path, index=False)

    print(f"Extracted text and {len(tables_data)} tables from {file_path.name}")

    return full_text, tables_data
