import logging
from pathlib import Path

import streamlit as st

from configs.config import Config
from src.vectorizers.ocr_vectorizer import OCRVectorizerAgent

logging.getLogger("PIL").setLevel(logging.ERROR)


def download_and_process_images(image_file: object | None = None) -> None:
    """Save an uploaded image, extract text using OCR, and build a FAISS index.

    Args:
    ----
        image_file (streamlit.uploaded_file_manager.UploadedFile | None):
            Uploaded image file object from Streamlit.

    Returns:
    -------
        None

    """
    if not image_file:
        st.warning("⚠️ No image file provided.")
        return

    try:
        # Safe file name and paths
        image_name = Path(image_file.name).name[:70]
        img_path = Config.paths.IMAGE_DIR / image_name
        Config.paths.IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        # Save image using Path.open()
        with img_path.open("wb") as f:
            f.write(image_file.getbuffer())
        st.info(f"🖼️ Saved uploaded image as `{img_path.name}`")

        # OCR processing
        vectorizer = OCRVectorizerAgent()
        chunks = vectorizer.process_image(img_path)
        if not chunks:
            st.warning(f"⚠️ No text extracted from `{img_path.name}`")
            return
        st.success(f"✅ Extracted {len(chunks)} text chunks from `{img_path.name}`")

        # Build FAISS index for all images in directory
        vectorizer.process_all_images_in_directory(Config.paths.IMAGE_DIR)
        index_name = img_path.stem.replace(" ", "_").lower()
        index_path = Config.paths.INDEX_DIR / f"index_faiss_{index_name}"
        st.success(f"💾 Saved FAISS index for `{img_path.name}` to `{index_path}`")

    except Exception as exc:  # noqa: BLE001
        error_msg = f"Error processing `{image_file.name}`. Exception: {exc}"
        st.error(f"❌ {error_msg}")
