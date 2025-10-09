import logging
from pathlib import Path

import streamlit as st

from configs.config import Config
from src.vectorizers.pdf_vectorizer import PDFVectorizerAgent

logging.getLogger("pdfminer").setLevel(logging.ERROR)


def download_and_process_pdfs(pdf_file: object | None = None) -> None:
    """Save an uploaded PDF, extract text, and build a FAISS index.

    Args:
    ----
        pdf_file (streamlit.uploaded_file_manager.UploadedFile | None):
            Uploaded PDF file object from Streamlit.

    Returns:
    -------
        None

    """
    if not pdf_file:
        st.warning("⚠️ No PDF file provided.")
        return

    vectorizer = PDFVectorizerAgent()

    try:
        # Safe file name and path
        pdf_name = Path(pdf_file.name).name[:70]
        pdf_path = Config.paths.PDF_DIR / pdf_name
        Config.paths.PDF_DIR.mkdir(parents=True, exist_ok=True)

        # Save PDF using Path.open()
        with pdf_path.open("wb") as f:
            f.write(pdf_file.getbuffer())
        st.info(f"📄 Saved uploaded PDF as `{pdf_path.name}`")

        # Extract text chunks
        chunks = vectorizer.process_pdf(pdf_path)
        if not chunks:
            st.warning(f"⚠️ No text extracted from `{pdf_path.name}`")
            return
        st.success(f"✅ Extracted {len(chunks)} text chunks from `{pdf_path.name}`")

        # Build FAISS index for all PDFs
        vectorizer.process_all_pdfs_in_directory(Config.paths.PDF_DIR)
        index_name = pdf_path.stem.replace(" ", "_").lower()
        index_path = Config.paths.INDEX_DIR / f"index_faiss_{index_name}"
        st.success(f"💾 Saved FAISS index for `{pdf_path.name}` to `{index_path}`")

    except Exception as exc:  # noqa: BLE001
        error_msg = f"Error processing `{pdf_file.name}`. Exception: {exc}"
        st.error(f"❌ {error_msg}")
