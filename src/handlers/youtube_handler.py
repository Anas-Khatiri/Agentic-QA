import logging

import streamlit as st

from configs.config import Config
from src.utils.youtube_content_extraction import extract_youtube_id, transcribe_youtube_audio
from src.vectorizers.youtube_vectorizer import YBVectorizerAgent

# Suppress noisy logs
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def download_and_process_youtube_contents(yt_url: str) -> None:
    """Download a YouTube video transcript, save it, and build a FAISS index.

    Args:
    ----
        yt_url (str): The URL of the YouTube video to process.

    Returns:
    -------
        None

    """
    if not yt_url.strip():
        st.warning("⚠️ No YouTube URL was provided.")
        return

    # Extract the YouTube video ID from its URL
    youtube_id = extract_youtube_id(url=yt_url)
    yb_txt_name = str(youtube_id)
    yb_txt_path = Config.paths.YOUTUBE_DIR / f"yb_{yb_txt_name}.txt"
    Config.paths.YOUTUBE_DIR.mkdir(parents=True, exist_ok=True)

    if yb_txt_path.exists():
        st.info(f"Transcript already exists: {yb_txt_path.name}. Skipping.")
    else:
        # Transcribe only if not already available
        transcribe_youtube_audio(youtube_url=yt_url, video_name=yb_txt_name)
        st.success(f"✅ Transcript downloaded: yb_{yb_txt_name}.txt")

    # Vectorize and index transcript
    vectorizer = YBVectorizerAgent()
    vectorizer.process_all_txt_in_directory(Config.paths.YOUTUBE_DIR)
    st.success("YouTube content processed and indexed!")
