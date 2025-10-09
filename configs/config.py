"""Agentic AI System configuration."""

from pathlib import Path
from typing import ClassVar

import torch


class PathConfig:
    """File and directory path configuration for input/output data.

    Automatically creates necessary directories if they don't exist.
    """

    # Base project directory
    BASE_DIR: ClassVar[Path] = Path(__file__).parent.parent.resolve()

    # Primary data directory
    DATA_DIR: ClassVar[Path] = BASE_DIR / "data"

    # Primary LLM models directory
    MODEL_DIR: ClassVar[Path] = Path("/models")

    # Subdirectories for different data sources and outputs
    PDF_DIR: ClassVar[Path] = DATA_DIR / "pdfs"
    IMAGE_DIR: ClassVar[Path] = DATA_DIR / "images"
    TXT_FROM_IMAGE_DIR: ClassVar[Path] = DATA_DIR / "text_from_image"
    INDEX_DIR: ClassVar[Path] = DATA_DIR / "vector_indices"
    TABLE_DIR: ClassVar[Path] = DATA_DIR / "tables"
    YOUTUBE_DIR: ClassVar[Path] = DATA_DIR / "youtube"
    FINANCIAL_DIR: ClassVar[Path] = DATA_DIR / "financial_results"
    GRAPH_DIR: ClassVar[Path] = DATA_DIR / "graphs"

    # Filenames for static CSV resources
    ANNOUNCEMENT_DATE_FILE_NAME: ClassVar[str] = "announcement_result_dates.csv"
    VEHICLE_SOLD_FILE_NAME: ClassVar[str] = "vehicles_sold_per_year.csv"

    # Ensure directories exist on first import
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True, parents=True)
    IMAGE_DIR.mkdir(exist_ok=True, parents=True)
    TXT_FROM_IMAGE_DIR.mkdir(exist_ok=True, parents=True)
    TABLE_DIR.mkdir(exist_ok=True, parents=True)
    YOUTUBE_DIR.mkdir(exist_ok=True, parents=True)
    FINANCIAL_DIR.mkdir(exist_ok=True, parents=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)


class ModelConfig:
    """Configuration for embedding models and language models (LLMs)."""

    device: ClassVar[str] = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_MODEL_NAME: ClassVar[str] = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_KWARGS: ClassVar[dict[str, str]] = {"device": device}

    LLM_NAME: ClassVar[str] = "HuggingFaceTB/SmolLM3-3B"
    LLM_MAX_TOKENS: ClassVar[int] = 500
    LLM_TEMPERATURE: ClassVar[float] = 0.6
    LLM_SAMPLE: ClassVar[bool] = False


class ProcessingConfig:
    """Chunking and retrieval settings used during document processing."""

    CHUNK_SIZE: ClassVar[int] = 1000
    CHUNK_OVERLAP: ClassVar[int] = 100
    RETRIEVER_K: ClassVar[int] = 5
    SIMILARITY_THRESHOLD: ClassVar[float] = 0.3


class AnnouncementPatternConfig:
    """Regex patterns and logic to extract financial announcement data.

    Used to identify French-style dates and agenda patterns in financial documents.
    """

    AGENDA_PATTERN: ClassVar[str] = r"Agenda\s+(\d{4})\s+des annonces financières"
    FRENCH_DATE_PATTERN: ClassVar[str] = r"\b(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|" r"juillet|août|septembre|octobre|novembre|décembre)\b"

    FRENCH_MONTHS: ClassVar[dict[str, str]] = {
        "janvier": "01",
        "février": "02",
        "mars": "03",
        "avril": "04",
        "mai": "05",
        "juin": "06",
        "juillet": "07",
        "août": "08",
        "septembre": "09",
        "octobre": "10",
        "novembre": "11",
        "décembre": "12",
    }


class Config:
    """Master access point to all configuration sections used across the app."""

    paths: ClassVar[type[PathConfig]] = PathConfig
    models: ClassVar[type[ModelConfig]] = ModelConfig
    processing: ClassVar[type[ProcessingConfig]] = ProcessingConfig
    announcement_patterns: ClassVar[type[AnnouncementPatternConfig]] = AnnouncementPatternConfig
