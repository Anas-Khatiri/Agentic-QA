from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from configs.config import Config
from src.utils.text_splitter import split_text_into_chunks


class YBVectorizerAgent:
    """Agent for processing YouTube transcripts. Chunks transcript text and builds a FAISS index per transcript file."""

    def __init__(self) -> None:
        """Initialize the YouTube transcript vectorizer with embedding model."""
        self.embedding = HuggingFaceEmbeddings(
            model_name=Config.models.EMBEDDING_MODEL_NAME,
            model_kwargs=Config.models.EMBEDDING_MODEL_KWARGS,
        )

    def process_yb_video(self, yb_text_file: str) -> list[Document]:
        """Read and chunk a single YouTube transcript file."""
        print(f"Processing YouTube transcript: {yb_text_file}")

        # Use Path.open() instead of open() for Ruff PTH123
        path = Path(yb_text_file)
        with path.open(encoding="utf-8") as f:
            yb_text = f.read()

        return split_text_into_chunks(yb_text)

    def process_all_txt_in_directory(self, directory: Path, save: bool = True) -> None:
        """Process all `.txt` transcript files in the directory. Each transcript is indexed separately with FAISS."""
        for yb_txt_file in directory.glob("*.txt"):
            try:
                # Normalize and define the name for this FAISS index
                index_name = yb_txt_file.stem.replace(" ", "_").lower()
                index_path = Config.paths.INDEX_DIR / f"index_faiss_{index_name}"

                if index_path.exists():
                    print(f"Index already exists for {yb_txt_file.name}. Skipping.")
                    continue

                # Extract content and convert it into chunks
                chunks = self.process_yb_video(str(yb_txt_file))
                if not chunks:
                    print(f"No content extracted from {yb_txt_file.name}. Skipping.")
                    continue

                # Convert chunks to Documents with metadata
                docs_with_metadata = [Document(page_content=chunk.page_content, metadata={"source": yb_txt_file.name}) for chunk in chunks]

                # Create and save FAISS index
                vectorstore = FAISS.from_documents(docs_with_metadata, self.embedding)

                # Save the FAISS index locally if requested
                if save:
                    vectorstore.save_local(index_path)
                    print(f"Saved FAISS index to {index_path}")

            except Exception as exc:  # noqa: BLE001
                print(f"Error processing {yb_txt_file.name}: {exc}")
