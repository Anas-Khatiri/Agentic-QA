from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from configs.config import Config
from src.utils.ocr_content_extraction import extract_image_content
from src.utils.text_splitter import split_text_into_chunks


class OCRVectorizerAgent:
    """Agent that processes image files by extracting text via OCR, splitting it into chunks, and indexing each image into FAISS for semantic search."""

    def __init__(self) -> None:
        """Initialize the OCRVectorizerAgent with embeddings and optional index."""
        self.embedding = HuggingFaceEmbeddings(
            model_name=Config.models.EMBEDDING_MODEL_NAME,
            model_kwargs=Config.models.EMBEDDING_MODEL_KWARGS,
        )

        # Path to a general FAISS index (optional, not used per image)
        self.index_path = str(Config.paths.INDEX_DIR / "faiss_index")
        self.vectorstore: FAISS | None = self.load_index()

    def load_index(self) -> FAISS | None:
        """Load an existing FAISS index from disk if it exists."""
        file_index = Path(self.index_path)
        if file_index.exists():
            print(f"Loading existing FAISS index from {self.index_path}")
            return FAISS.load_local(self.index_path, self.embedding)
        return None

    def process_image(self, img_path: Path) -> list[Document]:
        """Process a single image: extract text via OCR and split it into chunks."""
        print(f"Processing Image: {img_path.name}")
        text, _ = extract_image_content(img_path, img_path.name)
        return split_text_into_chunks(text)

    def process_all_images_in_directory(self, directory: Path, save: bool = True) -> None:
        """Process all image files in a directory, indexing each file separately. Optionally saves each FAISS index to disk."""
        for img_file in directory.glob("*.*"):
            if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tiff"]:
                continue

            try:
                # Normalize and define the name for this FAISS index
                index_name = img_file.stem.replace(" ", "_").lower()
                index_path = Config.paths.INDEX_DIR / f"index_faiss_{index_name}"

                # Skip if index already exists
                if index_path.exists():
                    print(f"Index already exists for {img_file.name}. Skipping.")
                    continue

                # Extract content and convert to chunks
                chunks = self.process_image(img_file)
                if not chunks:
                    print(f"No content extracted from {img_file.name}. Skipping.")
                    continue

                docs_with_metadata = [Document(page_content=chunk.page_content, metadata={"source": img_file.name}) for chunk in chunks]

                # Create FAISS index from chunks
                vectorstore = FAISS.from_documents(docs_with_metadata, self.embedding)

                # Save the index locally
                if save:
                    vectorstore.save_local(index_path)
                    print(f"Saved FAISS index to {index_path}")

            except Exception as exc:
                print(f"Error processing {img_file.name}: {exc}")
