from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from configs.config import Config
from src.utils.pdf_content_extraction import extract_pdf_content
from src.utils.text_splitter import split_text_into_chunks


class PDFVectorizerAgent:
    """Agent that processes PDF documents by extracting content, splitting it into chunks, and indexing it into FAISS for semantic search. Each chunk stores metadata with the source PDF filename."""

    def __init__(self) -> None:
        """Initialize the PDFVectorizerAgent with embeddings and optional index."""
        self.embedding = HuggingFaceEmbeddings(
            model_name=Config.models.EMBEDDING_MODEL_NAME,
            model_kwargs=Config.models.EMBEDDING_MODEL_KWARGS,
        )

        # Path to the base FAISS index
        self.index_path = str(Config.paths.INDEX_DIR / "faiss_index")
        self.vectorstore: FAISS | None = self.load_index()

    def load_index(self) -> FAISS | None:
        """Load an existing FAISS index from disk if it exists."""
        file_index = Path(self.index_path)
        if file_index.exists():
            print(f"Loading existing FAISS index from {self.index_path}")
            return FAISS.load_local(self.index_path, self.embedding)
        return None

    def process_pdf(self, pdf_path: Path) -> list[Document]:
        """Process a single PDF by extracting its text and splitting it into chunks."""
        print(f"Processing PDF: {pdf_path.name}")
        text, _ = extract_pdf_content(pdf_path, pdf_path.name)
        return split_text_into_chunks(text)

    def process_all_pdfs_in_directory(self, directory: Path, save: bool = True) -> None:
        """Process all PDF files in a directory. Each file is indexed separately. Optionally saves each index to disk to avoid reprocessing."""
        for pdf_file in directory.glob("*.pdf"):
            try:
                # Normalize and define the name for this FAISS index
                index_name = pdf_file.stem.replace(" ", "_").lower()
                index_path = Config.paths.INDEX_DIR / f"index_faiss_{index_name}"

                # Skip processing if the index already exists
                if index_path.exists():
                    print(f"Index already exists for {pdf_file.name}. Skipping.")
                    continue

                # Extract content and convert it into chunks
                chunks = self.process_pdf(pdf_file)
                if not chunks:
                    print(f"No content extracted from {pdf_file.name}. Skipping.")
                    continue

                docs_with_metadata = [Document(page_content=chunk.page_content, metadata={"source": pdf_file.name}) for chunk in chunks]

                # Create a FAISS vectorstore from the chunks with metadata
                vectorstore = FAISS.from_documents(docs_with_metadata, self.embedding)

                # Save the FAISS index locally if requested
                if save:
                    vectorstore.save_local(index_path)
                    print(f"Saved FAISS index to {index_path}")

            except Exception as exc:  # noqa: BLE001
                print(f"Error processing {pdf_file.name}: {exc}")
