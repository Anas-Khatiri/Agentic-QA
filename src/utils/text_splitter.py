from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from configs.config import Config


def split_text_into_chunks(text: str) -> list[Document]:
    """Split a long piece of text into smaller overlapping chunks. Useful for preparing text for vectorization and embedding models."""
    # Initialize a recursive character splitter with configured chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.processing.CHUNK_SIZE,  # Max characters per chunk
        chunk_overlap=Config.processing.CHUNK_OVERLAP,  # Overlap between consecutive chunks
    )

    # Wrap the input text in a Document object (LangChain expects this format)
    docs = [Document(page_content=text)]

    # Perform the actual splitting and return the list of chunks
    return splitter.split_documents(docs)
