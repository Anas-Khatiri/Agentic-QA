import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from configs.config import Config

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    error_msg = "Missing HUGGINGFACE_TOKEN in .env"
    raise ValueError(error_msg)


class QAAgent:
    """Question-answering agent using FAISS vector store and HuggingFace chat LLM."""

    def __init__(self) -> None:
        """Initialize embeddings, LLM client, and FAISS vector store."""
        # Initialize HuggingFace InferenceClient
        self.client: InferenceClient = InferenceClient(api_key=hf_token)

        # Initialize embeddings
        self.embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=Config.models.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )

        # Load FAISS vector store
        self.vectorstore: FAISS = self.load_all_indexes(Config.paths.INDEX_DIR)

    def load_all_indexes(self, base_dir: Path) -> FAISS:
        """Load and merge all FAISS indexes from the given directory."""
        base_store: FAISS | None = None
        index_dirs = sorted(base_dir.glob("index_faiss_*"))

        if not index_dirs:
            msg = f"No FAISS indexes found in {base_dir}"
            raise FileNotFoundError(msg)

        for path in index_dirs:
            if not path.is_dir():
                continue
            print(f"Loading index from: {path}")
            store = FAISS.load_local(
                str(path),
                self.embedding,
                allow_dangerous_deserialization=True,
            )
            if base_store is None:
                base_store = store
            else:
                base_store.merge_from(store)

        return base_store

    def build_prompt(self, context: str, question: str) -> str:
        """Construct the LLM prompt using context and user question."""
        system_instruction = (
            "You are a helpful and precise AI assistant. "
            "Use the context to answer the user's question as accurately as possible. "
            "If the context does not contain a clear answer, respond with: 'Not found in context.'"
        )
        prompt = f"{system_instruction}\n\n" f"Context:\n{context.strip()}\n\n" f"Question: {question.strip()}\n\n" f"Answer:"
        return prompt

    def answer_question(self, question: str) -> str:
        """Answer a user question using FAISS semantic search and LLM completion."""
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=Config.processing.RETRIEVER_K)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = self.build_prompt(context, question)

        try:
            # Use HuggingFace chat completions
            completion = self.client.chat.completions.create(
                model=Config.models.LLM_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.models.LLM_TEMPERATURE,
            )
            text: str = completion.choices[0].message["content"]
            return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()

        except Exception as exc:  # noqa: BLE001
            error_msg = f"Error generating answer. Question: {question!r}. Exception: {exc}"
            return f"❌ {error_msg}"
