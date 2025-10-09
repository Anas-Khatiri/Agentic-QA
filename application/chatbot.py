import logging

import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

from src.agents.qa_agent import QAAgent
from src.handlers.ocr_handler import download_and_process_images
from src.handlers.pdf_handler import download_and_process_pdfs
from src.handlers.visualizer_handler import (
    visualize_sales_vs_stock_correlation,
    visualize_stock_vs_index,
    visualize_vehicles_sold_per_year,
)
from src.handlers.youtube_handler import download_and_process_youtube_contents

# Suppress verbose logs
logging.getLogger("pdfminer").setLevel(logging.ERROR)


class Chatbot:
    """A Streamlit chatbot for document QA and visualization."""

    def __init__(self) -> None:
        """Initialize the Chatbot with optional memory and QA agent."""
        self.qa_agent: QAAgent | None = None
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(return_messages=True)

    # ---------------------------------------
    # PAGE 1: HOME — Upload, Process & Ask
    # ---------------------------------------
    def show_home(self) -> None:
        """Display the home page with upload, processing, and question answering interface."""
        self._render_header()
        uploaded_files = self._file_uploader()
        yt_url = self._youtube_input()

        if st.button("🚀 Process"):
            if uploaded_files:
                for file in uploaded_files:
                    self._process_uploaded_file(file)
            elif yt_url.strip():
                self._process_youtube_url(yt_url)
            else:
                st.warning("Please upload a file or provide a YouTube URL.")

        question = self._question_input()
        if question.strip():
            ai_answer = self._handle_question(question)
            self._log_and_display_answer(question, ai_answer)

    # ---------------- Helper Methods ----------------
    def _render_header(self) -> None:
        """Render the page header and description."""
        st.markdown(
            """
            <h1 style='text-align: center;'>🤖 Agentic Document QA System</h1>
            <p style='text-align: center;'>Upload PDFs or images, process YouTube videos, and ask questions from your data.</p>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("📂 Upload and Process Files or YouTube Videos")

    def _file_uploader(self) -> list | None:
        """Display file uploader widget."""
        return st.file_uploader(
            "📁 Upload PDF(s) or Image(s)",
            type=["pdf", "png", "jpg", "jpeg", "tiff"],
            accept_multiple_files=True,
        )

    def _youtube_input(self) -> str:
        """Display YouTube URL input widget."""
        return st.text_input("🎥 Or enter a YouTube video URL:")

    def _process_uploaded_file(self, uploaded_file: object) -> None:
        """Process a single uploaded PDF or image."""
        file_type = uploaded_file.type.lower()
        file_name = uploaded_file.name
        st.info(f"📄 Detected file: {file_name} ({file_type})")

        try:
            if "pdf" in file_type:
                st.info(f"Processing {file_name} as PDF...")
                download_and_process_pdfs(pdf_file=uploaded_file)
                st.success(f"✅ {file_name} processed successfully (PDF).")
                self._log_file_message(file_name, "PDF")
            elif any(ext in file_type for ext in ["image", "png", "jpg", "jpeg", "tiff"]):
                st.info(f"Processing {file_name} as image...")
                download_and_process_images(image_file=uploaded_file)
                st.success(f"✅ {file_name} processed successfully (image OCR).")
                self._log_file_message(file_name, "image")
            else:
                st.warning(f"⚠️ Unsupported file type: {file_name}")
        except (FileNotFoundError, ValueError) as e:
            st.error(f"❌ Error processing {file_name}: {e}")

    def _process_youtube_url(self, yt_url: str) -> None:
        """Process a YouTube video URL."""
        try:
            st.info("🎬 Processing YouTube video...")
            download_and_process_youtube_contents(yt_url=yt_url)
            st.success("✅ YouTube video processed and indexed.")
            self._log_file_message(yt_url, "YouTube URL")
        except (FileNotFoundError, ValueError) as e:
            st.error(f"❌ Error processing YouTube video: {e}")

    def _question_input(self) -> str:
        """Display question input widget."""
        st.divider()
        st.subheader("❓ Ask a Question")
        return st.text_input("💬 Ask a question:")

    def _handle_question(self, question: str) -> str:
        """Answer a user question, possibly generating visualizations."""
        q_lower = question.lower()

        if "vehicles sold per year" in q_lower:
            st.info("Generating graph: vehicles sold per year...")
            visualize_vehicles_sold_per_year(streamlit=True)
            return "📊 Generated graph: vehicles sold per year."

        if "stock price" in q_lower and "cac40" in q_lower:
            st.info("Generating graph: stock vs CAC40...")
            visualize_stock_vs_index(streamlit=True)
            return "📊 Compared Renault stock vs CAC40."

        if "correlation" in q_lower and "sales" in q_lower and "stock" in q_lower:
            st.info("Generating graph: sales vs stock correlation...")
            visualize_sales_vs_stock_correlation(streamlit=True)
            return "📊 Analyzed correlation between sales and stock."

        if self.qa_agent is None:
            try:
                self.qa_agent = QAAgent()
            except FileNotFoundError as e:
                st.error(f"❌ Error loading QA system: {e}")
                return "Error loading QA system."
        return self.qa_agent.answer_question(question)

    def _log_file_message(self, name: str, type_: str) -> None:
        """Log file upload or processing to conversation memory."""
        st.session_state.memory.chat_memory.add_message(HumanMessage(content=f"Uploaded {type_}: {name}"))
        st.session_state.memory.chat_memory.add_message(AIMessage(content=f"Processed {name} as {type_}."))

    def _log_and_display_answer(self, question: str, ai_answer: str) -> None:
        """Log question and answer, then display the answer."""
        st.session_state.memory.chat_memory.add_message(HumanMessage(content=question))
        st.session_state.memory.chat_memory.add_message(AIMessage(content=ai_answer))
        st.success(f"🧠 Answer: {ai_answer}")

    # ---------------------------------------
    # PAGE 2: CONVERSATION HISTORY
    # ---------------------------------------
    def show_history(self) -> None:
        """Display conversation history."""
        st.markdown("<h1 style='text-align: center;'>📝 Conversation History</h1>", unsafe_allow_html=True)
        messages = st.session_state.memory.chat_memory.messages
        if not messages:
            st.info("No conversation history yet.")
            return
        for msg in messages:
            role = "🧑 User" if isinstance(msg, HumanMessage) else "🤖 AI"
            st.markdown(f"**{role}:** {msg.content}")

    # ---------------------------------------
    # MAIN NAVIGATION
    # ---------------------------------------
    def run(self) -> None:
        """Run the chatbot application with sidebar navigation."""
        st.set_page_config(page_title="Agentic QA", layout="wide")
        st.sidebar.title("📘 Navigation")
        page = st.sidebar.radio("Go to:", ["🏠 Home", "💬 Conversation History"])
        if page == "🏠 Home":
            self.show_home()
        else:
            self.show_history()
