"""Entry point for the Streamlit chatbot application. Instantiates and runs the Chatbot dashboard."""

# Import the Chatbot class that defines the Streamlit application interface
from application.chatbot import Chatbot

# Check if this script is being run as the main program
if __name__ == "__main__":
    # Instantiate and launch the Streamlit chatbot dashboard
    Chatbot().run()
