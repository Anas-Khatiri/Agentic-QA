# -------------------------
# Dockerfile for Streamlit App
# -------------------------

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir pre-commit ruff

# Copy the app code
COPY . .

# Expose Streamlit port
EXPOSE 5000

# Streamlit settings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=5000
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# ✅ Ensure Hugging Face cache points to Railway volume
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models

# Start the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=5000"]
