# Use Python 3.12 Slim to match your environment
FROM python:3.12-slim-bookworm

# 1. Install System Dependencies
# 'poppler-utils' -> Required for pdf2image
# 'libmagic1', 'tesseract-ocr' -> Required for unstructured
# 'build-essential', 'curl' -> Standard tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    libmagic1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Working Directory
WORKDIR /app

# 3. Copy Requirements first (for caching layers)
COPY requirements.txt .

# 4. Install Python Dependencies
# We install PyTorch CPU version explicitly to save ~2GB of space
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Application Code
# This copies src/, data/, and your CSVs into the container
COPY . .

# 6. Set Environment Variables
# Ensure Python finds modules in the current directory
ENV PYTHONPATH=/app

