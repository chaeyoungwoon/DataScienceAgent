FROM python:3.11-slim

# System deps for matplotlib, reportlab, torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libfreetype6-dev \
        libpng-dev \
        pkg-config \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create runtime directories
RUN mkdir -p context data/raw data/cleaned data/processed \
             output/reports output/pipeline_results logs

# Non-root user for safety
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Kaggle credentials come from .env / docker-compose env vars
# HuggingFace model cache is volume-mounted to avoid re-downloads
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV MPLBACKEND=Agg

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
