FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libfreetype6-dev libpng-dev pkg-config git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p context data/raw data/cleaned data/processed \
             output/reports output/pipeline_results logs

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV MPLBACKEND=Agg

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
