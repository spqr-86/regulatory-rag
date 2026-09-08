# Application image — Streamlit UI (8502) and FastAPI (8503) share one image.
# The monitoring stack (Postgres + Grafana) is separate: see docker-compose.yml.
#
# The index and source documents are gitignored and NOT baked in — mount them:
#   docker build -t regulatory-rag .
#   docker run --rm -p 8502:8502 --env-file .env \
#     -v "$PWD/chroma_db:/app/chroma_db" -v "$PWD/source_docs:/app/source_docs" \
#     regulatory-rag
#   # API instead of UI:
#   docker run --rm -p 8503:8503 --env-file .env -v "$PWD/chroma_db:/app/chroma_db" \
#     regulatory-rag uvicorn api:app --host 0.0.0.0 --port 8503

FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch pulls the CUDA build by default; this project runs CPU-only.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt


FROM python:3.13-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app . .
RUN mkdir -p /app/chroma_db /app/source_docs /app/logs /app/.cache && chown -R app:app /app
USER app

EXPOSE 8502 8503

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8502/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port", "8502", "--server.address", "0.0.0.0"]
