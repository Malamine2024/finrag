# FinRAG: the API and the demo, in one image. The models stay in Ollama (on the host by default, or in the
# "ollama" service of docker-compose.yml); the index and the indicator table are mounted from ./data.
#
#   docker compose up --build            -> http://localhost:8501 (demo), http://localhost:8000/docs (API)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first: this layer is rebuilt only when the requirements change, not at every code change.
# Serving only: no Docling, PyTorch or ingestion tools (the index is built outside the image).
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src ./src

# Not root: the container only needs to read the code and write to the mounted data folder.
RUN useradd --create-home --uid 1000 finrag && chown -R finrag /app
USER finrag

EXPOSE 8000 8501
CMD ["uvicorn", "finrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
