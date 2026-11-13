# Technical Document Intelligence RAG — multi-stage Docker build (Stage 58)
# Builder installs pinned dependencies; runtime uses python:3.12-slim,
# non-root user (`rag`), and only the installed package plus data volumes.
# Secrets (`OPENAI_API_KEY`) are injected at deploy time, never baked.

FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage: minimal image, non-root, no build artifacts.
FROM python:3.12-slim AS runtime

# Non-root user (§10, isolation).
RUN useradd --create-home --shell /bin/bash rag
WORKDIR /app

# Copy installed packages from builder (same Python version).
COPY --from=builder /root/.local /home/rag/.local
ENV PATH=/home/rag/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Project code: only `app/`, `scripts/`, and config — no uploads or data.
COPY --chown=rag:rag app/ ./app/
COPY --chown=rag:rag scripts/ ./scripts/
COPY --chown=rag:rag .env.example ./
COPY --chown=rag:rag pyproject.toml ./
COPY --chown=rag:rag README.md ./

# Health endpoint is served by the API layer (§5.6); readiness checks
# the configured dependencies (`ChromaIndex`, `LLMClient`) before traffic.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=1)" || exit 1

USER rag
EXPOSE 8000

# Default: serve with `uvicorn` (§9). The image itself has no live OpenAI
# key baked in; `.env` is mounted at deploy time (§10, secrets).
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
