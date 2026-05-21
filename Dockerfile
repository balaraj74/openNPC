# --- OpenNPC Inference Server ---
# Build:  docker build -t opennpc-api .
# Run:    docker run -p 8787:8787 -e OPENNPC_API_KEY=change-me opennpc-api

FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache layer)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[api]"

# Copy source code
COPY opennpc/ opennpc/
COPY configs/ configs/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8787

ENV OPENNPC_ENV=production
ENV OPENNPC_DEBUG=false

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8787/health')" || exit 1

CMD ["uvicorn", "opennpc.api.service:app", "--host", "0.0.0.0", "--port", "8787", "--workers", "2"]
