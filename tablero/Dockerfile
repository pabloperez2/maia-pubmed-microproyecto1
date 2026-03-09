# Single-image repo: backend (FastAPI) + static frontend en /app/frontend
#
# Compatible con: Windows (Docker Desktop + WSL2), macOS (Intel & Apple Silicon),
# Linux (amd64 / arm64).  La imagen base es multi-arch; Docker elige
# automaticamente la variante correcta para cada plataforma.
#
# --platform se puede sobreescribir con:
#   docker buildx build --platform linux/amd64 ...   (forzar x86 en M-series)
FROM --platform=$BUILDPLATFORM python:3.11-slim

# -- Entorno Python ─────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/app/nltk_data \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# -- Dependencias del sistema ───────────────────────────────────────────────
# curl: usado en el HEALTHCHECK.
# dos2unix: convierte CRLF a LF si el repo fue clonado en Windows.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      dos2unix \
    && rm -rf /var/lib/apt/lists/*

# -- Dependencias Python ────────────────────────────────────────────────────
COPY backend/requirements.txt /app/backend/requirements.txt

RUN dos2unix /app/backend/requirements.txt \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

# -- Datos NLTK (descargados en build, no en runtime) ──────────────────────
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data', quiet=True); nltk.download('punkt_tab', download_dir='/app/nltk_data', quiet=True)"

# -- Codigo fuente ──────────────────────────────────────────────────────────
COPY backend  /app/backend
COPY frontend /app/frontend

RUN find /app/backend -name "*.py" -o -name "*.txt" | xargs dos2unix 2>/dev/null || true

# -- Variables de entorno por defecto ──────────────────────────────────────
# Sobreescribibles desde docker-compose o -e en docker run.
ENV MODEL_VERSION="allenai/scibert_scivocab_cased-v0.0.1" \
    MODEL_DIR="/app/backend/app/model/scibert_pubmed" \
    MAX_TEXT_LENGTH="8000" \
    MAX_RPM="20" \
    FRONTEND_DIR="/app/frontend" \
    ALLOWED_ORIGINS="*"

EXPOSE 8080

# -- Healthcheck ────────────────────────────────────────────────────────────
# start_period=60s: tiempo necesario para cargar SciBERT (419 MB) en CPU.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# -- Arranque ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
