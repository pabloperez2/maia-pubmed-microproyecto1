# Single-image repo: backend (FastAPI) + static frontend in /ui
#
# Compatible con: Windows (Docker Desktop + WSL2), macOS (Intel & Apple Silicon),
# Linux (amd64 / arm64).  La imagen base es multi-arch; Docker elige
# automáticamente la variante correcta para cada plataforma.
#
# --platform se puede sobreescribir con:
#   docker buildx build --platform linux/amd64 ...   (forzar x86 en M-series)
FROM --platform=$BUILDPLATFORM python:3.11-slim

# ── Entorno Python ─────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # NLTK buscará los datos aquí dentro del contenedor
    NLTK_DATA=/app/nltk_data \
    # Evita warning de pip sobre usuarios root dentro del contenedor
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Dependencias del sistema ───────────────────────────────────────────────
# curl: usado en el HEALTHCHECK (más portable que el módulo urllib en algunos
#        entornos Windows/WSL donde los señales de proceso difieren).
# dos2unix: convierte line-endings CRLF → LF por si el repo se clonó en
#            Windows con autocrlf=true (protege scripts y requirements.txt).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      dos2unix \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ────────────────────────────────────────────────────
COPY backend/requirements.txt /app/backend/requirements.txt

# Normaliza CRLF → LF antes de que pip lo lea (Windows safety)
RUN dos2unix /app/backend/requirements.txt \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

# ── Datos NLTK (descargados en build, no en runtime) ──────────────────────
RUN python - <<'EOF'
import nltk
for pkg in ("punkt", "punkt_tab"):
    nltk.download(pkg, download_dir="/app/nltk_data", quiet=True)
EOF

# ── Código fuente ──────────────────────────────────────────────────────────
COPY backend  /app/backend
COPY frontend /app/frontend

# Normaliza line-endings en todos los .py y .txt por si vienen de Windows
RUN find /app/backend -name "*.py" -o -name "*.txt" | xargs dos2unix 2>/dev/null || true

# ── Configuración runtime (sobreescribible por docker-compose / -e) ────────
ENV MODEL_VERSION="rhetoric-heuristic-v0.0.1" \
    MAX_TEXT_LENGTH="8000" \
    FRONTEND_DIR="/app/frontend"

EXPOSE 8080

# ── Healthcheck con curl (más robusto que urllib en todos los OS) ──────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# ── Arranque ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
