# MAIA · PubMed Microproyecto 1 — Repo único (Frontend + FastAPI + Docker)

Incluye:

- `frontend/` → interfaz web única: `maia-pubmed-microproyecto1.html`
- `backend/` → API FastAPI con endpoint `POST /api/predict`
- `Dockerfile` + `docker-compose.yml` → despliegue listo

## Ejecutar

### Con Docker
```bash
docker compose up --build
```

### Solo con FastAPI (sin Docker)
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

### Con variables de entorno personalizadas (sin Docker)

**Linux / macOS**
```bash
MAX_TEXT_LENGTH=8000 uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

**Windows CMD**
```cmd
set MAX_TEXT_LENGTH=8000 && uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

**Windows PowerShell**
```powershell
$env:MAX_TEXT_LENGTH="8000"; uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

Abrir:
- UI: http://localhost:8080/ui/maia-pubmed-microproyecto1.html
- Health: http://localhost:8080/health

## Variables de entorno

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `MODEL_VERSION` | `rhetoric-heuristic-v0.0.1` | Versión del modelo mostrada en la UI |
| `MAX_TEXT_LENGTH` | `8000` | Límite de caracteres por request |
| `FRONTEND_DIR` | `../../frontend` (relativo al módulo) | Ruta al directorio de archivos estáticos |

## API

`POST /api/predict`

Body:
```json
{ "text": "Pegue aquí el abstract..." }
```

Respuesta:
```json
{
  "model_version": "rhetoric-heuristic-v0.0.1",
  "sentences": [
    { "index": 1, "start": 0, "end": 95, "sentence": "...", "label": "Background", "confidence": 0.42 }
  ]
}
```

## Nota
El backend incluye una **heurística determinista** (mock). Para integrar al modelo real, reemplazar `predict_label()` en `backend/app/main.py` manteniendo las variables de salida (`label`, `confidence`).

## UI
La interfaz muestra el texto original resaltado por categoría, permite filtrar y ordenar las oraciones clasificadas, y lista los segmentos agrupados por etiqueta. El último análisis se persiste en `localStorage` y se restaura al recargar la página.
