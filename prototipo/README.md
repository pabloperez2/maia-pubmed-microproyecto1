# MAIA · PubMed Microproyecto 1 — Repo único (Frontend + FastAPI + Docker)

Incluye:

- `frontend/` → interfaz web única: `maia-pubmed-microproyecto1.html`
- `backend/` → API FastAPI con endpoint `POST /api/predict`
- `Dockerfile` + `docker-compose.yml` → despliegue listo

## Ejecutar

# Con Docker:
```bash
docker compose up --build
```

# Solo con FastAPI
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

Abrir:
- UI: http://localhost:8000/ui/maia-pubmed-microproyecto1.html
- Health: http://localhost:8000/health

## API

`POST /api/predict`

Body:
```json
{ "text": "Pegue aquí el abstract..." }
```

## Nota
El backend incluye una **heurística determinista** (mock). Para integrar tu modelo real, reemplaza `predict_label()` en `backend/app/main.py` manteniendo las variables de sálida (`label`, `confidence`).


## UI
La interfaz muestra el texto original resaltado por categoría y lista los segmentos hallados agrupados por etiqueta.


