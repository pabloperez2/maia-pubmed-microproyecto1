# MAIA · PubMed Microproyecto 1

Clasificador de oraciones de abstracts médicos. Usa **`allenai/scibert_scivocab_cased`**
como modelo base, fine-tuned sobre `armanc/pubmed-rct20k` (PubMed-RCT20k).
El modelo fine-tuned se identifica internamente como `allenai/scibert_scivocab_cased-v0.0.1`
donde `v0.0.1` es la versión del fine-tune, no una versión oficial de HuggingFace.

Clasifica cada oración en 5 categorías IMRAD:
`Background` · `Objective` · `Methods` · `Results` · `Conclusions`

```
solucion/
├── download_model.py          ← PASO 1: descargar modelo desde Google Drive
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── README.md
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       └── model/scibert_pubmed/  ← NO está en Git, se descarga con download_model.py
│           ├── config.json
│           ├── label_meta.json
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── training_args.bin
│           └── model.safetensors
└── frontend/
    └── maia-pubmed-microproyecto1.html
```

---

## Pre-requisitos

| Herramienta | Versión mínima | Verificar |
|---|---|---|
| Python | 3.10 | `python --version` |
| Docker Desktop / Docker Engine | 24.0 | `docker --version` |
| Docker Compose | 2.20 | `docker compose version` |
| Git | 2.x | `git --version` |

> **RAM recomendada:** 4 GB disponibles para el contenedor (carga del modelo SciBERT en CPU).

---

## Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/pabloperez2/maia-pubmed-microproyecto1.git
cd maia-pubmed-microproyecto1
```

---

## Paso 2 — Descargar el modelo desde Google Drive

Los 6 archivos del modelo **no están en el repositorio Git** (superan el límite de 100 MB de GitHub).
Se descargan desde Google Drive con el script incluido:

```bash
python download_model.py
```

El script:
- Instala `gdown` automáticamente si no está disponible
- Descarga los 6 archivos a `backend/app/model/scibert_pubmed/`
- Valida la integridad de cada archivo con MD5
- Si todos los archivos ya existen y son válidos, omite la descarga

**Salida esperada:**
```
Descargando modelo SciBERT completo desde Google Drive...
  FOLDER_ID : 1ue3EECOilm3U11mV3xctRNDDung25Mg0
  Destino   : .../backend/app/model/scibert_pubmed

Retrieving folder contents
Processing file 1/6: model.safetensors
Downloading... 100%|████████████| 419M/419M [00:38<00:00, 11.0MB/s]
...

Validando integridad de archivos descargados...
  model.safetensors              ✓  419.0 MB  md5 ✓
  tokenizer.json                 ✓  0.7 MB  md5 ✓
  tokenizer_config.json          ✓  0.0 MB  md5 ✓
  config.json                    ✓  0.0 MB  md5 ✓
  label_meta.json                ✓  0.0 MB  md5 ✓
  training_args.bin              ✓  0.0 MB  md5 ✓

✓ Modelo completo e íntegro en: .../backend/app/model/scibert_pubmed

Ahora puede construir el contenedor:
  docker compose build
  docker compose up -d
```

> **Sin credenciales:** No se requieren claves AWS ni tokens. Solo conexión a internet.

---

## Paso 3 — Instalar Docker

### Windows
1. Descargar **Docker Desktop** desde https://www.docker.com/products/docker-desktop
2. Ejecutar el instalador y reiniciar el sistema
3. Abrir Docker Desktop y esperar a que el motor arranque (ícono en la barra de tareas)
4. Verificar en terminal:
```powershell
docker --version
docker compose version
```

### macOS
```bash
# Con Homebrew:
brew install --cask docker

# Abrir Docker Desktop desde Aplicaciones y esperar a que arranque
docker --version
docker compose version
```

### Linux (Ubuntu / Debian)
```bash
# Instalar Docker Engine
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Agregar usuario al grupo docker (evita usar sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verificar
docker --version
docker compose version
```

---

## Paso 4 — Construir la imagen Docker

Desde la raíz del proyecto (donde está el `Dockerfile`):

```bash
docker compose build
```

> La primera vez descarga e instala las dependencias (~950 MB: torch, transformers, etc.).
> Las capas quedan cacheadas — builds posteriores son mucho más rápidos.

---

## Paso 5 — Levantar el contenedor

```bash
docker compose up -d
```

Ver logs en tiempo real:
```bash
docker compose logs -f
```

> El contenedor tarda **30–60 segundos** en estar listo la primera vez porque carga
> el modelo SciBERT (419 MB) en memoria. El healthcheck lo refleja como `starting`
> durante ese período — es comportamiento normal.

---

## Paso 6 — Verificar el despliegue

```bash
# Estado del contenedor (debe aparecer "healthy")
docker compose ps

# Health check de la API
curl http://localhost:8080/health
```

**Respuesta esperada:**
```json
{
  "status": "ok",
  "model_version": "allenai/scibert_scivocab_cased-v0.0.1",
  "model_backend": "scibert_finetuned",
  "uptime_s": 73.2,
  "max_text_length": 8000,
  "max_rpm": 20
}
```

| `model_backend` | Significado |
|---|---|
| `scibert_finetuned` | ✅ Modelo cargado correctamente |
| `heuristic_fallback` | ⚠️ Modelo no encontrado — verificar que `download_model.py` completó sin errores |

**Abrir la interfaz web:**
```
http://localhost:8080/
```

---

## Detener el contenedor

```bash
# Detener (conserva la imagen)
docker compose down

# Detener y eliminar la imagen construida
docker compose down --rmi local
```

---

## Ejecución local sin Docker

Útil para desarrollo y depuración. Requiere Python 3.10+ y las dependencias instaladas.

### 1 — Instalar dependencias

```bash
pip install -r backend/requirements.txt
```

### 2 — Arrancar el backend

#### Linux / macOS
```bash
MODEL_DIR=./backend/app/model/scibert_pubmed \
MODEL_VERSION=allenai/scibert_scivocab_cased-v0.0.1 \
MAX_TEXT_LENGTH=8000 \
MAX_RPM=20 \
uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

#### Windows — CMD
```cmd
set MODEL_DIR=.\backend\app\model\scibert_pubmed
set MODEL_VERSION=allenai/scibert_scivocab_cased-v0.0.1
set MAX_TEXT_LENGTH=8000
set MAX_RPM=20
uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

#### Windows — PowerShell
```powershell
$env:MODEL_DIR=".\backend\app\model\scibert_pubmed"
$env:MODEL_VERSION="allenai/scibert_scivocab_cased-v0.0.1"
$env:MAX_TEXT_LENGTH="8000"
$env:MAX_RPM="20"
uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

### 3 — Verificar

```bash
curl http://localhost:8080/health
```

> El modelo tarda **30–60 segundos** en cargar la primera vez.
> Revisar el campo `model_backend` en la respuesta — debe ser `scibert_finetuned`.

### Arranque mínimo (sin variables de entorno)

Si solo se quiere probar la heurística sin cargar el modelo:

#### Linux / macOS
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

#### Windows — CMD
```cmd
uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

#### Windows — PowerShell
```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
```

> Sin `MODEL_DIR`, el backend arranca en modo `heuristic_fallback` en menos de 1 segundo.

---

## Variables de entorno

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `MODEL_VERSION` | `allenai/scibert_scivocab_cased-v0.0.1` | Versión mostrada en la UI y en `/health` |
| `MODEL_DIR` | `./backend/app/model/scibert_pubmed` | Ruta al directorio del modelo |
| `MAX_TEXT_LENGTH` | `8000` | Límite de caracteres por request |
| `MAX_RPM` | `20` | Máximo requests por minuto por IP |
| `FRONTEND_DIR` | `/app/frontend` | Ruta a los archivos estáticos del SPA |
| `ALLOWED_ORIGINS` | `http://localhost:8080,...` | Orígenes CORS permitidos (coma separados) |

---

## API

### `POST /api/predict`

**Body:**
```json
{ "text": "Pegue aquí el abstract completo..." }
```

**Respuesta:**
```json
{
  "model_version": "allenai/scibert_scivocab_cased-v0.0.1",
  "sentences": [
    {
      "index": 1,
      "start": 0,
      "end": 95,
      "sentence": "This study evaluates...",
      "label": "Objective",
      "confidence": 0.934
    }
  ]
}
```

**Códigos de respuesta:**

| Código | Significado |
|---|---|
| `200` | Clasificación exitosa |
| `400` | Texto vacío o contiene patrones de inyección |
| `413` | Texto supera `MAX_TEXT_LENGTH` caracteres |
| `422` | Error de validación del body JSON |
| `429` | Rate limit excedido — reintentar en 60s |

### `GET /health`

```json
{
  "status": "ok",
  "model_version": "allenai/scibert_scivocab_cased-v0.0.1",
  "model_backend": "scibert_finetuned",
  "uptime_s": 120.4,
  "max_text_length": 8000,
  "max_rpm": 20
}
```

---

## Modelo

| Campo | Valor |
|---|---|
| Modelo base | `allenai/scibert_scivocab_cased` |
| Fine-tune | `armanc/pubmed-rct20k` (PubMed-RCT20k) |
| Versión fine-tune | `v0.0.1` (etiqueta interna del equipo) |
| Arquitectura | `BertForSequenceClassification` (BERT-base, 12 capas) |
| Vocabulario | 31 116 tokens científicos (SciBERT cased) |
| Formato de pesos | SafeTensors — `model.safetensors` (419 MB) |
| Clases | 5 — background, conclusions, methods, objective, results |

---

## Seguridad

- Validación de inyección XSS y SQLi en entrada (backend + frontend)
- Rate limiting: 20 RPM por IP (configurable con `MAX_RPM`)
- Headers de seguridad: `X-Frame-Options`, `X-Content-Type-Options`, `CSP`
- CORS restringido a orígenes en `ALLOWED_ORIGINS`
- Límite de payload configurable con `MAX_TEXT_LENGTH`
