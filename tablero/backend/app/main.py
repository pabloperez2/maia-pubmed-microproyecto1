# MAIA - UNIANDES
# Edwin Cifuentes
#
# Las oraciones se determinan segmentación NLTK, y es robusta ante abreviaturas médicas comunes.
# valor de confianza esta dado con fines de prueba en tablero, se deberá calcular por los modelos reales en el futuro, 
# esta en la cantidad de señales encontradas y la longitud de la oración.
# 

from __future__ import annotations

import os
import re
import time
import unicodedata
import collections
import threading
import nltk
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# NLTK – descarga el tokenizador de oraciones si no está disponible
# ---------------------------------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# ---------------------------------------------------------------------------
# Tipos y constantes
# ---------------------------------------------------------------------------
Label = Literal["Background", "Objective", "Methods", "Results", "Conclusions"]


# antes SciBERT_fine-tuned-v0.0.1
# ahora allenai/scibert_scivocab_cased-v0.0.1
MODEL_VERSION   = os.getenv("MODEL_VERSION", "allenai/scibert_scivocab_cased-v0.0.1")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "8000"))

_HERE        = Path(__file__).resolve().parent
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", str(_HERE / ".." / ".." / "frontend"))).resolve()

# ---------------------------------------------------------------------------
# Seguridad — DoS: rate limiting por IP
# ---------------------------------------------------------------------------
# Ventana deslizante simple en memoria (no requiere Redis para este scope).
# Límite: MAX_RPM requests por minuto por IP en /api/predict.
MAX_RPM        = int(os.getenv("MAX_RPM", "20"))
_rate_lock     = threading.Lock()
_rate_buckets: dict[str, collections.deque] = {}

def _is_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    window = 60.0
    with _rate_lock:
        if ip not in _rate_buckets:
            _rate_buckets[ip] = collections.deque()
        dq = _rate_buckets[ip]
        # Purgar timestamps fuera de la ventana
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= MAX_RPM:
            return True
        dq.append(now)
        return False

# ---------------------------------------------------------------------------
# Seguridad — XSS / SQLi: validación de contenido del texto
# ---------------------------------------------------------------------------
# Patrones que no tienen cabida legítima en un abstract biomédico en texto
# plano y que son marcadores inequívocos de inyección.
_BLOCK_PATTERNS = re.compile(
    r"(<\s*script|javascript\s*:|data\s*:|vbscript\s*:|"          # XSS vectors
    r"on\w+\s*=|<\s*/?\s*(iframe|object|embed|link|meta|style|"   # HTML injection
    r"form|input|svg|img\s+[^>]*onerror)|"
    r"union\s+select|drop\s+table|insert\s+into|"                 # SQLi keywords
    r"delete\s+from|update\s+\w+\s+set|exec\s*\(|"
    r"--\s*$|;\s*--|/\*.*\*/)",                                    # SQLi comments
    re.IGNORECASE | re.DOTALL,
)

# Proporción máxima de caracteres no imprimibles / de control permitida.
_MAX_NONPRINT_RATIO = 0.02

def _sanitize_text(v: str) -> str:
    """
    Rechaza texto con patrones de inyección claros.
    Normaliza unicode (NFC) y elimina caracteres de control salvo
    saltos de línea y tabulaciones (legítimos en abstracts multi-párrafo).
    """
    # 1. Normalizar unicode → NFC para evitar bypass por homoglifos
    v = unicodedata.normalize("NFC", v)

    # 2. Eliminar caracteres de control ilegítimos (U+0000–U+001F salvo \t \n \r)
    cleaned = "".join(
        ch for ch in v
        if unicodedata.category(ch) not in ("Cc", "Cs") or ch in "\t\n\r"
    )

    # 3. Verificar proporción de caracteres no imprimibles restantes
    non_print = sum(1 for ch in cleaned if unicodedata.category(ch) == "Cc")
    if len(cleaned) > 0 and non_print / len(cleaned) > _MAX_NONPRINT_RATIO:
        raise ValueError("El texto contiene demasiados caracteres de control.")

    # 4. Detectar patrones de inyección HTML/JS/SQL
    if _BLOCK_PATTERNS.search(cleaned):
        raise ValueError(
            "El texto contiene patrones no permitidos (posible intento de inyección)."
        )

    return cleaned

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Medical Abstract Sentence Classifier", version="0.0.1")

# ---------------------------------------------------------------------------
# CORS — restringir a orígenes conocidos (ajustar en producción)
# ---------------------------------------------------------------------------
# ALLOWED_ORIGINS="*" permite acceso desde cualquier origen (red local, etc.).
# Las protecciones XSS/SQLi, rate limiting y security headers siguen activas.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").strip()
_ALLOW_ALL_ORIGINS = _raw_origins == "*"
_ALLOWED_ORIGINS   = ["*"] if _ALLOW_ALL_ORIGINS else [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,          # no cookies/auth cross-origin — incluso con ALLOWED_ORIGINS=*
    allow_methods=["GET", "POST"],    # sólo lo que usa la app
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Middleware: Security headers (mitigan XSS, clickjacking, sniffing)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]          = "DENY"
    response.headers["X-XSS-Protection"]         = "1; mode=block"
    response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"]   = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "   # necesario para el <script> inline del SPA
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    return response

# ---------------------------------------------------------------------------
# Middleware: Rate limiting por IP en /api/predict (mitiga DoS)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/api/predict":
        ip = request.client.host if request.client else "unknown"
        if _is_rate_limited(ip):
            return JSONResponse(
                status_code=429,
                content={"detail": f"Demasiadas solicitudes. Límite: {MAX_RPM} por minuto."},
                headers={"Retry-After": "60"},
            )
    return await call_next(request)

_start_time = time.time()

# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, description="Texto fuente (abstract completo)")

    @field_validator("text")
    @classmethod
    def check_and_sanitize(cls, v: str) -> str:
        # 1. Límite de longitud (DoS)
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"El texto supera el límite permitido de {MAX_TEXT_LENGTH} caracteres "
                f"(recibido: {len(v)})."
            )
        # 2. Sanitización y detección de inyección (XSS / SQLi)
        return _sanitize_text(v)


class SentenceOut(BaseModel):
    index:      int
    start:      int   = Field(..., ge=0, description="Índice de inicio (0-based) en el texto original")
    end:        int   = Field(..., ge=0, description="Índice de fin (exclusivo) en el texto original")
    sentence:   str
    label:      Label
    # NOTA: confidence es un score heurístico artificial, NO una probabilidad calibrada.
    # Rangos orientativos: 0.40–0.55 = baja evidencia, 0.56–0.74 = evidencia media,
    # 0.75–0.90 = evidencia alta.  No debe interpretarse como probabilidad Bayesiana.
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictOut(BaseModel):
    model_version: str
    sentences:     List[SentenceOut]


# ---------------------------------------------------------------------------
# Segmentación con NLTK (robusta ante abreviaturas médicas)
# ---------------------------------------------------------------------------

# Abreviaturas médicas/científicas comunes que NLTK podría confundir con
# fin de oración.  Entrenamos el tokenizador con ellas.
_MEDICAL_ABBREVS = {
    "e.g", "i.e", "vs", "fig", "figs", "et al", "approx", "dept",
    "dr", "prof", "mr", "mrs", "ms", "jr", "sr",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "no", "vol", "p", "pp", "ed", "eds", "cf",
    # unidades clínicas
    "mmhg", "ml", "mg", "kg", "cm", "mm", "iu", "mcg", "μg", "l", "dl",
    "hr", "min", "sec", "wk", "mo",
}

def _build_tokenizer() -> nltk.tokenize.PunktSentenceTokenizer:
    """
    Construye un tokenizador Punkt entrenado con abreviaturas médicas.
    Si falla, devuelve el tokenizador estándar de inglés.
    """
    try:
        base = nltk.data.load("tokenizers/punkt_tab/english.pickle")
    except Exception:
        try:
            base = nltk.data.load("tokenizers/punkt/english.pickle")
        except Exception:
            base = nltk.tokenize.PunktSentenceTokenizer()

    # Ampliar las abreviaturas conocidas
    params = base._params  # type: ignore[attr-defined]
    params.abbrev_types.update(_MEDICAL_ABBREVS)
    return base


_tokenizer = _build_tokenizer()


def iter_sentence_spans(text: str):
    """
    Genera tuplas (start, end) para cada oración del texto usando NLTK Punkt.
    Preserva los offsets originales (incluyendo espacios) para que el frontend
    pueda resaltar exactamente el texto fuente.
    """
    if not text:
        return
    try:
        spans = list(_tokenizer.span_tokenize(text))
    except Exception:
        # Fallback ante errores inesperados: un solo span con todo el texto
        spans = [(0, len(text))]

    for start, end in spans:
        # Extender el span para incluir whitespace/salto de línea previo
        # hasta el siguiente inicio, preservando fidelidad al original.
        yield start, end


# ---------------------------------------------------------------------------
# Clasificador — carga el modelo SciBERT si existe, heurística si no
# ---------------------------------------------------------------------------
#
# El backend detecta automáticamente si MODEL_DIR apunta a un directorio
# con el modelo fine-tuned guardado por el notebook de entrenamiento.
# Si no existe o falla la carga, cae a la heurística determinista como fallback.
#
# Estructura esperada en MODEL_DIR:
#   config.json, pytorch_model.bin (o model.safetensors),
#   tokenizer.json, tokenizer_config.json, label_meta.json
#
MODEL_DIR = os.getenv("MODEL_DIR", "./backend/app/model/scibert_pubmed")    # e.g. ./model/scibert_pubmed

_ml_pipeline  = None   # transformers pipeline, cargado una vez al arrancar
_ml_label2id: dict = {}
_ml_id2label: dict = {}
_using_ml_model = False

def _load_ml_model():
    """Intenta cargar el modelo fine-tuned. Devuelve True si lo logra."""
    global _ml_pipeline, _ml_label2id, _ml_id2label, _using_ml_model
    if not MODEL_DIR:
        return False
    model_path = Path(MODEL_DIR).resolve()
    if not model_path.exists():
        import warnings
        warnings.warn(f"[MAIA] MODEL_DIR no encontrado: {model_path}. Usando heurística.", RuntimeWarning)
        return False
    try:
        import json
        from transformers import pipeline as hf_pipeline

        # Leer label mapping guardado por el notebook
        meta_file = model_path / "label_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            _ml_label2id = meta["label2id"]
            _ml_id2label = {int(k): v for k, v in meta["id2label"].items()}
        else:
            # Fallback: leer desde config.json de HuggingFace
            cfg = json.loads((model_path / "config.json").read_text())
            _ml_id2label  = {int(k): v for k, v in cfg.get("id2label", {}).items()}
            _ml_label2id  = {v: int(k) for k, v in _ml_id2label.items()}

        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1

        _ml_pipeline = hf_pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=device,
            truncation=True,
            max_length=128,
        )
        _using_ml_model = True
        print(f"[MAIA] Modelo SciBERT cargado desde: {model_path}")
        return True
    except Exception as exc:
        import warnings
        warnings.warn(f"[MAIA] No se pudo cargar el modelo ML: {exc}. Usando heurística.", RuntimeWarning)
        return False

_load_ml_model()


def predict_label_ml(sentence: str) -> tuple[Label, float]:
    """Clasificación usando el pipeline de transformers (SciBERT fine-tuned)."""
    result = _ml_pipeline(sentence)[0]
    raw_label = result["label"]           # e.g. "LABEL_2" o "methods"
    score     = float(result["score"])

    # Normalizar la etiqueta al formato capitalizado del API
    # El pipeline puede devolver el string directo del id2label del modelo
    _NORMALIZE = {
        "background":  "Background",
        "objective":   "Objective",
        "methods":     "Methods",
        "results":     "Results",
        "conclusions": "Conclusions",
    }
    if raw_label.upper().startswith("LABEL_"):
        # HuggingFace devuelve "LABEL_0", "LABEL_1", etc. → resolver con id2label
        idx = int(raw_label.split("_")[1])
        raw_label = _ml_id2label.get(idx, raw_label)

    label: Label = _NORMALIZE.get(raw_label.lower(), "Background")  # type: ignore
    # Cap de confianza: alineado con el rango de la heurística para la UI
    return label, round(min(score, 0.999), 3)


# ---------------------------------------------------------------------------
# Clasificador heurístico (fallback cuando no hay modelo ML)
# ---------------------------------------------------------------------------
_KW: dict[Label, list[str]] = {
    "Objective": [
        "objective", "aim", "purpose", "we aim", "this study aims",
        "our goal", "investigate", "to assess", "to evaluate", "to determine",
        "to examine", "to compare", "to identify", "to explore",
    ],
    "Methods": [
        "method", "methods", "we used", "we conducted", "randomized",
        "trial", "participants", "patients", "dataset", "procedure",
        "protocol", "recruited", "assigned", "inclusion criteria",
        "exclusion criteria", "cohort", "retrospective", "prospective",
        "double-blind", "placebo", "sample", "questionnaire", "survey",
        "intervention", "control group",
    ],
    "Results": [
        "results", "we found", "significant", "increased", "decreased",
        "improved", "associated", "accuracy", "auc", "p<", "p <",
        "odds ratio", "hazard ratio", "confidence interval", "ci ",
        "mean", "median", "difference", "reduction", "effect",
        "showed", "demonstrated", "revealed", "observed",
    ],
    "Conclusions": [
        "conclusion", "we conclude", "suggest", "indicate",
        "in summary", "therefore", "these findings", "in conclusion",
        "our results suggest", "collectively", "taken together",
        "future research", "further studies", "implication",
        "clinical practice", "may be useful", "warrants",
    ],
}


def predict_label_heuristic(sentence: str) -> tuple[Label, float]:
    """Heurística determinista por palabras clave (fallback)."""
    s = sentence.lower()

    def score(kws: list[str]) -> float:
        hits = sum(1 for k in kws if k in s)
        length_bonus = min(len(s) / 400.0, 0.08)
        return hits + length_bonus

    raw_scores: dict[str, float] = {
        "Background":  0.0,
        "Objective":   score(_KW["Objective"]),
        "Methods":     score(_KW["Methods"]),
        "Results":     score(_KW["Results"]),
        "Conclusions": score(_KW["Conclusions"]),
    }
    max_score = max(raw_scores.values())

    if max_score < 0.09:
        best: Label = "Background"
        conf = 0.42 + min(len(s) / 500.0, 0.08)
    else:
        priority: list[Label] = ["Results", "Methods", "Objective", "Conclusions", "Background"]
        best = max(priority, key=lambda k: raw_scores[k])  # type: ignore[arg-type]
        base   = 0.55
        signal = raw_scores[best]
        conf   = base + min(signal * 0.08, 0.30) + min(len(s) / 500.0, 0.05)
        conf   = max(0.0, min(conf, 0.90))

    return best, round(float(conf), 3)


def predict_label(sentence: str) -> tuple[Label, float]:
    """
    Punto de entrada único para la clasificación.
    Usa el modelo ML si está disponible; de lo contrario, la heurística.
    """
    if _using_ml_model:
        return predict_label_ml(sentence)
    return predict_label_heuristic(sentence)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/maia-pubmed-microproyecto1.html")


@app.get("/health")
def health():
    return {
        "status":          "ok",
        "model_version":   MODEL_VERSION,
        "model_backend":   "scibert_finetuned" if _using_ml_model else "heuristic_fallback",
        "uptime_s":        round(time.time() - _start_time, 1),
        "max_text_length": MAX_TEXT_LENGTH,
        "max_rpm":         MAX_RPM,
    }


@app.post("/api/predict", response_model=PredictOut)
async def predict(inp: PredictIn, request: Request):
    # Defensa DoS adicional: rechazar bodies excesivamente grandes antes de parsear
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > (MAX_TEXT_LENGTH * 4):
        raise HTTPException(status_code=413, detail="Payload demasiado grande.")

    text = inp.text
    outs: List[SentenceOut] = []
    idx = 1
    for (start, end) in iter_sentence_spans(text):
        frag = text[start:end]
        if frag.strip() == "":
            continue
        label, conf = predict_label(frag)
        outs.append(SentenceOut(
            index=idx, start=start, end=end,
            sentence=frag, label=label, confidence=conf,
        ))
        idx += 1
    return PredictOut(model_version=MODEL_VERSION, sentences=outs)


# ---------------------------------------------------------------------------
# Archivos estáticos del frontend
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
else:
    import warnings
    warnings.warn(
        f"[MAIA] FRONTEND_DIR no encontrado: {FRONTEND_DIR}. "
        "Ajusta la variable de entorno FRONTEND_DIR o la estructura del proyecto.",
        RuntimeWarning,
        stacklevel=1,
    )