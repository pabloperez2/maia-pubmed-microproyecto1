# MAIA - UNIANDES
# Edwin Cifuentes
#
# Solo busca palabras clave en las oraciones para definir una categoria
# Las oraciones se determinan segmentación NLTK, y es robusta ante abreviaturas médicas comunes.
# valor de confianza esta dado con fines de prueba en tablero, se deberá calcular por los modelos reales en el futuro, 
# esta en la cantidad de señales encontradas y la longitud de la oración.
# 

from __future__ import annotations

import os
import re
import time
import nltk
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
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

MODEL_VERSION   = os.getenv("MODEL_VERSION", "rhetoric-heuristic-v0.0.1")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "8000"))   # caracteres

# Directorio del frontend: resolvemos desde una variable de entorno o desde
# la ubicación del propio archivo (independiente del cwd de uvicorn).
_HERE          = Path(__file__).resolve().parent
FRONTEND_DIR   = Path(os.getenv("FRONTEND_DIR", str(_HERE / ".." / ".." / "frontend"))).resolve()

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Rhetorical Sentence Classifier", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_start_time = time.time()

# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, description="Texto fuente (abstract completo)")

    @field_validator("text")
    @classmethod
    def check_length(cls, v: str) -> str:
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"El texto supera el límite permitido de {MAX_TEXT_LENGTH} caracteres "
                f"(recibido: {len(v)})."
            )
        return v


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
# Clasificador heurístico
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


def predict_label(sentence: str) -> tuple[Label, float]:
    """
    Heurística determinista (mock): clasifica por palabras clave ponderadas.

    IMPORTANTE: El valor `confidence` es un score heurístico artificial.
    No debe interpretarse como probabilidad calibrada.
    - Sin evidencia de palabras clave → Background (score mínimo).
    - Con evidencia → score proporcional a la cantidad de palabras clave halladas.
    """
    s = sentence.lower()

    def score(kws: list[str]) -> float:
        hits = sum(1 for k in kws if k in s)
        # Bonus por longitud moderada (oraciones muy cortas son menos fiables)
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

    if max_score < 0.09:          # sin evidencia clara → Background
        best: Label = "Background"
        # Score bajo, confianza mínima para Background
        conf = 0.42 + min(len(s) / 500.0, 0.08)
    else:
        # Desempate por prioridad semántica (más específica primero)
        priority: list[Label] = ["Results", "Methods", "Objective", "Conclusions", "Background"]
        best = max(priority, key=lambda k: raw_scores[k])   # type: ignore[arg-type]

        # Confianza: base 0.55, sube con el número de señales, cap 0.90
        base = 0.55
        signal = raw_scores[best]
        conf   = base + min(signal * 0.08, 0.30) + min(len(s) / 500.0, 0.05)
        conf   = max(0.0, min(conf, 0.90))

    return best, round(float(conf), 3)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/maia-pubmed-microproyecto1.html")


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_version": MODEL_VERSION,
        "uptime_s":      round(time.time() - _start_time, 1),
        "max_text_length": MAX_TEXT_LENGTH,
    }


@app.post("/api/predict", response_model=PredictOut)
def predict(inp: PredictIn):
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