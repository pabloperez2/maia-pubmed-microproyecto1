# MAIA - UNIANDES
# Edwin Cifuentes

# Esqueleto de backEnd
# Utiliza FastAPI
# Solo busca palabras clave en las oraciones para definir una categoria
# Las oraciones se determinan por signos de puntación comunes
# valor de confianza no es preciso
# 

from __future__ import annotations

import os
import re
from typing import List, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

Label = Literal["Background", "Objective", "Methods", "Results", "Conclusions"]

MODEL_VERSION = os.getenv("MODEL_VERSION", "rhetoric-heuristic-v0.0.1")

app = FastAPI(title="Rhetorical Sentence Classifier", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, description="Texto fuente (abstract completo)")

class SentenceOut(BaseModel):
    index: int
    start: int = Field(..., ge=0, description="Índice de inicio (0-based) en el texto original")
    end: int = Field(..., ge=0, description="Índice de fin (exclusivo) en el texto original")
    sentence: str
    label: Label
    confidence: float = Field(..., ge=0.0, le=1.0)

class PredictOut(BaseModel):
    model_version: str
    sentences: List[SentenceOut]

def iter_sentence_spans(text: str):
    """
    Segmentación ligera con offsets sobre el texto original (preserva whitespace original).
    Divide por delimitadores . ! ? seguidos de espacio(s)/saltos de línea o fin de texto.
    Para producción, se debe considerar segmentación robusta con offsets (spaCy/Stanza).
    """
    if not text:
        return
    end_pat = re.compile(r"[.!?](?=(\s+|$))")
    start = 0
    for m in end_pat.finditer(text):
        end = m.end()
        yield (start, end)
        start = end
    if start < len(text):
        yield (start, len(text))

def predict_label(sentence: str) -> tuple[Label, float]:
    """
    Heurística determinista (mock): clasifica por palabras clave.
    - Si no hay evidencia de palabras clave, retorna Background como categoría por defecto.
    """
    s = sentence.lower()

    obj_kw = ["objective", "aim", "purpose", "we aim", "this study aims", "our goal", "investigate"]
    meth_kw = ["method", "methods", "we used", "we conducted", "randomized", "trial", "participants", "patients", "dataset", "procedure"]
    res_kw = ["results", "we found", "significant", "increased", "decreased", "improved", "associated", "accuracy", "auc", "p<", "p <"]
    con_kw = ["conclusion", "we conclude", "suggest", "indicate", "in summary", "therefore", "these findings"]

    def score(kws: List[str]) -> int:
        return sum(1 for k in kws if k in s)

    scores = {
        "Background": 0,
        "Objective": score(obj_kw),
        "Methods": score(meth_kw),
        "Results": score(res_kw),
        "Conclusions": score(con_kw),
    }

    max_score = max(scores.values())
    if max_score == 0:
        best: Label = "Background"
    else:
        # En caso de empate, aplicamos una prioridad razonable (más específica primero).
        priority = ["Results", "Methods", "Objective", "Conclusions", "Background"]
        best = max(priority, key=lambda k: scores[k])  # type: ignore[arg-type]

    base = 0.60
    conf = base + min(scores[best] * 0.10, 0.30) + min(len(s) / 300.0, 0.10)
    conf = max(0.0, min(conf, 0.99))

    return best, float(conf)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/maia-pubmed-microproyecto1.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

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
        outs.append(SentenceOut(index=idx, start=start, end=end, sentence=frag, label=label, confidence=conf))
        idx += 1
    return PredictOut(model_version=MODEL_VERSION, sentences=outs)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")
