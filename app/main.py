"""FastAPI service for Clinical Note Intelligence."""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query

from .config import settings
from .llm import build_llm_client
from .local_model import LocalClassifier
from .pipeline import Pipeline
from .schemas import (AnalyzeResponse, BatchRequest, BatchResponse, ClassificationResponse,
                      HealthResponse, Mode, NoteRequest)
from .text import clean_note_text

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = Pipeline(LocalClassifier(settings.model_dir), build_llm_client(), settings.llm_concurrency)
    yield


app = FastAPI(
    title="Clinical Note Intelligence API",
    description="Routes clinical transcriptions into 11 note groups (TF-IDF + Logistic Regression) "
                "and extracts structured fields with an LLM (Groq). Demo on de-identified sample notes, "
                "not for clinical use.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_pipeline() -> Pipeline:
    return app.state.pipeline


def validated(note: str) -> str:
    """Guardrail: reject empty / non-text / oversized input before any model sees it."""
    if len(note) > settings.max_note_length:
        raise HTTPException(413, f"Note exceeds {settings.max_note_length} characters.")
    if not any(ch.isalpha() for ch in clean_note_text(note)):
        raise HTTPException(422, "Note is empty or contains no readable text.")
    return note


def require_llm(pipe: Pipeline) -> None:
    if pipe.llm is None:
        raise HTTPException(503, "LLM is not configured. Set GROQ_API_KEY to enable this endpoint.")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(pipe: Pipeline = Depends(get_pipeline)):
    return HealthResponse(status="ok", local_model_loaded=True, llm_enabled=pipe.llm is not None,
                          llm_model=pipe.llm.model if pipe.llm else None)


@app.post("/classify", response_model=ClassificationResponse, tags=["classification"])
async def classify(req: NoteRequest,
                   mode: Mode = Query("local", description="local = TF-IDF model, llm = Groq with local fallback, "
                                                           "auto = same as llm when a key is set."),
                   pipe: Pipeline = Depends(get_pipeline)):
    if mode == "llm":
        require_llm(pipe)
    return await pipe.classify(validated(req.note), mode)


@app.post("/classify/batch", response_model=BatchResponse, tags=["classification"])
async def classify_batch(req: BatchRequest, mode: Mode = Query("local"), pipe: Pipeline = Depends(get_pipeline)):
    if len(req.notes) > settings.max_batch_size:
        raise HTTPException(413, f"At most {settings.max_batch_size} notes per request.")
    if mode == "llm":
        require_llm(pipe)
    notes = [validated(n) for n in req.notes]
    results = await pipe.classify_batch(notes, mode)
    return BatchResponse(count=len(results), results=results)


@app.post("/extract", tags=["extraction"])
async def extract(req: NoteRequest, pipe: Pipeline = Depends(get_pipeline)):
    """Structured fields (age, sex, note type, procedures, diagnoses, medications, anesthesia). Needs the LLM."""
    require_llm(pipe)
    extraction, fallback = await pipe.extract(validated(req.note))
    return {"extraction": extraction, "used_fallback": fallback}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["extraction"])
async def analyze(req: NoteRequest, mode: Mode = Query("local"), pipe: Pipeline = Depends(get_pipeline)):
    """End-to-end capstone: classification + extraction. Extraction falls back to an empty object without an LLM."""
    note = validated(req.note)
    classification = await pipe.classify(note, mode)
    extraction, fallback = await pipe.extract(note)
    return AnalyzeResponse(classification=classification, extraction=extraction, extraction_used_fallback=fallback)
