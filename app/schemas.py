"""Pydantic schemas: LLM output validation (from the notebook) and the public API models."""
import re
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .constants import GROUPS

GroupEnum = Enum("GroupEnum", {re.sub(r"[^0-9A-Za-z]+", "_", g).strip("_").upper(): g for g in GROUPS})


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()


_GROUP_LOOKUP = {_norm(g): g for g in GROUPS}


class NoteClassification(BaseModel):
    """What the LLM must return. Anything outside the 11 groups is rejected."""
    groups: List[GroupEnum] = Field(min_length=1, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("groups", mode="before")
    @classmethod
    def normalize_groups(cls, value):
        if isinstance(value, str):
            value = [value]
        cleaned = []
        for item in value:
            item = _GROUP_LOOKUP.get(_norm(item), item) if isinstance(item, str) else item
            if item not in cleaned:
                cleaned.append(item)
        return cleaned[:3]

    @field_validator("reasoning")
    @classmethod
    def trim_reasoning(cls, value):
        return value[:200]


class NoteExtraction(BaseModel):
    patient_age: Optional[float] = Field(default=None, ge=0, le=120)
    patient_sex: Literal["Male", "Female", "Unknown"] = "Unknown"
    note_type: Literal["Operative", "Consult", "Progress", "Discharge", "Imaging", "Other"] = "Other"
    procedures: List[str] = Field(default_factory=list)
    diagnoses: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    anesthesia_used: Optional[bool] = None


# ------------------------------- API models ---------------------------------
Mode = Literal["local", "llm", "auto"]


class NoteRequest(BaseModel):
    note: str = Field(description="Raw clinical transcription.", examples=[
        "PROCEDURE: Colonoscopy. The patient was sedated and the scope advanced to the cecum."])


class BatchRequest(BaseModel):
    notes: List[str] = Field(min_length=1)


class ClassificationResponse(BaseModel):
    groups: List[str]
    confidence: float
    source: Literal["local", "llm", "fallback"]
    scores: Optional[Dict[str, float]] = Field(default=None, description="Per-group probabilities (local model only).")


class BatchResponse(BaseModel):
    count: int
    results: List[ClassificationResponse]


class AnalyzeResponse(BaseModel):
    classification: ClassificationResponse
    extraction: NoteExtraction
    extraction_used_fallback: bool


class HealthResponse(BaseModel):
    status: str
    local_model_loaded: bool
    llm_enabled: bool
    llm_model: Optional[str] = None
