"""Async Groq client with schema validation and retries (the notebook's reliability layer)."""
import json
import logging
from typing import Optional, Type, TypeVar

from groq import AsyncGroq
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import settings
from .prompts import SYSTEM_PROMPT, build_classification_prompt, build_extraction_prompt
from .schemas import NoteClassification, NoteExtraction

log = logging.getLogger("clinical-notes.llm")
T = TypeVar("T", bound=BaseModel)


class PipelineError(Exception):
    """Single error type for network, JSON and schema failures, so the retry layer watches one thing."""


class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.model = model
        self._client = AsyncGroq(api_key=api_key)

    async def _call(self, prompt: str, max_tokens: int = 700) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
           retry=retry_if_exception_type(PipelineError), reraise=True)
    async def _run(self, prompt: str, schema: Type[T]) -> T:
        try:
            return schema(**json.loads(await self._call(prompt)))
        except Exception as exc:
            log.warning("LLM attempt failed: %s: %s", type(exc).__name__, str(exc)[:120])
            raise PipelineError(f"{type(exc).__name__}: {exc}") from exc

    async def classify(self, note: str) -> NoteClassification:
        return await self._run(build_classification_prompt(note), NoteClassification)

    async def extract(self, note: str) -> NoteExtraction:
        return await self._run(build_extraction_prompt(note), NoteExtraction)


def build_llm_client() -> Optional[LLMClient]:
    if not settings.groq_api_key:
        return None
    return LLMClient(settings.groq_api_key, settings.groq_model)
