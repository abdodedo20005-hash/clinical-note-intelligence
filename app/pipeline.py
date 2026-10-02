"""Orchestration: clean -> classify (local / LLM with fallback) -> extract (LLM, empty-but-valid fallback)."""
import asyncio
from typing import List, Optional, Tuple

from .llm import LLMClient, PipelineError
from .local_model import LocalClassifier
from .schemas import ClassificationResponse, NoteExtraction
from .text import clean_note_text


class Pipeline:
    def __init__(self, local: LocalClassifier, llm: Optional[LLMClient], concurrency: int = 5):
        self.local = local
        self.llm = llm
        self._sem = asyncio.Semaphore(concurrency)

    # ------------------------------ classification ------------------------------
    async def classify(self, note: str, mode: str = "local") -> ClassificationResponse:
        text = clean_note_text(note)
        use_llm = self.llm is not None and mode in ("llm", "auto")
        if use_llm:
            try:
                async with self._sem:
                    r = await self.llm.classify(text)
                return ClassificationResponse(groups=[g.value for g in r.groups], confidence=r.confidence, source="llm")
            except PipelineError:
                return await self._local_one(text, source="fallback")
        return await self._local_one(text, source="local")

    async def _local_one(self, text: str, source: str) -> ClassificationResponse:
        pred = (await asyncio.to_thread(self.local.predict, [text]))[0]
        return ClassificationResponse(**pred, source=source)

    async def classify_batch(self, notes: List[str], mode: str = "local") -> List[ClassificationResponse]:
        texts = [clean_note_text(n) for n in notes]
        if self.llm is None or mode == "local":
            preds = await asyncio.to_thread(self.local.predict, texts)  # one vectorised call
            return [ClassificationResponse(**p, source="local") for p in preds]
        return list(await asyncio.gather(*(self.classify(t, mode) for t in texts)))

    # -------------------------------- extraction --------------------------------
    async def extract(self, note: str) -> Tuple[NoteExtraction, bool]:
        """Returns (extraction, used_fallback). Requires an LLM; without one the empty object is returned."""
        if self.llm is None:
            return NoteExtraction(), True
        try:
            async with self._sem:
                return await self.llm.extract(clean_note_text(note)), False
        except PipelineError:
            return NoteExtraction(), True
