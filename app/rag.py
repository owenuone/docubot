import json
from dataclasses import asdict, dataclass

import numpy as np
from sqlalchemy import select

from app.config import settings
from app.db import Chunk, Document, QALog, SessionLocal
from app.llm import chat, embed

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the numbered context blocks provided.
- Reply in the same language as the question, in plain text without Markdown formatting.
- Cite the blocks you used inline, like [1] or [2].
- If the context does not contain the answer, say you don't know. Never make things up."""


@dataclass
class Source:
    document: str
    page: int | None
    text: str
    score: float

    @property
    def label(self) -> str:
        return f"{self.document} p.{self.page}" if self.page else self.document


class VectorIndex:
    """In-memory cosine search over all chunks; fine for thousands of chunks."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, int | None, str]] = []
        self._matrix = np.empty((0, 0), dtype=np.float32)

    async def reload(self) -> None:
        async with SessionLocal() as session:
            result = await session.execute(
                select(Document.filename, Chunk.page, Chunk.text, Chunk.embedding).join(
                    Document, Chunk.document_id == Document.id
                )
            )
            rows = result.all()
        self._rows = [(r.filename, r.page, r.text) for r in rows]
        self._matrix = (
            np.vstack([np.frombuffer(r.embedding, dtype=np.float32) for r in rows])
            if rows
            else np.empty((0, 0), dtype=np.float32)
        )

    def search(self, query: np.ndarray, k: int) -> list[Source]:
        if not self._rows:
            return []
        scores = self._matrix @ query
        top = np.argsort(scores)[::-1][:k]
        return [Source(*self._rows[i], score=float(scores[i])) for i in top]


index = VectorIndex()


def build_context(sources: list[Source]) -> str:
    if not sources:
        return "(no documents available)"
    return "\n\n".join(f"[{i}] ({s.label})\n{s.text}" for i, s in enumerate(sources, 1))


async def answer(
    question: str, history: list[dict[str, str]], user_id: int | None = None
) -> tuple[str, list[Source]]:
    query = (await embed([question]))[0]
    sources = index.search(query, settings.top_k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Context:\n{build_context(sources)}\n\nQuestion: {question}"},
    ]
    reply = await chat(messages)

    async with SessionLocal() as session:
        session.add(
            QALog(
                user_id=user_id,
                question=question,
                answer=reply,
                sources=json.dumps([asdict(s) for s in sources], ensure_ascii=False),
            )
        )
        await session.commit()

    return reply, sources
