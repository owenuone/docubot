import asyncio
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.config import settings
from app.db import Chunk, Document, SessionLocal
from app.llm import embed
from app.rag import index

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


class IngestError(Exception):
    pass


@dataclass
class Page:
    number: int | None
    text: str


def load_pages(path: Path) -> list[Page]:
    match path.suffix.lower():
        case ".pdf":
            reader = PdfReader(path)
            return [Page(i, p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]
        case ".md" | ".txt":
            return [Page(None, path.read_text(encoding="utf-8", errors="ignore"))]
        case suffix:
            raise IngestError(
                f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )


def split_text(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    step = size - overlap
    return [text[i : i + size] for i in range(0, max(len(text) - overlap, 1), step)]


async def ingest_file(path: Path, filename: str) -> tuple[Document, int]:
    loop = asyncio.get_running_loop()
    pages = await loop.run_in_executor(None, load_pages, path)
    pieces = [
        (page.number, chunk)
        for page in pages
        for chunk in split_text(page.text, settings.chunk_size, settings.chunk_overlap)
    ]
    if not pieces:
        raise IngestError(f"No extractable text found in '{filename}'.")

    vectors = await embed([text for _, text in pieces])

    async with SessionLocal() as session:
        doc = Document(filename=filename)
        session.add(doc)
        await session.flush()
        session.add_all(
            Chunk(document_id=doc.id, page=page, text=text, embedding=vec.tobytes())
            for (page, text), vec in zip(pieces, vectors)
        )
        await session.commit()

    await index.reload()
    return doc, len(pieces)
