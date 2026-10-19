import json
import secrets
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, func, select

from app.config import settings
from app.db import Chunk, Document, QALog, SessionLocal
from app.ingest import IngestError, ingest_file
from app.rag import answer, index


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.admin_api_token):
        raise HTTPException(status_code=401, detail="Invalid API key")


public = APIRouter()
router = APIRouter(dependencies=[Depends(require_api_key)])


class DocumentOut(BaseModel):
    id: int
    filename: str
    chunks: int
    created_at: datetime


class AskIn(BaseModel):
    question: str


class SourceOut(BaseModel):
    label: str
    score: float


class AskOut(BaseModel):
    answer: str
    sources: list[SourceOut]


class LogOut(BaseModel):
    id: int
    user_id: int | None
    question: str
    answer: str
    sources: list[dict]
    created_at: datetime


@public.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/documents")
async def list_documents() -> list[DocumentOut]:
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(Document, func.count(Chunk.id))
                .outerjoin(Chunk, Chunk.document_id == Document.id)
                .group_by(Document.id)
                .order_by(Document.id)
            )
        ).all()
    return [
        DocumentOut(id=d.id, filename=d.filename, chunks=n, created_at=d.created_at) for d, n in rows
    ]


@router.post("/documents", status_code=201)
async def upload_document(file: UploadFile) -> DocumentOut:
    filename = Path(file.filename or "document.txt").name
    path = settings.upload_dir / f"{uuid.uuid4().hex}_{filename}"
    path.write_bytes(await file.read())
    try:
        doc, chunk_count = await ingest_file(path, filename)
    except IngestError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))
    return DocumentOut(id=doc.id, filename=doc.filename, chunks=chunk_count, created_at=doc.created_at)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: int) -> None:
    async with SessionLocal() as session:
        if await session.get(Document, doc_id) is None:
            raise HTTPException(status_code=404, detail="Document not found")
        await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
        await session.execute(delete(Document).where(Document.id == doc_id))
        await session.commit()
    await index.reload()


@router.post("/ask")
async def ask(body: AskIn) -> AskOut:
    reply, sources = await answer(body.question, history=[])
    return AskOut(
        answer=reply, sources=[SourceOut(label=s.label, score=round(s.score, 4)) for s in sources]
    )


@router.get("/logs")
async def list_logs(limit: int = 20) -> list[LogOut]:
    async with SessionLocal() as session:
        logs = (await session.scalars(select(QALog).order_by(QALog.id.desc()).limit(limit))).all()
    return [
        LogOut(
            id=log.id,
            user_id=log.user_id,
            question=log.question,
            answer=log.answer,
            sources=json.loads(log.sources),
            created_at=log.created_at,
        )
        for log in logs
    ]
