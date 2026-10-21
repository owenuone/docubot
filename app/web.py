import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse
from loguru import logger
from openai import OpenAIError
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import Message, SessionLocal, Thread
from app.memory import reply

INDEX_HTML = Path(__file__).parent / "static" / "index.html"

VisitorId = Annotated[str, Header(pattern=r"^[0-9a-f-]{36}$")]

page = APIRouter()
router = APIRouter(prefix="/chat/api")
_requests: dict[str, deque[float]] = defaultdict(deque)


def is_rate_limited(key: str) -> bool:
    now = time.monotonic()
    calls = _requests[key]
    while calls and now - calls[0] > 60:
        calls.popleft()
    if len(calls) >= settings.rate_limit_per_minute:
        return True
    calls.append(now)
    return False


class ThreadOut(BaseModel):
    id: int
    title: str
    updated_at: datetime


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


class ThreadDetail(ThreadOut):
    summary: str
    summarized_upto: int
    messages: list[MessageOut]


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class ReplyOut(BaseModel):
    message: MessageOut
    thread: ThreadOut
    summary: str
    summarized_upto: int


async def get_thread(session: AsyncSession, thread_id: int, visitor_id: str) -> Thread:
    thread = await session.get(Thread, thread_id)
    if thread is None or thread.visitor_id != visitor_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


def thread_out(thread: Thread) -> ThreadOut:
    return ThreadOut(id=thread.id, title=thread.title, updated_at=thread.updated_at)


def message_out(message: Message) -> MessageOut:
    return MessageOut(
        id=message.id, role=message.role, content=message.content, created_at=message.created_at
    )


@page.get("/chat", include_in_schema=False)
async def chat_page() -> FileResponse:
    return FileResponse(INDEX_HTML)


@router.get("/threads")
async def list_threads(x_visitor_id: VisitorId) -> list[ThreadOut]:
    async with SessionLocal() as session:
        threads = await session.scalars(
            select(Thread)
            .where(Thread.visitor_id == x_visitor_id)
            .order_by(Thread.updated_at.desc(), Thread.id.desc())
        )
        return [thread_out(t) for t in threads]


@router.post("/threads", status_code=201)
async def create_thread(x_visitor_id: VisitorId) -> ThreadOut:
    async with SessionLocal() as session:
        thread = Thread(visitor_id=x_visitor_id)
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        return thread_out(thread)


@router.get("/threads/{thread_id}")
async def get_thread_detail(thread_id: int, x_visitor_id: VisitorId) -> ThreadDetail:
    async with SessionLocal() as session:
        thread = await get_thread(session, thread_id, x_visitor_id)
        messages = await session.scalars(
            select(Message).where(Message.thread_id == thread.id).order_by(Message.id)
        )
        return ThreadDetail(
            **thread_out(thread).model_dump(),
            summary=thread.summary,
            summarized_upto=thread.summarized_upto,
            messages=[message_out(m) for m in messages],
        )


@router.post("/threads/{thread_id}/messages")
async def send_message(
    thread_id: int, body: MessageIn, request: Request, x_visitor_id: VisitorId
) -> ReplyOut:
    # Rate limit by IP: visitor IDs are client-generated and trivial to rotate.
    if is_rate_limited(request.client.host if request.client else "unknown"):
        raise HTTPException(status_code=429, detail="You're sending messages too fast. Please wait a minute.")

    async with SessionLocal() as session:
        thread = await get_thread(session, thread_id, x_visitor_id)
        count = await session.scalar(
            select(func.count(Message.id)).where(Message.thread_id == thread.id)
        )
        if count >= settings.thread_message_limit:
            raise HTTPException(status_code=429, detail="This demo thread is full. Please start a new chat.")
        try:
            message = await reply(session, thread, body.content)
        except OpenAIError:
            logger.exception("Web chat failed for thread {}", thread.id)
            raise HTTPException(
                status_code=502, detail="The AI service is unavailable right now. Please try again."
            )
        return ReplyOut(
            message=message_out(message),
            thread=thread_out(thread),
            summary=thread.summary,
            summarized_upto=thread.summarized_upto,
        )


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: int, x_visitor_id: VisitorId) -> None:
    async with SessionLocal() as session:
        thread = await get_thread(session, thread_id, x_visitor_id)
        await session.execute(delete(Message).where(Message.thread_id == thread.id))
        await session.delete(thread)
        await session.commit()
