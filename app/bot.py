import html
import time
from collections import defaultdict, deque
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from loguru import logger
from openai import OpenAIError
from sqlalchemy import select

from app.config import settings
from app.db import Document, SessionLocal
from app.ingest import IngestError, ingest_file
from app.rag import Source, answer

TELEGRAM_MESSAGE_LIMIT = 4096

router = Router()
_history: dict[int, deque[dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=settings.history_turns * 2)
)
_requests: dict[int, deque[float]] = defaultdict(deque)


def is_rate_limited(user_id: int) -> bool:
    now = time.monotonic()
    calls = _requests[user_id]
    while calls and now - calls[0] > 60:
        calls.popleft()
    if len(calls) >= settings.rate_limit_per_minute:
        return True
    calls.append(now)
    return False


def format_sources(sources: list[Source]) -> str:
    labels = list(dict.fromkeys(s.label for s in sources))
    # <code> keeps Telegram from auto-linking names like "faq.md" as URLs.
    return f"\n\n📄 Sources: {', '.join(f'<code>{html.escape(l)}</code>' for l in labels)}" if labels else ""


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "Hi! I answer questions based on the uploaded documents.\n\n"
        "Just type your question.\n"
        "/docs – list available documents\n"
        "/reset – start a new conversation"
    )


@router.message(Command("reset"))
async def on_reset(message: Message) -> None:
    _history.pop(message.from_user.id, None)
    await message.answer("Conversation cleared.")


@router.message(Command("docs"))
async def on_docs(message: Message) -> None:
    async with SessionLocal() as session:
        names = (await session.scalars(select(Document.filename).order_by(Document.id))).all()
    text = "\n".join(f"• <code>{html.escape(n)}</code>" for n in names) if names else "No documents uploaded yet."
    await message.answer(text, parse_mode="HTML")


@router.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    if message.from_user.id not in settings.admin_ids:
        await message.answer(
            f"Only admins can upload documents. Your user ID is {message.from_user.id}."
        )
        return

    filename = Path(message.document.file_name or "document.txt").name
    path = settings.upload_dir / f"{message.document.file_unique_id}_{filename}"
    name = f"<code>{html.escape(filename)}</code>"
    status = await message.answer(f"⏳ Indexing {name}…", parse_mode="HTML")
    await bot.download(message.document, destination=path)

    try:
        _, chunk_count = await ingest_file(path, filename)
    except IngestError as e:
        await status.edit_text(f"❌ {html.escape(str(e))}", parse_mode="HTML")
        return
    except OpenAIError:
        logger.exception("Embedding failed for {}", filename)
        await status.edit_text("❌ Embedding service error, please try again later.")
        return
    await status.edit_text(f"✅ {name} indexed ({chunk_count} chunks).", parse_mode="HTML")


@router.message(F.text)
async def on_question(message: Message, bot: Bot) -> None:
    user_id = message.from_user.id
    if is_rate_limited(user_id):
        await message.answer("You're sending questions too fast. Please wait a minute.")
        return

    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        reply, sources = await answer(message.text, list(_history[user_id]), user_id)
    except OpenAIError:
        logger.exception("Answer failed for user {}", user_id)
        await message.answer("Sorry, the AI service is unavailable right now. Please try again later.")
        return

    _history[user_id].extend(
        [{"role": "user", "content": message.text}, {"role": "assistant", "content": reply}]
    )
    # Truncate before escaping so the cut can never split an HTML entity or tag.
    body = html.escape(reply[: TELEGRAM_MESSAGE_LIMIT - 500])
    await message.answer(body + format_sources(sources), parse_mode="HTML")
