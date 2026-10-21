from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import Message, Thread
from app.llm import chat

SYSTEM_PROMPT = """You are a helpful assistant for deep, ongoing work.
- Reply in the same language as the user, in plain text (simple lists are fine, no Markdown headings or tables).
- Use the long-term memory below as established context from earlier in this conversation."""

SUMMARY_PROMPT = """You maintain the long-term memory of a conversation.
Merge the existing memory with the new messages into one updated memory.
Keep facts, decisions, user preferences, names, numbers, open questions and the current task. Drop small talk.
Refer to the user as "the user"; never guess anyone's gender from their name.
Write concise bullet points in the conversation's language. Output only the memory."""

TITLE_LENGTH = 60


async def unsummarized(session: AsyncSession, thread: Thread) -> list[Message]:
    return list(
        await session.scalars(
            select(Message)
            .where(Message.thread_id == thread.id, Message.id > thread.summarized_upto)
            .order_by(Message.id)
        )
    )


async def compact(session: AsyncSession, thread: Thread) -> None:
    """Fold older messages into the rolling summary instead of silently dropping them."""
    messages = await unsummarized(session, thread)
    if len(messages) <= settings.memory_window:
        return
    old = messages[: len(messages) - settings.memory_keep_recent]
    transcript = "\n".join(f"{m.role}: {m.content}" for m in old)
    thread.summary = await chat(
        [
            {"role": "system", "content": SUMMARY_PROMPT},
            {
                "role": "user",
                "content": f"Existing memory:\n{thread.summary or '(empty)'}\n\nNew messages:\n{transcript}",
            },
        ]
    )
    thread.summarized_upto = old[-1].id


def build_prompt(thread: Thread, recent: list[Message]) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT
    if thread.summary:
        system += f"\n\nLong-term memory:\n{thread.summary}"
    return [{"role": "system", "content": system}, *({"role": m.role, "content": m.content} for m in recent)]


async def reply(session: AsyncSession, thread: Thread, text: str) -> Message:
    """Store the user's message, answer it with summary + recent context, and commit both.

    Nothing is committed if an LLM call fails, so a failed turn leaves the thread unchanged.
    """
    session.add(Message(thread_id=thread.id, role="user", content=text))
    await session.flush()
    await compact(session, thread)

    answer = await chat(build_prompt(thread, await unsummarized(session, thread)))
    message = Message(thread_id=thread.id, role="assistant", content=answer)
    session.add(message)
    if thread.title == "New chat":
        thread.title = text[:TITLE_LENGTH]
    thread.updated_at = func.now()
    await session.commit()
    await session.refresh(thread)
    await session.refresh(message)
    return message
