import json

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import Message, Thread
from app.llm import chat
from app.rag import Source, build_context, cited_labels, retrieve

SYSTEM_PROMPT = """You are a document assistant for long, ongoing conversations.
- Answer questions about products, policies and facts using ONLY the numbered context blocks in the latest message, and cite the blocks you used inline, like [1] or [2].
- If the context does not contain the answer, politely say you couldn't find that in the documents, without citing anything or mentioning "context" or "blocks". Never make things up.
- What the user told you earlier (their name, plans, preferences) is known from the conversation and the long-term memory below; use it freely, without citations.
- For greetings or small talk, reply briefly and offer to help; don't volunteer document content nobody asked for.
- Reply in the same language as the user, in plain text (simple lists are fine, no Markdown headings or tables)."""

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


def build_prompt(thread: Thread, recent: list[Message], sources: list[Source]) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT
    if thread.summary:
        system += f"\n\nLong-term memory:\n{thread.summary}"
    messages = [{"role": "system", "content": system}, *({"role": m.role, "content": m.content} for m in recent)]
    # Retrieved context goes only into the prompt; the stored message stays the user's own words.
    messages[-1]["content"] = f"Context:\n{build_context(sources)}\n\nQuestion: {messages[-1]['content']}"
    return messages


async def reply(session: AsyncSession, thread: Thread, text: str) -> Message:
    """Store the user's message, answer it from the documents with summary + recent context, and commit both.

    Nothing is committed if an LLM call fails, so a failed turn leaves the thread unchanged.
    """
    session.add(Message(thread_id=thread.id, role="user", content=text))
    await session.flush()
    await compact(session, thread)

    sources = await retrieve(text)
    answer = await chat(build_prompt(thread, await unsummarized(session, thread), sources))
    message = Message(
        thread_id=thread.id,
        role="assistant",
        content=answer,
        sources=json.dumps(cited_labels(answer, sources), ensure_ascii=False),
    )
    session.add(message)
    if thread.title == "New chat":
        thread.title = text[:TITLE_LENGTH]
    thread.updated_at = func.now()
    await session.commit()
    await session.refresh(thread)
    await session.refresh(message)
    return message
