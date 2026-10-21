import uuid

import pytest
from fastapi.testclient import TestClient

from app import memory
from app.config import settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def visitor() -> dict[str, str]:
    return {"X-Visitor-Id": str(uuid.uuid4())}


def test_chat_page_is_served(client: TestClient) -> None:
    resp = client.get("/chat")
    assert resp.status_code == 200
    assert "DocuBot Chat" in resp.text


def test_visitor_id_is_required(client: TestClient) -> None:
    assert client.get("/chat/api/threads").status_code == 422
    assert client.get("/chat/api/threads", headers={"X-Visitor-Id": "not-a-uuid"}).status_code == 422


def test_thread_is_persisted_and_titled(client: TestClient) -> None:
    headers = visitor()
    thread = client.post("/chat/api/threads", headers=headers).json()

    resp = client.post(
        f"/chat/api/threads/{thread['id']}/messages", json={"content": "Plan my novel"}, headers=headers
    )
    assert resp.status_code == 200
    assert resp.json()["message"]["role"] == "assistant"

    detail = client.get(f"/chat/api/threads/{thread['id']}", headers=headers).json()
    assert detail["title"] == "Plan my novel"
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


def test_visitors_cannot_see_each_others_threads(client: TestClient) -> None:
    owner, stranger = visitor(), visitor()
    thread = client.post("/chat/api/threads", headers=owner).json()

    assert client.get("/chat/api/threads", headers=stranger).json() == []
    assert client.get(f"/chat/api/threads/{thread['id']}", headers=stranger).status_code == 404
    assert client.delete(f"/chat/api/threads/{thread['id']}", headers=stranger).status_code == 404


def test_long_thread_is_folded_into_memory(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[list[dict[str, str]]] = []

    async def recording_chat(messages: list[dict[str, str]]) -> str:
        prompts.append(messages)
        if "long-term memory of a conversation" in messages[0]["content"]:
            return "- User's name is Ada"
        return "ok"

    monkeypatch.setattr(memory, "chat", recording_chat)
    monkeypatch.setattr(settings, "memory_window", 4)
    monkeypatch.setattr(settings, "memory_keep_recent", 2)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 100)

    headers = visitor()
    thread_id = client.post("/chat/api/threads", headers=headers).json()["id"]
    for text in ["My name is Ada", "I like tea", "Remember that"]:
        resp = client.post(f"/chat/api/threads/{thread_id}/messages", json={"content": text}, headers=headers)

    body = resp.json()
    assert body["summary"] == "- User's name is Ada"
    assert body["summarized_upto"] > 0

    last_prompt = prompts[-1]
    assert "User's name is Ada" in last_prompt[0]["content"]
    # Only the most recent messages are sent verbatim; the rest live in the summary.
    assert len(last_prompt) - 1 <= settings.memory_keep_recent
    assert last_prompt[-1] == {"role": "user", "content": "Remember that"}


def test_failed_llm_call_leaves_thread_unchanged(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import APIConnectionError

    async def failing_chat(messages: list[dict[str, str]]) -> str:
        raise APIConnectionError(request=None)

    monkeypatch.setattr(memory, "chat", failing_chat)
    headers = visitor()
    thread_id = client.post("/chat/api/threads", headers=headers).json()["id"]

    resp = client.post(f"/chat/api/threads/{thread_id}/messages", json={"content": "hi"}, headers=headers)
    assert resp.status_code == 502
    assert client.get(f"/chat/api/threads/{thread_id}", headers=headers).json()["messages"] == []
