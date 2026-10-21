import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Settings are read at import time, so the environment must be ready before importing app.*
_tmp = Path(tempfile.mkdtemp(prefix="docubot-test-"))
os.environ.update(
    TELEGRAM_BOT_TOKEN="",
    ADMIN_API_TOKEN="test-token",
    CHAT_API_KEY="dummy",
    EMBED_API_KEY="dummy",
    DATABASE_URL=f"sqlite+aiosqlite:///{_tmp / 'test.db'}",
    UPLOAD_DIR=str(_tmp / "uploads"),
)


async def fake_embed(texts: list[str]) -> np.ndarray:
    """Bag-of-words hashing embedding: offline, deterministic within a test run."""
    matrix = np.zeros((len(texts), 256), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().split():
            matrix[row, hash(word.strip(".,?!")) % 256] += 1
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


async def fake_chat(messages: list[dict[str, str]]) -> str:
    return "Fake answer [1]"


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ingest
    import app.memory
    import app.rag

    monkeypatch.setattr(app.ingest, "embed", fake_embed)
    monkeypatch.setattr(app.rag, "embed", fake_embed)
    monkeypatch.setattr(app.rag, "chat", fake_chat)
    monkeypatch.setattr(app.memory, "chat", fake_chat)
