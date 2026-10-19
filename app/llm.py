import numpy as np
from openai import AsyncOpenAI

from app.config import settings

# Most embedding APIs cap the number of inputs per request.
EMBED_BATCH_SIZE = 16

_chat_client = AsyncOpenAI(base_url=settings.chat_base_url, api_key=settings.chat_api_key)
_embed_client = AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)


async def embed(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        resp = await _embed_client.embeddings.create(
            model=settings.embed_model, input=texts[i : i + EMBED_BATCH_SIZE]
        )
        vectors.extend(item.embedding for item in resp.data)
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


async def chat(messages: list[dict[str, str]]) -> str:
    resp = await _chat_client.chat.completions.create(
        model=settings.chat_model, messages=messages, temperature=0.2
    )
    return resp.choices[0].message.content or ""
