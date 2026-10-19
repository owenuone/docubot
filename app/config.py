from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Telegram bot is optional so the REST API can run on its own (e.g. for testing).
    telegram_bot_token: str | None = None
    admin_ids: list[int] = Field(default_factory=list)
    admin_api_token: str = Field(min_length=1)

    # Any OpenAI-compatible provider works: DeepSeek, OpenAI, Qwen, GLM, Kimi...
    chat_base_url: str = "https://api.deepseek.com"
    chat_api_key: str = Field(min_length=1)
    chat_model: str = "deepseek-chat"

    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str = Field(min_length=1)
    embed_model: str = "BAAI/bge-m3"

    database_url: str = "sqlite+aiosqlite:///data/docubot.db"
    upload_dir: Path = Path("data/uploads")

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    history_turns: int = 4
    rate_limit_per_minute: int = 10


settings = Settings()
