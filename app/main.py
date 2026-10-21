import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from aiogram import Bot, Dispatcher
from fastapi import FastAPI
from loguru import logger

from app import api, web
from app.bot import router as bot_router
from app.config import settings
from app.db import init_db
from app.rag import index

logger.add("logs/docubot.log", rotation="1 day", retention="7 days", encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    await index.reload()

    bot: Bot | None = None
    polling: asyncio.Task | None = None
    if settings.telegram_bot_token:
        bot = Bot(settings.telegram_bot_token)
        dp = Dispatcher()
        dp.include_router(bot_router)
        # handle_signals=False: let uvicorn own SIGINT/SIGTERM.
        polling = asyncio.create_task(dp.start_polling(bot, handle_signals=False))
        logger.info("Telegram bot polling started")
    else:
        logger.warning("TELEGRAM_BOT_TOKEN not set, running REST API only")

    yield

    if polling:
        polling.cancel()
        with suppress(asyncio.CancelledError):
            await polling
    if bot:
        await bot.session.close()


app = FastAPI(title="DocuBot", version="0.1.0", lifespan=lifespan)
app.include_router(api.public)
app.include_router(api.router, prefix="/api")
app.include_router(web.page)
app.include_router(web.router)
