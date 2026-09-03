import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from configs.database import close_pool, open_pool
from configs.settings import settings
from src.api.routes import router
from src.ingestion.bootstrap import run_startup_bootstrap

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("rpids")

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_standard])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        open_pool()
        logger.info("database pool opened")
    except Exception as exc:
        logger.error("database pool could not be opened: %s", exc)

    await asyncio.to_thread(run_startup_bootstrap)

    yield

    close_pool()
    logger.info("database pool closed")


app = FastAPI(
    title="Retail Policy Intelligence & Decision Support System",
    version="4.0.0",
    description="Agentic compliance question answering over retail policy documents and operational records",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "rate limit exceeded", "limit": str(exc.detail)},
    )


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
