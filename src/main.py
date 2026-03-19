import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import get_settings
from src.controllers.health_controller import router as health_router
from src.controllers.ingestion_controller import router as ingestion_router
from src.data.database import close_db, connect_db
from src.services.kafka_producer_service import KafkaProducerService
from src.utils.exceptions import IngestionError
from src.workers.conversation_worker import ConversationWorker
from src.workers.evaluation_worker import EvaluationWorker
from src.workers.feedback_worker import FeedbackWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    await connect_db()
    await KafkaProducerService.start()

    worker_tasks = [
        asyncio.create_task(ConversationWorker().start()),
        asyncio.create_task(FeedbackWorker().start()),
        asyncio.create_task(EvaluationWorker().start()),
    ]
    logger.info("Application startup complete")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)

    await KafkaProducerService.stop()
    await close_db()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(IngestionError)
async def ingestion_error_handler(request: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": exc.message})


app.include_router(health_router, tags=["Health"])
app.include_router(
    ingestion_router,
    prefix=f"{settings.api_v1_prefix}/ingest",
    tags=["Ingestion"],
)
