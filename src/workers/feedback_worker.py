import logging

from src.config.settings import get_settings
from src.data.database import get_db
from src.data.repositories.feedback_repository import FeedbackRepository
from src.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)
settings = get_settings()


class FeedbackWorker(BaseWorker):
    def __init__(self) -> None:
        super().__init__(topics=[settings.kafka_topic_feedback])

    async def handle(self, payload: dict) -> None:
        db = get_db()
        repo = FeedbackRepository(db)
        await repo.upsert(payload)
        logger.info(f"Persisted feedback for: {payload.get('conversation_id')}")
