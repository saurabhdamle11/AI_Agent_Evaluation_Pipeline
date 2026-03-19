import logging

from src.config.settings import get_settings
from src.data.database import get_db
from src.data.repositories.conversation_repository import ConversationRepository
from src.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationWorker(BaseWorker):
    def __init__(self) -> None:
        super().__init__(topics=[settings.kafka_topic_conversations])

    async def handle(self, payload: dict) -> None:
        db = get_db()
        repo = ConversationRepository(db)
        await repo.upsert(payload)
        logger.info(f"Persisted conversation: {payload.get('conversation_id')}")
