import logging

from src.config.settings import get_settings
from src.data.database import get_db
from src.data.repositories.feedback_repository import FeedbackRepository
from src.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)
settings = get_settings()


class ProcessedFeedbackWorker(BaseWorker):
    """
    Consumes aggregated feedback from the 'feedback.processed' topic and
    stores the aggregation result back onto the raw feedback document in MongoDB.

    This keeps the raw feedback document as the single source of truth while
    enriching it with computed agreement metrics and routing decisions.
    """

    def __init__(self) -> None:
        super().__init__(topics=[settings.kafka_topic_feedback_processed])

    async def handle(self, payload: dict) -> None:
        db = get_db()
        repo = FeedbackRepository(db)
        conversation_id = payload["conversation_id"]
        await repo.update_aggregated(conversation_id, payload)
        logger.info(
            f"Stored aggregated feedback for: {conversation_id} "
            f"(routing={payload.get('routing')})"
        )
