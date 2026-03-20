import logging

from src.config.settings import get_settings
from src.data.database import get_db
from src.data.repositories.feedback_repository import FeedbackRepository
from src.services.feedback_aggregation_service import FeedbackAggregationService
from src.services.kafka_producer_service import KafkaProducerService
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

        aggregated = FeedbackAggregationService.aggregate(payload)
        await KafkaProducerService.publish(
            topic=settings.kafka_topic_feedback_processed,
            value=aggregated.model_dump(),
            key=aggregated.conversation_id,
        )
        logger.info(
            f"Published aggregated feedback for: {aggregated.conversation_id} "
            f"(routing={aggregated.routing}, kappa={aggregated.overall_kappa})"
        )
