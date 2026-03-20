import logging

from src.config.settings import get_settings
from src.data.database import get_db
from src.data.repositories.conversation_repository import ConversationRepository
from src.data.repositories.evaluation_repository import EvaluationRepository
from src.data.repositories.suggestion_repository import SuggestionRepository
from src.services.evaluation_service import EvaluationService
from src.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)
settings = get_settings()


class EvaluationWorker(BaseWorker):
    def __init__(self) -> None:
        super().__init__(topics=[settings.kafka_topic_evaluations])

    async def handle(self, payload: dict) -> None:
        conversation_id = payload.get("conversation_id")
        if not conversation_id:
            logger.warning("EvaluationWorker received message without conversation_id — skipping")
            return

        db = get_db()
        service = EvaluationService(
            conversation_repo=ConversationRepository(db),
            evaluation_repo=EvaluationRepository(db),
            suggestion_repo=SuggestionRepository(db),
        )
        result = await service.evaluate(conversation_id)
        if result:
            logger.info(f"Persisted evaluation {result.evaluation_id} for conversation {conversation_id}")
