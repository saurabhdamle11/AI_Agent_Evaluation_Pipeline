from src.config.settings import get_settings
from src.data.repositories.evaluation_repository import EvaluationRepository
from src.data.schemas.evaluation import EvaluationResult, TriggerEvaluationResponse
from src.services.kafka_producer_service import KafkaProducerService

settings = get_settings()


class EvaluationApiService:
    @staticmethod
    async def trigger_evaluation(conversation_id: str) -> TriggerEvaluationResponse:
        await KafkaProducerService.publish(
            topic=settings.kafka_topic_evaluations,
            value={"conversation_id": conversation_id},
            key=conversation_id,
        )
        return TriggerEvaluationResponse(conversation_id=conversation_id, status="queued")

    @staticmethod
    async def get_evaluation(
        evaluation_id: str,
        repo: EvaluationRepository,
    ) -> EvaluationResult | None:
        data = await repo.get_by_id(evaluation_id)
        return EvaluationResult(**data) if data else None

    @staticmethod
    async def list_evaluations_for_conversation(
        conversation_id: str,
        repo: EvaluationRepository,
    ) -> list[EvaluationResult]:
        rows = await repo.get_by_conversation_id(conversation_id)
        return [EvaluationResult(**row) for row in rows]
