from src.config.settings import get_settings
from src.data.schemas.ingestion import (
    BatchIngestionRequest,
    BatchIngestionResponse,
    ConversationIngestionRequest,
    FeedbackIngestionRequest,
    IngestionResponse,
)
from src.services.kafka_producer_service import KafkaProducerService

settings = get_settings()


class IngestionService:
    @staticmethod
    async def ingest_conversation(payload: ConversationIngestionRequest) -> IngestionResponse:
        await KafkaProducerService.publish(
            topic=settings.kafka_topic_conversations,
            value=payload.model_dump(),
            key=payload.conversation_id,
        )
        return IngestionResponse(conversation_id=payload.conversation_id, status="queued")

    @staticmethod
    async def ingest_batch(payload: BatchIngestionRequest) -> BatchIngestionResponse:
        for conv in payload.conversations:
            await KafkaProducerService.publish(
                topic=settings.kafka_topic_conversations,
                value=conv.model_dump(),
                key=conv.conversation_id,
            )
        return BatchIngestionResponse(queued=len(payload.conversations), status="queued")

    @staticmethod
    async def ingest_feedback(payload: FeedbackIngestionRequest) -> IngestionResponse:
        await KafkaProducerService.publish(
            topic=settings.kafka_topic_feedback,
            value=payload.model_dump(),
            key=payload.conversation_id,
        )
        return IngestionResponse(conversation_id=payload.conversation_id, status="queued")
