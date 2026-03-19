import pytest
from unittest.mock import AsyncMock, patch

from src.data.schemas.ingestion import (
    BatchIngestionRequest,
    ConversationIngestionRequest,
    ConversationMetadataSchema,
    FeedbackIngestionRequest,
    Role,
    TurnSchema,
)
from src.services.ingestion_service import IngestionService
from src.utils.exceptions import KafkaPublishError


def make_conv_request(conversation_id: str = "conv-123") -> ConversationIngestionRequest:
    return ConversationIngestionRequest(
        conversation_id=conversation_id,
        agent_version="1.0.0",
        turns=[
            TurnSchema(turn_id=1, role=Role.user, content="Hello", timestamp="2024-01-01T00:00:00Z")
        ],
        metadata=ConversationMetadataSchema(),
    )


@pytest.fixture
def mock_publish():
    with patch(
        "src.services.ingestion_service.KafkaProducerService.publish",
        new_callable=AsyncMock,
    ) as m:
        yield m


class TestIngestConversationService:
    async def test_publishes_to_conversations_topic(self, mock_publish):
        from src.config.settings import get_settings
        settings = get_settings()
        req = make_conv_request()
        await IngestionService.ingest_conversation(req)
        mock_publish.assert_awaited_once_with(
            topic=settings.kafka_topic_conversations,
            value=req.model_dump(),
            key="conv-123",
        )

    async def test_returns_queued_status(self, mock_publish):
        req = make_conv_request()
        resp = await IngestionService.ingest_conversation(req)
        assert resp.conversation_id == "conv-123"
        assert resp.status == "queued"

    async def test_propagates_kafka_error(self, mock_publish):
        mock_publish.side_effect = KafkaPublishError("connection refused")
        with pytest.raises(KafkaPublishError, match="connection refused"):
            await IngestionService.ingest_conversation(make_conv_request())


class TestIngestBatchService:
    async def test_publishes_once_per_conversation(self, mock_publish):
        req = BatchIngestionRequest(
            conversations=[make_conv_request("c1"), make_conv_request("c2"), make_conv_request("c3")]
        )
        await IngestionService.ingest_batch(req)
        assert mock_publish.await_count == 3

    async def test_returns_correct_queued_count(self, mock_publish):
        req = BatchIngestionRequest(
            conversations=[make_conv_request("c1"), make_conv_request("c2")]
        )
        resp = await IngestionService.ingest_batch(req)
        assert resp.queued == 2
        assert resp.status == "queued"

    async def test_empty_batch_returns_zero_queued(self, mock_publish):
        req = BatchIngestionRequest(conversations=[])
        resp = await IngestionService.ingest_batch(req)
        assert resp.queued == 0
        mock_publish.assert_not_awaited()

    async def test_each_conversation_uses_its_own_key(self, mock_publish):
        req = BatchIngestionRequest(
            conversations=[make_conv_request("c1"), make_conv_request("c2")]
        )
        await IngestionService.ingest_batch(req)
        keys = [call.kwargs["key"] for call in mock_publish.call_args_list]
        assert keys == ["c1", "c2"]


class TestIngestFeedbackService:
    async def test_publishes_to_feedback_topic(self, mock_publish):
        from src.config.settings import get_settings
        settings = get_settings()
        req = FeedbackIngestionRequest(conversation_id="conv-123", user_rating=4)
        await IngestionService.ingest_feedback(req)
        mock_publish.assert_awaited_once_with(
            topic=settings.kafka_topic_feedback,
            value=req.model_dump(),
            key="conv-123",
        )

    async def test_returns_queued_status(self, mock_publish):
        req = FeedbackIngestionRequest(conversation_id="conv-123")
        resp = await IngestionService.ingest_feedback(req)
        assert resp.conversation_id == "conv-123"
        assert resp.status == "queued"

    async def test_propagates_kafka_error(self, mock_publish):
        mock_publish.side_effect = KafkaPublishError("timeout")
        req = FeedbackIngestionRequest(conversation_id="conv-123", user_rating=3)
        with pytest.raises(KafkaPublishError, match="timeout"):
            await IngestionService.ingest_feedback(req)
