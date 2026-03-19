import pytest
from unittest.mock import AsyncMock, patch

from src.utils.exceptions import KafkaPublishError

CONV_PAYLOAD = {
    "conversation_id": "conv-123",
    "agent_version": "1.0.0",
    "turns": [
        {
            "turn_id": 1,
            "role": "user",
            "content": "Hello",
            "timestamp": "2024-01-01T00:00:00Z",
        }
    ],
    "metadata": {},
}

FEEDBACK_PAYLOAD = {
    "conversation_id": "conv-123",
    "user_rating": 4,
}


@pytest.fixture
def mock_publish():
    with patch(
        "src.services.kafka_producer_service.KafkaProducerService.publish",
        new_callable=AsyncMock,
    ) as m:
        yield m


class TestIngestConversation:
    async def test_returns_202_and_queued_status(self, client, mock_publish):
        resp = await client.post("/api/v1/ingest/conversation", json=CONV_PAYLOAD)
        assert resp.status_code == 202
        assert resp.json() == {"conversation_id": "conv-123", "status": "queued"}

    async def test_publishes_to_kafka_with_conversation_id_as_key(self, client, mock_publish):
        await client.post("/api/v1/ingest/conversation", json=CONV_PAYLOAD)
        mock_publish.assert_awaited_once()
        assert mock_publish.call_args.kwargs["key"] == "conv-123"

    async def test_publishes_to_conversations_topic(self, client, mock_publish):
        from src.config.settings import get_settings
        settings = get_settings()
        await client.post("/api/v1/ingest/conversation", json=CONV_PAYLOAD)
        assert mock_publish.call_args.kwargs["topic"] == settings.kafka_topic_conversations

    async def test_with_tool_calls(self, client, mock_publish):
        payload = {
            **CONV_PAYLOAD,
            "turns": [
                {
                    "turn_id": 1,
                    "role": "assistant",
                    "content": "Using search tool",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "tool_calls": [
                        {
                            "tool_name": "search",
                            "parameters": {"query": "foo"},
                            "result": {"hits": 5},
                            "latency_ms": 200,
                        }
                    ],
                }
            ],
        }
        resp = await client.post("/api/v1/ingest/conversation", json=payload)
        assert resp.status_code == 202

    async def test_missing_required_fields_returns_422(self, client, mock_publish):
        resp = await client.post("/api/v1/ingest/conversation", json={"conversation_id": "x"})
        assert resp.status_code == 422

    async def test_invalid_role_returns_422(self, client, mock_publish):
        payload = {
            **CONV_PAYLOAD,
            "turns": [{**CONV_PAYLOAD["turns"][0], "role": "robot"}],
        }
        resp = await client.post("/api/v1/ingest/conversation", json=payload)
        assert resp.status_code == 422

    async def test_kafka_failure_returns_500(self, client):
        with patch(
            "src.services.kafka_producer_service.KafkaProducerService.publish",
            new_callable=AsyncMock,
            side_effect=KafkaPublishError("broker unavailable"),
        ):
            resp = await client.post("/api/v1/ingest/conversation", json=CONV_PAYLOAD)
        assert resp.status_code == 500
        assert "broker unavailable" in resp.json()["detail"]


class TestIngestBatch:
    async def test_returns_202_with_queued_count(self, client, mock_publish):
        payload = {
            "conversations": [CONV_PAYLOAD, {**CONV_PAYLOAD, "conversation_id": "conv-456"}]
        }
        resp = await client.post("/api/v1/ingest/conversations/batch", json=payload)
        assert resp.status_code == 202
        body = resp.json()
        assert body["queued"] == 2
        assert body["status"] == "queued"

    async def test_publishes_once_per_conversation(self, client, mock_publish):
        payload = {
            "conversations": [CONV_PAYLOAD, {**CONV_PAYLOAD, "conversation_id": "conv-456"}]
        }
        await client.post("/api/v1/ingest/conversations/batch", json=payload)
        assert mock_publish.await_count == 2

    async def test_empty_batch_returns_202(self, client, mock_publish):
        resp = await client.post("/api/v1/ingest/conversations/batch", json={"conversations": []})
        assert resp.status_code == 202
        assert resp.json()["queued"] == 0
        mock_publish.assert_not_awaited()


class TestIngestFeedback:
    async def test_returns_202_and_queued_status(self, client, mock_publish):
        resp = await client.post("/api/v1/ingest/feedback", json=FEEDBACK_PAYLOAD)
        assert resp.status_code == 202
        assert resp.json() == {"conversation_id": "conv-123", "status": "queued"}

    async def test_publishes_to_feedback_topic(self, client, mock_publish):
        from src.config.settings import get_settings
        settings = get_settings()
        await client.post("/api/v1/ingest/feedback", json=FEEDBACK_PAYLOAD)
        mock_publish.assert_awaited_once()
        assert mock_publish.call_args.kwargs["topic"] == settings.kafka_topic_feedback

    async def test_with_all_optional_fields(self, client, mock_publish):
        payload = {
            "conversation_id": "conv-123",
            "user_rating": 5,
            "ops_review": {"quality": "excellent", "notes": "Very helpful"},
            "annotations": [
                {
                    "type": "tone",
                    "label": "positive",
                    "annotator_id": "ann-1",
                    "confidence": 0.95,
                }
            ],
        }
        resp = await client.post("/api/v1/ingest/feedback", json=payload)
        assert resp.status_code == 202

    async def test_invalid_rating_too_high_returns_422(self, client, mock_publish):
        resp = await client.post(
            "/api/v1/ingest/feedback",
            json={"conversation_id": "conv-123", "user_rating": 6},
        )
        assert resp.status_code == 422

    async def test_invalid_rating_too_low_returns_422(self, client, mock_publish):
        resp = await client.post(
            "/api/v1/ingest/feedback",
            json={"conversation_id": "conv-123", "user_rating": 0},
        )
        assert resp.status_code == 422

    async def test_invalid_confidence_out_of_range_returns_422(self, client, mock_publish):
        payload = {
            "conversation_id": "conv-123",
            "annotations": [
                {"type": "t", "label": "l", "annotator_id": "a", "confidence": 1.5}
            ],
        }
        resp = await client.post("/api/v1/ingest/feedback", json=payload)
        assert resp.status_code == 422

    async def test_invalid_ops_review_quality_returns_422(self, client, mock_publish):
        payload = {
            "conversation_id": "conv-123",
            "ops_review": {"quality": "amazing"},
        }
        resp = await client.post("/api/v1/ingest/feedback", json=payload)
        assert resp.status_code == 422
