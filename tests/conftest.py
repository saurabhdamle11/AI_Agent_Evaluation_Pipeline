import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture
async def client():
    with (
        patch("src.data.database.connect_db", new_callable=AsyncMock),
        patch("src.data.database.close_db", new_callable=AsyncMock),
        patch("src.services.kafka_producer_service.KafkaProducerService.start", new_callable=AsyncMock),
        patch("src.services.kafka_producer_service.KafkaProducerService.stop", new_callable=AsyncMock),
        patch("src.workers.conversation_worker.ConversationWorker.start", new_callable=AsyncMock),
        patch("src.workers.feedback_worker.FeedbackWorker.start", new_callable=AsyncMock),
    ):
        from src.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
