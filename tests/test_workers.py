import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.workers.conversation_worker import ConversationWorker
from src.workers.feedback_worker import FeedbackWorker
from src.workers.processed_feedback_worker import ProcessedFeedbackWorker

CONV_PAYLOAD = {
    "conversation_id": "conv-123",
    "agent_version": "1.0.0",
    "turns": [],
    "metadata": {},
}

FEEDBACK_PAYLOAD = {
    "conversation_id": "conv-123",
    "user_rating": 4,
}


class TestConversationWorker:
    async def test_handle_calls_repo_upsert_with_payload(self):
        mock_repo = AsyncMock()
        with (
            patch("src.workers.conversation_worker.get_db", return_value=MagicMock()),
            patch("src.workers.conversation_worker.ConversationRepository", return_value=mock_repo),
        ):
            worker = ConversationWorker()
            await worker.handle(CONV_PAYLOAD)
        mock_repo.upsert.assert_awaited_once_with(CONV_PAYLOAD)

    async def test_handle_initialises_repo_with_db(self):
        mock_db = MagicMock()
        mock_repo = AsyncMock()
        with (
            patch("src.workers.conversation_worker.get_db", return_value=mock_db),
            patch("src.workers.conversation_worker.ConversationRepository") as MockRepo,
        ):
            MockRepo.return_value = mock_repo
            worker = ConversationWorker()
            await worker.handle(CONV_PAYLOAD)
        MockRepo.assert_called_once_with(mock_db)

    async def test_handle_exception_propagates(self):
        mock_repo = AsyncMock()
        mock_repo.upsert.side_effect = RuntimeError("db error")
        with (
            patch("src.workers.conversation_worker.get_db", return_value=MagicMock()),
            patch("src.workers.conversation_worker.ConversationRepository", return_value=mock_repo),
        ):
            worker = ConversationWorker()
            with pytest.raises(RuntimeError, match="db error"):
                await worker.handle(CONV_PAYLOAD)



class TestFeedbackWorker:
    async def test_handle_calls_repo_upsert_with_payload(self):
        mock_repo = AsyncMock()
        with (
            patch("src.workers.feedback_worker.get_db", return_value=MagicMock()),
            patch("src.workers.feedback_worker.FeedbackRepository", return_value=mock_repo),
            patch("src.workers.feedback_worker.KafkaProducerService.publish", new_callable=AsyncMock),
        ):
            await FeedbackWorker().handle(FEEDBACK_PAYLOAD)
        mock_repo.upsert.assert_awaited_once_with(FEEDBACK_PAYLOAD)

    async def test_handle_initialises_repo_with_db(self):
        mock_db = MagicMock()
        mock_repo = AsyncMock()
        with (
            patch("src.workers.feedback_worker.get_db", return_value=mock_db),
            patch("src.workers.feedback_worker.FeedbackRepository") as MockRepo,
            patch("src.workers.feedback_worker.KafkaProducerService.publish", new_callable=AsyncMock),
        ):
            MockRepo.return_value = mock_repo
            await FeedbackWorker().handle(FEEDBACK_PAYLOAD)
        MockRepo.assert_called_once_with(mock_db)

    async def test_publishes_aggregated_to_processed_topic(self):
        mock_repo = AsyncMock()
        mock_publish = AsyncMock()
        with (
            patch("src.workers.feedback_worker.get_db", return_value=MagicMock()),
            patch("src.workers.feedback_worker.FeedbackRepository", return_value=mock_repo),
            patch("src.workers.feedback_worker.KafkaProducerService.publish", mock_publish),
        ):
            await FeedbackWorker().handle(FEEDBACK_PAYLOAD)
        mock_publish.assert_awaited_once()
        call_kwargs = mock_publish.call_args
        assert call_kwargs.kwargs["key"] == FEEDBACK_PAYLOAD["conversation_id"]

    async def test_handle_exception_propagates(self):
        mock_repo = AsyncMock()
        mock_repo.upsert.side_effect = RuntimeError("write failed")
        with (
            patch("src.workers.feedback_worker.get_db", return_value=MagicMock()),
            patch("src.workers.feedback_worker.FeedbackRepository", return_value=mock_repo),
            patch("src.workers.feedback_worker.KafkaProducerService.publish", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="write failed"):
                await FeedbackWorker().handle(FEEDBACK_PAYLOAD)


class TestProcessedFeedbackWorker:
    AGGREGATED_PAYLOAD = {
        "conversation_id": "conv-123",
        "user_rating": 4,
        "ops_review": None,
        "annotation_consensus": [],
        "overall_kappa": 1.0,
        "routing": "auto_label",
        "aggregated_at": "2026-03-20T00:00:00+00:00",
    }

    async def test_handle_calls_update_aggregated(self):
        mock_repo = AsyncMock()
        with (
            patch("src.workers.processed_feedback_worker.get_db", return_value=MagicMock()),
            patch("src.workers.processed_feedback_worker.FeedbackRepository", return_value=mock_repo),
        ):
            await ProcessedFeedbackWorker().handle(self.AGGREGATED_PAYLOAD)
        mock_repo.update_aggregated.assert_awaited_once_with(
            self.AGGREGATED_PAYLOAD["conversation_id"], self.AGGREGATED_PAYLOAD
        )

    async def test_handle_initialises_repo_with_db(self):
        mock_db = MagicMock()
        mock_repo = AsyncMock()
        with (
            patch("src.workers.processed_feedback_worker.get_db", return_value=mock_db),
            patch("src.workers.processed_feedback_worker.FeedbackRepository") as MockRepo,
        ):
            MockRepo.return_value = mock_repo
            await ProcessedFeedbackWorker().handle(self.AGGREGATED_PAYLOAD)
        MockRepo.assert_called_once_with(mock_db)

    async def test_handle_exception_propagates(self):
        mock_repo = AsyncMock()
        mock_repo.update_aggregated.side_effect = RuntimeError("db error")
        with (
            patch("src.workers.processed_feedback_worker.get_db", return_value=MagicMock()),
            patch("src.workers.processed_feedback_worker.FeedbackRepository", return_value=mock_repo),
        ):
            with pytest.raises(RuntimeError, match="db error"):
                await ProcessedFeedbackWorker().handle(self.AGGREGATED_PAYLOAD)
