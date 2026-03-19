from unittest.mock import AsyncMock, MagicMock, patch

from src.workers.conversation_worker import ConversationWorker
from src.workers.feedback_worker import FeedbackWorker

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
            import pytest
            with pytest.raises(RuntimeError, match="db error"):
                await worker.handle(CONV_PAYLOAD)


class TestFeedbackWorker:
    async def test_handle_calls_repo_upsert_with_payload(self):
        mock_repo = AsyncMock()
        with (
            patch("src.workers.feedback_worker.get_db", return_value=MagicMock()),
            patch("src.workers.feedback_worker.FeedbackRepository", return_value=mock_repo),
        ):
            worker = FeedbackWorker()
            await worker.handle(FEEDBACK_PAYLOAD)
        mock_repo.upsert.assert_awaited_once_with(FEEDBACK_PAYLOAD)

    async def test_handle_initialises_repo_with_db(self):
        mock_db = MagicMock()
        mock_repo = AsyncMock()
        with (
            patch("src.workers.feedback_worker.get_db", return_value=mock_db),
            patch("src.workers.feedback_worker.FeedbackRepository") as MockRepo,
        ):
            MockRepo.return_value = mock_repo
            worker = FeedbackWorker()
            await worker.handle(FEEDBACK_PAYLOAD)
        MockRepo.assert_called_once_with(mock_db)

    async def test_handle_exception_propagates(self):
        mock_repo = AsyncMock()
        mock_repo.upsert.side_effect = RuntimeError("write failed")
        with (
            patch("src.workers.feedback_worker.get_db", return_value=MagicMock()),
            patch("src.workers.feedback_worker.FeedbackRepository", return_value=mock_repo),
        ):
            worker = FeedbackWorker()
            import pytest
            with pytest.raises(RuntimeError, match="write failed"):
                await worker.handle(FEEDBACK_PAYLOAD)
