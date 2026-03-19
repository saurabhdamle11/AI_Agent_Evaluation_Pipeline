from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo import ReturnDocument

from src.data.repositories.conversation_repository import ConversationRepository
from src.data.repositories.feedback_repository import FeedbackRepository


def make_mock_db(collection_mock: AsyncMock) -> MagicMock:
    """Return a mock Motor database whose __getitem__ returns collection_mock."""
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection_mock)
    return db


def make_collection() -> AsyncMock:
    col = AsyncMock()
    col.find_one_and_replace = AsyncMock(return_value={})
    col.find_one = AsyncMock(return_value=None)
    return col


class TestConversationRepository:
    async def test_upsert_filters_by_conversation_id(self):
        col = make_collection()
        repo = ConversationRepository(make_mock_db(col))
        await repo.upsert({"conversation_id": "conv-123", "agent_version": "1.0"})
        filter_arg = col.find_one_and_replace.call_args.args[0]
        assert filter_arg == {"conversation_id": "conv-123"}

    async def test_upsert_sets_upsert_true(self):
        col = make_collection()
        repo = ConversationRepository(make_mock_db(col))
        await repo.upsert({"conversation_id": "conv-123"})
        assert col.find_one_and_replace.call_args.kwargs["upsert"] is True

    async def test_upsert_returns_after_document(self):
        col = make_collection()
        repo = ConversationRepository(make_mock_db(col))
        await repo.upsert({"conversation_id": "conv-123"})
        assert col.find_one_and_replace.call_args.kwargs["return_document"] == ReturnDocument.AFTER

    async def test_upsert_adds_ingested_at_timestamp(self):
        col = make_collection()
        repo = ConversationRepository(make_mock_db(col))
        data = {"conversation_id": "conv-123"}
        await repo.upsert(data)
        replacement = col.find_one_and_replace.call_args.args[1]
        assert "ingested_at" in replacement
        assert isinstance(replacement["ingested_at"], datetime)

    async def test_upsert_returns_result_from_mongo(self):
        col = make_collection()
        col.find_one_and_replace = AsyncMock(return_value={"conversation_id": "conv-123", "ingested_at": "ts"})
        repo = ConversationRepository(make_mock_db(col))
        result = await repo.upsert({"conversation_id": "conv-123"})
        assert result["conversation_id"] == "conv-123"

    async def test_get_by_id_queries_correct_filter(self):
        col = make_collection()
        col.find_one = AsyncMock(return_value={"conversation_id": "conv-123"})
        repo = ConversationRepository(make_mock_db(col))
        await repo.get_by_id("conv-123")
        col.find_one.assert_awaited_once_with({"conversation_id": "conv-123"}, {"_id": 0})

    async def test_get_by_id_returns_document(self):
        col = make_collection()
        col.find_one = AsyncMock(return_value={"conversation_id": "conv-123", "agent_version": "2.0"})
        repo = ConversationRepository(make_mock_db(col))
        result = await repo.get_by_id("conv-123")
        assert result["agent_version"] == "2.0"

    async def test_get_by_id_returns_none_when_not_found(self):
        col = make_collection()
        col.find_one = AsyncMock(return_value=None)
        repo = ConversationRepository(make_mock_db(col))
        result = await repo.get_by_id("nonexistent")
        assert result is None


class TestFeedbackRepository:
    async def test_upsert_filters_by_conversation_id(self):
        col = make_collection()
        repo = FeedbackRepository(make_mock_db(col))
        await repo.upsert({"conversation_id": "conv-456", "user_rating": 3})
        filter_arg = col.find_one_and_replace.call_args.args[0]
        assert filter_arg == {"conversation_id": "conv-456"}

    async def test_upsert_adds_updated_at(self):
        col = make_collection()
        repo = FeedbackRepository(make_mock_db(col))
        data = {"conversation_id": "conv-123"}
        await repo.upsert(data)
        replacement = col.find_one_and_replace.call_args.args[1]
        assert "updated_at" in replacement
        assert isinstance(replacement["updated_at"], datetime)

    async def test_upsert_sets_created_at_when_absent(self):
        col = make_collection()
        repo = FeedbackRepository(make_mock_db(col))
        data = {"conversation_id": "conv-123"}
        await repo.upsert(data)
        replacement = col.find_one_and_replace.call_args.args[1]
        assert "created_at" in replacement

    async def test_upsert_does_not_overwrite_existing_created_at(self):
        col = make_collection()
        repo = FeedbackRepository(make_mock_db(col))
        original_created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        data = {"conversation_id": "conv-123", "created_at": original_created_at}
        await repo.upsert(data)
        replacement = col.find_one_and_replace.call_args.args[1]
        assert replacement["created_at"] == original_created_at

    async def test_upsert_updated_at_is_newer_than_existing_created_at(self):
        col = make_collection()
        repo = FeedbackRepository(make_mock_db(col))
        old_created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        data = {"conversation_id": "conv-123", "created_at": old_created_at}
        await repo.upsert(data)
        replacement = col.find_one_and_replace.call_args.args[1]
        assert replacement["updated_at"] >= old_created_at

    async def test_upsert_sets_upsert_true(self):
        col = make_collection()
        repo = FeedbackRepository(make_mock_db(col))
        await repo.upsert({"conversation_id": "conv-123"})
        assert col.find_one_and_replace.call_args.kwargs["upsert"] is True

    async def test_get_by_conversation_id_queries_correct_filter(self):
        col = make_collection()
        col.find_one = AsyncMock(return_value={"conversation_id": "conv-123", "user_rating": 5})
        repo = FeedbackRepository(make_mock_db(col))
        await repo.get_by_conversation_id("conv-123")
        col.find_one.assert_awaited_once_with({"conversation_id": "conv-123"}, {"_id": 0})

    async def test_get_by_conversation_id_returns_document(self):
        col = make_collection()
        col.find_one = AsyncMock(return_value={"conversation_id": "conv-123", "user_rating": 5})
        repo = FeedbackRepository(make_mock_db(col))
        result = await repo.get_by_conversation_id("conv-123")
        assert result["user_rating"] == 5

    async def test_get_by_conversation_id_returns_none_when_not_found(self):
        col = make_collection()
        col.find_one = AsyncMock(return_value=None)
        repo = FeedbackRepository(make_mock_db(col))
        result = await repo.get_by_conversation_id("missing")
        assert result is None
