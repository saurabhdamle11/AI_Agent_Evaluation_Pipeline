import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from src.data.schemas.suggestion import Suggestion, SuggestionStatus, SuggestionType
from src.services.suggestion_service import SuggestionService

SUGGESTION_ID = "sug-001"
CONVERSATION_ID = "conv-001"


def make_suggestion_dict(
    suggestion_id=SUGGESTION_ID,
    status="pending",
    type="prompt",
    agent_version="1.0.0",
) -> dict:
    return Suggestion(
        suggestion_id=suggestion_id,
        type=type,
        conversation_ids=[CONVERSATION_ID],
        agent_version=agent_version,
        suggestion="Use more concise prompts.",
        rationale="Long prompts lead to longer latency.",
        confidence=0.85,
        expected_impact="10% latency reduction",
        status=status,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    ).model_dump()


@pytest.fixture
def repo():
    r = MagicMock()
    r.list_all = AsyncMock()
    r.get_by_id = AsyncMock()
    r.list_by_conversation_id = AsyncMock()
    r.update_status = AsyncMock()
    return r


class TestListSuggestions:
    async def test_returns_list_of_suggestions(self, repo):
        repo.list_all.return_value = [make_suggestion_dict(), make_suggestion_dict("sug-002")]
        results = await SuggestionService.list_suggestions(repo)
        assert len(results) == 2
        assert all(isinstance(r, Suggestion) for r in results)

    async def test_returns_empty_list_when_none(self, repo):
        repo.list_all.return_value = []
        results = await SuggestionService.list_suggestions(repo)
        assert results == []

    async def test_passes_status_filter_to_repo(self, repo):
        repo.list_all.return_value = []
        await SuggestionService.list_suggestions(repo, status=SuggestionStatus.applied)
        repo.list_all.assert_awaited_once_with(status="applied", type=None, agent_version=None)

    async def test_passes_type_filter_to_repo(self, repo):
        repo.list_all.return_value = []
        await SuggestionService.list_suggestions(repo, type=SuggestionType.tool)
        repo.list_all.assert_awaited_once_with(status=None, type="tool", agent_version=None)

    async def test_passes_agent_version_filter_to_repo(self, repo):
        repo.list_all.return_value = []
        await SuggestionService.list_suggestions(repo, agent_version="2.0.0")
        repo.list_all.assert_awaited_once_with(status=None, type=None, agent_version="2.0.0")

    async def test_no_filters_passes_all_none(self, repo):
        repo.list_all.return_value = []
        await SuggestionService.list_suggestions(repo)
        repo.list_all.assert_awaited_once_with(status=None, type=None, agent_version=None)


class TestGetSuggestion:
    async def test_returns_suggestion_when_found(self, repo):
        repo.get_by_id.return_value = make_suggestion_dict()
        result = await SuggestionService.get_suggestion(SUGGESTION_ID, repo)
        assert result is not None
        assert result.suggestion_id == SUGGESTION_ID

    async def test_returns_none_when_not_found(self, repo):
        repo.get_by_id.return_value = None
        result = await SuggestionService.get_suggestion(SUGGESTION_ID, repo)
        assert result is None

    async def test_queries_repo_with_correct_id(self, repo):
        repo.get_by_id.return_value = None
        await SuggestionService.get_suggestion(SUGGESTION_ID, repo)
        repo.get_by_id.assert_awaited_once_with(SUGGESTION_ID)

    async def test_result_is_suggestion_type(self, repo):
        repo.get_by_id.return_value = make_suggestion_dict()
        result = await SuggestionService.get_suggestion(SUGGESTION_ID, repo)
        assert isinstance(result, Suggestion)


class TestListSuggestionsForConversation:
    async def test_returns_suggestions_for_conversation(self, repo):
        repo.list_by_conversation_id.return_value = [make_suggestion_dict(), make_suggestion_dict("sug-002")]
        results = await SuggestionService.list_suggestions_for_conversation(CONVERSATION_ID, repo)
        assert len(results) == 2

    async def test_returns_empty_list_when_none(self, repo):
        repo.list_by_conversation_id.return_value = []
        results = await SuggestionService.list_suggestions_for_conversation(CONVERSATION_ID, repo)
        assert results == []

    async def test_queries_repo_with_correct_conversation_id(self, repo):
        repo.list_by_conversation_id.return_value = []
        await SuggestionService.list_suggestions_for_conversation(CONVERSATION_ID, repo)
        repo.list_by_conversation_id.assert_awaited_once_with(CONVERSATION_ID)


class TestUpdateStatus:
    async def test_returns_updated_suggestion(self, repo):
        updated = make_suggestion_dict(status="applied")
        repo.update_status.return_value = updated
        result = await SuggestionService.update_status(SUGGESTION_ID, SuggestionStatus.applied, repo)
        assert result is not None
        assert result.status == SuggestionStatus.applied

    async def test_returns_none_when_suggestion_not_found(self, repo):
        repo.update_status.return_value = None
        result = await SuggestionService.update_status(SUGGESTION_ID, SuggestionStatus.rejected, repo)
        assert result is None

    async def test_passes_correct_status_value_to_repo(self, repo):
        repo.update_status.return_value = make_suggestion_dict(status="rejected")
        await SuggestionService.update_status(SUGGESTION_ID, SuggestionStatus.rejected, repo)
        repo.update_status.assert_awaited_once_with(SUGGESTION_ID, "rejected")
