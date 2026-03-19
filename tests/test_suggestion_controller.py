import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from src.data.schemas.suggestion import Suggestion, SuggestionStatus, SuggestionType
from src.utils.dependencies import get_suggestion_repo

SUGGESTION_ID = "sug-abc"
CONVERSATION_ID = "conv-xyz"


def make_suggestion(
    suggestion_id=SUGGESTION_ID,
    status="pending",
    type="prompt",
) -> Suggestion:
    return Suggestion(
        suggestion_id=suggestion_id,
        type=type,
        conversation_ids=[CONVERSATION_ID],
        agent_version="1.0.0",
        suggestion="Shorten the system prompt.",
        rationale="Reduces token usage and latency.",
        confidence=0.9,
        expected_impact="Latency -15%",
        status=status,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_suggestion_repo():
    repo = MagicMock()
    repo.list_all = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.list_by_conversation_id = AsyncMock()
    repo.update_status = AsyncMock()
    return repo


@pytest.fixture
def client_with_repo(client, mock_suggestion_repo):
    from src.main import app
    app.dependency_overrides[get_suggestion_repo] = lambda: mock_suggestion_repo
    yield client, mock_suggestion_repo
    app.dependency_overrides.pop(get_suggestion_repo, None)


class TestListSuggestions:
    async def test_returns_200_with_list(self, client_with_repo):
        client, repo = client_with_repo
        repo.list_all.return_value = [make_suggestion().model_dump(), make_suggestion("sug-002").model_dump()]
        resp = await client.get("/api/v1/suggestions")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_returns_empty_list(self, client_with_repo):
        client, repo = client_with_repo
        repo.list_all.return_value = []
        resp = await client.get("/api/v1/suggestions")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_status_filter_is_passed_to_repo(self, client_with_repo):
        client, repo = client_with_repo
        repo.list_all.return_value = []
        await client.get("/api/v1/suggestions?status=applied")
        repo.list_all.assert_awaited_once_with(status="applied", type=None, agent_version=None)

    async def test_type_filter_is_passed_to_repo(self, client_with_repo):
        client, repo = client_with_repo
        repo.list_all.return_value = []
        await client.get("/api/v1/suggestions?type=tool")
        repo.list_all.assert_awaited_once_with(status=None, type="tool", agent_version=None)

    async def test_agent_version_filter_is_passed_to_repo(self, client_with_repo):
        client, repo = client_with_repo
        repo.list_all.return_value = []
        await client.get("/api/v1/suggestions?agent_version=2.0.0")
        repo.list_all.assert_awaited_once_with(status=None, type=None, agent_version="2.0.0")

    async def test_invalid_status_returns_422(self, client_with_repo):
        client, _ = client_with_repo
        resp = await client.get("/api/v1/suggestions?status=unknown")
        assert resp.status_code == 422


class TestGetSuggestion:
    async def test_returns_200_with_suggestion(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_id.return_value = make_suggestion().model_dump()
        resp = await client.get(f"/api/v1/suggestions/{SUGGESTION_ID}")
        assert resp.status_code == 200
        assert resp.json()["suggestion_id"] == SUGGESTION_ID

    async def test_returns_404_when_not_found(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_id.return_value = None
        resp = await client.get(f"/api/v1/suggestions/{SUGGESTION_ID}")
        assert resp.status_code == 404

    async def test_queries_repo_with_correct_id(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_id.return_value = make_suggestion().model_dump()
        await client.get(f"/api/v1/suggestions/{SUGGESTION_ID}")
        repo.get_by_id.assert_awaited_once_with(SUGGESTION_ID)

    async def test_response_includes_all_fields(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_id.return_value = make_suggestion().model_dump()
        body = (await client.get(f"/api/v1/suggestions/{SUGGESTION_ID}")).json()
        for field in ("suggestion_id", "type", "suggestion", "rationale", "confidence", "status"):
            assert field in body


class TestListSuggestionsForConversation:
    async def test_returns_200_with_list(self, client_with_repo):
        client, repo = client_with_repo
        repo.list_by_conversation_id.return_value = [make_suggestion().model_dump()]
        resp = await client.get(f"/api/v1/suggestions/conversation/{CONVERSATION_ID}")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_returns_empty_list_when_none(self, client_with_repo):
        client, repo = client_with_repo
        repo.list_by_conversation_id.return_value = []
        resp = await client.get(f"/api/v1/suggestions/conversation/{CONVERSATION_ID}")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_queries_repo_with_correct_conversation_id(self, client_with_repo):
        client, repo = client_with_repo
        repo.list_by_conversation_id.return_value = []
        await client.get(f"/api/v1/suggestions/conversation/{CONVERSATION_ID}")
        repo.list_by_conversation_id.assert_awaited_once_with(CONVERSATION_ID)


class TestUpdateSuggestionStatus:
    async def test_returns_200_with_updated_suggestion(self, client_with_repo):
        client, repo = client_with_repo
        repo.update_status.return_value = make_suggestion(status="applied").model_dump()
        resp = await client.patch(
            f"/api/v1/suggestions/{SUGGESTION_ID}/status",
            json={"status": "applied"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "applied"

    async def test_returns_404_when_not_found(self, client_with_repo):
        client, repo = client_with_repo
        repo.update_status.return_value = None
        resp = await client.patch(
            f"/api/v1/suggestions/{SUGGESTION_ID}/status",
            json={"status": "rejected"},
        )
        assert resp.status_code == 404

    async def test_invalid_status_returns_422(self, client_with_repo):
        client, _ = client_with_repo
        resp = await client.patch(
            f"/api/v1/suggestions/{SUGGESTION_ID}/status",
            json={"status": "invalid"},
        )
        assert resp.status_code == 422

    async def test_passes_status_value_to_repo(self, client_with_repo):
        client, repo = client_with_repo
        repo.update_status.return_value = make_suggestion(status="expired").model_dump()
        await client.patch(
            f"/api/v1/suggestions/{SUGGESTION_ID}/status",
            json={"status": "expired"},
        )
        repo.update_status.assert_awaited_once_with(SUGGESTION_ID, "expired")
