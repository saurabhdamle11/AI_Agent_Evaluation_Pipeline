import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.data.schemas.evaluation import (
    CoherenceDetails,
    EvaluationResult,
    EvaluationScores,
    ToolEvaluationDetails,
)
from src.utils.dependencies import get_evaluation_repo

EVALUATION_ID = "eval-abc-123"
CONVERSATION_ID = "conv-xyz-456"


def make_evaluation_result(
    evaluation_id=EVALUATION_ID,
    conversation_id=CONVERSATION_ID,
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_id=evaluation_id,
        conversation_id=conversation_id,
        scores=EvaluationScores(
            overall=0.85,
            heuristic=0.9,
            tool_accuracy=1.0,
            coherence=0.8,
            response_quality=0.5,
        ),
        issues_detected=[],
        created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_publish():
    with patch(
        "src.services.kafka_producer_service.KafkaProducerService.publish",
        new_callable=AsyncMock,
    ) as m:
        yield m


@pytest.fixture
def mock_eval_repo():
    repo = MagicMock()
    repo.get_by_id = AsyncMock()
    repo.get_by_conversation_id = AsyncMock()
    return repo


@pytest.fixture
def client_with_repo(client, mock_eval_repo):
    """Override the evaluation repo dependency with a mock."""
    from src.main import app
    app.dependency_overrides[get_evaluation_repo] = lambda: mock_eval_repo
    yield client, mock_eval_repo
    app.dependency_overrides.pop(get_evaluation_repo, None)


class TestTriggerEvaluation:
    async def test_returns_202_and_queued_status(self, client, mock_publish):
        resp = await client.post(
            "/api/v1/evaluations/trigger",
            json={"conversation_id": CONVERSATION_ID},
        )
        assert resp.status_code == 202
        assert resp.json() == {"conversation_id": CONVERSATION_ID, "status": "queued"}

    async def test_publishes_to_evaluations_topic(self, client, mock_publish):
        from src.config.settings import get_settings
        settings = get_settings()
        await client.post(
            "/api/v1/evaluations/trigger",
            json={"conversation_id": CONVERSATION_ID},
        )
        mock_publish.assert_awaited_once()
        assert mock_publish.call_args.kwargs["topic"] == settings.kafka_topic_evaluations

    async def test_publishes_with_conversation_id_as_key(self, client, mock_publish):
        await client.post(
            "/api/v1/evaluations/trigger",
            json={"conversation_id": CONVERSATION_ID},
        )
        assert mock_publish.call_args.kwargs["key"] == CONVERSATION_ID

    async def test_missing_conversation_id_returns_422(self, client, mock_publish):
        resp = await client.post("/api/v1/evaluations/trigger", json={})
        assert resp.status_code == 422

    async def test_kafka_error_returns_500(self, client, mock_publish):
        from src.utils.exceptions import KafkaPublishError
        mock_publish.side_effect = KafkaPublishError("broker down")
        resp = await client.post(
            "/api/v1/evaluations/trigger",
            json={"conversation_id": CONVERSATION_ID},
        )
        assert resp.status_code == 500


class TestGetEvaluation:
    async def test_returns_200_with_evaluation(self, client_with_repo):
        client, repo = client_with_repo
        result = make_evaluation_result()
        repo.get_by_id.return_value = result.model_dump()
        resp = await client.get(f"/api/v1/evaluations/{EVALUATION_ID}")
        assert resp.status_code == 200
        assert resp.json()["evaluation_id"] == EVALUATION_ID

    async def test_returns_404_when_not_found(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_id.return_value = None
        resp = await client.get(f"/api/v1/evaluations/{EVALUATION_ID}")
        assert resp.status_code == 404

    async def test_response_includes_scores(self, client_with_repo):
        client, repo = client_with_repo
        result = make_evaluation_result()
        repo.get_by_id.return_value = result.model_dump()
        resp = await client.get(f"/api/v1/evaluations/{EVALUATION_ID}")
        body = resp.json()
        assert "scores" in body
        assert body["scores"]["overall"] == 0.85

    async def test_queries_repo_with_correct_id(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_id.return_value = make_evaluation_result().model_dump()
        await client.get(f"/api/v1/evaluations/{EVALUATION_ID}")
        repo.get_by_id.assert_awaited_once_with(EVALUATION_ID)


class TestListEvaluationsForConversation:
    async def test_returns_200_with_list(self, client_with_repo):
        client, repo = client_with_repo
        results = [
            make_evaluation_result("eval-1").model_dump(),
            make_evaluation_result("eval-2").model_dump(),
        ]
        repo.get_by_conversation_id.return_value = results
        resp = await client.get(f"/api/v1/evaluations/conversation/{CONVERSATION_ID}")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_returns_empty_list_when_none_found(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_conversation_id.return_value = []
        resp = await client.get(f"/api/v1/evaluations/conversation/{CONVERSATION_ID}")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_queries_repo_with_correct_conversation_id(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_conversation_id.return_value = []
        await client.get(f"/api/v1/evaluations/conversation/{CONVERSATION_ID}")
        repo.get_by_conversation_id.assert_awaited_once_with(CONVERSATION_ID)

    async def test_response_items_include_evaluation_ids(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_conversation_id.return_value = [
            make_evaluation_result("eval-aaa").model_dump(),
            make_evaluation_result("eval-bbb").model_dump(),
        ]
        resp = await client.get(f"/api/v1/evaluations/conversation/{CONVERSATION_ID}")
        ids = [item["evaluation_id"] for item in resp.json()]
        assert ids == ["eval-aaa", "eval-bbb"]
