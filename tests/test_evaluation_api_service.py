import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.data.schemas.evaluation import EvaluationResult, EvaluationScores
from src.services.evaluation_api_service import EvaluationApiService
from src.utils.exceptions import KafkaPublishError

EVALUATION_ID = "eval-001"
CONVERSATION_ID = "conv-001"


def make_evaluation_dict(evaluation_id=EVALUATION_ID, conversation_id=CONVERSATION_ID) -> dict:
    return EvaluationResult(
        evaluation_id=evaluation_id,
        conversation_id=conversation_id,
        scores=EvaluationScores(overall=0.75, heuristic=0.8, tool_accuracy=0.9, coherence=0.7, response_quality=0.5),
        issues_detected=[],
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    ).model_dump()


@pytest.fixture
def mock_publish():
    with patch(
        "src.services.evaluation_api_service.KafkaProducerService.publish",
        new_callable=AsyncMock,
    ) as m:
        yield m


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.get_by_id = AsyncMock()
    repo.get_by_conversation_id = AsyncMock()
    return repo


class TestTriggerEvaluation:
    async def test_returns_queued_status(self, mock_publish):
        resp = await EvaluationApiService.trigger_evaluation(CONVERSATION_ID)
        assert resp.status_code if hasattr(resp, "status_code") else resp.status == "queued"
        assert resp.conversation_id == CONVERSATION_ID
        assert resp.status == "queued"

    async def test_publishes_to_evaluations_topic(self, mock_publish):
        from src.config.settings import get_settings
        settings = get_settings()
        await EvaluationApiService.trigger_evaluation(CONVERSATION_ID)
        mock_publish.assert_awaited_once_with(
            topic=settings.kafka_topic_evaluations,
            value={"conversation_id": CONVERSATION_ID},
            key=CONVERSATION_ID,
        )

    async def test_propagates_kafka_error(self, mock_publish):
        mock_publish.side_effect = KafkaPublishError("timeout")
        with pytest.raises(KafkaPublishError, match="timeout"):
            await EvaluationApiService.trigger_evaluation(CONVERSATION_ID)


class TestGetEvaluation:
    async def test_returns_evaluation_result_when_found(self, mock_repo):
        mock_repo.get_by_id.return_value = make_evaluation_dict()
        result = await EvaluationApiService.get_evaluation(EVALUATION_ID, mock_repo)
        assert result is not None
        assert result.evaluation_id == EVALUATION_ID

    async def test_returns_none_when_not_found(self, mock_repo):
        mock_repo.get_by_id.return_value = None
        result = await EvaluationApiService.get_evaluation(EVALUATION_ID, mock_repo)
        assert result is None

    async def test_queries_repo_with_correct_id(self, mock_repo):
        mock_repo.get_by_id.return_value = None
        await EvaluationApiService.get_evaluation(EVALUATION_ID, mock_repo)
        mock_repo.get_by_id.assert_awaited_once_with(EVALUATION_ID)

    async def test_result_is_evaluation_result_type(self, mock_repo):
        mock_repo.get_by_id.return_value = make_evaluation_dict()
        result = await EvaluationApiService.get_evaluation(EVALUATION_ID, mock_repo)
        assert isinstance(result, EvaluationResult)


class TestListEvaluationsForConversation:
    async def test_returns_list_of_results(self, mock_repo):
        mock_repo.get_by_conversation_id.return_value = [
            make_evaluation_dict("eval-1"),
            make_evaluation_dict("eval-2"),
        ]
        results = await EvaluationApiService.list_evaluations_for_conversation(CONVERSATION_ID, mock_repo)
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)

    async def test_returns_empty_list_when_none(self, mock_repo):
        mock_repo.get_by_conversation_id.return_value = []
        results = await EvaluationApiService.list_evaluations_for_conversation(CONVERSATION_ID, mock_repo)
        assert results == []

    async def test_queries_repo_with_correct_conversation_id(self, mock_repo):
        mock_repo.get_by_conversation_id.return_value = []
        await EvaluationApiService.list_evaluations_for_conversation(CONVERSATION_ID, mock_repo)
        mock_repo.get_by_conversation_id.assert_awaited_once_with(CONVERSATION_ID)

    async def test_preserves_order_of_results(self, mock_repo):
        mock_repo.get_by_conversation_id.return_value = [
            make_evaluation_dict("eval-first"),
            make_evaluation_dict("eval-second"),
            make_evaluation_dict("eval-third"),
        ]
        results = await EvaluationApiService.list_evaluations_for_conversation(CONVERSATION_ID, mock_repo)
        assert [r.evaluation_id for r in results] == ["eval-first", "eval-second", "eval-third"]
