import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.data.schemas.feedback import AggregatedFeedback, AnnotationConsensus, FeedbackResponse
from src.data.schemas.ingestion import AnnotationSchema, OpsReviewSchema
from src.utils.dependencies import get_feedback_repo

CONVERSATION_ID = "conv-001"
NOW = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def make_feedback_response(**overrides) -> FeedbackResponse:
    base = dict(
        conversation_id=CONVERSATION_ID,
        user_rating=4,
        ops_review=OpsReviewSchema(quality="good", notes="ok"),
        annotations=[
            AnnotationSchema(type="tone", label="positive", annotator_id="a1", confidence=0.9)
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    base.update(overrides)
    return FeedbackResponse(**base)


def make_aggregated_response(**overrides) -> AggregatedFeedback:
    base = dict(
        conversation_id=CONVERSATION_ID,
        user_rating=4,
        ops_review=None,
        annotation_consensus=[
            AnnotationConsensus(
                type="tone",
                consensus_label="positive",
                agreement_score=1.0,
                kappa=1.0,
                weighted_confidence=0.9,
                annotator_count=2,
            )
        ],
        overall_kappa=1.0,
        routing="auto_label",
        aggregated_at=NOW,
    )
    base.update(overrides)
    return AggregatedFeedback(**base)


@pytest.fixture
def mock_feedback_repo():
    repo = MagicMock()
    repo.get_by_conversation_id = AsyncMock()
    return repo


@pytest.fixture
def client_with_repo(client, mock_feedback_repo):
    from src.main import app
    app.dependency_overrides[get_feedback_repo] = lambda: mock_feedback_repo
    yield client, mock_feedback_repo
    app.dependency_overrides.pop(get_feedback_repo, None)


# ── GET /api/v1/feedback/{conversation_id} ────────────────────────────────────

class TestGetFeedback:
    async def test_returns_200_with_feedback(self, client_with_repo):
        client, repo = client_with_repo
        with MagicMock() as mock_service:
            from unittest.mock import patch
            with patch(
                "src.controllers.feedback_controller.FeedbackApiService.get_feedback",
                new_callable=AsyncMock,
                return_value=make_feedback_response(),
            ):
                resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}")
        assert resp.status_code == 200
        assert resp.json()["conversation_id"] == CONVERSATION_ID

    async def test_returns_404_when_not_found(self, client_with_repo):
        client, _ = client_with_repo
        with MagicMock():
            from unittest.mock import patch
            with patch(
                "src.controllers.feedback_controller.FeedbackApiService.get_feedback",
                new_callable=AsyncMock,
                return_value=None,
            ):
                resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    async def test_response_includes_user_rating(self, client_with_repo):
        client, _ = client_with_repo
        from unittest.mock import patch
        with patch(
            "src.controllers.feedback_controller.FeedbackApiService.get_feedback",
            new_callable=AsyncMock,
            return_value=make_feedback_response(user_rating=5),
        ):
            resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}")
        assert resp.json()["user_rating"] == 5

    async def test_response_includes_ops_review(self, client_with_repo):
        client, _ = client_with_repo
        from unittest.mock import patch
        with patch(
            "src.controllers.feedback_controller.FeedbackApiService.get_feedback",
            new_callable=AsyncMock,
            return_value=make_feedback_response(),
        ):
            resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}")
        assert resp.json()["ops_review"]["quality"] == "good"

    async def test_response_includes_annotations(self, client_with_repo):
        client, _ = client_with_repo
        from unittest.mock import patch
        with patch(
            "src.controllers.feedback_controller.FeedbackApiService.get_feedback",
            new_callable=AsyncMock,
            return_value=make_feedback_response(),
        ):
            resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}")
        annotations = resp.json()["annotations"]
        assert len(annotations) == 1
        assert annotations[0]["type"] == "tone"

    async def test_queries_repo_with_correct_conversation_id(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_conversation_id.return_value = None
        resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}")
        repo.get_by_conversation_id.assert_awaited_once_with(CONVERSATION_ID)


# ── GET /api/v1/feedback/{conversation_id}/aggregated ────────────────────────

class TestGetAggregatedFeedback:
    async def test_returns_200_with_aggregated(self, client_with_repo):
        client, _ = client_with_repo
        from unittest.mock import patch
        with patch(
            "src.controllers.feedback_controller.FeedbackApiService.get_aggregated_feedback",
            new_callable=AsyncMock,
            return_value=make_aggregated_response(),
        ):
            resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}/aggregated")
        assert resp.status_code == 200
        assert resp.json()["conversation_id"] == CONVERSATION_ID

    async def test_returns_404_when_not_found(self, client_with_repo):
        client, _ = client_with_repo
        from unittest.mock import patch
        with patch(
            "src.controllers.feedback_controller.FeedbackApiService.get_aggregated_feedback",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}/aggregated")
        assert resp.status_code == 404

    async def test_404_detail_mentions_processing(self, client_with_repo):
        client, _ = client_with_repo
        from unittest.mock import patch
        with patch(
            "src.controllers.feedback_controller.FeedbackApiService.get_aggregated_feedback",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}/aggregated")
        assert "processed" in resp.json()["detail"].lower()

    async def test_response_includes_overall_kappa(self, client_with_repo):
        client, _ = client_with_repo
        from unittest.mock import patch
        with patch(
            "src.controllers.feedback_controller.FeedbackApiService.get_aggregated_feedback",
            new_callable=AsyncMock,
            return_value=make_aggregated_response(overall_kappa=0.75),
        ):
            resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}/aggregated")
        assert resp.json()["overall_kappa"] == pytest.approx(0.75)

    async def test_response_includes_routing(self, client_with_repo):
        client, _ = client_with_repo
        from unittest.mock import patch
        with patch(
            "src.controllers.feedback_controller.FeedbackApiService.get_aggregated_feedback",
            new_callable=AsyncMock,
            return_value=make_aggregated_response(routing="human_review"),
        ):
            resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}/aggregated")
        assert resp.json()["routing"] == "human_review"

    async def test_response_includes_annotation_consensus(self, client_with_repo):
        client, _ = client_with_repo
        from unittest.mock import patch
        with patch(
            "src.controllers.feedback_controller.FeedbackApiService.get_aggregated_feedback",
            new_callable=AsyncMock,
            return_value=make_aggregated_response(),
        ):
            resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}/aggregated")
        consensus = resp.json()["annotation_consensus"]
        assert len(consensus) == 1
        assert consensus[0]["consensus_label"] == "positive"
        assert consensus[0]["kappa"] == pytest.approx(1.0)

    async def test_queries_repo_with_correct_conversation_id(self, client_with_repo):
        client, repo = client_with_repo
        repo.get_by_conversation_id.return_value = None
        resp = await client.get(f"/api/v1/feedback/{CONVERSATION_ID}/aggregated")
        repo.get_by_conversation_id.assert_awaited_once_with(CONVERSATION_ID)
