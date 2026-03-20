import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.data.schemas.feedback import AggregatedFeedback, FeedbackResponse
from src.services.feedback_api_service import FeedbackApiService


CONVERSATION_ID = "conv-001"
NOW = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def make_raw_doc(**overrides) -> dict:
    base = {
        "conversation_id": CONVERSATION_ID,
        "user_rating": 4,
        "ops_review": {"quality": "good", "notes": "ok"},
        "annotations": [
            {"type": "tone", "label": "positive", "annotator_id": "a1", "confidence": 0.9}
        ],
        "created_at": NOW,
        "updated_at": NOW,
    }
    base.update(overrides)
    return base


def make_aggregated_doc() -> dict:
    return {
        "conversation_id": CONVERSATION_ID,
        "user_rating": 4,
        "ops_review": None,
        "annotation_consensus": [
            {
                "type": "tone",
                "consensus_label": "positive",
                "agreement_score": 1.0,
                "kappa": 1.0,
                "weighted_confidence": 0.9,
                "annotator_count": 1,
            }
        ],
        "overall_kappa": 1.0,
        "routing": "auto_label",
        "aggregated_at": NOW,
    }


def make_repo(raw_doc=None) -> MagicMock:
    repo = MagicMock()
    repo.get_by_conversation_id = AsyncMock(return_value=raw_doc)
    return repo


# ── get_feedback ─────────────────────────────────────────────────────────────

class TestGetFeedback:
    async def test_returns_feedback_response_when_found(self):
        repo = make_repo(make_raw_doc())
        result = await FeedbackApiService.get_feedback(CONVERSATION_ID, repo)
        assert isinstance(result, FeedbackResponse)
        assert result.conversation_id == CONVERSATION_ID

    async def test_returns_none_when_not_found(self):
        repo = make_repo(None)
        result = await FeedbackApiService.get_feedback(CONVERSATION_ID, repo)
        assert result is None

    async def test_queries_repo_with_correct_conversation_id(self):
        repo = make_repo(make_raw_doc())
        await FeedbackApiService.get_feedback(CONVERSATION_ID, repo)
        repo.get_by_conversation_id.assert_awaited_once_with(CONVERSATION_ID)

    async def test_response_includes_user_rating(self):
        repo = make_repo(make_raw_doc(user_rating=5))
        result = await FeedbackApiService.get_feedback(CONVERSATION_ID, repo)
        assert result.user_rating == 5

    async def test_response_includes_annotations(self):
        repo = make_repo(make_raw_doc())
        result = await FeedbackApiService.get_feedback(CONVERSATION_ID, repo)
        assert len(result.annotations) == 1
        assert result.annotations[0].type == "tone"

    async def test_response_includes_timestamps(self):
        repo = make_repo(make_raw_doc())
        result = await FeedbackApiService.get_feedback(CONVERSATION_ID, repo)
        assert result.created_at == NOW
        assert result.updated_at == NOW

    async def test_optional_fields_can_be_none(self):
        doc = make_raw_doc(user_rating=None, ops_review=None, annotations=None)
        repo = make_repo(doc)
        result = await FeedbackApiService.get_feedback(CONVERSATION_ID, repo)
        assert result.user_rating is None
        assert result.ops_review is None
        assert result.annotations is None


# ── get_aggregated_feedback ───────────────────────────────────────────────────

class TestGetAggregatedFeedback:
    async def test_returns_aggregated_feedback_when_present(self):
        doc = make_raw_doc(aggregated=make_aggregated_doc())
        repo = make_repo(doc)
        result = await FeedbackApiService.get_aggregated_feedback(CONVERSATION_ID, repo)
        assert isinstance(result, AggregatedFeedback)
        assert result.conversation_id == CONVERSATION_ID

    async def test_returns_none_when_feedback_not_found(self):
        repo = make_repo(None)
        result = await FeedbackApiService.get_aggregated_feedback(CONVERSATION_ID, repo)
        assert result is None

    async def test_returns_none_when_aggregated_field_missing(self):
        repo = make_repo(make_raw_doc())  # no "aggregated" key
        result = await FeedbackApiService.get_aggregated_feedback(CONVERSATION_ID, repo)
        assert result is None

    async def test_queries_repo_with_correct_conversation_id(self):
        doc = make_raw_doc(aggregated=make_aggregated_doc())
        repo = make_repo(doc)
        await FeedbackApiService.get_aggregated_feedback(CONVERSATION_ID, repo)
        repo.get_by_conversation_id.assert_awaited_once_with(CONVERSATION_ID)

    async def test_routing_is_preserved(self):
        doc = make_raw_doc(aggregated={**make_aggregated_doc(), "routing": "human_review"})
        repo = make_repo(doc)
        result = await FeedbackApiService.get_aggregated_feedback(CONVERSATION_ID, repo)
        assert result.routing == "human_review"

    async def test_overall_kappa_is_preserved(self):
        doc = make_raw_doc(aggregated={**make_aggregated_doc(), "overall_kappa": 0.42})
        repo = make_repo(doc)
        result = await FeedbackApiService.get_aggregated_feedback(CONVERSATION_ID, repo)
        assert result.overall_kappa == pytest.approx(0.42)

    async def test_annotation_consensus_is_preserved(self):
        doc = make_raw_doc(aggregated=make_aggregated_doc())
        repo = make_repo(doc)
        result = await FeedbackApiService.get_aggregated_feedback(CONVERSATION_ID, repo)
        assert len(result.annotation_consensus) == 1
        assert result.annotation_consensus[0].type == "tone"
