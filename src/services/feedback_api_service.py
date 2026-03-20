from src.data.repositories.feedback_repository import FeedbackRepository
from src.data.schemas.feedback import AggregatedFeedback, FeedbackResponse


class FeedbackApiService:
    @staticmethod
    async def get_feedback(
        conversation_id: str,
        repo: FeedbackRepository,
    ) -> FeedbackResponse | None:
        data = await repo.get_by_conversation_id(conversation_id)
        return FeedbackResponse(**data) if data else None

    @staticmethod
    async def get_aggregated_feedback(
        conversation_id: str,
        repo: FeedbackRepository,
    ) -> AggregatedFeedback | None:
        data = await repo.get_by_conversation_id(conversation_id)
        if not data or "aggregated" not in data:
            return None
        return AggregatedFeedback(**data["aggregated"])
