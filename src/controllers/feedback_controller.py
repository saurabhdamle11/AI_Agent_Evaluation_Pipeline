from fastapi import APIRouter, Depends, HTTPException, status

from src.data.repositories.feedback_repository import FeedbackRepository
from src.data.schemas.feedback import AggregatedFeedback, FeedbackResponse
from src.services.feedback_api_service import FeedbackApiService
from src.utils.dependencies import get_feedback_repo

router = APIRouter()


@router.get(
    "/{conversation_id}",
    response_model=FeedbackResponse,
    summary="Get raw feedback for a conversation",
)
async def get_feedback(
    conversation_id: str,
    repo: FeedbackRepository = Depends(get_feedback_repo),
) -> FeedbackResponse:
    result = await FeedbackApiService.get_feedback(conversation_id, repo)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found",
        )
    return result


@router.get(
    "/{conversation_id}/aggregated",
    response_model=AggregatedFeedback,
    summary="Get aggregated feedback (inter-annotator agreement + routing) for a conversation",
)
async def get_aggregated_feedback(
    conversation_id: str,
    repo: FeedbackRepository = Depends(get_feedback_repo),
) -> AggregatedFeedback:
    result = await FeedbackApiService.get_aggregated_feedback(conversation_id, repo)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aggregated feedback not found — feedback may not have been processed yet",
        )
    return result
