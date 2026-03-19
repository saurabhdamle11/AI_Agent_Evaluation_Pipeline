from fastapi import APIRouter, Depends, HTTPException, status

from src.data.repositories.evaluation_repository import EvaluationRepository
from src.data.schemas.evaluation import (
    EvaluationResult,
    TriggerEvaluationRequest,
    TriggerEvaluationResponse,
)
from src.services.evaluation_api_service import EvaluationApiService
from src.utils.dependencies import get_evaluation_repo

router = APIRouter()


@router.post(
    "/trigger",
    response_model=TriggerEvaluationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger evaluation for a conversation",
)
async def trigger_evaluation(payload: TriggerEvaluationRequest) -> TriggerEvaluationResponse:
    return await EvaluationApiService.trigger_evaluation(payload.conversation_id)


@router.get(
    "/conversation/{conversation_id}",
    response_model=list[EvaluationResult],
    summary="List all evaluation results for a conversation",
)
async def list_evaluations_for_conversation(
    conversation_id: str,
    repo: EvaluationRepository = Depends(get_evaluation_repo),
) -> list[EvaluationResult]:
    return await EvaluationApiService.list_evaluations_for_conversation(conversation_id, repo)


@router.get(
    "/{evaluation_id}",
    response_model=EvaluationResult,
    summary="Get a single evaluation result by ID",
)
async def get_evaluation(
    evaluation_id: str,
    repo: EvaluationRepository = Depends(get_evaluation_repo),
) -> EvaluationResult:
    result = await EvaluationApiService.get_evaluation(evaluation_id, repo)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation not found")
    return result
