from fastapi import APIRouter, Depends, HTTPException, status

from src.data.repositories.evaluation_repository import EvaluationRepository
from src.data.repositories.feedback_repository import FeedbackRepository
from src.data.repositories.meta_evaluation_repository import MetaEvaluationRepository
from src.data.schemas.meta_evaluation import MetaEvaluationResult
from src.services.meta_evaluation_service import MetaEvaluationService
from src.utils.dependencies import (
    get_evaluation_repo,
    get_feedback_repo,
    get_meta_evaluation_repo,
)

router = APIRouter()


@router.post(
    "/run",
    response_model=MetaEvaluationResult,
    status_code=status.HTTP_200_OK,
    summary="Run meta-evaluation across all paired conversations",
)
async def run_meta_evaluation(
    evaluation_repo: EvaluationRepository = Depends(get_evaluation_repo),
    feedback_repo: FeedbackRepository = Depends(get_feedback_repo),
    meta_eval_repo: MetaEvaluationRepository = Depends(get_meta_evaluation_repo),
) -> MetaEvaluationResult:
    return await MetaEvaluationService.run(evaluation_repo, feedback_repo, meta_eval_repo)


@router.get(
    "/latest",
    response_model=MetaEvaluationResult,
    summary="Get the most recent meta-evaluation result",
)
async def get_latest_meta_evaluation(
    meta_eval_repo: MetaEvaluationRepository = Depends(get_meta_evaluation_repo),
) -> MetaEvaluationResult:
    result = await meta_eval_repo.get_latest()
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No meta-evaluation has been run yet",
        )
    return MetaEvaluationResult(**result)


@router.get(
    "/{meta_evaluation_id}",
    response_model=MetaEvaluationResult,
    summary="Get a meta-evaluation result by ID",
)
async def get_meta_evaluation(
    meta_evaluation_id: str,
    meta_eval_repo: MetaEvaluationRepository = Depends(get_meta_evaluation_repo),
) -> MetaEvaluationResult:
    result = await meta_eval_repo.get_by_id(meta_evaluation_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meta-evaluation not found",
        )
    return MetaEvaluationResult(**result)
