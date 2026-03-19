from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from src.data.repositories.suggestion_repository import SuggestionRepository
from src.data.schemas.suggestion import (
    Suggestion,
    SuggestionStatus,
    SuggestionType,
    UpdateSuggestionStatusRequest,
)
from src.services.suggestion_service import SuggestionService
from src.utils.dependencies import get_suggestion_repo

router = APIRouter()


@router.get(
    "",
    response_model=list[Suggestion],
    summary="List improvement suggestions with optional filters",
)
async def list_suggestions(
    status: Optional[SuggestionStatus] = None,
    type: Optional[SuggestionType] = None,
    agent_version: Optional[str] = None,
    repo: SuggestionRepository = Depends(get_suggestion_repo),
) -> list[Suggestion]:
    return await SuggestionService.list_suggestions(repo, status=status, type=type, agent_version=agent_version)


@router.get(
    "/conversation/{conversation_id}",
    response_model=list[Suggestion],
    summary="List suggestions linked to a conversation",
)
async def list_suggestions_for_conversation(
    conversation_id: str,
    repo: SuggestionRepository = Depends(get_suggestion_repo),
) -> list[Suggestion]:
    return await SuggestionService.list_suggestions_for_conversation(conversation_id, repo)


@router.get(
    "/{suggestion_id}",
    response_model=Suggestion,
    summary="Get a single suggestion by ID",
)
async def get_suggestion(
    suggestion_id: str,
    repo: SuggestionRepository = Depends(get_suggestion_repo),
) -> Suggestion:
    result = await SuggestionService.get_suggestion(suggestion_id, repo)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Suggestion not found")
    return result


@router.patch(
    "/{suggestion_id}/status",
    response_model=Suggestion,
    summary="Update the status of a suggestion",
)
async def update_suggestion_status(
    suggestion_id: str,
    payload: UpdateSuggestionStatusRequest,
    repo: SuggestionRepository = Depends(get_suggestion_repo),
) -> Suggestion:
    result = await SuggestionService.update_status(suggestion_id, payload.status, repo)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Suggestion not found")
    return result
