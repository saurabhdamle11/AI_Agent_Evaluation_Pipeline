from typing import Optional

from src.data.repositories.suggestion_repository import SuggestionRepository
from src.data.schemas.suggestion import Suggestion, SuggestionStatus, SuggestionType


class SuggestionService:
    @staticmethod
    async def list_suggestions(
        repo: SuggestionRepository,
        status: Optional[SuggestionStatus] = None,
        type: Optional[SuggestionType] = None,
        agent_version: Optional[str] = None,
    ) -> list[Suggestion]:
        rows = await repo.list_all(
            status=status.value if status else None,
            type=type.value if type else None,
            agent_version=agent_version,
        )
        return [Suggestion(**row) for row in rows]

    @staticmethod
    async def get_suggestion(suggestion_id: str, repo: SuggestionRepository) -> Suggestion | None:
        data = await repo.get_by_id(suggestion_id)
        return Suggestion(**data) if data else None

    @staticmethod
    async def list_suggestions_for_conversation(
        conversation_id: str,
        repo: SuggestionRepository,
    ) -> list[Suggestion]:
        rows = await repo.list_by_conversation_id(conversation_id)
        return [Suggestion(**row) for row in rows]

    @staticmethod
    async def update_status(
        suggestion_id: str,
        status: SuggestionStatus,
        repo: SuggestionRepository,
    ) -> Suggestion | None:
        data = await repo.update_status(suggestion_id, status.value)
        return Suggestion(**data) if data else None
