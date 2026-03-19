from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SuggestionType(str, Enum):
    prompt = "prompt"
    tool = "tool"


class SuggestionStatus(str, Enum):
    pending = "pending"
    applied = "applied"
    rejected = "rejected"
    expired = "expired"


class Suggestion(BaseModel):
    suggestion_id: str
    type: SuggestionType
    conversation_ids: list[str] = []
    agent_version: Optional[str] = None
    suggestion: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    expected_impact: Optional[str] = None
    status: SuggestionStatus = SuggestionStatus.pending
    created_at: datetime
    resolved_at: Optional[datetime] = None


class UpdateSuggestionStatusRequest(BaseModel):
    status: SuggestionStatus
