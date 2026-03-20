from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ToolCallSchema(BaseModel):
    tool_name: str
    parameters: dict[str, Any]
    result: Optional[dict[str, Any]] = None
    latency_ms: Optional[int] = None


class TurnSchema(BaseModel):
    turn_id: int
    role: Role
    content: str
    timestamp: str
    tool_calls: list[ToolCallSchema] = Field(default_factory=list)


class ConversationMetadataSchema(BaseModel):
    total_latency_ms: Optional[int] = None
    mission_completed: Optional[bool] = None


class ConversationIngestionRequest(BaseModel):
    conversation_id: str
    agent_version: str
    turns: list[TurnSchema]
    metadata: ConversationMetadataSchema


class BatchIngestionRequest(BaseModel):
    conversations: list[ConversationIngestionRequest]


# ── Feedback ──────────────────────────────────────────────────────────────────

class QualityEnum(str, Enum):
    poor = "poor"
    fair = "fair"
    good = "good"
    excellent = "excellent"


class OpsReviewSchema(BaseModel):
    quality: Optional[QualityEnum] = None
    notes: Optional[str] = None


class AnnotationSchema(BaseModel):
    type: str
    label: str
    annotator_id: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class FeedbackIngestionRequest(BaseModel):
    conversation_id: str
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    ops_review: Optional[OpsReviewSchema] = None
    annotations: Optional[list[AnnotationSchema]] = None


# ── Responses ─────────────────────────────────────────────────────────────────

class IngestionResponse(BaseModel):
    conversation_id: str
    status: str


class BatchIngestionResponse(BaseModel):
    queued: int
    status: str
