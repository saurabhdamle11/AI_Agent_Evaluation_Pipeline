from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class IssueSeverity(str, Enum):
    info = "info"
    warning = "warning"
    critical = "critical"


class IssueSchema(BaseModel):
    type: str
    severity: IssueSeverity
    description: str


class ToolEvaluationDetails(BaseModel):
    selection_accuracy: float = Field(ge=0.0, le=1.0)
    parameter_accuracy: float = Field(ge=0.0, le=1.0)
    hallucinated_params: int = Field(ge=0)
    execution_success: float = Field(ge=0.0, le=1.0)


class CoherenceDetails(BaseModel):
    consistency_score: float = Field(ge=0.0, le=1.0)
    contradiction_count: int = Field(ge=0)
    context_retention: float = Field(ge=0.0, le=1.0)


class EvaluationScores(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    response_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    tool_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    coherence: Optional[float] = Field(None, ge=0.0, le=1.0)
    heuristic: Optional[float] = Field(None, ge=0.0, le=1.0)
    tool_details: Optional[ToolEvaluationDetails] = None
    coherence_details: Optional[CoherenceDetails] = None


class EvaluationResult(BaseModel):
    evaluation_id: str
    conversation_id: str
    scores: EvaluationScores
    issues_detected: list[IssueSchema] = []
    created_at: datetime


class TriggerEvaluationRequest(BaseModel):
    conversation_id: str


class TriggerEvaluationResponse(BaseModel):
    conversation_id: str
    status: str


# Internal type passed between evaluators and the orchestrator — not persisted directly
class EvaluatorOutput(BaseModel):
    evaluator_name: str
    scores: dict[str, float]
    details: dict[str, Any] = {}
    issues: list[IssueSchema] = []
