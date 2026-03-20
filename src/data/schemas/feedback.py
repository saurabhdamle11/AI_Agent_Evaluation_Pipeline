from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from src.data.schemas.ingestion import AnnotationSchema, OpsReviewSchema


class AnnotationConsensus(BaseModel):
    type: str
    consensus_label: str
    agreement_score: float  # fraction of annotators who agree with majority label
    kappa: float            # pairwise Cohen's kappa across all annotator pairs
    weighted_confidence: float
    annotator_count: int


class AggregatedFeedback(BaseModel):
    conversation_id: str
    user_rating: Optional[int] = None
    ops_review: Optional[OpsReviewSchema] = None
    annotation_consensus: list[AnnotationConsensus]
    overall_kappa: float    # mean kappa across all annotation types
    routing: str            # "auto_label" or "human_review"
    aggregated_at: datetime


class FeedbackResponse(BaseModel):
    conversation_id: str
    user_rating: Optional[int] = None
    ops_review: Optional[OpsReviewSchema] = None
    annotations: Optional[list[AnnotationSchema]] = None
    created_at: datetime
    updated_at: datetime
