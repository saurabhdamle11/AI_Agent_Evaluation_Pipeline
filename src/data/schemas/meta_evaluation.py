from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EvaluatorMetrics(BaseModel):
    evaluator_type: str          # "heuristic" | "tool_evaluator" | "coherence" | "llm_judge"
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    correlation_with_human: float = Field(ge=-1.0, le=1.0)
    sample_size: int = Field(ge=0)


class BlindSpot(BaseModel):
    annotation_type: str         # annotation type consistently missed
    miss_rate: float = Field(ge=0.0, le=1.0)
    example_count: int = Field(ge=0)


class CalibrationAdjustment(BaseModel):
    evaluator_type: str
    weight_multiplier: float     # multiply the evaluator's current weight by this
    reason: str


class MetaEvaluationResult(BaseModel):
    meta_evaluation_id: str
    evaluator_metrics: list[EvaluatorMetrics]
    blind_spots: list[BlindSpot]
    calibration_adjustments: list[CalibrationAdjustment]
    sample_size: int = Field(ge=0)   # total conversation pairs analyzed
    created_at: datetime
