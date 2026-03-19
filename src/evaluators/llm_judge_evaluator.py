import logging

from src.data.schemas.evaluation import EvaluatorOutput, IssueSchema, IssueSeverity
from src.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class LLMJudgeEvaluator(BaseEvaluator):
    """
    Stub implementation pending Anthropic SDK integration.

    Will evaluate response quality, helpfulness, and factuality by submitting
    the conversation to an LLM judge (claude-sonnet-4-6). Until then, returns
    a neutral placeholder score of 0.5 so the orchestrator can still produce
    a complete evaluation result.
    """

    async def evaluate(self, conversation: dict) -> EvaluatorOutput:
        logger.warning("LLMJudgeEvaluator is a stub — Anthropic SDK integration pending")
        return EvaluatorOutput(
            evaluator_name="llm_judge",
            scores={"response_quality": 0.5},
            details={"stub": True},
            issues=[
                IssueSchema(
                    type="stub_evaluator",
                    severity=IssueSeverity.info,
                    message="LLM Judge score is a placeholder (0.5) — Anthropic SDK not yet integrated",
                )
            ],
        )
