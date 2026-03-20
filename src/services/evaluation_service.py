import asyncio
import logging
import uuid
from datetime import datetime, timezone

from src.data.repositories.conversation_repository import ConversationRepository
from src.data.repositories.evaluation_repository import EvaluationRepository
from src.data.repositories.suggestion_repository import SuggestionRepository
from src.data.schemas.evaluation import (
    CoherenceDetails,
    EvaluationResult,
    EvaluationScores,
    IssueSchema,
    IssueSeverity,
    ToolEvaluationDetails,
)
from src.evaluators.coherence_evaluator import CoherenceEvaluator
from src.evaluators.heuristic_evaluator import HeuristicEvaluator
from src.evaluators.llm_judge_evaluator import LLMJudgeEvaluator
from src.evaluators.tool_call_evaluator import ToolCallEvaluator

logger = logging.getLogger(__name__)

# Contribution of each score dimension to the overall score
_WEIGHTS: dict[str, float] = {
    "heuristic": 0.15,
    "tool_accuracy": 0.30,
    "coherence": 0.25,
    "response_quality": 0.30,
}

_TOOL_DETAIL_FIELDS = {"selection_accuracy", "parameter_accuracy", "hallucinated_params", "execution_success"}
_COHERENCE_DETAIL_FIELDS = {"consistency_score", "contradiction_count", "context_retention"}


class EvaluationService:
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        evaluation_repo: EvaluationRepository,
        suggestion_repo: SuggestionRepository | None = None,
    ) -> None:
        self.conversation_repo = conversation_repo
        self.evaluation_repo = evaluation_repo
        self.suggestion_repo = suggestion_repo
        self._evaluators = [
            HeuristicEvaluator(),
            ToolCallEvaluator(),
            CoherenceEvaluator(),
            LLMJudgeEvaluator(),
        ]

    async def evaluate(self, conversation_id: str) -> EvaluationResult | None:
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            logger.warning(f"Conversation not found: {conversation_id}")
            return None

        # Fan out to all evaluators concurrently; capture individual failures
        raw_outputs = await asyncio.gather(
            *[ev.evaluate(conversation) for ev in self._evaluators],
            return_exceptions=True,
        )

        merged_scores: dict[str, float] = {}
        merged_details: dict[str, dict] = {}
        all_issues: list[IssueSchema] = []

        for output in raw_outputs:
            if isinstance(output, BaseException):
                logger.error(f"Evaluator raised an exception: {output}")
                all_issues.append(IssueSchema(
                    type="evaluator_error",
                    severity=IssueSeverity.warning,
                    message=str(output),
                ))
                continue
            merged_scores.update(output.scores)
            if output.details:
                merged_details[output.evaluator_name] = output.details
            all_issues.extend(output.issues)

        # Build typed detail objects only when all required fields are present
        tool_detail_data = merged_details.get("tool_call", {})
        tool_details = (
            ToolEvaluationDetails(**tool_detail_data)
            if _TOOL_DETAIL_FIELDS.issubset(tool_detail_data)
            else None
        )

        coherence_detail_data = merged_details.get("coherence", {})
        coherence_details = (
            CoherenceDetails(**coherence_detail_data)
            if _COHERENCE_DETAIL_FIELDS.issubset(coherence_detail_data)
            else None
        )

        scores = EvaluationScores(
            overall=round(self._weighted_overall(merged_scores), 4),
            response_quality=merged_scores.get("response_quality"),
            tool_accuracy=merged_scores.get("tool_accuracy"),
            coherence=merged_scores.get("coherence"),
            heuristic=merged_scores.get("heuristic"),
            tool_details=tool_details,
            coherence_details=coherence_details,
        )

        result = EvaluationResult(
            evaluation_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            scores=scores,
            issues_detected=all_issues,
            created_at=datetime.now(timezone.utc),
        )

        await self.evaluation_repo.insert(result.model_dump())
        logger.info(
            f"Evaluation complete — id={result.evaluation_id} "
            f"conversation={conversation_id} overall={scores.overall}"
        )

        if self.suggestion_repo:
            await self._generate_suggestions(result, conversation.get("agent_version", "unknown"))

        return result

    async def _generate_suggestions(self, result: EvaluationResult, agent_version: str) -> None:
        from src.services.suggestion_generator import SuggestionGenerator
        candidates = SuggestionGenerator().generate(result, agent_version)
        for doc in candidates:
            fp = doc["_fingerprint"]
            if await self.suggestion_repo.exists_by_fingerprint(fp, agent_version):
                await self.suggestion_repo.append_conversation_id(fp, agent_version, result.conversation_id)
            else:
                await self.suggestion_repo.insert(doc)

    def _weighted_overall(self, scores: dict[str, float]) -> float:
        present = {k: v for k, v in _WEIGHTS.items() if k in scores}
        if not present:
            return 0.0
        total_weight = sum(present.values())
        return sum(_WEIGHTS[k] * scores[k] for k in present) / total_weight
