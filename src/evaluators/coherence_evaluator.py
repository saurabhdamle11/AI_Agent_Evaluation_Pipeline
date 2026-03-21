import logging

from src.data.schemas.evaluation import CoherenceDetails, EvaluatorOutput, IssueSchema, IssueSeverity
from src.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

_MIN_SUBSTANTIVE_CONTENT_LEN = 50


class CoherenceEvaluator(BaseEvaluator):
    async def evaluate(self, conversation: dict) -> EvaluatorOutput:
        issues: list[IssueSchema] = []
        turns = conversation.get("turns", [])

        if not turns:
            return EvaluatorOutput(
                evaluator_name="coherence",
                scores={"coherence": 0.0},
                issues=[IssueSchema(
                    type="empty_conversation",
                    severity=IssueSeverity.critical,
                    description="Conversation has no turns",
                )],
            )

        # consistency_score: check that non-system roles alternate
        non_system = [t for t in turns if t.get("role") != "system"]
        role_violations = 0
        for i in range(1, len(non_system)):
            if non_system[i].get("role") == non_system[i - 1].get("role"):
                role_violations += 1
                issues.append(IssueSchema(
                    type="role_alternation_violation",
                    severity=IssueSeverity.warning,
                    description=(
                        f"Consecutive turns with same role '{non_system[i].get('role')}' "
                        f"at non-system positions {i - 1} and {i}"
                    ),
                ))

        consistency_score = max(0.0, 1.0 - role_violations / max(len(non_system), 1))

        # context_retention: proxy combining response coverage and response depth
        user_turns = [t for t in turns if t.get("role") == "user"]
        assistant_turns = [t for t in turns if t.get("role") == "assistant"]

        if user_turns:
            response_ratio = min(len(assistant_turns) / len(user_turns), 1.0)
            substantive = sum(
                1 for t in assistant_turns
                if len(t.get("content", "")) >= _MIN_SUBSTANTIVE_CONTENT_LEN
            )
            substantive_ratio = substantive / len(assistant_turns) if assistant_turns else 0.0
            context_retention = (response_ratio + substantive_ratio) / 2
        else:
            context_retention = 1.0

        if context_retention < 0.5:
            issues.append(IssueSchema(
                type="low_context_retention",
                severity=IssueSeverity.warning,
                description=f"Low context retention score: {context_retention:.2f}",
            ))

        contradiction_count = 0

        coherence_score = round((consistency_score + context_retention) / 2, 4)

        details = CoherenceDetails(
            consistency_score=round(consistency_score, 4),
            contradiction_count=contradiction_count,
            context_retention=round(context_retention, 4),
        )

        return EvaluatorOutput(
            evaluator_name="coherence",
            scores={"coherence": coherence_score},
            details=details.model_dump(),
            issues=issues,
        )
