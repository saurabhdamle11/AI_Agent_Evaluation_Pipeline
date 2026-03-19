import logging

from src.data.schemas.evaluation import EvaluatorOutput, IssueSchema, IssueSeverity
from src.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

_TOTAL_LATENCY_WARNING_MS = 30_000   # 30 s
_TOOL_LATENCY_WARNING_MS = 5_000     # 5 s per tool call
_VALID_ROLES = {"user", "assistant", "system"}


class HeuristicEvaluator(BaseEvaluator):
    async def evaluate(self, conversation: dict) -> EvaluatorOutput:
        issues: list[IssueSchema] = []
        penalty = 0.0

        turns = conversation.get("turns", [])
        metadata = conversation.get("metadata", {})

        # Total latency threshold
        total_latency = metadata.get("total_latency_ms")
        if total_latency is not None and total_latency > _TOTAL_LATENCY_WARNING_MS:
            issues.append(IssueSchema(
                type="high_latency",
                severity=IssueSeverity.warning,
                message=f"Total latency {total_latency}ms exceeds {_TOTAL_LATENCY_WARNING_MS}ms threshold",
            ))
            penalty += 0.1

        for turn in turns:
            turn_id = turn.get("turn_id", "?")

            # Required fields
            for field in ("turn_id", "role", "content", "timestamp"):
                if not turn.get(field) and turn.get(field) != 0:
                    issues.append(IssueSchema(
                        type="missing_field",
                        severity=IssueSeverity.critical,
                        message=f"Turn {turn_id} missing required field: {field}",
                    ))
                    penalty += 0.15

            # Valid role
            if turn.get("role") not in _VALID_ROLES:
                issues.append(IssueSchema(
                    type="invalid_role",
                    severity=IssueSeverity.warning,
                    message=f"Turn {turn_id} has invalid role: {turn.get('role')!r}",
                ))
                penalty += 0.05

            # Empty content
            if isinstance(turn.get("content"), str) and turn["content"].strip() == "":
                issues.append(IssueSchema(
                    type="empty_content",
                    severity=IssueSeverity.warning,
                    message=f"Turn {turn_id} has empty content",
                ))
                penalty += 0.05

            # Per-tool-call latency
            for tc in turn.get("tool_calls") or []:
                tc_latency = tc.get("latency_ms")
                if tc_latency is not None and tc_latency > _TOOL_LATENCY_WARNING_MS:
                    issues.append(IssueSchema(
                        type="tool_high_latency",
                        severity=IssueSeverity.warning,
                        message=f"Tool '{tc.get('tool_name')}' latency {tc_latency}ms exceeds {_TOOL_LATENCY_WARNING_MS}ms",
                    ))
                    penalty += 0.05

        score = round(max(0.0, 1.0 - penalty), 4)
        return EvaluatorOutput(
            evaluator_name="heuristic",
            scores={"heuristic": score},
            issues=issues,
        )
