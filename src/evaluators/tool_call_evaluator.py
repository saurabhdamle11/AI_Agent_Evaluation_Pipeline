import logging

from src.data.schemas.evaluation import EvaluatorOutput, IssueSchema, IssueSeverity, ToolEvaluationDetails
from src.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class ToolCallEvaluator(BaseEvaluator):
    async def evaluate(self, conversation: dict) -> EvaluatorOutput:
        issues: list[IssueSchema] = []
        turns = conversation.get("turns", [])

        all_tool_calls = [
            tc
            for turn in turns
            for tc in (turn.get("tool_calls") or [])
        ]

        if not all_tool_calls:
            # No tool calls to evaluate — score is neutral
            return EvaluatorOutput(
                evaluator_name="tool_call",
                scores={"tool_accuracy": 1.0},
                issues=[],
            )

        # execution_success: ratio of calls with a non-None result
        executed = sum(1 for tc in all_tool_calls if tc.get("result") is not None)
        execution_success = executed / len(all_tool_calls)

        if execution_success < 1.0:
            failed = len(all_tool_calls) - executed
            issues.append(IssueSchema(
                type="tool_execution_failure",
                severity=IssueSeverity.warning,
                message=f"{failed}/{len(all_tool_calls)} tool calls returned no result (possible failure)",
            ))

        # parameter_accuracy: ratio of non-null parameter values
        # hallucinated_params: count of null-valued parameters
        total_params = 0
        null_params = 0
        for tc in all_tool_calls:
            for key, value in (tc.get("parameters") or {}).items():
                total_params += 1
                if value is None:
                    null_params += 1
                    issues.append(IssueSchema(
                        type="null_parameter",
                        severity=IssueSeverity.warning,
                        message=f"Tool '{tc.get('tool_name')}' has null value for param '{key}' — possible hallucination",
                    ))

        parameter_accuracy = (total_params - null_params) / total_params if total_params > 0 else 1.0
        hallucinated_params = null_params

        # selection_accuracy: proxy based on execution success
        # (true selection accuracy requires ground-truth labels)
        selection_accuracy = execution_success

        tool_accuracy = round((execution_success + parameter_accuracy + selection_accuracy) / 3, 4)

        details = ToolEvaluationDetails(
            selection_accuracy=round(selection_accuracy, 4),
            parameter_accuracy=round(parameter_accuracy, 4),
            hallucinated_params=hallucinated_params,
            execution_success=round(execution_success, 4),
        )

        return EvaluatorOutput(
            evaluator_name="tool_call",
            scores={"tool_accuracy": tool_accuracy},
            details=details.model_dump(),
            issues=issues,
        )
