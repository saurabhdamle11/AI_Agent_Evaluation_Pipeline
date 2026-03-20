"""
LLM-as-Judge evaluator using Claude via the Anthropic SDK.

Uses tool use to get reliable structured output: scores for response quality,
helpfulness, and factuality, plus any issues detected in the conversation.

On API failure the evaluator degrades gracefully — it logs the error and
returns a neutral (0.5) score with a warning issue so the rest of the
evaluation pipeline continues unaffected.
"""
import json
import logging

import anthropic

from src.config.settings import get_settings
from src.data.schemas.evaluation import EvaluatorOutput, IssueSchema, IssueSeverity
from src.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)
settings = get_settings()

# Tool definition used to force structured output from the judge
_EVALUATE_TOOL: anthropic.types.ToolParam = {
    "name": "evaluate_conversation",
    "description": (
        "Submit evaluation scores and detected issues for the AI agent conversation. "
        "All scores are floats between 0.0 (worst) and 1.0 (best)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "response_quality": {
                "type": "number",
                "description": "Weighted overall quality of the assistant's responses.",
            },
            "helpfulness": {
                "type": "number",
                "description": "How well the assistant addressed the user's needs.",
            },
            "factuality": {
                "type": "number",
                "description": "Factual accuracy and absence of hallucinated information.",
            },
            "issues": {
                "type": "array",
                "description": "Specific problems detected in the conversation.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "Short snake_case identifier for the issue type.",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warning", "critical"],
                        },
                        "message": {
                            "type": "string",
                            "description": "One-sentence description of the issue.",
                        },
                    },
                    "required": ["type", "severity", "message"],
                },
            },
        },
        "required": ["response_quality", "helpfulness", "factuality", "issues"],
    },
}

_SYSTEM_PROMPT = (
    "You are an expert, objective evaluator of AI agent conversations. "
    "Your role is to assess the quality of the assistant's responses — "
    "not the user's messages. Be critical but fair. "
    "Score each dimension on a continuous scale from 0.0 to 1.0."
)


def _format_conversation(conversation: dict) -> str:
    lines: list[str] = []
    for turn in conversation.get("turns", []):
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
        for tc in turn.get("tool_calls") or []:
            lines.append(
                f"  [TOOL CALL] {tc.get('tool_name')} "
                f"params={json.dumps(tc.get('parameters', {}))} "
                f"result={json.dumps(tc.get('result'))}"
            )
    return "\n".join(lines)


class LLMJudgeEvaluator(BaseEvaluator):
    """
    Evaluates response quality, helpfulness, and factuality using Claude.

    Scores (fed into weighted overall):
      - response_quality

    Details (informational, not weighted):
      - helpfulness, factuality
    """

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic()

    async def evaluate(self, conversation: dict) -> EvaluatorOutput:
        try:
            return await self._call_judge(conversation)
        except Exception as exc:
            logger.error(f"LLMJudgeEvaluator API call failed: {exc}")
            return EvaluatorOutput(
                evaluator_name="llm_judge",
                scores={"response_quality": 0.5},
                details={"error": str(exc)},
                issues=[
                    IssueSchema(
                        type="llm_judge_unavailable",
                        severity=IssueSeverity.warning,
                        message=f"LLM Judge could not complete evaluation: {exc}",
                    )
                ],
            )

    async def _call_judge(self, conversation: dict) -> EvaluatorOutput:
        conversation_text = _format_conversation(conversation)
        user_message = (
            f"Please evaluate the following AI agent conversation:\n\n"
            f"{conversation_text}\n\n"
            f"Use the evaluate_conversation tool to submit your assessment."
        )

        response = await self._client.messages.create(
            model=settings.llm_judge_model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=[_EVALUATE_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_message}],
        )

        tool_block = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )
        if tool_block is None:
            raise ValueError("LLM judge response contained no tool_use block")

        return self._parse_tool_input(tool_block.input)

    @staticmethod
    def _parse_tool_input(data: dict) -> EvaluatorOutput:
        def _clamp(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        issues = [
            IssueSchema(
                type=issue["type"],
                severity=IssueSeverity(issue["severity"]),
                message=issue["message"],
            )
            for issue in data.get("issues") or []
        ]

        return EvaluatorOutput(
            evaluator_name="llm_judge",
            scores={"response_quality": round(_clamp(data["response_quality"]), 4)},
            details={
                "helpfulness": round(_clamp(data["helpfulness"]), 4),
                "factuality": round(_clamp(data["factuality"]), 4),
            },
            issues=issues,
        )
