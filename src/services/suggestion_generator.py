import hashlib
import logging
import uuid
from datetime import datetime, timezone

from src.data.schemas.evaluation import EvaluationResult, IssueSeverity
from src.data.schemas.suggestion import SuggestionType

logger = logging.getLogger(__name__)

# Maps issue_type → (suggestion_type, suggestion_text, rationale, base_confidence, expected_impact)
_ISSUE_RULES: dict[str, tuple] = {
    "null_parameter": (
        SuggestionType.tool,
        "Add null/required field validation to tool parameter schema",
        "Null parameter values indicate the agent is hallucinating or inferring missing values. "
        "Adding explicit required field constraints to the tool schema prevents this.",
        0.75,
        "Reduce parameter hallucination rate",
    ),
    "tool_execution_failure": (
        SuggestionType.tool,
        "Add retry logic and fallback handling for tool execution failures",
        "Tool calls returning no result indicate execution failures. Retry logic with exponential "
        "backoff and graceful fallbacks improve reliability.",
        0.70,
        "Improve tool execution success rate",
    ),
    "tool_high_latency": (
        SuggestionType.tool,
        "Optimize or add caching for slow tool calls",
        "Tool latency exceeds threshold. Consider caching frequent queries or optimizing "
        "the tool implementation to reduce response time.",
        0.60,
        "Reduce tool call latency below threshold",
    ),
    "high_latency": (
        SuggestionType.prompt,
        "Reduce prompt complexity to lower total response latency",
        "Total response latency exceeds target. Simplifying the system prompt or reducing "
        "unnecessary reasoning steps can meaningfully cut latency.",
        0.55,
        "Bring total latency within target threshold",
    ),
    "low_context_retention": (
        SuggestionType.prompt,
        "Add explicit context summarization instructions between turns",
        "Low context retention score indicates the agent loses earlier context in multi-turn "
        "conversations. Instructing the agent to summarize key facts between turns helps.",
        0.70,
        "Improve context retention in multi-turn conversations",
    ),
    "role_alternation_violation": (
        SuggestionType.prompt,
        "Enforce strict turn-taking structure in the system prompt",
        "Consecutive turns with the same role violate expected conversation structure. Adding "
        "explicit turn-taking instructions to the system prompt prevents this pattern.",
        0.65,
        "Ensure proper conversation structure",
    ),
    "empty_content": (
        SuggestionType.prompt,
        "Add guardrails to prevent empty or blank assistant responses",
        "Empty assistant responses degrade user experience. Adding explicit non-empty response "
        "requirements to the system prompt mitigates this.",
        0.80,
        "Eliminate empty assistant responses",
    ),
    "missing_field": (
        SuggestionType.prompt,
        "Add output validation to ensure all required turn fields are populated",
        "Missing required fields in conversation turns indicate data quality issues in the "
        "agent output pipeline. Structured output enforcement or post-processing validation helps.",
        0.50,
        "Improve conversation data completeness",
    ),
    "invalid_role": (
        SuggestionType.prompt,
        "Restrict role values to the valid enum: user, assistant, system",
        "Invalid role values indicate a schema mismatch or upstream data issue in the agent. "
        "Enforcing role enum at the output layer prevents downstream processing failures.",
        0.65,
        "Prevent invalid role values in conversation logs",
    ),
    "empty_conversation": (
        SuggestionType.prompt,
        "Add a minimum turn count guard to prevent empty conversation submissions",
        "Conversations with no turns should not reach the evaluation pipeline. "
        "Add validation at the ingestion boundary to reject empty payloads.",
        0.85,
        "Prevent empty conversations from polluting evaluation results",
    ),
}

# Score-based suggestions — fired when a score is below threshold regardless of specific issues
_SCORE_RULES: list[tuple] = [
    # (score_key, threshold, suggestion_type, suggestion, rationale, confidence, expected_impact)
    (
        "tool_accuracy", 0.7,
        SuggestionType.tool,
        "Review and improve tool schema definitions and parameter descriptions",
        "Low tool accuracy score indicates systematic issues with tool selection or parameter "
        "extraction. Richer, more precise schema descriptions help the model use tools correctly.",
        0.60,
        "Improve overall tool accuracy score",
    ),
    (
        "coherence", 0.7,
        SuggestionType.prompt,
        "Add explicit multi-turn context management instructions to the system prompt",
        "Low coherence score indicates the agent struggles with context across turns. "
        "System-level instructions for context tracking and summarization improve this.",
        0.60,
        "Improve multi-turn coherence score",
    ),
    (
        "overall", 0.6,
        SuggestionType.prompt,
        "Conduct a general prompt review — overall quality is below acceptable threshold",
        "Overall evaluation score is below 0.6. A comprehensive prompt review targeting "
        "response quality, instruction clarity, and task alignment is recommended.",
        0.55,
        "Raise overall conversation quality above the 0.6 threshold",
    ),
]


def _fingerprint(suggestion_type: str, issue_key: str, agent_version: str) -> str:
    """16-char hex fingerprint for deduplication — same type+issue+agent_version = same fingerprint."""
    raw = f"{suggestion_type}:{issue_key}:{agent_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SuggestionGenerator:
    """
    Deterministic rule engine that maps EvaluationResult → improvement suggestions.

    Two rule sources:
    - Issue-driven: each entry in issues_detected is matched against _ISSUE_RULES
    - Score-driven:  scores below thresholds trigger _SCORE_RULES

    Each suggestion carries a _fingerprint for deduplication across evaluations.
    When the same pattern recurs for the same agent_version, the existing suggestion
    accumulates more conversation_ids rather than spawning duplicates.
    """

    def generate(self, evaluation: EvaluationResult, agent_version: str) -> list[dict]:
        """Return suggestion dicts ready to upsert into MongoDB."""
        suggestions: list[dict] = []
        seen: set[str] = set()

        # Issue-driven suggestions
        for issue in evaluation.issues_detected:
            rule = _ISSUE_RULES.get(issue.type)
            if not rule:
                logger.debug(f"No suggestion rule for issue type '{issue.type}' — skipping")
                continue

            stype, text, rationale, base_conf, impact = rule
            confidence = min(base_conf + 0.1, 1.0) if issue.severity == IssueSeverity.critical else base_conf

            fp = _fingerprint(stype.value, issue.type, agent_version)
            if fp in seen:
                continue
            seen.add(fp)

            suggestions.append(
                self._build(stype, text, rationale, confidence, impact, agent_version, evaluation.conversation_id, fp)
            )

        # Score-driven suggestions
        score_dict = evaluation.scores.model_dump()
        for score_key, threshold, stype, text, rationale, confidence, impact in _SCORE_RULES:
            val = score_dict.get(score_key)
            if val is not None and val < threshold:
                fp = _fingerprint(stype.value, f"low_{score_key}", agent_version)
                if fp in seen:
                    continue
                seen.add(fp)
                suggestions.append(
                    self._build(stype, text, rationale, confidence, impact, agent_version, evaluation.conversation_id, fp)
                )

        logger.info(
            f"SuggestionGenerator produced {len(suggestions)} candidate(s) "
            f"for conversation={evaluation.conversation_id} agent={agent_version}"
        )
        return suggestions

    @staticmethod
    def _build(
        stype: SuggestionType,
        suggestion_text: str,
        rationale: str,
        confidence: float,
        expected_impact: str,
        agent_version: str,
        conversation_id: str,
        fingerprint: str,
    ) -> dict:
        return {
            "suggestion_id": str(uuid.uuid4()),
            "type": stype.value,
            "conversation_ids": [conversation_id],
            "agent_version": agent_version,
            "suggestion": suggestion_text,
            "rationale": rationale,
            "confidence": round(confidence, 4),
            "expected_impact": expected_impact,
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            "_fingerprint": fingerprint,
        }
