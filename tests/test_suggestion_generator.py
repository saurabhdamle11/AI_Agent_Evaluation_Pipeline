from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.schemas.evaluation import (
    EvaluationResult,
    EvaluationScores,
    IssueSchema,
    IssueSeverity,
)
from src.data.schemas.suggestion import SuggestionType
from src.services.suggestion_generator import SuggestionGenerator, _fingerprint


def make_result(
    issues: list[IssueSchema] | None = None,
    overall: float = 0.9,
    tool_accuracy: float | None = None,
    coherence: float | None = None,
    conversation_id: str = "conv-001",
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_id="eval-001",
        conversation_id=conversation_id,
        scores=EvaluationScores(
            overall=overall,
            tool_accuracy=tool_accuracy,
            coherence=coherence,
        ),
        issues_detected=issues or [],
        created_at=datetime.now(timezone.utc),
    )


AGENT = "v1.0.0"
generator = SuggestionGenerator()


class TestFingerprintStability:
    def test_same_inputs_produce_same_fingerprint(self):
        assert _fingerprint("prompt", "high_latency", AGENT) == _fingerprint("prompt", "high_latency", AGENT)

    def test_different_issue_types_produce_different_fingerprints(self):
        assert _fingerprint("prompt", "high_latency", AGENT) != _fingerprint("prompt", "empty_content", AGENT)

    def test_different_agent_versions_produce_different_fingerprints(self):
        assert _fingerprint("prompt", "high_latency", "v1.0.0") != _fingerprint("prompt", "high_latency", "v2.0.0")

    def test_fingerprint_is_16_chars(self):
        assert len(_fingerprint("prompt", "high_latency", AGENT)) == 16


class TestIssueBasedSuggestions:
    def test_known_issue_type_generates_suggestion(self):
        result = make_result(issues=[IssueSchema(type="null_parameter", severity=IssueSeverity.warning, description="x")])
        suggestions = generator.generate(result, AGENT)
        assert len(suggestions) == 1
        assert suggestions[0]["type"] == SuggestionType.tool.value

    def test_unknown_issue_type_is_skipped(self):
        result = make_result(issues=[IssueSchema(type="unknown_issue_xyz", severity=IssueSeverity.info, description="x")])
        suggestions = generator.generate(result, AGENT)
        assert suggestions == []

    def test_critical_severity_boosts_confidence(self):
        warning_result = make_result(issues=[IssueSchema(type="null_parameter", severity=IssueSeverity.warning, description="x")])
        critical_result = make_result(issues=[IssueSchema(type="null_parameter", severity=IssueSeverity.critical, description="x")])
        warning_conf = generator.generate(warning_result, AGENT)[0]["confidence"]
        critical_conf = generator.generate(critical_result, AGENT)[0]["confidence"]
        assert critical_conf > warning_conf

    def test_duplicate_issue_types_produce_single_suggestion(self):
        issues = [
            IssueSchema(type="null_parameter", severity=IssueSeverity.warning, description="param A"),
            IssueSchema(type="null_parameter", severity=IssueSeverity.warning, description="param B"),
        ]
        result = make_result(issues=issues)
        suggestions = generator.generate(result, AGENT)
        assert len(suggestions) == 1

    def test_multiple_distinct_issues_produce_multiple_suggestions(self):
        issues = [
            IssueSchema(type="null_parameter", severity=IssueSeverity.warning, description="x"),
            IssueSchema(type="tool_execution_failure", severity=IssueSeverity.warning, description="y"),
            IssueSchema(type="high_latency", severity=IssueSeverity.warning, description="z"),
        ]
        result = make_result(issues=issues)
        suggestions = generator.generate(result, AGENT)
        assert len(suggestions) == 3

    def test_tool_issue_produces_tool_type_suggestion(self):
        result = make_result(issues=[IssueSchema(type="tool_execution_failure", severity=IssueSeverity.warning, description="x")])
        suggestions = generator.generate(result, AGENT)
        assert suggestions[0]["type"] == SuggestionType.tool.value

    def test_prompt_issue_produces_prompt_type_suggestion(self):
        result = make_result(issues=[IssueSchema(type="high_latency", severity=IssueSeverity.warning, description="x")])
        suggestions = generator.generate(result, AGENT)
        assert suggestions[0]["type"] == SuggestionType.prompt.value


class TestScoreBasedSuggestions:
    def test_low_tool_accuracy_generates_suggestion(self):
        result = make_result(tool_accuracy=0.5)
        suggestions = generator.generate(result, AGENT)
        fps = {s["_fingerprint"] for s in suggestions}
        expected_fp = _fingerprint("tool", "low_tool_accuracy", AGENT)
        assert expected_fp in fps

    def test_high_tool_accuracy_does_not_generate_suggestion(self):
        result = make_result(tool_accuracy=0.9)
        suggestions = generator.generate(result, AGENT)
        expected_fp = _fingerprint("tool", "low_tool_accuracy", AGENT)
        assert all(s["_fingerprint"] != expected_fp for s in suggestions)

    def test_low_coherence_generates_suggestion(self):
        result = make_result(coherence=0.5)
        suggestions = generator.generate(result, AGENT)
        fps = {s["_fingerprint"] for s in suggestions}
        assert _fingerprint("prompt", "low_coherence", AGENT) in fps

    def test_low_overall_generates_suggestion(self):
        result = make_result(overall=0.4)
        suggestions = generator.generate(result, AGENT)
        fps = {s["_fingerprint"] for s in suggestions}
        assert _fingerprint("prompt", "low_overall", AGENT) in fps

    def test_good_scores_produce_no_score_based_suggestions(self):
        result = make_result(overall=0.9, tool_accuracy=0.9, coherence=0.9)
        suggestions = generator.generate(result, AGENT)
        assert suggestions == []


class TestSuggestionShape:
    def test_required_fields_present(self):
        result = make_result(issues=[IssueSchema(type="empty_content", severity=IssueSeverity.warning, description="x")])
        s = generator.generate(result, AGENT)[0]
        for field in ("suggestion_id", "type", "conversation_ids", "agent_version",
                      "suggestion", "rationale", "confidence", "expected_impact",
                      "status", "created_at", "_fingerprint"):
            assert field in s, f"Missing field: {field}"

    def test_conversation_id_is_linked(self):
        result = make_result(
            issues=[IssueSchema(type="empty_content", severity=IssueSeverity.warning, description="x")],
            conversation_id="conv-abc",
        )
        s = generator.generate(result, AGENT)[0]
        assert "conv-abc" in s["conversation_ids"]

    def test_agent_version_is_set(self):
        result = make_result(issues=[IssueSchema(type="empty_content", severity=IssueSeverity.warning, description="x")])
        s = generator.generate(result, AGENT)[0]
        assert s["agent_version"] == AGENT

    def test_status_is_pending_by_default(self):
        result = make_result(issues=[IssueSchema(type="empty_content", severity=IssueSeverity.warning, description="x")])
        s = generator.generate(result, AGENT)[0]
        assert s["status"] == "pending"

    def test_confidence_is_within_bounds(self):
        issues = [IssueSchema(type=t, severity=IssueSeverity.critical, description="x")
                  for t in ("null_parameter", "empty_content", "tool_execution_failure")]
        result = make_result(issues=issues)
        for s in generator.generate(result, AGENT):
            assert 0.0 <= s["confidence"] <= 1.0


class TestEvaluationServiceIntegration:
    """Verify EvaluationService calls SuggestionGenerator and deduplicates correctly."""

    async def test_new_suggestion_is_inserted(self):
        from src.services.evaluation_service import EvaluationService

        # High latency triggers a high_latency issue → suggestion is generated
        conversation = {
            "conversation_id": "conv-001",
            "agent_version": "v1.0.0",
            "turns": [
                {"turn_id": 1, "role": "user", "content": "hello", "timestamp": "2024-01-01T00:00:00Z"},
                {"turn_id": 2, "role": "assistant", "content": "hi", "timestamp": "2024-01-01T00:00:01Z"},
            ],
            "metadata": {"total_latency_ms": 99999},
        }

        conv_repo = MagicMock()
        conv_repo.get_by_id = AsyncMock(return_value=conversation)

        eval_repo = MagicMock()
        eval_repo.insert = AsyncMock()

        sug_repo = MagicMock()
        sug_repo.exists_by_fingerprint = AsyncMock(return_value=False)
        sug_repo.insert = AsyncMock()
        sug_repo.append_conversation_id = AsyncMock()

        service = EvaluationService(conv_repo, eval_repo, sug_repo)
        await service.evaluate("conv-001")

        assert sug_repo.insert.called or sug_repo.append_conversation_id.called

    async def test_existing_suggestion_appends_conversation_id(self):
        from src.services.evaluation_service import EvaluationService

        conversation = {
            "conversation_id": "conv-002",
            "agent_version": "v1.0.0",
            "turns": [],
            "metadata": {"total_latency_ms": 99999},
        }

        conv_repo = MagicMock()
        conv_repo.get_by_id = AsyncMock(return_value=conversation)
        eval_repo = MagicMock()
        eval_repo.insert = AsyncMock()

        sug_repo = MagicMock()
        # Fingerprint already exists — should append, not insert
        sug_repo.exists_by_fingerprint = AsyncMock(return_value=True)
        sug_repo.insert = AsyncMock()
        sug_repo.append_conversation_id = AsyncMock()

        service = EvaluationService(conv_repo, eval_repo, sug_repo)
        await service.evaluate("conv-002")

        sug_repo.insert.assert_not_called()
        assert sug_repo.append_conversation_id.called
