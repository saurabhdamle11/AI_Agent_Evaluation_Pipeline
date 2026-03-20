import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.schemas.evaluation import EvaluatorOutput, IssueSchema, IssueSeverity
from src.services.evaluation_service import EvaluationService


def make_conversation(conversation_id="conv-001"):
    return {
        "conversation_id": conversation_id,
        "agent_version": "1.0.0",
        "turns": [
            {
                "turn_id": 1,
                "role": "user",
                "content": "Hello",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "turn_id": 2,
                "role": "assistant",
                "content": "Hi there! How can I help you today with your requests?",
                "timestamp": "2024-01-01T00:00:01Z",
            },
        ],
        "metadata": {"total_latency_ms": 800, "mission_completed": True},
    }


def make_repos(conversation=None):
    conv_repo = MagicMock()
    conv_repo.get_by_id = AsyncMock(return_value=conversation)
    eval_repo = MagicMock()
    eval_repo.insert = AsyncMock(return_value=None)
    return conv_repo, eval_repo


class TestEvaluationServiceEvaluate:
    async def test_returns_none_when_conversation_not_found(self):
        conv_repo, eval_repo = make_repos(conversation=None)
        service = EvaluationService(conv_repo, eval_repo)
        result = await service.evaluate("missing-id")
        assert result is None
        eval_repo.insert.assert_not_awaited()

    async def test_returns_evaluation_result_for_valid_conversation(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        result = await service.evaluate("conv-001")
        assert result is not None
        assert result.conversation_id == "conv-001"

    async def test_result_is_persisted_via_repo(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        result = await service.evaluate("conv-001")
        eval_repo.insert.assert_awaited_once()
        inserted = eval_repo.insert.call_args[0][0]
        assert inserted["conversation_id"] == "conv-001"
        assert inserted["evaluation_id"] == result.evaluation_id

    async def test_scores_have_all_dimensions(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        result = await service.evaluate("conv-001")
        assert result.scores.heuristic is not None
        assert result.scores.tool_accuracy is not None
        assert result.scores.coherence is not None
        assert result.scores.response_quality is not None

    async def test_overall_score_is_weighted_combination(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        result = await service.evaluate("conv-001")
        scores = result.scores
        # Manually compute expected weighted overall
        weights = {"heuristic": 0.15, "tool_accuracy": 0.30, "coherence": 0.25, "response_quality": 0.30}
        dim_scores = {
            "heuristic": scores.heuristic,
            "tool_accuracy": scores.tool_accuracy,
            "coherence": scores.coherence,
            "response_quality": scores.response_quality,
        }
        expected = sum(weights[k] * dim_scores[k] for k in weights) / sum(weights.values())
        assert abs(result.scores.overall - round(expected, 4)) < 1e-3

    async def test_overall_score_in_valid_range(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        result = await service.evaluate("conv-001")
        assert 0.0 <= result.scores.overall <= 1.0

    async def test_evaluator_exception_becomes_warning_issue(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        # Make one evaluator raise
        service._evaluators[0].evaluate = AsyncMock(side_effect=RuntimeError("boom"))
        result = await service.evaluate("conv-001")
        assert result is not None  # service does not crash
        issue_types = [i.type for i in result.issues_detected]
        assert "evaluator_error" in issue_types
        assert any(
            i.type == "evaluator_error" and i.severity == IssueSeverity.warning
            for i in result.issues_detected
        )

    async def test_all_evaluator_exceptions_still_produces_result(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        for ev in service._evaluators:
            ev.evaluate = AsyncMock(side_effect=RuntimeError("fail"))
        result = await service.evaluate("conv-001")
        assert result is not None
        assert len(result.issues_detected) == len(service._evaluators)

    async def test_issues_from_all_evaluators_are_merged(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        # Inject a sentinel issue into each evaluator so we can verify merging
        for ev in service._evaluators:
            ev.evaluate = AsyncMock(return_value=EvaluatorOutput(
                evaluator_name=ev.__class__.__name__,
                scores={"heuristic": 0.8},
                issues=[IssueSchema(type=f"test_issue_{ev.__class__.__name__}", severity=IssueSeverity.info, message="x")],
            ))
        result = await service.evaluate("conv-001")
        issue_types = {i.type for i in result.issues_detected}
        # All four evaluators' sentinel issues must appear in the merged list
        for ev_name in ("HeuristicEvaluator", "ToolCallEvaluator", "CoherenceEvaluator", "LLMJudgeEvaluator"):
            assert f"test_issue_{ev_name}" in issue_types

    async def test_result_has_unique_evaluation_id(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        r1 = await service.evaluate("conv-001")
        r2 = await service.evaluate("conv-001")
        assert r1.evaluation_id != r2.evaluation_id

    async def test_tool_details_populated_when_tool_calls_present(self):
        conversation = make_conversation()
        conversation["turns"][0]["tool_calls"] = [
            {"tool_name": "search", "parameters": {"q": "hello"}, "result": {"r": 1}, "latency_ms": 200}
        ]
        conv_repo, eval_repo = make_repos(conversation=conversation)
        service = EvaluationService(conv_repo, eval_repo)
        result = await service.evaluate("conv-001")
        assert result.scores.tool_details is not None

    async def test_coherence_details_always_populated(self):
        conv_repo, eval_repo = make_repos(conversation=make_conversation())
        service = EvaluationService(conv_repo, eval_repo)
        result = await service.evaluate("conv-001")
        assert result.scores.coherence_details is not None


class TestWeightedOverall:
    def test_all_scores_present_uses_full_weights(self):
        conv_repo, eval_repo = make_repos()
        service = EvaluationService(conv_repo, eval_repo)
        scores = {"heuristic": 1.0, "tool_accuracy": 1.0, "coherence": 1.0, "response_quality": 1.0}
        assert service._weighted_overall(scores) == 1.0

    def test_all_zeros_returns_zero(self):
        conv_repo, eval_repo = make_repos()
        service = EvaluationService(conv_repo, eval_repo)
        scores = {"heuristic": 0.0, "tool_accuracy": 0.0, "coherence": 0.0, "response_quality": 0.0}
        assert service._weighted_overall(scores) == 0.0

    def test_empty_scores_returns_zero(self):
        conv_repo, eval_repo = make_repos()
        service = EvaluationService(conv_repo, eval_repo)
        assert service._weighted_overall({}) == 0.0

    def test_partial_scores_normalizes_weights(self):
        conv_repo, eval_repo = make_repos()
        service = EvaluationService(conv_repo, eval_repo)
        # Only heuristic present (weight 0.15 / 0.15 = 1.0 normalized)
        scores = {"heuristic": 0.8}
        assert service._weighted_overall(scores) == pytest.approx(0.8)

    def test_unknown_score_key_is_ignored(self):
        conv_repo, eval_repo = make_repos()
        service = EvaluationService(conv_repo, eval_repo)
        scores = {"heuristic": 1.0, "unknown_dim": 0.0}
        # unknown_dim has no weight, so result is same as heuristic-only
        assert service._weighted_overall(scores) == pytest.approx(1.0)
