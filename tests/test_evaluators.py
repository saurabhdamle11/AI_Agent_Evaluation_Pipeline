import pytest
from unittest.mock import AsyncMock

from src.data.schemas.evaluation import IssueSeverity
from src.evaluators.heuristic_evaluator import HeuristicEvaluator
from src.evaluators.tool_call_evaluator import ToolCallEvaluator
from src.evaluators.coherence_evaluator import CoherenceEvaluator
from src.evaluators.llm_judge_evaluator import LLMJudgeEvaluator


def make_turn(
    turn_id=1,
    role="user",
    content="Hello there",
    timestamp="2024-01-01T00:00:00Z",
    tool_calls=None,
):
    turn = {
        "turn_id": turn_id,
        "role": role,
        "content": content,
        "timestamp": timestamp,
    }
    if tool_calls is not None:
        turn["tool_calls"] = tool_calls
    return turn


def make_tool_call(tool_name="search", parameters=None, result=None, latency_ms=None):
    return {
        "tool_name": tool_name,
        "parameters": parameters or {"query": "hello"},
        "result": result,
        "latency_ms": latency_ms,
    }


def make_conversation(turns=None, total_latency_ms=None, mission_completed=True):
    return {
        "conversation_id": "conv-test",
        "agent_version": "1.0.0",
        "turns": turns or [],
        "metadata": {
            "total_latency_ms": total_latency_ms,
            "mission_completed": mission_completed,
        },
    }


# ---------------------------------------------------------------------------
# HeuristicEvaluator
# ---------------------------------------------------------------------------


class TestHeuristicEvaluator:
    @pytest.fixture
    def evaluator(self):
        return HeuristicEvaluator()

    async def test_perfect_conversation_scores_one(self, evaluator):
        conv = make_conversation(
            turns=[make_turn(turn_id=1, role="user"), make_turn(turn_id=2, role="assistant", content="Sure!")],
            total_latency_ms=5000,
        )
        output = await evaluator.evaluate(conv)
        assert output.scores["heuristic"] == 1.0
        assert output.issues == []

    async def test_high_total_latency_adds_warning(self, evaluator):
        conv = make_conversation(turns=[make_turn()], total_latency_ms=31_000)
        output = await evaluator.evaluate(conv)
        issue_types = [i.type for i in output.issues]
        assert "high_latency" in issue_types
        severity = next(i.severity for i in output.issues if i.type == "high_latency")
        assert severity == IssueSeverity.warning
        assert output.scores["heuristic"] < 1.0

    async def test_missing_required_field_adds_critical_issue(self, evaluator):
        turn = {"turn_id": 1, "role": "user", "timestamp": "2024-01-01T00:00:00Z"}  # missing content
        conv = make_conversation(turns=[turn])
        output = await evaluator.evaluate(conv)
        assert any(i.type == "missing_field" and i.severity == IssueSeverity.critical for i in output.issues)

    async def test_invalid_role_adds_warning(self, evaluator):
        conv = make_conversation(turns=[make_turn(role="bot")])
        output = await evaluator.evaluate(conv)
        assert any(i.type == "invalid_role" for i in output.issues)

    async def test_empty_content_adds_warning(self, evaluator):
        conv = make_conversation(turns=[make_turn(content="   ")])
        output = await evaluator.evaluate(conv)
        assert any(i.type == "empty_content" for i in output.issues)

    async def test_high_tool_latency_adds_warning(self, evaluator):
        tc = make_tool_call(latency_ms=6_000, result={"ok": True})
        conv = make_conversation(turns=[make_turn(tool_calls=[tc])])
        output = await evaluator.evaluate(conv)
        assert any(i.type == "tool_high_latency" for i in output.issues)

    async def test_score_clamps_to_zero_with_many_issues(self, evaluator):
        # Many missing fields across multiple turns should clamp to 0.0
        turns = [
            {"turn_id": i, "role": "user", "content": "", "timestamp": ""}
            for i in range(10)
        ]
        conv = make_conversation(turns=turns)
        output = await evaluator.evaluate(conv)
        assert output.scores["heuristic"] >= 0.0

    async def test_no_tool_calls_produces_no_tool_issues(self, evaluator):
        conv = make_conversation(turns=[make_turn()])
        output = await evaluator.evaluate(conv)
        assert not any(i.type == "tool_high_latency" for i in output.issues)

    async def test_evaluator_name(self, evaluator):
        conv = make_conversation(turns=[make_turn()])
        output = await evaluator.evaluate(conv)
        assert output.evaluator_name == "heuristic"

    async def test_tool_latency_below_threshold_no_issue(self, evaluator):
        tc = make_tool_call(latency_ms=3_000, result={"ok": True})
        conv = make_conversation(turns=[make_turn(tool_calls=[tc])])
        output = await evaluator.evaluate(conv)
        assert not any(i.type == "tool_high_latency" for i in output.issues)


# ---------------------------------------------------------------------------
# ToolCallEvaluator
# ---------------------------------------------------------------------------


class TestToolCallEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ToolCallEvaluator()

    async def test_no_tool_calls_returns_neutral_score(self, evaluator):
        conv = make_conversation(turns=[make_turn()])
        output = await evaluator.evaluate(conv)
        assert output.scores["tool_accuracy"] == 1.0
        assert output.issues == []

    async def test_all_successful_calls_scores_one(self, evaluator):
        tc = make_tool_call(result={"data": "ok"}, parameters={"query": "hello"})
        conv = make_conversation(turns=[make_turn(tool_calls=[tc])])
        output = await evaluator.evaluate(conv)
        assert output.scores["tool_accuracy"] == 1.0

    async def test_failed_tool_call_lowers_score(self, evaluator):
        tc = make_tool_call(result=None)  # no result = failure
        conv = make_conversation(turns=[make_turn(tool_calls=[tc])])
        output = await evaluator.evaluate(conv)
        assert output.scores["tool_accuracy"] < 1.0
        assert any(i.type == "tool_execution_failure" for i in output.issues)

    async def test_null_parameter_detected_as_hallucination(self, evaluator):
        tc = make_tool_call(parameters={"query": None}, result={"ok": True})
        conv = make_conversation(turns=[make_turn(tool_calls=[tc])])
        output = await evaluator.evaluate(conv)
        assert any(i.type == "null_parameter" for i in output.issues)

    async def test_null_parameter_lowers_parameter_accuracy(self, evaluator):
        tc = make_tool_call(parameters={"a": None, "b": "value"}, result={"ok": True})
        conv = make_conversation(turns=[make_turn(tool_calls=[tc])])
        output = await evaluator.evaluate(conv)
        details = output.details
        assert details["parameter_accuracy"] == 0.5

    async def test_hallucinated_params_count(self, evaluator):
        tc = make_tool_call(parameters={"a": None, "b": None, "c": "ok"}, result={"ok": True})
        conv = make_conversation(turns=[make_turn(tool_calls=[tc])])
        output = await evaluator.evaluate(conv)
        assert output.details["hallucinated_params"] == 2

    async def test_tool_calls_across_multiple_turns(self, evaluator):
        tc1 = make_tool_call(tool_name="search", result={"r": 1})
        tc2 = make_tool_call(tool_name="fetch", result=None)  # failed
        conv = make_conversation(
            turns=[
                make_turn(turn_id=1, role="user", tool_calls=[tc1]),
                make_turn(turn_id=2, role="assistant", content="done", tool_calls=[tc2]),
            ]
        )
        output = await evaluator.evaluate(conv)
        assert output.details["execution_success"] == 0.5

    async def test_details_include_all_fields(self, evaluator):
        tc = make_tool_call(result={"ok": True})
        conv = make_conversation(turns=[make_turn(tool_calls=[tc])])
        output = await evaluator.evaluate(conv)
        for field in ("selection_accuracy", "parameter_accuracy", "hallucinated_params", "execution_success"):
            assert field in output.details

    async def test_evaluator_name(self, evaluator):
        conv = make_conversation(turns=[make_turn()])
        output = await evaluator.evaluate(conv)
        assert output.evaluator_name == "tool_call"


# ---------------------------------------------------------------------------
# CoherenceEvaluator
# ---------------------------------------------------------------------------


class TestCoherenceEvaluator:
    @pytest.fixture
    def evaluator(self):
        return CoherenceEvaluator()

    async def test_empty_conversation_scores_zero(self, evaluator):
        conv = make_conversation(turns=[])
        output = await evaluator.evaluate(conv)
        assert output.scores["coherence"] == 0.0
        assert any(i.type == "empty_conversation" and i.severity == IssueSeverity.critical for i in output.issues)

    async def test_well_alternating_turns_scores_high(self, evaluator):
        turns = [
            make_turn(turn_id=1, role="user", content="What is Python?"),
            make_turn(turn_id=2, role="assistant", content="Python is a high-level programming language with many features."),
            make_turn(turn_id=3, role="user", content="Tell me more."),
            make_turn(turn_id=4, role="assistant", content="It supports object-oriented, functional, and procedural programming paradigms."),
        ]
        conv = make_conversation(turns=turns)
        output = await evaluator.evaluate(conv)
        assert output.scores["coherence"] >= 0.8
        assert not any(i.type == "role_alternation_violation" for i in output.issues)

    async def test_consecutive_same_role_triggers_violation(self, evaluator):
        turns = [
            make_turn(turn_id=1, role="user"),
            make_turn(turn_id=2, role="user"),  # violation: two user turns in a row
            make_turn(turn_id=3, role="assistant", content="Okay."),
        ]
        conv = make_conversation(turns=turns)
        output = await evaluator.evaluate(conv)
        assert any(i.type == "role_alternation_violation" for i in output.issues)

    async def test_system_turns_excluded_from_alternation_check(self, evaluator):
        turns = [
            make_turn(turn_id=1, role="system", content="System prompt."),
            make_turn(turn_id=2, role="user", content="Hello."),
            make_turn(turn_id=3, role="assistant", content="Hi! How can I help you today with your questions?"),
        ]
        conv = make_conversation(turns=turns)
        output = await evaluator.evaluate(conv)
        assert not any(i.type == "role_alternation_violation" for i in output.issues)

    async def test_short_assistant_responses_lower_context_retention(self, evaluator):
        # All assistant responses are too short (< 50 chars)
        turns = [
            make_turn(turn_id=1, role="user", content="Question?"),
            make_turn(turn_id=2, role="assistant", content="Yes."),
        ]
        conv = make_conversation(turns=turns)
        output = await evaluator.evaluate(conv)
        details = output.details
        assert details["context_retention"] < 1.0

    async def test_context_retention_low_triggers_issue(self, evaluator):
        # user turn with no assistant response at all — low coverage
        turns = [
            make_turn(turn_id=1, role="user", content="Question one?"),
            make_turn(turn_id=2, role="user", content="Question two?"),
            make_turn(turn_id=3, role="assistant", content="Ok."),  # only one short response for 2 questions
        ]
        conv = make_conversation(turns=turns)
        output = await evaluator.evaluate(conv)
        assert any(i.type == "low_context_retention" for i in output.issues)

    async def test_details_include_all_fields(self, evaluator):
        conv = make_conversation(turns=[make_turn(turn_id=1, role="user")])
        output = await evaluator.evaluate(conv)
        for field in ("consistency_score", "contradiction_count", "context_retention"):
            assert field in output.details

    async def test_contradiction_count_is_zero_placeholder(self, evaluator):
        turns = [make_turn(turn_id=1, role="user"), make_turn(turn_id=2, role="assistant", content="Answer.")]
        conv = make_conversation(turns=turns)
        output = await evaluator.evaluate(conv)
        assert output.details["contradiction_count"] == 0

    async def test_evaluator_name(self, evaluator):
        conv = make_conversation(turns=[make_turn()])
        output = await evaluator.evaluate(conv)
        assert output.evaluator_name == "coherence"

    async def test_only_user_turns_no_assistant_responses(self, evaluator):
        turns = [make_turn(turn_id=1, role="user"), make_turn(turn_id=2, role="user")]
        conv = make_conversation(turns=turns)
        output = await evaluator.evaluate(conv)
        # Should not crash; context retention should be 0 (no assistant responses for user turns)
        assert output.scores["coherence"] >= 0.0


# ---------------------------------------------------------------------------
# LLMJudgeEvaluator
# ---------------------------------------------------------------------------


class TestLLMJudgeEvaluator:
    @pytest.fixture
    def evaluator(self):
        return LLMJudgeEvaluator()

    async def test_returns_neutral_placeholder_score(self, evaluator):
        conv = make_conversation(turns=[make_turn()])
        output = await evaluator.evaluate(conv)
        assert output.scores["response_quality"] == 0.5

    async def test_stub_issue_is_info_severity(self, evaluator):
        conv = make_conversation(turns=[make_turn()])
        output = await evaluator.evaluate(conv)
        assert any(i.type == "stub_evaluator" and i.severity == IssueSeverity.info for i in output.issues)

    async def test_details_contains_stub_flag(self, evaluator):
        conv = make_conversation(turns=[make_turn()])
        output = await evaluator.evaluate(conv)
        assert output.details.get("stub") is True

    async def test_evaluator_name(self, evaluator):
        conv = make_conversation(turns=[make_turn()])
        output = await evaluator.evaluate(conv)
        assert output.evaluator_name == "llm_judge"
