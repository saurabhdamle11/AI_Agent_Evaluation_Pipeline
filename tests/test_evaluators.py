import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


def _make_tool_block(input_data: dict) -> MagicMock:
    """Build a mock tool_use content block matching the Anthropic SDK shape."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = input_data
    return block


def _make_api_response(tool_input: dict) -> MagicMock:
    response = MagicMock()
    response.content = [_make_tool_block(tool_input)]
    return response


def _default_tool_input(**overrides) -> dict:
    base = {
        "response_quality": 0.85,
        "helpfulness": 0.9,
        "factuality": 0.8,
        "issues": [],
    }
    base.update(overrides)
    return base


class TestLLMJudgeEvaluator:
    @pytest.fixture
    def evaluator(self):
        with patch("src.evaluators.llm_judge_evaluator.anthropic.AsyncAnthropic"):
            return LLMJudgeEvaluator()

    def _patch_client(self, evaluator, tool_input: dict | None = None, exc: Exception | None = None):
        mock_create = AsyncMock()
        if exc:
            mock_create.side_effect = exc
        else:
            mock_create.return_value = _make_api_response(tool_input or _default_tool_input())
        evaluator._client.messages.create = mock_create
        return mock_create

    # ── happy path ──────────────────────────────────────────────────────────

    async def test_evaluator_name(self, evaluator):
        self._patch_client(evaluator)
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert output.evaluator_name == "llm_judge"

    async def test_response_quality_score_is_returned(self, evaluator):
        self._patch_client(evaluator, _default_tool_input(response_quality=0.75))
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert output.scores["response_quality"] == pytest.approx(0.75)

    async def test_helpfulness_and_factuality_in_details(self, evaluator):
        self._patch_client(evaluator, _default_tool_input(helpfulness=0.9, factuality=0.7))
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert output.details["helpfulness"] == pytest.approx(0.9)
        assert output.details["factuality"] == pytest.approx(0.7)

    async def test_issues_from_judge_are_returned(self, evaluator):
        tool_input = _default_tool_input(issues=[
            {"type": "vague_response", "severity": "warning", "message": "Response was too vague."}
        ])
        self._patch_client(evaluator, tool_input)
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert len(output.issues) == 1
        assert output.issues[0].type == "vague_response"
        assert output.issues[0].severity == IssueSeverity.warning

    async def test_no_issues_returns_empty_list(self, evaluator):
        self._patch_client(evaluator, _default_tool_input(issues=[]))
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert output.issues == []

    async def test_scores_are_clamped_to_valid_range(self, evaluator):
        self._patch_client(evaluator, _default_tool_input(response_quality=1.5))
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert output.scores["response_quality"] <= 1.0

    async def test_calls_api_with_correct_model(self, evaluator):
        from src.config.settings import get_settings
        mock_create = self._patch_client(evaluator)
        await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert mock_create.call_args.kwargs["model"] == get_settings().llm_judge_model

    async def test_tool_choice_forces_tool_use(self, evaluator):
        mock_create = self._patch_client(evaluator)
        await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert mock_create.call_args.kwargs["tool_choice"] == {"type": "any"}

    async def test_conversation_turns_included_in_prompt(self, evaluator):
        mock_create = self._patch_client(evaluator)
        conv = make_conversation(turns=[
            make_turn(role="user", content="What is 2+2?"),
            make_turn(role="assistant", content="It is 4."),
        ])
        await evaluator.evaluate(conv)
        user_message = mock_create.call_args.kwargs["messages"][0]["content"]
        assert "What is 2+2?" in user_message
        assert "It is 4." in user_message

    async def test_tool_calls_included_in_prompt(self, evaluator):
        mock_create = self._patch_client(evaluator)
        conv = make_conversation(turns=[
            make_turn(
                role="assistant",
                tool_calls=[make_tool_call(tool_name="web_search", parameters={"query": "test"})],
            )
        ])
        await evaluator.evaluate(conv)
        user_message = mock_create.call_args.kwargs["messages"][0]["content"]
        assert "web_search" in user_message

    # ── graceful degradation ─────────────────────────────────────────────────

    async def test_api_error_returns_neutral_score(self, evaluator):
        self._patch_client(evaluator, exc=RuntimeError("connection refused"))
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert output.scores["response_quality"] == pytest.approx(0.5)

    async def test_api_error_adds_warning_issue(self, evaluator):
        self._patch_client(evaluator, exc=RuntimeError("timeout"))
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert any(
            i.type == "llm_judge_unavailable" and i.severity == IssueSeverity.warning
            for i in output.issues
        )

    async def test_missing_tool_block_raises_and_degrades(self, evaluator):
        # Response with no tool_use block → ValueError → graceful fallback
        response = MagicMock()
        response.content = []  # no tool_use block
        evaluator._client.messages.create = AsyncMock(return_value=response)
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert output.scores["response_quality"] == pytest.approx(0.5)
        assert any(i.type == "llm_judge_unavailable" for i in output.issues)

    async def test_error_detail_stored_in_details(self, evaluator):
        self._patch_client(evaluator, exc=RuntimeError("boom"))
        output = await evaluator.evaluate(make_conversation(turns=[make_turn()]))
        assert "boom" in output.details.get("error", "")
