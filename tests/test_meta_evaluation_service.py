"""
Tests for MetaEvaluationService and its pure-function helpers.

All MongoDB I/O is mocked — no real database connection required.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.services.meta_evaluation_service import (
    GOOD_THRESHOLD,
    MetaEvaluationService,
    _human_score,
    _pearson,
    _prf,
    _weight_multiplier,
)
from src.data.schemas.meta_evaluation import MetaEvaluationResult


# ── helpers ───────────────────────────────────────────────────────────────────

def make_eval(conversation_id="conv-1", overall=0.8, response_quality=0.8,
              tool_accuracy=0.9, coherence=0.7, heuristic=0.85) -> dict:
    return {
        "conversation_id": conversation_id,
        "evaluation_id": f"eval-{conversation_id}",
        "scores": {
            "overall": overall,
            "response_quality": response_quality,
            "tool_accuracy": tool_accuracy,
            "coherence": coherence,
            "heuristic": heuristic,
        },
        "created_at": datetime(2026, 3, 20, tzinfo=timezone.utc),
    }


def make_feedback(conversation_id="conv-1", user_rating=4, quality="good",
                  annotations=None) -> dict:
    return {
        "conversation_id": conversation_id,
        "user_rating": user_rating,
        "annotations": annotations or [],
        "aggregated": {
            "ops_review": {"quality": quality},
            "overall_kappa": 1.0,
            "routing": "auto_label",
        },
    }


def make_repos(evaluations=None, feedbacks=None):
    eval_repo = MagicMock()
    eval_repo.get_all = AsyncMock(return_value=evaluations or [])
    feedback_repo = MagicMock()
    feedback_repo.get_all_with_aggregated = AsyncMock(return_value=feedbacks or [])
    meta_repo = MagicMock()
    meta_repo.insert = AsyncMock(return_value={})
    return eval_repo, feedback_repo, meta_repo


# ── _human_score ──────────────────────────────────────────────────────────────

class TestHumanScore:
    def test_user_rating_5_maps_to_1(self):
        assert _human_score({"user_rating": 5}) == pytest.approx(1.0)

    def test_user_rating_1_maps_to_0(self):
        assert _human_score({"user_rating": 1}) == pytest.approx(0.0)

    def test_user_rating_3_maps_to_half(self):
        assert _human_score({"user_rating": 3}) == pytest.approx(0.5)

    def test_ops_review_excellent_maps_to_1(self):
        fb = {"aggregated": {"ops_review": {"quality": "excellent"}}}
        assert _human_score(fb) == pytest.approx(1.0)

    def test_ops_review_poor_maps_to_0(self):
        fb = {"aggregated": {"ops_review": {"quality": "poor"}}}
        assert _human_score(fb) == pytest.approx(0.0)

    def test_both_signals_are_averaged(self):
        fb = {"user_rating": 5, "aggregated": {"ops_review": {"quality": "poor"}}}
        # (1.0 + 0.0) / 2 = 0.5
        assert _human_score(fb) == pytest.approx(0.5)

    def test_no_signals_returns_none(self):
        assert _human_score({}) is None

    def test_no_human_rating_uses_ops_review_only(self):
        fb = {"aggregated": {"ops_review": {"quality": "good"}}}
        assert _human_score(fb) == pytest.approx(0.67, abs=0.01)

    def test_unknown_quality_value_is_ignored(self):
        fb = {"user_rating": 4, "aggregated": {"ops_review": {"quality": "amazing"}}}
        assert _human_score(fb) == pytest.approx(0.75)  # only user_rating used


# ── _pearson ──────────────────────────────────────────────────────────────────

class TestPearson:
    def test_perfect_positive_correlation(self):
        xs = [0.1, 0.5, 0.9]
        ys = [0.1, 0.5, 0.9]
        assert _pearson(xs, ys) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        xs = [0.1, 0.5, 0.9]
        ys = [0.9, 0.5, 0.1]
        assert _pearson(xs, ys) == pytest.approx(-1.0)

    def test_no_variance_returns_zero(self):
        assert _pearson([0.5, 0.5, 0.5], [0.1, 0.5, 0.9]) == 0.0

    def test_single_point_returns_zero(self):
        assert _pearson([0.5], [0.5]) == 0.0


# ── _prf ─────────────────────────────────────────────────────────────────────

class TestPRF:
    def test_all_correct_predictions(self):
        preds  = [0.8, 0.8, 0.2, 0.2]
        truths = [0.8, 0.8, 0.2, 0.2]
        p, r, f1 = _prf(preds, truths)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_no_true_positives_returns_zeros(self):
        # evaluator always says "good" but all are actually "bad"
        preds  = [0.8, 0.9]
        truths = [0.2, 0.1]
        p, r, f1 = _prf(preds, truths)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_partial_recall(self):
        preds  = [0.2, 0.8]   # detects one bad, misses one bad
        truths = [0.2, 0.2]
        _, r, _ = _prf(preds, truths)
        assert r == pytest.approx(0.5)

    def test_f1_is_harmonic_mean(self):
        p, r, f1 = _prf([0.2, 0.8, 0.8], [0.2, 0.2, 0.8])
        # TP=1 (pred bad, true bad), FP=0, FN=1 (pred good, true bad)
        # precision=1.0, recall=0.5, f1=0.667
        assert f1 == pytest.approx(2 * p * r / (p + r), abs=1e-4)


# ── _weight_multiplier ────────────────────────────────────────────────────────

class TestWeightMultiplier:
    def test_low_f1_penalizes(self):
        mult, _ = _weight_multiplier(0.3)
        assert float(mult) == pytest.approx(0.7)

    def test_mid_f1_neutral(self):
        mult, _ = _weight_multiplier(0.6)
        assert float(mult) == pytest.approx(1.0)

    def test_high_f1_boosts(self):
        mult, _ = _weight_multiplier(0.85)
        assert float(mult) == pytest.approx(1.2)

    def test_reason_is_returned(self):
        _, reason = _weight_multiplier(0.3)
        assert len(reason) > 0


# ── MetaEvaluationService.run ─────────────────────────────────────────────────

class TestMetaEvaluationServiceRun:
    async def test_returns_meta_evaluation_result(self):
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval()],
            feedbacks=[make_feedback()],
        )
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        assert isinstance(result, MetaEvaluationResult)

    async def test_result_has_unique_id(self):
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval()],
            feedbacks=[make_feedback()],
        )
        r1 = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        r2 = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        assert r1.meta_evaluation_id != r2.meta_evaluation_id

    async def test_result_is_persisted(self):
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval()],
            feedbacks=[make_feedback()],
        )
        await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        meta_repo.insert.assert_awaited_once()

    async def test_sample_size_is_paired_count(self):
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval("c1"), make_eval("c2"), make_eval("c3")],
            feedbacks=[make_feedback("c1"), make_feedback("c2")],  # c3 has no feedback
        )
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        assert result.sample_size == 2

    async def test_no_pairs_returns_empty_metrics(self):
        eval_repo, fb_repo, meta_repo = make_repos(evaluations=[], feedbacks=[])
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        assert result.sample_size == 0
        assert result.evaluator_metrics == []

    async def test_feedback_without_human_signals_excluded(self):
        # Feedback with no user_rating and no ops_review → no human score → excluded
        fb = {"conversation_id": "c1", "aggregated": {}}
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval("c1")],
            feedbacks=[fb],
        )
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        assert result.sample_size == 0

    async def test_produces_metrics_for_all_four_evaluator_types(self):
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval("c1"), make_eval("c2")],
            feedbacks=[make_feedback("c1"), make_feedback("c2")],
        )
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        types = {m.evaluator_type for m in result.evaluator_metrics}
        assert types == {"llm_judge", "tool_evaluator", "coherence", "heuristic"}

    async def test_produces_calibration_for_each_evaluator(self):
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval("c1")],
            feedbacks=[make_feedback("c1")],
        )
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        metric_types = {m.evaluator_type for m in result.evaluator_metrics}
        calib_types = {c.evaluator_type for c in result.calibration_adjustments}
        assert metric_types == calib_types

    async def test_calibration_multipliers_are_valid(self):
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval()],
            feedbacks=[make_feedback()],
        )
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        for adj in result.calibration_adjustments:
            assert adj.weight_multiplier in (0.7, 1.0, 1.2)

    async def test_blind_spots_detected_for_missed_conversations(self):
        # Evaluator says good (overall=0.8), human says bad (rating=1 → score=0.0)
        # Annotation type "accuracy" → should appear as blind spot
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval("c1", overall=0.8)],
            feedbacks=[make_feedback(
                "c1",
                user_rating=1,
                annotations=[{"type": "accuracy", "label": "incorrect", "annotator_id": "a1"}],
            )],
        )
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        blind_spot_types = {b.annotation_type for b in result.blind_spots}
        assert "accuracy" in blind_spot_types

    async def test_no_blind_spots_when_evaluator_agrees_with_human(self):
        # Both say good — no misses, no blind spots
        eval_repo, fb_repo, meta_repo = make_repos(
            evaluations=[make_eval("c1", overall=0.9)],
            feedbacks=[make_feedback("c1", user_rating=5)],
        )
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        assert result.blind_spots == []

    async def test_blind_spots_sorted_by_miss_rate_descending(self):
        evals = [make_eval(f"c{i}", overall=0.8) for i in range(3)]
        fbs = [
            make_feedback("c0", user_rating=1, annotations=[
                {"type": "tone", "label": "bad", "annotator_id": "a1"},
                {"type": "accuracy", "label": "bad", "annotator_id": "a1"},
            ]),
            make_feedback("c1", user_rating=1, annotations=[
                {"type": "tone", "label": "bad", "annotator_id": "a1"},
            ]),
            make_feedback("c2", user_rating=5, annotations=[
                {"type": "accuracy", "label": "ok", "annotator_id": "a1"},
            ]),
        ]
        eval_repo, fb_repo, meta_repo = make_repos(evaluations=evals, feedbacks=fbs)
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        miss_rates = [b.miss_rate for b in result.blind_spots]
        assert miss_rates == sorted(miss_rates, reverse=True)

    async def test_metrics_sample_size_matches_paired_conversations(self):
        evals = [make_eval(f"c{i}") for i in range(3)]
        fbs   = [make_feedback(f"c{i}") for i in range(3)]
        eval_repo, fb_repo, meta_repo = make_repos(evaluations=evals, feedbacks=fbs)
        result = await MetaEvaluationService.run(eval_repo, fb_repo, meta_repo)
        for m in result.evaluator_metrics:
            assert m.sample_size == 3
