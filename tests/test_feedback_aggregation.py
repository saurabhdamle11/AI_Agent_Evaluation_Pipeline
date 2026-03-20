"""
Tests for FeedbackAggregationService.

Covers:
- Kappa computation (_pairwise_kappa)
- Weighted majority label selection
- Full aggregate() output: consensus, routing, overall_kappa
- Edge cases: no annotations, single annotator, unanimous agreement
"""
import pytest

from src.services.feedback_aggregation_service import (
    KAPPA_THRESHOLD,
    FeedbackAggregationService,
    _pairwise_kappa,
    _weighted_majority_label,
)


# ── _pairwise_kappa ──────────────────────────────────────────────────────────

class TestPairwiseKappa:
    def test_single_label_returns_one(self):
        assert _pairwise_kappa(["positive"]) == 1.0

    def test_perfect_agreement(self):
        assert _pairwise_kappa(["yes", "yes", "yes"]) == 1.0

    def test_perfect_disagreement_two_classes(self):
        # 2 annotators, each picks a different label → kappa = -1 (or near 0)
        # Po=0, Pe=0.25+0.25=0.5 → kappa = (0-0.5)/(1-0.5) = -1.0
        k = _pairwise_kappa(["a", "b"])
        assert k == pytest.approx(-1.0, abs=1e-6)

    def test_partial_agreement(self):
        # 3 raters: 2 agree, 1 disagrees
        labels = ["yes", "yes", "no"]
        k = _pairwise_kappa(labels)
        # Po = 1/3 (one agreeing pair out of 3), Pe = (2/3)^2 + (1/3)^2 = 5/9
        # kappa = (1/3 - 5/9) / (1 - 5/9) = (-2/9) / (4/9) = -0.5
        assert k == pytest.approx(-0.5, abs=1e-6)

    def test_majority_agreement(self):
        labels = ["good", "good", "good", "bad"]
        k = _pairwise_kappa(labels)
        assert -1.0 <= k <= 1.0

    def test_returns_float(self):
        assert isinstance(_pairwise_kappa(["a", "a", "b"]), float)


# ── _weighted_majority_label ─────────────────────────────────────────────────

class TestWeightedMajorityLabel:
    def test_unanimous(self):
        label, conf = _weighted_majority_label(["pos", "pos"], [0.9, 0.8])
        assert label == "pos"
        assert conf == pytest.approx(0.85, abs=1e-4)

    def test_majority_wins(self):
        label, conf = _weighted_majority_label(["a", "a", "b"], [0.9, 0.8, 1.0])
        assert label == "a"

    def test_confidence_breaks_tie(self):
        # "b" has higher total confidence → wins
        label, _ = _weighted_majority_label(["a", "b"], [0.3, 0.9])
        assert label == "b"

    def test_missing_confidence_uses_neutral_weight(self):
        # None → 0.5 neutral weight; "a" gets 0.9, "b" gets 0.5
        label, _ = _weighted_majority_label(["a", "b"], [0.9, None])
        assert label == "a"

    def test_all_missing_confidence(self):
        label, conf = _weighted_majority_label(["x", "x"], [None, None])
        assert label == "x"
        assert conf == pytest.approx(0.5, abs=1e-4)


# ── FeedbackAggregationService.aggregate ─────────────────────────────────────

def _make_feedback(**overrides) -> dict:
    base = {
        "conversation_id": "conv-001",
        "user_rating": 4,
        "ops_review": {"quality": "good", "notes": "looks fine"},
        "annotations": [
            {"type": "tone", "label": "positive", "annotator_id": "ann-1", "confidence": 0.9},
            {"type": "tone", "label": "positive", "annotator_id": "ann-2", "confidence": 0.85},
            {"type": "accuracy", "label": "correct", "annotator_id": "ann-1", "confidence": 0.95},
            {"type": "accuracy", "label": "correct", "annotator_id": "ann-2", "confidence": 0.8},
        ],
    }
    base.update(overrides)
    return base


class TestFeedbackAggregationService:
    def test_returns_aggregated_feedback(self):
        result = FeedbackAggregationService.aggregate(_make_feedback())
        assert result.conversation_id == "conv-001"
        assert result.user_rating == 4

    def test_consensus_labels_correct(self):
        result = FeedbackAggregationService.aggregate(_make_feedback())
        by_type = {c.type: c for c in result.annotation_consensus}
        assert by_type["tone"].consensus_label == "positive"
        assert by_type["accuracy"].consensus_label == "correct"

    def test_unanimous_kappa_is_one(self):
        result = FeedbackAggregationService.aggregate(_make_feedback())
        for consensus in result.annotation_consensus:
            assert consensus.kappa == pytest.approx(1.0, abs=1e-4)

    def test_routing_auto_label_when_high_agreement(self):
        result = FeedbackAggregationService.aggregate(_make_feedback())
        assert result.routing == "auto_label"

    def test_routing_human_review_when_low_agreement(self):
        feedback = _make_feedback(
            annotations=[
                {"type": "tone", "label": "positive", "annotator_id": "ann-1", "confidence": 0.9},
                {"type": "tone", "label": "negative", "annotator_id": "ann-2", "confidence": 0.9},
            ]
        )
        result = FeedbackAggregationService.aggregate(feedback)
        assert result.routing == "human_review"
        assert result.overall_kappa < KAPPA_THRESHOLD

    def test_no_annotations_returns_auto_label(self):
        feedback = _make_feedback(annotations=[])
        result = FeedbackAggregationService.aggregate(feedback)
        assert result.annotation_consensus == []
        assert result.overall_kappa == 1.0
        assert result.routing == "auto_label"

    def test_none_annotations_treated_as_empty(self):
        feedback = _make_feedback(annotations=None)
        result = FeedbackAggregationService.aggregate(feedback)
        assert result.annotation_consensus == []

    def test_single_annotator_kappa_is_one(self):
        feedback = _make_feedback(
            annotations=[
                {"type": "tone", "label": "positive", "annotator_id": "ann-1", "confidence": 0.7},
            ]
        )
        result = FeedbackAggregationService.aggregate(feedback)
        assert result.annotation_consensus[0].kappa == pytest.approx(1.0)
        assert result.annotation_consensus[0].annotator_count == 1

    def test_agreement_score_is_proportion(self):
        feedback = _make_feedback(
            annotations=[
                {"type": "tone", "label": "positive", "annotator_id": "a1", "confidence": 0.8},
                {"type": "tone", "label": "positive", "annotator_id": "a2", "confidence": 0.9},
                {"type": "tone", "label": "negative", "annotator_id": "a3", "confidence": 0.7},
            ]
        )
        result = FeedbackAggregationService.aggregate(feedback)
        tone = result.annotation_consensus[0]
        assert tone.agreement_score == pytest.approx(2 / 3, abs=1e-4)

    def test_multiple_annotation_types_aggregated_independently(self):
        result = FeedbackAggregationService.aggregate(_make_feedback())
        types = {c.type for c in result.annotation_consensus}
        assert types == {"tone", "accuracy"}

    def test_overall_kappa_is_mean_of_type_kappas(self):
        result = FeedbackAggregationService.aggregate(_make_feedback())
        kappas = [c.kappa for c in result.annotation_consensus]
        expected = round(sum(kappas) / len(kappas), 4)
        assert result.overall_kappa == pytest.approx(expected, abs=1e-4)

    def test_aggregated_at_is_set(self):
        result = FeedbackAggregationService.aggregate(_make_feedback())
        assert result.aggregated_at is not None

    def test_ops_review_passed_through(self):
        result = FeedbackAggregationService.aggregate(_make_feedback())
        assert result.ops_review is not None
        assert result.ops_review.quality == "good"

    def test_missing_confidence_handled(self):
        feedback = _make_feedback(
            annotations=[
                {"type": "tone", "label": "positive", "annotator_id": "a1", "confidence": None},
                {"type": "tone", "label": "positive", "annotator_id": "a2", "confidence": None},
            ]
        )
        result = FeedbackAggregationService.aggregate(feedback)
        tone = result.annotation_consensus[0]
        assert tone.consensus_label == "positive"
        assert tone.weighted_confidence == pytest.approx(0.5, abs=1e-4)
